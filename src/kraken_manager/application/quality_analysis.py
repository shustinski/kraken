"""Evaluate explicit quality rules against imported, immutable AnalysisRun results.

No measurements are generated here. The analysis engine owns computation and
the run store owns result import; this service consumes their actual records.
"""

from dataclasses import asdict
from datetime import UTC, datetime
from hashlib import sha256
from uuid import UUID, uuid5

from kraken_core.analysis_run_protocol import payload_sha256
from kraken_manager.application.authorization import AuthorizationPolicy
from kraken_manager.application.errors import ConflictError, NotFoundError
from kraken_manager.application.quality import QualityService
from kraken_manager.domain.common import DomainValidationError, freeze_mapping, require_non_empty, validate_uuid
from kraken_manager.domain.events import ActorSnapshot, EventEnvelope
from kraken_manager.domain.identity import Permission
from kraken_manager.domain.quality import QualityIssue, QualityMetricRule
from kraken_manager.domain.project import ProjectState


class QualityAnalysisService:
    RULE_FIELDS = {"metric_key", "value_field", "comparison", "threshold", "target_binding"}

    @staticmethod
    def signature(manifest, worker_runtime):
        if worker_runtime is None:
            raise DomainValidationError("Quality comparison requires a measured worker runtime; repeat the legacy analysis")
        return payload_sha256({"recipe": manifest.recipe.to_payload(), "runtime": worker_runtime.to_payload(),
                               "parameters": [item.to_payload() for item in manifest.parameters]})

    @staticmethod
    def summaries(store, project_id):
        runs = store.list_runs_for_project(project_id) if hasattr(store, "list_runs_for_project") else store.list_runs()
        return tuple({"run_id": run.run_id, "state": run.state, "total_frames": run.total_frames,
            "completed_frames": run.completed_frames, "failed_frames": run.failed_frames,
            "metric_keys": store.metric_keys(run.run_id),
            "bindings": tuple(binding.binding_key for binding in run.manifest.source_bindings)}
            for run in runs if str(run.project_id) == str(project_id))

    @staticmethod
    def _authorize(unit, profiles, project_id, principal, *, gitlab_identity_verified):
        project = unit.projections.get_project(validate_uuid(str(project_id)))
        if project is None:
            raise NotFoundError("Project was not found")
        if project.state is not ProjectState.ACTIVE:
            raise ConflictError("Archived projects cannot change analysis quality decisions")
        AuthorizationPolicy().require(principal=principal, storage=profiles.get(project.storage_profile),
            permission=Permission.MANAGE_REVIEW, roles=unit.acl.roles_for(project.id, principal.id),
            gitlab_identity_verified=gitlab_identity_verified)
        return project

    @classmethod
    def snapshot(cls, store, projections, project_id, run_id, metric_key=None):
        run = store.get_run(require_non_empty(run_id, field="run_id"))
        if run is None or str(run.project_id) != str(project_id):
            raise NotFoundError("Analysis run was not found in the project")
        if (run.state != "completed" or run.failed_frames or run.completed_frames != run.total_frames
                or run.imported_partitions != run.total_partitions or not run.total_partitions):
            raise ConflictError("Quality evaluation requires a fully imported completed AnalysisRun without failed frames")
        if getattr(run, "worker_runtime", None) is None:
            raise DomainValidationError("Quality evaluation requires a measured worker runtime; repeat the legacy analysis")
        frames = {}
        project = projections.get_project(project_id)
        partitions = store.all_partitions(run_id)
        if (project is None or run.manifest.project_id != str(project_id) or run.fingerprint != run.manifest.fingerprint
                or len(partitions) != run.total_partitions
                or {p.partition_index for p in partitions} != set(range(run.total_partitions))):
            raise DomainValidationError("Analysis manifest/partition identity is inconsistent")
        for partition in partitions:
            if (partition.project_id != str(project_id) or partition.run_id != run_id
                    or partition.run_fingerprint != run.fingerprint or partition.partition_count != run.total_partitions
                    or partition.recipe != run.manifest.recipe or partition.parameters != run.manifest.parameters):
                raise DomainValidationError("Analysis partition computation identity does not match its run")
            for frame in partition.frames:
                if frame.frame_id in frames or str(project.frame_id_at(frame.x, frame.y)) != frame.frame_id:
                    raise DomainValidationError("Analysis input frame identity/coordinates are inconsistent")
                for artifact in frame.artifacts:
                    version = projections.get_artifact_version(validate_uuid(artifact.artifact_version_id))
                    series = None if version is None else projections.get_artifact_series(version.series_id)
                    if (series is None or str(series.project_id) != str(project_id) or str(series.id) != artifact.artifact_id
                            or str(series.frame_id) != frame.frame_id or version.sha256 != artifact.sha256):
                        raise DomainValidationError("Analysis inputs do not resolve to the declared immutable project artifacts")
                frames[frame.frame_id] = frame
        if set(frames) != set(run.manifest.frame_ids) or len(frames) != run.total_frames:
            raise DomainValidationError("Analysis partitions do not cover the run frames exactly")
        rows = {}
        for raw in store.frame_results(run_id, metric_key or ""):
            row = dict(raw)
            identifier = row["frame_id"]
            if identifier in rows or identifier not in frames:
                raise DomainValidationError("Analysis results contain duplicate or undeclared frames")
            frame = frames[identifier]
            if row["status"] != "ready" or row["x"] != frame.x or row["y"] != frame.y:
                raise ConflictError("Quality evaluation requires ready analysis results with matching coordinates")
            rows[identifier] = row
        if set(rows) != set(frames):
            raise ConflictError("Analysis metric results are incomplete")
        return run, frames, rows

    @classmethod
    def create_findings(cls, unit, store, *, profiles, performers, project_id, principal,
                        run_id, rule, severity, idempotency_key, gitlab_identity_verified=False):
        project = cls._authorize(unit, profiles, project_id, principal, gitlab_identity_verified=gitlab_identity_verified)
        require_non_empty(idempotency_key, field="idempotency_key")
        if not isinstance(rule, dict) or set(rule) != cls.RULE_FIELDS:
            raise DomainValidationError("An analysis quality rule requires explicit metric, field, comparison, threshold and target binding")
        if severity not in {"blocker", "critical", "major", "minor", "improvement"}:
            raise DomainValidationError("Unknown quality severity")
        command = freeze_mapping({"run_id": run_id, "rule": rule, "severity": severity})
        prior = unit.event_store.find_by_idempotency_key(project.id, idempotency_key)
        if prior:
            event = prior[-1]
            if event.event_type != "quality.analysis_evaluated" or event.payload.get("command") != command:
                raise ConflictError("Idempotency key belongs to another analysis quality evaluation")
            return tuple(QualityIssue.from_dict(item) for item in event.payload["issues"])
        run, frames, rows = cls.snapshot(store, unit.projections, project.id, run_id, rule["metric_key"])
        policy = QualityMetricRule(**rule, analysis_signature=cls.signature(run.manifest, run.worker_runtime))
        if policy.target_binding not in run.manifest.recipe.expression.source_keys:
            raise DomainValidationError("The quality target binding must participate in the actual analysis expression")
        issues = []
        for identifier, frame in frames.items():
            value = rows[identifier].get(policy.value_field)
            failed = policy.failed(value)  # Missing/non-finite values never become a passing decision.
            targets = tuple(a for a in frame.artifacts if a.binding_key == policy.target_binding)
            if not targets:
                raise DomainValidationError("The selected target binding is absent from an analysis frame")
            if not failed:
                continue
            versions = tuple(a.artifact_version_id for a in targets)
            layers = {str(unit.projections.get_artifact_series(a.artifact_id).layer_id) for a in targets}
            child_key = f"qa:{sha256(idempotency_key.encode()).hexdigest()}:{identifier}"
            issue = QualityService.execute(unit, profiles=profiles, performers=performers, project_id=project.id,
                principal=principal, operation="create", idempotency_key=child_key,
                arguments={"title": f"{policy.metric_key}: {value:g} {policy.comparison} {policy.threshold:g}",
                    "description": f"Analysis run {run_id}, frame ({frame.x}, {frame.y}), stored {policy.value_field}={value:g}",
                    "severity": severity, "scope": {"layer_ids": sorted(layers), "frame_ids": [identifier]},
                    "source_version_ids": list(versions)}, analysis_rule=policy, analysis_run_id=run_id,
                analysis_observations=((identifier, value),), gitlab_identity_verified=gitlab_identity_verified)
            issues.append(issue)
        # A zero-finding evaluation is durable too, so retries cannot silently
        # change policy and reuse a command key after an apparently empty result.
        stream = f"quality-evaluation:{uuid5(UUID(str(project.id)), idempotency_key)}"
        event = EventEnvelope.create(stream_id=stream, project_id=project.id, revision=1,
            event_type="quality.analysis_evaluated", payload={"command": command, "issues": [i.to_dict() for i in issues],
                "run_fingerprint": run.fingerprint, "rule": asdict(policy)},
            actor=ActorSnapshot.from_principal(principal), idempotency_key=idempotency_key, recorded_at=datetime.now(UTC))
        unit.event_store.append(stream, expected_revision=0, events=(event,))
        return tuple(issues)

    @classmethod
    def recheck(cls, unit, store, *, profiles, performers, project_id, principal, issue_id,
                run_id, expected_revision, idempotency_key, note, gitlab_identity_verified=False):
        cls._authorize(unit, profiles, project_id, principal, gitlab_identity_verified=gitlab_identity_verified)
        issue = QualityService.load(unit.event_store, project_id, issue_id)
        policy = issue.analysis_rule
        if policy is None:
            raise DomainValidationError("This finding has no recorded analysis quality rule")
        run, frames, rows = cls.snapshot(store, unit.projections, project_id, run_id, policy.metric_key)
        if cls.signature(run.manifest, run.worker_runtime) != policy.analysis_signature:
            raise DomainValidationError("Recheck requires the same analysis recipe, parameters and runtime identity")
        _, original, _ = cls.snapshot(store, unit.projections, project_id, issue.analysis_run_id)
        observations = []
        versions = set()
        for identifier in issue.scope.frame_ids:
            if identifier not in frames or identifier not in original:
                raise DomainValidationError("Recheck does not cover every affected frame")
            previous = {a.binding_key: a for a in original[identifier].artifacts}
            current = {a.binding_key: a for a in frames[identifier].artifacts}
            if set(current) != set(previous):
                raise DomainValidationError("Recheck source bindings differ from the original analysis")
            for binding, artifact in current.items():
                if binding == policy.target_binding:
                    versions.add(artifact.artifact_version_id)
                elif artifact.artifact_version_id != previous[binding].artifact_version_id:
                    raise DomainValidationError("Recheck changed an unaffected reference input")
            value = rows[identifier].get(policy.value_field)
            policy.failed(value)
            observations.append((identifier, value))
        if versions != set(issue.correction_version_ids) or not versions:
            raise DomainValidationError("Recheck must actually analyze the recorded correction versions")
        passed = not any(policy.failed(value) for _, value in observations)
        return QualityService.execute(unit, profiles=profiles, performers=performers, project_id=project_id,
            principal=principal, issue_id=issue.id, expected_revision=expected_revision,
            operation="recheck", arguments={"passed": passed, "evidence_version_ids": sorted(versions), "note": note},
            idempotency_key=idempotency_key, analysis_run_id=run_id, analysis_observations=tuple(observations),
            gitlab_identity_verified=gitlab_identity_verified)

    @classmethod
    def readiness(cls, events, store, projections, project_id, run_id):
        try:
            run, frames, _ = cls.snapshot(store, projections, project_id, run_id)
        except ConflictError as exc:
            return {"run_id": run_id, "computational_ready": False, "ready": False, "reason": str(exc), "blocking_issue_ids": []}
        versions = {a.artifact_version_id for frame in frames.values() for a in frame.artifacts}
        blocked = QualityService.blocking_issues(events, project_id, versions, projections=projections)
        return {"run_id": run_id, "run_fingerprint": run.fingerprint, "computational_ready": True,
                "ready": not blocked, "reason": "" if not blocked else "Quality findings prohibit these input versions",
                "blocking_issue_ids": [issue.id for issue in blocked]}
