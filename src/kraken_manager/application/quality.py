"""Transactional quality commands over existing EventStore/UOW and artifacts."""

from __future__ import annotations

from datetime import UTC, datetime
from dataclasses import asdict
from uuid import UUID, uuid5

from .authorization import AuthorizationPolicy
from .errors import AuthorizationError, ConcurrencyError, ConflictError, NotFoundError
from kraken_manager.domain.common import DomainValidationError, freeze_mapping, require_non_empty, validate_uuid
from kraken_manager.domain.events import ActorSnapshot, EventEnvelope
from kraken_manager.domain.identity import Permission
from kraken_manager.domain.planning import WorkScope
from kraken_manager.domain.project import ProjectState, StructureState
from kraken_manager.domain.quality import QualityIssue, QualityState, QualityMetricRule


class QualityService:
    PERMISSIONS = {
        "create": Permission.MANAGE_REVIEW, "assign": Permission.ASSIGN_WORK,
        "correct": Permission.RETURN_REVIEW, "recheck": Permission.MANAGE_REVIEW,
        "accept": Permission.ACCEPT_REVIEW, "reject": Permission.ACCEPT_REVIEW,
        "exception": Permission.ACCEPT_REVIEW,
    }
    FIELDS = {
        "create": {"title", "description", "severity", "scope", "source_version_ids", "review_batch_id"},
        "assign": {"assignee_id", "work_order_id", "note"},
        "correct": {"evidence_version_ids", "note"},
        "recheck": {"evidence_version_ids", "passed", "note"},
        "accept": {"note"}, "reject": {"note"}, "exception": {"note"},
    }

    @staticmethod
    def frame_versions(projections, project_id, layer_id, x, y):
        from kraken_manager.domain.artifacts import deterministic_frame_series_id
        project = projections.get_project(validate_uuid(str(project_id)))
        layer = projections.get_layer(validate_uuid(str(layer_id)))
        if project is None or layer is None or layer.project_id != project.id:
            raise NotFoundError("Quality artifact layer was not found in the project")
        frame_id = project.frame_id_at(x, y)
        versions = []
        for representation in projections.list_representations(layer.id):
            series_id = deterministic_frame_series_id(representation.id, frame_id)
            if projections.get_artifact_series(series_id) is not None:
                versions.extend(projections.list_artifact_versions(series_id))
        return tuple(versions)

    @staticmethod
    def stream_id(issue_id):
        return f"quality-issue:{validate_uuid(str(issue_id))}"

    @classmethod
    def load(cls, events, project_id, issue_id):
        history = events.load_stream(cls.stream_id(issue_id))
        if not history:
            raise NotFoundError("Quality issue was not found")
        issue = QualityIssue.from_dict(history[-1].payload["issue"])
        if issue.project_id != str(project_id):
            raise NotFoundError("Quality issue belongs to another project")
        if issue.id != str(issue_id) or issue.revision + 1 != history[-1].revision:
            raise DomainValidationError("Quality stream identity/revision mismatch")
        return issue

    @staticmethod
    def list_issues(events, project_id):
        project_id = validate_uuid(str(project_id))
        latest = {}
        if hasattr(events, "list_project_events"):
            position = 0
            while batch := events.list_project_events(project_id, after_position=position, limit=1000):
                for position, event in batch:
                    if event.stream_id.startswith("quality-issue:"):
                        latest[event.stream_id] = event.payload["issue"]
        else:
            # Filesystem UoW exposes its backing log for cross-stream reads;
            # pending events are included so gates also work in a transaction.
            backing = getattr(events, "backing", events)
            for event in backing.iter_project():
                if event.stream_id.startswith("quality-issue:"):
                    latest[event.stream_id] = event.data["payload"]["issue"]
            for event in getattr(events, "_pending", ()):
                if event.stream_id.startswith("quality-issue:"):
                    latest[event.stream_id] = event.payload["issue"]
        issues = (QualityIssue.from_dict(payload) for payload in latest.values())
        return tuple(sorted((i for i in issues if i.project_id == project_id),
                            key=lambda i: (i.updated_at, i.id), reverse=True))

    @classmethod
    def blocking_issues(cls, events, project_id, artifact_version_ids, *, scope=None, projections=None):
        direct_versions = {validate_uuid(str(value)) for value in artifact_version_ids}
        versions = direct_versions
        if projections is not None:
            from kraken_manager.application.artifact_lineage import artifact_ancestors
            versions = artifact_ancestors(projections, project_id, versions)
        blocked = []
        for issue in cls.list_issues(events, project_id):
            if issue.state is QualityState.EXCEPTION or issue.severity not in {"major", "critical", "blocker"}:
                continue
            if issue.state is QualityState.ACCEPTED:
                bad_versions = set(issue.source_version_ids)
                bad_versions.update(v for review in issue.reviews if review.action == "correct"
                                    for v in review.evidence_version_ids if v not in issue.correction_version_ids)
                # An accepted correction certifies its branch, not every other
                # route by which a dependent artifact could ingest defective data.
                uncertified = (direct_versions if projections is None else artifact_ancestors(
                    projections, project_id, direct_versions, stop_at=issue.correction_version_ids))
                if bad_versions & uncertified:
                    blocked.append(issue)
                continue
            affected = set(issue.source_version_ids) | set(issue.correction_version_ids)
            affected.update(v for review in issue.reviews for v in review.evidence_version_ids)
            matches_scope = False
            if scope is not None:
                layers = not scope.layer_ids or not issue.scope.layer_ids or bool(set(scope.layer_ids) & set(issue.scope.layer_ids))
                frames = not scope.frame_ids or not issue.scope.frame_ids or bool(set(scope.frame_ids) & set(issue.scope.frame_ids))
                matches_scope = layers and frames
            if affected & versions or matches_scope:
                blocked.append(issue)
        return tuple(blocked)

    @classmethod
    def require_clear_inputs(cls, events, project_id, artifact_version_ids, *, scope=None, projections=None):
        blocked = cls.blocking_issues(events, project_id, artifact_version_ids, scope=scope, projections=projections)
        if blocked:
            raise ConflictError("Quality findings block acceptance or dispatch of these inputs: " + ", ".join(i.id for i in blocked))

    @classmethod
    def execute(cls, unit, *, profiles, performers, project_id, principal, operation, arguments,
                idempotency_key, issue_id=None, expected_revision=None, gitlab_identity_verified=False,
                analysis_rule: QualityMetricRule | None = None, analysis_run_id=None, analysis_observations=None):
        project = unit.projections.get_project(validate_uuid(str(project_id)))
        if project is None:
            raise NotFoundError("Project was not found")
        if project.state is not ProjectState.ACTIVE:
            raise ConflictError("Archived projects cannot change quality issues")
        if operation not in cls.FIELDS or not isinstance(arguments, dict) or set(arguments) - cls.FIELDS[operation]:
            raise DomainValidationError("Invalid quality command fields")
        AuthorizationPolicy().require(
            principal=principal, storage=profiles.get(project.storage_profile), permission=cls.PERMISSIONS[operation],
            roles=unit.acl.roles_for(project.id, principal.id), gitlab_identity_verified=gitlab_identity_verified,
        )
        require_non_empty(idempotency_key, field="idempotency_key")
        if operation == "create":
            if expected_revision not in {None, 0}:
                raise ConcurrencyError("A new issue requires revision zero")
            issue_id = str(uuid5(UUID(str(project.id)), f"quality:{idempotency_key}"))
        elif isinstance(expected_revision, bool) or not isinstance(expected_revision, int) or expected_revision < 0:
            raise ConcurrencyError("Expected quality revision is required")
        command = {"operation": operation, "issue_id": issue_id, "arguments": arguments}
        if analysis_run_id is not None:
            command["analysis_run_id"] = analysis_run_id
            command["analysis_rule"] = None if analysis_rule is None else asdict(analysis_rule)
            command["observations"] = analysis_observations
        signature = freeze_mapping(command)
        prior = unit.event_store.find_by_idempotency_key(project.id, idempotency_key)
        if prior:
            if prior[-1].stream_id != cls.stream_id(issue_id) or prior[-1].payload.get("command") != signature:
                raise ConflictError("Idempotency key belongs to another quality command")
            return QualityIssue.from_dict(prior[-1].payload["issue"])
        now = datetime.now(UTC)
        if operation == "create":
            scope = cls._scope(unit.projections, project, arguments["scope"])
            cls._versions(unit.projections, project, scope, arguments["source_version_ids"])
            batch_id = arguments.get("review_batch_id")
            if batch_id is not None:
                cls._owned_reference(unit.projections.get_review_batch(batch_id), project, scope, "review batch")
            issue = QualityIssue(
                id=issue_id, project_id=str(project.id), title=arguments["title"],
                description=arguments["description"], severity=arguments["severity"], scope=scope,
                source_version_ids=tuple(arguments["source_version_ids"]), review_batch_id=batch_id,
                created_by=str(principal.id), created_at=now, updated_at=now,
                analysis_run_id=analysis_run_id, analysis_rule=analysis_rule,
            )
            stream_revision = 0
        else:
            current = cls.load(unit.event_store, project.id, issue_id)
            if current.revision != expected_revision:
                raise ConcurrencyError(f"Expected revision {expected_revision}, found {current.revision}")
            args = dict(arguments)
            if operation == "recheck" and current.analysis_rule is not None:
                if analysis_run_id is None or analysis_observations is None:
                    raise DomainValidationError("Analysis findings must be rechecked against a completed stored analysis run")
                args["_analysis_run_id"] = analysis_run_id
                args["_observations"] = analysis_observations
            if operation == "assign":
                performer = None if performers is None else performers.get(validate_uuid(args["assignee_id"]))
                if performer is None or not performer.active:
                    raise DomainValidationError("Quality assignee is unknown or inactive")
                if args.get("work_order_id"):
                    order = unit.projections.get_work_order(validate_uuid(args["work_order_id"]))
                    cls._owned_reference(order, project, current.scope, "work order")
                    if str(order.assignee_id) != str(performer.id):
                        raise DomainValidationError("Correction work order has a different assignee")
            if "evidence_version_ids" in args:
                series = cls._versions(unit.projections, project, current.scope, args["evidence_version_ids"])
                if operation == "correct":
                    assigned = None if performers is None else performers.get(current.assignee_id)
                    if assigned is not None and assigned.principal_id is not None and str(assigned.principal_id) != str(principal.id):
                        # Managers may record a manual performer's return, but
                        # another corrector cannot impersonate a linked assignee.
                        AuthorizationPolicy().require(
                            principal=principal, storage=profiles.get(project.storage_profile),
                            permission=Permission.MANAGE_REVIEW, roles=unit.acl.roles_for(project.id, principal.id),
                            gitlab_identity_verified=gitlab_identity_verified,
                        )
                    source_series = cls._versions(unit.projections, project, current.scope, current.source_version_ids)
                    if series != source_series:
                        raise DomainValidationError("Corrections must version every affected artifact series")
            if operation in {"recheck", "accept"}:
                correction = next((r for r in reversed(current.reviews) if r.action == "correct"), None)
                if correction is not None and correction.actor_id == str(principal.id):
                    raise AuthorizationError("Correction authors cannot review their own correction")
            issue = current.change(operation, args, actor_id=str(principal.id), at=now)
            stream_revision = current.revision + 1
        event = EventEnvelope.create(
            stream_id=cls.stream_id(issue.id), project_id=project.id, revision=stream_revision + 1,
            event_type=f"quality.{operation}", payload={"schema_version": 1, "issue": issue.to_dict(), "command": signature},
            actor=ActorSnapshot.from_principal(principal), performer_id=issue.assignee_id,
            idempotency_key=idempotency_key, recorded_at=now,
        )
        unit.event_store.append(event.stream_id, expected_revision=stream_revision, events=(event,))
        return issue

    @staticmethod
    def _owned_reference(reference, project, scope, label):
        if reference is None or str(reference.project_id) != str(project.id):
            raise DomainValidationError(f"Quality {label} belongs to another project or is unknown")
        if scope.layer_ids and str(reference.layer_id) not in scope.layer_ids:
            raise DomainValidationError(f"Quality {label} is outside the issue layers")

    @classmethod
    def _scope(cls, projections, project, payload):
        if not isinstance(payload, dict):
            raise DomainValidationError("Quality scope must be an object")
        data = dict(payload)
        coordinates = data.pop("frame_coordinates", None)
        if coordinates is not None:
            if data.get("frame_ids"):
                raise DomainValidationError("Use frame IDs or coordinates, not both")
            data["frame_ids"] = tuple(str(project.frame_id_at(*xy)) for xy in coordinates)
        scope = WorkScope(**data)
        for identifier in scope.layer_ids:
            layer = projections.get_layer(identifier)
            if layer is None or layer.project_id != project.id or layer.state is StructureState.ARCHIVED:
                raise DomainValidationError("Quality scope references an unknown, foreign or archived layer")
        return scope

    @staticmethod
    def _versions(projections, project, scope, identifiers):
        if not isinstance(identifiers, (list, tuple)) or not identifiers:
            raise DomainValidationError("Quality evidence requires immutable artifact version IDs")
        series_ids = set()
        for identifier in identifiers:
            version = projections.get_artifact_version(validate_uuid(identifier))
            series = None if version is None else projections.get_artifact_series(version.series_id)
            if series is None or series.project_id != project.id:
                raise DomainValidationError("Quality artifact version is unknown or belongs to another project")
            if scope.layer_ids and str(series.layer_id) not in scope.layer_ids:
                raise DomainValidationError("Quality artifact is outside the issue layers")
            if scope.frame_ids and str(series.frame_id) not in scope.frame_ids:
                raise DomainValidationError("Quality artifact is outside the issue frames")
            series_ids.add(str(series.id))
        return series_ids
