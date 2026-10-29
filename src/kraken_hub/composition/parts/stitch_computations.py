"""Recoverable local stitch handoff and atomic layer-result import."""
from kraken_core.safe_files import open_regular_read
from kraken_manager.application.errors import ConflictError, NotFoundError
from kraken_manager.application.multilayer import MultilayerService
from kraken_manager.application.stitch_results import _context, fail_stitch_result, import_stitch_result
from kraken_manager.application.stitch_pipeline import check_stitch_owner
from kraken_manager.infrastructure.mosaic import MosaicReader


def validate_mosaic(path, plan, job_id):
    with MosaicReader(path) as reader:
        return reader.validate(expected_plan=plan, expected_job_id=job_id)


class StitchComputationsMixin:
    def stitch_block(self, project_id, job_id, column, row, *, coverage=False):
        from kraken_manager.infrastructure.mosaic import block_preview
        from kraken_manager.infrastructure.stitch_reads import published_mosaic

        with self._uow(str(project_id))() as unit, published_mosaic(
            unit.event_store, unit.projections, unit.blobs.path_for_read, project_id, job_id,
        ) as (reader, _version):
            return block_preview(reader, column, row, coverage=coverage)

    def stitch_vector_header(self, project_id, job_id):
        from kraken_manager.infrastructure.stitch_reads import published_mosaic, vector_header
        with self._uow(str(project_id))() as unit, published_mosaic(unit.event_store, unit.projections,
                unit.blobs.path_for_read, project_id, job_id) as (reader, version):
            return vector_header(reader, version, project_id)

    def stitch_vector_block(self, project_id, job_id, column, row):
        from kraken_manager.infrastructure.stitch_reads import published_mosaic, vector_block
        with self._uow(str(project_id))() as unit, published_mosaic(unit.event_store, unit.projections,
                unit.blobs.path_for_read, project_id, job_id) as (reader, version):
            return vector_block(reader, version, column, row)

    def export_stitch_vectors(self, *, project_id, job_id, destination, cancelled=lambda: False):
        from contextlib import ExitStack
        from kraken_manager.infrastructure.stitch_reads import published_mosaic, vector_header, vector_block
        from kraken_manager.infrastructure.vector_exchange import write_vector_exchange
        with ExitStack() as stack:
            with self._uow(str(project_id))() as unit:
                reader, version = stack.enter_context(published_mosaic(unit.event_store, unit.projections,
                    unit.blobs.path_for_read, project_id, job_id))
            # The accepted artifact is immutable. Do not keep the project write
            # lock while serializing or verifying a potentially large export.
            return write_vector_exchange(destination, vector_header(reader, version, project_id),
                lambda column, row: vector_block(reader, version, column, row), cancelled=cancelled)

    def import_stitch_observations(self, *, project_id, layer_id, principal, source_path, idempotency_key):
        """Import expert observations as exact JSON evidence and a physical revision."""
        import hashlib
        import json
        from uuid import UUID, uuid5
        from kraken_core.safe_files import open_regular_read
        from kraken_manager.application.dto import AddArtifactVersionCommand, CommandContext
        from kraken_manager.domain.artifacts import ArtifactScope

        with open_regular_read(source_path) as stream:
            data = stream.read(16 * 1024**2 + 1)
        if len(data) > 16 * 1024**2:
            raise ValueError("Stitch observations exceed the JSON size limit")
        arguments = json.loads(data.decode("utf-8-sig"))
        if not isinstance(arguments, dict) or str(arguments.get("layer_id", layer_id)) != str(layer_id):
            raise ValueError("Stitch observations belong to another layer")
        arguments = {**arguments, "layer_id": str(layer_id)}
        if set(arguments) - {"layer_id", "reference_frame_id", "tiles", "max_error_m", "max_condition"}:
            raise ValueError("Unknown stitch observation fields")
        series = self.create_artifact_series(principal=principal, project_id=project_id, layer_id=layer_id,
            scope=ArtifactScope.LAYER_ATTACHMENT, name="Stitch observations",
            series_id=str(uuid5(UUID(str(project_id)), "stitch-observations:" + idempotency_key)),
            idempotency_key=idempotency_key + ":series")
        digest = hashlib.sha256(data).hexdigest()
        version = self._commands(str(project_id)).execute(
            AddArtifactVersionCommand(context=CommandContext(actor=principal, idempotency_key=idempotency_key + ":evidence"),
                project_id=str(project_id), series_id=series.id, filename="stitch-observations.json", media_type="application/json",
                expected_series_revision=self.artifact_stream_revision(project_id, series.id), expected_sha256=digest), (data,))
        if version.sha256 != digest:
            raise ConflictError("Observation import key already belongs to another JSON document")
        return self.mutate_multilayer_model(project_id=project_id, principal=principal,
            expected_revision=self.get_multilayer_model(project_id)["model"]["revision"], operation="plan_layer_stitch",
            arguments={**arguments, "plan_version_id": str(version.id)}, idempotency_key=idempotency_key + ":geometry")

    def _stitch_context(self, unit, project_id, job_id, principal):
        project = unit.projections.get_project(str(project_id))
        if project is None:
            raise NotFoundError("Project not found")
        return _context(unit, str(project_id), str(job_id), principal, self.profiles.get(project.storage_profile),
                        unit.acl.roles_for(project.id, principal.id), False)

    def dispatch_stitch(self, *, project_id, job_id, principal, gateway):
        with self._uow(str(project_id))() as unit:
            service, model, request, manifest = self._stitch_context(unit, project_id, job_id, principal)
            stitch = next(s for s in model.stitches if (s.id, s.revision) == (request.stitch_id, request.stitch_revision))
            if request.state == "requested" and not service.stitch_current(model, stitch):
                raise ConflictError("Stitch geometry or source is stale")
            allowed = request.state != "requested" or check_stitch_owner(unit.event_store, unit.projections, manifest, dispatch=True)
        if not allowed:
            return self.get_multilayer_model(project_id)
        if request.state != "requested":
            return self.refresh_stitch(project_id=project_id, job_id=job_id, principal=principal, gateway=gateway)
        paths = {asset.asset_id: self.artifact_source_path(project_id, asset.asset_id) for asset in manifest.inputs}
        gateway.submit(manifest, paths)
        return self.get_multilayer_model(project_id)

    def import_stitch_publication(self, *, project_id, principal, publication, source_path):
        with self._uow(str(project_id))() as unit:
            project = unit.projections.get_project(str(project_id))
            model = import_stitch_result(unit, project_id=str(project_id), principal=principal,
                storage=self.profiles.get(project.storage_profile), roles=unit.acl.roles_for(project.id, principal.id),
                publication=publication, source_path=source_path, validate_mosaic=validate_mosaic, open_source=open_regular_read)
            unit.commit()
            return MultilayerService(unit.event_store, unit.projections).describe(model)

    def refresh_stitch(self, *, project_id, job_id, principal, gateway):
        with self._uow(str(project_id))() as unit:
            _, _, request, manifest = self._stitch_context(unit, project_id, job_id, principal)
        try:
            agent_job = gateway.get(str(job_id))
        except KeyError:
            if request.state == "requested":
                return self.dispatch_stitch(project_id=project_id, job_id=job_id, principal=principal, gateway=gateway)
            return self.get_multilayer_model(project_id)
        if request.state == "cancelled":
            if agent_job.state not in {"succeeded", "failed", "cancelled"}:
                gateway.cancel(agent_job)
            return self.get_multilayer_model(project_id)
        if request.state == "imported":
            if agent_job.state == "importing":
                gateway.complete_import(agent_job)
            return self.get_multilayer_model(project_id)
        if request.state == "failed":
            return self.get_multilayer_model(project_id)
        if agent_job.state in {"importing", "succeeded"}:
            publication, path = gateway.result(manifest)
            result = self.import_stitch_publication(project_id=project_id, principal=principal,
                publication=publication, source_path=path)
            if agent_job.state == "importing":
                gateway.complete_import(agent_job)
            return result
        if agent_job.state in {"failed", "cancelled"}:
            try:
                publication, _ = gateway.result(manifest)
                error = publication.report.get("error", agent_job.error)
            except (OSError, ValueError):
                error = agent_job.error
            with self._uow(str(project_id))() as unit:
                project = unit.projections.get_project(str(project_id))
                fail_stitch_result(unit, project_id=str(project_id), job_id=str(job_id), principal=principal,
                    storage=self.profiles.get(project.storage_profile), roles=unit.acl.roles_for(project.id, principal.id), error=error)
                unit.commit()
        return self.get_multilayer_model(project_id)
