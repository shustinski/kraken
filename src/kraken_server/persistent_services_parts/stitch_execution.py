"""Physical stitch intent -> existing leased Agent queue -> atomic layer import."""
import logging

from kraken_core.plugin_protocol import PluginJobManifestV2, PluginResultPublicationV2
from kraken_manager.application.multilayer import MultilayerService
from kraken_manager.application.stitch_results import _context, fail_stitch_result, reject_stitch_authorization
from kraken_manager.application.stitch_pipeline import check_stitch_owner
from kraken_manager.domain.project import ProjectState
from kraken_server.services import CommandContext, ConflictError


class StitchExecutionMixin:
    def _stitch_vector_read(self, project_id, job_id, column=None, row=None):
        import sqlite3
        from kraken_manager.infrastructure.stitch_reads import published_mosaic, vector_header, vector_block
        from kraken_manager.application.errors import NotFoundError as ModelNotFoundError, ConflictError as ModelConflictError
        from kraken_server.services import NotFoundError, ValidationError
        try:
            with published_mosaic(self.events, self.projections, self.uow_factory.blobs.path_for_read,
                    project_id, job_id) as (reader, version):
                if column is None:
                    return vector_header(reader, version, project_id)
                left, top, right, bottom = reader.renderer.bounds
                step = reader.renderer.limits.block_size
                if min(column, row) < 0 or column >= (right-left+step-1)//step or row >= (bottom-top+step-1)//step:
                    raise ValidationError("Vector block index is outside its grid")
                return vector_block(reader, version, column, row)
        except ModelNotFoundError as exc:
            raise NotFoundError(str(exc)) from exc
        except ModelConflictError as exc:
            raise ConflictError(str(exc)) from exc
        except (ConflictError, NotFoundError, ValidationError):
            raise
        except (ValueError, OSError, sqlite3.DatabaseError, KeyError, TypeError) as exc:
            raise ConflictError("Stored mosaic vectors failed integrity validation") from exc

    def stitch_vector_header(self, project_id, job_id):
        return self._stitch_vector_read(project_id, job_id)

    def stitch_vector_block(self, project_id, job_id, column, row):
        return self._stitch_vector_read(project_id, job_id, column, row)

    def stitch_block_preview(self, project_id, job_id, column, row, *, coverage=False):
        import base64
        import sqlite3
        from kraken_manager.domain.common import validate_uuid
        from kraken_manager.infrastructure.mosaic import MosaicReader, block_preview
        from kraken_server.services import NotFoundError, ValidationError

        try:
            model = MultilayerService(self.events, self.projections).load(str(project_id))
            identifier = validate_uuid(str(job_id))
            request = next((r for r in model.stitch_renders if r.job_id == identifier), None)
            if request is None or request.state != "imported":
                raise NotFoundError("Published mosaic was not found in this project")
            plan = next(s for s in model.stitches if (s.id, s.revision) == (request.stitch_id, request.stitch_revision))
            manifest = PluginJobManifestV2.from_json(request.manifest_json)
            version = self.projections.get_artifact_version(request.output.version_id)
            if version is None or version.blob is None or version.sha256 != request.output.sha256:
                raise ConflictError("Published mosaic artifact is missing or inconsistent")
            series = self.projections.get_artifact_series(version.series_id)
            if (series is None or str(series.project_id) != str(project_id) or str(series.layer_id) != plan.layer_id
                    or series.scope.value != "layer_attachment" or series.frame_id is not None or series.representation_id is not None):
                raise ConflictError("Mosaic artifact ownership does not match the requested layer")
            with MosaicReader(self.uow_factory.blobs.path_for_read(version.blob)) as reader:
                reader.check_request(expected_plan=plan, expected_job_id=identifier,
                    mode=manifest.parameters["mode"], block_size=manifest.parameters["block_size"])
                left, top, right, bottom = reader.renderer.bounds
                step = reader.renderer.limits.block_size
                if (any(isinstance(v, bool) or not isinstance(v, int) or v < 0 for v in (column, row))
                        or column >= (right-left+step-1)//step or row >= (bottom-top+step-1)//step):
                    raise ValidationError("Block index is outside the mosaic grid")
                data, metadata, x, y, columns, rows = block_preview(reader, column, row, coverage=coverage)
            metadata.update(output_version_id=str(version.id), output_sha256=version.sha256,
                            stitch_id=plan.id, stitch_revision=plan.revision)
            return {"image_base64": base64.b64encode(data).decode("ascii"), "metadata": metadata,
                    "x": x, "y": y, "columns": columns, "rows": rows}
        except (NotFoundError, ConflictError, ValidationError):
            raise
        except (ValueError, OSError, sqlite3.DatabaseError, KeyError, TypeError) as exc:
            raise ConflictError("Stored mosaic block failed integrity validation") from exc

    def dispatch_stitch_project(self, project_id):
        if self.agent_gateway is None or not hasattr(self.agent_gateway, "submit_stitch"):
            return
        service = MultilayerService(self.events, self.projections)
        model = service.load(str(project_id))
        project = service._project(project_id)
        for request in model.stitch_renders:
            try:
                self._dispatch_stitch_request(service, project, request)
            except Exception:
                logging.getLogger(__name__).exception("Stitch request remains pending: %s", request.job_id)

    def _dispatch_stitch_request(self, service, project, request):
        if request.state == "imported":
            self.agent_gateway.reconcile_imported_stitch(request)
            return
        if request.state == "failed":
            self.agent_gateway.cancel(request.job_id)
            return
        if request.state == "cancelled" or project.state is not ProjectState.ACTIVE:
            self.agent_gateway.cancel(request.job_id)
            return
        manifest = PluginJobManifestV2.from_json(request.manifest_json)
        with self.uow_factory() as unit:
            rejected = reject_stitch_authorization(unit, project_id=str(project.id), job_id=request.job_id,
                identities=self.identities, storage=self.profiles.get(project.storage_profile))
            if rejected is not None:
                unit.commit()
        if rejected is not None:
            self.agent_gateway.cancel(request.job_id)
            return
        actor = self._actor(manifest.actor_id)
        with self.uow_factory() as unit:
            current_service, current, active, canonical = _context(unit, str(project.id), request.job_id, actor,
                self.profiles.get(project.storage_profile), unit.acl.roles_for(project.id, actor.id), True)
            stitch = next(s for s in current.stitches if (s.id, s.revision) == (active.stitch_id, active.stitch_revision))
            if active.state != "requested":
                return
            reason = None
            try:
                if not check_stitch_owner(unit.event_store, unit.projections, canonical, dispatch=True):
                    return
            except ConflictError as exc:
                reason = str(exc)
            if reason is not None or not current_service.stitch_current(current, stitch):
                fail_stitch_result(unit, project_id=str(project.id), job_id=request.job_id, principal=actor,
                    storage=self.profiles.get(project.storage_profile), roles=unit.acl.roles_for(project.id, actor.id),
                    error=reason or "Stitch inputs or geometry became stale before dispatch", verified=True)
                unit.commit()
                stale = True
            else:
                stale = False
        if stale:
            self.agent_gateway.cancel(request.job_id)
            return
        self.agent_gateway.submit_stitch(canonical)
        # Cancellation can commit after the read but before enqueue. Its
        # durable physical state wins; import rechecks it independently.
        latest = service.load(str(project.id))
        active = next(r for r in latest.stitch_renders if r.job_id == request.job_id)
        if active.state != "requested":
            self.agent_gateway.cancel(request.job_id)

    def run_stitch_cycle(self):
        if self.agent_gateway is None or not hasattr(self.agent_gateway, "submit_stitch"):
            return 0
        count = 0
        for project in self.projections.list_projects(include_archived=True):
            try:
                self.dispatch_stitch_project(str(project.id))
                count += 1
            except Exception:
                logging.getLogger(__name__).exception("Stitch dispatch intent remains pending: %s", project.id)
        return count

    def import_stitch_agent_result(self, job_id, agent):
        manifest = self.agent_gateway.manifest(job_id, agent)
        if not isinstance(manifest, PluginJobManifestV2) or manifest.operation != "layer.stitch.v1":
            raise ConflictError("Leased job is not a stitch request")
        publications = self.agent_gateway.publications_for(job_id, agent)
        if len(publications) != 1:
            raise ConflictError("Exactly one final mosaic publication is required")
        result = PluginResultPublicationV2.from_dict(publications[0])
        if result.job_id != job_id:
            raise ConflictError("Mosaic publication belongs to another leased job")
        if not result.final or result.outcome not in {"succeeded", "failed", "cancelled"}:
            raise ConflictError("Partial mosaic publication is forbidden")
        if result.outcome != "succeeded":
            if result.outputs:
                raise ConflictError("Failed mosaic cannot publish outputs")
            return self.fail_stitch_agent_job(job_id, result.report.get("error", result.outcome))
        if len(result.outputs) != 1:
            raise ConflictError("Exactly one mosaic artifact is required")
        reference = self.agent_gateway.output_blob(job_id, result.outputs[0].asset_id)
        if reference.sha256 != result.outputs[0].sha256:
            raise ConflictError("Uploaded mosaic digest does not match its publication")
        # Uploaded content already lives in the same immutable BlobStore. The
        # shared importer verifies the complete block structure before commit.
        return self.commit_stitch_publication(manifest.project_id, result,
            self.uow_factory.blobs.path_for_read(reference), CommandContext(manifest.actor_id, "stitch-result:" + job_id, None))

    def fail_stitch_agent_job(self, job_id, error):
        manifest = self.agent_gateway.stitch_job_manifest(job_id)
        if manifest is None:
            raise ConflictError("Stitch job was not found")
        with self.uow_factory() as unit:
            service = MultilayerService(unit.event_store, unit.projections)
            project = service._project(manifest.project_id)
            rejected = reject_stitch_authorization(unit, project_id=manifest.project_id, job_id=job_id,
                identities=self.identities, storage=self.profiles.get(project.storage_profile))
            if rejected is not None:
                unit.commit()
                return service.describe(rejected)
            actor = self._actor(manifest.actor_id)
            model = fail_stitch_result(unit, project_id=manifest.project_id, job_id=job_id, principal=actor,
                storage=self.profiles.get(project.storage_profile), roles=unit.acl.roles_for(project.id, actor.id),
                error=error, verified=True)
            unit.commit()
            return service.describe(model)
