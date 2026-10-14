"""composition: import agent result."""
from __future__ import annotations
import json
from collections.abc import Mapping
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5
from kraken_core.plugin_protocol import (
    PluginResultManifest,
    PluginResultPublicationV2,
    parse_plugin_result_json,
)
from kraken_manager.application.dto import (
    CommandContext,
    ImportPluginResultCommand,
)
from kraken_manager.application.plugin_jobs import (
    ImportPluginResultHandler,
)
from kraken_manager.domain.artifacts import (
    ArtifactScope,
)
from kraken_manager.domain.common import (
    FrameId,
    PluginJobId,
)
from kraken_manager.domain.identity import (
    Principal,
)
from kraken_manager.domain.workflows import (
    PluginFrameOutcome,
    PluginFrameResultV1,
    PluginResultManifestV1,
    PluginResultOutcome,
)
from kraken_manager.infrastructure.plugin import (
    AgentStagingResultContentReader,
    domain_result_from_transport,
)


class ImportAgentResultMixin:
    def import_agent_result(
        self,
        *,
        principal: Principal,
        result_payload: Mapping[str, object],
        staging_root: Path | str,
        confirm_partial: bool = False,
    ):
        parsed = parse_plugin_result_json(
            json.dumps(dict(result_payload), ensure_ascii=False)
        )
        publication: PluginResultPublicationV2 | None = None
        if isinstance(parsed, PluginResultManifest):
            manifest = domain_result_from_transport(parsed)
        elif isinstance(parsed, PluginResultPublicationV2):
            publication = parsed
            frame_results = tuple(
                PluginFrameResultV1(
                    output_id=str(
                        uuid5(
                            NAMESPACE_URL,
                            f"kraken:v2-output:{parsed.job_id}:{asset.asset_id}",
                        )
                    ),
                    frame_id=FrameId(str(asset.frame_id)),
                    outcome=PluginFrameOutcome.SUCCEEDED,
                    relative_path=asset.relative_path,
                    sha256=asset.sha256,
                    media_type=asset.media_type,
                    role=asset.role,
                )
                for asset in parsed.outputs
                if asset.scope.value == "frame" and asset.frame_id is not None
            )
            manifest = PluginResultManifestV1(
                job_id=PluginJobId(parsed.job_id),
                plugin_name=parsed.plugin_id,
                plugin_version=parsed.plugin_version,
                results=frame_results,
                parameters_applied={
                    **dict(parsed.applied_parameters),
                    "v2_publication_id": parsed.publication_id,
                    "v2_sequence": parsed.sequence,
                },
                outcome=PluginResultOutcome(parsed.outcome),
                # The authoritative domain job keeps the backwards-compatible
                # V1 command contract while the transport may be V2.
                protocol_version="1.0",
            )
        else:
            raise TypeError("Unsupported Agent result")
        source_job = next(
            (
                job
                for job in self.plugin_jobs()
                if str(job.id) == str(manifest.job_id)
            ),
            None,
        )
        if source_job is None:
            raise ValueError("Agent result belongs to an unknown plugin job")
        result = ImportPluginResultHandler(
            self._uow(str(source_job.project_id)),
            self.profiles,
            self.clock,
            AgentStagingResultContentReader(staging_root),
        )(
            ImportPluginResultCommand(
                context=CommandContext(
                    actor=principal,
                    idempotency_key=f"agent-import:{manifest.job_id}",
                ),
                manifest=manifest,
                confirm_partial=confirm_partial or publication is not None,
            )
        )
        if publication is None:
            return result
        job = result.job
        for asset in publication.outputs:
            if asset.scope.value != "layer":
                continue
            series = self.create_artifact_series(
                principal=principal,
                project_id=job.project_id,
                layer_id=job.layer_id,
                scope=ArtifactScope.LAYER_ATTACHMENT,
                name=asset.role,
                idempotency_key=(
                    f"agent-v2-series:{publication.publication_id}:{asset.asset_id}"
                ),
            )
            source = (
                Path(staging_root)
                / str(publication.job_id)
                / Path(asset.relative_path)
            )
            self.add_managed_artifact_version(
                principal=principal,
                project_id=job.project_id,
                series_id=series.id,
                source=source,
                idempotency_key=(
                    f"agent-v2-version:{publication.publication_id}:{asset.asset_id}"
                ),
            )
        if publication.frame_values:
            self._record_workspace_event(
                principal=principal,
                project_id=job.project_id,
                stream_id=f"plugin-job:{job.id}",
                event_type="PluginFrameValuesPublishedV2",
                payload={
                    "plugin_job_id": str(job.id),
                    "publication_id": publication.publication_id,
                    "frame_values": {
                        frame_id: dict(values)
                        for frame_id, values in publication.frame_values.items()
                    },
                },
                idempotency_key=f"agent-v2-values:{publication.publication_id}",
            )
        return result
