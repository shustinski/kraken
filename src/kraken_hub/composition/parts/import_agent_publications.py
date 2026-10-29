"""composition: import agent publications."""
from __future__ import annotations
from collections.abc import Iterable, Mapping
from pathlib import Path
from kraken_core.plugin_protocol import (
    PluginResultPublicationV2,
)
from kraken_manager.domain.identity import (
    Principal,
)


class ImportAgentPublicationsMixin:
    def import_agent_publications(
        self,
        *,
        principal: Principal,
        publications: Iterable[Mapping[str, object]],
        staging_root: Path | str,
    ):
        parsed = tuple(
            PluginResultPublicationV2.from_dict(publication)
            for publication in publications
        )
        if not parsed:
            raise ValueError("Agent did not publish any V2 results")
        ordered = tuple(sorted(parsed, key=lambda item: item.sequence))
        first = ordered[0]
        if any(
            item.job_id != first.job_id
            or item.plugin_id != first.plugin_id
            or item.plugin_version != first.plugin_version
            for item in ordered
        ):
            raise ValueError("V2 publications do not belong to one plugin job")
        sequences = tuple(item.sequence for item in ordered)
        if len(sequences) != len(set(sequences)):
            raise ValueError("V2 publication sequence contains duplicates")
        if not ordered[-1].final:
            raise ValueError("The final V2 publication has not arrived yet")
        outputs = tuple(
            asset
            for publication in ordered
            for asset in publication.outputs
        )
        asset_ids = tuple(asset.asset_id for asset in outputs)
        if len(asset_ids) != len(set(asset_ids)):
            raise ValueError("V2 publications contain duplicate output asset IDs")
        frame_values: dict[str, dict[str, float]] = {}
        parameters: dict[str, object] = {}
        report: dict[str, object] = {}
        for publication in ordered:
            parameters.update(publication.applied_parameters)
            report.update(publication.report)
            for frame_id, measurements in publication.frame_values.items():
                frame_values.setdefault(frame_id, {}).update(measurements)
        parameters["v2_publication_ids"] = [
            publication.publication_id for publication in ordered
        ]
        final = ordered[-1]
        combined = PluginResultPublicationV2(
            job_id=final.job_id,
            publication_id=final.publication_id,
            sequence=final.sequence,
            plugin_id=final.plugin_id,
            plugin_version=final.plugin_version,
            outputs=outputs,
            outcome=final.outcome,
            applied_parameters=parameters,
            frame_values=frame_values,
            report=report,
            final=True,
        )
        return self.import_agent_result(
            principal=principal,
            result_payload=combined.to_dict(),
            staging_root=staging_root,
        )
