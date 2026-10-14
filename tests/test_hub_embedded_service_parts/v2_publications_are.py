"""test_hub_embedded_service: test v2 publications are."""
from __future__ import annotations
import hashlib
from pathlib import Path
from kraken_core.plugin_protocol import PluginAsset, PluginResultPublicationV2
from kraken_hub.composition import EmbeddedProjectService


class V2PublicationsAreMixin:
    def test_v2_publications_are_merged_before_domain_import(self) -> None:
        service = object.__new__(EmbeddedProjectService)
        captured = {}

        def import_result(**kwargs):
            captured.update(kwargs)
            return "imported"

        service.import_agent_result = import_result
        first = PluginResultPublicationV2(
            job_id="job-1",
            publication_id="publication-1",
            sequence=1,
            plugin_id="plugin",
            plugin_version="2",
            outputs=(
                PluginAsset(
                    asset_id="dataset-1",
                    role="dataset",
                    scope="layer",
                    relative_path="outputs/dataset.zip",
                    sha256=hashlib.sha256(b"dataset").hexdigest(),
                    media_type="application/zip",
                ),
            ),
            applied_parameters={"threshold": 0.5},
            frame_values={"frame-1": {"confidence": 0.7}},
        )
        final = PluginResultPublicationV2(
            job_id="job-1",
            publication_id="publication-2",
            sequence=2,
            plugin_id="plugin",
            plugin_version="2",
            outputs=(
                PluginAsset(
                    asset_id="model-1",
                    role="model",
                    scope="layer",
                    relative_path="outputs/model.bin",
                    sha256=hashlib.sha256(b"model").hexdigest(),
                    media_type="application/octet-stream",
                ),
            ),
            frame_values={"frame-1": {"quality": 0.9}},
            final=True,
        )

        result = service.import_agent_publications(
            principal=object(),
            publications=(final.to_dict(), first.to_dict()),
            staging_root=Path("."),
        )

        self.assertEqual("imported", result)
        merged = captured["result_payload"]
        self.assertEqual(
            ["dataset-1", "model-1"],
            [item["asset_id"] for item in merged["outputs"]],
        )
        self.assertEqual(
            {"confidence": 0.7, "quality": 0.9},
            merged["frame_values"]["frame-1"],
        )
        self.assertEqual(
            ["publication-1", "publication-2"],
            merged["applied_parameters"]["v2_publication_ids"],
        )
