"""Actual Karakal PNG output and Agent checks on a controlled four-pixel fixture."""
from dataclasses import replace
import hashlib
from io import BytesIO

from PIL import Image
import pytest

from kraken_agent.jobs import StagingWorkspace
from kraken_agent.runner import SubprocessPluginRunner
from kraken_core.analysis_protocol import AnalysisParameter
from kraken_core.analysis_run_protocol import AnalysisExpression, AnalysisRenderedMap, AnalysisPartitionResultManifest
from karakal.worker import execute
from karakal.core.headless import render_analysis_map
from tests.test_electrical_graph import png


def test_native_map_worker_and_agent_validate_exact_pixels_and_hash(tmp_path):
    from kraken_core.analysis_protocol import AnalysisArtifactInput, AnalysisFrameInput, AnalysisSourceRole
    from kraken_core.analysis_run_protocol import AnalysisPartitionJobManifest, AnalysisRecipe

    workspace = StagingWorkspace(tmp_path, "map-job")
    workspace.create()
    artifacts = []
    for key, pixels in (("A", [[255, 255], [0, 0]]), ("B", [[255, 0], [255, 0]])):
        data = png(pixels)
        relative = f"inputs/{key}.png"
        destination = workspace.resolve(relative)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)
        artifacts.append(AnalysisArtifactInput(key, AnalysisSourceRole.MODEL_OUTPUT, key, key + "-version", relative,
                                               "image/png", hashlib.sha256(data).hexdigest()))
    job = AnalysisPartitionJobManifest(job_id="map-job", run_id="run", partition_id="partition", project_id="project",
        partition_index=0, partition_count=1, run_fingerprint="saved-inputs",
        recipe=AnalysisRecipe(AnalysisExpression.binary("compare", AnalysisExpression.source("A"), AnalysisExpression.source("B"))),
        frames=(AnalysisFrameInput("frame", 1, 1, tuple(artifacts)),), parameters=(AnalysisParameter("_render_frame_id", "frame"),))
    workspace.write_manifest(job)
    result = execute(workspace.path / "job.json", workspace.path / "result.json", workspace.path / "progress.json", workspace.path)
    assert result.rendered_map is not None
    assert AnalysisPartitionResultManifest.from_payload(result.to_payload()) == result
    SubprocessPluginRunner._validate_analysis_result_contract(job, result)
    workspace.verify_result(result, job)
    path = workspace.resolve(result.rendered_map.relative_path)
    with Image.open(BytesIO(path.read_bytes())) as image:
        assert list(image.convert("RGB").get_flattened_data()) == [(255, 255, 255), (255, 0, 0), (0, 0, 255), (0, 0, 0)]
    assert render_analysis_map(job, "frame", workspace.path, workspace.path / "outputs") == path
    with pytest.raises(ValueError, match="unrequested"):
        SubprocessPluginRunner._validate_analysis_result_contract(replace(job, parameters=()), result)
    path.write_bytes(b"corrupted map")
    with pytest.raises(ValueError, match="checksum"):
        workspace.verify_result(result, job)


@pytest.mark.parametrize("change", [{"relative_path": "../escape.png"}, {"relative_path": "inputs/forged.png"},
                                    {"sha256": "x" * 64}, {"size_bytes": True}, {"size_bytes": 0},
                                    {"size_bytes": 256 * 1024 * 1024 + 1}])
def test_map_contract_rejects_unsafe_metadata(change):
    with pytest.raises(ValueError):
        AnalysisRenderedMap(**(dict(frame_id="frame", relative_path="outputs/map.png", sha256="a" * 64, size_bytes=10) | change))


@pytest.mark.parametrize("kind", ["invalid-image", "jpeg", "excessive-pixels"])
def test_server_rejects_invalid_or_unbounded_png_even_with_matching_hash(kind):
    from kraken_manager.application.errors import ConflictError
    from kraken_manager.domain.artifacts import BlobRef
    from kraken_server.persistent_services_parts.analysis_maps import AnalysisMapsMixin

    data = b"not an image"
    if kind == "jpeg":
        output = BytesIO()
        Image.new("RGB", (2, 2)).save(output, format="JPEG")
        data = output.getvalue()
    elif kind == "excessive-pixels":
        import struct
        import zlib

        def chunk(name, payload):
            return struct.pack(">I", len(payload)) + name + payload + struct.pack(">I", zlib.crc32(name + payload))

        data = (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", 8193, 8193, 8, 2, 0, 0, 0))
                + chunk(b"IDAT", zlib.compress(b"")) + chunk(b"IEND", b""))

    class Blobs:
        def open(self, _reference):
            return BytesIO(data)

    reference = BlobRef(hashlib.sha256(data).hexdigest(), len(data))
    with pytest.raises(ConflictError):
        AnalysisMapsMixin._verified_map_bytes(Blobs(), reference)
