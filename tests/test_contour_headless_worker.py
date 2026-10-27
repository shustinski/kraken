"""Real Contour processing in an Agent subprocess, without GUI automation."""

import hashlib
import io
import os
from pathlib import Path
import subprocess
import sys

from PIL import Image, ImageDraw

from kraken_core.plugin_protocol import PluginResultManifest
from tests.test_plugin_app_v1_bridges import _manifest, _workspace


def test_headless_contour_matches_the_existing_processing_algorithm(tmp_path):
    from contour.application.processing import ContourExtractionSettings, SaveOptions
    from contour.application.use_cases.processing import process_image_path

    image = Image.new("L", (64, 64), 0)
    ImageDraw.Draw(image).rectangle((10, 12, 45, 48), fill=255)
    encoded = io.BytesIO()
    image.save(encoded, format="PNG")
    payload = encoded.getvalue()
    manifest = _manifest(operation="frames.vectorize.v1", payload=payload, parameters={"execution_mode": "headless"})
    job, result = _workspace(tmp_path, manifest, payload)
    expected = tmp_path / "expected"
    expected.mkdir()
    processed = process_image_path(
        str(tmp_path / "inputs/staged-frame.png"),
        {},
        ContourExtractionSettings.from_dict({}),
        output_directory=str(expected),
        save_options=SaveOptions(save_cif=True),
    )
    assert processed.error is None
    environment = dict(
        os.environ, KRAKEN_JOB_MANIFEST=str(job), KRAKEN_RESULT_MANIFEST=str(result), KRAKEN_STAGING_ROOT=str(tmp_path)
    )
    root = Path(__file__).resolve().parents[1]
    process = subprocess.run(
        [sys.executable, str(root / "plugins/contour/__main__.py")],
        env=environment,
        cwd=root,
        capture_output=True,
        timeout=60,
    )
    assert process.returncode == 0, process.stderr.decode("utf-8", errors="replace")
    publication = PluginResultManifest.from_json(result.read_text(encoding="utf-8"))
    from kraken_core.plugins import load_plugin_metadata_from_directory

    assert publication.plugin_version == load_plugin_metadata_from_directory(root / "plugins/contour").version
    assert publication.outcome == "succeeded"
    assert len(publication.outputs) == 1
    output = publication.outputs[0]
    data = (tmp_path / output.relative_path).read_bytes()
    assert data == (expected / "staged-frame.cif").read_bytes()
    assert output.frame_id == "frame-1"
    assert output.sha256 == hashlib.sha256(data).hexdigest()
    assert publication.applied_parameters["execution_mode"] == "headless"
