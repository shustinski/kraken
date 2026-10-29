"""Exercise the deployed private worker, including its native CIF dependencies."""
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from uuid import uuid4


def probe_stitch_worker(directory):
    from PIL import Image
    from kraken_core.plugin_protocol import PluginAsset, PluginJobManifestV2, PluginResultPublicationV2
    from kraken_manager.domain.multilayer import ArtifactAnchor, CoordinateCalibration
    from kraken_manager.domain.stitch_plan import LayerStitchRevision, TilePlacement
    from kraken_manager.infrastructure.mosaic import MosaicReader
    from shapely.geometry import shape

    root = Path(directory)
    root.mkdir(exist_ok=False)
    results = []
    for kind in ("image", "vector", "invalid-vector"):
        staging = root / kind
        inputs = staging / "inputs"
        inputs.mkdir(parents=True)
        image_path = inputs / "source.png"
        Image.new("RGB", (12, 12), (255, 0, 0)).save(image_path)
        image_sha = hashlib.sha256(image_path.read_bytes()).hexdigest()
        layer_id, frame_id, image_id, plan_id, job_id = (str(uuid4()) for _ in range(5))
        calibration = CoordinateCalibration(layer_id, 0, ArtifactAnchor(image_id, image_sha), (1e-6, 1e-6))
        tile = TilePlacement(frame_id, calibration, 12, 12, "reference", (1, 0, 0, 0, 1, 0), catalog_xy=(1, 1))
        plan_path = inputs / "declared-fixture.json"
        plan_path.write_text('{"scope":"deployment fixture only, not measured geometry"}', encoding="utf-8")
        plan_sha = hashlib.sha256(plan_path.read_bytes()).hexdigest()
        plan = LayerStitchRevision(str(uuid4()), 0, 0, layer_id, ArtifactAnchor(plan_id, plan_sha), frame_id, (tile,))
        assets = [PluginAsset(image_id, "geometry-source" if kind != "image" else "source", "frame", "inputs/source.png",
            image_sha, "image/png", frame_id=frame_id, x=1, y=1),
            PluginAsset(plan_id, "stitch-plan", "layer", "inputs/declared-fixture.json", plan_sha, "application/json")]
        parameters = {"stitch_plan": asdict(plan), "mode": "rgb", "block_size": 16}
        if kind != "image":
            cif = inputs / "source.cif"
            # Explicit doubled bridge describes a 4x4 hole in an 8x8 hull.
            cif.write_text("DS 1 1 1;\nL NM;\n( S 12 12 );\n"
                "P 1 11 9 11 7 9 3 9 3 5 7 5 7 9 9 11 9 3 1 3 1 11;\n"
                "B 2 2 10 2 1 1;\nDF;\nE\n", encoding="utf-8")
            if kind == "invalid-vector":
                # The interactive primitive adapter requires dimensions before
                # geometry; this also exercises deployed codec error resources.
                cif.write_text("DS 1 1 1;\nL NM;\nB 2 2 6 6;\n( S 12 12 );\nDF;\nE\n", encoding="utf-8")
            vector_id, vector_sha = str(uuid4()), hashlib.sha256(cif.read_bytes()).hexdigest()
            assets.append(PluginAsset(vector_id, "source", "frame", "inputs/source.cif", vector_sha,
                "application/x-cif", frame_id=frame_id, x=1, y=1))
            parameters.update(content_kind="vector", content_inputs=[{"frame_id": frame_id, "artifact_version_id": vector_id, "sha256": vector_sha}])
        manifest = PluginJobManifestV2(job_id, "layer.stitch.v1", str(uuid4()), layer_id, str(uuid4()), tuple(assets), parameters=parameters)
        job_path, result_path = staging / "job.json", staging / "result.json"
        job_path.write_text(manifest.to_json(), encoding="utf-8")
        environment = dict(os.environ, KRAKEN_STAGING_ROOT=str(staging), KRAKEN_JOB_MANIFEST=str(job_path),
            KRAKEN_RESULT_MANIFEST=str(result_path), KRAKEN_PROGRESS_PATH=str(staging / "progress.json"))
        environment.pop("PYTHONPATH", None)
        command = [sys.executable, "--kraken-stitch-worker"] if getattr(sys, "frozen", False) else [sys.executable, "-m", "kraken_manager.infrastructure.stitch_worker"]
        options = {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}
        with (staging / "worker.log").open("wb") as log:
            process = subprocess.run(command, env=environment, cwd=staging, stdout=log, stderr=subprocess.STDOUT, timeout=60, **options)
        if process.returncode or not result_path.is_file():
            raise RuntimeError(f"Deployed {kind} worker failed; see {staging / 'worker.log'}")
        publication = PluginResultPublicationV2.from_json(result_path.read_text(encoding="utf-8"))
        if kind == "invalid-vector":
            error = publication.report.get("error", "")
            if publication.outcome != "failed" or publication.outputs or not error.startswith("Vector frame "):
                raise RuntimeError(f"Malformed CIF did not produce its contextual refusal: {publication.report}")
            if list(staging.rglob("mosaic.sqlite3")):
                raise RuntimeError("Malformed CIF left a published mosaic")
            results.append({"kind": kind, "job_id": job_id, "refused_without_output": True, "error": error})
            continue
        if publication.outcome != "succeeded" or len(publication.outputs) != 1:
            raise RuntimeError(f"Deployed {kind} stitch publication failed: {publication.report}")
        asset, = publication.outputs
        path = staging / asset.relative_path
        if hashlib.sha256(path.read_bytes()).hexdigest() != asset.sha256:
            raise RuntimeError("Deployed mosaic SHA differs from its publication")
        with MosaicReader(path) as reader:
            reader.validate(expected_plan=plan, expected_job_id=job_id)
            block = reader.read_block(0, 0)
            if block.coverage[0, 0] != 1 or block.pixels[0, 0].tolist() != [255, 0, 0]:
                raise RuntimeError("Deployed raster pixels/coverage differ")
            if kind == "vector":
                features = reader.read_vectors(0, 0)["features"]
                conductor, = [shape(f["geometry"]) for f in features if f["category"] == "conductor"]
                if conductor.area != 48 or len(conductor.interiors) != 1 or not any(f["category"] == "via" for f in features):
                    raise RuntimeError("Deployed CIF bridge/hole/rotated-box codec differs")
        results.append({"kind": kind, "job_id": job_id, "mosaic_sha256": asset.sha256})
    (root / "receipt.json").write_text(json.dumps({"status": "passed", "frozen": bool(getattr(sys, "frozen", False)), "results": results}, indent=2), encoding="utf-8")
    return results
