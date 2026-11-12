"""Export a sample for labelling, with a manifest that keeps it checkable later.

Layout of one export::

    <out>/
      photos/<frame>.png          the chip photo (copied as is)
      masks/<model>/<frame>.png   every model's mask (copied as is)
      prelabel/<frame>.png        the mask the operator starts from and corrects (one model's)
      overlay/<frame>.png         the photo with every finding drawn, coloured by diagnosis
      manifest.json               why each frame is here, all its findings, the plan and versions
      manifest.csv                the same, one row per frame, for a spreadsheet

When the corrected masks come back (core/ground_truth.py), the manifest says which detector
flagged what, so each detector's precision can be measured, and the random part of the sample
says how often the model is really wrong.
"""

from __future__ import annotations

import csv
import json
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np

from .domain import BuildResult
from .findings import (
    DIAG_BAD_DATA,
    DIAG_CONFIDENT_ERROR,
    DIAG_MODEL_ERROR,
    DIAG_NOT_ALIGNED,
    DIAG_UNCERTAIN,
    findings_to_payload,
)
from .sampling import SamplePlan, SampledFrame

try:
    import cv2
except Exception:  # pragma: no cover - OpenCV is a hard dependency of the app
    cv2 = None

MANIFEST_SCHEMA_VERSION = 1
MANIFEST_FILE = "manifest.json"

# BGR colours of the overlay by diagnosis.
_COLOURS = {
    DIAG_MODEL_ERROR: (0, 0, 255),
    DIAG_CONFIDENT_ERROR: (255, 0, 255),
    DIAG_BAD_DATA: (0, 200, 255),
    DIAG_NOT_ALIGNED: (255, 128, 0),
    DIAG_UNCERTAIN: (255, 255, 0),
}
_DEFAULT_COLOUR = (0, 255, 0)


def _copy(source: str | None, target: Path) -> str | None:
    if not source or not Path(source).is_file():
        return None
    target.parent.mkdir(parents=True, exist_ok=True)
    final = target.with_suffix(Path(source).suffix.lower() or target.suffix)
    shutil.copy2(source, final)
    return str(final)


def _read_gray(path: str) -> np.ndarray | None:
    data = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_GRAYSCALE) if data.size else None


def _write_png(path: Path, image: np.ndarray) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise OSError(f"cannot encode {path}")
    encoded.tofile(str(path))
    return str(path)


def _overlay(photo_path: str | None, frame: SampledFrame, target: Path) -> str | None:
    if cv2 is None or not photo_path:
        return None
    gray = _read_gray(photo_path)
    if gray is None:
        return None
    image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    height, width = gray.shape
    thickness = max(1, int(round(min(height, width) / 512)))
    for finding in frame.findings:
        colour = _COLOURS.get(finding.diagnosis, _DEFAULT_COLOUR)
        box = finding.pixel_box((height, width))
        if box is None:
            cv2.rectangle(image, (0, 0), (width - 1, height - 1), colour, 2 * thickness)
            continue
        x, y, w, h = (int(round(value)) for value in box)
        cv2.rectangle(image, (x, y), (x + max(1, w), y + max(1, h)), colour, thickness)
        label = f"{finding.kind}{'@' + finding.model_id if finding.model_id else ''}"
        cv2.putText(image, label, (x, max(10, y - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.35 * thickness, colour, 1, cv2.LINE_AA)
    return _write_png(target, image)


def export_sample(
    build_result: BuildResult,
    sample: tuple[SampledFrame, ...] | list[SampledFrame],
    destination: Path | str,
    *,
    plan: SamplePlan | None = None,
    prelabel_model: str | None = None,
) -> Path:
    """Write the sample to ``destination`` and return the manifest's path."""

    from ..version import APP_VERSION

    out = Path(destination)
    out.mkdir(parents=True, exist_ok=True)
    records = {str(record.key): record for record in build_result.records}
    model_ids = [str(spec.model_id) for spec in build_result.model_specs]
    prelabel = prelabel_model if prelabel_model in model_ids else (model_ids[0] if model_ids else None)
    rows = []
    detectors: dict[str, str] = {}
    for frame in sample:
        record = records.get(frame.key)
        if record is None:
            continue
        # The key may carry the image's extension and folders: one flat name without the extension.
        stem = Path(frame.key).with_suffix("").as_posix() if Path(frame.key).suffix else frame.key
        safe = "".join(char if char.isalnum() or char in "-_." else "_" for char in stem)
        files: dict[str, object] = {}
        files["photo"] = _copy(record.original_path, out / "photos" / f"{safe}.png")
        files["masks"] = {
            model_id: _copy((record.model_mask_paths or {}).get(model_id), out / "masks" / model_id / f"{safe}.png")
            for model_id in model_ids
        }
        if prelabel is not None:
            files["prelabel"] = _copy((record.model_mask_paths or {}).get(prelabel), out / "prelabel" / f"{safe}.png")
        files["overlay"] = _overlay(record.original_path, frame, out / "overlay" / f"{safe}.png")
        for finding in frame.findings:
            if finding.detector_version:
                detectors[finding.detector] = finding.detector_version
        rows.append(
            {
                "frame": frame.key,
                "strategy": frame.strategy,
                "reason": frame.reason,
                "metric_key": frame.metric_key,
                "metric_value": frame.metric_value,
                "findings": findings_to_payload(frame.findings)["findings"],
                "files": {
                    key: (str(Path(value).relative_to(out)) if isinstance(value, str) else
                          {k: str(Path(v).relative_to(out)) if v else None for k, v in value.items()} if isinstance(value, dict) else None)
                    for key, value in files.items()
                },
            }
        )
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "created": datetime.now().astimezone().isoformat(timespec="seconds"),
        "karakal_version": APP_VERSION,
        "models": model_ids,
        "prelabel_model": prelabel,
        "plan": None if plan is None else plan.to_dict(),
        "detectors": dict(sorted(detectors.items())),
        "frames": rows,
    }
    path = out / MANIFEST_FILE
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    with (out / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter=";")
        writer.writerow(["frame", "strategy", "reason", "metric", "value", "findings", "kinds"])
        for row in rows:
            kinds = sorted({item["kind"] for item in row["findings"]})
            writer.writerow(
                [row["frame"], row["strategy"], row["reason"], row["metric_key"] or "", "" if row["metric_value"] is None else f"{row['metric_value']:.3f}",
                 len(row["findings"]), " ".join(kinds)]
            )
    return path


def read_manifest(path: Path | str) -> dict[str, object]:
    file = Path(path)
    if file.is_dir():
        file = file / MANIFEST_FILE
    return json.loads(file.read_text(encoding="utf-8"))


__all__ = ["MANIFEST_FILE", "MANIFEST_SCHEMA_VERSION", "export_sample", "read_manifest"]
