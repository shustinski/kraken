from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import tempfile
import threading
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from neuralimage.active_learning.config import ActiveLearningConfig
from neuralimage.active_learning.scoring import score_prediction_uncertainty


@dataclass(frozen=True)
class UncertainRegion:
    x0: int
    y0: int
    x1: int
    y1: int
    score: float
    area: int
    reasons: tuple[str, ...]


@dataclass
class UncertainSampleRecord:
    sample_id: str
    score: float
    mean_confidence: float
    mean_entropy: float
    reasons: tuple[str, ...]
    rois: tuple[UncertainRegion, ...] = ()


def _atomic_save_image(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(dir=path.parent, suffix=path.suffix, delete=False)
    temporary = Path(handle.name)
    handle.close()
    try:
        Image.fromarray(array).save(temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(dir=path.parent, suffix='.tmp', mode='w', encoding='utf-8', delete=False)
    temporary = Path(handle.name)
    try:
        with handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _display_image(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image)
    if array.ndim == 3 and array.shape[0] in {1, 3}:
        array = array[0] if array.shape[0] == 1 else np.transpose(array, (1, 2, 0))
    if array.dtype == np.uint8:
        return array
    values = array.astype(np.float32)
    if values.size and float(np.nanmax(values)) <= 1.0:
        values *= 255.0
    return np.clip(np.nan_to_num(values), 0.0, 255.0).astype(np.uint8)


class ActiveLearningExporter:
    """Export full frames once with ranked uncertain ROIs and resumable manifests."""

    def __init__(self, config: ActiveLearningConfig):
        self.config = config
        self._exported = 0
        self._run_limit_lock = threading.Lock()

    @staticmethod
    def should_export(score_payload: dict[str, np.ndarray | float]) -> tuple[bool, tuple[str, ...]]:
        checks = (
            ('low_confidence', 'low_confidence'),
            ('high_entropy', 'high_entropy'),
            ('unstable', 'unstable_prediction'),
            ('disagreement', 'source_disagreement'),
        )
        reasons = tuple(reason for key, reason in checks if bool(np.any(score_payload.get(key, False))))
        return bool(reasons), reasons

    def _rank_rois(self, score_payload: dict[str, np.ndarray | float]) -> tuple[UncertainRegion, ...]:
        score = np.squeeze(np.asarray(score_payload['score'], dtype=np.float32))
        uncertain = score > 0.0
        if self.config.merge_distance:
            size = self.config.merge_distance * 2 + 1
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
            uncertain = cv2.morphologyEx(uncertain.astype(np.uint8), cv2.MORPH_CLOSE, kernel) > 0
        count, labels, stats, _centroids = cv2.connectedComponentsWithStats(uncertain.astype(np.uint8), connectivity=8)
        height, width = score.shape
        rois: list[UncertainRegion] = []
        reason_keys = {
            'low_confidence': 'low_confidence',
            'high_entropy': 'high_entropy',
            'unstable': 'unstable_prediction',
            'disagreement': 'source_disagreement',
        }
        for component_id in range(1, count):
            component = labels == component_id
            raw_area = int(component.sum())
            if raw_area < self.config.min_roi_area:
                continue
            x, y, w, h, _area = (int(value) for value in stats[component_id])
            padding = self.config.roi_padding
            x0, y0 = max(0, x - padding), max(0, y - padding)
            x1, y1 = min(width, x + w + padding), min(height, y + h + padding)
            reasons = tuple(
                reason
                for key, reason in reason_keys.items()
                if bool(np.any(np.asarray(score_payload[key])[component]))
            )
            rois.append(
                UncertainRegion(x0, y0, x1, y1, float(score[component].mean()), raw_area, reasons)
            )
        rois.sort(key=lambda region: (region.score, region.area), reverse=True)
        return tuple(rois[: self.config.max_rois_per_frame])

    @staticmethod
    def _existing_ids(manifest_path: Path) -> set[str]:
        if not manifest_path.exists():
            return set()
        ids: set[str] = set()
        for line in manifest_path.read_text(encoding='utf-8').splitlines():
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if payload.get('sample_id'):
                ids.add(str(payload['sample_id']))
        return ids

    @staticmethod
    def _safe_sample_id(sample_id: str, metadata: dict[str, object] | None) -> str:
        safe = re.sub(r'[^A-Za-z0-9_.-]+', '_', str(sample_id)).strip('._') or 'sample'
        if not metadata or not metadata.get('source_path'):
            return safe[:120]
        identity = json.dumps({'sample_id': sample_id, 'source': (metadata or {}).get('source_path')}, sort_keys=True)
        return f'{safe[:96]}_{hashlib.sha256(identity.encode("utf-8")).hexdigest()[:12]}'

    @staticmethod
    def _acquire_manifest_lock(export_root: Path, *, timeout_seconds: float = 5.0) -> tuple[int, Path] | None:
        export_root.mkdir(parents=True, exist_ok=True)
        lock_path = export_root / '.manifest.lock'
        deadline = time.monotonic() + max(0.1, float(timeout_seconds))
        while time.monotonic() < deadline:
            try:
                descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.write(descriptor, f'{os.getpid()}\n'.encode('ascii'))
                return descriptor, lock_path
            except FileExistsError:
                try:
                    if time.time() - lock_path.stat().st_mtime > 120.0:
                        lock_path.unlink(missing_ok=True)
                        continue
                except OSError:
                    pass
                time.sleep(0.01)
        return None

    @staticmethod
    def _release_manifest_lock(lock: tuple[int, Path]) -> None:
        descriptor, lock_path = lock
        try:
            os.close(descriptor)
        finally:
            lock_path.unlink(missing_ok=True)

    def _update_manifests(self, export_root: Path, payload: dict[str, object]) -> bool:
        lock = self._acquire_manifest_lock(export_root)
        if lock is None:
            return False
        try:
            manifest_path = export_root / 'manifest.jsonl'
            existing_lines = manifest_path.read_text(encoding='utf-8').splitlines() if manifest_path.exists() else []
            existing_rows: list[dict[str, object]] = []
            for line in existing_lines:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(row, dict):
                    existing_rows.append(row)
            sample_id = str(payload['sample_id'])
            if any(str(row.get('sample_id')) == sample_id for row in existing_rows):
                return False

            rows = [*existing_rows, payload]
            json_lines = [json.dumps(row, ensure_ascii=False, separators=(',', ':')) for row in rows]
            _atomic_write_text(manifest_path, '\n'.join(json_lines) + '\n')
            csv_path = export_root / 'manifest.csv'
            header = ('sample_id', 'source_sample_id', 'score', 'mean_confidence', 'mean_entropy', 'reasons', 'roi_count')
            handle = tempfile.NamedTemporaryFile(
                dir=export_root,
                suffix='.tmp',
                mode='w',
                encoding='utf-8',
                newline='',
                delete=False,
            )
            temporary = Path(handle.name)
            try:
                with handle:
                    writer = csv.DictWriter(handle, fieldnames=header)
                    writer.writeheader()
                    for row in rows:
                        writer.writerow({
                            'sample_id': row['sample_id'],
                            'source_sample_id': row.get('source_sample_id', ''),
                            'score': row['score'],
                            'mean_confidence': row['mean_confidence'],
                            'mean_entropy': row['mean_entropy'],
                            'reasons': ';'.join(row.get('reasons', [])),
                            'roi_count': len(row.get('rois', [])),
                        })
                os.replace(temporary, csv_path)
            finally:
                temporary.unlink(missing_ok=True)
            return True
        finally:
            self._release_manifest_lock(lock)

    def export_sample(
        self,
        *,
        export_root: Path,
        sample_id: str,
        image: np.ndarray,
        probabilities: np.ndarray,
        confidence: np.ndarray | None = None,
        ensemble_variance: np.ndarray | None = None,
        disagreement: np.ndarray | None = None,
        metadata: dict[str, object] | None = None,
    ) -> UncertainSampleRecord | None:
        if not self.config.enabled:
            return None
        probability_map = np.squeeze(np.asarray(probabilities, dtype=np.float32))
        score_payload = score_prediction_uncertainty(
            probability_map,
            confidence=np.squeeze(confidence) if confidence is not None else None,
            ensemble_variance=np.squeeze(ensemble_variance) if ensemble_variance is not None else None,
            disagreement=np.squeeze(disagreement) if disagreement is not None else None,
            low_confidence_threshold=self.config.low_confidence_threshold,
            high_entropy_threshold=self.config.high_entropy_threshold,
            instability_threshold=self.config.instability_threshold,
            disagreement_threshold=self.config.disagreement_threshold,
        )
        should_export, reasons = self.should_export(score_payload)
        if not should_export:
            return None
        rois = self._rank_rois(score_payload)
        if not rois:
            return None

        export_root = Path(export_root)
        manifest_path = export_root / 'manifest.jsonl'
        resolved_id = self._safe_sample_id(sample_id, metadata)
        existing_ids = self._existing_ids(manifest_path)
        if resolved_id in existing_ids:
            return None
        with self._run_limit_lock:
            if self._exported >= self.config.max_exports_per_run:
                return None
            self._exported += 1

        paths = {
            'image_path': export_root / 'images' / f'{resolved_id}.png',
            'prediction_path': export_root / 'probabilities' / f'{resolved_id}.png',
            'confidence_path': export_root / 'confidence' / f'{resolved_id}.png',
            'uncertainty_path': export_root / 'uncertainty' / f'{resolved_id}.png',
            'metadata_path': export_root / 'metadata' / f'{resolved_id}.json',
        }
        _atomic_save_image(paths['image_path'], _display_image(image))
        _atomic_save_image(paths['prediction_path'], np.round(np.clip(probability_map, 0.0, 1.0) * 255).astype(np.uint8))
        confidence_map = np.asarray(score_payload['confidence'], dtype=np.float32)
        _atomic_save_image(paths['confidence_path'], np.round(np.clip(confidence_map, 0.0, 1.0) * 255).astype(np.uint8))
        uncertainty_map = np.clip(1.0 - confidence_map, 0.0, 1.0)
        _atomic_save_image(paths['uncertainty_path'], np.round(uncertainty_map * 255).astype(np.uint8))

        record = UncertainSampleRecord(
            sample_id=resolved_id,
            score=float(max(region.score for region in rois)),
            mean_confidence=float(score_payload['mean_confidence']),
            mean_entropy=float(score_payload['mean_entropy']),
            reasons=reasons,
            rois=rois,
        )
        payload: dict[str, object] = {
            'sample_id': resolved_id,
            'source_sample_id': sample_id,
            'score': record.score,
            'mean_confidence': record.mean_confidence,
            'mean_entropy': record.mean_entropy,
            'reasons': list(record.reasons),
            'rois': [asdict(roi) for roi in rois],
            'exported_at': datetime.now(UTC).isoformat(),
            **{name: str(path.relative_to(export_root)) for name, path in paths.items()},
        }
        if metadata:
            payload.update(metadata)
        _atomic_write_text(paths['metadata_path'], json.dumps(payload, indent=2, ensure_ascii=False))

        if not self._update_manifests(export_root, payload):
            with self._run_limit_lock:
                self._exported = max(0, self._exported - 1)
            return None
        return record
