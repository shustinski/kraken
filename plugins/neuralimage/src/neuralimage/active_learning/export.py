from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from PIL import Image

from neuralimage.active_learning.config import ActiveLearningConfig
from neuralimage.active_learning.scoring import score_prediction_uncertainty


@dataclass
class UncertainSampleRecord:
    sample_id: str
    score: float
    mean_confidence: float
    mean_entropy: float
    reasons: tuple[str, ...]


class ActiveLearningExporter:
    """Export uncertain samples into a NeedsAnnotation dataset folder."""

    def __init__(self, config: ActiveLearningConfig):
        self.config = config
        self._exported = 0

    def should_export(self, score_payload: dict[str, np.ndarray | float]) -> tuple[bool, tuple[str, ...]]:
        reasons: list[str] = []
        if bool(np.any(score_payload.get('low_confidence', False))):
            reasons.append('low_confidence')
        if bool(np.any(score_payload.get('high_entropy', False))):
            reasons.append('high_entropy')
        if bool(np.any(score_payload.get('unstable', False))):
            reasons.append('unstable_prediction')
        return bool(reasons), tuple(reasons)

    def export_sample(
        self,
        *,
        export_root: Path,
        sample_id: str,
        image: np.ndarray,
        probabilities: np.ndarray,
        confidence: np.ndarray | None = None,
        metadata: dict[str, object] | None = None,
    ) -> UncertainSampleRecord | None:
        if not self.config.enabled:
            return None
        if self._exported >= self.config.max_exports_per_run:
            return None

        score_payload = score_prediction_uncertainty(
            probabilities,
            confidence=confidence,
            low_confidence_threshold=self.config.low_confidence_threshold,
            high_entropy_threshold=self.config.high_entropy_threshold,
            instability_threshold=self.config.instability_threshold,
        )
        should_export, reasons = self.should_export(score_payload)
        if not should_export:
            return None

        export_root.mkdir(parents=True, exist_ok=True)
        image_dir = export_root / 'images'
        prob_dir = export_root / 'predictions'
        meta_dir = export_root / 'metadata'
        image_dir.mkdir(parents=True, exist_ok=True)
        prob_dir.mkdir(parents=True, exist_ok=True)
        meta_dir.mkdir(parents=True, exist_ok=True)

        image_path = image_dir / f'{sample_id}.png'
        prob_path = prob_dir / f'{sample_id}.png'
        meta_path = meta_dir / f'{sample_id}.json'

        image_array = np.asarray(image)
        if image_array.ndim == 3 and image_array.shape[0] in {1, 3}:
            if image_array.shape[0] == 1:
                image_array = image_array[0]
            else:
                image_array = np.transpose(image_array, (1, 2, 0))
        if image_array.max() <= 1.0:
            image_array = (image_array * 255.0).astype(np.uint8)
        Image.fromarray(image_array).save(image_path)

        prob_array = np.clip(np.asarray(probabilities, dtype=np.float32), 0.0, 1.0)
        if prob_array.ndim == 3:
            prob_array = prob_array[0]
        Image.fromarray((prob_array * 255.0).astype(np.uint8)).save(prob_path)

        record = UncertainSampleRecord(
            sample_id=sample_id,
            score=float(np.max(score_payload['score'])),
            mean_confidence=float(score_payload['mean_confidence']),
            mean_entropy=float(score_payload['mean_entropy']),
            reasons=reasons,
        )
        payload = {
            'sample_id': sample_id,
            'score': record.score,
            'mean_confidence': record.mean_confidence,
            'mean_entropy': record.mean_entropy,
            'reasons': list(record.reasons),
            'exported_at': datetime.now(UTC).isoformat(),
            'image_path': str(image_path),
            'prediction_path': str(prob_path),
        }
        if metadata:
            payload.update(metadata)
        meta_path.write_text(json.dumps(payload, indent=2), encoding='utf-8')
        self._exported += 1
        return record
