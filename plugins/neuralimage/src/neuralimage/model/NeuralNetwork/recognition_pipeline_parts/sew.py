"""recognition_pipeline: sew."""
from __future__ import annotations
from pathlib import Path
from typing import Any, cast
import numpy as np
from PIL import Image
from neuralimage.active_learning.config import ActiveLearningConfig
from neuralimage.active_learning.export import ActiveLearningExporter
from neuralimage.lib.image_processing import sew_image

from neuralimage.model.NeuralNetwork.recognition_pipeline_parts.load_preview_array import _load_preview_array

def sew(
    save_dir: Path | str,
    item: dict[str, Any],
    jpeg_quality: int = 95,
    *,
    threshold: float | None = None,
    postprocess_kernel_size: int = 0,
    confidence_save_mode: str = 'off',
    lossless_binary_png: bool = False,
    active_learning: ActiveLearningConfig | None = None,
    active_learning_metadata: dict[str, object] | None = None,
) -> Path:
    from neuralimage.model.NeuralNetwork.recognition_pipeline_parts.imgsew import _confidence_output_root, _save_jpeg_atomic, _save_png_atomic
    save_binary_png = bool(lossless_binary_png and threshold is not None)
    output_name = Path(str(item['name'])).with_suffix('.png' if save_binary_png else '.jpg')
    output_path = Path(save_dir) / output_name
    output_path.parent.mkdir(parents=True, exist_ok=True)
    probability_image = cast(
        Image.Image,
        sew_image(
            base_image=item['baseim_size'],
            predictions=item['predicted_image'],
            overlap=item['overlap'],
            threshold=None,
            postprocess_kernel_size=0,
        ),
    )
    image = probability_image
    if threshold is not None:
        image = cast(
            Image.Image,
            sew_image(
                base_image=item['baseim_size'],
                predictions=item['predicted_image'],
                overlap=item['overlap'],
                threshold=threshold,
                postprocess_kernel_size=postprocess_kernel_size,
            ),
        )
    quality = max(1, min(100, int(jpeg_quality)))
    if save_binary_png:
        if image.mode != 'L':
            image = image.convert('L')
        _save_png_atomic(image, output_path)
    else:
        _save_jpeg_atomic(image, output_path, quality=quality)
    confidence_predictions = item.get('confidence_image')
    confidence_image: Image.Image | None = None
    if confidence_predictions is not None:
        confidence_image = cast(
            Image.Image,
            sew_image(
                base_image=item['baseim_size'],
                predictions=confidence_predictions,
                overlap=item['overlap'],
                threshold=None,
                postprocess_kernel_size=0,
            ),
        )
    if str(confidence_save_mode).strip().lower() == 'separate_grayscale' and confidence_image is not None:
        confidence_name = Path(str(item['name'])).with_suffix('.jpg')
        confidence_path = _confidence_output_root(save_dir) / confidence_name
        confidence_path.parent.mkdir(parents=True, exist_ok=True)
        _save_jpeg_atomic(confidence_image, confidence_path, quality=quality)
    variance_predictions = item.get('variance_image')
    variance_image: Image.Image | None = None
    if variance_predictions is not None:
        variance_image = cast(
            Image.Image,
            sew_image(
                base_image=item['baseim_size'],
                predictions=variance_predictions,
                overlap=item['overlap'],
                threshold=None,
                postprocess_kernel_size=0,
            ),
        )
    disagreement_predictions = item.get('disagreement_image')
    disagreement_image: Image.Image | None = None
    if disagreement_predictions is not None:
        disagreement_image = cast(
            Image.Image,
            sew_image(
                base_image=item['baseim_size'],
                predictions=disagreement_predictions,
                overlap=item['overlap'],
                threshold=None,
                postprocess_kernel_size=0,
            ),
        )
    if active_learning is not None and active_learning.enabled:
        source_path = item.get('source_path')
        source_image = _load_preview_array(source_path)
        if source_image is not None:
            metadata = dict(active_learning_metadata or {})
            metadata['source_path'] = str(source_path)
            ActiveLearningExporter(active_learning).export_sample(
                export_root=active_learning.resolved_export_dir(Path(save_dir)),
                sample_id=str(item['name']),
                image=source_image,
                probabilities=np.asarray(probability_image, dtype=np.float32) / 255.0,
                confidence=(
                    np.asarray(confidence_image, dtype=np.float32) / 255.0
                    if confidence_image is not None
                    else None
                ),
                ensemble_variance=(
                    np.asarray(variance_image, dtype=np.float32) / 255.0
                    if variance_image is not None
                    else None
                ),
                disagreement=(
                    np.asarray(disagreement_image, dtype=np.float32) / 255.0
                    if disagreement_image is not None
                    else None
                ),
                metadata=metadata,
            )
    return output_path
