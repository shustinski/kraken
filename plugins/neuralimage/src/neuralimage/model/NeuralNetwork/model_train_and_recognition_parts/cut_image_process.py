"""model_train_and_recognition: cut image process."""
from __future__ import annotations
from pathlib import Path
from neuralimage.model.NeuralNetwork.recognition_pipeline import (
    cut_image_prepare as _cut_image_prepare,
    create_batches as _create_batches,
    cut_image_process as _cut_image_process,
    get_array_from_image as _get_array_from_image,
    gpu_predict as _gpu_predict,
    imgpredict as _imgpredict,
    imgsew as _imgsew,
    sew as _sew,
    sew_from_queue as _sew_from_queue,
)

from neuralimage.model.NeuralNetwork.model_train_and_recognition_parts.module_state import STOP_TOKEN
from neuralimage.model.NeuralNetwork.model_train_and_recognition import NeuralRecognizer

class NeuralRecognitioner(NeuralRecognizer):
    """Backward-compatible alias for the legacy misspelled class name."""

def cut_image_process(
    cut_queue,
    cutted_queue,
    size,
    overlap,
    stop_event,
    use_context_branch: bool = False,
    use_cross_attention: bool = True,
    context_crop_size: tuple[int, int] | None = None,
    context_input_size: tuple[int, int] | None = None,
    compression_factor: int = 1,
    source_root: Path | None = None,
):
    from neuralimage.model.NeuralNetwork.model_train_and_recognition import _cut_image_process

    return _cut_image_process(
        cut_queue,
        cutted_queue,
        size,
        overlap,
        stop_event,
        use_context_branch=use_context_branch,
        use_cross_attention=use_cross_attention,
        context_crop_size=context_crop_size,
        context_input_size=context_input_size,
        compression_factor=compression_factor,
        source_root=source_root,
        stop_token=STOP_TOKEN,
    )

def cut_image_prepare(
    img_path: Path,
    segment_size: tuple[int, int, int],
    overlap: int,
    *,
    use_context_branch: bool = False,
    use_cross_attention: bool = True,
    context_crop_size: tuple[int, int] | None = None,
    context_input_size: tuple[int, int] | None = None,
    compression_factor: int = 1,
    source_root: Path | None = None,
):
    from neuralimage.model.NeuralNetwork.model_train_and_recognition import _cut_image_prepare

    return _cut_image_prepare(
        img_path,
        segment_size,
        overlap,
        use_context_branch=use_context_branch,
        use_cross_attention=use_cross_attention,
        context_crop_size=context_crop_size,
        context_input_size=context_input_size,
        compression_factor=compression_factor,
        source_root=source_root,
    )

def get_array_from_image(path, channels):
    return _get_array_from_image(path, channels)

def imgpredict(
    prediction_queue,
    predicted_queue,
    model_path,
    gpu,
    batch_size,
    stop_event,
    recognition_tta_enabled=False,
    confidence_tta_enabled=False,
):
    return _imgpredict(
        prediction_queue,
        predicted_queue,
        model_path,
        gpu,
        batch_size,
        stop_event,
        recognition_tta_enabled=recognition_tta_enabled,
        confidence_tta_enabled=confidence_tta_enabled,
        stop_token=STOP_TOKEN,
    )

def gpu_predict(
    img,
    model,
    device,
    batch_size,
    *,
    recognition_tta_enabled=False,
    confidence_tta_enabled=False,
):
    from neuralimage.model.NeuralNetwork.model_train_and_recognition import _gpu_predict

    return _gpu_predict(
        img,
        model,
        device,
        batch_size,
        recognition_tta_enabled=recognition_tta_enabled,
        confidence_tta_enabled=confidence_tta_enabled,
    )

def create_batches(tensor_data, batch_size):
    return _create_batches(tensor_data, batch_size)

def imgsew(
    outputDir,
    sew_queue,
    sewed_queue,
    stop_event,
    jpeg_quality=95,
    threshold=None,
    postprocess_kernel_size=0,
    confidence_save_mode='off',
):
    return _imgsew(
        outputDir,
        sew_queue,
        sewed_queue,
        jpeg_quality,
        stop_event,
        threshold=threshold,
        postprocess_kernel_size=postprocess_kernel_size,
        confidence_save_mode=confidence_save_mode,
        stop_token=STOP_TOKEN,
    )

def sew_from_queue(
    output_dir,
    sew_queue,
    sewed_queue,
    jpeg_quality=95,
    threshold=None,
    postprocess_kernel_size=0,
    confidence_save_mode='off',
):
    return _sew_from_queue(
        output_dir,
        sew_queue,
        sewed_queue,
        jpeg_quality=jpeg_quality,
        threshold=threshold,
        postprocess_kernel_size=postprocess_kernel_size,
        confidence_save_mode=confidence_save_mode,
    )

def sew(save_dir, item, jpeg_quality=95, *, threshold=None, postprocess_kernel_size=0, confidence_save_mode='off'):
    from neuralimage.model.NeuralNetwork.model_train_and_recognition import _sew

    return _sew(
        save_dir,
        item,
        jpeg_quality=jpeg_quality,
        threshold=threshold,
        postprocess_kernel_size=postprocess_kernel_size,
        confidence_save_mode=confidence_save_mode,
    )
