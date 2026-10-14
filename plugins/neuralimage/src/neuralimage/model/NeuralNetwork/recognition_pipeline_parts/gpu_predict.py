"""recognition_pipeline: gpu predict."""
from __future__ import annotations
import os
from typing import Any, cast
from contextlib import suppress
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from neuralimage.uncertainty.config import UncertaintyConfig
from neuralimage.model.NeuralNetwork.blocks import extract_confidence_output, extract_mask_outputs

def gpu_predict(
    img: dict[str, Any],
    model: nn.Module,
    device: torch.device,
    batch_size: int,
    *,
    recognition_tta_enabled: bool = False,
    confidence_tta_enabled: bool = False,
    uncertainty: UncertaintyConfig | None = None,
) -> dict[str, Any]:
    predicted_image = np.empty_like(img['cutted_image'])
    confidence_image = np.empty_like(img['cutted_image'])
    variance_image = np.zeros_like(img['cutted_image'])
    disagreement_image = np.zeros_like(img['cutted_image'])
    parts_in_image = len(img['cutted_image'])

    source_batches = img['cutted_image']
    if torch.is_tensor(source_batches):
        tensor_data = source_batches.float()
    else:
        tensor_data = torch.from_numpy(source_batches).float()
    context_batches = img.get('context_image')
    context_tensor_data: torch.Tensor | None = None
    if context_batches is not None:
        if torch.is_tensor(context_batches):
            context_tensor_data = context_batches.float()
        else:
            context_tensor_data = torch.from_numpy(context_batches).float()
    global_batches = img.get('global_image')
    global_tensor_data: torch.Tensor | None = None
    if global_batches is not None:
        if torch.is_tensor(global_batches):
            global_tensor_data = global_batches.float()
        else:
            global_tensor_data = torch.from_numpy(global_batches).float()
    coords_batches = img.get('patch_coords_norm')
    coords_tensor_data: torch.Tensor | None = None
    if coords_batches is not None:
        if torch.is_tensor(coords_batches):
            coords_tensor_data = coords_batches.float()
        else:
            coords_tensor_data = torch.from_numpy(coords_batches).float()
    use_amp = device.type == 'cuda'
    if use_amp and hasattr(tensor_data, 'pin_memory'):
        with suppress(Exception):
            tensor_data = tensor_data.pin_memory()
    if use_amp and context_tensor_data is not None and hasattr(context_tensor_data, 'pin_memory'):
        with suppress(Exception):
            context_tensor_data = context_tensor_data.pin_memory()
    if use_amp and global_tensor_data is not None and hasattr(global_tensor_data, 'pin_memory'):
        with suppress(Exception):
            global_tensor_data = global_tensor_data.pin_memory()
    if use_amp and coords_tensor_data is not None and hasattr(coords_tensor_data, 'pin_memory'):
        with suppress(Exception):
            coords_tensor_data = coords_tensor_data.pin_memory()
    min_prob = float('inf')
    max_prob = float('-inf')
    mean_acc = 0.0
    processed_batches = 0
    non_finite_values = 0
    fp32_retries = 0
    tta_uncertainty_enabled = bool(
        confidence_tta_enabled
        or (uncertainty is not None and uncertainty.uses_tta_variance())
    )
    tta_flip = bool(
        recognition_tta_enabled
        or tta_uncertainty_enabled
    )
    multiscale_values = str(os.getenv('NEURALIMAGE_MS_SCALES', '1.0')).strip()
    ms_scales: list[float] = []
    for token in multiscale_values.split(','):
        token = token.strip()
        if not token:
            continue
        try:
            scale = float(token)
        except ValueError:
            continue
        if scale <= 0.0:
            continue
        ms_scales.append(scale)
    if not ms_scales:
        ms_scales = [1.0]

    def _extract_prediction_tensors(outputs: Any) -> tuple[torch.Tensor, torch.Tensor | None]:
        mask_outputs = extract_mask_outputs(outputs)
        if torch.is_tensor(mask_outputs):
            primary = mask_outputs
        elif isinstance(mask_outputs, (list, tuple)) and mask_outputs and torch.is_tensor(mask_outputs[0]):
            primary = cast(torch.Tensor, mask_outputs[0])
        else:
            raise TypeError('Recognition model must return a tensor or a non-empty sequence of tensors.')
        confidence = extract_confidence_output(outputs)
        return primary, confidence

    def _select_global_batch(
        global_tensor: torch.Tensor | None,
        start_index: int,
        end_index: int,
    ) -> torch.Tensor | None:
        if global_tensor is None:
            return None
        if global_tensor.ndim == 3:
            batch_len = int(end_index - start_index)
            return global_tensor.unsqueeze(0).expand(batch_len, -1, -1, -1)
        if global_tensor.ndim == 4:
            if int(global_tensor.shape[0]) == 1:
                batch_len = int(end_index - start_index)
                return global_tensor.expand(batch_len, -1, -1, -1)
            return global_tensor[start_index:end_index]
        raise ValueError(f'global_image must be CHW or NCHW, got {tuple(global_tensor.shape)!r}.')

    def _flip_patch_coords_norm_x(coords: torch.Tensor | None) -> torch.Tensor | None:
        if coords is None:
            return None
        flipped = coords.clone()
        x0 = coords[:, 0].clone()
        x1 = coords[:, 2].clone()
        flipped[:, 0] = 1.0 - x1
        flipped[:, 2] = 1.0 - x0
        return torch.clamp(flipped, 0.0, 1.0)

    def _predict_once(
        batch_tensor: torch.Tensor,
        context_batch_tensor: torch.Tensor | None = None,
        global_batch_tensor: torch.Tensor | None = None,
        patch_coords_tensor: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            if context_batch_tensor is None and global_batch_tensor is None and patch_coords_tensor is None:
                return _extract_prediction_tensors(model(batch_tensor))
            model_input: dict[str, torch.Tensor] = {'local_image': batch_tensor}
            if context_batch_tensor is not None:
                model_input['context_image'] = context_batch_tensor
            if global_batch_tensor is not None:
                model_input['global_image'] = global_batch_tensor
                model_input.setdefault('context_image', global_batch_tensor)
            if patch_coords_tensor is not None:
                model_input['patch_coords_norm'] = patch_coords_tensor
            return _extract_prediction_tensors(model(model_input))

    def _predict_with_multi_scale_and_tta(
        batch_tensor: torch.Tensor,
        context_batch_tensor: torch.Tensor | None = None,
        global_batch_tensor: torch.Tensor | None = None,
        patch_coords_tensor: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        base_h, base_w = int(batch_tensor.shape[-2]), int(batch_tensor.shape[-1])
        mask_acc = None
        confidence_acc: torch.Tensor | None = None
        confidence_from_mask_acc: torch.Tensor | None = None
        uncertainty_probability_acc: torch.Tensor | None = None
        uncertainty_probability_sq_acc: torch.Tensor | None = None
        uncertainty_weight = 0.0
        mask_weight = 0.0
        confidence_weight = 0.0
        for scale in ms_scales:
            if abs(scale - 1.0) < 1e-8:
                scaled = batch_tensor
            else:
                scaled_h = max(8, int(round(base_h * scale)))
                scaled_w = max(8, int(round(base_w * scale)))
                scaled = F.interpolate(
                    batch_tensor,
                    size=(scaled_h, scaled_w),
                    mode='bilinear',
                    align_corners=False,
                )
            logits, confidence_logits = _predict_once(
                scaled,
                context_batch_tensor,
                global_batch_tensor,
                patch_coords_tensor,
            )
            if logits.shape[-2:] != (base_h, base_w):
                logits = F.interpolate(logits, size=(base_h, base_w), mode='bilinear', align_corners=False)
            if mask_acc is None:
                mask_acc = torch.zeros_like(logits)
            mask_acc += logits
            mask_weight += 1.0
            probability = torch.sigmoid(logits)
            if uncertainty_probability_acc is None:
                uncertainty_probability_acc = torch.zeros_like(probability)
                uncertainty_probability_sq_acc = torch.zeros_like(probability)
            uncertainty_probability_acc += probability
            uncertainty_probability_sq_acc += probability.square()
            uncertainty_weight += 1.0
            if confidence_logits is not None and confidence_logits.shape[-2:] != (base_h, base_w):
                confidence_logits = F.interpolate(
                    confidence_logits,
                    size=(base_h, base_w),
                    mode='bilinear',
                    align_corners=False,
                )
            if confidence_logits is not None:
                if confidence_acc is None:
                    confidence_acc = torch.zeros_like(logits)
                confidence_acc += confidence_logits
            else:
                if confidence_from_mask_acc is None:
                    confidence_from_mask_acc = torch.zeros_like(logits)
                confidence_from_mask_acc += logits
            confidence_weight += 1.0
            if tta_flip:
                flipped_context = None
                flipped_global = None
                flipped_coords = None
                if global_batch_tensor is not None:
                    flipped_global = torch.flip(global_batch_tensor, dims=[-1])
                    flipped_context = flipped_global
                    flipped_coords = _flip_patch_coords_norm_x(patch_coords_tensor)
                elif context_batch_tensor is not None:
                    flipped_context = torch.flip(context_batch_tensor, dims=[-1])
                logits_h, confidence_h = _predict_once(
                    torch.flip(scaled, dims=[-1]),
                    flipped_context,
                    flipped_global,
                    flipped_coords,
                )
                logits_h = torch.flip(logits_h, dims=[-1])
                if logits_h.shape[-2:] != (base_h, base_w):
                    logits_h = F.interpolate(logits_h, size=(base_h, base_w), mode='bilinear', align_corners=False)
                if bool(recognition_tta_enabled):
                    mask_acc += logits_h
                    mask_weight += 1.0
                if tta_uncertainty_enabled:
                    probability_h = torch.sigmoid(logits_h)
                    uncertainty_probability_acc += probability_h
                    uncertainty_probability_sq_acc += probability_h.square()
                    uncertainty_weight += 1.0
                if confidence_h is not None:
                    confidence_h = torch.flip(confidence_h, dims=[-1])
                    if confidence_h.shape[-2:] != (base_h, base_w):
                        confidence_h = F.interpolate(
                            confidence_h,
                            size=(base_h, base_w),
                            mode='bilinear',
                            align_corners=False,
                        )
                if tta_uncertainty_enabled:
                    if confidence_h is not None:
                        if confidence_acc is None:
                            confidence_acc = torch.zeros_like(logits_h)
                        confidence_acc += confidence_h
                    else:
                        if confidence_from_mask_acc is None:
                            confidence_from_mask_acc = torch.zeros_like(logits_h)
                        confidence_from_mask_acc += logits_h
                    confidence_weight += 1.0
        if mask_acc is None:
            raise RuntimeError('Recognition TTA accumulator is empty.')
        averaged_mask = mask_acc / max(mask_weight, 1.0)
        if (
            tta_uncertainty_enabled
            and uncertainty_probability_acc is not None
            and uncertainty_probability_sq_acc is not None
            and uncertainty_weight > 1.0
        ):
            mean_probability = uncertainty_probability_acc / uncertainty_weight
            variance = torch.clamp(
                uncertainty_probability_sq_acc / uncertainty_weight - mean_probability.square(),
                min=0.0,
            )
            confidence_probability = torch.clamp(1.0 - 4.0 * variance, min=1e-6, max=1.0 - 1e-6)
            averaged_confidence = torch.logit(confidence_probability)
            tta_variance = variance
        elif confidence_acc is not None:
            averaged_confidence = confidence_acc / max(confidence_weight, 1.0)
            tta_variance = None
        elif confidence_from_mask_acc is not None:
            fallback_probability = torch.sigmoid(confidence_from_mask_acc / max(confidence_weight, 1.0))
            fallback_confidence = torch.clamp(
                torch.abs(fallback_probability - 0.5) * 2.0,
                min=1e-6,
                max=1.0 - 1e-6,
            )
            averaged_confidence = torch.logit(fallback_confidence)
            tta_variance = None
        else:
            averaged_confidence = None
            tta_variance = None
        return averaged_mask, averaged_confidence, tta_variance

    with torch.inference_mode():
        for batch_index, start in enumerate(range(0, parts_in_image, batch_size)):
            end = min(start + batch_size, parts_in_image)
            batch = tensor_data[start:end].to(device, non_blocking=use_amp)
            context_batch = None
            if context_tensor_data is not None:
                context_batch = context_tensor_data[start:end].to(device, non_blocking=use_amp)
            global_batch = _select_global_batch(global_tensor_data, start, end)
            if global_batch is not None:
                global_batch = global_batch.to(device, non_blocking=use_amp)
            coords_batch = None
            if coords_tensor_data is not None:
                coords_batch = coords_tensor_data[start:end].to(device, non_blocking=use_amp)
            outputs, confidence_outputs, tta_variance = _predict_with_multi_scale_and_tta(
                batch,
                context_batch,
                global_batch,
                coords_batch,
            )

            uncertainty_method = (
                str(getattr(uncertainty, 'method', '')).strip().lower()
                if uncertainty is not None and uncertainty.enabled
                else ''
            )
            batch_variance = tta_variance
            batch_disagreement = None
            if uncertainty_method in {'mc_dropout', 'combined'}:
                module_states = {
                    module: (module.training, getattr(module, 'p', None))
                    for module in model.modules()
                }
                model.eval()
                for module in model.modules():
                    if isinstance(module, nn.modules.dropout._DropoutNd):
                        module.p = float(min(max(uncertainty.mc_dropout_rate, 0.0), 0.999))
                        module.train(True)
                mc_probabilities: list[torch.Tensor] = []
                try:
                    for _ in range(max(2, int(uncertainty.mc_dropout_samples))):
                        mc_logits, _mc_confidence, _mc_tta_variance = _predict_with_multi_scale_and_tta(
                            batch,
                            context_batch,
                            global_batch,
                            coords_batch,
                        )
                        mc_probabilities.append(torch.sigmoid(mc_logits))
                finally:
                    for module, (training, dropout_rate) in module_states.items():
                        if dropout_rate is not None and hasattr(module, 'p'):
                            module.p = dropout_rate
                        module.train(training)
                stacked_mc = torch.stack(mc_probabilities, dim=0)
                mc_mean = stacked_mc.mean(dim=0)
                mc_variance = stacked_mc.var(dim=0, unbiased=False)
                mc_confidence = torch.clamp(1.0 - 4.0 * mc_variance, min=1e-6, max=1.0 - 1e-6)
                regular_probability = torch.sigmoid(outputs)
                batch_disagreement = torch.abs(mc_mean - regular_probability)
                batch_variance = (
                    mc_variance
                    if batch_variance is None
                    else 0.5 * (batch_variance + mc_variance)
                )
                if uncertainty_method == 'mc_dropout':
                    outputs = torch.logit(torch.clamp(mc_mean, min=1e-6, max=1.0 - 1e-6))
                    confidence_outputs = torch.logit(mc_confidence)
                else:
                    existing_confidence = (
                        torch.sigmoid(confidence_outputs)
                        if confidence_outputs is not None
                        else torch.abs(regular_probability - 0.5) * 2.0
                    )
                    confidence_outputs = torch.logit(
                        torch.clamp(0.5 * (existing_confidence + mc_confidence), min=1e-6, max=1.0 - 1e-6)
                    )

            if not bool(torch.isfinite(outputs).all()):
                fp32_retries += 1
                outputs, confidence_outputs = _predict_once(
                    batch.float(),
                    context_batch.float() if context_batch is not None else None,
                    global_batch.float() if global_batch is not None else None,
                    coords_batch.float() if coords_batch is not None else None,
                )

            finite_mask = torch.isfinite(outputs)
            if not bool(finite_mask.all()):
                non_finite_values += int((~finite_mask).sum().item())
                outputs = torch.nan_to_num(outputs, nan=0.0, posinf=20.0, neginf=-20.0)

            outputs = torch.sigmoid(outputs)
            if not bool(torch.isfinite(outputs).all()):
                finite_after_sigmoid = torch.isfinite(outputs)
                non_finite_values += int((~finite_after_sigmoid).sum().item())
                outputs = torch.nan_to_num(outputs, nan=0.0, posinf=1.0, neginf=0.0)
            if confidence_outputs is None:
                confidence_outputs = torch.abs(outputs - 0.5) * 2.0
            else:
                confidence_outputs = torch.sigmoid(confidence_outputs)
            if not bool(torch.isfinite(confidence_outputs).all()):
                confidence_outputs = torch.nan_to_num(confidence_outputs, nan=0.0, posinf=1.0, neginf=0.0)
            confidence_outputs = torch.clamp(confidence_outputs, min=0.0, max=1.0)

            predictions = outputs.detach().cpu().numpy()
            confidence_predictions = confidence_outputs.detach().cpu().numpy()
            batch_min = float(np.min(predictions))
            batch_max = float(np.max(predictions))
            batch_mean = float(np.mean(predictions))
            min_prob = min(min_prob, batch_min)
            max_prob = max(max_prob, batch_max)
            mean_acc += batch_mean
            processed_batches += 1
            predicted_image[start:end] = predictions[: end - start]
            confidence_image[start:end] = confidence_predictions[: end - start]
            if batch_variance is not None:
                variance_image[start:end] = batch_variance.detach().cpu().numpy()[: end - start]
            if batch_disagreement is not None:
                disagreement_image[start:end] = batch_disagreement.detach().cpu().numpy()[: end - start]

    img['predicted_image'] = predicted_image
    img['confidence_image'] = confidence_image
    if (
        uncertainty is not None
        and uncertainty.enabled
        and str(uncertainty.method).strip().lower() in {'tta_variance', 'mc_dropout', 'combined'}
    ):
        img['variance_image'] = variance_image
        img['disagreement_image'] = disagreement_image
    if processed_batches > 0:
        img['_prediction_stats'] = {
            'min': float(min_prob),
            'max': float(max_prob),
            'mean': float(mean_acc / processed_batches),
            'non_finite': int(non_finite_values),
            'fp32_retries': int(fp32_retries),
        }
    else:
        img['_prediction_stats'] = {
            'min': 0.0,
            'max': 0.0,
            'mean': 0.0,
            'non_finite': int(non_finite_values),
            'fp32_retries': int(fp32_retries),
        }
    return img
