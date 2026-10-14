"""model_train_and_recognition: run validation epoch."""
from __future__ import annotations
import math
from collections.abc import Callable, Mapping, Sized
from typing import Any, ContextManager, cast
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from neuralimage.metrics.segmentation import compute_segmentation_metrics
from neuralimage.targets.dataset_hooks import extract_primary_label

from neuralimage.model.NeuralNetwork.model_train_and_recognition_parts.module_state import VALIDATION_THRESHOLD_CANDIDATES


class RunValidationEpochMixin:
    def _run_validation_epoch(
        self,
        epoch: int,
        device: torch.device,
        bce_criterion: nn.Module,
        autocast_ctx: Callable[[], ContextManager[Any]],
    ) -> dict[str, float] | None:
        if self._val_dataloader is None:
            return None
        val_dataset_len = len(cast(Sized, self._val_dataloader.dataset))
        if val_dataset_len == 0:
            return None

        was_training = bool(self._model.training)
        self._model.eval()
        val_loss = 0.0
        val_loss_samples = 0
        frame_loss_sums: dict[str, float] = {}
        frame_loss_counts: dict[str, int] = {}
        correct = 0
        total = 0
        true_positive = 0.0
        false_positive = 0.0
        false_negative = 0.0
        threshold_counts = {
            float(threshold): {
                'correct': 0,
                'total': 0,
                'tp': 0.0,
                'fp': 0.0,
                'fn': 0.0,
            }
            for threshold in VALIDATION_THRESHOLD_CANDIDATES
        }
        skipped_non_finite_batches = 0
        advanced_metrics_enabled = self._advanced_validation_enabled()
        advanced_metric_totals: dict[str, float] = {}
        advanced_metric_count = 0
        validation_export_cache = self._create_validation_export_cache()
        export_saved_images = 0
        evaluation_model = self._base_model if isinstance(self._model, DDP) else self._model
        try:
            validation_total = max(1, int(len(self._val_dataloader)))
        except TypeError:
            validation_batch_size = max(1, int(getattr(self._val_dataloader, 'batch_size', 1) or 1))
            validation_total = max(1, math.ceil(val_dataset_len / validation_batch_size))
        base_validation_dataset = self._unwrap_validation_dataset(self._val_dataloader.dataset)
        frame_lengths_fn = getattr(base_validation_dataset, 'validation_frame_lengths', None)
        if callable(frame_lengths_fn):
            validation_frame_total = len(frame_lengths_fn())
        else:
            validation_frame_total = len(getattr(base_validation_dataset, 'samples', ())) or val_dataset_len
        self._bus.put([
            'metrics',
            {'type': 'validation_progress', 'current': 0, 'total': validation_total},
        ])
        self._bus.put([
            'logging',
            f'Validation epoch {epoch + 1} started: frames={validation_frame_total}, batches={validation_total}.',
        ])
        with torch.no_grad():
            for validation_index, batch in enumerate(self._val_dataloader):
                data, target, _sample_indices = self._split_batch(batch)
                inputs = self._move_batch_input_to_device(data, device)
                image = self._extract_local_image(inputs)
                label = self._move_batch_target_to_device(target, device)
                with autocast_ctx():
                    outputs = evaluation_model(inputs)
                    per_sample_loss = self._compute_per_sample_loss(outputs, label, bce_criterion)
                    loss = per_sample_loss.mean()

                if not self._is_finite_tensor(loss):
                    skipped_non_finite_batches += 1
                    self._bus.put([
                        'metrics',
                        {
                            'type': 'validation_progress',
                            'current': int(validation_index + 1),
                            'total': validation_total,
                        },
                    ])
                    continue

                val_loss += loss.item() * image.size(0)
                val_loss_samples += int(image.size(0))
                sample_losses = per_sample_loss.detach().float().cpu().flatten()
                if _sample_indices is not None:
                    if isinstance(_sample_indices, torch.Tensor):
                        resolved_indices = [int(value) for value in _sample_indices.detach().cpu().flatten().tolist()]
                    else:
                        resolved_indices = [int(value) for value in _sample_indices]
                else:
                    resolved_indices = []
                validation_dataset = getattr(self._val_dataloader, 'dataset', None)
                frame_key_fn = getattr(validation_dataset, 'frame_key', None)
                for sample_offset, sample_loss in enumerate(sample_losses.tolist()):
                    if sample_offset < len(resolved_indices) and callable(frame_key_fn):
                        frame_key = str(frame_key_fn(resolved_indices[sample_offset]))
                    else:
                        frame_key = f'batch_{validation_index:08d}'
                    frame_loss_sums[frame_key] = frame_loss_sums.get(frame_key, 0.0) + float(sample_loss)
                    frame_loss_counts[frame_key] = frame_loss_counts.get(frame_key, 0) + 1
                probs = torch.sigmoid(self._sanitize_outputs_for_loss(outputs))
                confidence_output = self._extract_confidence_output_tensor(outputs)
                confidence_tensor = (
                    torch.sigmoid(confidence_output)
                    if confidence_output is not None
                    else None
                )
                self._collect_validation_export_batch(
                    validation_export_cache,
                    sample_indices=_sample_indices,
                    probs=probs,
                    confidence=confidence_tensor,
                    saved_images=export_saved_images,
                )
                export_saved_images += int(probs.shape[0])
                preds = probs >= 0.5
                primary_label = extract_primary_label(label) if isinstance(label, Mapping) else label
                label_bin = self._sanitize_labels_for_loss(primary_label) >= 0.5
                if advanced_metrics_enabled and (
                    not self._advanced_validation_full_frame()
                    or
                    validation_export_cache is None or validation_export_cache.mode != 'no_cut'
                ):
                    confidence_np = None
                    boundary_tolerance, include_hd95, confidence_bins = self._advanced_validation_options()
                    if confidence_tensor is not None:
                        confidence_np = confidence_tensor.detach().cpu().numpy()
                    batch_size = int(probs.shape[0])
                    for sample_index in range(batch_size):
                        sample_metrics = compute_segmentation_metrics(
                            probs[sample_index, 0].detach().cpu().numpy(),
                            primary_label[sample_index, 0].detach().cpu().numpy(),
                            threshold=float(self._recommended_inference_threshold),
                            confidence=(
                                confidence_np[sample_index, 0]
                                if confidence_np is not None
                                else None
                            ),
                            boundary_tolerance=boundary_tolerance,
                            include_hd95=include_hd95,
                            confidence_bins=confidence_bins,
                        )
                        for metric_name, metric_value in sample_metrics.as_dict().items():
                            if metric_name == 'confidence_histogram':
                                continue
                            if isinstance(metric_value, (int, float)):
                                advanced_metric_totals[metric_name] = (
                                    advanced_metric_totals.get(metric_name, 0.0) + float(metric_value)
                                )
                    advanced_metric_count += batch_size
                correct += (preds == label_bin).sum().item()
                total += label_bin.numel()
                preds_f = preds.float()
                label_f = label_bin.float()
                true_positive += float((preds_f * label_f).sum().item())
                false_positive += float((preds_f * (1.0 - label_f)).sum().item())
                false_negative += float(((1.0 - preds_f) * label_f).sum().item())
                for threshold, counts in threshold_counts.items():
                    threshold_preds = probs >= float(threshold)
                    counts['correct'] += int((threshold_preds == label_bin).sum().item())
                    counts['total'] += int(label_bin.numel())
                    threshold_preds_f = threshold_preds.float()
                    counts['tp'] += float((threshold_preds_f * label_f).sum().item())
                    counts['fp'] += float((threshold_preds_f * (1.0 - label_f)).sum().item())
                    counts['fn'] += float(((1.0 - threshold_preds_f) * label_f).sum().item())
                self._bus.put([
                    'metrics',
                    {
                        'type': 'validation_progress',
                        'current': int(validation_index + 1),
                        'total': validation_total,
                    },
                ])

        if skipped_non_finite_batches > 0:
            self._bus.put([
                'logging',
                (
                    'Validation warning: '
                    f'skipped {skipped_non_finite_batches} batch(es) with non-finite loss.'
                ),
            ])
            validation_export_cache = None
        if val_loss_samples <= 0:
            self._bus.put(['logging', 'Validation warning: all batches were skipped due to non-finite loss.'])
            self._model.train(was_training)
            return None

        frame_losses = [
            frame_loss_sums[frame_key] / frame_loss_counts[frame_key]
            for frame_key in frame_loss_sums
            if frame_loss_counts.get(frame_key, 0) > 0
        ]
        avg_val_loss = (
            sum(frame_losses) / len(frame_losses)
            if frame_losses
            else val_loss / val_loss_samples
        )
        base_metrics = self._compute_binary_metrics(
            true_positive=true_positive,
            false_positive=false_positive,
            false_negative=false_negative,
            correct=correct,
            total=total,
        )
        threshold_metrics = {
            float(threshold): self._compute_binary_metrics(
                true_positive=float(counts['tp']),
                false_positive=float(counts['fp']),
                false_negative=float(counts['fn']),
                correct=int(counts['correct']),
                total=int(counts['total']),
            )
            for threshold, counts in threshold_counts.items()
        }
        best_threshold, best_threshold_metrics = self._pick_best_validation_threshold(threshold_metrics)
        self._recommended_inference_threshold = float(best_threshold)
        stitched_metrics, stitched_frame_count = (
            self._compute_stitched_validation_metrics(
                validation_export_cache,
                threshold=float(best_threshold),
            )
            if self._advanced_validation_full_frame()
            else ({}, 0)
        )
        if stitched_frame_count > 0:
            advanced_metric_totals = stitched_metrics
            advanced_metric_count = 1
        self._save_validation_binary_predictions(
            epoch=epoch,
            device=device,
            autocast_ctx=autocast_ctx,
            threshold=float(best_threshold),
            export_cache=validation_export_cache,
        )
        self._model.train(was_training)

        return {
            'loss': float(avg_val_loss),
            'accuracy': float(base_metrics['accuracy']),
            'iou': float(base_metrics['iou']),
            'dice': float(base_metrics['dice']),
            'f1': float(base_metrics['f1']),
            'best_threshold': float(best_threshold),
            'best_threshold_accuracy': float(best_threshold_metrics['accuracy']),
            'best_threshold_iou': float(best_threshold_metrics['iou']),
            'best_threshold_dice': float(best_threshold_metrics['dice']),
            'best_threshold_f1': float(best_threshold_metrics['f1']),
            **(
                {
                    metric_name: float(
                        metric_total
                        if metric_name in {
                            'connected_component_difference',
                            'wire_break_count',
                            'false_bridge_count',
                            'missed_component_count',
                            'spurious_component_count',
                            'foreground_component_delta',
                            'background_component_delta',
                            'topology_violation_count',
                        }
                        else metric_total / max(1, advanced_metric_count)
                    )
                    for metric_name, metric_total in advanced_metric_totals.items()
                }
                if advanced_metrics_enabled and advanced_metric_count > 0
                else {}
            ),
        }
