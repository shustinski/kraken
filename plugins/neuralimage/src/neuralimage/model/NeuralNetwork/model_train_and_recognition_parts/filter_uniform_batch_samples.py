"""model_train_and_recognition: filter uniform batch samples."""
from __future__ import annotations
from collections.abc import Mapping
from typing import Any
import torch
from neuralimage.targets.dataset_hooks import extract_primary_label
from neuralimage.training.batch_transfer import move_batch_to_device

from neuralimage.model.NeuralNetwork.model_train_and_recognition_parts.collect_memory_metrics import _collect_memory_metrics


class FilterUniformBatchSamplesMixin:
    def _publish_epoch_start(self, epoch: int, run_context: _RunContext, distributed: bool) -> None:
        self._model.train()
        if distributed:
            sampler = getattr(self._train_dataloader, 'sampler', None)
            set_epoch = getattr(sampler, 'set_epoch', None)
            if callable(set_epoch):
                set_epoch(epoch)
        self._bus.put(['logging', f'Начало эпохи [{epoch + 1}/{self._epochs}]'])
        current_lr = float(run_context.optimizer.param_groups[0]['lr'])
        self._bus.put(['logging', f'Текущий learning rate: {current_lr:.8f}'])
        self._bus.put([
            'metrics',
            {'type': 'train_epoch_progress', 'current': int(epoch), 'total': int(self._epochs)},
        ])
        self._bus.put([
            'metrics',
            {'type': 'train_batch_progress', 'current': 0, 'total': int(run_context.train_size)},
        ])
        self._bus.put([
            'metrics',
            {'type': 'validation_progress', 'current': 0, 'total': 0},
        ])
        memory_payload = _collect_memory_metrics()
        if memory_payload is not None:
            self._bus.put(['metrics', {'type': 'system_memory', **memory_payload, 'epoch': int(epoch + 1)}])
        self._bus.put(['logging', ' '])

    @staticmethod
    def _split_batch(batch: Any) -> tuple[Any, Any, Any]:
        if isinstance(batch, (list, tuple)) and len(batch) == 3:
            data, target, sample_indices = batch
            return data, target, sample_indices
        data, target = batch
        return data, target, None

    @staticmethod
    def _move_batch_input_to_device(data: Any, device: torch.device) -> Any:
        if not torch.is_tensor(data) and not isinstance(data, Mapping):
            raise TypeError(f'Unsupported batch input type: {type(data)!r}')
        return move_batch_to_device(data, device)

    @staticmethod
    def _move_batch_target_to_device(target: Any, device: torch.device) -> Any:
        if not torch.is_tensor(target) and not isinstance(target, Mapping):
            raise TypeError(f'Unsupported batch target type: {type(target)!r}')
        return move_batch_to_device(target, device)

    @staticmethod
    def _create_cuda_timing_events(device: torch.device, count: int) -> tuple[Any, ...] | None:
        if device.type != 'cuda':
            return None
        return tuple(torch.cuda.Event(enable_timing=True) for _ in range(max(0, int(count))))

    @staticmethod
    def _record_cuda_timing_event(event: Any, device: torch.device) -> None:
        if event is None or device.type != 'cuda':
            return
        event.record(torch.cuda.current_stream(device=device))

    @staticmethod
    def _elapsed_cuda_event_ms(start_event: Any, end_event: Any) -> float:
        return float(start_event.elapsed_time(end_event))

    @staticmethod
    def _loss_value_to_float(loss: torch.Tensor | float) -> float:
        if torch.is_tensor(loss):
            return float(loss.detach().to(device='cpu', dtype=torch.float32).item())
        return float(loss)

    @staticmethod
    def _extract_local_image(data: Any) -> torch.Tensor:
        if isinstance(data, Mapping):
            local_image = data.get('local_image')
            if not torch.is_tensor(local_image):
                raise TypeError('Expected "local_image" tensor in model input mapping.')
            return local_image
        if not torch.is_tensor(data):
            raise TypeError(f'Unsupported local batch input type: {type(data)!r}')
        return data

    @staticmethod
    def _extract_context_image(data: Any) -> torch.Tensor | None:
        if not isinstance(data, Mapping):
            return None
        context_image = data.get('context_image')
        if context_image is None:
            return None
        if not torch.is_tensor(context_image):
            raise TypeError('Expected "context_image" tensor in model input mapping.')
        return context_image

    @staticmethod
    def _filter_batch_input(data: Any, valid_mask: torch.Tensor) -> Any:
        if isinstance(data, Mapping):
            filtered: dict[str, Any] = {}
            for key, value in data.items():
                if torch.is_tensor(value) and value.ndim > 0 and int(value.shape[0]) == int(valid_mask.shape[0]):
                    filtered[str(key)] = value[valid_mask.detach().to(device=value.device)]
                else:
                    filtered[str(key)] = value
            return filtered
        if torch.is_tensor(data):
            return data[valid_mask]
        raise TypeError(f'Unsupported batch input type: {type(data)!r}')

    def _filter_uniform_batch_samples(
        self,
        image: Any,
        label: torch.Tensor | Mapping[str, torch.Tensor],
        sample_indices: Any,
    ) -> tuple[Any, torch.Tensor | Mapping[str, torch.Tensor], Any, int, bool]:
        if not self._skip_uniform_labels:
            return image, label, sample_indices, 0, True
        primary_label = extract_primary_label(label) if isinstance(label, Mapping) else label
        label_flat = self._sanitize_labels_for_loss(primary_label).reshape(primary_label.shape[0], -1)
        label_bin = label_flat >= 0.5
        is_all_zero = (~label_bin).all(dim=1)
        is_all_one = label_bin.all(dim=1)
        valid_mask = ~(is_all_zero | is_all_one)
        skipped_here = int((~valid_mask).sum().item())
        if not bool(valid_mask.any()):
            return image, label, sample_indices, skipped_here, False

        image = self._filter_batch_input(image, valid_mask)
        if isinstance(label, Mapping):
            label = {
                key: (
                    value[valid_mask.detach().to(device=value.device)]
                    if torch.is_tensor(value)
                    and value.ndim > 0
                    and int(value.shape[0]) == int(valid_mask.shape[0])
                    else value
                )
                for key, value in label.items()
            }
        else:
            label = label[valid_mask]
        if sample_indices is None:
            return image, label, None, skipped_here, True
        if torch.is_tensor(sample_indices):
            sample_indices = sample_indices[valid_mask.detach().to(device=sample_indices.device)]
        else:
            sample_indices_tensor = torch.as_tensor(sample_indices)
            sample_indices = sample_indices_tensor[valid_mask.detach().to(device='cpu')]
        return image, label, sample_indices, skipped_here, True

    @staticmethod
    def _permute_sample_indices(sample_indices: Any, permutation: torch.Tensor) -> Any:
        if sample_indices is None:
            return None
        if torch.is_tensor(sample_indices):
            return sample_indices[permutation.detach().to(device=sample_indices.device)]
        sample_indices_tensor = torch.as_tensor(sample_indices, dtype=torch.long)
        return sample_indices_tensor[permutation.detach().to(device='cpu')]
