"""model_train_and_recognition: update loss aware sampling."""
from __future__ import annotations
from collections.abc import Mapping
from typing import Any, Iterator, cast
import torch

from neuralimage.model.NeuralNetwork.model_train_and_recognition_parts.stop_process_descendants import _SupportsLossAwareSampling


class UpdateLossAwareSamplingMixin:
    def _update_loss_aware_sampling(
        self,
        *,
        run_context: _RunContext,
        sample_indices: Any,
        per_sample_loss: torch.Tensor,
        mixup_pair_indices: Any = None,
        mixup_lambdas: torch.Tensor | None = None,
    ) -> None:
        if not run_context.supports_loss_aware_sampling or sample_indices is None:
            return
        if torch.is_tensor(sample_indices):
            sample_idx_tensor = sample_indices.detach().to(device='cpu', dtype=torch.long).flatten()
        else:
            sample_idx_tensor = torch.as_tensor(sample_indices, dtype=torch.long).flatten()
        sample_loss_tensor = per_sample_loss.detach().to(device='cpu', dtype=torch.float32).flatten()
        sample_loss_tensor = torch.nan_to_num(sample_loss_tensor, nan=1.0, posinf=50.0, neginf=0.0)
        if sample_idx_tensor.numel() != sample_loss_tensor.numel():
            return

        sampler = cast(_SupportsLossAwareSampling, run_context.train_sampler)
        if mixup_pair_indices is None or mixup_lambdas is None:
            sampler.update_batch_losses(sample_idx_tensor, sample_loss_tensor)
            return

        lambda_tensor = mixup_lambdas.detach().to(device='cpu', dtype=torch.float32).flatten()
        if lambda_tensor.numel() == 1 and sample_loss_tensor.numel() > 1:
            lambda_tensor = lambda_tensor.expand_as(sample_loss_tensor)
        if lambda_tensor.numel() != sample_loss_tensor.numel():
            sampler.update_batch_losses(sample_idx_tensor, sample_loss_tensor)
            return

        if torch.is_tensor(mixup_pair_indices):
            pair_idx_tensor = mixup_pair_indices.detach().to(device='cpu', dtype=torch.long).flatten()
        else:
            pair_idx_tensor = torch.as_tensor(mixup_pair_indices, dtype=torch.long).flatten()
        if pair_idx_tensor.numel() != sample_loss_tensor.numel():
            sampler.update_batch_losses(sample_idx_tensor, sample_loss_tensor)
            return

        sampler.update_batch_losses(sample_idx_tensor, sample_loss_tensor * lambda_tensor)
        sampler.update_batch_losses(pair_idx_tensor, sample_loss_tensor * (1.0 - lambda_tensor))

    @staticmethod
    def _format_tensor_debug_stats(tensor: torch.Tensor) -> str:
        detached = tensor.detach()
        shape = tuple(int(dim) for dim in detached.shape)
        dtype = str(detached.dtype)
        device = str(detached.device)
        finite_mask = torch.isfinite(detached)
        finite_count = int(finite_mask.sum().item())
        total_count = int(detached.numel())
        nan_count = int(torch.isnan(detached).sum().item())
        posinf_count = int(torch.isposinf(detached).sum().item())
        neginf_count = int(torch.isneginf(detached).sum().item())
        stats = (
            f'shape={shape}, dtype={dtype}, device={device}, '
            f'finite={finite_count}/{total_count}, nan={nan_count}, +inf={posinf_count}, -inf={neginf_count}'
        )
        if finite_count <= 0:
            return stats
        finite_values = detached[finite_mask].to(dtype=torch.float32)
        min_value = float(finite_values.min().item())
        max_value = float(finite_values.max().item())
        mean_value = float(finite_values.mean().item())
        return f'{stats}, min={min_value:.6g}, max={max_value:.6g}, mean={mean_value:.6g}'

    def _iter_named_input_tensors(self, data: Any, *, prefix: str = 'inputs') -> Iterator[tuple[str, torch.Tensor]]:
        if torch.is_tensor(data):
            yield prefix, data
            return
        if isinstance(data, Mapping):
            for key, value in data.items():
                child_prefix = f'{prefix}.{key}'
                if torch.is_tensor(value):
                    yield child_prefix, value
                elif isinstance(value, Mapping):
                    yield from self._iter_named_input_tensors(value, prefix=child_prefix)

    def _find_first_non_finite_input_tensor(self, data: Any) -> tuple[str, torch.Tensor] | None:
        for name, tensor in self._iter_named_input_tensors(data):
            if not self._is_finite_tensor(tensor):
                return name, tensor
        return None

    def _find_first_non_finite_parameter(self) -> tuple[str, torch.Tensor] | None:
        for name, param in self._model.named_parameters():
            if not self._is_finite_tensor(param):
                return name, param
        return None

    def _find_first_non_finite_buffer(self) -> tuple[str, torch.Tensor] | None:
        for name, buffer in self._model.named_buffers():
            if not self._is_finite_tensor(buffer):
                return name, buffer
        return None

    def _find_first_non_finite_gradient(self) -> tuple[str, torch.Tensor] | None:
        for name, param in self._model.named_parameters():
            gradient = param.grad
            if gradient is None:
                continue
            if not self._is_finite_tensor(gradient):
                return name, gradient
        return None
