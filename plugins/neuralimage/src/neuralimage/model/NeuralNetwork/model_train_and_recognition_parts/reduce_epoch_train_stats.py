"""model_train_and_recognition: reduce epoch train stats."""
from __future__ import annotations
import torch
import torch.distributed as dist


class ReduceEpochTrainStatsMixin:
    @staticmethod
    def _reduce_epoch_train_stats(
        epoch_stats: _EpochStats,
        *,
        distributed: bool,
        device: torch.device,
    ) -> tuple[float, float]:
        loss_sum_tensor = epoch_stats.train_loss_sum
        if distributed:
            reduce_tensor = torch.zeros(2, dtype=torch.float64, device=device)
            if loss_sum_tensor is not None:
                reduce_tensor[0] = loss_sum_tensor.detach().to(device=device, dtype=torch.float64)
            reduce_tensor[1] = float(epoch_stats.train_samples_count)
            dist.all_reduce(reduce_tensor, op=dist.ReduceOp.SUM)
            return float(reduce_tensor[0].item()), max(1.0, float(reduce_tensor[1].item()))
        local_loss_sum = 0.0
        if loss_sum_tensor is not None:
            local_loss_sum = float(loss_sum_tensor.detach().to(device='cpu', dtype=torch.float64).item())
        return local_loss_sum, float(max(1, epoch_stats.train_samples_count))
