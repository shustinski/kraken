"""model_train_and_recognition: has non finite gradients."""
from __future__ import annotations
import torch
import torch.distributed as dist


class HasNonFiniteGradientsMixin:
    def _has_non_finite_gradients(self) -> bool:
        if self._find_first_non_finite_gradient() is not None:
            return True
        loss_weighter = getattr(self, '_loss_weighter', None)
        return bool(
            loss_weighter is not None
            and any(
                parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all())
                for parameter in loss_weighter.parameters()
            )
        )

    def _synchronize_loss_weighter_gradients(self) -> None:
        loss_weighter = getattr(self, '_loss_weighter', None)
        if loss_weighter is None or not dist.is_available() or not dist.is_initialized():
            return
        world_size = max(1, int(dist.get_world_size()))
        for parameter in loss_weighter.parameters():
            if parameter.grad is None:
                continue
            dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM)
            parameter.grad.div_(float(world_size))
