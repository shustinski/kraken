"""model_train_and_recognition: stop process descendants."""
from __future__ import annotations
from typing import TYPE_CHECKING
from typing import Any, Protocol
import torch

def _stop_process_descendants(processes: list[Any], *, force: bool, timeout: float) -> None:
    if not processes:
        return
    action_name = 'kill' if force else 'terminate'
    for child in reversed(processes):
        try:
            getattr(child, action_name)()
        except Exception:
            pass
    try:
        import psutil

        _gone, alive = psutil.wait_procs(processes, timeout=max(0.0, float(timeout)))
    except Exception:
        return
    if force:
        for child in alive:
            try:
                child.kill()
            except Exception:
                pass
        try:
            psutil.wait_procs(alive, timeout=max(0.0, float(timeout)))
        except Exception:
            pass

def _ddp_worker_entry(rank: int, trainer: 'TrainerProcess', world_size: int, master_port: int) -> None:
    trainer._run_ddp_worker(rank=rank, world_size=world_size, master_port=master_port)

class _SupportsSetEpoch(Protocol):
    def set_epoch(self) -> None:
        ...

class _SupportsLossAwareSampling(Protocol):
    def update_batch_losses(self, sample_indices: torch.Tensor, sample_losses: torch.Tensor) -> None:
        ...

    def resize(self, size: int, *, reset: bool = False) -> None:
        ...

if TYPE_CHECKING:
    from neuralimage.model.NeuralNetwork.model_train_and_recognition import TrainerProcess
