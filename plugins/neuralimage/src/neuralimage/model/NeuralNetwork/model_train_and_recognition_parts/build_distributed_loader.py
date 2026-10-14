"""model_train_and_recognition: build distributed loader."""
from __future__ import annotations
import os
import datetime
import socket
from pathlib import Path
from typing import Any
import torch
import torch.distributed as dist
import torch.multiprocessing as torch_mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, RandomSampler, Sampler
from torch.utils.data.distributed import DistributedSampler

from neuralimage.model.NeuralNetwork.model_train_and_recognition_parts.module_state import CHECKPOINT_SUFFIX

from neuralimage.model.NeuralNetwork.model_train_and_recognition_parts.combined_optimizer import _NoOpQueue

from neuralimage.model.NeuralNetwork.model_train_and_recognition_parts.distributed_loss_aware_sampler import _DistributedLossAwareSampler

from neuralimage.model.NeuralNetwork.model_train_and_recognition_parts.stop_process_descendants import _ddp_worker_entry


class BuildDistributedLoaderMixin:
    @staticmethod
    def _format_elapsed_duration(seconds: float) -> str:
        total_seconds = int(max(0.0, round(float(seconds))))
        hours, remainder = divmod(total_seconds, 3600)
        minutes, secs = divmod(remainder, 60)
        return f'{hours:02d}:{minutes:02d}:{secs:02d}'

    def _save_training_log(self) -> Path:
        log_path = self._save_path.parent / 'training_log.txt'
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_text = '\n'.join(str(line) for line in self._training_log_lines)
        if log_text and not log_text.endswith('\n'):
            log_text += '\n'
        log_path.write_text(log_text, encoding='utf-8')
        return log_path

    @staticmethod
    def _find_free_port() -> int:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(('127.0.0.1', 0))
            sock.listen(1)
            return int(sock.getsockname()[1])

    @staticmethod
    def _safe_loader_len(loader: DataLoader | None) -> int:
        if loader is None:
            return 0
        try:
            return int(len(loader))
        except Exception:
            return 0

    @staticmethod
    def _build_distributed_loader(base_loader: DataLoader, rank: int, world_size: int, shuffle: bool) -> DataLoader:
        dataset = base_loader.dataset
        original_sampler = getattr(base_loader, 'sampler', None)
        if hasattr(original_sampler, 'update_batch_losses'):
            sampler: Sampler[int] = _DistributedLossAwareSampler(
                original_sampler,
                rank=rank,
                world_size=world_size,
                drop_last=bool(getattr(base_loader, 'drop_last', False)),
            )
        else:
            sampler = DistributedSampler(
                dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=shuffle,
                drop_last=bool(getattr(base_loader, 'drop_last', False)),
            )
        loader_kwargs: dict[str, Any] = {
            'dataset': dataset,
            'batch_size': int(getattr(base_loader, 'batch_size', 1) or 1),
            'sampler': sampler,
            'num_workers': int(getattr(base_loader, 'num_workers', 0)),
            'pin_memory': bool(getattr(base_loader, 'pin_memory', False)),
            'drop_last': bool(getattr(base_loader, 'drop_last', False)),
        }
        collate_fn = getattr(base_loader, 'collate_fn', None)
        if collate_fn is not None:
            loader_kwargs['collate_fn'] = collate_fn
        if loader_kwargs['num_workers'] > 0:
            prefetch_factor = getattr(base_loader, 'prefetch_factor', None)
            if prefetch_factor is not None:
                loader_kwargs['prefetch_factor'] = prefetch_factor
            loader_kwargs['persistent_workers'] = bool(getattr(base_loader, 'persistent_workers', False))
        return DataLoader(**loader_kwargs)

    def _configure_ddp_dataloaders(self, *, rank: int, world_size: int) -> None:
        if self._train_dataloader is None:
            raise RuntimeError('Train dataloader is not initialized for DDP.')

        sampler_obj = getattr(self._train_dataloader, 'sampler', None)
        train_shuffle = bool(getattr(self._train_dataloader, 'shuffle', False))
        if (not train_shuffle) and isinstance(sampler_obj, RandomSampler):
            train_shuffle = True
        self._train_dataloader = self._build_distributed_loader(
            self._train_dataloader,
            rank=rank,
            world_size=world_size,
            shuffle=train_shuffle,
        )

        if self._val_dataloader is None:
            return
        if rank == 0:
            return
        # Validation is executed only on rank 0. Non-main ranks should not shard
        # or iterate validation data because this distorts metrics and adds
        # unnecessary loader overhead.
        self._val_dataloader = None

    def _has_multi_gpu_cuda(self) -> bool:
        if not torch.cuda.is_available():
            return False
        return torch.cuda.device_count() > 1

    def _log_multi_gpu_unavailable(self, requested_mode: str) -> None:
        if requested_mode == 'off':
            self._bus.put(['logging', 'Multi-GPU mode is disabled in settings.'])
            return
        if not torch.cuda.is_available():
            self._bus.put(['logging', f'multi-GPU mode "{requested_mode}" skipped: CUDA is unavailable.'])
            return
        gpu_count = int(torch.cuda.device_count())
        if gpu_count <= 1:
            self._bus.put(['logging', f'multi-GPU mode "{requested_mode}" skipped: found {gpu_count} GPU.'])
            return
        if requested_mode == 'distributeddataparallel' and os.name == 'nt':
            self._bus.put(['logging', 'DDP requested but disabled on Windows in this build.'])
            return
        self._bus.put(['logging', f'multi-GPU mode "{requested_mode}" unavailable; using single GPU.'])

    def _should_use_ddp(self) -> bool:
        if self._multi_gpu_mode == 'off':
            self._bus.put(['logging', 'Режим multi GPU отключен в настройках.'])
            return False
        if self._multi_gpu_mode != 'distributeddataparallel':
            return False
        if not self._has_multi_gpu_cuda():
            return False
        if os.name == 'nt':
            # On Windows, DDP+gloo can hang during rendezvous on some setups.
            return False
        return True

    def _should_use_data_parallel(self) -> bool:
        if self._multi_gpu_mode != 'dataparallel':
            return False
        return self._has_multi_gpu_cuda()

    def _run_ddp(self, world_size: int) -> None:
        master_port = self._find_free_port()
        self._bus.put(['logging', f'Включен режим multi GPU: DistributedDataParallel на {world_size} GPU.'])
        torch_mp.spawn(
            _ddp_worker_entry,
            args=(self, world_size, master_port),
            nprocs=world_size,
            join=True,
        )

    def _run_ddp_worker(self, rank: int, world_size: int, master_port: int) -> None:
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = str(master_port)
        backend = 'nccl'
        if os.name == 'nt' or not dist.is_nccl_available():
            backend = 'gloo'
        dist.init_process_group(
            backend=backend,
            rank=rank,
            world_size=world_size,
            init_method=f'tcp://127.0.0.1:{master_port}',
            timeout=datetime.timedelta(seconds=120),
        )
        try:
            torch.cuda.set_device(rank)
            device = torch.device(f'cuda:{rank}')
            if rank != 0:
                self._bus = _NoOpQueue()
                self._show_batch_preview = False

            self._configure_ddp_dataloaders(rank=rank, world_size=world_size)

            self._model.to(device)
            self._try_compile_model(is_main_process=(rank == 0), device=device)
            self._model = DDP(self._model, device_ids=[rank], output_device=rank)
            self._run_impl(device=device, rank=rank, world_size=world_size, distributed=True)
        finally:
            dist.destroy_process_group()

    def _checkpoint_path(self) -> Path:
        return self._save_path.with_suffix(CHECKPOINT_SUFFIX)
