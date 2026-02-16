import os
import datetime
import time
import socket
import sys
from dataclasses import dataclass
from time import sleep
from pathlib import Path
from queue import Empty
import os

from collections.abc import Callable, Sized
from contextlib import nullcontext
from multiprocessing.synchronize import Event as MpEvent
from typing import Any, ContextManager, Protocol, cast

import multiprocessing as mp
import threading

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as torch_mp
from torch import optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data.distributed import DistributedSampler

from lib import System
from model.NeuralNetwork.dataset import CustomDataset
from lib.data_interfaces import (
    RecognitionParameters,
    OptimizerParameters,
    OptimizerName,
    MixedPrecisionMode,
    EarlyStoppingParameters,
    WarmupParameters,
)
from lib.file_func import filter_images
from lib.func import get_input_channels
from lib.image_processing import cut_image, sew_image
from lib.images import save_color
from lib.message_bus import AbstractMessageBus

CHECKPOINT_SUFFIX = '.ckpt'
STOP_TOKEN = '__STOP__'


class _NoOpQueue:
    def put(self, item: Any) -> None:
        return


def _ddp_worker_entry(rank: int, trainer: 'TrainerProcess', world_size: int, master_port: int) -> None:
    trainer._run_ddp_worker(rank=rank, world_size=world_size, master_port=master_port)


class _SupportsSetEpoch(Protocol):
    def set_epoch(self) -> None:
        ...


def _release_torch_memory() -> None:
    if not torch.cuda.is_available():
        return
    try:
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    except Exception:
        # Best-effort release; ignore cleanup failures.
        pass


def _collect_memory_metrics() -> dict[str, float] | None:
    ram_mb: float | None = None
    try:
        import psutil  # type: ignore

        ram_mb = float(psutil.Process(os.getpid()).memory_info().rss) / (1024.0 * 1024.0)
    except Exception:
        ram_mb = None

    vram_alloc_mb: float | None = None
    vram_reserved_mb: float | None = None
    if torch.cuda.is_available():
        try:
            vram_alloc_mb = float(torch.cuda.memory_allocated()) / (1024.0 * 1024.0)
            vram_reserved_mb = float(torch.cuda.memory_reserved()) / (1024.0 * 1024.0)
        except Exception:
            vram_alloc_mb = None
            vram_reserved_mb = None

    if ram_mb is None and vram_alloc_mb is None and vram_reserved_mb is None:
        return None

    payload: dict[str, float] = {}
    if ram_mb is not None:
        payload['ram_mb'] = ram_mb
    if vram_alloc_mb is not None:
        payload['vram_allocated_mb'] = vram_alloc_mb
    if vram_reserved_mb is not None:
        payload['vram_reserved_mb'] = vram_reserved_mb
    return payload


class ModelTrainer(threading.Thread):
    def __init__(self,train_dataloader:DataLoader, val_dataloader:DataLoader | None,
                 model:nn.Module, save_path:Path,epochs:int, message_bus:AbstractMessageBus,
                 callback:Callable[..., None]|None = None,
                 optimizer_params: OptimizerParameters | None = None,
                 mixed_precision: MixedPrecisionMode = MixedPrecisionMode.bf16,
                 early_stopping_params: EarlyStoppingParameters | None = None,
                 warmup_params: WarmupParameters | None = None,
                 resume_from_checkpoint: bool = False,
                 use_multi_gpu: bool = True,
                 show_batch_preview: bool = True):
        super().__init__()
        self._train_dataloader = train_dataloader
        self._val_dataloader = val_dataloader
        self._model = model
        self._save_path = save_path
        self._epochs = epochs
        self._bus = message_bus
        self.callback = callback
        self._optimizer_params = optimizer_params or OptimizerParameters()
        self._mixed_precision = mixed_precision
        self._early_stopping_params = early_stopping_params or EarlyStoppingParameters()
        self._warmup_params = warmup_params or WarmupParameters()
        self._resume_from_checkpoint = resume_from_checkpoint
        self._use_multi_gpu = use_multi_gpu
        self._show_batch_preview = show_batch_preview
        self._stop_event = threading.Event()
        self.message_queue = mp.Queue()
        self.succeeded = False
        self.error_message: str | None = None


    def run(self):
        training_process = None
        try:
            if self._train_dataloader is None or self._model is None:
                self.error_message = 'Ошибка: отсутствуют данные обучения или модель для запуска процесса.'
                self._bus.publish('error', self.error_message)
                return

            train_dataloader = cast(DataLoader, self._train_dataloader)
            val_dataloader = cast(DataLoader | None, self._val_dataloader)
            model = cast(nn.Module, self._model)

            training_process = TrainerProcess(train_dataloader, val_dataloader, model,
                                              self._save_path, self._epochs, self.message_queue,
                                              optimizer_params=self._optimizer_params,
                                              mixed_precision=self._mixed_precision,
                                              early_stopping_params=self._early_stopping_params,
                                              warmup_params=self._warmup_params,
                                              resume_from_checkpoint=self._resume_from_checkpoint,
                                              use_multi_gpu=self._use_multi_gpu,
                                              show_batch_preview=self._show_batch_preview)
            training_process.start()
            current_time = time.perf_counter()
            while training_process.is_alive():
                if self._stop_event.is_set():
                    training_process.kill()
                    break
                if not self.message_queue.empty():
                    new_message = self.message_queue.get()
                    elapsed_seconds = int(time.perf_counter() - current_time)
                    elapsed_hours, remainder = divmod(elapsed_seconds, 3600)
                    elapsed_minutes, elapsed_secs = divmod(remainder, 60)
                    elapsed_suffix = f' Время с прошлого сообщения: {elapsed_hours:02d}:{elapsed_minutes:02d}:{elapsed_secs:02d}'
                    payload = new_message[1]
                    if isinstance(payload, str) and new_message[0] != 'error':
                        payload = payload + elapsed_suffix
                    if new_message[0] == 'error' and isinstance(payload, str):
                        self.error_message = payload
                    self._bus.publish(new_message[0], payload)
                time.sleep(1)

            training_process.join()

            while not self.message_queue.empty():
                queued_message = self.message_queue.get()
                topic = queued_message[0]
                payload = queued_message[1]
                if topic == 'error' and isinstance(payload, str):
                    self.error_message = payload
                self._bus.publish(topic, payload)

            if self._stop_event.is_set():
                self.succeeded = False
                return

            exit_code = training_process.exitcode if training_process is not None else 1
            if exit_code not in (0, None):
                if self.error_message is None:
                    self.error_message = f'Ошибка обучения: процесс завершился с кодом {exit_code}.'
                    self._bus.publish('error', self.error_message)
                self.succeeded = False
                return

            self.succeeded = True
            if self.callback is not None:
                self.callback()
            print("Model saved successfully!")
        finally:
            if training_process is not None and training_process.is_alive():
                training_process.kill()
                training_process.join(timeout=5)
            try:
                self.message_queue.close()
            except Exception:
                pass
            # Drop heavyweight references from parent thread after process completion.
            self._model = None
            self._train_dataloader = None
            self._val_dataloader = None
            _release_torch_memory()

    def stop(self):
        self._stop_event.set()

class TrainerProcess(mp.Process):
    def __init__(self,train_dataloader:DataLoader, val_dataloader:DataLoader | None,
                 model:nn.Module, save_path:Path, epochs:int, message_bus:mp.Queue,
                 callback:Callable[...,None]|None = None,
                 optimizer_params: OptimizerParameters | None = None,
                 mixed_precision: MixedPrecisionMode = MixedPrecisionMode.bf16,
                 early_stopping_params: EarlyStoppingParameters | None = None,
                 warmup_params: WarmupParameters | None = None,
                 resume_from_checkpoint: bool = False,
                 use_multi_gpu: bool = True,
                 show_batch_preview: bool = True):
        super().__init__()
        self._train_dataloader = train_dataloader
        self._val_dataloader = val_dataloader
        self._model = model
        self._save_path = save_path
        self._epochs = epochs
        self._bus = message_bus
        self.callback = callback
        self._optimizer_params = optimizer_params or OptimizerParameters()
        self._mixed_precision = mixed_precision
        self._early_stopping_params = early_stopping_params or EarlyStoppingParameters()
        self._warmup_params = warmup_params or WarmupParameters()
        self._resume_from_checkpoint = resume_from_checkpoint
        self._use_multi_gpu = use_multi_gpu
        self._show_batch_preview = show_batch_preview

    @property
    def _base_model(self) -> nn.Module:
        """Return the original model instance even if wrapped by DDP/DataParallel."""
        if isinstance(self._model, (DDP, nn.DataParallel)):
            return self._model.module
        return self._model

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

    def _should_use_ddp(self) -> bool:
        if not self._use_multi_gpu:
            self._bus.put(['logging', 'Режим multi GPU отключен в настройках.'])
            return False
        if not torch.cuda.is_available():
            return False
        return torch.cuda.device_count() > 1

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
        dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
        try:
            torch.cuda.set_device(rank)
            device = torch.device(f'cuda:{rank}')
            if rank != 0:
                self._bus = _NoOpQueue()
                self._show_batch_preview = False

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
            if self._val_dataloader is not None:
                self._val_dataloader = self._build_distributed_loader(
                    self._val_dataloader,
                    rank=rank,
                    world_size=world_size,
                    shuffle=False,
                )

            self._model.to(device)
            self._try_compile_model(is_main_process=(rank == 0))
            self._model = DDP(self._model, device_ids=[rank], output_device=rank)
            self._run_impl(device=device, rank=rank, world_size=world_size, distributed=True)
        finally:
            dist.destroy_process_group()

    def _checkpoint_path(self) -> Path:
        return self._save_path.with_suffix(CHECKPOINT_SUFFIX)

    def _save_checkpoint(
        self,
        completed_epoch: int,
        optimizer,
        scaler,
        scheduler,
        early_stopping_best_loss: float | None = None,
        early_stopping_bad_epochs: int = 0,
        early_stopping_best_epoch: int = 0,
        early_stopping_best_model_state: dict[str, torch.Tensor] | None = None,
    ) -> None:
        checkpoint = {
            'version': 1,
            'completed_epoch': completed_epoch,
            'epochs': self._epochs,
            'model_state_dict': self._base_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'optimizer_name': self._optimizer_params.name.value,
            'learning_rate': self._optimizer_params.learning_rate,
            'weight_decay': self._optimizer_params.weight_decay,
            'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
            'early_stopping_best_loss': early_stopping_best_loss,
            'early_stopping_bad_epochs': early_stopping_bad_epochs,
            'early_stopping_best_epoch': early_stopping_best_epoch,
            'early_stopping_best_model_state': early_stopping_best_model_state,
        }
        torch.save(checkpoint, self._checkpoint_path())

    def _load_checkpoint_if_available(self, optimizer, scaler, scheduler) -> tuple[int, dict[str, Any]]:
        if not self._resume_from_checkpoint:
            return 0, {}

        checkpoint_path = self._checkpoint_path()
        if not checkpoint_path.exists():
            self._bus.put(['logging', 'Файл контрольной точки не найден. Обучение начнется с первой эпохи.'])
            return 0, {}

        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            model_state = checkpoint.get('model_state_dict')
            optimizer_state = checkpoint.get('optimizer_state_dict')
            scaler_state = checkpoint.get('scaler_state_dict')
            scheduler_state = checkpoint.get('scheduler_state_dict')
            completed_epoch = int(checkpoint.get('completed_epoch', 0))
            if model_state:
                self._base_model.load_state_dict(model_state)
            if optimizer_state:
                optimizer.load_state_dict(optimizer_state)
            if scaler_state:
                scaler.load_state_dict(scaler_state)
            if scheduler is not None and scheduler_state:
                scheduler.load_state_dict(scheduler_state)

            start_epoch = max(0, min(completed_epoch, self._epochs))
            self._bus.put([
                'logging',
                f'Параметры восстановлены из контрольной точки {checkpoint_path.name}. Последняя завершенная эпоха: {completed_epoch}.',
            ])
            return start_epoch, checkpoint
        except Exception as error:
            self._bus.put(['logging', f'Ошибка загрузки контрольной точки: {error}. Обучение начнется с первой эпохи.'])
            return 0, {}

    def _create_warmup_scheduler(self, optimizer, train_steps_per_epoch: int):
        warmup = self._warmup_params
        if not warmup.enabled:
            return None
        if train_steps_per_epoch <= 0:
            return None

        warmup_epochs = max(1, int(warmup.epochs))
        warmup_steps = warmup_epochs * train_steps_per_epoch
        start_factor = float(min(max(warmup.start_factor, 0.0), 1.0))

        def lr_lambda(step: int) -> float:
            if step >= warmup_steps:
                return 1.0
            progress = (step + 1) / warmup_steps
            return start_factor + (1.0 - start_factor) * progress

        return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    def _resolve_target_epochs(self, start_epoch: int) -> int:
        """
        In fine-tuning mode the UI epoch value is treated as "additional epochs".
        For fresh training it is treated as total epochs.
        """
        if self._resume_from_checkpoint and start_epoch > 0:
            return start_epoch + int(self._epochs)
        return int(self._epochs)

    def _run_validation_epoch(
        self,
        device: torch.device,
        criterion: nn.Module,
        autocast_ctx: Callable[[], ContextManager[Any]],
    ) -> dict[str, float] | None:
        if self._val_dataloader is None:
            return None
        val_dataset_len = len(cast(Sized, self._val_dataloader.dataset))
        if val_dataset_len == 0:
            return None

        self._model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        true_positive = 0.0
        false_positive = 0.0
        false_negative = 0.0
        with torch.no_grad():
            for data, target in self._val_dataloader:
                image = data.to(device, non_blocking=True)
                label = target.to(device, non_blocking=True)
                with autocast_ctx():
                    outputs = self._model(image)
                    loss = criterion(outputs, label)

                val_loss += loss.item() * image.size(0)
                probs = torch.sigmoid(outputs)
                preds = probs >= 0.5
                label_bin = label >= 0.5
                correct += (preds == label_bin).sum().item()
                total += label_bin.numel()
                preds_f = preds.float()
                label_f = label_bin.float()
                true_positive += float((preds_f * label_f).sum().item())
                false_positive += float((preds_f * (1.0 - label_f)).sum().item())
                false_negative += float(((1.0 - preds_f) * label_f).sum().item())

        avg_val_loss = val_loss / val_dataset_len
        val_accuracy = (correct / total) if total else 0.0
        iou_denom = true_positive + false_positive + false_negative
        dice_denom = (2.0 * true_positive) + false_positive + false_negative
        if iou_denom == 0.0:
            iou = 1.0
            dice = 1.0
            f1 = 1.0
        else:
            iou = true_positive / iou_denom
            dice = (2.0 * true_positive) / dice_denom if dice_denom > 0.0 else 0.0
            precision_denom = true_positive + false_positive
            recall_denom = true_positive + false_negative
            precision = true_positive / precision_denom if precision_denom > 0.0 else 0.0
            recall = true_positive / recall_denom if recall_denom > 0.0 else 0.0
            f1_denom = precision + recall
            f1 = (2.0 * precision * recall) / f1_denom if f1_denom > 0.0 else 0.0

        return {
            'loss': float(avg_val_loss),
            'accuracy': float(val_accuracy),
            'iou': float(iou),
            'dice': float(dice),
            'f1': float(f1),
        }

    def _create_optimizer(self):
        params = self._optimizer_params
        if params.name == OptimizerName.adam:
            adam_kwargs: dict[str, Any] = {'lr': params.learning_rate, 'weight_decay': params.weight_decay}
            if self._can_use_fused_optim():
                adam_kwargs['fused'] = True
            try:
                return optim.Adam(self._model.parameters(), **adam_kwargs)
            except TypeError:
                adam_kwargs.pop('fused', None)
                return optim.Adam(self._model.parameters(), **adam_kwargs)
        if params.name == OptimizerName.adamw:
            return self._create_adamw_optimizer(params)
        if params.name == OptimizerName.adamw_muon:
            return self._create_adamw_muon_optimizer(params)
        raise ValueError(f'Unsupported optimizer: {params.name}')

    def _can_use_fused_optim(self) -> bool:
        if not torch.cuda.is_available():
            return False
        try:
            return any(p.is_cuda for p in self._model.parameters())
        except Exception:
            return False

    def _resolve_mixed_precision(self, device: torch.device) -> tuple[MixedPrecisionMode, torch.dtype | None, bool]:
        if device.type != 'cuda':
            return MixedPrecisionMode.off, None, False

        mode = self._mixed_precision
        if mode == MixedPrecisionMode.off:
            return mode, None, False
        if mode == MixedPrecisionMode.fp16:
            return mode, torch.float16, True
        if mode == MixedPrecisionMode.bf16:
            is_bf16_supported = False
            try:
                is_bf16_supported = bool(torch.cuda.is_bf16_supported())
            except Exception:
                is_bf16_supported = False
            if is_bf16_supported:
                return mode, torch.bfloat16, False
            return MixedPrecisionMode.fp16, torch.float16, True
        return MixedPrecisionMode.off, None, False

    def _build_adamw_param_groups(self, params: OptimizerParameters) -> list[dict[str, Any]]:
        norm_param_ids: set[int] = set()
        batch_norm_base = getattr(nn.modules.batchnorm, '_BatchNorm', ())
        norm_types = (nn.GroupNorm, batch_norm_base) if batch_norm_base else (nn.GroupNorm,)

        for module in self._base_model.modules():
            if isinstance(module, norm_types):
                for p in module.parameters(recurse=False):
                    if p.requires_grad:
                        norm_param_ids.add(id(p))

        decay_params: list[torch.nn.Parameter] = []
        no_decay_params: list[torch.nn.Parameter] = []
        for param in self._model.parameters():
            if not param.requires_grad:
                continue
            if id(param) in norm_param_ids:
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        param_groups: list[dict[str, Any]] = []
        if decay_params:
            param_groups.append({'params': decay_params, 'weight_decay': params.weight_decay})
        if no_decay_params:
            param_groups.append({'params': no_decay_params, 'weight_decay': 0.0})
        return param_groups

    def _create_adamw_optimizer(self, params: OptimizerParameters):
        use_fused = self._can_use_fused_optim()
        param_groups = self._build_adamw_param_groups(params)
        if param_groups:
            self._bus.put([
                'logging',
                f'AdamW: исключены norm-параметры из weight decay (групп: {len(param_groups)}).',
            ])
            adamw_kwargs: dict[str, Any] = {'lr': params.learning_rate}
            if use_fused:
                adamw_kwargs['fused'] = True
            try:
                return optim.AdamW(param_groups, **adamw_kwargs)
            except TypeError:
                adamw_kwargs.pop('fused', None)
                return optim.AdamW(param_groups, **adamw_kwargs)
        adamw_kwargs = {'lr': params.learning_rate, 'weight_decay': params.weight_decay}
        if use_fused:
            adamw_kwargs['fused'] = True
        try:
            return optim.AdamW(self._model.parameters(), **adamw_kwargs)
        except TypeError:
            adamw_kwargs.pop('fused', None)
            return optim.AdamW(self._model.parameters(), **adamw_kwargs)

    def _create_adamw_muon_optimizer(self, params: OptimizerParameters):
        """
        Prepared hook for Muon integration.
        Tries to use optional Muon package; if unavailable falls back to AdamW.
        """
        muon_cls = None
        for module_name in ('muon', 'optimizers.muon', 'torch_optimizer'):
            try:
                module = __import__(module_name, fromlist=['Muon'])
                muon_cls = getattr(module, 'Muon', None)
                if muon_cls is not None:
                    break
            except Exception:
                continue

        if muon_cls is None:
            self._bus.put([
                'logging',
                'Оптимизатор Muon недоступен. Используется AdamW.',
            ])
            return self._create_adamw_optimizer(params)

        return muon_cls(self._model.parameters(), lr=params.learning_rate, weight_decay=params.weight_decay)

    @staticmethod
    def _tensor_to_preview_array(tensor: torch.Tensor) -> np.ndarray:
        arr = tensor.detach().cpu().float().numpy()
        if arr.ndim == 3 and arr.shape[0] in (1, 3):
            arr = np.transpose(arr, (1, 2, 0))
        elif arr.ndim == 3:
            arr = arr[:, :, 0]

        if arr.ndim == 3 and arr.shape[2] == 1:
            arr = arr[:, :, 0]

        min_val = float(arr.min()) if arr.size else 0.0
        max_val = float(arr.max()) if arr.size else 0.0
        if 0.0 <= min_val and max_val <= 1.0:
            arr = arr * 255.0
        elif max_val > min_val:
            arr = (arr - min_val) * (255.0 / (max_val - min_val))
        else:
            arr = np.zeros_like(arr)
        return np.clip(arr, 0, 255).astype(np.uint8)

    def _try_compile_model(self, is_main_process: bool = True) -> None:
        if bool(getattr(sys, 'frozen', False)):
            if is_main_process:
                self._bus.put(['logging', 'torch.compile отключен в сборке PyInstaller.'])
            return
        compile_fn = getattr(torch, 'compile', None)
        if compile_fn is None:
            if is_main_process:
                self._bus.put(['logging', 'torch.compile недоступен в этой версии PyTorch.'])
            return
        try:
            self._model = compile_fn(self._model, mode='max-autotune', dynamic=False)
            if is_main_process:
                self._bus.put(['logging', 'Включен torch.compile (mode=max-autotune).'])
        except Exception as error:
            if is_main_process:
                self._bus.put(['logging', f'torch.compile отключен (fallback): {error}'])

    def run(self):
        try:
            if self._should_use_ddp():
                self._run_ddp(world_size=torch.cuda.device_count())
                return

            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self._model.to(device)
            self._run_impl(device=device, rank=0, world_size=1, distributed=False)
        except Exception as error:
            self._bus.put(['error', f'Критическая ошибка обучения: {error}'])
            raise

    def _run_impl(self, device: torch.device, rank: int, world_size: int, distributed: bool):
        is_main_process = (rank == 0)
        if device.type == 'cuda':
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        if not distributed:
            self._try_compile_model(is_main_process=is_main_process)

        criterion = nn.BCEWithLogitsLoss()
        optimizer = self._create_optimizer()

        # criterion = nn.BCEWithLogitsLoss()
        # optimizer = torch.optim.AdamW(self._model.parameters(), lr=1e-3, weight_decay=1e-4)
        # #
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
        #                                                        T_max=self._epochs,
        #                                                        eta_min=1e-4)

        if is_main_process:
            self._bus.put(['logging', f'Обнаружено устройство для обучения: {device}'])
        train_size = len(self._train_dataloader) if self._train_dataloader is not None else 0
        # Keep metric traffic bounded, otherwise IPC/UI overhead can dominate on fast GPUs.
        train_metric_stride = max(1, (train_size + 39) // 40)
        train_progress_stride = max(1, (train_size + 39) // 40)
        train_log_stride = max(25, train_metric_stride)
        preview_stride = max(100, train_metric_stride)
        self._bus.put([
            'metrics',
            {'type': 'train_epoch_progress', 'current': 0, 'total': int(self._epochs)},
        ])
        self._bus.put([
            'metrics',
            {'type': 'train_batch_progress', 'current': 0, 'total': int(train_size)},
        ])
        memory_payload = _collect_memory_metrics()
        if memory_payload is not None:
            self._bus.put(['metrics', {'type': 'system_memory', **memory_payload}])

        self._bus.put(['logging', 'Подготовка данных и запуск цикла обучения'])
        resolved_mp_mode, autocast_dtype, scaler_enabled = self._resolve_mixed_precision(device)
        if resolved_mp_mode != self._mixed_precision:
            self._bus.put([
                'logging',
                f'Режим mixed precision "{self._mixed_precision.value}" недоступен на {device.type}. Используется режим "{resolved_mp_mode.value}".',
            ])
        else:
            self._bus.put(['logging', f'Режим mixed precision: {resolved_mp_mode.value}.'])

        scaler = torch.cuda.amp.GradScaler(enabled=scaler_enabled)
        autocast_ctx: Callable[[], ContextManager[Any]] = (
            (lambda: torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=True))
            if autocast_dtype is not None
            else (lambda: nullcontext())
        )
        scheduler = self._create_warmup_scheduler(optimizer, train_size)
        if scheduler is not None:
            self._bus.put([
                'logging',
                f'Warmup включен: эпох={int(self._warmup_params.epochs)}, start_factor={float(self._warmup_params.start_factor):.4f}.',
            ])

        start_epoch, checkpoint = self._load_checkpoint_if_available(optimizer, scaler, scheduler)
        early_stopping_best_loss = checkpoint.get('early_stopping_best_loss')
        if early_stopping_best_loss is not None:
            early_stopping_best_loss = float(early_stopping_best_loss)
        else:
            early_stopping_best_loss = None
        early_stopping_bad_epochs = int(checkpoint.get('early_stopping_bad_epochs', 0))
        early_stopping_best_epoch = int(checkpoint.get('early_stopping_best_epoch', 0))
        early_stopping_best_model_state = checkpoint.get('early_stopping_best_model_state')

        has_validation = bool(self._val_dataloader is not None)
        if has_validation and self._val_dataloader is not None:
            has_validation = len(cast(Sized, self._val_dataloader.dataset)) > 0
        early_stopping_enabled = bool(self._early_stopping_params.enabled and has_validation)
        early_stopping_patience = max(0, int(self._early_stopping_params.patience))
        early_stopping_min_delta = max(0.0, float(self._early_stopping_params.min_delta))
        if self._early_stopping_params.enabled and not has_validation:
            self._bus.put([
                'logging',
                'Early stopping включен, но валидационный датасет отсутствует. Early stopping отключен.',
            ])
        elif early_stopping_enabled:
            self._bus.put([
                'logging',
                f'Early stopping включен: patience={early_stopping_patience}, min_delta={early_stopping_min_delta:.6f}.',
            ])

        target_epochs = self._resolve_target_epochs(start_epoch)
        if target_epochs != self._epochs:
            self._bus.put([
                'logging',
                f'Режим дообучения: продолжение с эпохи {start_epoch}. Будет добавлено {self._epochs} эпох (до {target_epochs}).',
            ])
            self._epochs = target_epochs
            self._bus.put([
                'metrics',
                {'type': 'train_epoch_progress', 'current': int(start_epoch), 'total': int(self._epochs)},
            ])

        if start_epoch >= self._epochs:
            self._bus.put(['logging', 'Все эпохи из контрольной точки уже выполнены. Дополнительное обучение не требуется.'])
        for epoch in range(start_epoch, self._epochs):
            self._model.train()  # Set model to training mode
            if distributed:
                sampler = getattr(self._train_dataloader, 'sampler', None)
                if isinstance(sampler, DistributedSampler):
                    sampler.set_epoch(epoch)
            self._bus.put(['logging', f'Начало эпохи [{epoch + 1}/{self._epochs}]'])
            current_lr = float(optimizer.param_groups[0]['lr'])
            self._bus.put(['logging', f'Текущий learning rate: {current_lr:.8f}'])
            self._bus.put([
                'metrics',
                {'type': 'train_epoch_progress', 'current': int(epoch + 1), 'total': int(self._epochs)},
            ])
            self._bus.put([
                'metrics',
                {'type': 'train_batch_progress', 'current': 0, 'total': int(train_size)},
            ])
            memory_payload = _collect_memory_metrics()
            if memory_payload is not None:
                self._bus.put(['metrics', {'type': 'system_memory', **memory_payload, 'epoch': int(epoch + 1)}])
            self._bus.put(['logging'," "])
            train_loss = 0
            train_samples_count = 0
            epoch_data_wait_ms = 0.0
            epoch_forward_ms = 0.0
            epoch_backward_ms = 0.0
            epoch_optimizer_ms = 0.0
            epoch_total_ms = 0.0
            prev_batch_end = time.perf_counter()
            train_dataset = self._train_dataloader.dataset
            if hasattr(train_dataset, 'set_epoch'):
                cast(_SupportsSetEpoch, train_dataset).set_epoch()
            for i, (data,target) in enumerate(self._train_dataloader):
                batch_start = time.perf_counter()
                data_wait_ms = (batch_start - prev_batch_end) * 1000.0
                image, label = data.to(device, non_blocking=True), target.to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)
                # Forward pass
                forward_start = time.perf_counter()
                with autocast_ctx():
                    outputs = self._model(image)
                    loss = criterion(outputs, label)  # Compare output with first channel
                forward_ms = (time.perf_counter() - forward_start) * 1000.0

                if self._show_batch_preview and (i % preview_stride == 0 or i == train_size - 1):
                    preview_image = self._tensor_to_preview_array(data[0])
                    preview_label = self._tensor_to_preview_array(target[0])
                    preview_outputs = self._tensor_to_preview_array(torch.sigmoid(outputs[0].detach()))
                    self._bus.put([
                        'metrics',
                        {
                            'type': 'train_batch_preview',
                            'epoch': int(epoch + 1),
                            'batch_index': int(i + 1),
                            'image': preview_image,
                            'label': preview_label,
                            'outputs': preview_outputs,
                        },
                    ])

                # Backward pass
                backward_start = time.perf_counter()
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm(self._model.parameters(), 1.0)
                backward_ms = (time.perf_counter() - backward_start) * 1000.0
                optimizer_start = time.perf_counter()
                scaler.step(optimizer)
                scaler.update()
                if scheduler is not None:
                    scheduler.step()
                optimizer_ms = (time.perf_counter() - optimizer_start) * 1000.0
                batch_total_ms = (time.perf_counter() - batch_start) * 1000.0
                prev_batch_end = time.perf_counter()

                epoch_data_wait_ms += data_wait_ms
                epoch_forward_ms += forward_ms
                epoch_backward_ms += backward_ms
                epoch_optimizer_ms += optimizer_ms
                epoch_total_ms += batch_total_ms


                loss_coeff = loss.item()
                train_loss += loss_coeff * image.size(0)
                train_samples_count += int(image.size(0))
                if (i % train_metric_stride == 0) or (i == train_size - 1):
                    self._bus.put([
                        'metrics',
                        {
                            'type': 'train_batch',
                            'epoch': epoch + 1,
                            'batch_index': i + 1,
                            'loss': float(loss_coeff),
                        },
                    ])
                    self._bus.put([
                        'metrics',
                        {
                            'type': 'train_perf',
                            'epoch': epoch + 1,
                            'batch_index': i + 1,
                            'data_wait_ms': float(data_wait_ms),
                            'forward_ms': float(forward_ms),
                            'backward_ms': float(backward_ms),
                            'optimizer_ms': float(optimizer_ms),
                            'total_ms': float(batch_total_ms),
                        },
                    ])
                if (i % train_progress_stride == 0) or (i == train_size - 1):
                    self._bus.put([
                        'metrics',
                        {'type': 'train_batch_progress', 'current': int(i + 1), 'total': int(train_size)},
                    ])

                if (i % train_log_stride == 0) or (i == train_size - 1):
                    loss = loss.item()
                    self._bus.put(['training', f'Эпоха [{epoch + 1}/{self._epochs}] '
                                               f'Потеря: {loss:>7f} '
                                               f'Пакет: [{i:>5d}/{train_size:>5d}] '
                                               f'| шаг: {batch_total_ms:.1f} мс'])

            if distributed:
                reduce_tensor = torch.tensor([train_loss, float(train_samples_count)], dtype=torch.float64, device=device)
                dist.all_reduce(reduce_tensor, op=dist.ReduceOp.SUM)
                global_train_loss = float(reduce_tensor[0].item())
                global_train_samples = max(1.0, float(reduce_tensor[1].item()))
            else:
                global_train_loss = float(train_loss)
                global_train_samples = float(max(1, train_samples_count))

            if is_main_process and global_train_samples > 0:
                avg_train_loss = global_train_loss / global_train_samples
                self._bus.put([
                    'metrics',
                    {'type': 'train_epoch', 'epoch': epoch + 1, 'loss': float(avg_train_loss)},
                ])
                self._bus.put(['logging', f'Средняя потеря на обучающей выборке: {avg_train_loss}'])
                avg_denom = max(1, train_size)
                self._bus.put([
                    'metrics',
                    {
                        'type': 'train_perf_epoch',
                        'epoch': epoch + 1,
                        'data_wait_ms': float(epoch_data_wait_ms / avg_denom),
                        'forward_ms': float(epoch_forward_ms / avg_denom),
                        'backward_ms': float(epoch_backward_ms / avg_denom),
                        'optimizer_ms': float(epoch_optimizer_ms / avg_denom),
                        'total_ms': float(epoch_total_ms / avg_denom),
                    },
                ])


            validation_result = None
            if is_main_process:
                validation_result = self._run_validation_epoch(device, criterion, autocast_ctx)
            if validation_result is not None:
                avg_val_loss = validation_result['loss']
                val_accuracy = validation_result['accuracy']
                val_iou = validation_result['iou']
                val_dice = validation_result['dice']
                val_f1 = validation_result['f1']
                self._bus.put([
                    'logging',
                    (
                        f'Эпоха [{epoch + 1}/{self._epochs}] '
                        f'Валидационная потеря: {avg_val_loss:.6f} | '
                        f'Валидационная точность: {val_accuracy:.4%} | '
                        f'IoU: {val_iou:.4%} | Dice: {val_dice:.4%} | F1: {val_f1:.4%}'
                    ),
                ])
                self._bus.put([
                    'metrics',
                    {
                        'type': 'val_epoch',
                        'epoch': epoch + 1,
                        'loss': float(avg_val_loss),
                        'accuracy': float(val_accuracy),
                        'iou': float(val_iou),
                        'dice': float(val_dice),
                        'f1': float(val_f1),
                    },
                ])
                if early_stopping_enabled:
                    improved = (
                        early_stopping_best_loss is None
                        or (early_stopping_best_loss - avg_val_loss) > early_stopping_min_delta
                    )
                    if improved:
                        early_stopping_best_loss = float(avg_val_loss)
                        early_stopping_bad_epochs = 0
                        early_stopping_best_epoch = int(epoch + 1)
                        if self._early_stopping_params.restore_best_weights:
                            early_stopping_best_model_state = {
                                key: value.detach().cpu().clone()
                                for key, value in self._base_model.state_dict().items()
                            }
                        self._bus.put([
                            'logging',
                            f'Early stopping: новый лучший результат на эпохе {epoch + 1} (val_loss={avg_val_loss:.6f}).',
                        ])
                    else:
                        early_stopping_bad_epochs += 1
                        self._bus.put([
                            'logging',
                            f'Early stopping: без улучшения {early_stopping_bad_epochs}/{early_stopping_patience}.',
                        ])

            if is_main_process:
                torch.save(self._base_model, self._save_path)
                self._save_checkpoint(
                    epoch + 1,
                    optimizer,
                    scaler,
                    scheduler,
                    early_stopping_best_loss=early_stopping_best_loss,
                    early_stopping_bad_epochs=early_stopping_bad_epochs,
                    early_stopping_best_epoch=early_stopping_best_epoch,
                    early_stopping_best_model_state=early_stopping_best_model_state,
                )

            should_stop = bool(
                early_stopping_enabled
                and early_stopping_bad_epochs > 0
                and early_stopping_bad_epochs >= early_stopping_patience
            )
            if distributed:
                stop_tensor = torch.tensor([1 if (should_stop and is_main_process) else 0], device=device, dtype=torch.int32)
                dist.broadcast(stop_tensor, src=0)
                should_stop = bool(int(stop_tensor.item()))

            if should_stop:
                if is_main_process:
                    self._bus.put([
                        'logging',
                        f'Early stopping сработал на эпохе {epoch + 1}. Лучший val_loss достигнут на эпохе {early_stopping_best_epoch}.',
                    ])
                    if self._early_stopping_params.restore_best_weights and early_stopping_best_model_state:
                        self._base_model.load_state_dict(early_stopping_best_model_state)
                        self._bus.put(['logging', 'Восстановлены лучшие веса модели по валидации.'])
                break

            if is_main_process:
                memory_payload = _collect_memory_metrics()
                if memory_payload is not None:
                    self._bus.put(['metrics', {'type': 'system_memory', **memory_payload, 'epoch': int(epoch + 1)}])

            if distributed:
                dist.barrier()

        if is_main_process:
            torch.save(self._base_model, self._save_path)
        if self.callback is not None and is_main_process:
            self.callback()
        if is_main_process:
            print("Model saved successfully!")


@dataclass(frozen=True)
class RecognitionRuntimePlan:
    threads: int
    devices: list[torch.device]
    gpu_count: int
    use_multiprocessing: bool
    cut_workers: int
    predict_workers: int
    sew_workers: int


class NeuralRecognizer(threading.Thread):

    def __init__(self, recognition_parameters:RecognitionParameters,
                 message_bus:AbstractMessageBus, callback:Callable[..., None]|None = None  ):
        super().__init__()
        self.number_of_threads:int = 1
        self.devices_list: list[torch.device] = []
        self._parameters = recognition_parameters
        self._bus = message_bus
        self.callback = callback
        img_files = self._parameters.source_files
        names = [img_file.name for img_file in img_files]
        result_folder = self._parameters.result_folder
        result_folder.mkdir(parents=True, exist_ok=True)
        self._bus.publish('logging', f'Изображений в очереди на распознавание: {len(names)}')
        self.colors = 1
        self.model:nn.Module|None = None
        self._thread_stop_event = threading.Event()
        self.stop_event = mp.Event()

    def run(self, multithreading=False):
        try:
            self.prepare_model()
            runtime_plan = self._build_runtime_plan(multithreading=multithreading)
            self.number_of_threads = runtime_plan.threads
            self.devices_list = runtime_plan.devices

            if runtime_plan.use_multiprocessing:
                self.run_multiprocessing(runtime_plan)
            else:
                if multithreading and not isinstance(self._parameters.model, (str, Path)):
                    self._bus.publish(
                        'logging',
                        'Многопроцессное распознавание доступно только при загрузке модели из файла. Переход в однопоточный режим.',
                    )
                self.run_one_thread()
        finally:
            self.model = None
            _release_torch_memory()
            if self.callback is not None:
                self.callback()

    def _build_runtime_plan(self, multithreading: bool) -> RecognitionRuntimePlan:
        source_count = len(self._parameters.source_files)
        cpu_threads = mp.cpu_count()
        threads = min(source_count, cpu_threads)
        devices, gpu_count = self._resolve_devices()

        can_use_multiprocessing_model = isinstance(self._parameters.model, (str, Path))
        enough_threads_for_pipeline = (threads - gpu_count) >= 3
        use_multiprocessing = bool(multithreading and can_use_multiprocessing_model and enough_threads_for_pipeline)

        predict_workers = max(1, len(devices))
        remaining_workers = max(2, threads - predict_workers)
        cut_workers = max(1, remaining_workers // 2)
        sew_workers = max(1, remaining_workers - cut_workers)

        return RecognitionRuntimePlan(
            threads=threads,
            devices=devices,
            gpu_count=gpu_count,
            use_multiprocessing=use_multiprocessing,
            cut_workers=cut_workers,
            predict_workers=predict_workers,
            sew_workers=sew_workers,
        )

    @staticmethod
    def _resolve_devices() -> tuple[list[torch.device], int]:
        if not torch.cuda.is_available():
            return [torch.device('cpu')], 0
        gpu_count = System.check_gpu_availability()
        if gpu_count <= 0:
            return [torch.device('cpu')], 0
        return [torch.device(f'cuda:{gpu}') for gpu in range(gpu_count)], gpu_count

    def prepare_model(self):
        loaded_model: str | Path | nn.Module = self._parameters.model
        if isinstance(loaded_model, (str, Path)):
            loaded_model = torch.load(loaded_model, weights_only=False)
        if not isinstance(loaded_model, nn.Module):
            raise TypeError('Recognition model must be a torch.nn.Module or a model path.')
        if not bool(getattr(sys, 'frozen', False)):
            compile_fn = getattr(torch, 'compile', None)
            if compile_fn is not None:
                try:
                    loaded_model = compile_fn(loaded_model, mode='max-autotune', dynamic=False)
                    self._bus.publish('logging', 'Распознавание: включен torch.compile (mode=max-autotune).')
                except Exception as error:
                    self._bus.publish('logging', f'Распознавание: torch.compile отключен (fallback): {error}')
        self.model = loaded_model
        self.colors = get_input_channels(loaded_model)

    def run_multiprocessing(self, runtime_plan: RecognitionRuntimePlan | None = None):
        if runtime_plan is None:
            runtime_plan = self._build_runtime_plan(multithreading=True)
        cut_queue = mp.Queue()
        predict_queue = mp.Queue(maxsize=max(4, len(self.devices_list) * 4))
        sew_queue = mp.Queue(maxsize=max(4, len(self.devices_list) * 4))
        sewed_queue = mp.Queue()

        for image in self._parameters.source_files:
            cut_queue.put(image)

        result_folder = self._parameters.result_folder
        batch_size = self._parameters.batch_size
        overlap = self._parameters.overlap
        len_frames = len(self._parameters.source_files)

        predict_workers_count = runtime_plan.predict_workers
        cut_workers_count = runtime_plan.cut_workers
        sew_workers_count = runtime_plan.sew_workers

        for _ in range(cut_workers_count):
            cut_queue.put(STOP_TOKEN)

        shape = (self.colors, self._parameters.part_size[0], self._parameters.part_size[1])
        self._bus.publish(
            'metrics',
            {'type': 'recognition_progress', 'current': 0, 'total': int(len_frames)},
        )
        self._bus.publish(
            'logging',
            f'План рабочих процессов: нарезка={cut_workers_count}, предсказание={predict_workers_count}, сборка={sew_workers_count}',
        )

        cut_processes: list[mp.Process] = []
        predict_processes: list[mp.Process] = []
        all_processes: list[mp.Process] = []

        self._bus.publish('logging', 'Запуск процессов нарезки изображений')
        for _ in range(cut_workers_count):
            process = mp.Process(
                target=cut_image_process,
                args=(cut_queue, predict_queue, shape, overlap, self.stop_event),
            )
            process.start()
            cut_processes.append(process)
            all_processes.append(process)

        self._bus.publish('logging', 'Запуск процессов предсказания')
        model_source = cast(str | Path, self._parameters.model)
        for worker_idx in range(predict_workers_count):
            device = self.devices_list[min(worker_idx, len(self.devices_list) - 1)]
            process = mp.Process(
                target=imgpredict,
                args=(predict_queue, sew_queue, model_source, device, batch_size, self.stop_event),
            )
            process.start()
            predict_processes.append(process)
            all_processes.append(process)

        self._bus.publish('logging', 'Запуск процессов сборки изображений')
        for _ in range(sew_workers_count):
            process = mp.Process(
                target=imgsew,
                args=(result_folder, sew_queue, sewed_queue, self.stop_event),
            )
            process.start()
            all_processes.append(process)

        current_time = time.perf_counter()
        now = datetime.datetime.now()
        sewed_in_general = 0
        predict_stopped = False
        sew_stopped = False

        try:
            while sewed_in_general < len_frames:
                if self.stop_event.is_set():
                    break

                if not predict_stopped and all(not proc.is_alive() for proc in cut_processes):
                    for _ in range(predict_workers_count):
                        predict_queue.put(STOP_TOKEN)
                    predict_stopped = True

                if predict_stopped and (not sew_stopped) and all(not proc.is_alive() for proc in predict_processes):
                    for _ in range(sew_workers_count):
                        sew_queue.put(STOP_TOKEN)
                    sew_stopped = True

                try:
                    sewed_queue.get(timeout=0.2)
                    now_new = datetime.datetime.now() - now
                    now_new = now_new - datetime.timedelta(microseconds=now_new.microseconds)
                    sewed_in_general += 1
                    time_for_frame = round(time.perf_counter() - current_time, 3)
                    self._bus.publish(
                        'logging',
                        f'Кадр: {sewed_in_general}/{len_frames}. Время на кадр: {time_for_frame} сек. Прошло: {now_new}',
                    )
                    self._bus.publish(
                        'metrics',
                        {'type': 'recognition_progress', 'current': int(sewed_in_general), 'total': int(len_frames)},
                    )
                    current_time = time.perf_counter()
                except Empty:
                    pass

                failed_process = next((proc for proc in all_processes if proc.exitcode not in (None, 0)), None)
                if failed_process is not None:
                    self._bus.publish(
                        'logging',
                        f'Завершение дочернего процесса с ошибкой. pid={failed_process.pid}, код={failed_process.exitcode}.',
                    )
                    self.stop_event.set()
                    break
        finally:
            if not predict_stopped:
                for _ in range(predict_workers_count):
                    predict_queue.put(STOP_TOKEN)
            if not sew_stopped:
                for _ in range(sew_workers_count):
                    sew_queue.put(STOP_TOKEN)

            for process in all_processes:
                process.join(timeout=5)
            for process in all_processes:
                if process.is_alive():
                    process.terminate()
                    process.join(timeout=5)

    def run_one_thread(self):
        model = self.model
        if model is None:
            raise RuntimeError('Model is not prepared before recognition start.')
        if not self.devices_list:
            self.devices_list = [torch.device("cpu")]
        device = self.devices_list[0]

        model.eval()
        model.to(device)

        self._bus.publish('logging', 'Запуск распознавания в однопоточном режиме')
        self._bus.publish(
            'metrics',
            {'type': 'recognition_progress', 'current': 0, 'total': int(len(self._parameters.source_files))},
        )
        memory_payload = _collect_memory_metrics()
        if memory_payload is not None:
            self._bus.publish('metrics', {'type': 'system_memory', **memory_payload})

        current_time = time.perf_counter()
        now = datetime.datetime.now()
        shape = (self.colors,self._parameters.part_size[0], self._parameters.part_size[1])

        for i,img in enumerate(self._parameters.source_files):
            if self._thread_stop_event.is_set():
                break
            cutted_image = cut_image_prepare(img, shape, self._parameters.overlap)
            predicted_image = gpu_predict(cutted_image, model, device, self._parameters.batch_size)
            sew(self._parameters.result_folder, predicted_image)

            now_new = datetime.datetime.now()
            now_new = now_new - now
            now_new = now_new - datetime.timedelta(microseconds=now_new.microseconds)
            time_for_frame = round(time.perf_counter() - current_time, 3)
            self._bus.publish(
                'logging',
                f'Кадр: {i + 1}/{len(self._parameters.source_files)}. Время на кадр: {time_for_frame} сек. Прошло: {now_new}',
            )
            self._bus.publish(
                'metrics',
                {
                    'type': 'recognition_progress',
                    'current': int(i + 1),
                    'total': int(len(self._parameters.source_files)),
                },
            )
            memory_payload = _collect_memory_metrics()
            if memory_payload is not None:
                self._bus.publish('metrics', {'type': 'system_memory', **memory_payload})
            current_time = time.perf_counter()

    def stop(self):
        self._thread_stop_event.set()
        self.stop_event.set()


class NeuralRecognitioner(NeuralRecognizer):
    """Backward-compatible alias for the legacy misspelled class name."""

def cut_image_process(cut_queue, cutted_queue, size, overlap, stop_event):
    while not stop_event.is_set():
        try:
            img_path = cut_queue.get(timeout=0.2)
        except Empty:
            continue
        if img_path == STOP_TOKEN:
            break
        images_output_parameters = cut_image_prepare(img_path, size, overlap)
        cutted_queue.put(images_output_parameters)


def cut_image_prepare(img_path:Path, segment_size:tuple[int,int,int], overlap:int):
    img_dict = {'baseim_size': None, 'segment_size': None, 'overlap': None,
                'cutted_image': None, 'name': img_path.name}

    Image.MAX_IMAGE_PIXELS = None

    channels = segment_size[0]
    with Image.open(img_path) as startimg:
        if not (startimg.mode == 'L') and channels == 1:  # if rgb image predicts as grayscale
            startimg = startimg.convert("L")
        img_dict['baseim_size'] = startimg.size
        # convert image to numpy array
        work_image = np.array(startimg).astype('float32')
    img_dict['segment_size'] = segment_size
    img_dict['overlap'] = overlap

    # shape[0] - height, shape[1] - width
    if channels == 1:
        work_image = np.reshape(work_image, (channels, work_image.shape[0], work_image.shape[1]))
    else:
        work_image = work_image.transpose(2, 0, 1)

    img_dict['cutted_image'] = cut_image(work_image, segment_size, overlap)

    return img_dict

def get_array_from_image(path, channels):
    Image.MAX_IMAGE_PIXELS = None
    with Image.open(path) as startimg:
        work_image = np.array(startimg).astype('float32')
    # shape[0] - height, shape[1] - width
    work_image = work_image.transpose(2, 0, 1)
    return work_image


def imgpredict(prediction_queue, predicted_queue, model_path, gpu, batch_size, stop_event: MpEvent):
    model = torch.load(model_path, weights_only=False)
    model.eval()
    model.to(gpu)
    if gpu.type == 'cuda':
        torch.backends.cudnn.benchmark = True

    while not stop_event.is_set():
        try:
            item = prediction_queue.get(timeout=0.2)
        except Empty:
            continue
        if item == STOP_TOKEN:
            break
        item = gpu_predict(item, model, gpu, batch_size)
        item.pop('cutted_image', None)
        predicted_queue.put(item)


def gpu_predict(img, model, device, batch_size):
    predicted_image = np.empty_like(img['cutted_image'])
    parts_in_image = len(img['cutted_image'])

    tensor_data = torch.from_numpy(img['cutted_image']).float()
    batches = create_batches(tensor_data, batch_size)
    use_amp = (device.type == 'cuda')
    with torch.inference_mode():
        for i, batch in enumerate(batches):
            batch = batch.to(device, non_blocking=use_amp)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                outputs = model(batch)
                outputs = torch.sigmoid(outputs)
            numpy_predictions = outputs.detach().cpu().numpy()
            if batch_size * (i + 1) < parts_in_image:
                predicted_image[batch_size * i: batch_size * (i + 1)] = numpy_predictions
            else:
                predicted_image[batch_size * i:] = numpy_predictions

    img['predicted_image'] = predicted_image


    return img


def create_batches(tensor_data, batch_size):
    """Split tensor_data into batches of size batch_size."""
    n_samples = len(tensor_data)
    for i in range(0, n_samples, batch_size):
        yield tensor_data[i:i + batch_size]


def imgsew(outputDir, sew_queue, sewed_queue, stop_event: MpEvent):
    while not stop_event.is_set():
        try:
            item = sew_queue.get(timeout=0.2)
        except Empty:
            continue
        if item == STOP_TOKEN:
            break
        sew(outputDir, item)
        sewed_queue.put(item['name'])


def sew_from_queue(output_dir, sew_queue, sewed_queue):
    item = sew_queue.get()
    sew(output_dir, item)
    sewed_queue.put(item['name'])


def sew(save_dir, item):
    name =  '.'.join(item['name'].split('.')[:-1]) + '.jpg'
    full_name = os.path.join(save_dir, name)
    big_result = cast(
        Image.Image,
        sew_image(
            base_image=item['baseim_size'],
            predictions=item['predicted_image'],
            overlap=item['overlap'],
        ),
    )
    # big_result = big_result.resize(item['baseim_size'])
    # big_result = img_crop_border(crop_border * 2, big_result)
    big_result.save(full_name)



if __name__ == '__main__':
    result_tesnor = get_array_from_image('D:/NN/PCB/nn_test/samples/img_0000.jpg', 3)
    for i,channel in enumerate('RGB'):
        resimg = Image.fromarray(result_tesnor[i].astype('uint8'), mode='L')
        save_color(resimg, channel, f'D:/NN/PCB/nn_test/samples/img_0000_{channel}.jpg')



