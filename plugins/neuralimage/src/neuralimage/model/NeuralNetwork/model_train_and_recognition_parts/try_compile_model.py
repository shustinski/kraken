"""model_train_and_recognition: try compile model."""
from __future__ import annotations
import os
import sys
from typing import Any
import numpy as np
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

from neuralimage.model.NeuralNetwork.model_train_and_recognition_parts.combined_optimizer import (
    MuonOptimizerUnavailableError,
    TrainingInstabilityError,
)

from neuralimage.model.NeuralNetwork.model_train_and_recognition_parts.epoch_stats import _TrainLoopStrides

from neuralimage.model.NeuralNetwork.model_train_and_recognition_parts.recognizer_process import (
    _get_torch_compile_unavailable_reason,
    _resolve_torch_compile_mode,
)


class TryCompileModelMixin:
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

    def _uses_variable_patch_sizes(self) -> bool:
        batch_sampler = getattr(getattr(self, '_train_dataloader', None), 'batch_sampler', None)
        return bool(getattr(batch_sampler, 'variable_patch_sizes', False))

    def _try_compile_model(self, is_main_process: bool = True, device: torch.device | None = None) -> None:
        if self._uses_variable_patch_sizes():
            self._torch_compile_active = False
            if is_main_process:
                self._bus.put([
                    'logging',
                    'torch.compile disabled: random patch sizes require dynamic input geometry.',
                ])
            return
        if bool(getattr(sys, 'frozen', False)):
            self._torch_compile_active = False
            if is_main_process:
                self._bus.put(['logging', 'torch.compile отключен в сборке PyInstaller.'])
            return
        if isinstance(self._model, nn.DataParallel):
            self._torch_compile_active = False
            if is_main_process:
                self._bus.put(['logging', 'torch.compile skipped: nn.DataParallel model wrapper is active.'])
            return
        if not self._env_bool('NEURALIMAGE_TORCH_COMPILE', False):
            self._torch_compile_active = False
            if is_main_process:
                self._bus.put(['logging', 'torch.compile disabled by NEURALIMAGE_TORCH_COMPILE=0.'])
            return
        compile_fn = getattr(torch, 'compile', None)
        if compile_fn is None:
            self._torch_compile_active = False
            if is_main_process:
                self._bus.put(['logging', 'torch.compile недоступен в этой версии PyTorch.'])
            return
        target_device_type = device.type if device is not None else ('cuda' if torch.cuda.is_available() else 'cpu')
        compile_unavailable_reason = _get_torch_compile_unavailable_reason(target_device_type)
        if compile_unavailable_reason is not None:
            self._torch_compile_active = False
            if is_main_process:
                self._bus.put(['logging', f'torch.compile disabled: {compile_unavailable_reason}'])
            return
        try:
            self._uncompiled_model = self._model
            compile_mode, mode_reason = _resolve_torch_compile_mode(target_device_type, device)
            self._model = compile_fn(self._model, mode=compile_mode, dynamic=False)
            self._torch_compile_active = True
            if is_main_process:
                self._bus.put(['logging', f'torch.compile enabled (mode={compile_mode}, reason={mode_reason}).'])
        except Exception as error:
            self._torch_compile_active = False
            if is_main_process:
                self._bus.put(['logging', f'torch.compile отключен (fallback): {error}'])

    @staticmethod
    def _is_torch_compile_runtime_failure(error: Exception) -> bool:
        current: BaseException | None = error
        for _ in range(6):
            if current is None:
                break
            message = str(current)
            if type(current).__name__ == 'BackendCompilerFailed':
                return True
            if "backend='inductor'" in message:
                return True
            if 'triton_key' in message:
                return True
            current = current.__cause__ if current.__cause__ is not None else current.__context__
        return False

    def _disable_torch_compile_runtime_fallback(self, error: Exception) -> bool:
        if not self._torch_compile_active:
            return False
        if self._uncompiled_model is None:
            return False
        if isinstance(self._model, (DDP, nn.DataParallel)):
            return False
        if not self._is_torch_compile_runtime_failure(error):
            return False
        self._model = self._uncompiled_model
        self._torch_compile_active = False
        self._bus.put(['logging', f'torch.compile disabled at runtime (fallback): {error}'])
        return True

    def _forward_model(self, image: Any) -> Any:
        try:
            return self._model(image)
        except Exception as error:
            if self._disable_torch_compile_runtime_fallback(error):
                return self._model(image)
            raise

    @staticmethod
    def _create_grad_scaler(device_type: str, enabled: bool) -> Any:
        amp_module = getattr(torch, 'amp', None)
        grad_scaler_cls = getattr(amp_module, 'GradScaler', None) if amp_module is not None else None
        if grad_scaler_cls is not None:
            try:
                return grad_scaler_cls(device_type, enabled=enabled)
            except TypeError:
                try:
                    return grad_scaler_cls(enabled=enabled)
                except TypeError:
                    pass
        return torch.cuda.amp.GradScaler(enabled=enabled)

    def run(self):
        try:
            requested_multi_gpu_mode = self._multi_gpu_mode
            if self._should_use_ddp():
                self._run_ddp(world_size=torch.cuda.device_count())
                return

            if (
                requested_multi_gpu_mode == 'distributeddataparallel'
                and self._has_multi_gpu_cuda()
                and os.name == 'nt'
            ):
                gpu_count = int(torch.cuda.device_count())
                self._bus.put([
                    'logging',
                    f'DDP requested on Windows; using nn.DataParallel on {gpu_count} GPU as fallback.',
                ])
                self._model = nn.DataParallel(self._model)
                device = torch.device('cuda:0')
            elif self._should_use_data_parallel():
                gpu_count = int(torch.cuda.device_count())
                self._bus.put([
                    'logging',
                    f'Включен режим multi GPU: nn.DataParallel на {gpu_count} GPU.',
                ])
                self._model = nn.DataParallel(self._model)
                device = torch.device('cuda:0')
            else:
                if requested_multi_gpu_mode != 'off':
                    self._log_multi_gpu_unavailable(requested_multi_gpu_mode)
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self._model.to(device)
            self._run_impl(device=device, rank=0, world_size=1, distributed=False)
        except MuonOptimizerUnavailableError as error:
            self._bus.put(['error', str(error)])
            raise
        except TrainingInstabilityError as error:
            self._bus.put(['error', str(error)])
            raise
        except Exception as error:
            self._bus.put(['error', f'Критическая ошибка обучения: {error}'])
            raise

    @staticmethod
    def _build_train_loop_strides(train_size: int, log_update_frequency: int = 0) -> _TrainLoopStrides:
        if log_update_frequency > 0:
            metric_stride = int(log_update_frequency)
            preview_stride = 2*int(log_update_frequency)
            log_stride = int(log_update_frequency)
        else:
             metric_stride = 10
             preview_stride = 20
             log_stride = 10
        return _TrainLoopStrides(
            metric=metric_stride,
            progress=metric_stride,
            log=log_stride,
            preview=preview_stride,
        )
