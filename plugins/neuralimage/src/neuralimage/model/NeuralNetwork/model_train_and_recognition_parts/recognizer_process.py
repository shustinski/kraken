"""model_train_and_recognition: recognizer process."""
from __future__ import annotations
import os
import sys
import importlib.util
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast
import multiprocessing as mp
import threading
import torch
from neuralimage.lib.data_interfaces import (
    RecognitionParameters,
)
from neuralimage.lib.message_bus import AbstractMessageBus

from neuralimage.model.NeuralNetwork.model_train_and_recognition_parts.module_state import (
    _MAX_AUTOTUNE_MIN_SMS,
    _TORCH_COMPILE_MODES,
)

def _resolve_memory_root_process(process):
    parent = process.parent()
    if parent is None:
        return process
    try:
        current_cmdline = ' '.join(str(part) for part in process.cmdline()).lower()
    except Exception:
        current_cmdline = ''
    try:
        parent_cmdline = ' '.join(str(part) for part in parent.cmdline()).lower()
    except Exception:
        parent_cmdline = ''
    try:
        same_executable = Path(parent.exe()).resolve() == Path(process.exe()).resolve()
    except Exception:
        same_executable = False
    if same_executable and ('neuralimage' in current_cmdline or 'neuralimage' in parent_cmdline):
        return parent
    return process

def _is_module_available(module_name: str) -> bool:
    try:
        return importlib.util.find_spec(module_name) is not None
    except Exception:
        return False

def _get_torch_compile_unavailable_reason(target_device_type: str) -> str | None:
    if target_device_type != 'cuda':
        return None
    if not _is_module_available('triton'):
        return 'Triton is not installed for CUDA backend.'
    try:
        from torch.utils import _triton as torch_triton
    except Exception:
        torch_triton = None

    if torch_triton is not None:
        has_triton_package = getattr(torch_triton, 'has_triton_package', None)
        if callable(has_triton_package):
            try:
                if not bool(has_triton_package()):
                    return 'Triton package is incompatible with this PyTorch build.'
                return None
            except Exception as error:
                return f'Triton compatibility probe failed: {error}'

    try:
        from triton.compiler.compiler import triton_key
    except Exception as error:
        return f'Triton package is incompatible with this PyTorch build: {error}'
    if triton_key is None:
        return 'Triton package is incompatible with this PyTorch build.'
    return None

def _is_debugger_attached() -> bool:
    gettrace = getattr(sys, 'gettrace', None)
    if not callable(gettrace):
        return False
    try:
        return bool(gettrace())
    except Exception:
        return False

def _resolve_torch_compile_mode(
    target_device_type: str,
    device: torch.device | None = None,
) -> tuple[str, str]:
    raw_mode = str(os.getenv('NEURALIMAGE_TORCH_COMPILE_MODE', '')).strip().lower()
    if raw_mode in _TORCH_COMPILE_MODES:
        return raw_mode, 'env'

    if target_device_type != 'cuda' or not torch.cuda.is_available():
        return 'default', 'device'

    try:
        if device is not None and device.type == 'cuda' and device.index is not None:
            props = torch.cuda.get_device_properties(device.index)
        else:
            props = torch.cuda.get_device_properties(torch.cuda.current_device())
        sm_count = int(getattr(props, 'multi_processor_count', 0))
    except Exception:
        return 'default', 'fallback'

    if sm_count < _MAX_AUTOTUNE_MIN_SMS:
        return 'reduce-overhead', f'sm={sm_count}'
    return 'max-autotune', f'sm={sm_count}'

@dataclass(frozen=True)
class RecognitionRuntimePlan:
    threads: int
    devices: list[torch.device]
    gpu_count: int
    use_multiprocessing: bool
    cut_workers: int
    predict_workers: int
    sew_workers: int

class _QueueMessageBus:
    def __init__(self, queue: mp.Queue):
        self._queue = queue

    def publish(self, topic: str, payload: Any) -> None:
        self._queue.put([topic, payload])

class RecognizerProcess(mp.Process):
    def __init__(
        self,
        recognition_parameters: RecognitionParameters,
        message_bus: mp.Queue,
        stop_event: Any,
        multithreading: bool = False,
    ):
        super().__init__()
        self._recognition_parameters = recognition_parameters
        self._bus = message_bus
        self._stop_event = stop_event
        self._multithreading = bool(multithreading)

    def _start_stop_watcher(self, recognizer: NeuralRecognizer) -> threading.Thread:
        def _watch_stop_signal() -> None:
            self._stop_event.wait()
            recognizer.stop()

        watcher = threading.Thread(target=_watch_stop_signal, daemon=True)
        watcher.start()
        return watcher

    def run(self):
        recognizer = NeuralRecognizer(
            self._recognition_parameters,
            message_bus=cast(AbstractMessageBus, _QueueMessageBus(self._bus)),
            callback=None,
        )
        self._start_stop_watcher(recognizer)
        try:
            recognizer.run(multithreading=self._multithreading)
        except Exception as error:
            self._bus.put(['error', f'Recognition error: {error}'])
            raise
        finally:
            recognizer.stop()
