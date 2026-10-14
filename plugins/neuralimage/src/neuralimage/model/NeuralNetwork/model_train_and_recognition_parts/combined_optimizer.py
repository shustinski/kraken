"""model_train_and_recognition: combined optimizer."""
from __future__ import annotations
import datetime
from queue import Empty
from collections.abc import Callable, Mapping
from typing import Any
from torch import optim

from neuralimage.model.NeuralNetwork.model_train_and_recognition_parts.stop_process_descendants import _stop_process_descendants

class _CombinedOptimizer(optim.Optimizer):
    def __init__(self, optimizers: list[optim.Optimizer]):
        if not optimizers:
            raise ValueError('Combined optimizer requires at least one optimizer.')

        params = [
            param
            for optimizer in optimizers
            for group in optimizer.param_groups
            for param in group.get('params', [])
        ]
        super().__init__(params, {})
        self._optimizers = optimizers
        self.param_groups = [
            group
            for optimizer in optimizers
            for group in optimizer.param_groups
        ]

    def zero_grad(self, set_to_none: bool = True) -> None:
        for optimizer in self._optimizers:
            optimizer.zero_grad(set_to_none=set_to_none)

    def step(self, closure: Callable[[], float] | None = None):
        loss = None
        for index, optimizer in enumerate(self._optimizers):
            if index == 0 and closure is not None:
                loss = optimizer.step(closure)
            else:
                optimizer.step()
        return loss

    def state_dict(self) -> dict[str, Any]:
        return {
            'optimizer_state_dicts': [optimizer.state_dict() for optimizer in self._optimizers],
        }

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        optimizer_state_dicts = state_dict.get('optimizer_state_dicts')
        if not isinstance(optimizer_state_dicts, list):
            raise ValueError('Combined optimizer checkpoint is missing inner optimizer states.')
        for optimizer, optimizer_state_dict in zip(self._optimizers, optimizer_state_dicts, strict=False):
            optimizer.load_state_dict(optimizer_state_dict)
        self.param_groups = [
            group
            for optimizer in self._optimizers
            for group in optimizer.param_groups
        ]

class TrainingInstabilityError(RuntimeError):
    pass

class _TrainingPauseRequested(RuntimeError):
    def __init__(self, epoch: int, next_batch: int):
        self.epoch = int(epoch)
        self.next_batch = int(next_batch)
        super().__init__(f'Training paused at epoch {epoch}, batch {next_batch}.')

class MuonOptimizerUnavailableError(RuntimeError):
    pass

class _NoOpQueue:
    def put(self, item: Any) -> None:
        return

class _RecordingQueue:
    def __init__(self, queue: Any, log_lines: list[str]) -> None:
        self._queue = queue
        self._log_lines = log_lines

    def put(self, item: Any) -> None:
        try:
            if isinstance(item, (list, tuple)) and len(item) >= 2 and str(item[0]) == 'logging':
                timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                for line in str(item[1]).splitlines() or ['']:
                    self._log_lines.append(f'[{timestamp}] {line}')
        except Exception:
            pass
        self._queue.put(item)

    def log(self, message: str) -> None:
        self.put(['logging', message])

def _drain_process_queue(
    queue: Any,
    publish_message: Callable[[Any], None],
) -> None:
    if queue is None:
        return
    reader = getattr(queue, '_reader', None)
    poll_fn = getattr(reader, 'poll', None)
    get_fn = getattr(queue, 'get', None)
    if callable(poll_fn) and callable(get_fn):
        while True:
            try:
                if not bool(poll_fn(0)):
                    break
                queued_message = get_fn()
            except (Empty, EOFError, OSError, ValueError):
                break
            publish_message(queued_message)
        return
    get_nowait = getattr(queue, 'get_nowait', None)
    if callable(get_nowait):
        while True:
            try:
                queued_message = get_nowait()
            except Empty:
                break
            except (EOFError, OSError, ValueError):
                break
            publish_message(queued_message)
        return
    empty_fn = getattr(queue, 'empty', None)
    if not callable(empty_fn) or not callable(get_fn):
        return
    while True:
        try:
            if bool(empty_fn()):
                break
            queued_message = get_fn()
        except Empty:
            break
        except (EOFError, OSError, ValueError):
            break
        publish_message(queued_message)

def _join_process_with_escalation(process: Any, *, join_timeout: float = 5.0) -> None:
    if process is None:
        return
    try:
        process.join(timeout=join_timeout)
    except Exception:
        pass
    if not process.is_alive():
        return

    # A training process can own DataLoader/DDP workers.  Terminating only the
    # direct multiprocessing.Process on Windows reparents those workers and
    # leaves NeuralImage processes behind after the UI has closed.  Snapshot
    # and stop the complete tree before the direct child loses ownership of it.
    descendants = _get_process_descendants(getattr(process, 'pid', None))
    _stop_process_descendants(descendants, force=False, timeout=min(join_timeout, 1.0))
    try:
        process.terminate()
    except Exception:
        pass
    try:
        process.join(timeout=join_timeout)
    except Exception:
        pass
    if not process.is_alive():
        _stop_process_descendants(descendants, force=True, timeout=join_timeout)
        return
    descendants.extend(
        child for child in _get_process_descendants(getattr(process, 'pid', None))
        if child not in descendants
    )
    _stop_process_descendants(descendants, force=True, timeout=join_timeout)
    try:
        process.kill()
    except Exception:
        pass
    try:
        process.join(timeout=join_timeout)
    except Exception:
        pass

def _get_process_descendants(pid: Any) -> list[Any]:
    if pid is None:
        return []
    try:
        import psutil

        return list(psutil.Process(int(pid)).children(recursive=True))
    except Exception:
        return []
