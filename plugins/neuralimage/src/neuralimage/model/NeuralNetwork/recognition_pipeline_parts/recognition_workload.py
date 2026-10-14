"""recognition_pipeline: recognition workload."""
from __future__ import annotations
import os
import multiprocessing as mp
import traceback
from dataclasses import dataclass
from pathlib import Path
from queue import Empty
from typing import Any
import torch
from neuralimage.active_learning.config import ActiveLearningConfig
from neuralimage.preprocessing.config import PreprocessingConfig
from neuralimage.uncertainty.config import UncertaintyConfig

from neuralimage.model.NeuralNetwork.recognition_pipeline_parts.module_state import _QUEUE_EMPTY

def _resolve_relative_frame_name(img_path: Path, source_root: Path | None) -> str:
    frame_name = img_path.name
    if source_root is None:
        return frame_name

    source_root_path = Path(source_root)
    try:
        return img_path.relative_to(source_root_path).as_posix()
    except ValueError:
        pass

    try:
        return img_path.resolve().relative_to(source_root_path.resolve()).as_posix()
    except Exception:
        pass

    # Windows path casing and mixed absolute/relative forms can break relative_to().
    try:
        rel_path = os.path.relpath(str(img_path), start=str(source_root_path))
    except Exception:
        return frame_name
    rel_path_normalized = rel_path.replace('\\', '/')
    if rel_path_normalized.startswith('../') or rel_path_normalized == '..':
        return frame_name
    return Path(rel_path_normalized).as_posix()

@dataclass(frozen=True)
class RecognitionWorkload:
    source_files: list[Path]
    result_folder: Path
    part_size: tuple[int, int]
    overlap: int
    batch_size: int
    colors: int
    jpeg_quality: int
    binarize_output: bool
    threshold: float | None
    postprocess_enabled: bool
    postprocess_kernel_size: int
    recognition_tta_enabled: bool
    confidence_tta_enabled: bool
    confidence_save_mode: str
    devices: list[torch.device]
    model_source: str | Path
    use_context_branch: bool = False
    use_cross_attention: bool = True
    context_crop_size: tuple[int, int] | None = None
    context_input_size: tuple[int, int] | None = None
    compression_factor: int = 1
    source_root: Path | None = None
    preprocessing: PreprocessingConfig | None = None
    active_learning: ActiveLearningConfig | None = None
    active_learning_metadata: dict[str, object] | None = None
    uncertainty: UncertaintyConfig | None = None

    @property
    def frame_count(self) -> int:
        return int(len(self.source_files))

    @property
    def segment_shape(self) -> tuple[int, int, int]:
        return (self.colors, self.part_size[0], self.part_size[1])

@dataclass(frozen=True)
class WorkerCounts:
    cut: int
    predict: int
    sew: int

@dataclass(frozen=True)
class RuntimeWorkerConfig:
    workload: RecognitionWorkload
    worker_counts: WorkerCounts
    stop_token: str

@dataclass
class PipelineQueues:
    cut: mp.Queue
    predict: mp.Queue
    sew: mp.Queue
    sewed: mp.Queue
    errors: mp.Queue

    def close(self) -> None:
        for queue in (self.cut, self.predict, self.sew, self.sewed, self.errors):
            try:
                queue.close()
            except Exception:
                pass

@dataclass
class WorkerGroups:
    cut: list[mp.Process]
    predict: list[mp.Process]
    sew: list[mp.Process]

    def all(self) -> list[mp.Process]:
        return [*self.cut, *self.predict, *self.sew]

class ProgressReporter:
    def __init__(self, publish: Publisher, total_frames: int):
        self._publish = publish
        self._total_frames = int(total_frames)

    def publish_started(self) -> None:
        self._publish(
            'metrics',
            {'type': 'recognition_progress', 'current': 0, 'total': int(self._total_frames)},
        )

    def publish_worker_plan(self, worker_counts: WorkerCounts) -> None:
        self._publish(
            'logging',
            (
                'Recognition worker plan: '
                f'cut={worker_counts.cut}, '
                f'predict={worker_counts.predict}, '
                f'sew={worker_counts.sew}'
            ),
        )

    def publish_frame(self, current_frame: int) -> None:
        self._publish(
            'metrics',
            {
                'type': 'recognition_progress',
                'current': int(current_frame),
                'total': int(self._total_frames),
            },
        )

def _try_get_queue_item(queue: Any, *, timeout: float) -> Any:
    """Poll a multiprocessing queue without surfacing Empty in debuggers."""
    if queue is None:
        return _QUEUE_EMPTY
    reader = getattr(queue, '_reader', None)
    if reader is not None:
        try:
            if not bool(reader.poll(timeout)):
                return _QUEUE_EMPTY
            # After poll() reports readiness, a blocking get() avoids get_nowait()
            # spuriously raising Empty in debugger-attached multiprocessing runs.
            return queue.get()
        except (Empty, EOFError, OSError, ValueError):
            return _QUEUE_EMPTY

    try:
        return queue.get(timeout=timeout)
    except Empty:
        return _QUEUE_EMPTY

def _report_worker_exception(error_queue: mp.Queue | None, worker_name: str) -> None:
    if error_queue is None:
        return
    try:
        error_queue.put(
            {
                'worker': str(worker_name),
                'pid': int(os.getpid()),
                'traceback': traceback.format_exc(),
            }
        )
    except Exception:
        pass
