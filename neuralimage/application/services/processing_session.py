from __future__ import annotations

from dataclasses import dataclass

from application.dto import MainWindowState, SettingsState
from .processing_queue import ActiveTaskMutationError, ProcessingTaskQueue, QueuedTask


TaskType = QueuedTask[MainWindowState, SettingsState]


@dataclass(frozen=True, slots=True)
class QueueTaskSnapshot:
    task_id: int
    work_mode: str
    status: str


@dataclass(frozen=True, slots=True)
class StartNextTaskDecision:
    task: TaskType | None
    worker_busy: bool = False


@dataclass(frozen=True, slots=True)
class CompleteActiveTaskResult:
    task: TaskType | None
    stop_requested: bool = False


class ProcessingSession:
    def __init__(self) -> None:
        self._queue: ProcessingTaskQueue[MainWindowState, SettingsState] = ProcessingTaskQueue()
        self._stop_requested = False

    @property
    def active_task(self) -> TaskType | None:
        return self._queue.active_task

    def enqueue_task(self, main_state: MainWindowState, settings_state: SettingsState) -> TaskType:
        return self._queue.enqueue(main_state, settings_state)

    def get_task_by_index(self, index: int) -> TaskType | None:
        return self._queue.task_by_index(index)

    def remove_task_by_index(self, index: int) -> TaskType | None:
        return self._queue.remove_by_index(index)

    def toggle_pause_by_index(self, index: int) -> TaskType | None:
        return self._queue.toggle_pause_by_index(index)

    def request_stop(self) -> TaskType | None:
        active_task = self.active_task
        if active_task is not None:
            self._stop_requested = True
        return active_task

    def next_task_to_start(self, *, worker_running: bool) -> StartNextTaskDecision:
        if worker_running:
            return StartNextTaskDecision(task=None, worker_busy=True)
        next_task = self._queue.activate_next_ready()
        if next_task is not None:
            self._stop_requested = False
        return StartNextTaskDecision(task=next_task, worker_busy=False)

    def drop_task(self, task_id: int) -> TaskType | None:
        active_task = self.active_task
        was_active = active_task is not None and active_task.task_id == task_id
        removed_task = self._queue.remove_task(task_id)
        if was_active:
            self._stop_requested = False
        return removed_task

    def complete_active_task(self) -> CompleteActiveTaskResult:
        completed_task = self._queue.complete_active()
        result = CompleteActiveTaskResult(task=completed_task, stop_requested=self._stop_requested)
        self._stop_requested = False
        return result

    def queue_snapshot(self) -> tuple[QueueTaskSnapshot, ...]:
        active_task = self.active_task
        active_task_id = active_task.task_id if active_task is not None else None
        items: list[QueueTaskSnapshot] = []
        for task in self._queue.tasks:
            status = 'queued'
            if task.paused:
                status = 'paused'
            if task.task_id == active_task_id:
                status = 'running'
            items.append(
                QueueTaskSnapshot(
                    task_id=task.task_id,
                    work_mode=str(task.main_window_state.work_mode or 'unknown'),
                    status=status,
                )
            )
        return tuple(items)


__all__ = [
    'ActiveTaskMutationError',
    'CompleteActiveTaskResult',
    'ProcessingSession',
    'QueueTaskSnapshot',
    'QueuedTask',
    'StartNextTaskDecision',
]
