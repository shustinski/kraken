from __future__ import annotations

from dataclasses import dataclass

from neuralimage.application.dto import MainWindowState, SettingsState
from .processing_queue import (
    TASK_FINISHED_ERROR,
    TASK_FINISHED_SUCCESS,
    TASK_PAUSED,
    TASK_RUNNING,
    TASK_WAITING,
    ActiveTaskMutationError,
    ProcessingTaskQueue,
    QueuedTask,
)


TaskType = QueuedTask[MainWindowState, SettingsState]


@dataclass(frozen=True, slots=True)
class QueueTaskSnapshot:
    task_id: int
    work_mode: str
    status: str
    display_name: str = ''
    error_message: str = ''
    progress_current: int = 0
    progress_total: int = 0
    owner_username: str = ''
    owner_display_name: str = ''


@dataclass(frozen=True, slots=True)
class StartNextTaskDecision:
    task: TaskType | None
    worker_busy: bool = False


@dataclass(frozen=True, slots=True)
class CompleteActiveTaskResult:
    task: TaskType | None
    stop_requested: bool = False
    paused: bool = False


class ProcessingSession:
    def __init__(self) -> None:
        self._queue: ProcessingTaskQueue[MainWindowState, SettingsState] = ProcessingTaskQueue()
        self._stop_requested = False
        self._pause_requested = False

    @property
    def active_task(self) -> TaskType | None:
        return self._queue.active_task

    def enqueue_task(
        self,
        main_state: MainWindowState,
        settings_state: SettingsState,
        *,
        owner_username: str = '',
        owner_display_name: str = '',
        display_name: str = '',
    ) -> TaskType:
        return self._queue.enqueue(
            main_state,
            settings_state,
            owner_username=owner_username,
            owner_display_name=owner_display_name,
            display_name=display_name,
        )

    def get_task_by_index(self, index: int) -> TaskType | None:
        return self._queue.task_by_index(index)

    def remove_task_by_index(self, index: int) -> TaskType | None:
        return self._queue.remove_by_index(index)

    def toggle_pause_by_index(self, index: int) -> TaskType | None:
        return self._queue.toggle_pause_by_index(index)

    def resume_task_by_index(self, index: int) -> TaskType | None:
        return self._queue.resume_by_index(index)

    def rename_task_by_index(self, index: int, display_name: str) -> TaskType | None:
        return self._queue.rename_by_index(index, display_name)

    def retry_task_by_index(self, index: int) -> TaskType | None:
        return self._queue.retry_task_by_index(index)

    def move_task_by_index(self, source_index: int, target_index: int) -> TaskType | None:
        return self._queue.move_by_index(source_index, target_index)

    def move_task_up_by_index(self, index: int) -> TaskType | None:
        return self._queue.move_up_by_index(index)

    def move_task_down_by_index(self, index: int) -> TaskType | None:
        return self._queue.move_down_by_index(index)

    def request_stop(self) -> TaskType | None:
        active_task = self.active_task
        if active_task is not None:
            self._stop_requested = True
        return active_task

    def request_pause_active(self) -> TaskType | None:
        active_task = self.active_task
        if active_task is not None:
            self._stop_requested = True
            self._pause_requested = True
        return active_task

    def mark_active_paused_after_stop(self) -> TaskType | None:
        paused_task = self._queue.pause_active()
        self._stop_requested = False
        self._pause_requested = False
        return paused_task

    def set_active_error(self, error_message: str) -> TaskType | None:
        return self._queue.set_active_error(error_message)

    def update_active_progress(self, current: int, total: int) -> TaskType | None:
        return self._queue.set_progress_for_active(current, total)

    def next_task_to_start(self, *, worker_running: bool) -> StartNextTaskDecision:
        if worker_running:
            return StartNextTaskDecision(task=None, worker_busy=True)
        next_task = self._queue.activate_next_ready()
        if next_task is not None:
            self._stop_requested = False
            self._pause_requested = False
        return StartNextTaskDecision(task=next_task, worker_busy=False)

    def drop_task(self, task_id: int) -> TaskType | None:
        active_task = self.active_task
        was_active = active_task is not None and active_task.task_id == task_id
        removed_task = self._queue.remove_task(task_id)
        if was_active:
            self._stop_requested = False
            self._pause_requested = False
        return removed_task

    def complete_active_task(self) -> CompleteActiveTaskResult:
        if self._pause_requested:
            paused_task = self.mark_active_paused_after_stop()
            return CompleteActiveTaskResult(task=paused_task, stop_requested=True, paused=True)
        active_task = self.active_task
        error_message = str(getattr(active_task, 'error_message', '') or '') if active_task is not None else ''
        success = not error_message and not self._stop_requested
        completed_task = self._queue.complete_active(success=success, error_message=error_message)
        result = CompleteActiveTaskResult(task=completed_task, stop_requested=self._stop_requested)
        self._stop_requested = False
        self._pause_requested = False
        return result

    def queue_snapshot(self) -> tuple[QueueTaskSnapshot, ...]:
        active_task = self.active_task
        active_task_id = active_task.task_id if active_task is not None else None
        items: list[QueueTaskSnapshot] = []
        for task in self._queue.tasks:
            status = str(task.status or TASK_WAITING)
            if task.paused and status not in (TASK_FINISHED_SUCCESS, TASK_FINISHED_ERROR):
                status = TASK_PAUSED
            if task.task_id == active_task_id:
                status = TASK_RUNNING
            items.append(
                QueueTaskSnapshot(
                    task_id=task.task_id,
                    work_mode=str(task.main_window_state.work_mode or 'unknown'),
                    status=status,
                    display_name=str(task.display_name or ''),
                    error_message=str(task.error_message or ''),
                    progress_current=int(task.progress.current),
                    progress_total=int(task.progress.total),
                    owner_username=str(task.owner_username or ''),
                    owner_display_name=str(task.owner_display_name or task.owner_username or ''),
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
