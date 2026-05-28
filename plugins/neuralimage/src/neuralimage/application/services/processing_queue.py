from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Generic, TypeVar


MainStateT = TypeVar('MainStateT')
SettingsStateT = TypeVar('SettingsStateT')

TASK_WAITING = 'waiting'
TASK_RUNNING = 'in_progress'
TASK_PAUSED = 'paused'
TASK_FINISHED_SUCCESS = 'finished_success'
TASK_FINISHED_ERROR = 'finished_error'
TERMINAL_TASK_STATUSES = frozenset({TASK_FINISHED_SUCCESS, TASK_FINISHED_ERROR})
RECOGNITION_WORK_MODES = frozenset({'train_and_recognition', 'recognition_only', 'further_training'})
TRAINING_NAME_FROM_SAMPLE_PARENT_MODES = frozenset({'train_only', 'continue_training'})


@dataclass(frozen=True, slots=True)
class TaskProgress:
    current: int = 0
    total: int = 0


def _default_display_name(main_window_state: MainStateT) -> str:
    work_mode = str(getattr(main_window_state, 'work_mode', '') or '')
    if work_mode in TRAINING_NAME_FROM_SAMPLE_PARENT_MODES:
        sample_folder = str(getattr(main_window_state, 'sample_folder', '') or '').strip()
        if sample_folder:
            parent_name = Path(sample_folder).parent.name
            if parent_name:
                return parent_name
    source_folder = str(getattr(main_window_state, 'source_folder', '') or '').strip()
    if source_folder:
        return Path(source_folder).name
    return ''


@dataclass(slots=True)
class QueuedTask(Generic[MainStateT, SettingsStateT]):
    task_id: int
    main_window_state: MainStateT
    settings_state: SettingsStateT
    owner_username: str = ''
    owner_display_name: str = ''
    paused: bool = False
    display_name: str = ''
    status: str = TASK_WAITING
    error_message: str = ''
    progress: TaskProgress = TaskProgress()

    @property
    def is_finished(self) -> bool:
        return self.status in TERMINAL_TASK_STATUSES

    @property
    def is_running(self) -> bool:
        return self.status == TASK_RUNNING


class ActiveTaskMutationError(RuntimeError):
    def __init__(self, task_id: int, action: str):
        self.task_id = int(task_id)
        self.action = str(action)
        super().__init__(f'Active task #{self.task_id} cannot be changed by "{self.action}".')


class ProcessingTaskQueue(Generic[MainStateT, SettingsStateT]):
    def __init__(self) -> None:
        self._tasks: list[QueuedTask[MainStateT, SettingsStateT]] = []
        self._next_task_id = 1
        self._active_task_id: int | None = None

    def __len__(self) -> int:
        return len(self._tasks)

    @property
    def tasks(self) -> tuple[QueuedTask[MainStateT, SettingsStateT], ...]:
        return tuple(self._tasks)

    @property
    def active_task(self) -> QueuedTask[MainStateT, SettingsStateT] | None:
        return self._find_by_id(self._active_task_id)

    def task_by_index(self, index: int) -> QueuedTask[MainStateT, SettingsStateT] | None:
        return self._get_by_index(index)

    def enqueue(
        self,
        main_window_state: MainStateT,
        settings_state: SettingsStateT,
        *,
        owner_username: str = '',
        owner_display_name: str = '',
        display_name: str = '',
    ) -> QueuedTask[MainStateT, SettingsStateT]:
        task = QueuedTask(
            task_id=self._next_task_id,
            main_window_state=main_window_state,
            settings_state=settings_state,
            owner_username=str(owner_username or ''),
            owner_display_name=str(owner_display_name or owner_username or ''),
            display_name=str(display_name or _default_display_name(main_window_state)),
        )
        self._next_task_id += 1
        self._tasks.append(task)
        return task

    def retry_task_by_index(self, index: int) -> QueuedTask[MainStateT, SettingsStateT] | None:
        task = self._get_by_index(index)
        if task is None or task.status != TASK_FINISHED_ERROR:
            return None
        retry_task = QueuedTask(
            task_id=self._next_task_id,
            main_window_state=task.main_window_state,
            settings_state=task.settings_state,
            owner_username=task.owner_username,
            owner_display_name=task.owner_display_name,
            display_name=task.display_name,
        )
        self._next_task_id += 1
        self._tasks.insert(index + 1, retry_task)
        return retry_task

    def remove_by_index(self, index: int) -> QueuedTask[MainStateT, SettingsStateT] | None:
        task = self._get_by_index(index)
        if task is None:
            return None
        self._ensure_not_active(task, action='remove')
        return self._tasks.pop(index)

    def toggle_pause_by_index(self, index: int) -> QueuedTask[MainStateT, SettingsStateT] | None:
        task = self._get_by_index(index)
        if task is None:
            return None
        self._ensure_not_active(task, action='pause')
        if task.is_finished:
            return task
        task.paused = not task.paused
        task.status = TASK_PAUSED if task.paused else TASK_WAITING
        return task

    def activate_next_ready(self) -> QueuedTask[MainStateT, SettingsStateT] | None:
        next_task = next(
            (
                task
                for task in self._tasks
                if not task.paused and task.status not in TERMINAL_TASK_STATUSES and task.status != TASK_RUNNING
            ),
            None,
        )
        self._active_task_id = next_task.task_id if next_task is not None else None
        if next_task is not None:
            next_task.status = TASK_RUNNING
            next_task.error_message = ''
            next_task.progress = TaskProgress()
        return next_task

    def pause_active(self) -> QueuedTask[MainStateT, SettingsStateT] | None:
        active_task = self.active_task
        if active_task is None:
            return None
        active_task.paused = True
        active_task.status = TASK_PAUSED
        self._active_task_id = None
        return active_task

    def resume_by_index(self, index: int) -> QueuedTask[MainStateT, SettingsStateT] | None:
        task = self._get_by_index(index)
        if task is None or task.is_finished or task.is_running:
            return task
        task.paused = False
        task.status = TASK_WAITING
        return task

    def remove_task(self, task_id: int) -> QueuedTask[MainStateT, SettingsStateT] | None:
        for index, task in enumerate(self._tasks):
            if task.task_id != task_id:
                continue
            removed_task = self._tasks.pop(index)
            if removed_task.task_id == self._active_task_id:
                self._active_task_id = None
            return removed_task
        return None

    def complete_active(self, *, success: bool = True, error_message: str = '') -> QueuedTask[MainStateT, SettingsStateT] | None:
        active_task = self.active_task
        if active_task is None:
            return None
        active_task.paused = False
        active_task.status = TASK_FINISHED_SUCCESS if success else TASK_FINISHED_ERROR
        active_task.error_message = '' if success else str(error_message or '')
        active_task.progress = TaskProgress(1, 1)
        self._active_task_id = None
        return active_task

    def rename_by_index(self, index: int, display_name: str) -> QueuedTask[MainStateT, SettingsStateT] | None:
        task = self._get_by_index(index)
        if task is None:
            return None
        task.display_name = str(display_name)
        return task

    def set_progress_for_active(self, current: int, total: int) -> QueuedTask[MainStateT, SettingsStateT] | None:
        active_task = self.active_task
        if active_task is None:
            return None
        active_task.progress = TaskProgress(max(0, int(current)), max(0, int(total)))
        return active_task

    def set_active_error(self, error_message: str) -> QueuedTask[MainStateT, SettingsStateT] | None:
        active_task = self.active_task
        if active_task is None:
            return None
        active_task.error_message = str(error_message or '')
        return active_task

    def move_by_index(self, source_index: int, target_index: int) -> QueuedTask[MainStateT, SettingsStateT] | None:
        task = self._get_by_index(source_index)
        if task is None:
            return None
        self._ensure_not_active(task, action='move')
        if task.is_finished:
            return None
        target_index = max(0, min(int(target_index), len(self._tasks) - 1))
        if source_index == target_index:
            return task
        removed = self._tasks.pop(source_index)
        self._tasks.insert(target_index, removed)
        return removed

    def move_up_by_index(self, index: int) -> QueuedTask[MainStateT, SettingsStateT] | None:
        return self.move_by_index(index, index - 1)

    def move_down_by_index(self, index: int) -> QueuedTask[MainStateT, SettingsStateT] | None:
        return self.move_by_index(index, index + 1)

    def is_active(self, task: QueuedTask[MainStateT, SettingsStateT]) -> bool:
        return task.task_id == self._active_task_id

    def _get_by_index(self, index: int) -> QueuedTask[MainStateT, SettingsStateT] | None:
        if index < 0 or index >= len(self._tasks):
            return None
        return self._tasks[index]

    def _find_by_id(self, task_id: int | None) -> QueuedTask[MainStateT, SettingsStateT] | None:
        if task_id is None:
            return None
        return next((task for task in self._tasks if task.task_id == task_id), None)

    def _ensure_not_active(self, task: QueuedTask[MainStateT, SettingsStateT], *, action: str) -> None:
        if self.is_active(task):
            raise ActiveTaskMutationError(task.task_id, action)
