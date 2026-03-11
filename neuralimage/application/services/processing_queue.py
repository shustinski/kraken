from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, TypeVar


MainStateT = TypeVar('MainStateT')
SettingsStateT = TypeVar('SettingsStateT')


@dataclass(slots=True)
class QueuedTask(Generic[MainStateT, SettingsStateT]):
    task_id: int
    main_window_state: MainStateT
    settings_state: SettingsStateT
    paused: bool = False


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
    ) -> QueuedTask[MainStateT, SettingsStateT]:
        task = QueuedTask(
            task_id=self._next_task_id,
            main_window_state=main_window_state,
            settings_state=settings_state,
        )
        self._next_task_id += 1
        self._tasks.append(task)
        return task

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
        task.paused = not task.paused
        return task

    def activate_next_ready(self) -> QueuedTask[MainStateT, SettingsStateT] | None:
        next_task = next((task for task in self._tasks if not task.paused), None)
        self._active_task_id = next_task.task_id if next_task is not None else None
        return next_task

    def remove_task(self, task_id: int) -> QueuedTask[MainStateT, SettingsStateT] | None:
        for index, task in enumerate(self._tasks):
            if task.task_id != task_id:
                continue
            removed_task = self._tasks.pop(index)
            if removed_task.task_id == self._active_task_id:
                self._active_task_id = None
            return removed_task
        return None

    def complete_active(self) -> QueuedTask[MainStateT, SettingsStateT] | None:
        active_task = self.active_task
        if active_task is None:
            return None
        return self.remove_task(active_task.task_id)

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
