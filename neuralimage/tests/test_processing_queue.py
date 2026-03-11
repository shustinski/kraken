import pytest

from application.services import ActiveTaskMutationError, ProcessingTaskQueue
from application.dto import MainWindowState, SettingsState


def _make_states(work_mode: str) -> tuple[MainWindowState, SettingsState]:
    return MainWindowState(work_mode=work_mode), SettingsState()


def test_enqueue_assigns_incrementing_task_ids():
    queue = ProcessingTaskQueue[MainWindowState, SettingsState]()

    first = queue.enqueue(*_make_states('train_only'))
    second = queue.enqueue(*_make_states('recognition_only'))

    assert first.task_id == 1
    assert second.task_id == 2
    assert len(queue.tasks) == 2


def test_remove_by_index_is_noop_for_invalid_row():
    queue = ProcessingTaskQueue[MainWindowState, SettingsState]()

    queue.enqueue(*_make_states('train_only'))

    assert queue.remove_by_index(-1) is None
    assert queue.remove_by_index(5) is None
    assert len(queue.tasks) == 1


def test_task_by_index_returns_task_or_none():
    queue = ProcessingTaskQueue[MainWindowState, SettingsState]()

    first = queue.enqueue(*_make_states('train_only'))

    assert queue.task_by_index(0) is first
    assert queue.task_by_index(-1) is None
    assert queue.task_by_index(3) is None


def test_remove_by_index_rejects_active_task():
    queue = ProcessingTaskQueue[MainWindowState, SettingsState]()

    active = queue.enqueue(*_make_states('train_only'))
    queue.enqueue(*_make_states('recognition_only'))
    queue.activate_next_ready()

    with pytest.raises(ActiveTaskMutationError) as exc:
        queue.remove_by_index(0)

    assert exc.value.task_id == active.task_id
    assert exc.value.action == 'remove'


def test_toggle_pause_by_index_switches_state_for_non_active_task():
    queue = ProcessingTaskQueue[MainWindowState, SettingsState]()

    queue.enqueue(*_make_states('train_only'))
    paused_task = queue.enqueue(*_make_states('recognition_only'))

    toggled = queue.toggle_pause_by_index(1)

    assert toggled is paused_task
    assert paused_task.paused is True

    toggled = queue.toggle_pause_by_index(1)

    assert toggled is paused_task
    assert paused_task.paused is False


def test_toggle_pause_by_index_rejects_active_task():
    queue = ProcessingTaskQueue[MainWindowState, SettingsState]()

    active = queue.enqueue(*_make_states('train_only'))
    queue.activate_next_ready()

    with pytest.raises(ActiveTaskMutationError) as exc:
        queue.toggle_pause_by_index(0)

    assert exc.value.task_id == active.task_id
    assert exc.value.action == 'pause'


def test_activate_next_ready_skips_paused_tasks():
    queue = ProcessingTaskQueue[MainWindowState, SettingsState]()

    first = queue.enqueue(*_make_states('train_only'))
    second = queue.enqueue(*_make_states('recognition_only'))
    first.paused = True

    active = queue.activate_next_ready()

    assert active is second
    assert queue.active_task is second


def test_complete_active_removes_task_and_clears_active_selection():
    queue = ProcessingTaskQueue[MainWindowState, SettingsState]()

    first = queue.enqueue(*_make_states('train_only'))
    second = queue.enqueue(*_make_states('recognition_only'))
    queue.activate_next_ready()

    completed = queue.complete_active()

    assert completed is first
    assert queue.active_task is None
    assert queue.tasks == (second,)


def test_remove_task_clears_active_when_active_task_removed():
    queue = ProcessingTaskQueue[MainWindowState, SettingsState]()

    first = queue.enqueue(*_make_states('train_only'))
    queue.enqueue(*_make_states('recognition_only'))
    queue.activate_next_ready()

    removed = queue.remove_task(first.task_id)

    assert removed is first
    assert queue.active_task is None
