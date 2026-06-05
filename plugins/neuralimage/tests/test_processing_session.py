import pytest

from neuralimage.application.dto import MainWindowState, SettingsState
from neuralimage.application.services import ActiveTaskMutationError, ProcessingSession


def _make_states(work_mode: str) -> tuple[MainWindowState, SettingsState]:
    return MainWindowState(work_mode=work_mode), SettingsState()


def test_queue_snapshot_reflects_running_paused_and_queued_tasks():
    session = ProcessingSession()

    first = session.enqueue_task(*_make_states('train_only'), owner_username='alice', owner_display_name='Alice')
    second = session.enqueue_task(*_make_states('recognition_only'), owner_username='bob', owner_display_name='Bob')
    third = session.enqueue_task(*_make_states('further_training'))
    session.toggle_pause_by_index(1)

    decision = session.next_task_to_start(worker_running=False)

    assert decision.task is first
    assert [(item.task_id, item.status, item.owner_username, item.owner_display_name) for item in session.queue_snapshot()] == [
        (first.task_id, 'in_progress', 'alice', 'Alice'),
        (second.task_id, 'paused', 'bob', 'Bob'),
        (third.task_id, 'waiting', '', ''),
    ]


def test_next_task_to_start_reports_busy_worker_without_mutating_session():
    session = ProcessingSession()
    session.enqueue_task(*_make_states('train_only'))

    decision = session.next_task_to_start(worker_running=True)

    assert decision.worker_busy is True
    assert decision.task is None
    assert [item.status for item in session.queue_snapshot()] == ['waiting']


def test_request_stop_marks_completion_as_stopped():
    session = ProcessingSession()
    task = session.enqueue_task(*_make_states('train_only'))
    session.next_task_to_start(worker_running=False)

    active = session.request_stop()
    result = session.complete_active_task()

    assert active is task
    assert result.task is task
    assert result.stop_requested is True


def test_get_task_by_index_returns_enqueued_task():
    session = ProcessingSession()
    first = session.enqueue_task(*_make_states('train_only'))
    session.enqueue_task(*_make_states('recognition_only'))

    assert session.get_task_by_index(0) is first
    assert session.get_task_by_index(-1) is None
    assert session.get_task_by_index(8) is None


def test_drop_active_task_resets_stop_state_for_next_task():
    session = ProcessingSession()
    first = session.enqueue_task(*_make_states('train_only'))
    second = session.enqueue_task(*_make_states('recognition_only'))
    session.next_task_to_start(worker_running=False)
    session.request_stop()

    removed = session.drop_task(first.task_id)
    next_task = session.next_task_to_start(worker_running=False).task
    result = session.complete_active_task()

    assert removed is first
    assert next_task is second
    assert result.task is second
    assert result.stop_requested is False
    assert result.task.status == 'finished_success'


def test_remove_and_pause_still_reject_active_task_mutation():
    session = ProcessingSession()
    session.enqueue_task(*_make_states('train_only'))
    session.next_task_to_start(worker_running=False)

    with pytest.raises(ActiveTaskMutationError):
        session.remove_task_by_index(0)

    with pytest.raises(ActiveTaskMutationError):
        session.toggle_pause_by_index(0)


def test_error_completion_keeps_failed_task_and_next_task_runs():
    session = ProcessingSession()
    first = session.enqueue_task(*_make_states('train_only'))
    second = session.enqueue_task(*_make_states('recognition_only'))
    session.next_task_to_start(worker_running=False)

    session.set_active_error('broken')
    result = session.complete_active_task()
    next_task = session.next_task_to_start(worker_running=False).task

    assert result.task is first
    assert first.status == 'finished_error'
    assert first.error_message == 'broken'
    assert next_task is second


def test_pause_active_marks_task_paused_without_starting_next():
    session = ProcessingSession()
    first = session.enqueue_task(*_make_states('train_only'))
    session.enqueue_task(*_make_states('recognition_only'))
    session.next_task_to_start(worker_running=False)

    session.request_pause_active()
    result = session.complete_active_task()

    assert result.task is first
    assert result.paused is True
    assert first.status == 'paused'
    assert session.active_task is None
