import pytest
from types import SimpleNamespace

from neuralimage.application.dto import MainWindowState, SettingsState
from neuralimage.application.services import ActiveTaskMutationError, ProcessingSession
from neuralimage.presenter.task_flow import on_metrics_message


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


def test_queue_keeps_epoch_batch_and_validation_progress_independently():
    session = ProcessingSession()
    task = session.enqueue_task(*_make_states('train_only'))
    session.next_task_to_start(worker_running=False)
    updates: list[tuple[str, int, int, int]] = []
    presenter = SimpleNamespace(
        _processing_session=session,
        view=SimpleNamespace(
            update_task_queue_item_training_progress=lambda task_id, **values: updates.append(
                (
                    'epoch'
                    if 'epoch_current' in values
                    else ('validation' if 'validation_current' in values else 'batch'),
                    task_id,
                    values.get(
                        'epoch_current',
                        values.get('batch_current', values.get('validation_current')),
                    ),
                    values.get(
                        'epoch_total',
                        values.get('batch_total', values.get('validation_total')),
                    ),
                )
            ),
        ),
    )

    on_metrics_message(presenter, {'type': 'train_epoch_progress', 'current': 2, 'total': 8})
    on_metrics_message(presenter, {'type': 'train_batch_progress', 'current': 40, 'total': 100})
    on_metrics_message(presenter, {'type': 'validation_progress', 'current': 3, 'total': 12})

    snapshot = session.queue_snapshot()[0]
    assert updates == [
        ('epoch', task.task_id, 2, 8),
        ('batch', task.task_id, 40, 100),
        ('validation', task.task_id, 3, 12),
    ]
    assert (snapshot.epoch_progress_current, snapshot.epoch_progress_total) == (2, 8)
    assert (snapshot.batch_progress_current, snapshot.batch_progress_total) == (40, 100)
    assert (snapshot.validation_progress_current, snapshot.validation_progress_total) == (3, 12)
    assert snapshot.progress_kind == 'validation_progress'


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


def test_pause_request_exposes_pausing_then_allows_next_task():
    session = ProcessingSession()
    first = session.enqueue_task(*_make_states('train_only'))
    second = session.enqueue_task(*_make_states('recognition_only'))
    session.next_task_to_start(worker_running=False)

    session.request_pause_active()
    assert session.queue_snapshot()[0].status == 'pausing'

    result = session.complete_active_task()
    next_task = session.next_task_to_start(worker_running=False).task

    assert result.paused is True
    assert first.status == 'paused'
    assert next_task is second


def test_stopped_task_has_distinct_status_and_restarts_in_place():
    session = ProcessingSession()
    task = session.enqueue_task(*_make_states('train_only'))
    session.next_task_to_start(worker_running=False)
    session.request_stop()

    result = session.complete_active_task()
    assert task.status == 'stopped'
    restarted = session.restart_task_by_index(0)

    assert result.task is task
    assert task.status == 'waiting'
    assert restarted is task
    assert restarted.task_id == 1
