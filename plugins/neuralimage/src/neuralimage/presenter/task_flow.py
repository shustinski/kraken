from __future__ import annotations

import os
from dataclasses import replace

from neuralimage.application.dto import clone_main_window_state
from neuralimage.application.services import ActiveTaskMutationError, build_processing_start_error_message, build_workflow_parameters
from neuralimage.application.services.training_artifacts import build_training_artifact_dir
from neuralimage.infrastructure.config.state_store import WORKFLOW_SNAPSHOT_FILENAME, save_workflow_snapshot
from neuralimage.lib.data_interfaces import WorkMode


def on_start_requested(presenter) -> None:
    presenter._save_windows_to_qsettings()
    presenter._update_main_window_state()
    presenter._update_settings_window_state()
    validation_error = build_processing_start_error_message(
        presenter.main_window_state,
        presenter.settings_state,
    )
    if validation_error:
        presenter.view.show_warning.emit(validation_error)
        presenter.message_bus.publish('logging', validation_error.replace('\n', ' | '))
        return

    task = presenter._processing_session.enqueue_task(
        main_state=clone_main_window_state(presenter.main_window_state),
        settings_state=replace(presenter.settings_state),
    )
    presenter.message_bus.publish('logging', f'Задача #{task.task_id} добавлена в очередь.')
    presenter._refresh_queue_view(selected_task_id=task.task_id)
    presenter._start_next_task_if_possible()


def on_stop_requested(presenter) -> None:
    if presenter.neuaral_handler is None:
        return
    presenter.neuaral_handler.stop()
    active_task = presenter._processing_session.request_stop()
    if active_task is not None:
        presenter.message_bus.publish('logging', f'Остановлена активная задача #{active_task.task_id}.')


def remove_queue_row(presenter, row: int) -> None:
    try:
        task = presenter._processing_session.remove_task_by_index(row)
    except ActiveTaskMutationError as error:
        presenter.message_bus.publish('logging', f'Нельзя убрать активную задачу #{error.task_id}.')
        return
    if task is None:
        return
    presenter.message_bus.publish('logging', f'Задача #{task.task_id} удалена из очереди.')
    queue_size = len(presenter._processing_session.queue_snapshot())
    presenter._refresh_queue_view(selected_row=min(row, queue_size - 1))


def on_queue_properties_requested(presenter, row: int, *, dialog_cls) -> None:
    task = presenter._processing_session.get_task_by_index(row)
    snapshot = presenter._processing_session.queue_snapshot()
    if task is None or row < 0 or row >= len(snapshot):
        return

    dialog = dialog_cls(
        task_id=task.task_id,
        status=snapshot[row].status,
        paused=task.paused,
        main_window_state=task.main_window_state,
        settings_state=task.settings_state,
        parent=presenter.view,
    )
    dialog.restore_requested.connect(presenter._on_task_restore_requested)
    dialog.exec()


def on_task_restore_requested(presenter, main_state, settings_state) -> None:
    presenter._restore_task_state_to_ui(main_state, settings_state)


def on_queue_pause_toggle_requested(presenter) -> None:
    row = presenter.view.get_selected_queue_row()
    try:
        task = presenter._processing_session.toggle_pause_by_index(row)
    except ActiveTaskMutationError as error:
        presenter.message_bus.publish('logging', f'Нельзя поставить на паузу активную задачу #{error.task_id}.')
        return
    if task is None:
        return
    state = 'поставлена на паузу' if task.paused else 'снята с паузы'
    presenter.message_bus.publish('logging', f'Задача #{task.task_id} {state}.')
    presenter._refresh_queue_view(selected_task_id=task.task_id)
    presenter._start_next_task_if_possible()


def refresh_queue_view(presenter, *, selected_row: int = -1, selected_task_id: int | None = None) -> None:
    items: list[str] = []
    resolved_selected_row = selected_row
    status_map = {
        'queued': 'в очереди',
        'paused': 'на паузе',
        'running': 'выполняется',
    }
    for index, task in enumerate(presenter._processing_session.queue_snapshot()):
        status = status_map.get(task.status, task.status)
        items.append(f'#{task.task_id} | {task.work_mode} | {status}')
        if selected_task_id is not None and task.task_id == selected_task_id:
            resolved_selected_row = index
    presenter.view.set_task_queue_items(items, resolved_selected_row)


def start_next_task_if_possible(presenter) -> None:
    handler = presenter.neuaral_handler
    is_running = False
    if handler is not None:
        is_running_method = getattr(handler, 'isRunning', None)
        if callable(is_running_method):
            is_running = bool(is_running_method())

    decision = presenter._processing_session.next_task_to_start(worker_running=is_running)
    if decision.worker_busy:
        return
    next_task = decision.task
    if next_task is None:
        presenter.view.toggle_start_stop.emit(False)
        return

    presenter._refresh_queue_view(selected_task_id=next_task.task_id)
    presenter.view.toggle_start_stop.emit(True)
    presenter._start_task(next_task)


def start_task(
    presenter,
    task,
    *,
    handler_thread_cls,
    workflow_builder=build_workflow_parameters,
    artifact_dir_builder=build_training_artifact_dir,
    workflow_snapshot_saver=save_workflow_snapshot,
    workflow_snapshot_filename: str = WORKFLOW_SNAPSHOT_FILENAME,
) -> None:
    os.environ['NEURALIMAGE_TORCH_COMPILE'] = '1' if task.settings_state.torch_compile_enabled else '0'
    presenter.message_bus.publish(
        'logging',
        f'torch.compile {"enabled" if task.settings_state.torch_compile_enabled else "disabled"} by UI setting.',
    )
    work_mode, training_settings, recognition_parameters = workflow_builder(
        task.main_window_state,
        task.settings_state,
    )
    if work_mode is None:
        presenter.message_bus.publish('error', f'Задача #{task.task_id}: не удалось определить режим работы.')
        presenter._processing_session.drop_task(task.task_id)
        presenter.view.toggle_start_stop.emit(False)
        presenter._refresh_queue_view()
        presenter._start_next_task_if_possible()
        return

    if work_mode in (
        WorkMode.train_only,
        WorkMode.train_and_recognition,
        WorkMode.further_training,
    ):
        try:
            artifact_dir = artifact_dir_builder(
                task.main_window_state,
                task.settings_state,
                work_mode,
            )
            training_settings.artifact_dir = artifact_dir
            snapshot_path = workflow_snapshot_saver(
                task.main_window_state,
                task.settings_state,
                destination=artifact_dir / workflow_snapshot_filename,
                workflow_snapshot=(work_mode, training_settings, recognition_parameters),
            )
            presenter.message_bus.publish('logging', f'Артефакты запуска будут сохранены в {artifact_dir}.')
            presenter.message_bus.publish('logging', f'Параметры запуска сохранены в {snapshot_path}.')
        except OSError as error:
            presenter.message_bus.publish('error', f'Не удалось сохранить параметры запуска: {error}')

    presenter.neuaral_handler = handler_thread_cls(
        work_mode=work_mode,
        recognition_parameters=recognition_parameters,
        tranining_parameters=training_settings,
        message_bus=presenter.message_bus,
        callback=presenter._on_stop_requested,
    )
    presenter.neuaral_handler.ask.connect(presenter._thread_ask)
    presenter.neuaral_handler.finished.connect(presenter._on_task_finished)
    presenter.neuaral_handler.start()
    presenter.message_bus.publish('logging', f'Запущена задача #{task.task_id}.')


def on_task_finished(presenter) -> None:
    result = presenter._processing_session.complete_active_task()
    if result.task is not None:
        if result.stop_requested:
            presenter.message_bus.publish('logging', f'Задача #{result.task.task_id} остановлена.')
        else:
            presenter.message_bus.publish('logging', f'Задача #{result.task.task_id} завершена.')
    presenter.neuaral_handler = None
    presenter._refresh_queue_view()
    presenter._start_next_task_if_possible()
