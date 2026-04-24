import os

from application.dto import MainWindowState, SettingsState
from lib.data_interfaces import WorkMode, normalize_validation_source, normalize_work_mode


def _is_existing_dir(path: str) -> bool:
    return bool(path and os.path.isdir(path))


def _is_existing_file(path: str) -> bool:
    return bool(path and os.path.isfile(path))


def get_processing_start_blockers(
    state: MainWindowState,
    settings_state: SettingsState | None = None,
) -> list[str]:
    blockers: list[str] = []
    work_mode = normalize_work_mode(state.work_mode)
    training_mode = work_mode in {
        WorkMode.train_only.value,
        WorkMode.train_and_recognition.value,
        WorkMode.further_training.value,
    }
    epochs_ok = bool(state.epochs)
    if training_mode and not epochs_ok:
        blockers.append('\u0423\u043a\u0430\u0436\u0438\u0442\u0435 \u0447\u0438\u0441\u043b\u043e \u044d\u043f\u043e\u0445 \u0431\u043e\u043b\u044c\u0448\u0435 \u043d\u0443\u043b\u044f.')

    src_ok = _is_existing_dir(state.source_folder)
    res_ok = _is_existing_dir(state.result_folder)
    sample_ok = _is_existing_dir(state.sample_folder)
    label_ok = _is_existing_dir(state.label_folder)
    model_ok = _is_existing_file(state.model_path)

    if work_mode == WorkMode.train_and_recognition.value:
        if not src_ok:
            blockers.append(
                '\u0412\u044b\u0431\u0435\u0440\u0438\u0442\u0435 \u0441\u0443\u0449\u0435\u0441\u0442\u0432\u0443\u044e\u0449\u0443\u044e '
                '\u043f\u0430\u043f\u043a\u0443 \u0441 \u0438\u0441\u0445\u043e\u0434\u043d\u044b\u043c\u0438 \u0438\u0437\u043e\u0431'
                '\u0440\u0430\u0436\u0435\u043d\u0438\u044f\u043c\u0438.'
            )
        if not res_ok:
            blockers.append(
                '\u0412\u044b\u0431\u0435\u0440\u0438\u0442\u0435 \u0441\u0443\u0449\u0435\u0441\u0442\u0432\u0443\u044e\u0449\u0443\u044e '
                '\u043f\u0430\u043f\u043a\u0443 \u0434\u043b\u044f \u0440\u0435\u0437\u0443\u043b\u044c\u0442\u0430\u0442\u043e\u0432.'
            )
        if not sample_ok:
            blockers.append(
                '\u0412\u044b\u0431\u0435\u0440\u0438\u0442\u0435 \u0441\u0443\u0449\u0435\u0441\u0442\u0432\u0443\u044e\u0449\u0443\u044e '
                '\u043f\u0430\u043f\u043a\u0443 \u0441 \u043e\u0431\u0443\u0447\u0430\u044e\u0449\u0438\u043c\u0438 \u0438\u0437\u043e'
                '\u0431\u0440\u0430\u0436\u0435\u043d\u0438\u044f\u043c\u0438.'
            )
        if not label_ok:
            blockers.append(
                '\u0412\u044b\u0431\u0435\u0440\u0438\u0442\u0435 \u0441\u0443\u0449\u0435\u0441\u0442\u0432\u0443\u044e\u0449\u0443\u044e '
                '\u043f\u0430\u043f\u043a\u0443 \u0441 \u043e\u0431\u0443\u0447\u0430\u044e\u0449\u0438\u043c\u0438 \u043c\u0435\u0442'
                '\u043a\u0430\u043c\u0438.'
            )
    elif work_mode == WorkMode.recognition_only.value:
        if not src_ok:
            blockers.append(
                '\u0412\u044b\u0431\u0435\u0440\u0438\u0442\u0435 \u0441\u0443\u0449\u0435\u0441\u0442\u0432\u0443\u044e\u0449\u0443\u044e '
                '\u043f\u0430\u043f\u043a\u0443 \u0441 \u0438\u0441\u0445\u043e\u0434\u043d\u044b\u043c\u0438 \u0438\u0437\u043e\u0431'
                '\u0440\u0430\u0436\u0435\u043d\u0438\u044f\u043c\u0438.'
            )
        if not res_ok:
            blockers.append(
                '\u0412\u044b\u0431\u0435\u0440\u0438\u0442\u0435 \u0441\u0443\u0449\u0435\u0441\u0442\u0432\u0443\u044e\u0449\u0443\u044e '
                '\u043f\u0430\u043f\u043a\u0443 \u0434\u043b\u044f \u0440\u0435\u0437\u0443\u043b\u044c\u0442\u0430\u0442\u043e\u0432.'
            )
        if not model_ok:
            blockers.append(
                '\u0412\u044b\u0431\u0435\u0440\u0438\u0442\u0435 \u0441\u0443\u0449\u0435\u0441\u0442\u0432\u0443\u044e\u0449\u0438\u0439 '
                '\u0444\u0430\u0439\u043b \u043c\u043e\u0434\u0435\u043b\u0438.'
            )
    elif work_mode == WorkMode.further_training.value:
        if not src_ok:
            blockers.append(
                '\u0412\u044b\u0431\u0435\u0440\u0438\u0442\u0435 \u0441\u0443\u0449\u0435\u0441\u0442\u0432\u0443\u044e\u0449\u0443\u044e '
                '\u043f\u0430\u043f\u043a\u0443 \u0441 \u0438\u0441\u0445\u043e\u0434\u043d\u044b\u043c\u0438 \u0438\u0437\u043e\u0431'
                '\u0440\u0430\u0436\u0435\u043d\u0438\u044f\u043c\u0438.'
            )
        if not res_ok:
            blockers.append(
                '\u0412\u044b\u0431\u0435\u0440\u0438\u0442\u0435 \u0441\u0443\u0449\u0435\u0441\u0442\u0432\u0443\u044e\u0449\u0443\u044e '
                '\u043f\u0430\u043f\u043a\u0443 \u0434\u043b\u044f \u0440\u0435\u0437\u0443\u043b\u044c\u0442\u0430\u0442\u043e\u0432.'
            )
        if not sample_ok:
            blockers.append(
                '\u0412\u044b\u0431\u0435\u0440\u0438\u0442\u0435 \u0441\u0443\u0449\u0435\u0441\u0442\u0432\u0443\u044e\u0449\u0443\u044e '
                '\u043f\u0430\u043f\u043a\u0443 \u0441 \u043e\u0431\u0443\u0447\u0430\u044e\u0449\u0438\u043c\u0438 \u0438\u0437\u043e'
                '\u0431\u0440\u0430\u0436\u0435\u043d\u0438\u044f\u043c\u0438.'
            )
        if not label_ok:
            blockers.append(
                '\u0412\u044b\u0431\u0435\u0440\u0438\u0442\u0435 \u0441\u0443\u0449\u0435\u0441\u0442\u0432\u0443\u044e\u0449\u0443\u044e '
                '\u043f\u0430\u043f\u043a\u0443 \u0441 \u043e\u0431\u0443\u0447\u0430\u044e\u0449\u0438\u043c\u0438 \u043c\u0435\u0442'
                '\u043a\u0430\u043c\u0438.'
            )
        if not model_ok:
            blockers.append(
                '\u0412\u044b\u0431\u0435\u0440\u0438\u0442\u0435 \u0441\u0443\u0449\u0435\u0441\u0442\u0432\u0443\u044e\u0449\u0438\u0439 '
                '\u0444\u0430\u0439\u043b \u043c\u043e\u0434\u0435\u043b\u0438.'
            )
    elif work_mode == WorkMode.train_only.value:
        if not sample_ok:
            blockers.append(
                '\u0412\u044b\u0431\u0435\u0440\u0438\u0442\u0435 \u0441\u0443\u0449\u0435\u0441\u0442\u0432\u0443\u044e\u0449\u0443\u044e '
                '\u043f\u0430\u043f\u043a\u0443 \u0441 \u043e\u0431\u0443\u0447\u0430\u044e\u0449\u0438\u043c\u0438 \u0438\u0437\u043e'
                '\u0431\u0440\u0430\u0436\u0435\u043d\u0438\u044f\u043c\u0438.'
            )
        if not label_ok:
            blockers.append(
                '\u0412\u044b\u0431\u0435\u0440\u0438\u0442\u0435 \u0441\u0443\u0449\u0435\u0441\u0442\u0432\u0443\u044e\u0449\u0443\u044e '
                '\u043f\u0430\u043f\u043a\u0443 \u0441 \u043e\u0431\u0443\u0447\u0430\u044e\u0449\u0438\u043c\u0438 \u043c\u0435\u0442'
                '\u043a\u0430\u043c\u0438.'
            )
    else:
        blockers.append(
            '\u0412\u044b\u0431\u0435\u0440\u0438\u0442\u0435 \u043a\u043e\u0440\u0440\u0435\u043a\u0442\u043d\u044b\u0439 '
            '\u0440\u0435\u0436\u0438\u043c \u0440\u0430\u0431\u043e\u0442\u044b.'
        )

    if settings_state is None:
        return blockers

    if not training_mode or not bool(getattr(settings_state, 'use_validation', False)):
        return blockers

    validation_source = normalize_validation_source(getattr(settings_state, 'validation_source', 'split'))
    if validation_source != 'external':
        return blockers

    validation_image_ok = _is_existing_dir(str(getattr(settings_state, 'validation_image_folder', '')))
    validation_label_ok = _is_existing_dir(str(getattr(settings_state, 'validation_label_folder', '')))
    if not validation_image_ok:
        blockers.append(
            '\u0412\u044b\u0431\u0435\u0440\u0438\u0442\u0435 \u0441\u0443\u0449\u0435\u0441\u0442\u0432\u0443\u044e\u0449\u0443\u044e '
            '\u043f\u0430\u043f\u043a\u0443 \u0441 \u0432\u0430\u043b\u0438\u0434\u0430\u0446\u0438\u043e\u043d\u043d\u044b\u043c\u0438 '
            '\u0438\u0437\u043e\u0431\u0440\u0430\u0436\u0435\u043d\u0438\u044f\u043c\u0438.'
        )
    if not validation_label_ok:
        blockers.append(
            '\u0412\u044b\u0431\u0435\u0440\u0438\u0442\u0435 \u0441\u0443\u0449\u0435\u0441\u0442\u0432\u0443\u044e\u0449\u0443\u044e '
            '\u043f\u0430\u043f\u043a\u0443 \u0441 \u0432\u0430\u043b\u0438\u0434\u0430\u0446\u0438\u043e\u043d\u043d\u044b\u043c\u0438 '
            '\u043c\u0435\u0442\u043a\u0430\u043c\u0438.'
        )
    return blockers


def build_processing_start_error_message(
    state: MainWindowState,
    settings_state: SettingsState | None = None,
) -> str:
    blockers = get_processing_start_blockers(state, settings_state)
    if not blockers:
        return ''
    return '\u0417\u0430\u043f\u0443\u0441\u043a \u043d\u0435\u0432\u043e\u0437\u043c\u043e\u0436\u0435\u043d:\n- ' + '\n- '.join(blockers)


def can_start_processing(state: MainWindowState, settings_state: SettingsState | None = None) -> bool:
    return not get_processing_start_blockers(state, settings_state)
