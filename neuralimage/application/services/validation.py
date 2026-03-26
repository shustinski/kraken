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
    epochs_ok = bool(state.epochs)
    if not epochs_ok:
        blockers.append('Укажите число эпох больше нуля.')

    src_ok = _is_existing_dir(state.source_folder)
    res_ok = _is_existing_dir(state.result_folder)
    sample_ok = _is_existing_dir(state.sample_folder)
    label_ok = _is_existing_dir(state.label_folder)
    model_ok = _is_existing_file(state.model_path)

    if work_mode == WorkMode.train_and_recognition.value:
        if not src_ok:
            blockers.append('Выберите существующую папку с исходными изображениями.')
        if not res_ok:
            blockers.append('Выберите существующую папку для результатов.')
        if not sample_ok:
            blockers.append('Выберите существующую папку с обучающими изображениями.')
        if not label_ok:
            blockers.append('Выберите существующую папку с обучающими метками.')
    elif work_mode == WorkMode.recognition_only.value:
        if not src_ok:
            blockers.append('Выберите существующую папку с исходными изображениями.')
        if not res_ok:
            blockers.append('Выберите существующую папку для результатов.')
        if not model_ok:
            blockers.append('Выберите существующий файл модели.')
    elif work_mode == WorkMode.further_training.value:
        if not src_ok:
            blockers.append('Выберите существующую папку с исходными изображениями.')
        if not res_ok:
            blockers.append('Выберите существующую папку для результатов.')
        if not sample_ok:
            blockers.append('Выберите существующую папку с обучающими изображениями.')
        if not label_ok:
            blockers.append('Выберите существующую папку с обучающими метками.')
        if not model_ok:
            blockers.append('Выберите существующий файл модели.')
    elif work_mode == WorkMode.train_only.value:
        if not sample_ok:
            blockers.append('Выберите существующую папку с обучающими изображениями.')
        if not label_ok:
            blockers.append('Выберите существующую папку с обучающими метками.')
    else:
        blockers.append('Выберите корректный режим работы.')

    if settings_state is None:
        return blockers

    training_mode = work_mode in {
        WorkMode.train_only.value,
        WorkMode.train_and_recognition.value,
        WorkMode.further_training.value,
    }
    if not training_mode or not bool(getattr(settings_state, 'use_validation', False)):
        return blockers

    validation_source = normalize_validation_source(getattr(settings_state, 'validation_source', 'split'))
    if validation_source != 'external':
        return blockers

    validation_image_ok = _is_existing_dir(str(getattr(settings_state, 'validation_image_folder', '')))
    validation_label_ok = _is_existing_dir(str(getattr(settings_state, 'validation_label_folder', '')))
    if not validation_image_ok:
        blockers.append('Выберите существующую папку с валидационными изображениями.')
    if not validation_label_ok:
        blockers.append('Выберите существующую папку с валидационными метками.')
    return blockers


def build_processing_start_error_message(
    state: MainWindowState,
    settings_state: SettingsState | None = None,
) -> str:
    blockers = get_processing_start_blockers(state, settings_state)
    if not blockers:
        return ''
    return 'Запуск невозможен:\n- ' + '\n- '.join(blockers)


def can_start_processing(state: MainWindowState, settings_state: SettingsState | None = None) -> bool:
    return not get_processing_start_blockers(state, settings_state)
