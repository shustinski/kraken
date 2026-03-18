import os

from application.dto import MainWindowState, SettingsState
from lib.data_interfaces import WorkMode, normalize_validation_source, normalize_work_mode


def can_start_processing(state: MainWindowState, settings_state: SettingsState | None = None) -> bool:
    src_ok = bool(state.source_folder and os.path.isdir(state.source_folder))
    res_ok = bool(state.result_folder and os.path.isdir(state.result_folder))
    training_ok = bool(
        state.sample_folder
        and state.label_folder
        and os.path.isdir(state.sample_folder)
        and os.path.isdir(state.label_folder)
    )
    model_ok = bool(state.model_path and os.path.isfile(state.model_path))
    epochs_ok = bool(state.epochs)

    work_mode = normalize_work_mode(state.work_mode)
    if work_mode == WorkMode.train_and_recognition.value:
        basic_ok = epochs_ok and src_ok and res_ok and training_ok
    elif work_mode == WorkMode.recognition_only.value:
        basic_ok = epochs_ok and src_ok and res_ok and model_ok
    elif work_mode == WorkMode.further_training.value:
        basic_ok = epochs_ok and src_ok and res_ok and training_ok and model_ok
    elif work_mode == WorkMode.train_only.value:
        basic_ok = epochs_ok and training_ok
    else:
        basic_ok = False

    if not basic_ok or settings_state is None:
        return basic_ok

    training_mode = work_mode in {
        WorkMode.train_only.value,
        WorkMode.train_and_recognition.value,
        WorkMode.further_training.value,
    }
    if not training_mode or not bool(getattr(settings_state, 'use_validation', False)):
        return basic_ok

    validation_source = normalize_validation_source(getattr(settings_state, 'validation_source', 'split'))
    if validation_source != 'external':
        return basic_ok

    validation_image_ok = bool(
        getattr(settings_state, 'validation_image_folder', '')
        and os.path.isdir(str(getattr(settings_state, 'validation_image_folder', '')))
    )
    validation_label_ok = bool(
        getattr(settings_state, 'validation_label_folder', '')
        and os.path.isdir(str(getattr(settings_state, 'validation_label_folder', '')))
    )
    return basic_ok and validation_image_ok and validation_label_ok
