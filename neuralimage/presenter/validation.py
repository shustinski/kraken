import os

from lib.data_interfaces import WorkMode, normalize_work_mode
from view.window_dataclasses import MainWindowState


def can_start_processing(state: MainWindowState) -> bool:
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
    if work_mode == WorkMode.train_only.value:
        return epochs_ok and training_ok
    if work_mode == WorkMode.train_and_recognition.value:
        return epochs_ok and src_ok and res_ok and training_ok
    if work_mode == WorkMode.recognition_only.value:
        return epochs_ok and src_ok and res_ok and model_ok
    if work_mode == WorkMode.further_training.value:
        return epochs_ok and src_ok and res_ok and training_ok and model_ok
    return False
