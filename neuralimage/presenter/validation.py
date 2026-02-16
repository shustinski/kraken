import os

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

    ok = src_ok and res_ok and epochs_ok
    if state.work_mode == 'train_and_recognition':
        return ok and training_ok
    if state.work_mode == 'recognintion_only':
        return ok and model_ok
    if state.work_mode == 'futher_training':
        return ok and training_ok and model_ok
    return ok
