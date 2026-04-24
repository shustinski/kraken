from itertools import product
from pathlib import Path

from neuralimage.application.dto import MainWindowState, SettingsState
from neuralimage.application.services.workflow_mapper import build_workflow_parameters
from tests.helpers import make_test_dir


def test_workflow_mapper_applies_boolean_and_mode_combinations():
    source = make_test_dir("workflow_matrix_source")
    result = make_test_dir("workflow_matrix_result")
    sample = make_test_dir("workflow_matrix_sample")
    label = make_test_dir("workflow_matrix_label")
    model_path = source / "model.pth"

    combinations_checked = 0
    for (
        work_mode,
        color_mode,
        sample_cut_mode,
        crop_enabled,
        resize_enabled,
        additional_augmentation,
    ) in product(
        (
            "train_only",
            "train_and_recognition",
            "recognition_only",
            "further_training",
        ),
        ("RGB", "ЧБ"),
        ("online", "disk"),
        (False, True),
        (False, True),
        (False, True),
    ):
        main_state = MainWindowState(
            work_mode=work_mode,
            source_folder=str(source),
            result_folder=str(result),
            sample_folder=str(sample),
            label_folder=str(label),
            model_path=str(model_path),
            epochs=3,
        )
        settings_state = SettingsState(
            color_mode=color_mode,
            sample_cut_mode=sample_cut_mode,
            crop_enabled=crop_enabled,
            resize_enabled=resize_enabled,
            additional_augmentation=additional_augmentation,
            model="M 720k",
            sample_size=(64, 64),
        )

        resolved_mode, training, recognition = build_workflow_parameters(main_state, settings_state)

        assert resolved_mode is not None
        assert resolved_mode.value == work_mode
        assert training.cut_mode.value == sample_cut_mode
        assert training.prepare.enable_crop is crop_enabled
        assert training.prepare.enable_resize is resize_enabled
        assert training.generation.additional_augmentation is additional_augmentation

        expected_channels = 3 if color_mode == "RGB" else 1
        assert training.colors == expected_channels
        assert training.generation.channels == expected_channels

        if work_mode in ("train_only", "train_and_recognition"):
            assert recognition.model == settings_state.model
        else:
            assert recognition.model == Path(main_state.model_path)

        combinations_checked += 1

    assert combinations_checked == 128
