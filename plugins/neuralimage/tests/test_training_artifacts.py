from neuralimage.application.dto import MainWindowState, SettingsState
from neuralimage.application.services.training_artifacts import build_training_artifact_dir
from neuralimage.lib.data_interfaces import WorkMode
from tests.helpers import make_test_dir


def test_build_training_artifact_dir_uses_model_name_and_creates_unique_folder():
    root = make_test_dir('training_artifacts_root')
    sample_dir = root / 'samples'
    label_dir = root / 'labels'
    sample_dir.mkdir()
    label_dir.mkdir()

    main_state = MainWindowState(
        work_mode='train_only',
        sample_folder=str(sample_dir),
        label_folder=str(label_dir),
    )
    settings_state = SettingsState(model='Mock Model')

    first_dir = build_training_artifact_dir(main_state, settings_state, WorkMode.train_only)
    second_dir = build_training_artifact_dir(main_state, settings_state, WorkMode.train_only)

    assert first_dir.exists() is True
    assert second_dir.exists() is True
    assert first_dir.parent == root
    assert second_dir.parent == root
    assert first_dir != second_dir
    assert first_dir.name.startswith('Mock_Model_')
