from application.dto import MainWindowState, SettingsState
from application.services.validation import can_start_processing
from tests.helpers import make_test_dir


def test_can_start_processing_train_and_recognition_requires_dataset():
    tmp_path = make_test_dir("presenter_validation_train")
    state = MainWindowState(
        work_mode='train_and_recognition',
        source_folder=str(tmp_path),
        result_folder=str(tmp_path),
        sample_folder='',
        label_folder='',
        epochs=1,
    )
    assert can_start_processing(state) is False


def test_can_start_processing_recognition_requires_model():
    tmp_path = make_test_dir("presenter_validation_rec")
    model = tmp_path / 'm.pth'
    model.write_text('x', encoding='utf-8')
    state = MainWindowState(
        work_mode='recognition_only',
        source_folder=str(tmp_path),
        result_folder=str(tmp_path),
        model_path=str(model),
        epochs=1,
    )
    assert can_start_processing(state) is True


def test_can_start_processing_train_only_does_not_require_source_or_result():
    tmp_path = make_test_dir("presenter_validation_train_only")
    state = MainWindowState(
        work_mode='train_only',
        source_folder='',
        result_folder='',
        sample_folder=str(tmp_path),
        label_folder=str(tmp_path),
        epochs=1,
    )
    assert can_start_processing(state) is True


def test_can_start_processing_external_validation_requires_validation_paths():
    tmp_path = make_test_dir("presenter_validation_external")
    train_images = tmp_path / 'train_images'
    train_labels = tmp_path / 'train_labels'
    train_images.mkdir()
    train_labels.mkdir()

    state = MainWindowState(
        work_mode='train_only',
        sample_folder=str(train_images),
        label_folder=str(train_labels),
        epochs=1,
    )
    settings = SettingsState(
        use_validation=True,
        validation_source='external',
        validation_image_folder='',
        validation_label_folder='',
    )
    assert can_start_processing(state, settings) is False

    validation_images = tmp_path / 'validation_images'
    validation_labels = tmp_path / 'validation_labels'
    validation_images.mkdir()
    validation_labels.mkdir()
    settings.validation_image_folder = str(validation_images)
    settings.validation_label_folder = str(validation_labels)
    assert can_start_processing(state, settings) is True
