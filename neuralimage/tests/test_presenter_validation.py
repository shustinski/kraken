from view.window_dataclasses import MainWindowState
from presenter.validation import can_start_processing
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
