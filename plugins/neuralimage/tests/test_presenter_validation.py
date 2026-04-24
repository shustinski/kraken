from neuralimage.application.dto import MainWindowState, SettingsState
from neuralimage.application.services.validation import (
    build_processing_start_error_message,
    can_start_processing,
    get_processing_start_blockers,
)
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


def test_get_processing_start_blockers_returns_specific_messages():
    state = MainWindowState(
        work_mode='recognition_only',
        source_folder='',
        result_folder='',
        model_path='',
        epochs=0,
    )

    blockers = get_processing_start_blockers(state)

    assert '\u0423\u043a\u0430\u0436\u0438\u0442\u0435 \u0447\u0438\u0441\u043b\u043e \u044d\u043f\u043e\u0445 \u0431\u043e\u043b\u044c\u0448\u0435 \u043d\u0443\u043b\u044f.' not in blockers
    assert (
        '\u0412\u044b\u0431\u0435\u0440\u0438\u0442\u0435 \u0441\u0443\u0449\u0435\u0441\u0442\u0432\u0443\u044e\u0449\u0443\u044e \u043f\u0430\u043f\u043a\u0443 '
        '\u0441 \u0438\u0441\u0445\u043e\u0434\u043d\u044b\u043c\u0438 \u0438\u0437\u043e\u0431\u0440\u0430\u0436\u0435\u043d\u0438\u044f\u043c\u0438.'
    ) in blockers
    assert (
        '\u0412\u044b\u0431\u0435\u0440\u0438\u0442\u0435 \u0441\u0443\u0449\u0435\u0441\u0442\u0432\u0443\u044e\u0449\u0443\u044e \u043f\u0430\u043f\u043a\u0443 '
        '\u0434\u043b\u044f \u0440\u0435\u0437\u0443\u043b\u044c\u0442\u0430\u0442\u043e\u0432.'
    ) in blockers
    assert (
        '\u0412\u044b\u0431\u0435\u0440\u0438\u0442\u0435 \u0441\u0443\u0449\u0435\u0441\u0442\u0432\u0443\u044e\u0449\u0438\u0439 \u0444\u0430\u0439\u043b '
        '\u043c\u043e\u0434\u0435\u043b\u0438.'
    ) in blockers


def test_can_start_processing_recognition_only_ignores_empty_epochs():
    tmp_path = make_test_dir("presenter_validation_rec_epochs")
    model = tmp_path / 'm.pth'
    model.write_text('x', encoding='utf-8')
    state = MainWindowState(
        work_mode='recognition_only',
        source_folder=str(tmp_path),
        result_folder=str(tmp_path),
        model_path=str(model),
        epochs=0,
    )

    assert can_start_processing(state) is True


def test_build_processing_start_error_message_formats_blockers_for_dialog():
    state = MainWindowState(work_mode='', epochs=0)

    message = build_processing_start_error_message(state)

    assert message.startswith('\u0417\u0430\u043f\u0443\u0441\u043a \u043d\u0435\u0432\u043e\u0437\u043c\u043e\u0436\u0435\u043d:\n- ')
    assert '\u0412\u044b\u0431\u0435\u0440\u0438\u0442\u0435 \u043a\u043e\u0440\u0440\u0435\u043a\u0442\u043d\u044b\u0439 \u0440\u0435\u0436\u0438\u043c \u0440\u0430\u0431\u043e\u0442\u044b.' in message
