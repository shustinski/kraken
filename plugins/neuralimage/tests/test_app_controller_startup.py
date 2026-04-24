from neuralimage.controller.app_controller import (
    apply_backend_unavailable_ui_state,
    format_backend_unavailable_message,
)


class _FakeSignal:
    def __init__(self) -> None:
        self.events: list[str] = []

    def emit(self, value: str) -> None:
        self.events.append(value)


class _FakeButton:
    def __init__(self) -> None:
        self.tooltip = ''

    def setToolTip(self, value: str) -> None:
        self.tooltip = value


class _FakeWindow:
    def __init__(self) -> None:
        self._title = 'NeuralImage v5.9.0'
        self.btn_start = _FakeButton()
        self.log_message = _FakeSignal()
        self.show_warning = _FakeSignal()

    def windowTitle(self) -> str:
        return self._title

    def setWindowTitle(self, value: str) -> None:
        self._title = value


def test_format_backend_unavailable_message_explains_disabled_start_button():
    message = format_backend_unavailable_message(RuntimeError('DLL load failed'))

    assert "кнопка 'Запуск' недоступна" in message
    assert 'Microsoft Visual C++ Redistributable' in message
    assert 'DLL load failed' in message


def test_apply_backend_unavailable_ui_state_marks_window_and_emits_notice():
    window = _FakeWindow()
    message = 'Backend unavailable'

    apply_backend_unavailable_ui_state(window, message)

    assert window.windowTitle().endswith('[UI only]')
    assert window.btn_start.tooltip == message
    assert window.log_message.events == [f'Запуск недоступен: {message}']
    assert window.show_warning.events == [message]
