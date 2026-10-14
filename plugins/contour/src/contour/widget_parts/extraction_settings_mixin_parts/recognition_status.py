"""extraction_settings_mixin: recognition status."""
from __future__ import annotations


class RecognitionStatusMixin:
    def _set_recognition_status(self, kind: str, message: str | None = None) -> None:
        if not hasattr(self, "recognition_status_label"):
            return
        if kind == "disabled":
            self.recognition_status_label.clear()
            return
        if self._ui_language == "ru":
            texts = {
                "idle": "Готово",
                "disabled": "Извлечение отключено",
                "updating": "Выполняется обработка…",
                "error": "Ошибка",
            }
        else:
            texts = {
                "idle": "Ready",
                "disabled": "Recognition off",
                "updating": "Updating…",
                "error": "Error",
            }
        text = message or texts.get(kind, texts["idle"])
        self.recognition_status_label.setText(text)
