"""processing_mixin: prompt transition vector save."""
from __future__ import annotations
from typing import Any
from contour.widget_parts._imports import (
    QCheckBox,
    QMessageBox,
    TransitionPromptChoice,
)


class PromptTransitionVectorSaveMixin:
    def _prompt_transition_vector_save_dialog(self: Any) -> TransitionPromptChoice:
        msg = QMessageBox(self)
        msg.setIcon(QMessageBox.Icon.Warning)
        msg.setWindowTitle(
            self._tr(
                "unsaved_vectors_dialog_title",
                "Несохранённые изменения" if self._ui_language == "ru" else "Unsaved changes",
            )
        )
        msg.setText(
            self._tr(
                "unsaved_vectors_dialog_text",
                "Сохранить изменения?" if self._ui_language == "ru" else "Save changes?",
            )
        )
        save_button = msg.addButton(
            "Сохранить" if self._ui_language == "ru" else "Save",
            QMessageBox.ButtonRole.AcceptRole,
        )
        discard_button = msg.addButton(
            "Не сохранять" if self._ui_language == "ru" else "Don't save",
            QMessageBox.ButtonRole.DestructiveRole,
        )
        cancel_button = msg.addButton(
            "Отмена" if self._ui_language == "ru" else "Cancel",
            QMessageBox.ButtonRole.RejectRole,
        )
        autosave_checkbox = QCheckBox(
            "Автосохранение при переходе к следующему кадру"
            if self._ui_language == "ru"
            else "Autosave on next frame"
        )
        autosave_checkbox.setChecked(False)
        msg.setCheckBox(autosave_checkbox)
        msg.setDefaultButton(save_button)
        result = msg.exec()
        clicked = msg.clickedButton()
        if clicked is None:
            if result == QMessageBox.StandardButton.Cancel:
                return TransitionPromptChoice.CANCEL
            if result == QMessageBox.StandardButton.Discard:
                return TransitionPromptChoice.DISCARD
            if result == QMessageBox.StandardButton.Save and autosave_checkbox.isChecked() and hasattr(
                self, "autosave_on_frame_transition_checkbox"
            ):
                self.autosave_on_frame_transition_checkbox.setChecked(True)
            return TransitionPromptChoice.SAVE
        if clicked is cancel_button:
            return TransitionPromptChoice.CANCEL
        if clicked is discard_button:
            return TransitionPromptChoice.DISCARD
        if clicked is save_button and autosave_checkbox.isChecked() and hasattr(
            self, "autosave_on_frame_transition_checkbox"
        ):
            self.autosave_on_frame_transition_checkbox.setChecked(True)
        return TransitionPromptChoice.SAVE
