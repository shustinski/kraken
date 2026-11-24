"""Pause/continue glue between a managed Kraken job and the Contour widget.

Every function is a no-op for standalone Contour: the widget only carries a
``_managed_batch_resume`` after the Kraken bridge attached one.
"""

from __future__ import annotations

import logging
from typing import Any

from .local_batch_resume import LocalBatchResumeError, ManagedBatchResume

_LOGGER = logging.getLogger(__name__)


def _ru(widget: Any) -> bool:
    return getattr(widget, "_ui_language", "ru") == "ru"


def _text(widget: Any, ru: str, en: str) -> str:
    return ru if _ru(widget) else en


def _log(widget: Any, message: str) -> None:
    append = getattr(widget, "_append_log", None)
    if callable(append):
        append(message)
    else:
        _LOGGER.info("%s", message)


def managed_settings_snapshot(widget: Any) -> dict[str, Any]:
    """Every widget input that changes the generated CIFs, in Contour's JSON format."""
    return {
        "format": "contour-recognition-settings",
        "version": 2,
        "pipeline": widget.get_pipeline(),
        "recognition_settings": widget._current_contour_settings().to_dict(),
        "save_options": widget._current_save_options().to_dict(),
    }


def _paused_message(widget: Any, resume: ManagedBatchResume) -> str:
    done = len(resume.completed())
    return _text(
        widget,
        f"На паузе: готово {done} из {resume.total} кадров, осталось {resume.total - done}.",
        f"Paused: {done} of {resume.total} frames done, {resume.total - done} remaining.",
    )


def attach_managed_resume(widget: Any, session: Any) -> ManagedBatchResume | None:
    """Open the job's local state, restore its settings and show it paused."""
    widget._managed_batch_resume = None
    widget._managed_batch_resume_error = None
    try:
        resume = ManagedBatchResume.open(session)
    except LocalBatchResumeError as exc:
        _LOGGER.error("%s", exc)
        widget._managed_batch_resume_error = str(exc)
        _log(
            widget,
            _text(
                widget,
                f"Не удалось восстановить локальное состояние задания: {exc}. Исходные файлы сохранены; "
                "пакетное распознавание заблокировано.",
                f"Local job state could not be restored: {exc}. Original files are kept; batch recognition is blocked.",
            ),
        )
        return None
    widget._managed_batch_resume = resume
    settings = resume.state.current_settings
    if settings is not None:
        from .application.processing import ContourExtractionSettings

        if isinstance(settings.get("pipeline"), dict):
            widget.set_pipeline(dict(settings["pipeline"]))
        if isinstance(settings.get("recognition_settings"), dict):
            widget._set_extraction_settings(ContourExtractionSettings.from_dict(dict(settings["recognition_settings"])))
        _log(
            widget,
            _text(
                widget,
                "Восстановлены настройки распознавания этого задания.",
                "Restored this job's recognition settings.",
            ),
        )
    if resume.completed():
        _log(widget, _paused_message(widget, resume))
        if resume.has_unknown():
            _log(
                widget,
                _text(
                    widget,
                    "Для части готовых CIF исходные настройки неизвестны: они созданы до сохранения снимков "
                    "настроек. Продолжение потребует подтверждения.",
                    "Original settings are unknown for some finished CIFs: they predate settings snapshots. "
                    "Continuing will ask for confirmation.",
                ),
            )
    refresh_managed_batch_labels(widget)
    return resume


def refresh_managed_batch_labels(widget: Any) -> None:
    resume = getattr(widget, "_managed_batch_resume", None)
    if resume is None or not hasattr(widget, "stop_batch_button"):
        return
    widget.stop_batch_button.setText(_text(widget, "Пауза", "Pause"))
    widget.stop_batch_button.setAccessibleName(_text(widget, "Пауза", "Pause"))
    done = len(resume.completed())
    if 0 < done < resume.total:
        widget.batch_button.setText(_text(widget, "Продолжить", "Continue"))
        widget.batch_button.setAccessibleName(_text(widget, "Продолжить", "Continue"))


def managed_batch_paths(widget: Any) -> list[str] | None:
    """Frames to process now, or ``None`` when nothing should start."""
    error = getattr(widget, "_managed_batch_resume_error", None)
    if error:
        _log(widget, _text(widget, f"Пакет заблокирован: {error}", f"Batch blocked: {error}"))
        return None
    resume: ManagedBatchResume = widget._managed_batch_resume
    remaining = resume.remaining_paths()
    if not remaining:
        _log(widget, _text(widget, "Все кадры задания уже распознаны.", "Every frame of this job is done."))
        return None
    settings = managed_settings_snapshot(widget)
    if resume.needs_confirmation(settings):
        question = _text(
            widget,
            "Уже готовые CIF созданы с другими или неизвестными параметрами. Если продолжить, старые и новые "
            "CIF будут иметь разные параметры обработки; для каждого кадра это будет записано в результат. "
            "Продолжить?",
            "Finished CIFs were made with different or unknown parameters. Continuing will mix processing "
            "parameters between old and new CIFs; each frame's parameters are recorded in the result. Continue?",
        )
        if not widget._confirm_processing_question(question):
            return None
    try:
        return resume.begin(settings)
    except OSError as exc:
        _log(
            widget,
            _text(
                widget, f"Не удалось сохранить снимок настроек: {exc}", f"Could not save the settings snapshot: {exc}"
            ),
        )
        return None


def managed_progress(widget: Any, current: int, total: int) -> tuple[int, int]:
    resume = getattr(widget, "_managed_batch_resume", None)
    if resume is None or not resume.running:
        return current, total
    return resume.overall_progress(current)


def on_managed_batch_finished(widget: Any) -> None:
    resume = getattr(widget, "_managed_batch_resume", None)
    if resume is None or not resume.running:
        return
    resume.finish()
    if resume.remaining_paths():
        _log(widget, _paused_message(widget, resume))
    refresh_managed_batch_labels(widget)


__all__ = [
    "attach_managed_resume",
    "managed_batch_paths",
    "managed_progress",
    "managed_settings_snapshot",
    "on_managed_batch_finished",
    "refresh_managed_batch_labels",
]
