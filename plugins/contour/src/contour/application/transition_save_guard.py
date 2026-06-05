"""Pure helpers for navigation when polygon overlays may be unsaved."""

from __future__ import annotations

from enum import Enum


class TransitionPromptChoice(Enum):
    SAVE = "save"
    DISCARD = "discard"
    CANCEL = "cancel"


def navigation_allowed_after_autosave_attempt(*, dirty: bool, save_ok: bool) -> bool:
    """Autosave-on-transition path: proceed unless there are unsaved changes and save failed."""

    if not dirty:
        return True
    return save_ok


def navigation_allowed_after_prompt(*, dirty: bool, choice: TransitionPromptChoice, save_ok: bool | None) -> bool:
    """Interactive Save / Don't save / Cancel path (``save_ok`` used only when choice is SAVE)."""

    if not dirty:
        return True
    if choice == TransitionPromptChoice.CANCEL:
        return False
    if choice == TransitionPromptChoice.DISCARD:
        return True
    if choice == TransitionPromptChoice.SAVE:
        return bool(save_ok)
    return False


__all__ = ["TransitionPromptChoice", "navigation_allowed_after_autosave_attempt", "navigation_allowed_after_prompt"]
