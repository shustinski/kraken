"""Tests for transition / unsaved-vector guard helpers."""

from __future__ import annotations

import unittest

from contour.application.transition_save_guard import (
    TransitionPromptChoice,
    navigation_allowed_after_autosave_attempt,
    navigation_allowed_after_prompt,
)


class TransitionSaveGuardTests(unittest.TestCase):
    def test_clean_autosave_branch(self) -> None:
        self.assertTrue(navigation_allowed_after_autosave_attempt(dirty=False, save_ok=False))

    def test_dirty_autosave_save_ok(self) -> None:
        self.assertTrue(navigation_allowed_after_autosave_attempt(dirty=True, save_ok=True))

    def test_dirty_autosave_save_error_blocks(self) -> None:
        self.assertFalse(navigation_allowed_after_autosave_attempt(dirty=True, save_ok=False))

    def test_clean_prompt_branch(self) -> None:
        self.assertTrue(
            navigation_allowed_after_prompt(dirty=False, choice=TransitionPromptChoice.CANCEL, save_ok=False)
        )

    def test_dirty_save_choice_invokes_save_semantics_via_ok_flag(self) -> None:
        self.assertTrue(
            navigation_allowed_after_prompt(dirty=True, choice=TransitionPromptChoice.SAVE, save_ok=True)
        )
        self.assertFalse(
            navigation_allowed_after_prompt(dirty=True, choice=TransitionPromptChoice.SAVE, save_ok=False)
        )

    def test_dirty_discard_allows(self) -> None:
        self.assertTrue(
            navigation_allowed_after_prompt(dirty=True, choice=TransitionPromptChoice.DISCARD, save_ok=False)
        )

    def test_dirty_cancel_blocks(self) -> None:
        self.assertFalse(
            navigation_allowed_after_prompt(dirty=True, choice=TransitionPromptChoice.CANCEL, save_ok=True)
        )


if __name__ == "__main__":
    unittest.main()
