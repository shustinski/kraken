from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from contour.adapters.qt.object_validity import qt_object_is_valid, safe_viewport


class QtObjectValidityTests(unittest.TestCase):
    def test_qt_object_is_valid_returns_false_for_none(self) -> None:
        self.assertFalse(qt_object_is_valid(None))

    @patch("contour.adapters.qt.object_validity.shiboken6")
    def test_qt_object_is_valid_uses_shiboken_when_available(self, shiboken6: MagicMock) -> None:
        widget = object()
        shiboken6.isValid.return_value = True
        self.assertTrue(qt_object_is_valid(widget))
        shiboken6.isValid.assert_called_once_with(widget)

    @patch("contour.adapters.qt.object_validity.shiboken6", None)
    def test_safe_viewport_returns_none_for_deleted_widget(self) -> None:
        widget = MagicMock()
        widget.viewport.side_effect = RuntimeError("wrapped C/C++ object has been deleted")
        self.assertIsNone(safe_viewport(widget))

    @patch("contour.adapters.qt.object_validity.shiboken6")
    def test_safe_viewport_returns_viewport_when_valid(self, shiboken6: MagicMock) -> None:
        widget = MagicMock()
        viewport = object()
        widget.viewport.return_value = viewport
        shiboken6.isValid.return_value = True
        self.assertIs(safe_viewport(widget), viewport)


if __name__ == "__main__":
    unittest.main()
