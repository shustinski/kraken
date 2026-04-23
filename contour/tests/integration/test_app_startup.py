"""Integration smoke test: full application bootstrap via :func:`assemble_application`."""

from __future__ import annotations

import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


class ApplicationBootstrapTests(unittest.TestCase):
    """Exercise the CLI bootstrap path end-to-end without a display."""

    def test_assemble_application_creates_view_and_presenter(self) -> None:
        from polygon_widget.application.bootstrap import assemble_application

        components = assemble_application(argv=["--no-qss"])
        try:
            self.assertIsNotNone(components.app)
            self.assertIsNotNone(components.view)
            self.assertIsNotNone(components.presenter)
            self.assertIsNotNone(components.model)
            self.assertTrue(hasattr(components.view, "widget"))
        finally:
            components.view.close()
            components.view.deleteLater()
            components.app.processEvents()


if __name__ == "__main__":
    unittest.main()
