from __future__ import annotations

import unittest
from pathlib import Path

from polygon_widget.application.styles import _rewrite_relative_urls


class StylesheetTests(unittest.TestCase):
    def test_rewrite_relative_urls_uses_absolute_filesystem_paths(self) -> None:
        base_dir = Path(__file__).resolve().parents[2] / "polygon_widget" / "resources" / "styles"

        rewritten = _rewrite_relative_urls(
            'QCheckBox::indicator:checked { image: url(icons/check_light.svg); }',
            base_dir,
        )

        expected_path = (base_dir / "icons" / "check_light.svg").resolve().as_posix()
        self.assertIn(f'url("{expected_path}")', rewritten)
        self.assertNotIn("file:/", rewritten)


if __name__ == "__main__":
    unittest.main()
