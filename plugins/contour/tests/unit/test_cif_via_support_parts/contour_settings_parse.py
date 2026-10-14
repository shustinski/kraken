"""test_cif_via_support: test contour settings parse."""
from __future__ import annotations
from contour.application.processing import ContourExtractionSettings


class ContourSettingsParseMixin:
    def test_contour_settings_parse_fixed_via_values_from_dict(self) -> None:
        settings = ContourExtractionSettings.from_dict(
            {
                "extraction_profile": "vias",
                "object_type": "via",
                "output_mode": "box",
                "via_size_mode": "fixed",
                "fixed_via_widths": "8, 10; 12",
                "fixed_via_heights": [8, "10", 12.0],
            }
        )

        self.assertEqual(settings.via_size_mode, "fixed")
        self.assertEqual(settings.fixed_via_widths, [8, 10, 12])
        self.assertEqual(settings.fixed_via_heights, [8, 10, 12])
