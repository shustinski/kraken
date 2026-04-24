from __future__ import annotations

import unittest

from polygon_widget.domain import compute_polygon_metrics


class GeometryTests(unittest.TestCase):
    def test_compute_polygon_metrics_for_rectangle(self) -> None:
        area, perimeter, bbox = compute_polygon_metrics(
            [
                (10.0, 5.0),
                (18.0, 5.0),
                (18.0, 11.0),
                (10.0, 11.0),
            ]
        )

        self.assertEqual(area, 48.0)
        self.assertEqual(perimeter, 28.0)
        self.assertEqual(bbox, (10, 5, 9, 7))

    def test_compute_polygon_metrics_for_segment(self) -> None:
        area, perimeter, bbox = compute_polygon_metrics([(1.2, 3.4), (4.8, 3.4)])

        self.assertEqual(area, 0.0)
        self.assertAlmostEqual(perimeter, 7.2)
        self.assertEqual(bbox, (1, 3, 4, 1))


if __name__ == "__main__":
    unittest.main()
