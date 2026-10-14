"""test_vector_geometry_postprocess: test repair keyhole spikes."""
from __future__ import annotations
from contour.application.vector_geometry_postprocess import (
    polygon_description_is_invalid,
    repair_invalid_polygon_descriptions,
)
from contour.domain import PolygonData, compute_polygon_metrics

from test_vector_geometry_postprocess_parts.rect import _rect


class RepairKeyholeSpikesMixin:
    def test_repair_invalid_polygon_descriptions_keeps_other_polygons(self) -> None:
        keepers = [
            _rect(0.0, 0.0, 40.0, 40.0, 1),
            _rect(100.0, 0.0, 140.0, 40.0, 2),
            _rect(200.0, 0.0, 240.0, 40.0, 3),
        ]
        keyhole = PolygonData(
            id=10,
            points=[
                (0.0, 100.0),
                (80.0, 100.0),
                (80.0, 180.0),
                (0.0, 180.0),
                (0.0, 140.0),
                (40.0, 140.0),
                (40.0, 150.0),
                (20.0, 150.0),
                (20.0, 130.0),
                (40.0, 130.0),
                (40.0, 140.0),
                (0.0, 140.0),
            ],
        )
        self.assertFalse(polygon_description_is_invalid(keyhole))
        self.assertEqual(keyhole.description_invalid_reason(), "repeated_vertex")

        repaired = repair_invalid_polygon_descriptions([*keepers, keyhole])

        self.assertEqual(len({polygon.id for polygon in repaired}), len(repaired))
        for keeper in keepers:
            match = next(polygon for polygon in repaired if polygon.points == keeper.points)
            self.assertEqual(match.id, keeper.id)
            self.assertFalse(match.is_hole)
        self.assertFalse(any(polygon_description_is_invalid(polygon) for polygon in repaired))
        self.assertTrue(any(polygon.is_hole for polygon in repaired))
        self.assertGreaterEqual(len(repaired), len(keepers) + 2)

    def test_repair_keyhole_with_spikes_keeps_authored_geometry(self) -> None:
        """0560-style: repair must split keyholes, not reshape via shapely dissolve."""

        # Incomplete multi-keyhole bar with A->B->A spikes on two hole rims.
        points = [
            (1958.0, 1805.0),
            (1937.0, 1805.0),
            (1931.0, 1799.0),
            (1825.0, 1800.0),
            (1820.0, 1805.0),
            (1823.0, 1811.0),
            (1828.0, 1813.0),
            (1862.0, 1814.0),
            (1915.0, 1812.0),
            (1928.0, 1813.0),
            (1933.0, 1812.0),
            (1937.0, 1807.0),
            (1937.0, 1805.0),
            (1958.0, 1805.0),
            (1799.0, 1805.0),
            (1795.0, 1801.0),
            (1790.0, 1799.0),
            (1689.0, 1800.0),
            (1683.0, 1806.0),
            (1684.0, 1809.0),
            (1689.0, 1813.0),
            (1794.0, 1813.0),
            (1799.0, 1809.0),
            (1799.0, 1805.0),
            (1958.0, 1805.0),
            (1958.0, 1804.0),
            (1525.0, 1804.0),
            (1519.0, 1800.0),
            (1416.0, 1800.0),
            (1413.0, 1801.0),
            (1408.0, 1807.0),
            (1414.0, 1813.0),
            (1511.0, 1813.0),
            (1512.0, 1827.0),
            (1511.0, 1813.0),
            (1521.0, 1813.0),
            (1525.0, 1809.0),
            (1525.0, 1804.0),
            (1958.0, 1804.0),
            (1963.0, 1800.0),
            (1998.0, 1800.0),
            (1998.0, 1775.0),
            (1990.0, 1774.0),
            (1.0, 1776.0),
            (1.0, 1794.0),
            (1011.0, 1793.0),
            (1016.0, 1794.0),
            (1026.0, 1800.0),
            (1052.0, 1801.0),
            (1066.0, 1800.0),
            (1077.0, 1794.0),
            (1201.0, 1793.0),
            (1216.0, 1800.0),
            (1232.0, 1799.0),
            (1385.0, 1801.0),
            (1389.0, 1806.0),
            (1382.0, 1813.0),
            (1.0, 1813.0),
            (1.0, 1828.0),
            (1790.0, 1828.0),
            (1937.0, 1826.0),
            (1998.0, 1827.0),
            (1998.0, 1813.0),
            (1963.0, 1813.0),
            (1959.0, 1809.0),
            (1662.0, 1809.0),
            (1660.0, 1802.0),
            (1655.0, 1799.0),
            (1552.0, 1800.0),
            (1546.0, 1805.0),
            (1546.0, 1807.0),
            (1552.0, 1813.0),
            (1589.0, 1813.0),
            (1589.0, 1827.0),
            (1589.0, 1813.0),
            (1657.0, 1813.0),
            (1662.0, 1809.0),
            (1959.0, 1809.0),
            (1957.0, 1807.0),
            (1958.0, 1805.0),
        ]
        area, perimeter, bbox = compute_polygon_metrics(points)
        keyhole = PolygonData(id=1, points=points, area=area, perimeter=perimeter, bbox=bbox)
        self.assertEqual(keyhole.description_invalid_reason(), "repeated_vertex")

        repaired = repair_invalid_polygon_descriptions([keyhole])
        outers = [polygon for polygon in repaired if not polygon.is_hole]
        holes = [polygon for polygon in repaired if polygon.is_hole]

        self.assertFalse(any(polygon_description_is_invalid(polygon) for polygon in repaired))
        self.assertEqual(len(outers), 1)
        self.assertEqual(len(holes), 4)
        # Shapely dissolve used to spawn a tiny floating outer (~576) and merge holes.
        self.assertFalse(any(polygon.area < 1000.0 for polygon in outers))
        self.assertAlmostEqual(outers[0].area, 79595.0, delta=1.0)
        self.assertEqual(outers[0].bbox[1], bbox[1])
        self.assertEqual(outers[0].bbox[3], bbox[3])
