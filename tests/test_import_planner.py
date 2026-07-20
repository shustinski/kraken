from __future__ import annotations

import unittest

from kraken_manager.application.imports import ImportMappingMode, ImportPlanner, ImportSource


class ImportPlannerTests(unittest.TestCase):
    def test_xy_mapping_reports_sparse_coverage_without_materializing_empty_frames(self) -> None:
        plan = ImportPlanner().plan(
            width=3,
            height=2,
            sources=(
                ImportSource("a", "metal_1_1.png", 10),
                ImportSource("b", "metal_3_2.png", 20),
            ),
            mode=ImportMappingMode.XY_FILENAME,
        )
        self.assertTrue(plan.ready)
        self.assertEqual([(1, 1), (3, 2)], [(item.x, item.y) for item in plan.items])
        self.assertEqual(4, plan.missing_coordinates)
        self.assertEqual(30, plan.total_bytes)

    def test_row_major_regex_and_explicit_mapping(self) -> None:
        planner = ImportPlanner()
        row = planner.plan(
            width=2,
            height=2,
            sources=(ImportSource("a", "frame_3.tif", 1),),
            mode=ImportMappingMode.ROW_MAJOR_SUFFIX,
        )
        self.assertEqual((1, 2), (row.items[0].x, row.items[0].y))
        regex = planner.plan(
            width=5,
            height=5,
            sources=(ImportSource("a", "x04-y02.png", 1),),
            mode=ImportMappingMode.REGEX,
            regex=r"x(?P<x>\d+)-y(?P<y>\d+)",
        )
        self.assertEqual((4, 2), (regex.items[0].x, regex.items[0].y))
        explicit = planner.plan(
            width=1,
            height=1,
            sources=(ImportSource("a", "unknown.bin", 1),),
            mode=ImportMappingMode.EXPLICIT,
            explicit={"a": (1, 1)},
        )
        self.assertTrue(explicit.ready)

    def test_conflicts_and_out_of_grid_block_commit(self) -> None:
        plan = ImportPlanner().plan(
            width=2,
            height=2,
            sources=(
                ImportSource("a", "first_1_1.png", 1),
                ImportSource("b", "second_1_1.png", 1),
                ImportSource("c", "third_9_9.png", 1),
            ),
            mode=ImportMappingMode.XY_FILENAME,
        )
        self.assertFalse(plan.ready)
        self.assertIn("duplicate_coordinate", {issue.code for issue in plan.issues})
        self.assertIn("outside_grid", {issue.code for issue in plan.issues})


if __name__ == "__main__":
    unittest.main()
