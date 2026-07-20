from __future__ import annotations

import tempfile
import unittest
from importlib.util import find_spec
from datetime import UTC, datetime, timedelta
from pathlib import Path

from kraken_manager.infrastructure.reports import ActivityRecord, ReportFilters, ReportService


class StatisticsReportTests(unittest.TestCase):
    def test_metrics_distinguish_unique_frames_and_operations(self) -> None:
        now = datetime.now(UTC)
        records = (
            ActivityRecord("e1", now, "frame.vectorized", "p", "l", frame_id="f", performer_id="worker"),
            ActivityRecord("e2", now, "artifact.vector.created", "p", "l", frame_id="f", performer_id="worker"),
            ActivityRecord("e3", now, "artifact.imported", "p", "l", frame_id="f", bytes_count=123),
            ActivityRecord("e4", now, "review.changes_requested", "p", "l", frame_id="f"),
        )
        filters = ReportFilters(now - timedelta(seconds=1), now + timedelta(seconds=1))
        metrics = ReportService().aggregate(records, filters)
        self.assertEqual(2, metrics.values["vectorization_operations"])
        self.assertEqual(1, metrics.values["unique_first_vectorized_frames"])
        self.assertEqual(123, metrics.values["imported_bytes"])
        self.assertEqual(("e1",), metrics.event_ids_by_metric["unique_first_vectorized_frames"])

    def test_application_event_names_are_classified_from_payload(self) -> None:
        now = datetime.now(UTC)
        records = (
            ActivityRecord(
                "version",
                now,
                "ArtifactVersionCreated",
                "p",
                "l",
                frame_id="f",
                bytes_count=50,
                payload={"filename": "1_1.cif", "source": "import"},
            ),
            ActivityRecord(
                "representation",
                now,
                "RepresentationCreated",
                "p",
                "l",
                payload={"kind": "image", "source": "NeuralImage", "binary": True},
            ),
        )
        filters = ReportFilters(now - timedelta(seconds=1), now + timedelta(seconds=1))

        metrics = ReportService().aggregate(records, filters)

        self.assertEqual(1, metrics.values["created_artifact_versions"])
        self.assertEqual(1, metrics.values["vectorization_operations"])
        self.assertEqual(1, metrics.values["imported_files"])
        self.assertEqual(50, metrics.values["imported_bytes"])
        self.assertEqual(1, metrics.values["binary_representations"])

    def test_first_vectorization_is_computed_over_complete_supplied_history(self) -> None:
        now = datetime.now(UTC)
        records = (
            ActivityRecord("old", now - timedelta(days=2), "FrameVectorized", "p", "l", frame_id="f1"),
            ActivityRecord("repeat", now, "FrameVectorized", "p", "l", frame_id="f1"),
            ActivityRecord("first", now, "FrameVectorized", "p", "l", frame_id="f2"),
        )
        filters = ReportFilters(now - timedelta(hours=1), now + timedelta(hours=1))

        metrics = ReportService().aggregate(iter(records), filters)

        self.assertEqual(2, metrics.values["vectorization_operations"])
        self.assertEqual(1, metrics.values["unique_first_vectorized_frames"])
        self.assertEqual(("first",), metrics.event_ids_by_metric["unique_first_vectorized_frames"])

    def test_csv_is_utf8_normalized_event_journal(self) -> None:
        now = datetime.now(UTC)
        record = ActivityRecord("событие", now, "artifact.imported", "проект", payload={"заметка": "тест"})
        with tempfile.TemporaryDirectory() as temporary:
            output = ReportService().write_csv(
                Path(temporary) / "report.csv",
                (record,),
                ReportFilters(now - timedelta(seconds=1), now + timedelta(seconds=1)),
            )
            contents = output.read_text(encoding="utf-8")
            self.assertIn("событие", contents)
            self.assertIn("заметка", contents)

    def test_csv_neutralizes_spreadsheet_formulas(self) -> None:
        now = datetime.now(UTC)
        record = ActivityRecord(
            "=HYPERLINK(\"https://invalid\")",
            now,
            "artifact.imported",
            "+project",
            actor_id="  @actor",
            tool="-tool",
        )
        with tempfile.TemporaryDirectory() as temporary:
            output = ReportService().write_csv(
                Path(temporary) / "report.csv",
                (record,),
                ReportFilters(now - timedelta(seconds=1), now + timedelta(seconds=1)),
            )
            contents = output.read_text(encoding="utf-8")

        self.assertIn("'=HYPERLINK", contents)
        self.assertIn("'+project", contents)
        self.assertIn("'  @actor", contents)
        self.assertIn("'-tool", contents)

    def test_csv_can_stream_an_already_sorted_cursor(self) -> None:
        now = datetime.now(UTC)
        consumed: list[str] = []

        def cursor():
            for event_id in ("one", "two"):
                consumed.append(event_id)
                yield ActivityRecord(event_id, now, "ProjectCreated", "p")

        with tempfile.TemporaryDirectory() as temporary:
            ReportService().write_csv(
                Path(temporary) / "report.csv",
                cursor(),
                ReportFilters(now - timedelta(seconds=1), now + timedelta(seconds=1)),
                assume_sorted=True,
            )

        self.assertEqual(["one", "two"], consumed)

    @unittest.skipUnless(find_spec("openpyxl"), "openpyxl is an optional reports dependency")
    def test_xlsx_neutralizes_spreadsheet_formulas(self) -> None:
        from openpyxl import load_workbook

        now = datetime.now(UTC)
        record = ActivityRecord("=1+1", now, "ProjectCreated", "+project", tool="@tool")
        with tempfile.TemporaryDirectory() as temporary:
            output = ReportService().write_xlsx(
                Path(temporary) / "report.xlsx",
                (record,),
                ReportFilters(now - timedelta(seconds=1), now + timedelta(seconds=1)),
            )
            workbook = load_workbook(output, read_only=True, data_only=False)
            events = workbook["Events"]
            row = next(events.iter_rows(min_row=2, values_only=False))
            values = tuple(cell.value for cell in row)
            data_types = tuple(cell.data_type for cell in row)
            workbook.close()

        self.assertEqual("'=1+1", values[0])
        self.assertNotEqual("f", data_types[0])
        self.assertEqual("'+project", values[3])
        self.assertEqual("'@tool", values[8])


if __name__ == "__main__":
    unittest.main()
