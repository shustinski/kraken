"""Popup text and table rows stay inside a size Windows can place."""

from __future__ import annotations

import pytest

from kraken_manager.presentation.qt.text_bounds import (
    POPUP_TEXT_LIMIT,
    TABLE_ROW_HEIGHT_LIMIT,
    bounded_popup_text,
    cap_table_row_heights,
)


def test_popup_text_keeps_a_short_prefix_of_an_unbounded_document() -> None:
    text = "line\n" * 5000
    bounded = bounded_popup_text(text)
    assert len(bounded) <= POPUP_TEXT_LIMIT
    assert bounded.endswith("…")
    assert "line" in bounded


def test_short_popup_text_is_unchanged() -> None:
    assert bounded_popup_text("06e05318-bdd7-4cd6-9bcd-a2d3b384c520") == (
        "06e05318-bdd7-4cd6-9bcd-a2d3b384c520"
    )


def test_table_row_height_is_capped_without_a_display(qapp) -> None:
    pytest.importorskip("PyQt6")
    from PyQt6.QtWidgets import QTableWidget

    del qapp
    table = QTableWidget(1, 1)
    table.setRowHeight(0, 36974)
    cap_table_row_heights(table)
    assert table.rowHeight(0) == TABLE_ROW_HEIGHT_LIMIT
    assert table.rowHeight(0) < 32767
