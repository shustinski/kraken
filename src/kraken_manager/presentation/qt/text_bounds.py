"""Keep popup and table text from requesting a multi-screen geometry.

Qt's tooltip and menu roll effect copies the popup's full size. A tooltip or
wrapped cell whose text is an unbounded document becomes tens of thousands of
pixels tall, and Windows then rejects the position as a signed 16-bit overflow.
"""

from __future__ import annotations

POPUP_TEXT_LIMIT = 1200
TABLE_ROW_HEIGHT_LIMIT = 96


def bounded_popup_text(text: object, *, limit: int = POPUP_TEXT_LIMIT) -> str:
    value = str(text or "")
    if len(value) <= limit:
        return value
    return value[: limit - 1].rstrip() + "…"


def cap_table_row_heights(table, maximum: int = TABLE_ROW_HEIGHT_LIMIT) -> None:
    for row in range(table.rowCount()):
        if table.rowHeight(row) > maximum:
            table.setRowHeight(row, maximum)
