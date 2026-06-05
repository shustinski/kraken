"""Single source of truth for editor tool shortcuts (sequences + names for UI).

Shortcuts are installed on the graphics view with WidgetShortcut context so they
do not fire while typing in other widgets.
"""

from __future__ import annotations

from PyQt6.QtGui import QKeySequence

from .tools import EditorTool

_TOOL_SEQUENCE_STRINGS: dict[EditorTool, str | None] = {
    EditorTool.SELECT: "V",
    EditorTool.SELECT_AREA: None,
    EditorTool.PAN: "H",
    EditorTool.RULER: "K",
    EditorTool.ADD_POLYGON: "P",
    EditorTool.BRUSH: "B",
    EditorTool.ADD_VIA: "U",
    EditorTool.ADD_VERTEX: "A",
    EditorTool.DELETE_VERTEX: "D",
    EditorTool.MOVE_VERTEX: "M",
    EditorTool.DELETE_POLYGON: "E",
}


def tool_shortcut_sequence(tool: EditorTool) -> QKeySequence | None:
    raw = _TOOL_SEQUENCE_STRINGS.get(tool)
    if not raw:
        return None
    return QKeySequence(raw)


def tool_shortcut_native_text(tool: EditorTool) -> str:
    seq = tool_shortcut_sequence(tool)
    return seq.toString(QKeySequence.SequenceFormat.NativeText) if seq is not None else ""


def append_shortcut_to_tooltip(description: str, shortcut_native: str) -> str:
    if not shortcut_native:
        return description
    return f"{description}\n({shortcut_native})"


_EDITOR_TOOL_SHORT_LABELS: dict[EditorTool, tuple[str, str]] = {
    EditorTool.SELECT: ("Выбор", "Select"),
    EditorTool.PAN: ("Панорама", "Pan"),
    EditorTool.RULER: ("Линейка", "Ruler"),
    EditorTool.ADD_POLYGON: ("Полигон", "Polygon"),
    EditorTool.BRUSH: ("Кисть", "Brush"),
    EditorTool.ADD_VIA: ("Переход", "Via"),
    EditorTool.ADD_VERTEX: ("Добавить вершину", "Add vertex"),
    EditorTool.DELETE_VERTEX: ("Удалить вершину", "Delete vertex"),
    EditorTool.MOVE_VERTEX: ("Перемещение вершины", "Move vertex"),
    EditorTool.DELETE_POLYGON: ("Удалить полигон", "Delete polygon"),
}


def editor_tool_hotkey_rows(*, ru: bool) -> list[tuple[str, str]]:
    """(human action name, native shortcut text) for help UI."""
    order = [
        EditorTool.SELECT,
        EditorTool.PAN,
        EditorTool.RULER,
        EditorTool.ADD_POLYGON,
        EditorTool.BRUSH,
        EditorTool.ADD_VIA,
        EditorTool.ADD_VERTEX,
        EditorTool.DELETE_VERTEX,
        EditorTool.MOVE_VERTEX,
        EditorTool.DELETE_POLYGON,
    ]
    rows: list[tuple[str, str]] = []
    for tool in order:
        seq_text = tool_shortcut_native_text(tool)
        if not seq_text:
            continue
        label = _EDITOR_TOOL_SHORT_LABELS[tool][0 if ru else 1]
        rows.append((label, seq_text))
    return rows


def build_editor_hotkeys_plain_text(*, ru: bool) -> str:
    """Multi-line reference for the help dialog (tools + general)."""
    lines: list[str] = []
    header = "Редактор — горячие клавиши (фокус на изображении)" if ru else "Editor hotkeys (image view focused)"
    lines.append(header)
    lines.append("")
    lines.append("— Инструменты —" if ru else "— Tools —")
    for label, key in editor_tool_hotkey_rows(ru=ru):
        lines.append(f"{label}: {key}")
    lines.append("")
    lines.append("— Общие —" if ru else "— General —")
    for label, key in editor_misc_hotkey_lines(ru=ru):
        lines.append(f"{label}: {key}")
    return "\n".join(lines)


def editor_misc_hotkey_lines(*, ru: bool) -> list[tuple[str, str]]:
    """(action description, key text) for help dialog — labels are human-facing."""
    undo = QKeySequence(QKeySequence.StandardKey.Undo).toString(QKeySequence.SequenceFormat.NativeText)
    redo = QKeySequence(QKeySequence.StandardKey.Redo).toString(QKeySequence.SequenceFormat.NativeText)
    copy = QKeySequence(QKeySequence.StandardKey.Copy).toString(QKeySequence.SequenceFormat.NativeText)
    cut = QKeySequence(QKeySequence.StandardKey.Cut).toString(QKeySequence.SequenceFormat.NativeText)
    paste = QKeySequence(QKeySequence.StandardKey.Paste).toString(QKeySequence.SequenceFormat.NativeText)
    if ru:
        return [
            ("Отменить", undo),
            ("Вернуть", redo),
            ("Копировать выделение", copy),
            ("Вырезать выделение", cut),
            ("Вставить", paste),
            ("Удалить выделенные полигоны", "Del"),
            ("Снять выделение / отменить вставку", "Esc"),
            ("Переключить видимость векторов", "Пробел"),
            ("Завершить полигон по точкам", "Enter"),
            ("Масштаб (колесо)", "Ctrl+колесо"),
            ("Прокрутка по горизонтали", "Shift+колесо"),
            ("В режиме полигона — временно точки или прямоугольник", "Shift"),
            ("Добавить к выделению / убрать из выделения", "Ctrl+клик"),
        ]
    return [
        ("Undo", undo),
        ("Redo", redo),
        ("Copy selection", copy),
        ("Cut selection", cut),
        ("Paste", paste),
        ("Delete selected polygons", "Del"),
        ("Clear selection / cancel paste", "Esc"),
        ("Toggle vector overlay visibility", "Space"),
        ("Finish point polygon", "Enter"),
        ("Zoom (wheel)", "Ctrl+wheel"),
        ("Scroll horizontally", "Shift+wheel"),
        ("In polygon tool: temporarily points vs rectangle", "Shift"),
        ("Add/remove from selection", "Ctrl+click"),
    ]
