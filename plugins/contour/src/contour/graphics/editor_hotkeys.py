"""Single source of truth for editor tool shortcuts (sequences + names for UI).

Shortcuts are installed on the graphics view with WidgetShortcut context so they
do not fire while typing in other widgets.
"""

from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QKeyEvent, QKeySequence

from ..infrastructure.runtime_config import config_string
from .tools import EditorTool

_DEFAULT_TOOL_SEQUENCE_STRINGS: dict[EditorTool, str | None] = {
    EditorTool.SELECT: "V",
    EditorTool.SELECT_AREA: None,
    EditorTool.PAN: "H",
    EditorTool.RULER: "K",
    EditorTool.ADD_POLYGON: "P",
    EditorTool.BRUSH: "B",
    EditorTool.TRACE_PEN: "T",
    EditorTool.ADD_VIA: "U",
    EditorTool.ADD_VERTEX: "A",
    EditorTool.DELETE_VERTEX: "D",
    EditorTool.MOVE_VERTEX: "M",
    EditorTool.ANTIALIAS: None,
    EditorTool.DELETE_POLYGON: None,
}


def tool_shortcut_sequence(tool: EditorTool) -> QKeySequence | None:
    default = _DEFAULT_TOOL_SEQUENCE_STRINGS.get(tool)
    raw = config_string("shortcuts", f"tool_{tool.value}", default or "")
    if not raw:
        return None
    return QKeySequence(raw)


def shortcut_sequence(action: str, default: str) -> QKeySequence | None:
    raw = config_string("shortcuts", action, default)
    return QKeySequence(raw) if raw else None


def shortcut_native_text(action: str, default: str) -> str:
    raw = config_string("shortcuts", action, default)
    return " / ".join(
        QKeySequence(item.strip()).toString(QKeySequence.SequenceFormat.NativeText)
        for item in raw.split(";")
        if item.strip()
    )


def event_matches_shortcut(event: QKeyEvent, action: str, default: str) -> bool:
    raw = config_string("shortcuts", action, default)
    modifier_keys = {
        "shift": Qt.Key.Key_Shift,
        "ctrl": Qt.Key.Key_Control,
        "control": Qt.Key.Key_Control,
        "alt": Qt.Key.Key_Alt,
        "meta": Qt.Key.Key_Meta,
    }
    alternatives = [item.strip() for item in raw.split(";") if item.strip()]
    if any(event.key() == modifier_keys.get(item.lower()) for item in alternatives):
        return True
    actual = QKeySequence(event.keyCombination())
    return any(actual.matches(QKeySequence(item)) == QKeySequence.SequenceMatch.ExactMatch for item in alternatives)


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
    EditorTool.TRACE_PEN: ("Трасса", "Trace"),
    EditorTool.ADD_VIA: ("Переход", "Via"),
    EditorTool.ADD_VERTEX: ("Добавить вершину", "Add vertex"),
    EditorTool.DELETE_VERTEX: ("Удалить вершину", "Delete vertex"),
    EditorTool.MOVE_VERTEX: ("Перемещение вершины", "Move vertex"),
    EditorTool.ANTIALIAS: ("Антиалиасинг", "Antialias"),
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
        EditorTool.TRACE_PEN,
        EditorTool.ADD_VIA,
        EditorTool.ADD_VERTEX,
        EditorTool.DELETE_VERTEX,
        EditorTool.MOVE_VERTEX,
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
    undo = shortcut_native_text("undo", "Ctrl+Z")
    redo = shortcut_native_text("redo", "Ctrl+Y")
    copy = shortcut_native_text("copy", "Ctrl+C")
    cut = shortcut_native_text("cut", "Ctrl+X")
    paste = shortcut_native_text("paste", "Ctrl+V")
    delete_selection = shortcut_native_text("delete_selection", "Del")
    cancel = shortcut_native_text("cancel", "Esc")
    hold_vectors = shortcut_native_text("hold_vectors", "Space")
    fit_view = shortcut_native_text("fit_view", "F")
    hold_source = shortcut_native_text("hold_source_image", "X")
    finish_polygon = shortcut_native_text("finish_polygon", "Enter")
    cycle_mode = shortcut_native_text("cycle_tool_mode", "Shift")
    if ru:
        return [
            ("Отменить", undo),
            ("Вернуть", redo),
            ("Копировать выделение", copy),
            ("Вырезать выделение", cut),
            ("Вставить", paste),
            ("Удалить выделенные полигоны", delete_selection),
            ("Снять выделение / отменить вставку", cancel),
            ("Переключить видимость векторов", hold_vectors),
            ("Подогнать изображение под окно", fit_view),
            ("Временно скрыть векторы и перемещать изображение", "Средняя кнопка мыши"),
            ("Временно показать исходное изображение без фильтров", hold_source),
            ("Завершить полигон по точкам", finish_polygon),
            ("Масштаб (колесо)", "Ctrl+колесо"),
            ("Прокрутка по горизонтали", "Shift+колесо"),
            ("В режиме полигона — временно точки или прямоугольник", cycle_mode),
            ("Добавить к выделению / убрать из выделения", "Ctrl+клик"),
        ]
    return [
        ("Undo", undo),
        ("Redo", redo),
        ("Copy selection", copy),
        ("Cut selection", cut),
        ("Paste", paste),
        ("Delete selected polygons", delete_selection),
        ("Clear selection / cancel paste", cancel),
        ("Temporarily hide vector overlays", hold_vectors),
        ("Fit image to view", fit_view),
        ("Temporarily hide vectors and pan", "Middle mouse button"),
        ("Temporarily show source image without filters", hold_source),
        ("Finish point polygon", finish_polygon),
        ("Zoom (wheel)", "Ctrl+wheel"),
        ("Scroll horizontally", "Shift+wheel"),
        ("In polygon tool: temporarily points vs rectangle", cycle_mode),
        ("Add/remove from selection", "Ctrl+click"),
    ]
