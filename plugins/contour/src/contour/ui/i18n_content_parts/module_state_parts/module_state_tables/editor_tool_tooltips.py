"""EDITOR_TOOL_TOOLTIPS."""
from __future__ import annotations
from contour.graphics_view import EditorTool

EDITOR_TOOL_TOOLTIPS: dict[EditorTool, tuple[str, str]] = {
    EditorTool.SELECT: (
        "Выбор: клик по полигону; перетаскивание с пустого места — рамка; Alt + перетаскивание — переместить полигон. "
        "Ctrl добавляет или убирает из выделения.",
        "Select: click a polygon; drag on empty space for a marquee; Alt + drag moves a polygon. "
        "Ctrl adds or toggles items in the selection.",
    ),
    EditorTool.SELECT_AREA: (
        "Выбор: клик по полигону; перетаскивание с пустого места — рамка; Alt + перетаскивание — переместить полигон. "
        "Ctrl добавляет или убирает из выделения.",
        "Select: click a polygon; drag on empty space for a marquee; Alt + drag moves a polygon. "
        "Ctrl adds or toggles items in the selection.",
    ),
    EditorTool.PAN: (
        "Перемещение изображения без изменения полигонов.",
        "Pan the image without editing polygons.",
    ),
    EditorTool.RULER: (
        "Измерение расстояния на изображении перетаскиванием мыши.",
        "Measure distance on the image by dragging the mouse.",
    ),
    EditorTool.ADD_POLYGON: (
        "Создание нового полигона точками или прямоугольником. Правая кнопка вырезает область из существующих полигонов.",
        "Create a new polygon with points or a rectangle. Right-click cuts an area out of existing polygons.",
    ),
    EditorTool.BRUSH: (
        "Рисование или стирание области кистью. Круг под курсором показывает текущую толщину кисти.",
        "Draw or erase an area with the brush. The circle under the cursor shows the current brush width.",
    ),
    EditorTool.TRACE_PEN: (
        "Перо дорожки: кликайте по вершинам центральной линии, затем дважды кликните или нажмите Enter. Shift привязывает сегменты к 45 градусам; цепочки правой кнопкой стирают.",
        "Trace pen: click trace centerline vertices, then double-click or press Enter to commit. Shift snaps segments to 45 degrees; right-click chains erase.",
    ),
    EditorTool.ADD_VIA: (
        "Поставить контакт заданной ширины и высоты в месте клика.",
        "Place a contact of the configured width and height at the click position.",
    ),
    EditorTool.ADD_VERTEX: (
        "Сначала выберите полигон кликом, затем добавляйте вершины на его границу. Enter или Escape завершают работу с полигоном.",
        "Click a polygon to select it, then add vertices along its boundary. Press Enter or Escape to finish with the current polygon.",
    ),
    EditorTool.DELETE_VERTEX: (
        "Удалить вершину выбранного полигона. Режим удаления задает, удаляется одна точка или область.",
        "Delete vertices from the selected polygon. The delete mode controls whether one point or an area is removed.",
    ),
    EditorTool.MOVE_VERTEX: (
        "Переместить вершину или ребро полигона. Вершина и ребро подсвечиваются при наведении.",
        "Move a polygon vertex or edge. The target vertex or edge is highlighted on hover.",
    ),
    EditorTool.ANTIALIAS: (
        "Сгладить полигон под курсором: показывает вершины и уменьшает их число по клику.",
        "Shows vertices for the polygon under the cursor and reduces its vertex count on click.",
    ),
    EditorTool.DELETE_POLYGON: (
        "Удалить полигон, по которому вы кликнете.",
        "Delete the polygon you click.",
    ),
}
