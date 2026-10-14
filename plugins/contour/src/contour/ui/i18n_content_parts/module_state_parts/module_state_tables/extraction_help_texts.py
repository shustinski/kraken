"""EXTRACTION_HELP_TEXTS."""
from __future__ import annotations
from contour.graphics_view import EditorTool

EXTRACTION_HELP_TEXTS: LocalizedTextMap = {
    "extraction_profile": (
        "Выбирает независимый набор настроек для проводников или переходных отверстий.",
        "Selects an independent settings profile for conductors or contacts.",
    ),
    "retrieval_mode": (
        "Определяет, искать ли только внешние контуры или всю иерархию вложенных контуров.",
        "Controls whether only outer contours or the full nested contour hierarchy is extracted.",
    ),
    "approximation_mode": (
        "Задаёт, насколько подробно хранится контур: компактно или со всеми промежуточными точками.",
        "Controls how densely contour points are stored: compact or full detail.",
    ),
    "epsilon": (
        "Сила упрощения контура. Больше значение уменьшает число вершин и сглаживает форму.",
        "Contour simplification strength. Higher values reduce vertices and smooth the shape.",
    ),
    "epsilon_mode": (
        "Переключает epsilon между абсолютным значением в пикселях и долей от периметра контура.",
        "Switches epsilon between an absolute pixel value and a fraction of contour perimeter.",
    ),
    "min_area": (
        "Отсекает слишком маленькие объекты по площади.",
        "Rejects objects that are too small by area.",
    ),
    "max_area": (
        "Отсекает слишком большие объекты по площади.",
        "Rejects objects that are too large by area.",
    ),
    "min_perimeter": (
        "Отсекает короткие шумовые контуры по длине границы.",
        "Rejects short noisy contours by boundary length.",
    ),
    "max_perimeter": (
        "Отсекает слишком длинные и рваные контуры.",
        "Rejects overly long or ragged contours.",
    ),
    "min_points": (
        "Минимальное число вершин после аппроксимации. Помогает убрать вырожденные фигуры.",
        "Minimum number of vertices after approximation. Helps remove degenerate shapes.",
    ),
    "min_bbox_width": (
        "Минимальная ширина ограничивающего прямоугольника объекта.",
        "Minimum object bounding-box width.",
    ),
    "max_bbox_width": (
        "Максимальная ширина ограничивающего прямоугольника объекта.",
        "Maximum object bounding-box width.",
    ),
    "min_bbox_height": (
        "Минимальная высота ограничивающего прямоугольника объекта.",
        "Minimum object bounding-box height.",
    ),
    "max_bbox_height": (
        "Максимальная высота ограничивающего прямоугольника объекта.",
        "Maximum object bounding-box height.",
    ),
    "min_aspect_ratio": (
        "Минимальное отношение ширины к высоте. Полезно для отбора вытянутых объектов.",
        "Minimum width-to-height ratio. Useful for selecting elongated objects.",
    ),
    "max_aspect_ratio": (
        "Максимальное отношение ширины к высоте. Полезно для отсечения слишком вытянутых форм.",
        "Maximum width-to-height ratio. Useful for rejecting overly elongated shapes.",
    ),
    "exclude_border_touching": (
        "Убирает объекты, касающиеся границы изображения. Полезно против обрезанных фрагментов.",
        "Removes objects touching the image border. Useful against cropped fragments.",
    ),
    "min_solidity": (
        "Фильтр по заполненности относительно выпуклой оболочки. Больше значение оставляет более плотные формы.",
        "Compactness filter relative to the convex hull. Higher values keep denser shapes.",
    ),
    "min_extent": (
        "Фильтр по заполненности bbox. Больше значение оставляет объекты, лучше заполняющие свой прямоугольник.",
        "Bounding-box fill filter. Higher values keep objects that fill their box better.",
    ),
    "min_polygon_width": (
        "Минимальная допустимая ширина полигона в пикселях. "
        "Позволяет отсекать слишком тонкие линии, шумовые артефакты и ложные узкие сегменты. "
        "Если значение слишком маленькое — будет больше ложных полигонов. "
        "Если слишком большое — можно потерять реальные узкие дорожки или контакты.",
        "Minimum allowed polygon width in pixels; 0 disables the filter. Uses a robust local-thickness estimate.",
    ),
    "via_white_threshold": (
        "Добавляет к маске пиксели ярче заданного порога.",
        "Adds pixels brighter than the selected threshold to the mask.",
    ),
    "via_black_threshold": (
        "Добавляет к маске пиксели темнее заданного порога.",
        "Adds pixels darker than the selected threshold to the mask.",
    ),
    "via_threshold_range": (
        "Добавляет к маске пиксели, попадающие в заданный диапазон яркости.",
        "Adds pixels inside the selected intensity range to the mask.",
    ),
    "via_min_roundness": (
        "Минимальная округлость контакта в процентах: 100 близко к окружности, 0 отключает фильтр.",
        "Minimum contact roundness in percent: 100 is close to a circle, 0 disables the filter.",
    ),
    "via_size_mode": (
        "Переключает отбор контактов между диапазоном и точными значениями.",
        "Switches contact filtering between a size range and exact size values.",
    ),
    "min_via_width": (
        "Минимальная ширина контакта в профиле контактов.",
        "Minimum contact width in the contact profile.",
    ),
    "max_via_width": (
        "Максимальная ширина контакта в профиле контактов.",
        "Maximum contact width in the contact profile.",
    ),
    "min_via_height": (
        "Минимальная высота контакта в профиле контактов.",
        "Minimum contact height in the contact profile.",
    ),
    "max_via_height": (
        "Максимальная высота контакта в профиле контактов.",
        "Maximum contact height in the contact profile.",
    ),
    "fixed_via_widths": (
        "Список ширин контактов. Каждая ширина связана с высотой в той же позиции.",
        "Contact widths list. Each width is paired with the height at the same position.",
    ),
    "fixed_via_heights": (
        "Список высот контактов. Каждая высота связана с шириной в той же позиции.",
        "Contact heights list. Each height is paired with the width at the same position.",
    ),
    "min_hierarchy_depth": (
        "Минимальная глубина в иерархии контуров. Ноль означает внешние контуры.",
        "Minimum contour hierarchy depth. Zero means outer contours.",
    ),
    "min_inner_hole_area": (
        "Минимальная площадь отверстия для заливки. Меньшие отверстия удаляются при извлечении проводников и ручной постобработке.",
        "Minimum hole area to fill. Smaller holes are removed during conductor extraction and manual post-processing.",
    ),
    "max_hierarchy_depth": (
        "Максимальная глубина в иерархии контуров.",
        "Maximum contour hierarchy depth.",
    ),
    "max_hole_area_ratio": (
        "Ограничивает размер внутренней дырки относительно родительского контура.",
        "Limits inner-hole size relative to its parent contour.",
    ),
    "delta": (
        "RGB-допуск для выбранных цветов.",
        "RGB tolerance for the selected colors.",
    ),
    "min_component_area": (
        "Минимальная площадь белого компонента бинарной маски.",
        "Minimum white-component area in the binary mask.",
    ),
    "max_component_area": (
        "Максимальная площадь белого компонента бинарной маски. 0 означает без ограничения.",
        "Maximum white-component area. Zero means unlimited.",
    ),
    "min_component_perimeter": (
        "Минимальный периметр белого компонента бинарной маски.",
        "Minimum white-component perimeter in the binary mask.",
    ),
    "max_component_perimeter": (
        "Максимальный периметр белого компонента бинарной маски. 0 означает без ограничения.",
        "Maximum white-component perimeter. Zero means unlimited.",
    ),
}
