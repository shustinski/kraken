from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
from PyQt6.QtCore import QEvent, QPointF, QRectF, QSize, QSettings, QSignalBlocker, Qt, QThreadPool, QTimer, pyqtSignal
from PyQt6.QtGui import QBrush, QColor, QIcon, QPainter, QPainterPath, QPen, QPixmap, QPolygonF
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QAbstractSpinBox,
    QButtonGroup,
    QCheckBox,
    QColorDialog,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QStyle,
    QTabWidget,
    QTreeWidget,
    QTreeWidgetItem,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from .adapters.qt.image_conversion import cv_to_qimage
from .adapters.qt.preview import AutoTuneRunnable, PreparedImageRunnable, PreviewImageView, PreviewProcessingRunnable
from .application.dto import PersistedPaths
from .application.processing import (
    VIA_SIZE_MODE_FIXED,
    VIA_SIZE_MODE_RANGE,
    ContourExtractionSettings,
    DisplaySettings,
    SaveOptions,
    normalize_via_size_mode,
)
from .application.services import WorkspaceSession
from .application.use_cases import (
    AutoTuneResult,
    PreparedImageRequest,
    PreviewProcessingRequest,
    build_prepared_image_signature,
    build_preview_request_signature,
    index_cif_directory,
    load_input_directory,
)
from .batch_processor import BatchProcessor
from .contour_extractor import APPROXIMATION_MODE_MAP, RETRIEVAL_MODE_MAP
from .domain import PolygonData
from .graphics_view import BrushMode, DeleteVertexMode, EditorTool, PolygonCreateMode, PolygonEditorView
from .infrastructure import WidgetDisplaySettingsStore, WidgetPathSettingsStore
from .i18n import active_language, tr
from .pipeline import (
    PreprocessingPipeline,
    available_operations,
    get_choice_display_label,
    get_operation_descriptor,
    get_operation_display_name,
    get_parameter_display_label,
)
from .serializers import export_dataset_frame, load_polygons_cif, save_polygons_cif, save_result_bundle
from .utils import is_image_path, load_image_color, scan_image_files


LocalizedTextMap = dict[str, tuple[str, str]]


FRAME_STATUS_ROLE = int(Qt.ItemDataRole.UserRole) + 1
FRAME_STATUS_UNCHANGED = "unchanged"
FRAME_STATUS_VIEWED = "viewed"
FRAME_STATUS_MODIFIED = "modified"
VIA_PRESETS_SETTINGS_KEY = "via_search/user_presets"


class PipelineListWidget(QListWidget):
    deletePressed = pyqtSignal()
    orderChanged = pyqtSignal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setDragDropMode(QAbstractItemView.DragDropMode.InternalMove)
        self.setDefaultDropAction(Qt.DropAction.MoveAction)
        self.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)

    def keyPressEvent(self, event) -> None:
        if event.key() in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace):
            self.deletePressed.emit()
            event.accept()
            return
        super().keyPressEvent(event)

    def dropEvent(self, event) -> None:
        super().dropEvent(event)
        self.orderChanged.emit()

    def contextMenuEvent(self, event) -> None:
        if self.currentRow() < 0:
            return
        menu = QMenu(self)
        delete_action = menu.addAction("Удалить" if active_language() == "ru" else "Delete")
        if menu.exec(event.globalPos()) == delete_action:
            self.deletePressed.emit()


EXTRACTION_HELP_TEXTS: LocalizedTextMap = {
    "extraction_profile": (
        "Выбирает независимый набор настроек для проводников или переходных отверстий.",
        "Selects an independent settings profile for conductors or vias.",
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
        "Минимальная округлость via в процентах: 100 близко к окружности, 0 отключает фильтр.",
        "Minimum via roundness in percent: 100 is close to a circle, 0 disables the filter.",
    ),
    "via_size_mode": (
        "РџРµСЂРµРєР»СЋС‡Р°РµС‚ РѕС‚Р±РѕСЂ via РјРµР¶РґСѓ РґРёР°РїР°Р·РѕРЅРѕРј Рё С‚РѕС‡РЅС‹РјРё Р·РЅР°С‡РµРЅРёСЏРјРё.",
        "Switches via filtering between a size range and exact size values.",
    ),
    "min_via_width": (
        "Минимальная ширина переходного отверстия в via-профиле.",
        "Minimum via width in the via profile.",
    ),
    "max_via_width": (
        "Максимальная ширина переходного отверстия в via-профиле.",
        "Maximum via width in the via profile.",
    ),
    "min_via_height": (
        "Минимальная высота переходного отверстия в via-профиле.",
        "Minimum via height in the via profile.",
    ),
    "max_via_height": (
        "Максимальная высота переходного отверстия в via-профиле.",
        "Maximum via height in the via profile.",
    ),
    "fixed_via_widths": (
        "РЎРїРёСЃРѕРє С€РёСЂРёРЅ via. РљР°Р¶РґР°СЏ С€РёСЂРёРЅР° СЃРІСЏР·Р°РЅР° СЃ РІС‹СЃРѕС‚РѕР№ РІ С‚РѕР№ Р¶Рµ РїРѕР·РёС†РёРё.",
        "Via widths list. Each width is paired with the height at the same position.",
    ),
    "fixed_via_heights": (
        "РЎРїРёСЃРѕРє РІС‹СЃРѕС‚ via. РљР°Р¶РґР°СЏ РІС‹СЃРѕС‚Р° СЃРІСЏР·Р°РЅР° СЃ С€РёСЂРёРЅРѕР№ РІ С‚РѕР№ Р¶Рµ РїРѕР·РёС†РёРё.",
        "Via heights list. Each height is paired with the width at the same position.",
    ),
    "min_hierarchy_depth": (
        "Минимальная глубина в иерархии контуров. Ноль означает внешние контуры.",
        "Minimum contour hierarchy depth. Zero means outer contours.",
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


PIPELINE_PARAMETER_HELP_TEXTS: LocalizedTextMap = {
    "kernel_size": (
        "Размер окна обработки. Большее окно сильнее сглаживает или меняет форму маски, но теряет мелкие детали.",
        "Processing window size. Larger windows smooth or reshape the mask more aggressively but lose fine detail.",
    ),
    "iterations": (
        "Число повторов морфологической операции. Больше повторов усиливает эффект.",
        "Number of morphology repeats. More iterations make the effect stronger.",
    ),
    "shape": (
        "Форма структурирующего элемента. Rect даёт жёсткую геометрию, ellipse мягче, cross тоньше воздействует на линии.",
        "Structuring element shape. Rect is strict, ellipse softer, cross affects thin lines more gently.",
    ),
    "sigma_x": (
        "Сила гауссова размытия. Чем больше, тем мягче края и меньше шум.",
        "Gaussian blur strength. Higher values soften edges and reduce noise more.",
    ),
    "diameter": (
        "Размер области bilateral-фильтрации. Большее значение сильнее сглаживает текстуры.",
        "Bilateral neighborhood size. Higher values smooth textures more strongly.",
    ),
    "sigma_color": (
        "Насколько сильно фильтр сглаживает различия по яркости.",
        "How strongly the filter smooths intensity differences.",
    ),
    "sigma_space": (
        "Насколько широко bilateral учитывает соседние пиксели по расстоянию.",
        "How far the bilateral filter looks spatially.",
    ),
    "clip_limit": (
        "Ограничение усиления локального контраста для CLAHE.",
        "Local contrast boost limit for CLAHE.",
    ),
    "tile_grid_size": (
        "Размер сетки локального выравнивания контраста в CLAHE.",
        "Local contrast grid size for CLAHE.",
    ),
    "alpha": (
        "Множитель контраста. Больше значение делает различия яркостей заметнее.",
        "Contrast multiplier. Higher values increase tonal separation.",
    ),
    "beta": (
        "Сдвиг яркости. Положительное значение осветляет, отрицательное затемняет.",
        "Brightness offset. Positive brightens, negative darkens.",
    ),
    "gamma": (
        "Нелинейная коррекция яркости. Меньше 1 осветляет тени, больше 1 затемняет.",
        "Nonlinear brightness correction. Below 1 brightens shadows, above 1 darkens them.",
    ),
    "threshold": (
        "Порог бинаризации. Пиксели по одну сторону порога становятся фоном, по другую объектом.",
        "Binarization threshold. Pixels on one side become background, on the other become foreground.",
    ),
    "max_value": (
        "Значение, которым заполняются пиксели после пороговой операции.",
        "Output value assigned by thresholding.",
    ),
    "threshold_type": (
        "Тип пороговой операции: прямой, инверсный и другие варианты преобразования.",
        "Thresholding mode: direct, inverted, and other conversion variants.",
    ),
    "adaptive_method": (
        "Способ локального расчёта порога для adaptive threshold.",
        "Method used to compute local thresholds in adaptive thresholding.",
    ),
    "block_size": (
        "Размер локального окна для adaptive threshold.",
        "Local window size for adaptive thresholding.",
    ),
    "c_value": (
        "Смещение локального порога. Меняет агрессивность отделения объекта от фона.",
        "Local threshold offset. Changes how aggressively foreground is separated from background.",
    ),
    "threshold_mode": (
        "Способ построения первичной маски перед уточнением по границам.",
        "How the initial mask is built before edge-guided refinement.",
    ),
    "edge_detector": (
        "Метод поиска резких перепадов яркости для уточнения границы маски.",
        "Method used to find intensity edges for mask boundary refinement.",
    ),
    "edge_percentile": (
        "Порог силы градиента для Sobel. Больше значение оставляет только более резкие края.",
        "Sobel gradient-strength cutoff. Higher values keep only sharper edges.",
    ),
    "correction_radius": (
        "Максимальный сдвиг границы маски в пикселях при поиске ближайшего яркостного края.",
        "Maximum mask-boundary shift in pixels while looking for the nearest intensity edge.",
    ),
    "fill_holes": (
        "Заполняет внутренние отверстия после уточнения границы.",
        "Fills inner holes after boundary refinement.",
    ),
    "threshold1": (
        "Нижний порог Canny. Определяет чувствительность к слабым границам.",
        "Lower Canny threshold. Controls sensitivity to weak edges.",
    ),
    "threshold2": (
        "Верхний порог Canny. Определяет, какие границы считаются уверенными.",
        "Upper Canny threshold. Controls which edges are considered strong.",
    ),
    "aperture_size": (
        "Размер фильтра Собеля в Canny. Больше значение даёт более гладкую оценку градиента.",
        "Sobel kernel size in Canny. Larger values produce a smoother gradient estimate.",
    ),
    "l2gradient": (
        "Использует более точную, но немного более тяжёлую оценку градиента в Canny.",
        "Uses a more precise but slightly heavier gradient computation in Canny.",
    ),
    "width": (
        "Целевая ширина для resize или ширина выделяемой области для crop.",
        "Target width for resize or extracted region width for crop.",
    ),
    "height": (
        "Целевая высота для resize или высота выделяемой области для crop.",
        "Target height for resize or extracted region height for crop.",
    ),
    "keep_aspect": (
        "Сохраняет пропорции изображения при изменении размера.",
        "Preserves image aspect ratio during resizing.",
    ),
    "interpolation": (
        "Метод интерполяции при изменении размера изображения.",
        "Interpolation method used during resizing.",
    ),
    "x": (
        "Координата левого края области crop.",
        "Left coordinate of the crop region.",
    ),
    "y": (
        "Координата верхнего края области crop.",
        "Top coordinate of the crop region.",
    ),
    "amount": (
        "Сила повышения резкости. Больше значение сильнее подчёркивает контуры.",
        "Sharpening strength. Higher values emphasize edges more strongly.",
    ),
    "sigma": (
        "Ширина предварительного размытия перед повышением резкости.",
        "Pre-blur width used before sharpening.",
    ),
    "h": (
        "Сила подавления шума в Non-Local Means.",
        "Noise suppression strength in Non-Local Means denoising.",
    ),
    "template_window_size": (
        "Размер шаблона для сравнения текстур при denoise.",
        "Template size used for texture comparison during denoising.",
    ),
    "search_window_size": (
        "Размер области поиска похожих фрагментов при denoise.",
        "Search area size for similar patches during denoising.",
    ),
}
EXTRACTION_HELP_TEXTS.update(
    {
        "extraction_profile": (
            "Выбирает, что сейчас настраивается: проводники или переходные отверстия. У каждого профиля свои фильтры и свой результат векторизации.",
            "Selects what is being configured now: conductors or vias. Each profile has its own filters and vectorization result.",
        ),
        "retrieval_mode": (
            "Определяет, искать только внешние границы объектов или также вложенные контуры и отверстия внутри них.",
            "Controls whether only outer object borders are found or nested contours and holes are included too.",
        ),
        "approximation_mode": (
            "Задает подробность контура. Более подробный режим сохраняет больше точек, компактный делает форму проще.",
            "Controls contour detail. Full detail keeps more points, compact mode simplifies the shape.",
        ),
        "epsilon": (
            "Сглаживает найденный контур. Чем больше значение, тем меньше лишних вершин, но мелкие детали могут пропасть.",
            "Smooths the detected contour. Higher values remove more extra vertices, but small details can disappear.",
        ),
        "epsilon_mode": (
            "Меняет смысл сглаживания: фиксированное число пикселей или доля от длины контура. Относительный режим удобнее для объектов разного размера.",
            "Changes smoothing units: fixed pixels or a fraction of contour length. Relative mode is better for objects of different sizes.",
        ),
        "min_area": (
            "Отбрасывает объекты с площадью меньше этого значения. Используйте, чтобы убрать мелкий шум.",
            "Rejects objects with an area below this value. Use it to remove small noise.",
        ),
        "max_area": (
            "Отбрасывает объекты с площадью больше этого значения. Ноль обычно означает без верхнего ограничения.",
            "Rejects objects with an area above this value. Zero usually means no upper limit.",
        ),
        "min_perimeter": (
            "Отбрасывает контуры с короткой границей. Помогает убрать мелкие случайные точки и обрывки.",
            "Rejects contours with a short border. Helps remove tiny random specks and fragments.",
        ),
        "max_perimeter": (
            "Отбрасывает контуры со слишком длинной границей. Помогает убрать рваные или слишком крупные области.",
            "Rejects contours with an overly long border. Helps remove ragged or too-large regions.",
        ),
        "min_points": (
            "Минимальное число вершин у готового полигона. Увеличьте, если нужно убрать вырожденные треугольники и линии.",
            "Minimum vertex count for the final polygon. Increase it to remove degenerate triangles and lines.",
        ),
        "min_bbox_width": (
            "Минимальная ширина прямоугольника вокруг объекта. Объекты уже этого значения не попадут в результат.",
            "Minimum width of the object's bounding rectangle. Narrower objects are excluded.",
        ),
        "max_bbox_width": (
            "Максимальная ширина прямоугольника вокруг объекта. Ноль обычно означает без верхнего ограничения.",
            "Maximum width of the object's bounding rectangle. Zero usually means no upper limit.",
        ),
        "min_bbox_height": (
            "Минимальная высота прямоугольника вокруг объекта. Объекты ниже этого значения не попадут в результат.",
            "Minimum height of the object's bounding rectangle. Shorter objects are excluded.",
        ),
        "max_bbox_height": (
            "Максимальная высота прямоугольника вокруг объекта. Ноль обычно означает без верхнего ограничения.",
            "Maximum height of the object's bounding rectangle. Zero usually means no upper limit.",
        ),
        "min_aspect_ratio": (
            "Минимальное отношение ширины к высоте. Помогает оставить объекты нужной вытянутости.",
            "Minimum width-to-height ratio. Helps keep objects with the required elongation.",
        ),
        "max_aspect_ratio": (
            "Максимальное отношение ширины к высоте. Помогает убрать слишком вытянутые или слишком плоские объекты.",
            "Maximum width-to-height ratio. Helps remove objects that are too elongated or too flat.",
        ),
        "exclude_border_touching": (
            "Если включено, объекты, касающиеся края изображения, не сохраняются. Полезно для обрезанных фрагментов.",
            "When enabled, objects touching the image border are excluded. Useful for cropped fragments.",
        ),
        "min_solidity": (
            "Требуемая заполненность формы относительно ее выпуклой оболочки. Чем выше значение, тем сильнее отсекаются рваные и вогнутые области.",
            "Required shape fill relative to its convex hull. Higher values reject ragged and strongly concave regions.",
        ),
        "min_extent": (
            "Требуемая заполненность прямоугольника вокруг объекта. Чем выше значение, тем плотнее объект должен занимать свой bbox.",
            "Required fill of the object's bounding rectangle. Higher values require the object to occupy its box more tightly.",
        ),
        "via_white_threshold": (
            "Добавляет в маску via пиксели светлее заданного порога. Включайте для поиска светлых переходных отверстий.",
            "Adds pixels brighter than the threshold to the via mask. Enable it to detect bright vias.",
        ),
        "via_white_range": (
            "Добавляет в маску via пиксели, яркость которых попадает в диапазон для белых переходных отверстий.",
            "Adds pixels whose brightness falls inside the white-via range to the via mask.",
        ),
        "via_black_threshold": (
            "Добавляет в маску via пиксели темнее заданного порога. Включайте для поиска темных переходных отверстий.",
            "Adds pixels darker than the threshold to the via mask. Enable it to detect dark vias.",
        ),
        "via_black_range": (
            "Добавляет в маску via пиксели, яркость которых попадает в диапазон для черных переходных отверстий.",
            "Adds pixels whose brightness falls inside the black-via range to the via mask.",
        ),
        "via_detector_methods": (
            "Включает методы поиска via. Градиент ищет круговой перепад яркости, Hough ищет окружности, остальные методы помогают на бинарных и шумных кадрах.",
            "Enables via detection methods. Gradient finds circular brightness edges, Hough finds circles, and the other methods help on binary and noisy frames.",
        ),
        "via_detector_gradient_method": (
            "Ищет via по круговому перепаду яркости на границе контакта. Используйте для размытых или слабо отличимых via, когда важнее форма круглого края, а не абсолютная яркость.",
            "Finds vias by a circular brightness edge around the contact. Use it for blurred or weak vias when the round edge shape matters more than absolute brightness.",
        ),
        "via_detector_spot_method": (
            "Ищет компактные светлые или темные точки, которые выделяются относительно ближайшего фона. Хорошо работает для маленьких ярких контактов на дорожках и шумном фоне.",
            "Finds compact bright or dark spots that stand out from nearby background. Works well for small bright contacts on traces and noisy background.",
        ),
        "via_detector_hough_method": (
            "Ищет окружности через HoughCircles. Включайте, когда via выглядят как кольца или четкие круглые пятна; отключайте, если появляются лишние круги на дорожках и вертикальных границах.",
            "Finds circles with HoughCircles. Enable it when vias look like rings or clear round spots; disable it if traces and vertical edges create extra circles.",
        ),
        "via_detector_components_method": (
            "Ищет связные области после порогов и морфологии, затем фильтрует их по размеру и форме. Полезно на бинарных кадрах и там, где каждый via становится отдельным пятном.",
            "Finds connected regions after thresholds and morphology, then filters them by size and shape. Useful on binary frames and when each via becomes a separate blob.",
        ),
        "via_detector_contours_method": (
            "Ищет контуры областей-кандидатов и оценивает их геометрию. Используйте, когда контуры вокруг via уже хорошо извлекаются, но нужно отсеять дорожки и рваные области.",
            "Finds contours of candidate regions and evaluates their geometry. Use it when via contours are already extracted well but traces and ragged regions need filtering.",
        ),
        "via_detector_morphology_method": (
            "Ищет центры via после морфологического сглаживания и distance transform. Помогает склеить шумные пиксели в устойчивые круглые кандидаты на зернистых изображениях.",
            "Finds via centers after morphological smoothing and distance transform. Helps turn noisy pixels into stable round candidates on grainy images.",
        ),
        "via_detector_template_method": (
            "Ищет области, похожие на сохраненные пользователем образцы via. Используйте, когда внешний вид контактов стабилен; качество сильно зависит от выбранных шаблонов.",
            "Finds regions similar to user-saved via samples. Use it when contact appearance is stable; quality depends strongly on the selected templates.",
        ),
        "via_detector_blob_method": (
            "Ищет круглые пятна через blob detector с проверкой площади, округлости и выпуклости. Полезно для чистых точечных via; на сильном шуме может давать лишние кандидаты.",
            "Finds round spots with a blob detector using area, circularity, and convexity checks. Useful for clean dot-like vias; may add false candidates on heavy noise.",
        ),
        "via_gradient_min_strength": (
            "Минимальная сила кругового перепада яркости. Увеличьте значение, если появляются ложные via на шуме; уменьшите, если слабые контакты пропадают.",
            "Minimum circular edge strength. Increase it to remove noisy false vias; decrease it when weak contacts disappear.",
        ),
        "via_gradient_min_coverage": (
            "Какая часть окружности должна иметь заметный перепад яркости. Больше значение требует более замкнутую круглую границу via.",
            "How much of the circle must have a visible brightness edge. Higher values require a more complete circular via boundary.",
        ),
        "via_spot_min_contrast": (
            "Минимальная локальная яркость точки относительно ближайшего фона. Увеличьте, если находятся шум и дорожки; уменьшите, если слабые контакты пропадают.",
            "Minimum local brightness of a dot relative to nearby background. Increase it when noise and traces are detected; decrease it when weak contacts are missed.",
        ),
        "via_spot_min_roundness": (
            "Минимальная компактность яркой точки. Увеличьте, чтобы отсеять вытянутые участки дорожек и края; уменьшите для размытых контактов.",
            "Minimum compactness of the bright dot. Increase it to reject elongated traces and edges; decrease it for blurred contacts.",
        ),
        "via_spot_line_suppression": (
            "\u041f\u043e\u0434\u0430\u0432\u043b\u044f\u0435\u0442 \u0434\u043b\u0438\u043d\u043d\u044b\u0435 \u0433\u043e\u0440\u0438\u0437\u043e\u043d\u0442\u0430\u043b\u044c\u043d\u044b\u0435 \u0438 \u0432\u0435\u0440\u0442\u0438\u043a\u0430\u043b\u044c\u043d\u044b\u0435 \u0434\u043e\u0440\u043e\u0436\u043a\u0438 \u043f\u0435\u0440\u0435\u0434 \u043f\u043e\u0438\u0441\u043a\u043e\u043c \u043a\u0440\u0443\u0433\u043b\u044b\u0445 \u0442\u043e\u0447\u0435\u043a. \u0423\u0432\u0435\u043b\u0438\u0447\u044c\u0442\u0435, \u0435\u0441\u043b\u0438 \u043d\u0430 \u0434\u043e\u0440\u043e\u0436\u043a\u0430\u0445 \u043f\u043e\u044f\u0432\u043b\u044f\u044e\u0442\u0441\u044f \u043b\u043e\u0436\u043d\u044b\u0435 via; \u0443\u043c\u0435\u043d\u044c\u0448\u0438\u0442\u0435, \u0435\u0441\u043b\u0438 via \u0441\u043b\u0438\u0448\u043a\u043e\u043c \u0432\u044b\u0442\u0435\u0440\u043b\u0438\u0441\u044c.",
            "Suppresses long horizontal and vertical traces before round-dot detection. Increase it when traces create false vias; decrease it if real vias are erased.",
        ),
        "via_hough_edge_threshold": (
            "Порог сильного края для HoughCircles. Увеличьте, чтобы игнорировать слабый шум; уменьшите, чтобы находить менее контрастные окружности.",
            "Strong-edge threshold for HoughCircles. Increase it to ignore weak noise; decrease it to find lower-contrast circles.",
        ),
        "via_hough_accumulator_threshold": (
            "Сколько голосов нужно окружности в HoughCircles. Больше значение дает меньше, но надежнее кандидатов; меньше значение ищет слабые via.",
            "How many votes a Hough circle needs. Higher values produce fewer, stronger candidates; lower values find weaker vias.",
        ),
        "via_component_min_score": (
            "Минимальный отклик для кандидатов, найденных связанными компонентами. Увеличьте, чтобы убрать слабые пятна после бинаризации.",
            "Minimum response for connected-component candidates. Increase it to remove weak spots after binarization.",
        ),
        "via_contour_min_score": (
            "Минимальный отклик для кандидатов, найденных по контурам. Увеличьте, чтобы оставить только контуры с выраженным локальным сигналом.",
            "Minimum response for contour candidates. Increase it to keep only contours with a clear local signal.",
        ),
        "via_morphology_peak_scale": (
            "Минимальный размер центра via при поиске морфологических пиков относительно ожидаемого размера. Больше значение отсекает мелкие шумовые центры.",
            "Minimum via-center peak size relative to the expected size. Higher values reject small noisy centers.",
        ),
        "via_template_min_score": (
            "Минимальное совпадение с круглым шаблоном. Больше значение требует более похожую на круг область.",
            "Minimum circular-template match. Higher values require an area that looks more like a circle.",
        ),
        "via_templates": (
            "Список шаблонов via. Нажмите выбор шаблона и протяните рамку по переходному отверстию на изображении; все шаблоны используются при поиске.",
            "List of via templates. Click pick template and drag a rectangle over a via in the image; all templates are used during detection.",
        ),
        "via_blob_min_circularity": (
            "Минимальная округлость для Blob detector. Больше значение сильнее отбрасывает вытянутые и рваные пятна.",
            "Minimum circularity for Blob detector. Higher values reject elongated and ragged spots more strongly.",
        ),
        "reset_via_search": (
            "Возвращает методы поиска via и их параметры к значениям по умолчанию. Сохраненные шаблоны не удаляются.",
            "Restores via search methods and their parameters to defaults. Saved templates are not removed.",
        ),
        "via_noisy_traces_preset": (
            "Быстро настраивает поиск для кадров, где яркие круглые via находятся на фоне длинных дорожек и шума. Включает методы Точки и Градиент, усиливает подавление дорожек и отключает методы, которые часто дают лишние срабатывания.",
            "Quick setup for frames where bright round vias sit on long traces and noisy background. Enables Spots and Gradient, increases trace suppression, and disables methods that often create extra false hits.",
        ),
        "via_blurred_preset": (
            "Быстро настраивает поиск для слабых или размытых via. Понижает пороги, допускает менее компактную форму и оставляет инспектор включенным, чтобы можно было проверить найденные точки кликом.",
            "Quick setup for weak or blurred vias. Lowers thresholds, allows less compact spots, and keeps the inspector enabled so found points can be checked by click.",
        ),
        "via_preset_selector": (
            "Список встроенных и сохраненных пользователем пресетов поиска via. Пресет меняет только параметры распознавания, шаблоны и размеры via остаются на месте.",
            "List of built-in and user-saved via search presets. A preset changes recognition parameters only; templates and via sizes stay unchanged.",
        ),
        "debug_candidates": (
            "Включает инспектор распознавания via. После обработки кликните по найденной via, чтобы увидеть метод поиска, причину принятия и основные численные признаки.",
            "Enables the via recognition inspector. After processing, click a detected via to see the search method, acceptance reason, and main numeric features.",
        ),
        "via_threshold_range": (
            "Добавляет в маску via пиксели, яркость которых попадает в заданный диапазон. Удобно, когда via имеет средний тон.",
            "Adds pixels whose intensity falls inside the selected range to the via mask. Useful for mid-tone vias.",
        ),
        "via_min_roundness": (
            "Минимальная похожесть via на круг в процентах. Увеличьте, чтобы убрать вытянутые пятна; 0 отключает фильтр.",
            "Minimum via circularity in percent. Increase it to reject elongated spots; 0 disables the filter.",
        ),
        "via_size_mode": (
            "Выбирает способ отбора via по размеру: общий диапазон ширины/высоты или список точных размеров.",
            "Chooses how vias are filtered by size: a width/height range or a list of exact sizes.",
        ),
        "min_via_width": (
            "Минимальная ширина переходного отверстия. Более узкие объекты будут отброшены.",
            "Minimum via width. Narrower objects are rejected.",
        ),
        "max_via_width": (
            "Максимальная ширина переходного отверстия. Ноль означает без верхнего ограничения.",
            "Maximum via width. Zero means no upper limit.",
        ),
        "min_via_height": (
            "Минимальная высота переходного отверстия. Более низкие объекты будут отброшены.",
            "Minimum via height. Shorter objects are rejected.",
        ),
        "max_via_height": (
            "Максимальная высота переходного отверстия. Ноль означает без верхнего ограничения.",
            "Maximum via height. Zero means no upper limit.",
        ),
        "fixed_via_widths": (
            "Ширина via из списка точных размеров. Пара X/Y в одной строке описывает один допустимый размер.",
            "Via width in the exact-size list. The X/Y pair in one row describes one allowed size.",
        ),
        "fixed_via_heights": (
            "Высота via из списка точных размеров. Пара X/Y в одной строке описывает один допустимый размер.",
            "Via height in the exact-size list. The X/Y pair in one row describes one allowed size.",
        ),
        "min_hierarchy_depth": (
            "Минимальная глубина вложенности контура. Ноль означает внешний контур; большие значения выбирают внутренние контуры.",
            "Minimum contour nesting depth. Zero means an outer contour; higher values select inner contours.",
        ),
        "max_hierarchy_depth": (
            "Максимальная глубина вложенности контура. Ограничивает, насколько глубоко искать внутренние контуры.",
            "Maximum contour nesting depth. Limits how deep inner contours are accepted.",
        ),
        "max_hole_area_ratio": (
            "Максимальная площадь отверстия внутри объекта относительно внешнего контура. Уменьшите, чтобы убрать объекты с большими дырками.",
            "Maximum inner-hole area relative to the outer contour. Lower it to reject objects with large holes.",
        ),
        "delta": (
            "Допуск к выбранному цвету при цветовой бинаризации. Большее значение захватывает больше похожих оттенков.",
            "Tolerance around selected colors for color binarization. Higher values include more similar shades.",
        ),
        "min_component_area": (
            "Минимальная площадь белой области в бинарной маске. Меньшие области удаляются как шум.",
            "Minimum area of a white region in the binary mask. Smaller regions are removed as noise.",
        ),
        "max_component_area": (
            "Максимальная площадь белой области в бинарной маске. Ноль означает без верхнего ограничения.",
            "Maximum area of a white region in the binary mask. Zero means no upper limit.",
        ),
        "min_component_perimeter": (
            "Минимальная длина границы белой области в бинарной маске. Короткие области удаляются.",
            "Minimum border length of a white region in the binary mask. Shorter regions are removed.",
        ),
        "max_component_perimeter": (
            "Максимальная длина границы белой области в бинарной маске. Ноль означает без верхнего ограничения.",
            "Maximum border length of a white region in the binary mask. Zero means no upper limit.",
        ),
    }
)

PIPELINE_PARAMETER_HELP_TEXTS.update(
    {
        "kernel_size": (
            "Размер окна обработки. Большее окно сильнее сглаживает или меняет форму маски, но может съесть мелкие детали.",
            "Processing window size. A larger window smooths or reshapes the mask more strongly, but can remove fine details.",
        ),
        "iterations": (
            "Сколько раз повторить операцию. Чем больше повторов, тем сильнее эффект.",
            "How many times to repeat the operation. More repeats make the effect stronger.",
        ),
        "shape": (
            "Форма ядра обработки: прямоугольник действует жестче, эллипс мягче, крест слабее влияет на диагонали.",
            "Processing kernel shape: rectangle is stricter, ellipse is softer, cross affects diagonals less.",
        ),
        "sigma_x": (
            "Сила гауссова размытия. Увеличьте, чтобы убрать шум; уменьшите, чтобы сохранить резкие края.",
            "Gaussian blur strength. Increase it to reduce noise; decrease it to keep sharper edges.",
        ),
        "diameter": (
            "Размер области bilateral-фильтра. Большее значение сильнее сглаживает текстуру вокруг пикселя.",
            "Bilateral filter neighborhood size. Higher values smooth texture around each pixel more strongly.",
        ),
        "sigma_color": (
            "Насколько сильно фильтр сглаживает различия по яркости и цвету. Больше значение объединяет более разные оттенки.",
            "How strongly the filter smooths brightness and color differences. Higher values merge more different tones.",
        ),
        "sigma_space": (
            "Насколько далеко фильтр смотрит вокруг пикселя. Большее значение учитывает более широкую область.",
            "How far around each pixel the filter looks. Higher values use a wider area.",
        ),
        "clip_limit": (
            "Ограничивает усиление локального контраста. Уменьшите, если появляются пересвеченные пятна или шум.",
            "Limits local contrast amplification. Lower it if glare spots or noise appear.",
        ),
        "tile_grid_size": (
            "Размер участков для локального контраста. Меньшие участки дают более локальный эффект, большие действуют плавнее.",
            "Tile size for local contrast. Smaller tiles give a more local effect, larger tiles act more smoothly.",
        ),
        "alpha": (
            "Коэффициент контраста. Больше 1 усиливает различия, меньше 1 делает изображение мягче.",
            "Contrast multiplier. Above 1 increases differences, below 1 softens the image.",
        ),
        "beta": (
            "Сдвиг яркости. Положительное значение осветляет изображение, отрицательное затемняет.",
            "Brightness offset. Positive values brighten the image, negative values darken it.",
        ),
        "gamma": (
            "Нелинейная коррекция яркости. Меньше 1 осветляет темные детали, больше 1 затемняет средние тона.",
            "Nonlinear brightness correction. Below 1 brightens dark details, above 1 darkens midtones.",
        ),
        "threshold": (
            "Порог бинаризации. Пиксели по одну сторону порога станут фоном, по другую - объектом.",
            "Binarization threshold. Pixels on one side become background, on the other side become foreground.",
        ),
        "max_value": (
            "Яркость, которую получают пиксели объекта после пороговой операции. Обычно оставляют 255 для белой маски.",
            "Output brightness assigned to foreground pixels after thresholding. Usually keep 255 for a white mask.",
        ),
        "threshold_type": (
            "Направление порога: обычное оставляет светлые объекты, инверсное - темные объекты.",
            "Threshold direction: normal keeps bright objects, inverted keeps dark objects.",
        ),
        "adaptive_method": (
            "Как считать локальный порог: среднее проще, Gaussian мягче учитывает пиксели рядом с центром окна.",
            "How to compute the local threshold: mean is simpler, Gaussian weights pixels near the window center more smoothly.",
        ),
        "block_size": (
            "Размер локального окна для адаптивного порога. Больше окно лучше при плавной подсветке, меньше - при мелких изменениях.",
            "Local window size for adaptive thresholding. Larger works better for smooth lighting, smaller for fine changes.",
        ),
        "c_value": (
            "Сдвиг локального порога. Меняет агрессивность отделения объекта от фона.",
            "Local threshold offset. Changes how aggressively foreground is separated from background.",
        ),
        "threshold_mode": (
            "Как строится начальная маска перед уточнением границ: вручную, Otsu или адаптивно.",
            "How the initial mask is built before edge refinement: manual, Otsu, or adaptive.",
        ),
        "edge_detector": (
            "Метод поиска резких границ для уточнения маски. Canny строже, Sobel проще и мягче.",
            "Edge detection method for mask refinement. Canny is stricter, Sobel is simpler and softer.",
        ),
        "edge_percentile": (
            "Порог силы границы для Sobel. Большее значение оставляет только самые резкие края.",
            "Sobel edge strength cutoff. Higher values keep only the sharpest edges.",
        ),
        "correction_radius": (
            "На сколько пикселей можно сдвигать границу маски к ближайшему найденному краю.",
            "How many pixels the mask border may move toward the nearest detected edge.",
        ),
        "threshold1": (
            "Нижний порог Canny. Уменьшите, чтобы видеть слабые границы; увеличьте, чтобы убрать шум.",
            "Lower Canny threshold. Lower it to detect weak edges; raise it to reduce noise.",
        ),
        "threshold2": (
            "Верхний порог Canny. Определяет, какие границы считаются уверенными.",
            "Upper Canny threshold. Controls which edges are considered strong.",
        ),
        "aperture_size": (
            "Размер фильтра Sobel внутри Canny. Большее значение дает более плавную оценку границы.",
            "Sobel filter size inside Canny. Higher values produce a smoother edge estimate.",
        ),
        "l2gradient": (
            "Включает более точный расчет силы границы в Canny. Обычно дает чуть стабильнее результат, но работает тяжелее.",
            "Uses a more precise Canny edge-strength calculation. Usually slightly more stable but heavier.",
        ),
        "fill_holes": (
            "Заполняет внутренние дырки в маске после уточнения границ. Отключите, если отверстия должны сохраниться.",
            "Fills inner holes in the mask after edge refinement. Disable it if holes should remain.",
        ),
        "min_component_area": (
            "Минимальная площадь белой области. Меньшие компоненты удаляются.",
            "Minimum area of a white component. Smaller components are removed.",
        ),
        "max_component_area": (
            "Максимальная площадь белой области. Ноль означает без верхнего ограничения.",
            "Maximum area of a white component. Zero means no upper limit.",
        ),
        "min_component_perimeter": (
            "Минимальная длина границы белой области. Короткие компоненты удаляются.",
            "Minimum border length of a white component. Shorter components are removed.",
        ),
        "max_component_perimeter": (
            "Максимальная длина границы белой области. Ноль означает без верхнего ограничения.",
            "Maximum border length of a white component. Zero means no upper limit.",
        ),
        "distance_ratio": (
            "Чувствительность разделения слипшихся объектов. Большее значение требует более выраженных центров объектов.",
            "Sensitivity for splitting touching objects. Higher values require more distinct object centers.",
        ),
        "min_peak_area": (
            "Минимальная площадь центра объекта при разделении watershed. Увеличьте, чтобы не дробить шум.",
            "Minimum object-center area for watershed splitting. Increase it to avoid splitting noise.",
        ),
        "background_iterations": (
            "Сколько раз расширять фон перед watershed. Больше повторов увереннее отделяет фон, но может сжать объекты.",
            "How many times to expand the background before watershed. More repeats separate background more firmly but can shrink objects.",
        ),
        "width": (
            "Ширина результата при изменении размера или ширина вырезаемой области при crop.",
            "Result width for resize, or extracted area width for crop.",
        ),
        "height": (
            "Высота результата при изменении размера или высота вырезаемой области при crop.",
            "Result height for resize, or extracted area height for crop.",
        ),
        "keep_aspect": (
            "Сохраняет пропорции изображения при изменении размера. Отключайте только если допустимо растяжение.",
            "Preserves image proportions during resize. Disable only when stretching is acceptable.",
        ),
        "interpolation": (
            "Метод пересчета пикселей при изменении размера. Влияет на резкость и сглаживание результата.",
            "Pixel resampling method for resizing. Affects sharpness and smoothness of the result.",
        ),
        "scale": (
            "Во сколько раз изменить размер изображения. Значение больше 1 увеличивает, меньше 1 уменьшает.",
            "Resize multiplier. Values above 1 enlarge the image, below 1 shrink it.",
        ),
        "x": (
            "Левая координата области, которая будет вырезана из изображения.",
            "Left coordinate of the region that will be cropped from the image.",
        ),
        "y": (
            "Верхняя координата области, которая будет вырезана из изображения.",
            "Top coordinate of the region that will be cropped from the image.",
        ),
        "amount": (
            "Сила повышения резкости. Большее значение сильнее подчеркивает края и шум.",
            "Sharpening strength. Higher values emphasize edges and noise more strongly.",
        ),
        "sigma": (
            "Ширина предварительного размытия для резкости. Большее значение делает усиление более широким и мягким.",
            "Pre-blur width for sharpening. Higher values make the enhancement wider and softer.",
        ),
        "h": (
            "Сила подавления шума. Большее значение убирает больше шума, но может сгладить полезные детали.",
            "Noise suppression strength. Higher values remove more noise but can smooth useful details.",
        ),
        "template_window_size": (
            "Размер образца, по которому denoise сравнивает текстуры. Обычно меняют редко.",
            "Sample size used by denoise to compare textures. Usually changed rarely.",
        ),
        "search_window_size": (
            "Размер области поиска похожих фрагментов для denoise. Больше область может лучше чистить шум, но работает медленнее.",
            "Search area size for similar patches in denoise. Larger areas can clean noise better but run slower.",
        ),
    }
)

PIPELINE_CONTROL_TOOLTIPS: LocalizedTextMap = {
    "add_step_button": (
        "Добавляет выбранный фильтр из списка в текущий pipeline.",
        "Adds the selected filter from the list to the current pipeline.",
    ),
    "remove_step_button": (
        "Удаляет выбранный шаг из pipeline. Остальные шаги сохраняются.",
        "Removes the selected step from the pipeline. Other steps stay unchanged.",
    ),
    "move_up_button": (
        "Перемещает выбранный фильтр выше. Порядок важен: верхние фильтры применяются раньше.",
        "Moves the selected filter up. Order matters: upper filters run earlier.",
    ),
    "move_down_button": (
        "Перемещает выбранный фильтр ниже. Нижние фильтры применяются позже.",
        "Moves the selected filter down. Lower filters run later.",
    ),
    "auto_apply_checkbox": (
        "Автоматически пересчитывает текущее изображение после изменения pipeline или его параметров.",
        "Automatically reprocesses the current image after pipeline or parameter changes.",
    ),
    "apply_current_button": (
        "Применяет текущий pipeline к открытому изображению вручную.",
        "Manually applies the current pipeline to the open image.",
    ),
    "save_json_button": (
        "Сохраняет текущий набор фильтров и их параметры в JSON-файл.",
        "Saves the current filter list and parameters to a JSON file.",
    ),
    "load_json_button": (
        "Загружает pipeline из JSON-файла и заменяет текущий список фильтров.",
        "Loads a pipeline from a JSON file and replaces the current filter list.",
    ),
    "auto_tune_button": (
        "Подбирает параметры фильтров по нарисованным полигонам, используя их как эталон результата.",
        "Tunes filter parameters using the drawn polygons as the target result.",
    ),
}

EDITOR_TOOL_TOOLTIPS: dict[EditorTool, tuple[str, str]] = {
    EditorTool.SELECT: (
        "Выбор и перемещение полигонов на изображении.",
        "Select and move polygons on the image.",
    ),
    EditorTool.SELECT_AREA: (
        "Выделение объектов рамкой. Потяните прямоугольник вокруг нужных полигонов; Ctrl добавляет к текущему выделению.",
        "Select objects with a rectangle. Drag around polygons; Ctrl adds to the current selection.",
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
        "Создание нового полигона точками или прямоугольником.",
        "Create a new polygon with points or a rectangle.",
    ),
    EditorTool.BRUSH: (
        "Рисование или стирание области кистью. Круг под курсором показывает текущую толщину кисти.",
        "Draw or erase an area with the brush. The circle under the cursor shows the current brush width.",
    ),
    EditorTool.ADD_VIA: (
        "Поставить переходное отверстие заданной ширины и высоты в месте клика.",
        "Place a via of the configured width and height at the click position.",
    ),
    EditorTool.ADD_VERTEX: (
        "Добавить вершину на ближайший участок границы выбранного полигона.",
        "Add a vertex to the nearest edge of the selected polygon.",
    ),
    EditorTool.DELETE_VERTEX: (
        "Удалить вершину выбранного полигона. Режим удаления задает, удаляется одна точка или область.",
        "Delete vertices from the selected polygon. The delete mode controls whether one point or an area is removed.",
    ),
    EditorTool.MOVE_VERTEX: (
        "Переместить отдельную вершину выбранного полигона.",
        "Move a single vertex of the selected polygon.",
    ),
    EditorTool.DELETE_POLYGON: (
        "Удалить полигон, по которому вы кликнете.",
        "Delete the polygon you click.",
    ),
}

EDITOR_ACTION_TOOLTIPS: LocalizedTextMap = {
    "undo_button": (
        "Отменяет последнее изменение полигонов.",
        "Undoes the last polygon edit.",
    ),
    "redo_button": (
        "Возвращает последнее отмененное изменение полигонов.",
        "Redoes the last undone polygon edit.",
    ),
    "zoom_in_button": (
        "Увеличивает изображение в окне просмотра.",
        "Zooms in on the image view.",
    ),
    "zoom_out_button": (
        "Уменьшает изображение в окне просмотра.",
        "Zooms out of the image view.",
    ),
    "fit_button": (
        "Подгоняет изображение целиком под размер окна просмотра.",
        "Fits the whole image into the view.",
    ),
}

GENERAL_CONTROL_TOOLTIPS: LocalizedTextMap = {
    "input_dir": (
        "Папка с изображениями, которые появятся в списке файлов.",
        "Folder with images that will appear in the file list.",
    ),
    "cif_dir": (
        "Папка с CIF-разметкой для наложения на изображения. Можно оставить пустой, если CIF не нужен.",
        "Folder with CIF annotations to overlay on images. Leave empty if CIF is not needed.",
    ),
    "output_dir": (
        "Папка, куда сохраняются результаты обработки и векторизации.",
        "Folder where processing and vectorization results are saved.",
    ),
    "dataset_dir": (
        "Папка датасета, куда экспортируется текущий кадр в режиме подготовки данных.",
        "Dataset folder where the current frame is exported during dataset preparation.",
    ),
    "browse_input": (
        "Выбрать папку с исходными изображениями.",
        "Choose the folder with source images.",
    ),
    "browse_cif": (
        "Выбрать папку с CIF-разметкой для наложения.",
        "Choose the folder with CIF annotations for overlay.",
    ),
    "browse_output": (
        "Выбрать папку для сохранения результатов.",
        "Choose the folder for saved results.",
    ),
    "browse_dataset": (
        "Выбрать папку датасета для экспорта кадров.",
        "Choose the dataset folder for frame export.",
    ),
    "refresh_files": (
        "Перечитать список изображений из выбранной входной папки.",
        "Reload the image list from the selected input folder.",
    ),
    "image_list": (
        "Список найденных изображений. Выбор файла открывает его для просмотра и обработки.",
        "List of found images. Selecting a file opens it for viewing and processing.",
    ),
    "process_current": (
        "Обработать только выбранное изображение текущими настройками.",
        "Process only the selected image with the current settings.",
    ),
    "start_batch": (
        "Запустить обработку всех изображений из списка.",
        "Start processing all images in the list.",
    ),
    "stop_batch": (
        "Остановить пакетную обработку после текущих выполняемых задач.",
        "Stop batch processing after the currently running tasks finish.",
    ),
    "save_current": (
        "Сохранить результат для текущего изображения в выходную папку.",
        "Save the current image result to the output folder.",
    ),
    "export_dataset": (
        "Экспортировать текущий кадр и разметку в папку датасета.",
        "Export the current frame and annotation to the dataset folder.",
    ),
    "dataset_mode": (
        "После сохранения помечает кадр как подготовленный для датасета и помогает не перепутать уже обработанные файлы.",
        "After saving, marks the frame as prepared for the dataset and helps distinguish already processed files.",
    ),
    "max_workers": (
        "Сколько изображений можно обрабатывать параллельно в пакетном режиме. Больше потоков быстрее, но сильнее нагружает компьютер.",
        "How many images can be processed in parallel during batch mode. More workers can be faster but load the computer more.",
    ),
    "save_svg": (
        "Сохранять SVG-файл с найденными полигонами вместе с результатом.",
        "Save an SVG file with detected polygons together with the result.",
    ),
    "save_preview": (
        "Сохранять картинку предпросмотра с наложенными полигонами.",
        "Save a preview image with polygons overlaid.",
    ),
    "external_color": (
        "Цвет внешней границы обычных полигонов на предпросмотре.",
        "Color of regular polygon outer borders in the preview.",
    ),
    "hole_color": (
        "Цвет внутренних отверстий полигонов на предпросмотре.",
        "Color of polygon inner holes in the preview.",
    ),
    "selected_color": (
        "Цвет полигона, который сейчас выбран в редакторе.",
        "Color of the polygon currently selected in the editor.",
    ),
    "vertex_color": (
        "Цвет точек-вершин, которые показываются на полигонах.",
        "Color of vertex points shown on polygons.",
    ),
    "line_width": (
        "Толщина линий полигонов на экране. Не меняет координаты и результат векторизации.",
        "Polygon line width on screen. Does not change coordinates or vectorization results.",
    ),
    "vertex_size": (
        "Размер точек-вершин на экране. Не влияет на геометрию полигонов.",
        "Size of vertex points on screen. Does not affect polygon geometry.",
    ),
    "fill_opacity": (
        "Прозрачность заливки полигонов на предпросмотре. Ноль скрывает заливку, единица делает ее непрозрачной.",
        "Polygon fill opacity in the preview. Zero hides the fill, one makes it opaque.",
    ),
    "show_vertices": (
        "Показывать точки-вершины полигонов в редакторе.",
        "Show polygon vertex points in the editor.",
    ),
    "show_labels": (
        "Показывать номера полигонов на изображении.",
        "Show polygon IDs on the image.",
    ),
    "polygon_mode": (
        "Способ создания полигона: по отдельным точкам или прямоугольником.",
        "Polygon creation method: point by point or as a rectangle.",
    ),
    "brush_mode": (
        "Режим движения кисти. Свободный рисует как ведете мышь, режим 45 градусов ограничивает направление.",
        "Brush movement mode. Freeform follows the mouse, 45-degree mode constrains direction.",
    ),
    "brush_size": (
        "Толщина кисти в пикселях изображения. Круг под курсором показывает этот размер.",
        "Brush width in image pixels. The circle under the cursor shows this size.",
    ),
    "delete_vertex_mode": (
        "Как удалять вершины: одну ближайшую точку или все точки внутри области.",
        "How vertices are deleted: one nearest point or all points inside an area.",
    ),
    "editor_via_width": (
        "Ширина via, которое ставится инструментом Via при клике по изображению.",
        "Width of the via placed by the Via tool when clicking the image.",
    ),
    "editor_via_height": (
        "Высота via, которое ставится инструментом Via при клике по изображению.",
        "Height of the via placed by the Via tool when clicking the image.",
    ),
}


PIPELINE_OPERATION_GROUPS: tuple[tuple[str, tuple[str, str], tuple[str, ...]], ...] = (
    (
        "smoothing",
        ("Сглаживание и шум", "Smoothing and noise"),
        ("gaussian_blur", "median_blur", "bilateral_filter", "denoise"),
    ),
    (
        "contrast",
        ("Контраст и тон", "Contrast and tone"),
        ("clahe", "histogram_equalization", "brightness_contrast", "gamma_correction", "sharpen"),
    ),
    (
        "thresholding",
        ("Пороговая обработка", "Thresholding"),
        ("color_binarize", "threshold", "adaptive_threshold", "otsu_threshold", "edge_guided_threshold", "invert"),
    ),
    (
        "binary",
        ("Бинарные фильтры", "Binary filters"),
        ("binary_fill_holes", "binary_filter_area", "binary_filter_perimeter", "watershed_split"),
    ),
    (
        "morphology",
        ("Морфология", "Morphology"),
        ("morph_open", "morph_close", "erode", "dilate", "gradient", "tophat", "blackhat"),
    ),
    (
        "geometry",
        ("Геометрия и границы", "Geometry and edges"),
        ("canny", "resize", "scale_resize", "crop"),
    ),
)


PIPELINE_OPERATION_HELP_TEXTS: dict[str, dict[str, tuple[str, str]]] = {
    "gaussian_blur": {
        "summary": (
            "Мягко размывает изображение и подавляет мелкий шум перед бинаризацией.",
            "Smoothly blurs the image and suppresses fine noise before thresholding.",
        ),
        "use": (
            "Используйте, когда объект читается, но края слегка шумят.",
            "Use it when the object is visible but the edges are slightly noisy.",
        ),
    },
    "median_blur": {
        "summary": (
            "Хорошо убирает одиночные выбросы и соль-перец, сохраняя границы лучше обычного blur.",
            "Removes salt-and-pepper outliers while preserving edges better than a regular blur.",
        ),
        "use": (
            "Подходит для бинарных масок и изображений с точечным шумом.",
            "Best for binary masks and images with impulse noise.",
        ),
    },
    "bilateral_filter": {
        "summary": (
            "Сглаживает внутри областей, но старается сохранить контуры.",
            "Smooths within regions while trying to preserve edges.",
        ),
        "use": (
            "Полезен, когда нужно уменьшить текстурный шум и не размыть границы проводника.",
            "Useful when you need to reduce texture noise without washing out conductor edges.",
        ),
    },
    "clahe": {
        "summary": (
            "Усиливает локальный контраст по частям изображения.",
            "Boosts local contrast in small image regions.",
        ),
        "use": (
            "Применяйте при неравномерной подсветке или слабом локальном контрасте.",
            "Apply it under uneven illumination or weak local contrast.",
        ),
    },
    "histogram_equalization": {
        "summary": (
            "Растягивает общий контраст по всему изображению.",
            "Expands overall contrast across the whole image.",
        ),
        "use": (
            "Подходит для равномерно тусклых изображений без сильной локальной засветки.",
            "Good for uniformly dull images without strong local glare.",
        ),
    },
    "brightness_contrast": {
        "summary": (
            "Линейно меняет яркость и контраст.",
            "Linearly adjusts brightness and contrast.",
        ),
        "use": (
            "Используйте для грубой подстройки перед threshold.",
            "Use it for coarse correction before thresholding.",
        ),
    },
    "gamma_correction": {
        "summary": (
            "Нелинейно перераспределяет яркости, сильнее влияя на тени и свет.",
            "Nonlinearly redistributes tones, affecting shadows and highlights differently.",
        ),
        "use": (
            "Подходит, когда нужно поднять тёмные детали или приглушить пересвет.",
            "Useful when you need to lift dark details or tame highlights.",
        ),
    },
    "threshold": {
        "summary": (
            "Преобразует изображение в маску по одному глобальному порогу.",
            "Converts the image into a mask using a single global threshold.",
        ),
        "use": (
            "Работает хорошо при стабильном фоне и понятном разделении яркостей.",
            "Works well with a stable background and clear intensity separation.",
        ),
    },
    "adaptive_threshold": {
        "summary": (
            "Строит маску по локальному порогу для каждой области изображения.",
            "Builds a mask using a local threshold for each image region.",
        ),
        "use": (
            "Используйте при градиентной подсветке и неоднородном фоне.",
            "Use it under lighting gradients and non-uniform backgrounds.",
        ),
    },
    "otsu_threshold": {
        "summary": (
            "Автоматически подбирает глобальный порог по гистограмме.",
            "Automatically chooses a global threshold from the histogram.",
        ),
        "use": (
            "Хороший стартовый вариант, когда вручную порог ещё не известен.",
            "A good starting option when you do not yet know the right manual threshold.",
        ),
    },
    "morph_open": {
        "summary": (
            "Сначала сужает, потом расширяет маску. Убирает мелкие шумовые точки.",
            "Erodes then dilates the mask. Removes small foreground specks.",
        ),
        "use": (
            "Полезно для очистки случайных точек перед поиском контуров.",
            "Useful for cleaning isolated specks before contour extraction.",
        ),
    },
    "morph_close": {
        "summary": (
            "Сначала расширяет, потом сужает маску. Закрывает мелкие разрывы и дырки.",
            "Dilates then erodes the mask. Closes small gaps and holes.",
        ),
        "use": (
            "Используйте, когда проводник рвётся или контур состоит из щелей.",
            "Use it when a conductor breaks apart or the contour has small gaps.",
        ),
    },
    "erode": {
        "summary": (
            "Сужает светлые области и убирает тонкие выступы.",
            "Shrinks bright regions and removes thin protrusions.",
        ),
        "use": (
            "Подходит для отделения слипшихся объектов и удаления утолщений.",
            "Useful for separating touching objects and trimming thick edges.",
        ),
    },
    "dilate": {
        "summary": (
            "Расширяет светлые области и укрепляет тонкие элементы.",
            "Expands bright regions and reinforces thin structures.",
        ),
        "use": (
            "Помогает восстановить разорванные линии и усилить слабые проводники.",
            "Helps reconnect broken lines and strengthen weak conductors.",
        ),
    },
    "gradient": {
        "summary": (
            "Оставляет в основном границу объекта как разность между dilate и erode.",
            "Keeps mainly the object boundary as the difference between dilate and erode.",
        ),
        "use": (
            "Используйте для выделения краёв, когда важен контур, а не заливка.",
            "Use it when edges matter more than filled regions.",
        ),
    },
    "tophat": {
        "summary": (
            "Выделяет маленькие светлые детали на более тёмном фоне.",
            "Highlights small bright details on a darker background.",
        ),
        "use": (
            "Полезен для поиска мелких светлых отверстий или точек.",
            "Useful for finding small bright vias or spots.",
        ),
    },
    "blackhat": {
        "summary": (
            "Выделяет маленькие тёмные детали на более светлом фоне.",
            "Highlights small dark details on a lighter background.",
        ),
        "use": (
            "Полезен для тёмных отверстий или канавок на светлом поле.",
            "Useful for dark vias or grooves on a bright field.",
        ),
    },
    "canny": {
        "summary": (
            "Строит карту границ по резким перепадам яркости.",
            "Builds an edge map from strong intensity changes.",
        ),
        "use": (
            "Используйте, когда объект хорошо описывается линиями границ.",
            "Use it when the object is best represented by edge lines.",
        ),
    },
    "edge_guided_threshold": {
        "summary": (
            "Строит заполненную бинарную маску и уточняет её границу по яркостным краям Sobel или Canny.",
            "Builds a filled binary mask and refines its boundary using Sobel or Canny intensity edges.",
        ),
        "use": (
            "Используйте, когда обычный порог видит объект, но граница уезжает из-за размытия или неоднородной яркости.",
            "Use it when thresholding sees the object but blur or uneven intensity shifts the boundary.",
        ),
    },
    "invert": {
        "summary": (
            "Меняет светлое и тёмное местами.",
            "Swaps bright and dark regions.",
        ),
        "use": (
            "Нужно, когда объект и фон перепутаны относительно ожидаемой бинаризации.",
            "Useful when foreground and background polarity are reversed for the expected thresholding flow.",
        ),
    },
    "resize": {
        "summary": (
            "Меняет размер изображения для нормализации масштаба объектов.",
            "Changes image size to normalize object scale.",
        ),
        "use": (
            "Полезно, если изображения приходят в разных разрешениях.",
            "Useful when images arrive at different resolutions.",
        ),
    },
    "crop": {
        "summary": (
            "Обрезает изображение до рабочей области.",
            "Crops the image to a region of interest.",
        ),
        "use": (
            "Используйте, если полезная область известна заранее и не нужно обрабатывать весь кадр.",
            "Use it when the useful region is known in advance and the full frame is unnecessary.",
        ),
    },
    "sharpen": {
        "summary": (
            "Подчёркивает локальные перепады и делает границы резче.",
            "Emphasizes local changes and sharpens edges.",
        ),
        "use": (
            "Полезно перед пороговой обработкой, если границы объекта размыты.",
            "Useful before thresholding when object edges are soft.",
        ),
    },
    "denoise": {
        "summary": (
            "Убирает шум с сохранением общей структуры изображения.",
            "Removes noise while preserving the overall image structure.",
        ),
        "use": (
            "Используйте на шумных снимках перед более агрессивными шагами pipeline.",
            "Use it on noisy captures before more aggressive pipeline steps.",
        ),
    },
}


def _localized_text(mapping: LocalizedTextMap, key: str, language: str) -> str:
    entry = mapping.get(key, ("", ""))
    return entry[0] if language == "ru" else entry[1]


class PolygonExtractionWidget(QWidget):
    imageProcessed = pyqtSignal(str, list)
    batchProgress = pyqtSignal(int, int)
    batchFinished = pyqtSignal()
    polygonsEdited = pyqtSignal()
    logMessage = pyqtSignal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("polygonExtractionWidget")
        self._ui_language = active_language()
        self._path_settings_store = WidgetPathSettingsStore()
        self._display_settings_store = WidgetDisplaySettingsStore()
        self._workspace = WorkspaceSession()
        self._pipeline = PreprocessingPipeline()
        self._display_settings = DisplaySettings()
        self._contour_settings_profiles = {
            "conductors": ContourExtractionSettings(
                extraction_profile="conductors",
                object_type="conductor",
                output_mode="polygon",
                min_polygon_angle=90.0,
            ),
            "vias": ContourExtractionSettings(
                extraction_profile="vias",
                object_type="via",
                output_mode="box",
                min_solidity=0.6,
                min_extent=0.5,
                min_aspect_ratio=0.5,
                max_aspect_ratio=2.0,
            ),
        }
        self._active_extraction_profile = "conductors"
        self._ignore_extraction_profile_change = False
        self._ignore_pipeline_item_change = False
        self._suspend_fixed_via_updates = False
        self._restoring_display_settings = False
        self._fixed_via_rows: list[dict[str, QWidget]] = []
        self._parameter_widgets: dict[str, QWidget] = {}
        self._updating_views = False
        self._batch_progress_enabled = False
        self._progress_status_key = "idle_status"
        self._progress_status_kwargs: dict[str, object] = {}
        self._preview_thread_pool = QThreadPool(self)
        self._preview_thread_pool.setMaxThreadCount(1)
        self._preview_thread_pool.setExpiryTimeout(-1)
        self._preview_update_timer = QTimer(self)
        self._preview_update_timer.setSingleShot(True)
        self._preview_update_timer.setInterval(180)
        self._preview_update_timer.timeout.connect(self._start_pending_preview_processing)
        self._preview_request_serial = 0
        self._preview_running_request_id: int | None = None
        self._preview_pending_request: PreviewProcessingRequest | None = None
        self._preview_running_signature: tuple[str, str, str] | None = None
        self._preview_pending_signature: tuple[str, str, str] | None = None
        self._help_menu: QMenu | None = None
        self._color_pick_pipeline_row: int | None = None
        self._via_template_images: list[np.ndarray] = []
        self._viewed_image_paths: set[str] = set()
        self._user_via_presets: dict[str, dict[str, object]] = self._load_user_via_presets()
        self._extra_layers: list[dict[str, object]] = []
        self._prepared_image_thread_pool = QThreadPool(self)
        self._prepared_image_thread_pool.setMaxThreadCount(1)
        self._prepared_image_thread_pool.setExpiryTimeout(-1)
        self._prepared_image_request_serial = 0
        self._prepared_image_running_request_id: int | None = None
        self._prepared_image_pending_request: PreparedImageRequest | None = None
        self._prepared_image_running_signature: tuple[str, str] | None = None
        self._prepared_image_pending_signature: tuple[str, str] | None = None
        self._auto_tune_thread_pool = QThreadPool(self)
        self._auto_tune_thread_pool.setMaxThreadCount(1)
        self._auto_tune_thread_pool.setExpiryTimeout(-1)
        self._auto_tune_request_serial = 0
        self._auto_tune_running_request_id: int | None = None
        self._neighbor_image_cache: dict[str, object] = {}

        self._batch_processor = BatchProcessor(self)
        self._batch_processor.set_ui_language(self._ui_language)
        self._batch_processor.resultReady.connect(self._on_batch_result)
        self._batch_processor.progressChanged.connect(self._on_batch_progress)
        self._batch_processor.finished.connect(self._on_batch_finished)
        self._batch_processor.errorOccurred.connect(self._on_batch_error)
        self._batch_processor.logMessage.connect(self._append_log)

        self._build_ui()
        self._apply_compact_ui_style()
        self._disable_spinbox_wheel_changes()
        self._restore_persisted_paths()
        self._restore_persisted_display_settings()
        self._populate_pipeline_operations()
        self._populate_pipeline_list()
        self._apply_display_settings()
        self._set_extraction_settings(self._contour_settings_profiles[self._active_extraction_profile])
        self.set_ui_language(self._ui_language)

    def _build_ui(self) -> None:
        root_layout = QVBoxLayout(self)

        self.main_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.main_splitter.setChildrenCollapsible(False)
        self.main_splitter.splitterMoved.connect(self._on_main_splitter_moved)
        root_layout.addWidget(self.main_splitter, 1)

        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setMinimumWidth(360)
        left_scroll.setMaximumWidth(560)
        controls_container = QWidget()
        left_scroll.setWidget(controls_container)
        controls_layout = QVBoxLayout(controls_container)
        self.control_tabs = self._build_tabs()
        controls_layout.addWidget(self.control_tabs, 1)
        self.main_splitter.addWidget(left_scroll)
        self.visual_panel = self._build_visual_panel()
        self.main_splitter.addWidget(self.visual_panel)
        self.right_tabs = QTabWidget()
        self.right_tabs.setUsesScrollButtons(True)
        self.right_tabs.setMinimumWidth(280)
        self.right_tabs.setMaximumWidth(440)
        self.files_tab = self._build_files_tab()
        self.right_tabs.addTab(self.files_tab, "Files")
        self.main_splitter.addWidget(self.right_tabs)
        self.main_splitter.setStretchFactor(0, 0)
        self.main_splitter.setStretchFactor(1, 1)
        self.main_splitter.setStretchFactor(2, 0)
        self.main_splitter.setSizes([380, 1000, 320])

    def _apply_compact_ui_style(self) -> None:
        self.setStyleSheet(
            """
            #polygonExtractionWidget {
                font-size: 12px;
            }
            #polygonExtractionWidget QLabel,
            #polygonExtractionWidget QCheckBox,
            #polygonExtractionWidget QGroupBox {
                font-size: 12px;
            }
            #polygonExtractionWidget QPushButton {
                min-height: 28px;
                padding: 4px 10px;
                font-size: 12px;
            }
            #polygonExtractionWidget QToolButton {
                padding: 2px;
            }
            #polygonExtractionWidget QLineEdit,
            #polygonExtractionWidget QComboBox,
            #polygonExtractionWidget QSpinBox,
            #polygonExtractionWidget QDoubleSpinBox {
                min-height: 26px;
                padding: 2px 6px;
                font-size: 12px;
            }
            #polygonExtractionWidget QTabBar::tab {
                min-height: 24px;
                padding: 4px 10px;
                font-size: 12px;
            }
            #polygonExtractionWidget QListWidget {
                font-size: 12px;
            }
            #polygonExtractionWidget QProgressBar {
                min-height: 18px;
                max-height: 18px;
            }
            """
        )

    def _build_path_panel(self) -> QWidget:
        self.path_group = QGroupBox("Input / Output")
        layout = QVBoxLayout(self.path_group)

        self.input_dir_edit = QLineEdit()
        self.cif_dir_edit = QLineEdit()
        self.output_dir_edit = QLineEdit()
        self.dataset_dir_edit = QLineEdit()
        self.input_dir_label = QLabel("Input directory")
        self.cif_dir_label = QLabel("CIF overlay directory")
        self.output_dir_label = QLabel("Output directory")
        self.dataset_dir_label = QLabel("Dataset directory")
        self.browse_input_button = QPushButton()
        self.browse_cif_button = QPushButton()
        self.browse_output_button = QPushButton()
        self.browse_dataset_button = QPushButton()
        self.refresh_button = QPushButton()
        folder_icon = self.style().standardIcon(QStyle.StandardPixmap.SP_DirOpenIcon)
        for button in (
            self.browse_input_button,
            self.browse_cif_button,
            self.browse_output_button,
            self.browse_dataset_button,
        ):
            self._configure_icon_only_button(button, folder_icon)
        self._configure_icon_only_button(self.refresh_button, self._refresh_files_icon())

        self.browse_input_button.clicked.connect(self._select_input_directory)
        self.browse_cif_button.clicked.connect(self._select_cif_directory)
        self.browse_output_button.clicked.connect(self._select_output_directory)
        self.browse_dataset_button.clicked.connect(self._select_dataset_directory)
        self.refresh_button.clicked.connect(self.refresh_image_list)
        self.input_dir_edit.editingFinished.connect(self._apply_input_directory_edit)
        self.cif_dir_edit.editingFinished.connect(self._apply_cif_directory_edit)
        self.output_dir_edit.editingFinished.connect(self._apply_output_directory_edit)
        self.dataset_dir_edit.editingFinished.connect(self._apply_dataset_directory_edit)

        for label, edit, button in [
            (self.input_dir_label, self.input_dir_edit, self.browse_input_button),
            (self.cif_dir_label, self.cif_dir_edit, self.browse_cif_button),
            (self.output_dir_label, self.output_dir_edit, self.browse_output_button),
            (self.dataset_dir_label, self.dataset_dir_edit, self.browse_dataset_button),
        ]:
            row = QWidget()
            row_layout = QHBoxLayout(row)
            row_layout.setContentsMargins(0, 0, 0, 0)
            row_layout.setSpacing(6)
            row_layout.addWidget(edit, 1)
            row_layout.addWidget(button)
            layout.addWidget(label)
            layout.addWidget(row)
        layout.addWidget(self.refresh_button)
        return self.path_group

    def _build_paths_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        self.path_panel = self._build_path_panel()
        layout.addWidget(self.path_panel)
        layout.addStretch(1)
        return tab

    def _build_tabs(self) -> QWidget:
        tabs = QTabWidget()
        tabs.setUsesScrollButtons(True)
        self.paths_tab = self._build_paths_tab()
        self.pipeline_tab = self._build_pipeline_tab()
        self.extraction_tab = self._build_extraction_tab()
        self.display_tab = self._build_display_tab()
        tabs.addTab(self.paths_tab, "Paths")
        tabs.addTab(self.pipeline_tab, "Pipeline")
        tabs.addTab(self.extraction_tab, "Extraction")
        tabs.addTab(self.display_tab, "Display")
        return tabs

    def _restore_persisted_paths(self) -> None:
        paths = self._path_settings_store.load()

        if paths.output_directory:
            self.set_output_directory(paths.output_directory)
        if paths.dataset_directory:
            self.set_dataset_directory(paths.dataset_directory)
        if paths.cif_directory:
            self.set_cif_directory(paths.cif_directory)
        if paths.input_directory:
            self.set_input_directory(paths.input_directory)

    def _save_persisted_paths(self) -> None:
        self._path_settings_store.save(
            PersistedPaths(
                input_directory=self.input_dir_edit.text().strip(),
                cif_directory=self.cif_dir_edit.text().strip(),
                output_directory=self.output_dir_edit.text().strip(),
                dataset_directory=self.dataset_dir_edit.text().strip(),
            )
        )

    def _restore_persisted_display_settings(self) -> None:
        payload = self._display_settings_store.load()
        self._display_settings = DisplaySettings.from_dict(payload)
        if not hasattr(self, "line_width_spin"):
            return

        blockers = [
            QSignalBlocker(self.line_width_spin),
            QSignalBlocker(self.vertex_size_spin),
            QSignalBlocker(self.fill_opacity_spin),
            QSignalBlocker(self.show_vertices_checkbox),
            QSignalBlocker(self.show_labels_checkbox),
            QSignalBlocker(self.random_object_colors_checkbox),
            QSignalBlocker(self.show_neighbor_frames_checkbox),
            QSignalBlocker(self.neighbor_columns_spin),
            QSignalBlocker(self.neighbor_max_grid_spin),
            QSignalBlocker(self.neighbor_opacity_spin),
            QSignalBlocker(self.neighbor_overlap_spin),
        ]
        self._restoring_display_settings = True
        try:
            self._update_color_button(self.external_color_button, self._display_settings.external_color)
            self._update_color_button(self.hole_color_button, self._display_settings.hole_color)
            self._update_color_button(self.selected_color_button, self._display_settings.selected_color)
            self._update_color_button(self.vertex_color_button, self._display_settings.vertex_color)
            self.line_width_spin.setValue(float(self._display_settings.line_width))
            self.vertex_size_spin.setValue(float(self._display_settings.vertex_size))
            self.fill_opacity_spin.setValue(float(self._display_settings.fill_opacity))
            self.show_vertices_checkbox.setChecked(bool(self._display_settings.show_vertices))
            self.show_labels_checkbox.setChecked(bool(self._display_settings.show_labels))
            self.random_object_colors_checkbox.setChecked(bool(payload.get("random_object_colors", False)))
            self.show_neighbor_frames_checkbox.setChecked(bool(payload.get("show_neighbor_frames", False)))
            self.neighbor_columns_spin.setValue(max(1, int(payload.get("neighbor_columns", 3))))
            self.neighbor_max_grid_spin.setValue(self._odd_neighbor_grid_size(int(payload.get("neighbor_max_grid", 7))))
            self.neighbor_opacity_spin.setValue(float(payload.get("neighbor_opacity", 0.35)))
            self.neighbor_overlap_spin.setValue(max(0, int(payload.get("neighbor_overlap_pixels", 0))))
            self._restore_main_splitter_sizes(payload.get("main_splitter_sizes"))
        finally:
            self._restoring_display_settings = False
            del blockers
        self._sync_neighbor_frames()

    def _current_display_settings_payload(self) -> dict[str, object]:
        return {
            **self._display_settings.to_dict(),
            "random_object_colors": bool(self.random_object_colors_checkbox.isChecked()),
            "show_neighbor_frames": bool(self.show_neighbor_frames_checkbox.isChecked()),
            "neighbor_columns": int(self.neighbor_columns_spin.value()),
            "neighbor_max_grid": int(self.neighbor_max_grid_spin.value()),
            "neighbor_opacity": float(self.neighbor_opacity_spin.value()),
            "neighbor_overlap_pixels": int(self.neighbor_overlap_spin.value()),
            "main_splitter_sizes": self.main_splitter.sizes() if hasattr(self, "main_splitter") else [],
        }

    def _save_persisted_display_settings(self) -> None:
        if self._restoring_display_settings or not hasattr(self, "line_width_spin"):
            return
        self._display_settings_store.save(self._current_display_settings_payload())

    def _restore_main_splitter_sizes(self, raw_sizes: object) -> None:
        if not hasattr(self, "main_splitter"):
            return
        if not isinstance(raw_sizes, (list, tuple)):
            return
        try:
            sizes = [max(1, int(value)) for value in raw_sizes]
        except (TypeError, ValueError):
            return
        if len(sizes) != self.main_splitter.count() or sum(sizes) <= 0:
            return
        self.main_splitter.setSizes(sizes)

    def _on_main_splitter_moved(self, *_args) -> None:
        self._save_persisted_display_settings()

    def _build_files_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)

        self.image_list = QListWidget()
        self.image_list.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.image_list.currentItemChanged.connect(self._on_image_item_changed)
        self.images_label = QLabel("Images")
        layout.addWidget(self.images_label)
        layout.addWidget(self.image_list, 1)

        self.run_group = QGroupBox("Run")
        run_layout = QGridLayout(self.run_group)
        self.process_current_button = QPushButton()
        self.process_current_button.clicked.connect(self.process_current_image)
        self.batch_button = QPushButton()
        self.batch_button.clicked.connect(self.start_batch_processing)
        self.stop_batch_button = QPushButton()
        self.stop_batch_button.clicked.connect(self.stop_batch_processing)
        self._configure_icon_only_button(
            self.process_current_button,
            self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPlay),
        )
        self._configure_icon_only_button(
            self.batch_button,
            self.style().standardIcon(QStyle.StandardPixmap.SP_MediaSeekForward),
        )
        self._configure_icon_only_button(
            self.stop_batch_button,
            self.style().standardIcon(QStyle.StandardPixmap.SP_MediaStop),
        )
        self.save_current_button = QPushButton("Save current result")
        self.save_current_button.clicked.connect(self.save_current_result)
        self.export_dataset_button = QPushButton("Export frame to dataset")
        self.export_dataset_button.clicked.connect(self.export_current_frame_to_dataset)
        self.dataset_mode_checkbox = QCheckBox("Dataset mode")
        self.max_workers_spin = QSpinBox()
        self.max_workers_spin.setRange(1, 32)
        self.max_workers_spin.setValue(4)
        self.max_workers_label = QLabel("Max workers")
        run_buttons_row = QWidget()
        run_buttons_layout = QHBoxLayout(run_buttons_row)
        run_buttons_layout.setContentsMargins(0, 0, 0, 0)
        run_buttons_layout.setSpacing(6)
        run_buttons_layout.addWidget(self.process_current_button)
        run_buttons_layout.addWidget(self.batch_button)
        run_buttons_layout.addWidget(self.stop_batch_button)
        run_buttons_layout.addStretch(1)
        run_layout.addWidget(run_buttons_row, 0, 0, 1, 2)
        run_layout.addWidget(self.max_workers_label, 1, 0)
        run_layout.addWidget(self.max_workers_spin, 1, 1)
        run_layout.addWidget(self.save_current_button, 2, 0, 1, 2)
        run_layout.addWidget(self.export_dataset_button, 3, 0, 1, 2)
        run_layout.addWidget(self.dataset_mode_checkbox, 4, 0, 1, 2)
        layout.addWidget(self.run_group)
        self.batch_progress_bar = QProgressBar()
        self.batch_progress_bar.setRange(0, 100)
        self.batch_progress_bar.setValue(0)
        self.batch_progress_bar.setFormat("%p% (%v/%m)")
        self.batch_progress_bar.setTextVisible(True)
        self.batch_progress_bar.setVisible(False)
        layout.addWidget(self.batch_progress_bar)
        return tab

    def _build_pipeline_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)

        self.available_filters_group = QGroupBox("Available filters")
        available_layout = QVBoxLayout(self.available_filters_group)
        self.operation_tree = QTreeWidget()
        self.operation_tree.setHeaderHidden(True)
        self.operation_tree.setRootIsDecorated(True)
        self.operation_tree.setUniformRowHeights(True)
        self.operation_tree.currentItemChanged.connect(self._on_available_operation_selected)
        self.operation_tree.itemDoubleClicked.connect(self._on_available_operation_activated)
        self.operation_tree.setMinimumHeight(180)
        available_layout.addWidget(self.operation_tree, 1)

        self.parameters_group = QGroupBox("Step parameters")
        parameters_scroll = QScrollArea()
        parameters_scroll.setWidgetResizable(True)
        parameters_scroll.setMinimumHeight(170)
        parameters_widget = QWidget()
        self.parameters_form = QFormLayout(parameters_widget)
        self._configure_compact_form(self.parameters_form)
        parameters_scroll.setWidget(parameters_widget)
        group_layout = QVBoxLayout(self.parameters_group)
        group_layout.addWidget(parameters_scroll)

        self.pipeline_steps_group = QGroupBox("Applied filters")
        steps_layout = QVBoxLayout(self.pipeline_steps_group)

        self.pipeline_list = PipelineListWidget()
        self.pipeline_list.currentRowChanged.connect(self._on_pipeline_step_selected)
        self.pipeline_list.itemChanged.connect(self._on_pipeline_item_changed)
        self.pipeline_list.deletePressed.connect(self._remove_pipeline_step)
        self.pipeline_list.orderChanged.connect(self._sync_pipeline_order_from_list)
        self.pipeline_list.setMinimumHeight(180)
        steps_layout.addWidget(self.pipeline_list, 1)

        apply_row = QWidget()
        apply_layout = QGridLayout(apply_row)
        apply_layout.setContentsMargins(0, 0, 0, 0)
        self.auto_apply_checkbox = QCheckBox("Auto apply")
        self.auto_apply_checkbox.setChecked(True)
        self.auto_apply_checkbox.hide()
        self.save_pipeline_button = QPushButton("Save JSON")
        self.save_pipeline_button.clicked.connect(self._save_pipeline_json)
        self.load_pipeline_button = QPushButton("Load JSON")
        self.load_pipeline_button.clicked.connect(self._load_pipeline_json)
        self.auto_tune_button = QPushButton("Auto-fit from drawing")
        self.auto_tune_button.clicked.connect(self._start_auto_tune_from_reference)
        self.auto_tune_button.setToolTip("Tunes filter parameters using the drawn polygons as the target result")
        apply_layout.addWidget(self.save_pipeline_button, 0, 0)
        apply_layout.addWidget(self.load_pipeline_button, 0, 1)
        apply_layout.addWidget(self.auto_tune_button, 1, 0, 1, 2)
        apply_layout.setColumnStretch(0, 1)
        apply_layout.setColumnStretch(1, 1)
        steps_layout.addWidget(apply_row)

        self.pipeline_help_group = QGroupBox("Filter help")
        help_layout = QVBoxLayout(self.pipeline_help_group)
        self.pipeline_help_title = QLabel()
        self.pipeline_help_title.setWordWrap(True)
        self.pipeline_help_title.setFixedHeight(28)
        self.pipeline_help_summary = QLabel()
        self.pipeline_help_summary.setWordWrap(True)
        self.pipeline_help_summary.setFixedHeight(58)
        self.pipeline_help_use = QLabel()
        self.pipeline_help_use.setWordWrap(True)
        self.pipeline_help_use.setFixedHeight(74)
        preview_row = QWidget()
        preview_layout = QHBoxLayout(preview_row)
        preview_layout.setContentsMargins(0, 0, 0, 0)
        before_column = QVBoxLayout()
        self.pipeline_help_before_title = QLabel("Before")
        self.pipeline_help_before_image = QLabel()
        self.pipeline_help_before_image.setFixedSize(190, 132)
        self.pipeline_help_before_image.setAlignment(Qt.AlignmentFlag.AlignCenter)
        before_column.addWidget(self.pipeline_help_before_title)
        before_column.addWidget(self.pipeline_help_before_image)
        after_column = QVBoxLayout()
        self.pipeline_help_after_title = QLabel("After")
        self.pipeline_help_after_image = QLabel()
        self.pipeline_help_after_image.setFixedSize(190, 132)
        self.pipeline_help_after_image.setAlignment(Qt.AlignmentFlag.AlignCenter)
        after_column.addWidget(self.pipeline_help_after_title)
        after_column.addWidget(self.pipeline_help_after_image)
        preview_layout.addLayout(before_column)
        preview_layout.addLayout(after_column)
        help_layout.addWidget(self.pipeline_help_title)
        help_layout.addWidget(self.pipeline_help_summary)
        help_layout.addWidget(self.pipeline_help_use)
        help_layout.addWidget(preview_row)

        layout.addWidget(self.available_filters_group)
        layout.addWidget(self.parameters_group)
        layout.addWidget(self.pipeline_steps_group)
        layout.addWidget(self.pipeline_help_group)
        layout.addStretch(1)
        return tab

    def _build_extraction_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        container = QWidget()
        container_layout = QVBoxLayout(container)
        container_layout.setContentsMargins(0, 0, 0, 0)

        self.contour_group = QGroupBox("Contour extraction")
        contour_layout = QVBoxLayout(self.contour_group)
        self.profile_group = QGroupBox("Profile")
        self.profile_form = QFormLayout(self.profile_group)
        self._configure_compact_form(self.profile_form)
        self.extraction_profile_combo = QComboBox()
        self.extraction_profile_combo.addItem("Conductors", "conductors")
        self.extraction_profile_combo.addItem("Vias", "vias")
        self.profile_form.addRow("Extraction profile", self.extraction_profile_combo)
        self.extraction_profile_label_widget = self.profile_form.labelForField(self.extraction_profile_combo)

        self.basic_filters_group = QGroupBox("Basic filters")
        self.basic_filters_form = QFormLayout(self.basic_filters_group)
        self._configure_compact_form(self.basic_filters_form)
        self.retrieval_mode_combo = QComboBox()
        for mode_name in RETRIEVAL_MODE_MAP:
            self.retrieval_mode_combo.addItem(mode_name, mode_name)
        self.retrieval_mode_combo.setCurrentIndex(self.retrieval_mode_combo.findData("RETR_EXTERNAL"))
        self.approximation_mode_combo = QComboBox()
        for mode_name in APPROXIMATION_MODE_MAP:
            self.approximation_mode_combo.addItem(mode_name, mode_name)
        self.approximation_mode_combo.setCurrentIndex(self.approximation_mode_combo.findData("CHAIN_APPROX_SIMPLE"))
        self.epsilon_spin = QDoubleSpinBox()
        self.epsilon_spin.setRange(0.0, 1000.0)
        self.epsilon_spin.setDecimals(3)
        self.epsilon_spin.setValue(2.0)
        self.epsilon_relative_checkbox = QCheckBox("Relative to contour perimeter")
        self.min_area_spin = QDoubleSpinBox()
        self.min_area_spin.setRange(0.0, 1_000_000_000.0)
        self.min_area_spin.setValue(10.0)
        self.max_area_spin = QDoubleSpinBox()
        self.max_area_spin.setRange(0.0, 1_000_000_000.0)
        self.max_area_spin.setValue(0.0)
        self.min_perimeter_spin = QDoubleSpinBox()
        self.min_perimeter_spin.setRange(0.0, 1_000_000_000.0)
        self.min_perimeter_spin.setValue(10.0)
        self.max_perimeter_spin = QDoubleSpinBox()
        self.max_perimeter_spin.setRange(0.0, 1_000_000_000.0)
        self.max_perimeter_spin.setValue(0.0)
        self.area_range_widget = self._build_range_row(self.min_area_spin, self.max_area_spin)
        self.perimeter_range_widget = self._build_range_row(self.min_perimeter_spin, self.max_perimeter_spin)
        self.min_points_spin = QSpinBox()
        self.min_points_spin.setRange(3, 10_000)
        self.min_points_spin.setValue(3)

        self.geometry_filters_group = QGroupBox("Geometry filters")
        self.geometry_filters_form = QFormLayout(self.geometry_filters_group)
        self._configure_compact_form(self.geometry_filters_form)
        self.min_bbox_width_spin = QSpinBox()
        self.min_bbox_width_spin.setRange(0, 100_000)
        self.min_bbox_width_spin.setValue(0)
        self.max_bbox_width_spin = QSpinBox()
        self.max_bbox_width_spin.setRange(0, 100_000)
        self.max_bbox_width_spin.setValue(0)
        self.min_bbox_height_spin = QSpinBox()
        self.min_bbox_height_spin.setRange(0, 100_000)
        self.min_bbox_height_spin.setValue(0)
        self.max_bbox_height_spin = QSpinBox()
        self.max_bbox_height_spin.setRange(0, 100_000)
        self.max_bbox_height_spin.setValue(0)
        self.min_aspect_ratio_spin = QDoubleSpinBox()
        self.min_aspect_ratio_spin.setRange(0.0, 1_000.0)
        self.min_aspect_ratio_spin.setDecimals(3)
        self.min_aspect_ratio_spin.setSingleStep(0.05)
        self.min_aspect_ratio_spin.setValue(0.0)
        self.max_aspect_ratio_spin = QDoubleSpinBox()
        self.max_aspect_ratio_spin.setRange(0.0, 1_000.0)
        self.max_aspect_ratio_spin.setDecimals(3)
        self.max_aspect_ratio_spin.setSingleStep(0.05)
        self.max_aspect_ratio_spin.setValue(0.0)
        self.bbox_width_range_widget = self._build_range_row(self.min_bbox_width_spin, self.max_bbox_width_spin)
        self.bbox_height_range_widget = self._build_range_row(self.min_bbox_height_spin, self.max_bbox_height_spin)
        self.aspect_ratio_range_widget = self._build_range_row(self.min_aspect_ratio_spin, self.max_aspect_ratio_spin)
        self.exclude_border_touching_checkbox = QCheckBox("Exclude")
        self.min_solidity_spin = QDoubleSpinBox()
        self.min_solidity_spin.setRange(0.0, 1.0)
        self.min_solidity_spin.setDecimals(3)
        self.min_solidity_spin.setSingleStep(0.05)
        self.min_solidity_spin.setValue(0.0)
        self.min_extent_spin = QDoubleSpinBox()
        self.min_extent_spin.setRange(0.0, 1.0)
        self.min_extent_spin.setDecimals(3)
        self.min_extent_spin.setSingleStep(0.05)
        self.min_extent_spin.setValue(0.0)
        self.min_polygon_angle_spin = QDoubleSpinBox()
        self.min_polygon_angle_spin.setRange(0.0, 180.0)
        self.min_polygon_angle_spin.setDecimals(1)
        self.min_polygon_angle_spin.setSingleStep(5.0)
        self.min_polygon_angle_spin.setValue(90.0)

        self.conductor_group = QGroupBox("Conductor gradient")
        self.conductor_form = QFormLayout(self.conductor_group)
        self._configure_compact_form(self.conductor_form)
        self.conductor_gradient_checkbox = QCheckBox("Enabled")
        self.conductor_gradient_min_strength_spin = QDoubleSpinBox()
        self.conductor_gradient_min_strength_spin.setRange(0.0, 255.0)
        self.conductor_gradient_min_strength_spin.setDecimals(1)
        self.conductor_gradient_min_strength_spin.setSingleStep(1.0)
        self.conductor_gradient_min_strength_spin.setValue(18.0)
        self.conductor_gradient_band_radius_spin = QSpinBox()
        self.conductor_gradient_band_radius_spin.setRange(0, 25)
        self.conductor_gradient_band_radius_spin.setValue(3)

        self.via_group = QGroupBox("Via constraints")
        self.via_form = QFormLayout(self.via_group)
        self._configure_compact_form(self.via_form)
        self.via_size_mode_combo = QComboBox()
        self.via_size_mode_combo.addItem("Range", VIA_SIZE_MODE_RANGE)
        self.via_size_mode_combo.addItem("Fixed values", VIA_SIZE_MODE_FIXED)
        self.via_white_range_checkbox = QCheckBox("White range")
        self.via_white_range_checkbox.setChecked(True)
        self.via_white_range_min_spin = QSpinBox()
        self.via_white_range_min_spin.setRange(0, 255)
        self.via_white_range_min_spin.setValue(200)
        self.via_white_range_max_spin = QSpinBox()
        self.via_white_range_max_spin.setRange(0, 255)
        self.via_white_range_max_spin.setValue(255)
        self.via_white_range_widget = self._build_range_row(self.via_white_range_min_spin, self.via_white_range_max_spin)
        self.via_black_range_checkbox = QCheckBox("Black range")
        self.via_black_range_min_spin = QSpinBox()
        self.via_black_range_min_spin.setRange(0, 255)
        self.via_black_range_min_spin.setValue(0)
        self.via_black_range_max_spin = QSpinBox()
        self.via_black_range_max_spin.setRange(0, 255)
        self.via_black_range_max_spin.setValue(30)
        self.via_black_range_widget = self._build_range_row(self.via_black_range_min_spin, self.via_black_range_max_spin)
        self.via_detector_methods_widget = QWidget()
        self.via_detector_methods_layout = QGridLayout(self.via_detector_methods_widget)
        self.via_detector_methods_layout.setContentsMargins(0, 0, 0, 0)
        self.via_detector_methods_layout.setHorizontalSpacing(10)
        self.via_detector_methods_layout.setVerticalSpacing(4)
        self.via_detector_gradient_checkbox = QCheckBox("Gradient")
        self.via_detector_gradient_checkbox.setChecked(True)
        self.via_detector_spot_checkbox = QCheckBox("Spots")
        self.via_detector_spot_checkbox.setChecked(True)
        self.via_detector_hough_checkbox = QCheckBox("Hough")
        self.via_detector_hough_checkbox.setChecked(True)
        self.via_detector_components_checkbox = QCheckBox("Components")
        self.via_detector_components_checkbox.setChecked(True)
        self.via_detector_contours_checkbox = QCheckBox("Contours")
        self.via_detector_contours_checkbox.setChecked(True)
        self.via_detector_morphology_checkbox = QCheckBox("Morphology")
        self.via_detector_morphology_checkbox.setChecked(True)
        self.via_detector_template_checkbox = QCheckBox("Template")
        self.via_detector_template_checkbox.setChecked(True)
        self.via_detector_blob_checkbox = QCheckBox("Blob")
        self.via_detector_blob_checkbox.setChecked(True)
        for row_index, (left_checkbox, right_checkbox) in enumerate(
            [
                (self.via_white_range_checkbox, self.via_black_range_checkbox),
                (self.via_detector_gradient_checkbox, self.via_detector_spot_checkbox),
                (self.via_detector_hough_checkbox, self.via_detector_components_checkbox),
                (self.via_detector_contours_checkbox, self.via_detector_morphology_checkbox),
                (self.via_detector_template_checkbox, self.via_detector_blob_checkbox),
            ]
        ):
            self.via_detector_methods_layout.addWidget(left_checkbox, row_index, 0)
            if right_checkbox is not None:
                self.via_detector_methods_layout.addWidget(right_checkbox, row_index, 1)
        self.via_spot_min_contrast_spin = QDoubleSpinBox()
        self.via_spot_min_contrast_spin.setRange(0.0, 255.0)
        self.via_spot_min_contrast_spin.setDecimals(1)
        self.via_spot_min_contrast_spin.setSingleStep(1.0)
        self.via_spot_min_contrast_spin.setValue(18.0)
        self.via_spot_min_roundness_spin = QDoubleSpinBox()
        self.via_spot_min_roundness_spin.setRange(0.0, 100.0)
        self.via_spot_min_roundness_spin.setDecimals(1)
        self.via_spot_min_roundness_spin.setSingleStep(1.0)
        self.via_spot_min_roundness_spin.setValue(45.0)
        self.via_spot_line_suppression_spin = QDoubleSpinBox()
        self.via_spot_line_suppression_spin.setRange(0.0, 1.0)
        self.via_spot_line_suppression_spin.setDecimals(2)
        self.via_spot_line_suppression_spin.setSingleStep(0.05)
        self.via_spot_line_suppression_spin.setValue(0.65)
        self.via_gradient_min_strength_spin = QDoubleSpinBox()
        self.via_gradient_min_strength_spin.setRange(0.0, 255.0)
        self.via_gradient_min_strength_spin.setDecimals(1)
        self.via_gradient_min_strength_spin.setSingleStep(1.0)
        self.via_gradient_min_strength_spin.setValue(12.0)
        self.via_gradient_min_coverage_spin = QDoubleSpinBox()
        self.via_gradient_min_coverage_spin.setRange(0.0, 1.0)
        self.via_gradient_min_coverage_spin.setDecimals(3)
        self.via_gradient_min_coverage_spin.setSingleStep(0.01)
        self.via_gradient_min_coverage_spin.setValue(0.24)
        self.via_hough_edge_threshold_spin = QDoubleSpinBox()
        self.via_hough_edge_threshold_spin.setRange(1.0, 1000.0)
        self.via_hough_edge_threshold_spin.setDecimals(1)
        self.via_hough_edge_threshold_spin.setSingleStep(5.0)
        self.via_hough_edge_threshold_spin.setValue(80.0)
        self.via_hough_accumulator_threshold_spin = QDoubleSpinBox()
        self.via_hough_accumulator_threshold_spin.setRange(1.0, 1000.0)
        self.via_hough_accumulator_threshold_spin.setDecimals(1)
        self.via_hough_accumulator_threshold_spin.setSingleStep(1.0)
        self.via_hough_accumulator_threshold_spin.setValue(10.0)
        self.via_component_min_score_spin = QDoubleSpinBox()
        self.via_component_min_score_spin.setRange(0.0, 255.0)
        self.via_component_min_score_spin.setDecimals(1)
        self.via_component_min_score_spin.setSingleStep(1.0)
        self.via_component_min_score_spin.setValue(0.0)
        self.via_contour_min_score_spin = QDoubleSpinBox()
        self.via_contour_min_score_spin.setRange(0.0, 255.0)
        self.via_contour_min_score_spin.setDecimals(1)
        self.via_contour_min_score_spin.setSingleStep(1.0)
        self.via_contour_min_score_spin.setValue(0.0)
        self.via_morphology_peak_scale_spin = QDoubleSpinBox()
        self.via_morphology_peak_scale_spin.setRange(0.01, 2.0)
        self.via_morphology_peak_scale_spin.setDecimals(3)
        self.via_morphology_peak_scale_spin.setSingleStep(0.01)
        self.via_morphology_peak_scale_spin.setValue(0.18)
        self.via_template_min_score_spin = QDoubleSpinBox()
        self.via_template_min_score_spin.setRange(0.0, 1.0)
        self.via_template_min_score_spin.setDecimals(3)
        self.via_template_min_score_spin.setSingleStep(0.01)
        self.via_template_min_score_spin.setValue(0.35)
        self.via_templates_widget = QWidget()
        self.via_templates_layout = QVBoxLayout(self.via_templates_widget)
        self.via_templates_layout.setContentsMargins(0, 0, 0, 0)
        self.via_templates_layout.setSpacing(6)
        self.via_template_list = QListWidget()
        self.via_template_list.setMaximumHeight(96)
        self.via_template_list.setIconSize(QSize(56, 56))
        self.via_templates_layout.addWidget(self.via_template_list)
        via_template_buttons = QWidget()
        via_template_buttons_layout = QHBoxLayout(via_template_buttons)
        via_template_buttons_layout.setContentsMargins(0, 0, 0, 0)
        self.add_via_template_button = QPushButton("Pick template")
        self.add_via_template_button.setCheckable(True)
        self.remove_via_template_button = QPushButton("Remove selected")
        self.clear_via_templates_button = QPushButton("Clear templates")
        via_template_buttons_layout.addWidget(self.add_via_template_button)
        via_template_buttons_layout.addWidget(self.remove_via_template_button)
        via_template_buttons_layout.addWidget(self.clear_via_templates_button)
        self.via_templates_layout.addWidget(via_template_buttons)
        self.via_blob_min_circularity_spin = QDoubleSpinBox()
        self.via_blob_min_circularity_spin.setRange(0.0, 1.0)
        self.via_blob_min_circularity_spin.setDecimals(3)
        self.via_blob_min_circularity_spin.setSingleStep(0.01)
        self.via_blob_min_circularity_spin.setValue(0.35)
        self.via_preset_combo = QComboBox()
        self.apply_via_preset_button = QPushButton("Apply preset")
        self.save_via_preset_button = QPushButton("Save preset")
        self.delete_via_preset_button = QPushButton("Delete preset")
        self.via_preset_widget = QWidget()
        via_preset_layout = QGridLayout(self.via_preset_widget)
        via_preset_layout.setContentsMargins(0, 0, 0, 0)
        via_preset_layout.setHorizontalSpacing(6)
        via_preset_layout.setVerticalSpacing(6)
        via_preset_layout.addWidget(self.via_preset_combo, 0, 0, 1, 3)
        via_preset_layout.addWidget(self.apply_via_preset_button, 1, 0)
        via_preset_layout.addWidget(self.save_via_preset_button, 1, 1)
        via_preset_layout.addWidget(self.delete_via_preset_button, 1, 2)
        self._refresh_via_preset_combo()
        self.noisy_traces_via_preset_button = QPushButton("Noisy traces preset")
        self.blurred_via_preset_button = QPushButton("Blurred vias preset")
        self.reset_via_search_button = QPushButton("Reset via search")
        self.debug_candidates_checkbox = QCheckBox("Debug recognition")
        self.via_roundness_spin = QDoubleSpinBox()
        self.via_roundness_spin.setRange(0.0, 100.0)
        self.via_roundness_spin.setDecimals(1)
        self.via_roundness_spin.setSingleStep(1.0)
        self.via_roundness_spin.setValue(5.0)
        self.min_via_width_spin = QSpinBox()
        self.min_via_width_spin.setRange(0, 100_000)
        self.min_via_width_spin.setValue(0)
        self.max_via_width_spin = QSpinBox()
        self.max_via_width_spin.setRange(0, 100_000)
        self.max_via_width_spin.setValue(0)
        self.min_via_height_spin = QSpinBox()
        self.min_via_height_spin.setRange(0, 100_000)
        self.min_via_height_spin.setValue(0)
        self.max_via_height_spin = QSpinBox()
        self.max_via_height_spin.setRange(0, 100_000)
        self.max_via_height_spin.setValue(0)
        self.via_width_range_widget = self._build_range_row(self.min_via_width_spin, self.max_via_width_spin)
        self.via_height_range_widget = self._build_range_row(self.min_via_height_spin, self.max_via_height_spin)
        self.fixed_vias_widget = QWidget()
        self.fixed_vias_widget.setObjectName("fixedViaArea")
        self.fixed_vias_layout = QVBoxLayout(self.fixed_vias_widget)
        self.fixed_vias_layout.setContentsMargins(10, 10, 10, 10)
        self.fixed_vias_layout.setSpacing(8)
        self.fixed_vias_widget.setStyleSheet(
            "#fixedViaArea { background-color: rgba(255, 255, 255, 0.03); border: 1px solid rgba(255, 255, 255, 0.10); border-radius: 8px; }"
            "#fixedViaArea QLabel, #fixedViaArea QSpinBox, #fixedViaArea QPushButton { border: none; background: transparent; }"
        )
        self.fixed_via_rows_widget = QWidget()
        self.fixed_via_rows_layout = QVBoxLayout(self.fixed_via_rows_widget)
        self.fixed_via_rows_layout.setContentsMargins(0, 0, 0, 0)
        self.fixed_via_rows_layout.setSpacing(6)
        self.fixed_vias_layout.addWidget(self.fixed_via_rows_widget)
        self.fixed_via_add_button = QPushButton("+")
        self.fixed_via_add_button.setMinimumHeight(38)
        self.fixed_via_add_button.setStyleSheet(
            "QPushButton { background-color: #2fbf71; color: white; font-size: 22px; font-weight: 700; border-radius: 8px; }"
            "QPushButton:hover { background-color: #28a764; }"
            "QPushButton:pressed { background-color: #229157; }"
        )
        self.fixed_via_add_button.clicked.connect(self._add_fixed_via_row)
        self.fixed_vias_layout.addWidget(self.fixed_via_add_button)

        self.topology_group = QGroupBox("Hierarchy and holes")
        self.topology_form = QFormLayout(self.topology_group)
        self._configure_compact_form(self.topology_form)
        self.min_hierarchy_depth_spin = QSpinBox()
        self.min_hierarchy_depth_spin.setRange(0, 100)
        self.min_hierarchy_depth_spin.setValue(0)
        self.max_hierarchy_depth_spin = QSpinBox()
        self.max_hierarchy_depth_spin.setRange(0, 100)
        self.max_hierarchy_depth_spin.setValue(0)
        self.max_hole_area_ratio_spin = QDoubleSpinBox()
        self.max_hole_area_ratio_spin.setRange(0.0, 10.0)
        self.max_hole_area_ratio_spin.setDecimals(3)
        self.max_hole_area_ratio_spin.setSingleStep(0.05)
        self.max_hole_area_ratio_spin.setValue(0.0)
        self.extraction_profile_combo.currentIndexChanged.connect(self._on_extraction_profile_changed)
        self.retrieval_mode_combo.currentIndexChanged.connect(self._on_extraction_settings_changed)
        self.approximation_mode_combo.currentIndexChanged.connect(self._on_extraction_settings_changed)
        self.epsilon_spin.valueChanged.connect(self._on_extraction_settings_changed)
        self.epsilon_relative_checkbox.stateChanged.connect(self._on_extraction_settings_changed)
        self.min_area_spin.valueChanged.connect(self._on_extraction_settings_changed)
        self.max_area_spin.valueChanged.connect(self._on_extraction_settings_changed)
        self.min_perimeter_spin.valueChanged.connect(self._on_extraction_settings_changed)
        self.min_points_spin.valueChanged.connect(self._on_extraction_settings_changed)
        self.max_perimeter_spin.valueChanged.connect(self._on_extraction_settings_changed)
        self.min_bbox_width_spin.valueChanged.connect(self._on_extraction_settings_changed)
        self.max_bbox_width_spin.valueChanged.connect(self._on_extraction_settings_changed)
        self.min_bbox_height_spin.valueChanged.connect(self._on_extraction_settings_changed)
        self.max_bbox_height_spin.valueChanged.connect(self._on_extraction_settings_changed)
        self.min_aspect_ratio_spin.valueChanged.connect(self._on_extraction_settings_changed)
        self.max_aspect_ratio_spin.valueChanged.connect(self._on_extraction_settings_changed)
        self.exclude_border_touching_checkbox.stateChanged.connect(self._on_extraction_settings_changed)
        self.min_solidity_spin.valueChanged.connect(self._on_extraction_settings_changed)
        self.min_extent_spin.valueChanged.connect(self._on_extraction_settings_changed)
        self.min_polygon_angle_spin.valueChanged.connect(self._on_extraction_settings_changed)
        self.conductor_gradient_checkbox.stateChanged.connect(self._on_extraction_settings_changed)
        self.conductor_gradient_min_strength_spin.valueChanged.connect(self._on_extraction_settings_changed)
        self.conductor_gradient_band_radius_spin.valueChanged.connect(self._on_extraction_settings_changed)
        self.via_size_mode_combo.currentIndexChanged.connect(self._on_via_size_mode_changed)
        self.via_white_range_checkbox.stateChanged.connect(self._on_extraction_settings_changed)
        self.via_white_range_min_spin.valueChanged.connect(self._on_extraction_settings_changed)
        self.via_white_range_max_spin.valueChanged.connect(self._on_extraction_settings_changed)
        self.via_black_range_checkbox.stateChanged.connect(self._on_extraction_settings_changed)
        self.via_black_range_min_spin.valueChanged.connect(self._on_extraction_settings_changed)
        self.via_black_range_max_spin.valueChanged.connect(self._on_extraction_settings_changed)
        self.via_detector_gradient_checkbox.stateChanged.connect(self._on_extraction_settings_changed)
        self.via_detector_spot_checkbox.stateChanged.connect(self._on_extraction_settings_changed)
        self.via_detector_hough_checkbox.stateChanged.connect(self._on_extraction_settings_changed)
        self.via_detector_components_checkbox.stateChanged.connect(self._on_extraction_settings_changed)
        self.via_detector_contours_checkbox.stateChanged.connect(self._on_extraction_settings_changed)
        self.via_detector_morphology_checkbox.stateChanged.connect(self._on_extraction_settings_changed)
        self.via_detector_template_checkbox.stateChanged.connect(self._on_extraction_settings_changed)
        self.via_detector_blob_checkbox.stateChanged.connect(self._on_extraction_settings_changed)
        self.via_spot_min_contrast_spin.valueChanged.connect(self._on_extraction_settings_changed)
        self.via_spot_min_roundness_spin.valueChanged.connect(self._on_extraction_settings_changed)
        self.via_spot_line_suppression_spin.valueChanged.connect(self._on_extraction_settings_changed)
        self.via_gradient_min_strength_spin.valueChanged.connect(self._on_extraction_settings_changed)
        self.via_gradient_min_coverage_spin.valueChanged.connect(self._on_extraction_settings_changed)
        self.via_hough_edge_threshold_spin.valueChanged.connect(self._on_extraction_settings_changed)
        self.via_hough_accumulator_threshold_spin.valueChanged.connect(self._on_extraction_settings_changed)
        self.via_component_min_score_spin.valueChanged.connect(self._on_extraction_settings_changed)
        self.via_contour_min_score_spin.valueChanged.connect(self._on_extraction_settings_changed)
        self.via_morphology_peak_scale_spin.valueChanged.connect(self._on_extraction_settings_changed)
        self.via_template_min_score_spin.valueChanged.connect(self._on_extraction_settings_changed)
        self.via_blob_min_circularity_spin.valueChanged.connect(self._on_extraction_settings_changed)
        self.add_via_template_button.toggled.connect(self._set_via_template_pick_active)
        self.remove_via_template_button.clicked.connect(self._remove_selected_via_template)
        self.clear_via_templates_button.clicked.connect(self._clear_via_templates)
        self.apply_via_preset_button.clicked.connect(self._apply_selected_via_preset)
        self.save_via_preset_button.clicked.connect(self._save_current_via_preset)
        self.delete_via_preset_button.clicked.connect(self._delete_selected_via_preset)
        self.noisy_traces_via_preset_button.clicked.connect(self._apply_noisy_traces_via_preset)
        self.blurred_via_preset_button.clicked.connect(self._apply_blurred_via_preset)
        self.reset_via_search_button.clicked.connect(self._reset_via_search_parameters)
        self.debug_candidates_checkbox.stateChanged.connect(self._on_extraction_settings_changed)
        self.via_roundness_spin.valueChanged.connect(self._on_extraction_settings_changed)
        self.min_via_width_spin.valueChanged.connect(self._on_extraction_settings_changed)
        self.max_via_width_spin.valueChanged.connect(self._on_extraction_settings_changed)
        self.min_via_height_spin.valueChanged.connect(self._on_extraction_settings_changed)
        self.max_via_height_spin.valueChanged.connect(self._on_extraction_settings_changed)
        self.min_hierarchy_depth_spin.valueChanged.connect(self._on_extraction_settings_changed)
        self.max_hierarchy_depth_spin.valueChanged.connect(self._on_extraction_settings_changed)
        self.max_hole_area_ratio_spin.valueChanged.connect(self._on_extraction_settings_changed)

        self.basic_filters_form.addRow("Retrieval mode", self.retrieval_mode_combo)
        self.retrieval_mode_label_widget = self.basic_filters_form.labelForField(self.retrieval_mode_combo)
        self.basic_filters_form.addRow("Approximation mode", self.approximation_mode_combo)
        self.approximation_mode_label_widget = self.basic_filters_form.labelForField(self.approximation_mode_combo)
        self.basic_filters_form.addRow("Epsilon", self.epsilon_spin)
        self.epsilon_label_widget = self.basic_filters_form.labelForField(self.epsilon_spin)
        self.basic_filters_form.addRow("Epsilon mode", self.epsilon_relative_checkbox)
        self.epsilon_mode_label_widget = self.basic_filters_form.labelForField(self.epsilon_relative_checkbox)
        self.basic_filters_form.addRow("Area range", self.area_range_widget)
        self.min_area_label_widget = self.basic_filters_form.labelForField(self.area_range_widget)
        self.max_area_label_widget = None
        self.basic_filters_form.addRow("Perimeter range", self.perimeter_range_widget)
        self.min_perimeter_label_widget = self.basic_filters_form.labelForField(self.perimeter_range_widget)
        self.max_perimeter_label_widget = None
        self.basic_filters_form.addRow("Min point count", self.min_points_spin)
        self.min_point_count_label_widget = self.basic_filters_form.labelForField(self.min_points_spin)

        self.geometry_filters_form.addRow("BBox width range", self.bbox_width_range_widget)
        self.min_bbox_width_label_widget = self.geometry_filters_form.labelForField(self.bbox_width_range_widget)
        self.max_bbox_width_label_widget = None
        self.geometry_filters_form.addRow("BBox height range", self.bbox_height_range_widget)
        self.min_bbox_height_label_widget = self.geometry_filters_form.labelForField(self.bbox_height_range_widget)
        self.max_bbox_height_label_widget = None
        self.geometry_filters_form.addRow("Aspect ratio range", self.aspect_ratio_range_widget)
        self.min_aspect_ratio_label_widget = self.geometry_filters_form.labelForField(self.aspect_ratio_range_widget)
        self.max_aspect_ratio_label_widget = None
        self.geometry_filters_form.addRow("Border handling", self.exclude_border_touching_checkbox)
        self.border_handling_label_widget = self.geometry_filters_form.labelForField(self.exclude_border_touching_checkbox)
        self.geometry_filters_form.addRow("Min solidity", self.min_solidity_spin)
        self.min_solidity_label_widget = self.geometry_filters_form.labelForField(self.min_solidity_spin)
        self.geometry_filters_form.addRow("Min extent", self.min_extent_spin)
        self.min_extent_label_widget = self.geometry_filters_form.labelForField(self.min_extent_spin)
        self.geometry_filters_form.addRow("Min polygon angle", self.min_polygon_angle_spin)
        self.min_polygon_angle_label_widget = self.geometry_filters_form.labelForField(self.min_polygon_angle_spin)

        self.conductor_form.addRow("Gradient boundaries", self.conductor_gradient_checkbox)
        self.conductor_gradient_enabled_label_widget = self.conductor_form.labelForField(self.conductor_gradient_checkbox)
        self.conductor_form.addRow("Min edge", self.conductor_gradient_min_strength_spin)
        self.conductor_gradient_min_strength_label_widget = self.conductor_form.labelForField(self.conductor_gradient_min_strength_spin)
        self.conductor_form.addRow("Boundary band", self.conductor_gradient_band_radius_spin)
        self.conductor_gradient_band_radius_label_widget = self.conductor_form.labelForField(self.conductor_gradient_band_radius_spin)

        self.via_form.addRow("Via size mode", self.via_size_mode_combo)
        self.via_size_mode_label_widget = self.via_form.labelForField(self.via_size_mode_combo)
        self.via_form.addRow("White range", self.via_white_range_widget)
        self.via_white_range_label_widget = self.via_form.labelForField(self.via_white_range_widget)
        self.via_form.addRow("Black range", self.via_black_range_widget)
        self.via_black_range_label_widget = self.via_form.labelForField(self.via_black_range_widget)
        self.via_form.addRow("Detection methods", self.via_detector_methods_widget)
        self.via_detector_methods_label_widget = self.via_form.labelForField(self.via_detector_methods_widget)
        self.via_form.addRow("Spot contrast", self.via_spot_min_contrast_spin)
        self.via_spot_min_contrast_label_widget = self.via_form.labelForField(self.via_spot_min_contrast_spin)
        self.via_form.addRow("Spot roundness", self.via_spot_min_roundness_spin)
        self.via_spot_min_roundness_label_widget = self.via_form.labelForField(self.via_spot_min_roundness_spin)
        self.via_form.addRow("Spot trace suppression", self.via_spot_line_suppression_spin)
        self.via_spot_line_suppression_label_widget = self.via_form.labelForField(self.via_spot_line_suppression_spin)
        self.via_form.addRow("Gradient edge", self.via_gradient_min_strength_spin)
        self.via_gradient_min_strength_label_widget = self.via_form.labelForField(self.via_gradient_min_strength_spin)
        self.via_form.addRow("Gradient coverage", self.via_gradient_min_coverage_spin)
        self.via_gradient_min_coverage_label_widget = self.via_form.labelForField(self.via_gradient_min_coverage_spin)
        self.via_form.addRow("Hough edge", self.via_hough_edge_threshold_spin)
        self.via_hough_edge_threshold_label_widget = self.via_form.labelForField(self.via_hough_edge_threshold_spin)
        self.via_form.addRow("Hough votes", self.via_hough_accumulator_threshold_spin)
        self.via_hough_accumulator_threshold_label_widget = self.via_form.labelForField(self.via_hough_accumulator_threshold_spin)
        self.via_form.addRow("Components score", self.via_component_min_score_spin)
        self.via_component_min_score_label_widget = self.via_form.labelForField(self.via_component_min_score_spin)
        self.via_form.addRow("Contours score", self.via_contour_min_score_spin)
        self.via_contour_min_score_label_widget = self.via_form.labelForField(self.via_contour_min_score_spin)
        self.via_form.addRow("Morphology peaks", self.via_morphology_peak_scale_spin)
        self.via_morphology_peak_scale_label_widget = self.via_form.labelForField(self.via_morphology_peak_scale_spin)
        self.via_form.addRow("Template score", self.via_template_min_score_spin)
        self.via_template_min_score_label_widget = self.via_form.labelForField(self.via_template_min_score_spin)
        self.via_form.addRow("Templates", self.via_templates_widget)
        self.via_templates_label_widget = self.via_form.labelForField(self.via_templates_widget)
        self.via_form.addRow("Blob circularity", self.via_blob_min_circularity_spin)
        self.via_blob_min_circularity_label_widget = self.via_form.labelForField(self.via_blob_min_circularity_spin)
        self.via_form.addRow("Saved presets", self.via_preset_widget)
        self.via_preset_label_widget = self.via_form.labelForField(self.via_preset_widget)
        self.via_form.addRow("Preset", self.noisy_traces_via_preset_button)
        self.noisy_traces_via_preset_label_widget = self.via_form.labelForField(self.noisy_traces_via_preset_button)
        self.via_form.addRow("Preset", self.blurred_via_preset_button)
        self.blurred_via_preset_label_widget = self.via_form.labelForField(self.blurred_via_preset_button)
        self.via_form.addRow("Reset", self.reset_via_search_button)
        self.reset_via_search_label_widget = self.via_form.labelForField(self.reset_via_search_button)
        self.via_form.addRow("Debug", self.debug_candidates_checkbox)
        self.debug_candidates_label_widget = self.via_form.labelForField(self.debug_candidates_checkbox)
        self.via_form.addRow("Roundness", self.via_roundness_spin)
        self.via_roundness_label_widget = self.via_form.labelForField(self.via_roundness_spin)
        self.via_form.addRow("Via width range", self.via_width_range_widget)
        self.min_via_width_label_widget = self.via_form.labelForField(self.via_width_range_widget)
        self.max_via_width_label_widget = None
        self.via_form.addRow("Via height range", self.via_height_range_widget)
        self.min_via_height_label_widget = self.via_form.labelForField(self.via_height_range_widget)
        self.max_via_height_label_widget = None
        self.via_form.addRow("Fixed vias", self.fixed_vias_widget)
        self.fixed_vias_label_widget = self.via_form.labelForField(self.fixed_vias_widget)
        self._update_via_size_controls_state()

        self.topology_form.addRow("Min hierarchy depth", self.min_hierarchy_depth_spin)
        self.min_hierarchy_depth_label_widget = self.topology_form.labelForField(self.min_hierarchy_depth_spin)
        self.topology_form.addRow("Max hierarchy depth (0 = unlimited)", self.max_hierarchy_depth_spin)
        self.max_hierarchy_depth_label_widget = self.topology_form.labelForField(self.max_hierarchy_depth_spin)
        self.topology_form.addRow("Max hole area ratio (0 = unlimited)", self.max_hole_area_ratio_spin)
        self.max_hole_area_ratio_label_widget = self.topology_form.labelForField(self.max_hole_area_ratio_spin)

        for group in [
            self.profile_group,
            self.basic_filters_group,
            self.geometry_filters_group,
            self.conductor_group,
            self.via_group,
            self.topology_group,
        ]:
            contour_layout.addWidget(group)
        container_layout.addWidget(self.contour_group)

        self.save_group = QGroupBox("Save options")
        save_layout = QVBoxLayout(self.save_group)
        self.save_cif_checkbox = QCheckBox("CIF")
        self.save_cif_checkbox.setChecked(True)
        self.save_csv_checkbox = QCheckBox("CSV")
        self.save_txt_checkbox = QCheckBox("TXT")
        self.save_svg_checkbox = QCheckBox("SVG preview")
        self.save_preview_checkbox = QCheckBox("Overlay preview image")
        self.save_preview_checkbox.setChecked(True)
        for checkbox in [
            self.save_cif_checkbox,
            self.save_csv_checkbox,
            self.save_txt_checkbox,
            self.save_svg_checkbox,
            self.save_preview_checkbox,
        ]:
            save_layout.addWidget(checkbox)
        container_layout.addWidget(self.save_group)
        container_layout.addStretch(1)
        scroll.setWidget(container)
        layout.addWidget(scroll, 1)
        return tab

    def _build_display_tab(self) -> QWidget:
        tab = QWidget()
        self.display_form = QFormLayout(tab)

        self.external_color_button = self._build_color_button(self._display_settings.external_color, self._choose_external_color)
        self.hole_color_button = self._build_color_button(self._display_settings.hole_color, self._choose_hole_color)
        self.selected_color_button = self._build_color_button(self._display_settings.selected_color, self._choose_selected_color)
        self.vertex_color_button = self._build_color_button(self._display_settings.vertex_color, self._choose_vertex_color)
        self.line_width_spin = QDoubleSpinBox()
        self.line_width_spin.setRange(1.0, 20.0)
        self.line_width_spin.setValue(self._display_settings.line_width)
        self.vertex_size_spin = QDoubleSpinBox()
        self.vertex_size_spin.setRange(2.0, 30.0)
        self.vertex_size_spin.setValue(self._display_settings.vertex_size)
        self.fill_opacity_spin = QDoubleSpinBox()
        self.fill_opacity_spin.setRange(0.0, 1.0)
        self.fill_opacity_spin.setSingleStep(0.05)
        self.fill_opacity_spin.setValue(self._display_settings.fill_opacity)
        self.show_vertices_checkbox = QCheckBox("Show vertices")
        self.show_vertices_checkbox.setChecked(self._display_settings.show_vertices)
        self.show_labels_checkbox = QCheckBox("Show polygon IDs")
        self.show_labels_checkbox.setChecked(self._display_settings.show_labels)
        self.random_object_colors_checkbox = QCheckBox("Random object colors")
        self.show_neighbor_frames_checkbox = QCheckBox("Show neighboring frames")
        self.neighbor_columns_spin = QSpinBox()
        self.neighbor_columns_spin.setRange(1, 1000)
        self.neighbor_columns_spin.setValue(3)
        self.neighbor_max_grid_spin = QSpinBox()
        self.neighbor_max_grid_spin.setRange(3, 7)
        self.neighbor_max_grid_spin.setSingleStep(2)
        self.neighbor_max_grid_spin.setValue(7)
        self.neighbor_opacity_spin = QDoubleSpinBox()
        self.neighbor_opacity_spin.setRange(0.05, 1.0)
        self.neighbor_opacity_spin.setSingleStep(0.05)
        self.neighbor_opacity_spin.setValue(0.35)
        self.neighbor_overlap_spin = QSpinBox()
        self.neighbor_overlap_spin.setRange(0, 100_000)
        self.neighbor_overlap_spin.setValue(0)
        self.extra_layers_widget = QWidget()
        extra_layers_layout = QVBoxLayout(self.extra_layers_widget)
        extra_layers_layout.setContentsMargins(0, 0, 0, 0)
        extra_layers_layout.setSpacing(6)
        self.extra_layers_list = QListWidget()
        self.extra_layers_list.setMaximumHeight(100)
        extra_layers_layout.addWidget(self.extra_layers_list)
        self.extra_layer_path_widget = QWidget()
        extra_layer_path_layout = QHBoxLayout(self.extra_layer_path_widget)
        extra_layer_path_layout.setContentsMargins(0, 0, 0, 0)
        extra_layer_path_layout.setSpacing(6)
        self.extra_layer_path_edit = QLineEdit()
        self.extra_layer_path_browse_button = QPushButton("...")
        self.extra_layer_path_browse_button.setFixedWidth(34)
        extra_layer_path_layout.addWidget(self.extra_layer_path_edit, 1)
        extra_layer_path_layout.addWidget(self.extra_layer_path_browse_button)
        extra_layer_buttons = QWidget()
        extra_layer_buttons_layout = QHBoxLayout(extra_layer_buttons)
        extra_layer_buttons_layout.setContentsMargins(0, 0, 0, 0)
        self.add_extra_layers_button = QPushButton("Add images")
        self.remove_extra_layer_button = QPushButton("Remove")
        extra_layer_buttons_layout.addWidget(self.add_extra_layers_button)
        extra_layer_buttons_layout.addWidget(self.remove_extra_layer_button)
        extra_layers_layout.addWidget(extra_layer_buttons)
        self.extra_layer_visible_checkbox = QCheckBox("Layer visible")
        self.extra_layer_opacity_spin = QDoubleSpinBox()
        self.extra_layer_opacity_spin.setRange(0.0, 1.0)
        self.extra_layer_opacity_spin.setSingleStep(0.05)
        self.extra_layer_opacity_spin.setValue(0.35)
        self.extra_layer_dx_spin = QDoubleSpinBox()
        self.extra_layer_dx_spin.setRange(-1_000_000.0, 1_000_000.0)
        self.extra_layer_dx_spin.setDecimals(2)
        self.extra_layer_dy_spin = QDoubleSpinBox()
        self.extra_layer_dy_spin.setRange(-1_000_000.0, 1_000_000.0)
        self.extra_layer_dy_spin.setDecimals(2)

        for widget in [
            self.line_width_spin,
            self.vertex_size_spin,
            self.fill_opacity_spin,
            self.show_vertices_checkbox,
            self.show_labels_checkbox,
            self.random_object_colors_checkbox,
        ]:
            if isinstance(widget, QCheckBox):
                widget.stateChanged.connect(self._apply_display_settings)
            else:
                widget.valueChanged.connect(self._apply_display_settings)
        self.show_neighbor_frames_checkbox.stateChanged.connect(self._on_neighbor_display_settings_changed)
        self.neighbor_columns_spin.valueChanged.connect(self._on_neighbor_display_settings_changed)
        self.neighbor_max_grid_spin.valueChanged.connect(self._on_neighbor_display_settings_changed)
        self.neighbor_opacity_spin.valueChanged.connect(self._on_neighbor_display_settings_changed)
        self.neighbor_overlap_spin.valueChanged.connect(self._on_neighbor_display_settings_changed)
        self.extra_layers_list.currentRowChanged.connect(self._on_extra_layer_selected)
        self.add_extra_layers_button.clicked.connect(self._load_extra_layers)
        self.remove_extra_layer_button.clicked.connect(self._remove_selected_extra_layer)
        self.extra_layer_path_browse_button.clicked.connect(self._browse_selected_extra_layer_path)
        self.extra_layer_path_edit.editingFinished.connect(self._on_extra_layer_path_changed)
        self.extra_layer_visible_checkbox.stateChanged.connect(self._on_extra_layer_controls_changed)
        self.extra_layer_opacity_spin.valueChanged.connect(self._on_extra_layer_controls_changed)
        self.extra_layer_dx_spin.valueChanged.connect(self._on_extra_layer_controls_changed)
        self.extra_layer_dy_spin.valueChanged.connect(self._on_extra_layer_controls_changed)

        self.display_form.addRow("External contour", self.external_color_button)
        self.external_color_label_widget = self.display_form.labelForField(self.external_color_button)
        self.display_form.addRow("Hole contour", self.hole_color_button)
        self.hole_color_label_widget = self.display_form.labelForField(self.hole_color_button)
        self.display_form.addRow("Selected contour", self.selected_color_button)
        self.selected_color_label_widget = self.display_form.labelForField(self.selected_color_button)
        self.display_form.addRow("Vertex color", self.vertex_color_button)
        self.vertex_color_label_widget = self.display_form.labelForField(self.vertex_color_button)
        self.display_form.addRow("Line width", self.line_width_spin)
        self.line_width_label_widget = self.display_form.labelForField(self.line_width_spin)
        self.display_form.addRow("Vertex size", self.vertex_size_spin)
        self.vertex_size_label_widget = self.display_form.labelForField(self.vertex_size_spin)
        self.display_form.addRow("Fill opacity", self.fill_opacity_spin)
        self.fill_opacity_label_widget = self.display_form.labelForField(self.fill_opacity_spin)
        self.display_form.addRow(self.show_vertices_checkbox)
        self.display_form.addRow(self.show_labels_checkbox)
        self.display_form.addRow(self.random_object_colors_checkbox)
        self.display_form.addRow(self.show_neighbor_frames_checkbox)
        self.display_form.addRow("Frames per row", self.neighbor_columns_spin)
        self.neighbor_columns_label_widget = self.display_form.labelForField(self.neighbor_columns_spin)
        self.display_form.addRow("Max neighbor grid", self.neighbor_max_grid_spin)
        self.neighbor_max_grid_label_widget = self.display_form.labelForField(self.neighbor_max_grid_spin)
        self.display_form.addRow("Neighbor opacity", self.neighbor_opacity_spin)
        self.neighbor_opacity_label_widget = self.display_form.labelForField(self.neighbor_opacity_spin)
        self.display_form.addRow("Frame overlap", self.neighbor_overlap_spin)
        self.neighbor_overlap_label_widget = self.display_form.labelForField(self.neighbor_overlap_spin)
        self.display_form.addRow("Additional layers", self.extra_layers_widget)
        self.extra_layers_label_widget = self.display_form.labelForField(self.extra_layers_widget)
        self.display_form.addRow("Layer path", self.extra_layer_path_widget)
        self.extra_layer_path_label_widget = self.display_form.labelForField(self.extra_layer_path_widget)
        self.display_form.addRow(self.extra_layer_visible_checkbox)
        self.display_form.addRow("Layer opacity", self.extra_layer_opacity_spin)
        self.extra_layer_opacity_label_widget = self.display_form.labelForField(self.extra_layer_opacity_spin)
        self.display_form.addRow("Layer dX", self.extra_layer_dx_spin)
        self.extra_layer_dx_label_widget = self.display_form.labelForField(self.extra_layer_dx_spin)
        self.display_form.addRow("Layer dY", self.extra_layer_dy_spin)
        self.extra_layer_dy_label_widget = self.display_form.labelForField(self.extra_layer_dy_spin)
        return tab

    def _build_help_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        self.help_scroll = QScrollArea()
        self.help_scroll.setWidgetResizable(True)
        self.help_container = QWidget()
        self.help_layout = QVBoxLayout(self.help_container)
        self.help_layout.setContentsMargins(0, 0, 0, 0)
        self.help_scroll.setWidget(self.help_container)
        layout.addWidget(self.help_scroll, 1)
        self._rebuild_help_cards()
        return tab

    def _clear_layout_widgets(self, layout: QVBoxLayout) -> None:
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            child_layout = item.layout()
            if widget is not None:
                widget.deleteLater()
            elif child_layout is not None:
                self._clear_layout_widgets(child_layout)  # type: ignore[arg-type]

    @staticmethod
    def _build_help_sample_image() -> np.ndarray:
        image = np.full((180, 260), 38, dtype=np.uint8)
        cv2.rectangle(image, (18, 18), (110, 90), 190, thickness=-1)
        cv2.circle(image, (176, 60), 26, 230, thickness=-1)
        cv2.circle(image, (176, 60), 10, 70, thickness=-1)
        cv2.line(image, (20, 136), (236, 120), 160, thickness=6)
        cv2.line(image, (22, 154), (236, 154), 210, thickness=4)
        cv2.putText(image, "A1", (126, 138), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 240, 2, cv2.LINE_AA)
        noise = np.random.default_rng(42).normal(0, 12, image.shape).astype(np.int16)
        return np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    def _operation_help_entry(self, operation_name: str) -> tuple[str, str]:
        entry = PIPELINE_OPERATION_HELP_TEXTS.get(operation_name, {})
        summary_pair = entry.get("summary", ("", ""))
        use_pair = entry.get("use", ("", ""))
        summary = summary_pair[0] if self._ui_language == "ru" else summary_pair[1]
        use_case = use_pair[0] if self._ui_language == "ru" else use_pair[1]
        if not summary:
            summary = "Преобразование обрабатывает изображение перед извлечением контуров." if self._ui_language == "ru" else "This transformation preprocesses the image before contour extraction."
        if not use_case:
            use_case = "Используйте, когда этот эффект приближает изображение к удобной бинарной маске." if self._ui_language == "ru" else "Use it when the effect moves the image toward a cleaner binary mask."
        return summary, use_case

    def _pipeline_parameter_tooltip(self, operation_name: str, parameter_name: str) -> str:
        del operation_name
        return _localized_text(PIPELINE_PARAMETER_HELP_TEXTS, parameter_name, self._ui_language)

    def _pixmap_for_help_image(self, image: np.ndarray) -> QPixmap:
        return QPixmap.fromImage(cv_to_qimage(image)).scaled(
            190,
            132,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )

    def _rebuild_help_cards(self) -> None:
        if not hasattr(self, "help_layout"):
            return
        self._clear_layout_widgets(self.help_layout)
        intro = QLabel(
            "Ниже показано, как каждое преобразование меняет один и тот же тестовый кадр. Это помогает понять, когда шаг уместен в pipeline."
            if self._ui_language == "ru"
            else "Below, each transformation is applied to the same synthetic sample image so you can see what it changes and when to use it."
        )
        intro.setWordWrap(True)
        self.help_layout.addWidget(intro)
        sample_image = self._build_help_sample_image()
        before_pixmap = self._pixmap_for_help_image(sample_image)
        for descriptor in available_operations():
            card = QGroupBox(get_operation_display_name(descriptor.type_name, self._ui_language))
            card_layout = QVBoxLayout(card)
            summary, use_case = self._operation_help_entry(descriptor.type_name)
            summary_label = QLabel(summary)
            summary_label.setWordWrap(True)
            use_label = QLabel(
                ("Когда использовать: " if self._ui_language == "ru" else "When to use: ") + use_case
            )
            use_label.setWordWrap(True)
            images_row = QWidget()
            images_layout = QHBoxLayout(images_row)
            images_layout.setContentsMargins(0, 0, 0, 0)
            before_box = QVBoxLayout()
            before_title = QLabel("До" if self._ui_language == "ru" else "Before")
            before_image = QLabel()
            before_image.setPixmap(before_pixmap)
            before_box.addWidget(before_title)
            before_box.addWidget(before_image)
            after_box = QVBoxLayout()
            after_title = QLabel("После" if self._ui_language == "ru" else "After")
            after_image = QLabel()
            try:
                processed = descriptor.handler(sample_image.copy(), descriptor.default_parameters())
            except Exception:
                processed = sample_image
            after_image.setPixmap(self._pixmap_for_help_image(processed))
            after_box.addWidget(after_title)
            after_box.addWidget(after_image)
            images_layout.addLayout(before_box)
            images_layout.addLayout(after_box)
            card_layout.addWidget(summary_label)
            card_layout.addWidget(use_label)
            card_layout.addWidget(images_row)
            self.help_layout.addWidget(card)
        self.help_layout.addStretch(1)

    def help_menu_title(self) -> str:
        return self._tr("tab_help")

    def attach_help_menu(self, menu: QMenu) -> None:
        self._help_menu = menu
        self._refresh_help_menu()

    def _refresh_help_menu(self) -> None:
        if self._help_menu is None:
            return
        self._help_menu.clear()
        overview_action = self._help_menu.addAction(
            self._tr("help_all_filters_action", "Все преобразования" if self._ui_language == "ru" else "All transformations")
        )
        overview_action.triggered.connect(lambda _checked=False: self._show_help_dialog())
        self._help_menu.addSeparator()
        for group_key, labels, operations in PIPELINE_OPERATION_GROUPS:
            submenu = self._help_menu.addMenu(labels[0] if self._ui_language == "ru" else labels[1])
            submenu.setObjectName(f"helpMenu_{group_key}")
            for operation_name in operations:
                action = submenu.addAction(get_operation_display_name(operation_name, self._ui_language))
                action.triggered.connect(lambda _checked=False, op=operation_name: self._show_help_dialog(op))

    def _show_help_dialog(self, operation_name: str | None = None) -> None:
        dialog = QDialog(self)
        dialog.setWindowTitle(
            self._tr("tab_help") if operation_name is None else get_operation_display_name(operation_name, self._ui_language)
        )
        dialog.resize(960, 720)
        layout = QVBoxLayout(dialog)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        container = QWidget()
        help_layout = QVBoxLayout(container)
        help_layout.setContentsMargins(0, 0, 0, 0)
        scroll.setWidget(container)
        layout.addWidget(scroll, 1)
        self._populate_help_cards(
            help_layout,
            [operation_name] if operation_name is not None else self._all_operation_names(),
        )
        dialog.exec()

    def _populate_help_cards(self, layout: QVBoxLayout, operation_names: list[str]) -> None:
        self._clear_layout_widgets(layout)
        intro = QLabel(
            self._tr(
                "help_intro_text",
                "Ниже показано, как каждое преобразование меняет один и тот же тестовый кадр. Это помогает понять, когда шаг уместен в pipeline."
                if self._ui_language == "ru"
                else "Each transformation below is applied to the same sample image so you can see its effect and when to use it.",
            )
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)
        sample_image = self._build_help_sample_image()
        before_pixmap = self._pixmap_for_help_image(sample_image)
        for operation_name in operation_names:
            descriptor = get_operation_descriptor(operation_name)
            card = QGroupBox(get_operation_display_name(descriptor.type_name, self._ui_language))
            card_layout = QVBoxLayout(card)
            summary, use_case = self._operation_help_entry(descriptor.type_name)
            summary_label = QLabel(summary)
            summary_label.setWordWrap(True)
            use_label = QLabel(
                ("Когда использовать: " if self._ui_language == "ru" else "When to use: ") + use_case
            )
            use_label.setWordWrap(True)
            images_row = QWidget()
            images_layout = QHBoxLayout(images_row)
            images_layout.setContentsMargins(0, 0, 0, 0)
            before_box = QVBoxLayout()
            before_title = QLabel("До" if self._ui_language == "ru" else "Before")
            before_image = QLabel()
            before_image.setPixmap(before_pixmap)
            before_box.addWidget(before_title)
            before_box.addWidget(before_image)
            after_box = QVBoxLayout()
            after_title = QLabel("После" if self._ui_language == "ru" else "After")
            after_image = QLabel()
            try:
                processed = descriptor.handler(sample_image.copy(), descriptor.default_parameters())
            except Exception:
                processed = sample_image
            after_image.setPixmap(self._pixmap_for_help_image(processed))
            after_box.addWidget(after_title)
            after_box.addWidget(after_image)
            images_layout.addLayout(before_box)
            images_layout.addLayout(after_box)
            card_layout.addWidget(summary_label)
            card_layout.addWidget(use_label)
            card_layout.addWidget(images_row)
            layout.addWidget(card)
        layout.addStretch(1)

    def _all_operation_names(self) -> list[str]:
        return [descriptor.type_name for descriptor in available_operations()]

    def _selected_available_operation_name(self) -> str | None:
        if not hasattr(self, "operation_tree"):
            return None
        item = self.operation_tree.currentItem()
        if item is None:
            return None
        operation_name = item.data(0, Qt.ItemDataRole.UserRole)
        return str(operation_name) if operation_name else None

    def _find_operation_tree_item(self, operation_name: str) -> QTreeWidgetItem | None:
        if not hasattr(self, "operation_tree"):
            return None
        for index in range(self.operation_tree.topLevelItemCount()):
            group_item = self.operation_tree.topLevelItem(index)
            for child_index in range(group_item.childCount()):
                child_item = group_item.child(child_index)
                if child_item.data(0, Qt.ItemDataRole.UserRole) == operation_name:
                    return child_item
        return None

    def _update_pipeline_help_preview(self, operation_name: str | None) -> None:
        if not hasattr(self, "pipeline_help_title"):
            return
        if not operation_name:
            self.pipeline_help_title.clear()
            self.pipeline_help_summary.clear()
            self.pipeline_help_use.clear()
            self.pipeline_help_before_image.clear()
            self.pipeline_help_after_image.clear()
            return
        descriptor = get_operation_descriptor(operation_name)
        summary, use_case = self._operation_help_entry(operation_name)
        sample_image = self._build_help_sample_image()
        self.pipeline_help_title.setText(get_operation_display_name(operation_name, self._ui_language))
        self.pipeline_help_summary.setText(summary)
        self.pipeline_help_use.setText(
            ("Когда использовать: " if self._ui_language == "ru" else "When to use: ") + use_case
        )
        self.pipeline_help_before_image.setPixmap(self._pixmap_for_help_image(sample_image))
        try:
            processed = descriptor.handler(sample_image.copy(), descriptor.default_parameters())
        except Exception:
            processed = sample_image
        self.pipeline_help_after_image.setPixmap(self._pixmap_for_help_image(processed))

    def _on_available_operation_selected(self, current: QTreeWidgetItem | None, _previous: QTreeWidgetItem | None) -> None:
        operation_name = current.data(0, Qt.ItemDataRole.UserRole) if current is not None else None
        self._update_pipeline_help_preview(str(operation_name) if operation_name else None)

    def _on_available_operation_activated(self, item: QTreeWidgetItem, _column: int) -> None:
        if item.data(0, Qt.ItemDataRole.UserRole):
            self._add_pipeline_step()

    def _set_field_tooltip(self, label_widget: QLabel | None, field_widget: QWidget, help_key: str) -> None:
        tooltip = _localized_text(EXTRACTION_HELP_TEXTS, help_key, self._ui_language)
        if label_widget is not None:
            label_widget.setToolTip(tooltip)
        field_widget.setToolTip(tooltip)

    def _renumber_fixed_via_rows(self) -> None:
        for index, row in enumerate(self._fixed_via_rows, start=1):
            label = row["label"]
            if isinstance(label, QLabel):
                label.setText(f"via{index}")

    def _clear_fixed_via_rows(self) -> None:
        while self._fixed_via_rows:
            row = self._fixed_via_rows.pop()
            widget = row["widget"]
            if isinstance(widget, QWidget):
                self.fixed_via_rows_layout.removeWidget(widget)
                widget.deleteLater()

    def _fixed_via_pairs(self) -> list[tuple[int, int]]:
        pairs: list[tuple[int, int]] = []
        for row in self._fixed_via_rows:
            width_spin = row["width_spin"]
            height_spin = row["height_spin"]
            if isinstance(width_spin, QSpinBox) and isinstance(height_spin, QSpinBox):
                pairs.append((int(width_spin.value()), int(height_spin.value())))
        return pairs

    def _delete_fixed_via_row(self, row_widget: QWidget) -> None:
        for index, row in enumerate(self._fixed_via_rows):
            if row["widget"] is row_widget:
                self._fixed_via_rows.pop(index)
                self.fixed_via_rows_layout.removeWidget(row_widget)
                row_widget.deleteLater()
                self._renumber_fixed_via_rows()
                if not self._suspend_fixed_via_updates:
                    self._on_extraction_settings_changed()
                return

    def _add_fixed_via_row(self, *_args, width: int = 1, height: int = 1) -> None:
        row_widget = QWidget()
        row_layout = QHBoxLayout(row_widget)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.setSpacing(8)

        via_label = QLabel("")
        via_label.setMinimumWidth(44)
        width_spin = QSpinBox()
        width_spin.setRange(1, 100_000)
        width_spin.setValue(max(1, int(width)))
        width_spin.setPrefix("X ")
        height_spin = QSpinBox()
        height_spin.setRange(1, 100_000)
        height_spin.setValue(max(1, int(height)))
        height_spin.setPrefix("Y ")
        remove_button = QPushButton("-")
        remove_button.setFixedWidth(36)
        remove_button.setMinimumHeight(30)
        remove_button.setStyleSheet(
            "QPushButton { background-color: #d64545; color: white; font-size: 18px; font-weight: 700; border-radius: 6px; }"
            "QPushButton:hover { background-color: #bf3838; }"
            "QPushButton:pressed { background-color: #a93030; }"
        )

        width_spin.valueChanged.connect(self._on_extraction_settings_changed)
        height_spin.valueChanged.connect(self._on_extraction_settings_changed)
        remove_button.clicked.connect(lambda _checked=False, widget=row_widget: self._delete_fixed_via_row(widget))

        self._fixed_via_rows.append(
            {
                "widget": row_widget,
                "label": via_label,
                "width_spin": width_spin,
                "height_spin": height_spin,
                "remove_button": remove_button,
            }
        )

        row_layout.addWidget(via_label)
        row_layout.addWidget(width_spin, 1)
        row_layout.addWidget(height_spin, 1)
        row_layout.addWidget(remove_button)
        self.fixed_via_rows_layout.addWidget(row_widget)
        self._renumber_fixed_via_rows()

        width_spin.setToolTip(_localized_text(EXTRACTION_HELP_TEXTS, "fixed_via_widths", self._ui_language))
        height_spin.setToolTip(_localized_text(EXTRACTION_HELP_TEXTS, "fixed_via_heights", self._ui_language))
        remove_button.setToolTip(
            "Удаляет эту строку с допустимым размером via из списка."
            if self._ui_language == "ru"
            else "Removes this allowed via-size row from the list."
        )

        if not self._suspend_fixed_via_updates:
            self._on_extraction_settings_changed()

    def _apply_extraction_tooltips(self) -> None:
        self._set_field_tooltip(self.extraction_profile_label_widget, self.extraction_profile_combo, "extraction_profile")
        self._set_field_tooltip(self.retrieval_mode_label_widget, self.retrieval_mode_combo, "retrieval_mode")
        self._set_field_tooltip(self.approximation_mode_label_widget, self.approximation_mode_combo, "approximation_mode")
        self._set_field_tooltip(self.epsilon_label_widget, self.epsilon_spin, "epsilon")
        self._set_field_tooltip(self.epsilon_mode_label_widget, self.epsilon_relative_checkbox, "epsilon_mode")
        self._set_field_tooltip(self.min_area_label_widget, self.min_area_spin, "min_area")
        self._set_field_tooltip(self.max_area_label_widget, self.max_area_spin, "max_area")
        self._set_field_tooltip(self.min_perimeter_label_widget, self.min_perimeter_spin, "min_perimeter")
        self._set_field_tooltip(self.max_perimeter_label_widget, self.max_perimeter_spin, "max_perimeter")
        self._set_field_tooltip(self.min_point_count_label_widget, self.min_points_spin, "min_points")
        self._set_field_tooltip(self.min_bbox_width_label_widget, self.min_bbox_width_spin, "min_bbox_width")
        self._set_field_tooltip(self.max_bbox_width_label_widget, self.max_bbox_width_spin, "max_bbox_width")
        self._set_field_tooltip(self.min_bbox_height_label_widget, self.min_bbox_height_spin, "min_bbox_height")
        self._set_field_tooltip(self.max_bbox_height_label_widget, self.max_bbox_height_spin, "max_bbox_height")
        self._set_field_tooltip(self.min_aspect_ratio_label_widget, self.min_aspect_ratio_spin, "min_aspect_ratio")
        self._set_field_tooltip(self.max_aspect_ratio_label_widget, self.max_aspect_ratio_spin, "max_aspect_ratio")
        self._set_field_tooltip(self.border_handling_label_widget, self.exclude_border_touching_checkbox, "exclude_border_touching")
        self._set_field_tooltip(self.min_solidity_label_widget, self.min_solidity_spin, "min_solidity")
        self._set_field_tooltip(self.min_extent_label_widget, self.min_extent_spin, "min_extent")
        self._set_field_tooltip(self.via_size_mode_label_widget, self.via_size_mode_combo, "via_size_mode")
        self._set_field_tooltip(self.via_white_range_label_widget, self.via_white_range_widget, "via_white_range")
        self._set_field_tooltip(self.via_black_range_label_widget, self.via_black_range_widget, "via_black_range")
        self._set_field_tooltip(self.via_detector_methods_label_widget, self.via_detector_methods_widget, "via_detector_methods")
        self._set_field_tooltip(
            self.via_gradient_min_strength_label_widget,
            self.via_gradient_min_strength_spin,
            "via_gradient_min_strength",
        )
        self._set_field_tooltip(
            self.via_gradient_min_coverage_label_widget,
            self.via_gradient_min_coverage_spin,
            "via_gradient_min_coverage",
        )
        self._set_field_tooltip(self.via_spot_min_contrast_label_widget, self.via_spot_min_contrast_spin, "via_spot_min_contrast")
        self._set_field_tooltip(self.via_spot_min_roundness_label_widget, self.via_spot_min_roundness_spin, "via_spot_min_roundness")
        self._set_field_tooltip(
            self.via_spot_line_suppression_label_widget,
            self.via_spot_line_suppression_spin,
            "via_spot_line_suppression",
        )
        self._set_field_tooltip(
            self.via_hough_edge_threshold_label_widget,
            self.via_hough_edge_threshold_spin,
            "via_hough_edge_threshold",
        )
        self._set_field_tooltip(
            self.via_hough_accumulator_threshold_label_widget,
            self.via_hough_accumulator_threshold_spin,
            "via_hough_accumulator_threshold",
        )
        self._set_field_tooltip(self.via_component_min_score_label_widget, self.via_component_min_score_spin, "via_component_min_score")
        self._set_field_tooltip(self.via_contour_min_score_label_widget, self.via_contour_min_score_spin, "via_contour_min_score")
        self._set_field_tooltip(self.via_morphology_peak_scale_label_widget, self.via_morphology_peak_scale_spin, "via_morphology_peak_scale")
        self._set_field_tooltip(self.via_template_min_score_label_widget, self.via_template_min_score_spin, "via_template_min_score")
        self._set_field_tooltip(self.via_templates_label_widget, self.via_templates_widget, "via_templates")
        self._set_field_tooltip(self.via_blob_min_circularity_label_widget, self.via_blob_min_circularity_spin, "via_blob_min_circularity")
        self._set_field_tooltip(self.via_preset_label_widget, self.via_preset_widget, "via_preset_selector")
        self._set_field_tooltip(
            self.noisy_traces_via_preset_label_widget,
            self.noisy_traces_via_preset_button,
            "via_noisy_traces_preset",
        )
        self._set_field_tooltip(
            self.blurred_via_preset_label_widget,
            self.blurred_via_preset_button,
            "via_blurred_preset",
        )
        self._set_field_tooltip(self.reset_via_search_label_widget, self.reset_via_search_button, "reset_via_search")
        self.add_via_template_button.setToolTip(_localized_text(EXTRACTION_HELP_TEXTS, "via_templates", self._ui_language))
        self.remove_via_template_button.setToolTip(
            "Удаляет выбранный шаблон via из списка." if self._ui_language == "ru" else "Removes the selected via template from the list."
        )
        self.clear_via_templates_button.setToolTip(
            "Удаляет все сохраненные шаблоны via из списка." if self._ui_language == "ru" else "Removes all saved via templates from the list."
        )
        for checkbox, tooltip_key in (
            (self.via_white_range_checkbox, "via_white_range"),
            (self.via_black_range_checkbox, "via_black_range"),
            (self.via_detector_gradient_checkbox, "via_detector_gradient_method"),
            (self.via_detector_spot_checkbox, "via_detector_spot_method"),
            (self.via_detector_hough_checkbox, "via_detector_hough_method"),
            (self.via_detector_components_checkbox, "via_detector_components_method"),
            (self.via_detector_contours_checkbox, "via_detector_contours_method"),
            (self.via_detector_morphology_checkbox, "via_detector_morphology_method"),
            (self.via_detector_template_checkbox, "via_detector_template_method"),
            (self.via_detector_blob_checkbox, "via_detector_blob_method"),
        ):
            detector_tooltip = _localized_text(EXTRACTION_HELP_TEXTS, tooltip_key, self._ui_language)
            checkbox.setToolTip(detector_tooltip)
            checkbox.setStatusTip(detector_tooltip)
        self._set_field_tooltip(self.debug_candidates_label_widget, self.debug_candidates_checkbox, "debug_candidates")
        self._set_field_tooltip(self.via_roundness_label_widget, self.via_roundness_spin, "via_min_roundness")
        self._set_field_tooltip(self.min_via_width_label_widget, self.min_via_width_spin, "min_via_width")
        self._set_field_tooltip(self.max_via_width_label_widget, self.max_via_width_spin, "max_via_width")
        self._set_field_tooltip(self.min_via_height_label_widget, self.min_via_height_spin, "min_via_height")
        self._set_field_tooltip(self.max_via_height_label_widget, self.max_via_height_spin, "max_via_height")
        self._set_field_tooltip(self.fixed_vias_label_widget, self.fixed_vias_widget, "fixed_via_widths")
        self.fixed_via_add_button.setToolTip(
            "Добавляет еще одну допустимую пару ширины и высоты via."
            if self._ui_language == "ru"
            else "Adds another allowed via width and height pair."
        )
        for row in self._fixed_via_rows:
            width_spin = row["width_spin"]
            height_spin = row["height_spin"]
            remove_button = row["remove_button"]
            if isinstance(width_spin, QSpinBox):
                width_spin.setToolTip(_localized_text(EXTRACTION_HELP_TEXTS, "fixed_via_widths", self._ui_language))
            if isinstance(height_spin, QSpinBox):
                height_spin.setToolTip(_localized_text(EXTRACTION_HELP_TEXTS, "fixed_via_heights", self._ui_language))
            if isinstance(remove_button, QPushButton):
                remove_button.setToolTip(
                    "Удаляет эту строку с допустимым размером via из списка."
                    if self._ui_language == "ru"
                    else "Removes this allowed via-size row from the list."
                )
        self._set_field_tooltip(self.min_hierarchy_depth_label_widget, self.min_hierarchy_depth_spin, "min_hierarchy_depth")
        self._set_field_tooltip(self.max_hierarchy_depth_label_widget, self.max_hierarchy_depth_spin, "max_hierarchy_depth")
        self._set_field_tooltip(self.max_hole_area_ratio_label_widget, self.max_hole_area_ratio_spin, "max_hole_area_ratio")

    def _update_via_size_controls_state(self) -> None:
        fixed_mode = normalize_via_size_mode(self.via_size_mode_combo.currentData()) == VIA_SIZE_MODE_FIXED
        range_widgets = [
            (self.min_via_width_label_widget, self.via_width_range_widget),
            (self.min_via_height_label_widget, self.via_height_range_widget),
        ]
        fixed_widgets = [
            (self.fixed_vias_label_widget, self.fixed_vias_widget),
        ]
        for label_widget, field_widget in range_widgets:
            if label_widget is not None:
                label_widget.setVisible(not fixed_mode)
            field_widget.setVisible(not fixed_mode)
        for label_widget, field_widget in fixed_widgets:
            if label_widget is not None:
                label_widget.setVisible(fixed_mode)
            field_widget.setVisible(fixed_mode)
        self._update_via_threshold_controls_state()

    def _update_via_threshold_controls_state(self) -> None:
        white_enabled = self.via_white_range_checkbox.isChecked()
        self.via_white_range_min_spin.setEnabled(white_enabled)
        self.via_white_range_max_spin.setEnabled(white_enabled)
        if self.via_white_range_label_widget is not None:
            self.via_white_range_label_widget.setVisible(white_enabled)
        self.via_white_range_widget.setVisible(white_enabled)
        black_enabled = self.via_black_range_checkbox.isChecked()
        self.via_black_range_min_spin.setEnabled(black_enabled)
        self.via_black_range_max_spin.setEnabled(black_enabled)
        if self.via_black_range_label_widget is not None:
            self.via_black_range_label_widget.setVisible(black_enabled)
        self.via_black_range_widget.setVisible(black_enabled)
        detector_rows = [
            (
                self.via_detector_gradient_checkbox.isChecked(),
                [
                    (self.via_gradient_min_strength_label_widget, self.via_gradient_min_strength_spin),
                    (self.via_gradient_min_coverage_label_widget, self.via_gradient_min_coverage_spin),
                ],
            ),
            (
                self.via_detector_spot_checkbox.isChecked(),
                [
                    (self.via_spot_min_contrast_label_widget, self.via_spot_min_contrast_spin),
                    (self.via_spot_min_roundness_label_widget, self.via_spot_min_roundness_spin),
                    (self.via_spot_line_suppression_label_widget, self.via_spot_line_suppression_spin),
                ],
            ),
            (
                self.via_detector_hough_checkbox.isChecked(),
                [
                    (self.via_hough_edge_threshold_label_widget, self.via_hough_edge_threshold_spin),
                    (self.via_hough_accumulator_threshold_label_widget, self.via_hough_accumulator_threshold_spin),
                ],
            ),
            (self.via_detector_components_checkbox.isChecked(), [(self.via_component_min_score_label_widget, self.via_component_min_score_spin)]),
            (self.via_detector_contours_checkbox.isChecked(), [(self.via_contour_min_score_label_widget, self.via_contour_min_score_spin)]),
            (self.via_detector_morphology_checkbox.isChecked(), [(self.via_morphology_peak_scale_label_widget, self.via_morphology_peak_scale_spin)]),
            (
                self.via_detector_template_checkbox.isChecked(),
                [
                    (self.via_template_min_score_label_widget, self.via_template_min_score_spin),
                    (self.via_templates_label_widget, self.via_templates_widget),
                ],
            ),
            (self.via_detector_blob_checkbox.isChecked(), [(self.via_blob_min_circularity_label_widget, self.via_blob_min_circularity_spin)]),
        ]
        for visible, rows in detector_rows:
            for label_widget, field_widget in rows:
                if label_widget is not None:
                    label_widget.setVisible(visible)
                field_widget.setVisible(visible)
                field_widget.setEnabled(visible)

    def _update_extraction_profile_controls_state(self) -> None:
        is_via_profile = self._active_extraction_profile == "vias"
        self.conductor_group.setEnabled(not is_via_profile)
        self.conductor_group.setVisible(not is_via_profile)
        self.via_group.setEnabled(is_via_profile)
        self.via_group.setVisible(is_via_profile)
        self.topology_group.setVisible(not is_via_profile)

    def _build_visual_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)

        self.editor_group = QGroupBox("Image / polygon editor")
        editor_layout = QVBoxLayout(self.editor_group)
        self.polygon_editor = PolygonEditorView()
        self.polygon_editor.polygonsEdited.connect(self._on_polygons_edited)
        self.polygon_editor.logRequested.connect(self._append_log)
        self.polygon_editor.imageClicked.connect(self._on_editor_image_clicked)
        self.polygon_editor.imageRegionSelected.connect(self._on_editor_image_region_selected)
        self.polygon_editor.rulerMeasurementChanged.connect(self._update_ruler_status)
        self.polygon_editor.toolChanged.connect(self._on_editor_tool_changed)
        self.polygon_editor.zoomChanged.connect(lambda _zoom: self._sync_neighbor_frames())
        self.polygon_editor.neighborFrameActivated.connect(self._on_neighbor_frame_activated)
        self.polygon_editor.viaDebugRequested.connect(self._on_via_debug_requested)
        self.editor_toolbar = self._build_editor_toolbar()
        editor_layout.addWidget(self.editor_toolbar)
        editor_layout.addWidget(self.polygon_editor, 1)

        layout.addWidget(self.editor_group, 1)
        return panel

    def _build_editor_toolbar(self) -> QWidget:
        toolbar = QWidget()
        layout = QHBoxLayout(toolbar)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        self._tool_button_group = QButtonGroup(self)
        self._tool_button_group.setExclusive(True)
        self._tool_buttons: dict[EditorTool, QToolButton] = {}
        for text, tool in [
            ("Select", EditorTool.SELECT),
            ("Select Area", EditorTool.SELECT_AREA),
            ("Pan", EditorTool.PAN),
            ("Ruler", EditorTool.RULER),
            ("Add Polygon", EditorTool.ADD_POLYGON),
            ("Brush", EditorTool.BRUSH),
            ("Via", EditorTool.ADD_VIA),
            ("Add Vertex", EditorTool.ADD_VERTEX),
            ("Delete Vertex", EditorTool.DELETE_VERTEX),
            ("Move Vertex", EditorTool.MOVE_VERTEX),
            ("Delete Polygon", EditorTool.DELETE_POLYGON),
        ]:
            button = QToolButton()
            self._configure_toolbar_button(button, self._create_editor_tool_icon(tool), text, checkable=True)
            button.clicked.connect(lambda checked=False, tool_value=tool: self.polygon_editor.set_tool(tool_value))
            self._tool_button_group.addButton(button)
            self._tool_buttons[tool] = button
            layout.addWidget(button)
            if tool == EditorTool.SELECT:
                button.setChecked(True)

        self.polygon_mode_label = QLabel("Polygon")
        self.polygon_mode_combo = QComboBox()
        self.polygon_mode_combo.addItem(self._mode_text("polygon_points"), PolygonCreateMode.POINTS)
        self.polygon_mode_combo.addItem(self._mode_text("polygon_rectangle"), PolygonCreateMode.RECTANGLE)
        self.polygon_mode_combo.currentIndexChanged.connect(
            lambda _index: self.polygon_editor.set_polygon_create_mode(self.polygon_mode_combo.currentData())
        )
        layout.addWidget(self.polygon_mode_label)
        layout.addWidget(self.polygon_mode_combo)

        self.brush_mode_label = QLabel("Brush")
        self.brush_mode_combo = QComboBox()
        self.brush_mode_combo.addItem(self._mode_text("brush_freeform"), BrushMode.FREEFORM)
        self.brush_mode_combo.addItem(self._mode_text("brush_45deg"), BrushMode.ANGLED)
        self.brush_mode_combo.currentIndexChanged.connect(
            lambda _index: self.polygon_editor.set_brush_mode(self.brush_mode_combo.currentData())
        )
        layout.addWidget(self.brush_mode_label)
        layout.addWidget(self.brush_mode_combo)

        self.brush_size_label = QLabel("Толщина" if self._ui_language == "ru" else "Width")
        self.brush_size_spin = QSpinBox()
        self.brush_size_spin.setRange(1, 256)
        self.brush_size_spin.setValue(12)
        self.brush_size_spin.setFixedWidth(68)
        self.brush_size_spin.valueChanged.connect(
            lambda value: self.polygon_editor.set_brush_thickness(float(value))
        )
        layout.addWidget(self.brush_size_label)
        layout.addWidget(self.brush_size_spin)

        self.via_width_label = QLabel("Via W")
        self.via_width_spin = QSpinBox()
        self.via_width_spin.setRange(1, 100_000)
        self.via_width_spin.setValue(12)
        self.via_width_spin.setFixedWidth(74)
        self.via_height_label = QLabel("Via H")
        self.via_height_spin = QSpinBox()
        self.via_height_spin.setRange(1, 100_000)
        self.via_height_spin.setValue(12)
        self.via_height_spin.setFixedWidth(74)
        self.via_width_spin.valueChanged.connect(lambda _value: self._sync_editor_via_size())
        self.via_height_spin.valueChanged.connect(lambda _value: self._sync_editor_via_size())
        layout.addWidget(self.via_width_label)
        layout.addWidget(self.via_width_spin)
        layout.addWidget(self.via_height_label)
        layout.addWidget(self.via_height_spin)

        self.delete_vertex_mode_label = QLabel("Delete")
        self.delete_vertex_mode_combo = QComboBox()
        self.delete_vertex_mode_combo.addItem(self._mode_text("delete_single"), DeleteVertexMode.SINGLE)
        self.delete_vertex_mode_combo.addItem(self._mode_text("delete_area"), DeleteVertexMode.AREA)
        self.delete_vertex_mode_combo.currentIndexChanged.connect(
            lambda _index: self.polygon_editor.set_delete_vertex_mode(self.delete_vertex_mode_combo.currentData())
        )
        layout.addWidget(self.delete_vertex_mode_label)
        layout.addWidget(self.delete_vertex_mode_combo)

        self.ruler_status_label = QLabel("")
        self.ruler_status_label.setMinimumWidth(180)
        self.ruler_status_label.setVisible(False)
        layout.addWidget(self.ruler_status_label)

        self.undo_button = QToolButton()
        self._configure_toolbar_button(self.undo_button, self._create_editor_action_icon("undo"), "Undo")
        self.undo_button.clicked.connect(self.polygon_editor.undo)
        self.redo_button = QToolButton()
        self._configure_toolbar_button(self.redo_button, self._create_editor_action_icon("redo"), "Redo")
        self.redo_button.clicked.connect(self.polygon_editor.redo)
        self.zoom_in_button = QToolButton()
        self._configure_toolbar_button(self.zoom_in_button, self._create_editor_action_icon("zoom_in"), "Zoom +")
        self.zoom_in_button.clicked.connect(self.polygon_editor.zoom_in)
        self.zoom_out_button = QToolButton()
        self._configure_toolbar_button(self.zoom_out_button, self._create_editor_action_icon("zoom_out"), "Zoom -")
        self.zoom_out_button.clicked.connect(self.polygon_editor.zoom_out)
        self.fit_button = QToolButton()
        self._configure_toolbar_button(self.fit_button, self._create_editor_action_icon("fit"), "Fit")
        self.fit_button.clicked.connect(self.polygon_editor.fit_to_view)

        for button in [
            self.undo_button,
            self.redo_button,
            self.zoom_in_button,
            self.zoom_out_button,
            self.fit_button,
        ]:
            layout.addWidget(button)

        self.preview_busy_label = QLabel(self._busy_indicator_text())
        self.preview_busy_progress = QProgressBar()
        self.preview_busy_progress.setRange(0, 0)
        self.preview_busy_progress.setTextVisible(False)
        self.preview_busy_progress.setFixedWidth(88)
        self.preview_busy_label.setVisible(False)
        self.preview_busy_progress.setVisible(False)
        layout.addWidget(self.preview_busy_label)
        layout.addWidget(self.preview_busy_progress)
        layout.addStretch(1)
        self.polygon_editor.set_polygon_create_mode(self.polygon_mode_combo.currentData())
        self.polygon_editor.set_brush_mode(self.brush_mode_combo.currentData())
        self.polygon_editor.set_brush_thickness(float(self.brush_size_spin.value()))
        self._sync_editor_via_size()
        self.polygon_editor.set_delete_vertex_mode(self.delete_vertex_mode_combo.currentData())
        return toolbar

    def _sync_editor_via_size(self) -> None:
        self.polygon_editor.set_via_size(float(self.via_width_spin.value()), float(self.via_height_spin.value()))

    def _configure_toolbar_button(
        self,
        button: QToolButton,
        icon: QIcon,
        text: str,
        *,
        checkable: bool = False,
    ) -> None:
        button.setIcon(icon)
        button.setIconSize(QSize(self._toolbar_icon_size_px(), self._toolbar_icon_size_px()))
        button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonIconOnly)
        button.setToolTip(text)
        button.setStatusTip(text)
        button.setAccessibleName(text)
        button.setAutoRaise(False)
        button.setFixedSize(self._toolbar_button_size_px(), self._toolbar_button_size_px())
        button.setCheckable(checkable)

    def _create_editor_tool_icon(self, tool: EditorTool) -> QIcon:
        canvas_size = self._toolbar_icon_canvas_size_px()
        pixmap = QPixmap(canvas_size, canvas_size)
        pixmap.fill(Qt.GlobalColor.transparent)

        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        scale_factor = canvas_size / 28.0
        painter.scale(scale_factor, scale_factor)

        stroke = QColor("#FFFFFF")
        neutral = QColor("#E2E8F0")
        accent = QColor("#38BDF8")
        success = QColor("#4ADE80")
        warning = QColor("#FDBA74")
        danger = QColor("#FB7185")

        if tool == EditorTool.SELECT:
            self._paint_select_icon(painter, stroke)
        elif tool == EditorTool.SELECT_AREA:
            self._paint_select_area_icon(painter, stroke, accent)
        elif tool == EditorTool.PAN:
            self._paint_pan_icon(painter, stroke, accent)
        elif tool == EditorTool.RULER:
            self._paint_ruler_icon(painter, stroke, warning)
        elif tool == EditorTool.ADD_POLYGON:
            self._paint_polygon_badge_icon(painter, stroke, accent, "+")
        elif tool == EditorTool.BRUSH:
            self._paint_brush_icon(painter, stroke, success)
        elif tool == EditorTool.ADD_VIA:
            self._paint_via_icon(painter, stroke, QColor("#A78BFA"))
        elif tool == EditorTool.ADD_VERTEX:
            self._paint_vertex_edit_icon(painter, stroke, neutral, success, "+")
        elif tool == EditorTool.DELETE_VERTEX:
            self._paint_vertex_edit_icon(painter, stroke, neutral, danger, "-")
        elif tool == EditorTool.MOVE_VERTEX:
            self._paint_move_vertex_icon(painter, stroke, warning)
        elif tool == EditorTool.DELETE_POLYGON:
            self._paint_polygon_badge_icon(painter, stroke, danger, "x")
        painter.end()
        return QIcon(pixmap)

    def _create_editor_action_icon(self, action: str) -> QIcon:
        canvas_size = self._toolbar_icon_canvas_size_px()
        pixmap = QPixmap(canvas_size, canvas_size)
        pixmap.fill(Qt.GlobalColor.transparent)

        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        scale_factor = canvas_size / 28.0
        painter.scale(scale_factor, scale_factor)
        stroke = QColor("#FFFFFF")
        accent = QColor("#38BDF8")

        if action == "undo":
            self._paint_history_icon(painter, stroke, mirrored=False)
        elif action == "redo":
            self._paint_history_icon(painter, stroke, mirrored=True)
        elif action == "zoom_in":
            self._paint_zoom_icon(painter, stroke, accent, add=True)
        elif action == "zoom_out":
            self._paint_zoom_icon(painter, stroke, accent, add=False)
        else:
            self._paint_fit_icon(painter, stroke, accent)
        painter.end()
        return QIcon(pixmap)

    @staticmethod
    def _toolbar_icon_size_px() -> int:
        return 28

    @staticmethod
    def _toolbar_button_size_px() -> int:
        return 34

    @staticmethod
    def _toolbar_icon_canvas_size_px() -> int:
        return 72

    def _paint_select_icon(self, painter: QPainter, stroke: QColor) -> None:
        path = QPainterPath()
        path.moveTo(5.5, 4.0)
        path.lineTo(5.5, 21.0)
        path.lineTo(10.0, 16.5)
        path.lineTo(12.8, 23.0)
        path.lineTo(16.0, 21.8)
        path.lineTo(13.2, 15.6)
        path.lineTo(20.8, 15.6)
        path.closeSubpath()
        painter.setPen(QPen(stroke, 1.9, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
        painter.fillPath(path, QBrush(QColor("#FFFFFF")))
        painter.drawPath(path)

    def _paint_select_area_icon(self, painter: QPainter, stroke: QColor, accent: QColor) -> None:
        painter.setPen(QPen(accent, 1.8, Qt.PenStyle.DashLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
        painter.setBrush(QBrush(Qt.BrushStyle.NoBrush))
        painter.drawRect(QRectF(5.0, 6.0, 16.0, 13.0))
        self._paint_select_icon(painter, stroke)

    def _paint_pan_icon(self, painter: QPainter, stroke: QColor, accent: QColor) -> None:
        center = QPointF(14.0, 14.0)
        painter.setPen(QPen(stroke, 2.2, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        painter.drawLine(QPointF(14.0, 5.0), QPointF(14.0, 23.0))
        painter.drawLine(QPointF(5.0, 14.0), QPointF(23.0, 14.0))
        self._draw_arrow_head(painter, QPointF(14.0, 5.0), QPointF(14.0, 2.0))
        self._draw_arrow_head(painter, QPointF(14.0, 23.0), QPointF(14.0, 26.0))
        self._draw_arrow_head(painter, QPointF(5.0, 14.0), QPointF(2.0, 14.0))
        self._draw_arrow_head(painter, QPointF(23.0, 14.0), QPointF(26.0, 14.0))
        painter.setPen(QPen(accent, 1.8))
        painter.setBrush(QBrush(accent))
        painter.drawEllipse(QRectF(center.x() - 2.2, center.y() - 2.2, 4.4, 4.4))

    def _paint_ruler_icon(self, painter: QPainter, stroke: QColor, accent: QColor) -> None:
        start = QPointF(5.0, 19.5)
        end = QPointF(23.0, 8.5)
        painter.setPen(QPen(stroke, 2.0, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        painter.drawLine(start, end)
        painter.setPen(QPen(accent, 1.6, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        for tick in range(1, 5):
            base_x = 6.5 + tick * 3.4
            base_y = 18.6 - tick * 2.1
            painter.drawLine(QPointF(base_x, base_y), QPointF(base_x - 1.0, base_y - 1.8))
        self._draw_vertex_marker(painter, start, stroke, QColor("#FFFFFF"), radius=1.6)
        self._draw_vertex_marker(painter, end, stroke, QColor("#FFFFFF"), radius=1.6)

    def _paint_polygon_badge_icon(
        self,
        painter: QPainter,
        stroke: QColor,
        badge_color: QColor,
        badge_symbol: str,
    ) -> None:
        polygon = QPolygonF(
            [
                QPointF(4.5, 18.5),
                QPointF(8.6, 7.0),
                QPointF(18.0, 8.6),
                QPointF(20.0, 18.0),
                QPointF(12.0, 22.0),
            ]
        )
        painter.setPen(QPen(stroke, 2.1, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawPolygon(polygon)
        for point in polygon:
            self._draw_vertex_marker(painter, point, stroke, QColor("#FFFFFF"), radius=1.9)
        self._draw_badge(painter, QPointF(20.5, 6.5), badge_color, badge_symbol)

    def _paint_vertex_edit_icon(
        self,
        painter: QPainter,
        stroke: QColor,
        neutral: QColor,
        badge_color: QColor,
        badge_symbol: str,
    ) -> None:
        polyline = [QPointF(4.5, 18.0), QPointF(11.0, 8.0), QPointF(19.0, 18.2)]
        painter.setPen(QPen(stroke, 2.1, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
        painter.drawPolyline(QPolygonF(polyline))
        self._draw_vertex_marker(painter, polyline[0], stroke, QColor("#FFFFFF"), radius=1.8)
        self._draw_vertex_marker(painter, polyline[2], stroke, QColor("#FFFFFF"), radius=1.8)
        self._draw_vertex_marker(painter, polyline[1], stroke, neutral, radius=2.4)
        self._draw_badge(painter, QPointF(20.0, 6.5), badge_color, badge_symbol)

    def _paint_move_vertex_icon(self, painter: QPainter, stroke: QColor, accent: QColor) -> None:
        polygon = QPolygonF(
            [
                QPointF(4.5, 18.4),
                QPointF(8.8, 8.0),
                QPointF(17.0, 9.0),
                QPointF(19.6, 17.5),
                QPointF(11.4, 21.2),
            ]
        )
        painter.setPen(QPen(stroke, 2.1, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawPolygon(polygon)
        target = QPointF(17.0, 9.0)
        self._draw_vertex_marker(painter, target, stroke, accent, radius=2.5)
        painter.setPen(QPen(accent, 1.9, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        painter.drawLine(QPointF(17.0, 4.2), QPointF(17.0, 13.8))
        painter.drawLine(QPointF(12.2, 9.0), QPointF(21.8, 9.0))
        self._draw_arrow_head(painter, QPointF(17.0, 4.2), QPointF(17.0, 1.6))
        self._draw_arrow_head(painter, QPointF(17.0, 13.8), QPointF(17.0, 16.4))
        self._draw_arrow_head(painter, QPointF(12.2, 9.0), QPointF(9.6, 9.0))
        self._draw_arrow_head(painter, QPointF(21.8, 9.0), QPointF(24.4, 9.0))

    def _paint_brush_icon(self, painter: QPainter, stroke: QColor, accent: QColor) -> None:
        painter.setPen(QPen(stroke, 2.0, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
        path = QPainterPath()
        path.moveTo(7.0, 20.5)
        path.cubicTo(9.0, 15.0, 13.0, 10.0, 18.0, 6.5)
        path.lineTo(21.0, 9.5)
        path.cubicTo(17.5, 14.5, 12.5, 18.5, 7.0, 20.5)
        painter.drawPath(path)
        painter.setBrush(QBrush(accent))
        painter.setPen(QPen(accent, 1.0))
        painter.drawEllipse(QRectF(18.8, 5.2, 4.6, 4.6))

    def _paint_via_icon(self, painter: QPainter, stroke: QColor, accent: QColor) -> None:
        painter.setPen(QPen(stroke, 2.0, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRect(QRectF(6.0, 7.0, 16.0, 14.0))
        painter.setPen(QPen(accent, 1.6, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        painter.setBrush(QBrush(accent))
        painter.drawEllipse(QRectF(10.0, 9.0, 8.0, 8.0))
        painter.drawLine(QPointF(14.0, 4.5), QPointF(14.0, 7.0))
        painter.drawLine(QPointF(14.0, 21.0), QPointF(14.0, 23.5))
        painter.drawLine(QPointF(3.5, 14.0), QPointF(6.0, 14.0))
        painter.drawLine(QPointF(22.0, 14.0), QPointF(24.5, 14.0))

    def _paint_history_icon(self, painter: QPainter, stroke: QColor, mirrored: bool) -> None:
        painter.setPen(QPen(stroke, 2.2, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
        path = QPainterPath()
        if mirrored:
            path.moveTo(7.0, 8.0)
            path.cubicTo(13.0, 3.5, 22.5, 6.0, 22.0, 14.0)
            path.cubicTo(21.5, 21.0, 13.5, 23.5, 8.5, 20.0)
            painter.drawPath(path)
            self._draw_arrow_head(painter, QPointF(7.2, 8.0), QPointF(3.8, 9.0))
        else:
            path.moveTo(21.0, 8.0)
            path.cubicTo(15.0, 3.5, 5.5, 6.0, 6.0, 14.0)
            path.cubicTo(6.5, 21.0, 14.5, 23.5, 19.5, 20.0)
            painter.drawPath(path)
            self._draw_arrow_head(painter, QPointF(20.8, 8.0), QPointF(24.2, 9.0))

    def _paint_zoom_icon(self, painter: QPainter, stroke: QColor, accent: QColor, *, add: bool) -> None:
        painter.setPen(QPen(stroke, 2.1, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawEllipse(QRectF(5.0, 5.0, 12.0, 12.0))
        painter.drawLine(QPointF(15.2, 15.2), QPointF(22.8, 22.8))
        painter.setPen(QPen(accent, 2.0, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        painter.drawLine(QPointF(8.5, 11.0), QPointF(13.5, 11.0))
        if add:
            painter.drawLine(QPointF(11.0, 8.5), QPointF(11.0, 13.5))

    def _paint_fit_icon(self, painter: QPainter, stroke: QColor, accent: QColor) -> None:
        painter.setPen(QPen(stroke, 2.0, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        painter.drawRect(QRectF(7.0, 7.0, 14.0, 14.0))
        painter.setPen(QPen(accent, 2.0, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        painter.drawLine(QPointF(5.0, 10.0), QPointF(9.0, 10.0))
        painter.drawLine(QPointF(10.0, 5.0), QPointF(10.0, 9.0))
        painter.drawLine(QPointF(19.0, 5.0), QPointF(19.0, 9.0))
        painter.drawLine(QPointF(19.0, 19.0), QPointF(19.0, 23.0))
        painter.drawLine(QPointF(5.0, 19.0), QPointF(9.0, 19.0))
        painter.drawLine(QPointF(19.0, 19.0), QPointF(23.0, 19.0))

    def _draw_vertex_marker(
        self,
        painter: QPainter,
        point: QPointF,
        stroke: QColor,
        fill: QColor,
        radius: float,
    ) -> None:
        painter.setPen(QPen(stroke, 1.2))
        painter.setBrush(QBrush(fill))
        painter.drawEllipse(QRectF(point.x() - radius, point.y() - radius, radius * 2.0, radius * 2.0))

    def _draw_badge(self, painter: QPainter, center: QPointF, color: QColor, symbol: str) -> None:
        badge_rect = QRectF(center.x() - 4.3, center.y() - 4.3, 8.6, 8.6)
        painter.setPen(QPen(color.darker(120), 1.0))
        painter.setBrush(QBrush(color))
        painter.drawEllipse(badge_rect)
        painter.setPen(QPen(QColor("#FFFFFF"), 1.7, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        if symbol == "+":
            painter.drawLine(QPointF(center.x() - 2.2, center.y()), QPointF(center.x() + 2.2, center.y()))
            painter.drawLine(QPointF(center.x(), center.y() - 2.2), QPointF(center.x(), center.y() + 2.2))
        elif symbol == "-":
            painter.drawLine(QPointF(center.x() - 2.2, center.y()), QPointF(center.x() + 2.2, center.y()))
        else:
            painter.drawLine(
                QPointF(center.x() - 1.9, center.y() - 1.9),
                QPointF(center.x() + 1.9, center.y() + 1.9),
            )
            painter.drawLine(
                QPointF(center.x() - 1.9, center.y() + 1.9),
                QPointF(center.x() + 1.9, center.y() - 1.9),
            )

    def _draw_arrow_head(self, painter: QPainter, base: QPointF, tip: QPointF) -> None:
        vector_x = tip.x() - base.x()
        vector_y = tip.y() - base.y()
        if abs(vector_x) >= abs(vector_y):
            direction = 1.0 if vector_x >= 0 else -1.0
            left = QPointF(base.x() + 1.6 * direction, base.y() - 1.4)
            right = QPointF(base.x() + 1.6 * direction, base.y() + 1.4)
        else:
            direction = 1.0 if vector_y >= 0 else -1.0
            left = QPointF(base.x() - 1.4, base.y() + 1.6 * direction)
            right = QPointF(base.x() + 1.4, base.y() + 1.6 * direction)
        painter.drawLine(base, left)
        painter.drawLine(base, right)

    def _tr(self, key: str, default: str = "", **kwargs) -> str:
        return tr(key, default=default, language=self._ui_language, **kwargs)

    def _set_common_tooltip(self, widget: QWidget | None, key: str) -> None:
        if widget is None:
            return
        tooltip = _localized_text(GENERAL_CONTROL_TOOLTIPS, key, self._ui_language)
        widget.setToolTip(tooltip)
        widget.setStatusTip(tooltip)

    def _mode_text(self, key: str) -> str:
        if self._ui_language == "ru":
            mapping = {
                "polygon_points": "По точкам",
                "polygon_rectangle": "Прямоугольник",
                "brush_freeform": "Произвольная",
                "brush_45deg": "45° шаг",
                "delete_single": "Вершина",
                "delete_area": "Область",
            }
        else:
            mapping = {
                "polygon_points": "By points",
                "polygon_rectangle": "Rectangle",
                "brush_freeform": "Freeform",
                "brush_45deg": "45° constrained",
                "delete_single": "Single vertex",
                "delete_area": "Area",
            }
        return mapping[key]

    def _busy_indicator_text(self) -> str:
        return "Обработка..." if self._ui_language == "ru" else "Processing..."

    def _set_progress_status(self, key: str, **kwargs) -> None:
        self._progress_status_key = key
        self._progress_status_kwargs = dict(kwargs)

    def set_ui_language(self, language: str | None) -> None:
        self._ui_language = active_language(language)
        self._batch_processor.set_ui_language(self._ui_language)
        if hasattr(self, "polygon_editor"):
            self.polygon_editor.set_ui_language(self._ui_language)
        self._retranslate_ui()

    def _retranslate_ui(self) -> None:
        if not hasattr(self, "control_tabs"):
            return
        selected_operation = self._selected_available_operation_name()
        selected_pipeline_row = self.pipeline_list.currentRow() if hasattr(self, "pipeline_list") else -1

        self.path_group.setTitle(self._tr("path_panel_title"))
        self.input_dir_label.setText(self._tr("input_directory_label"))
        self.cif_dir_label.setText(self._tr("cif_overlay_directory_label"))
        self.output_dir_label.setText(self._tr("output_directory_label"))
        self.dataset_dir_label.setText(self._tr("dataset_directory_label"))
        for button, accessible_name in (
            (self.browse_input_button, self._tr("browse_input_button")),
            (self.browse_cif_button, self._tr("browse_cif_button")),
            (self.browse_output_button, self._tr("browse_output_button")),
            (self.browse_dataset_button, self._tr("browse_dataset_button")),
            (self.refresh_button, self._tr("refresh_files_button")),
        ):
            button.setText("")
            button.setAccessibleName(accessible_name)
        for widget, tooltip_key in (
            (self.input_dir_label, "input_dir"),
            (self.input_dir_edit, "input_dir"),
            (self.cif_dir_label, "cif_dir"),
            (self.cif_dir_edit, "cif_dir"),
            (self.output_dir_label, "output_dir"),
            (self.output_dir_edit, "output_dir"),
            (self.dataset_dir_label, "dataset_dir"),
            (self.dataset_dir_edit, "dataset_dir"),
            (self.browse_input_button, "browse_input"),
            (self.browse_cif_button, "browse_cif"),
            (self.browse_output_button, "browse_output"),
            (self.browse_dataset_button, "browse_dataset"),
            (self.refresh_button, "refresh_files"),
        ):
            self._set_common_tooltip(widget, tooltip_key)

        self.control_tabs.setTabText(0, self._tr("tab_paths"))
        self.control_tabs.setTabText(1, self._tr("tab_pipeline"))
        self.control_tabs.setTabText(2, self._tr("tab_extraction"))
        self.control_tabs.setTabText(3, self._tr("tab_display"))
        if hasattr(self, "right_tabs"):
            self.right_tabs.setTabText(0, self._tr("tab_files"))

        self.images_label.setText(self._tr("images_label"))
        self.run_group.setTitle(self._tr("run_group_title"))
        for button, accessible_name in (
            (self.process_current_button, self._tr("process_current_button")),
            (self.batch_button, self._tr("start_batch_button")),
            (self.stop_batch_button, self._tr("stop_batch_button")),
        ):
            button.setText("")
            button.setAccessibleName(accessible_name)
        self.save_current_button.setText(self._tr("save_current_button"))
        self.export_dataset_button.setText(self._tr("export_dataset_button"))
        self.dataset_mode_checkbox.setText(self._tr("dataset_mode_checkbox"))
        self.max_workers_label.setText(self._tr("max_workers_label"))
        for widget, tooltip_key in (
            (self.image_list, "image_list"),
            (self.images_label, "image_list"),
            (self.process_current_button, "process_current"),
            (self.batch_button, "start_batch"),
            (self.stop_batch_button, "stop_batch"),
            (self.save_current_button, "save_current"),
            (self.export_dataset_button, "export_dataset"),
            (self.dataset_mode_checkbox, "dataset_mode"),
            (self.max_workers_label, "max_workers"),
            (self.max_workers_spin, "max_workers"),
        ):
            self._set_common_tooltip(widget, tooltip_key)

        self.available_filters_group.setTitle(
            self._tr("available_filters_group_title", "Фильтры pipeline" if self._ui_language == "ru" else "Pipeline filters")
        )
        self.pipeline_steps_group.setTitle(
            self._tr("applied_filters_group_title", "Примененные фильтры" if self._ui_language == "ru" else "Applied filters")
        )
        self.pipeline_help_group.setTitle(
            self._tr("pipeline_help_group_title", "Справка по фильтру" if self._ui_language == "ru" else "Filter help")
        )
        self.pipeline_help_before_title.setText("До" if self._ui_language == "ru" else "Before")
        self.pipeline_help_after_title.setText("После" if self._ui_language == "ru" else "After")
        self.save_pipeline_button.setText(self._tr("save_json_button"))
        self.load_pipeline_button.setText(self._tr("load_json_button"))
        self.auto_tune_button.setText(
            self._tr(
                "auto_tune_button",
                "Автоподбор по рисунку" if self._ui_language == "ru" else "Auto-fit from drawing",
            )
        )
        self.auto_tune_button.setToolTip(
            self._tr(
                "auto_tune_button_tooltip",
                "Использовать текущие нарисованные полигоны как эталон"
                if self._ui_language == "ru"
                else "Use the currently drawn polygons as the fitting target",
            )
        )
        for widget, tooltip_key in (
            (self.save_pipeline_button, "save_json_button"),
            (self.load_pipeline_button, "load_json_button"),
            (self.auto_tune_button, "auto_tune_button"),
        ):
            tooltip = _localized_text(PIPELINE_CONTROL_TOOLTIPS, tooltip_key, self._ui_language)
            widget.setToolTip(tooltip)
            widget.setStatusTip(tooltip)
        self.parameters_group.setTitle(self._tr("step_parameters_group"))

        self.contour_group.setTitle(self._tr("contour_extraction_group"))
        self.profile_group.setTitle(self._tr("extraction_profile_group_title", "Профиль" if self._ui_language == "ru" else "Profile"))
        self.basic_filters_group.setTitle(self._tr("basic_filters_group_title", "Базовые фильтры" if self._ui_language == "ru" else "Basic filters"))
        self.geometry_filters_group.setTitle(self._tr("geometry_filters_group_title", "Геометрия" if self._ui_language == "ru" else "Geometry"))
        self.via_group.setTitle(self._tr("via_constraints_group_title", "Ограничения via" if self._ui_language == "ru" else "Via constraints"))
        self.topology_group.setTitle(self._tr("topology_group_title", "Иерархия и отверстия" if self._ui_language == "ru" else "Hierarchy and holes"))
        if self.extraction_profile_label_widget is not None:
            self.extraction_profile_label_widget.setText(self._tr("extraction_profile_label"))
        self.extraction_profile_combo.setItemText(0, self._tr("extraction_profile_conductors"))
        self.extraction_profile_combo.setItemText(1, self._tr("extraction_profile_vias"))
        if self.retrieval_mode_label_widget is not None:
            self.retrieval_mode_label_widget.setText(self._tr("retrieval_mode_label"))
        if self.approximation_mode_label_widget is not None:
            self.approximation_mode_label_widget.setText(self._tr("approximation_mode_label"))
        if self.epsilon_label_widget is not None:
            self.epsilon_label_widget.setText(self._tr("epsilon_label"))
        if self.epsilon_mode_label_widget is not None:
            self.epsilon_mode_label_widget.setText(self._tr("epsilon_mode_label"))
        self.epsilon_relative_checkbox.setText(self._tr("epsilon_relative_checkbox"))
        if self.min_area_label_widget is not None:
            self.min_area_label_widget.setText(self._tr("area_range_label", "Диапазон площади" if self._ui_language == "ru" else "Area range"))
        if self.min_perimeter_label_widget is not None:
            self.min_perimeter_label_widget.setText(self._tr("perimeter_range_label", "Диапазон периметра" if self._ui_language == "ru" else "Perimeter range"))
        if self.min_point_count_label_widget is not None:
            self.min_point_count_label_widget.setText(self._tr("min_point_count_label"))
        if self.min_bbox_width_label_widget is not None:
            self.min_bbox_width_label_widget.setText(self._tr("bbox_width_range_label", "Диапазон ширины bbox" if self._ui_language == "ru" else "BBox width range"))
        if self.min_bbox_height_label_widget is not None:
            self.min_bbox_height_label_widget.setText(self._tr("bbox_height_range_label", "Диапазон высоты bbox" if self._ui_language == "ru" else "BBox height range"))
        if self.min_aspect_ratio_label_widget is not None:
            self.min_aspect_ratio_label_widget.setText(self._tr("aspect_ratio_range_label", "Диапазон aspect ratio" if self._ui_language == "ru" else "Aspect ratio range"))
        if self.border_handling_label_widget is not None:
            self.border_handling_label_widget.setText(self._tr("border_handling_label"))
        self.exclude_border_touching_checkbox.setText(
            self._tr("exclude_border_touching_checkbox_short", "Исключать" if self._ui_language == "ru" else "Exclude")
        )
        if self.min_solidity_label_widget is not None:
            self.min_solidity_label_widget.setText(self._tr("min_solidity_label"))
        if self.min_extent_label_widget is not None:
            self.min_extent_label_widget.setText(self._tr("min_extent_label"))
        if self.via_size_mode_label_widget is not None:
            self.via_size_mode_label_widget.setText(
                self._tr("via_size_mode_label", "Режим размеров via" if self._ui_language == "ru" else "Via size mode")
            )
        self.via_size_mode_combo.setItemText(
            0,
            self._tr("via_size_mode_range", "Диапазон" if self._ui_language == "ru" else "Range"),
        )
        self.via_size_mode_combo.setItemText(
            1,
            self._tr("via_size_mode_fixed", "Фиксированные значения" if self._ui_language == "ru" else "Fixed values"),
        )
        if self.via_white_range_label_widget is not None:
            self.via_white_range_label_widget.setText(
                self._tr("via_white_range_label", "Диапазон белых" if self._ui_language == "ru" else "White range")
            )
        self.via_white_range_checkbox.setText("Вкл." if self._ui_language == "ru" else "Enabled")
        if self.via_black_range_label_widget is not None:
            self.via_black_range_label_widget.setText(
                self._tr("via_black_range_label", "Диапазон чёрных" if self._ui_language == "ru" else "Black range")
            )
        self.via_black_range_checkbox.setText("Вкл." if self._ui_language == "ru" else "Enabled")
        if self.via_detector_methods_label_widget is not None:
            self.via_detector_methods_label_widget.setText(
                self._tr("via_detector_methods_label", "Методы поиска" if self._ui_language == "ru" else "Detection methods")
            )
        self.via_white_range_checkbox.setText(self._tr("via_white_range_method", "Диапазон белых" if self._ui_language == "ru" else "White range"))
        self.via_black_range_checkbox.setText(self._tr("via_black_range_method", "Диапазон черных" if self._ui_language == "ru" else "Black range"))
        self.via_detector_gradient_checkbox.setText(
            self._tr("via_detector_gradient", "Градиент" if self._ui_language == "ru" else "Gradient")
        )
        self.via_detector_spot_checkbox.setText(self._tr("via_detector_spot", "Точки" if self._ui_language == "ru" else "Spots"))
        self.via_detector_hough_checkbox.setText(self._tr("via_detector_hough", "Hough" if self._ui_language == "ru" else "Hough"))
        self.via_detector_components_checkbox.setText(
            self._tr("via_detector_components", "Компоненты" if self._ui_language == "ru" else "Components")
        )
        self.via_detector_contours_checkbox.setText(
            self._tr("via_detector_contours", "Контуры" if self._ui_language == "ru" else "Contours")
        )
        self.via_detector_morphology_checkbox.setText(
            self._tr("via_detector_morphology", "Морфология" if self._ui_language == "ru" else "Morphology")
        )
        self.via_detector_template_checkbox.setText(
            self._tr("via_detector_template", "Шаблон" if self._ui_language == "ru" else "Template")
        )
        self.via_detector_blob_checkbox.setText(self._tr("via_detector_blob", "Blob" if self._ui_language == "ru" else "Blob"))
        if self.via_gradient_min_strength_label_widget is not None:
            self.via_gradient_min_strength_label_widget.setText(
                self._tr("via_gradient_min_strength_label", "Мин. перепад" if self._ui_language == "ru" else "Min edge")
            )
        if self.via_gradient_min_coverage_label_widget is not None:
            self.via_gradient_min_coverage_label_widget.setText(
                self._tr("via_gradient_min_coverage_label", "Доля окружности" if self._ui_language == "ru" else "Circle coverage")
            )
        if self.via_spot_min_contrast_label_widget is not None:
            self.via_spot_min_contrast_label_widget.setText(
                self._tr("via_spot_min_contrast_label", "Точки: контраст" if self._ui_language == "ru" else "Spots contrast")
            )
        if self.via_spot_min_roundness_label_widget is not None:
            self.via_spot_min_roundness_label_widget.setText(
                self._tr("via_spot_min_roundness_label", "Точки: компактность" if self._ui_language == "ru" else "Spots compactness")
            )
        if self.via_spot_line_suppression_label_widget is not None:
            self.via_spot_line_suppression_label_widget.setText(
                self._tr(
                    "via_spot_line_suppression_label",
                    "\u0422\u043e\u0447\u043a\u0438: \u0434\u043e\u0440\u043e\u0436\u043a\u0438" if self._ui_language == "ru" else "Spots traces",
                )
            )
        if self.via_hough_edge_threshold_label_widget is not None:
            self.via_hough_edge_threshold_label_widget.setText(
                self._tr("via_hough_edge_threshold_label", "Hough: край" if self._ui_language == "ru" else "Hough edge")
            )
        if self.via_hough_accumulator_threshold_label_widget is not None:
            self.via_hough_accumulator_threshold_label_widget.setText(
                self._tr("via_hough_accumulator_threshold_label", "Hough: голоса" if self._ui_language == "ru" else "Hough votes")
            )
        if self.via_component_min_score_label_widget is not None:
            self.via_component_min_score_label_widget.setText(
                self._tr("via_component_min_score_label", "Компоненты: отклик" if self._ui_language == "ru" else "Components score")
            )
        if self.via_contour_min_score_label_widget is not None:
            self.via_contour_min_score_label_widget.setText(
                self._tr("via_contour_min_score_label", "Контуры: отклик" if self._ui_language == "ru" else "Contours score")
            )
        if self.via_morphology_peak_scale_label_widget is not None:
            self.via_morphology_peak_scale_label_widget.setText(
                self._tr("via_morphology_peak_scale_label", "Морфология: пик" if self._ui_language == "ru" else "Morphology peak")
            )
        if self.via_template_min_score_label_widget is not None:
            self.via_template_min_score_label_widget.setText(
                self._tr("via_template_min_score_label", "Шаблон: совпадение" if self._ui_language == "ru" else "Template score")
            )
        if self.via_templates_label_widget is not None:
            self.via_templates_label_widget.setText(self._tr("via_templates_label", "Шаблоны" if self._ui_language == "ru" else "Templates"))
        self.add_via_template_button.setText(
            self._tr("add_via_template_button", "Выделить шаблон" if self._ui_language == "ru" else "Pick template")
        )
        self.remove_via_template_button.setText(
            self._tr("remove_via_template_button", "Удалить выбранный" if self._ui_language == "ru" else "Remove selected")
        )
        self.clear_via_templates_button.setText(
            self._tr("clear_via_templates_button", "Удалить все" if self._ui_language == "ru" else "Clear all")
        )
        if self.via_blob_min_circularity_label_widget is not None:
            self.via_blob_min_circularity_label_widget.setText(
                self._tr("via_blob_min_circularity_label", "Blob: округлость" if self._ui_language == "ru" else "Blob circularity")
            )
        if self.noisy_traces_via_preset_label_widget is not None:
            self.noisy_traces_via_preset_label_widget.setText("")
        if self.via_preset_label_widget is not None:
            self.via_preset_label_widget.setText(
                self._tr("via_preset_label", "Пресеты поиска via" if self._ui_language == "ru" else "Via search presets")
            )
        self.apply_via_preset_button.setText(self._tr("apply_via_preset_button", "Применить" if self._ui_language == "ru" else "Apply"))
        self.save_via_preset_button.setText(self._tr("save_via_preset_button", "Сохранить" if self._ui_language == "ru" else "Save"))
        self.delete_via_preset_button.setText(self._tr("delete_via_preset_button", "Удалить" if self._ui_language == "ru" else "Delete"))
        self.noisy_traces_via_preset_button.setText(
            self._tr(
                "noisy_traces_via_preset_button",
                "Пресет: яркие via на дорожках" if self._ui_language == "ru" else "Preset: bright vias on traces",
            )
        )
        if self.blurred_via_preset_label_widget is not None:
            self.blurred_via_preset_label_widget.setText("")
        self.blurred_via_preset_button.setText(
            self._tr(
                "blurred_via_preset_button",
                "Пресет: слабые/размытые via" if self._ui_language == "ru" else "Preset: weak/blurred vias",
            )
        )
        self._refresh_via_preset_combo()
        if self.reset_via_search_label_widget is not None:
            self.reset_via_search_label_widget.setText("")
        self.reset_via_search_button.setText(
            self._tr("reset_via_search_button", "Сбросить параметры поиска via" if self._ui_language == "ru" else "Reset via search parameters")
        )
        if self.debug_candidates_label_widget is not None:
            self.debug_candidates_label_widget.setText(
                self._tr("debug_candidates_label", "Отладка via" if self._ui_language == "ru" else "Via debug")
            )
        self.debug_candidates_checkbox.setText(
            self._tr("debug_candidates_checkbox", "Проверять по клику" if self._ui_language == "ru" else "Inspect by click")
        )
        if self.via_roundness_label_widget is not None:
            self.via_roundness_label_widget.setText(self._tr("via_roundness_label", "Округлость" if self._ui_language == "ru" else "Roundness"))
        if self.min_via_width_label_widget is not None:
            self.min_via_width_label_widget.setText(self._tr("via_width_range_label", "Диапазон ширины via" if self._ui_language == "ru" else "Via width range"))
        if self.min_via_height_label_widget is not None:
            self.min_via_height_label_widget.setText(self._tr("via_height_range_label", "Диапазон высоты via" if self._ui_language == "ru" else "Via height range"))
        if self.fixed_vias_label_widget is not None:
            self.fixed_vias_label_widget.setText(
                self._tr("fixed_vias_label", "Фиксированные via" if self._ui_language == "ru" else "Fixed vias")
            )
        if self.min_hierarchy_depth_label_widget is not None:
            self.min_hierarchy_depth_label_widget.setText(self._tr("min_hierarchy_depth_label"))
        if self.max_hierarchy_depth_label_widget is not None:
            self.max_hierarchy_depth_label_widget.setText(self._tr("max_hierarchy_depth_label"))
        if self.max_hole_area_ratio_label_widget is not None:
            self.max_hole_area_ratio_label_widget.setText(self._tr("max_hole_area_ratio_label"))
        self.save_group.setTitle(self._tr("save_options_group"))
        self.save_svg_checkbox.setText(self._tr("save_svg_checkbox"))
        self.save_preview_checkbox.setText(self._tr("save_preview_checkbox"))
        self._set_common_tooltip(self.save_svg_checkbox, "save_svg")
        self._set_common_tooltip(self.save_preview_checkbox, "save_preview")
        self._apply_extraction_tooltips()
        self._renumber_fixed_via_rows()
        self._update_extraction_profile_controls_state()

        if self.external_color_label_widget is not None:
            self.external_color_label_widget.setText(self._tr("external_contour_label"))
        if self.hole_color_label_widget is not None:
            self.hole_color_label_widget.setText(self._tr("hole_contour_label"))
        if self.selected_color_label_widget is not None:
            self.selected_color_label_widget.setText(self._tr("selected_contour_label"))
        if self.vertex_color_label_widget is not None:
            self.vertex_color_label_widget.setText(self._tr("vertex_color_label"))
        if self.line_width_label_widget is not None:
            self.line_width_label_widget.setText(self._tr("line_width_label"))
        if self.vertex_size_label_widget is not None:
            self.vertex_size_label_widget.setText(self._tr("vertex_size_label"))
        if self.fill_opacity_label_widget is not None:
            self.fill_opacity_label_widget.setText(self._tr("fill_opacity_label"))
        self.show_vertices_checkbox.setText(self._tr("show_vertices_checkbox"))
        self.show_labels_checkbox.setText(self._tr("show_labels_checkbox"))
        self.random_object_colors_checkbox.setText(
            self._tr("random_object_colors_checkbox", "Случайные цвета объектов" if self._ui_language == "ru" else "Random object colors")
        )
        self.show_neighbor_frames_checkbox.setText(
            self._tr("show_neighbor_frames_checkbox", "Показывать соседние кадры" if self._ui_language == "ru" else "Show neighboring frames")
        )
        if self.neighbor_columns_label_widget is not None:
            self.neighbor_columns_label_widget.setText(
                self._tr("neighbor_columns_label", "Кадров в строке" if self._ui_language == "ru" else "Frames per row")
            )
        if self.neighbor_max_grid_label_widget is not None:
            self.neighbor_max_grid_label_widget.setText(
                self._tr("neighbor_max_grid_label", "Макс. сетка" if self._ui_language == "ru" else "Max grid")
            )
        if self.neighbor_opacity_label_widget is not None:
            self.neighbor_opacity_label_widget.setText(
                self._tr("neighbor_opacity_label", "Прозрачность соседей" if self._ui_language == "ru" else "Neighbor opacity")
            )
        if self.neighbor_overlap_label_widget is not None:
            self.neighbor_overlap_label_widget.setText(
                self._tr("neighbor_overlap_label", "Пересечение кадров" if self._ui_language == "ru" else "Frame overlap")
            )
        if self.extra_layers_label_widget is not None:
            self.extra_layers_label_widget.setText(
                self._tr("extra_layers_label", "Дополнительные слои" if self._ui_language == "ru" else "Additional layers")
            )
        if self.extra_layer_path_label_widget is not None:
            self.extra_layer_path_label_widget.setText(
                self._tr("extra_layer_path_label", "Путь слоя" if self._ui_language == "ru" else "Layer path")
            )
        self.add_extra_layers_button.setText(
            self._tr("add_extra_layers_button", "Добавить изображения" if self._ui_language == "ru" else "Add images")
        )
        self.remove_extra_layer_button.setText(
            self._tr("remove_extra_layer_button", "Удалить слой" if self._ui_language == "ru" else "Remove layer")
        )
        self.extra_layer_visible_checkbox.setText(
            self._tr("extra_layer_visible_checkbox", "Показывать выбранный слой" if self._ui_language == "ru" else "Show selected layer")
        )
        if self.extra_layer_opacity_label_widget is not None:
            self.extra_layer_opacity_label_widget.setText(
                self._tr("extra_layer_opacity_label", "Прозрачность слоя" if self._ui_language == "ru" else "Layer opacity")
            )
        if self.extra_layer_dx_label_widget is not None:
            self.extra_layer_dx_label_widget.setText(
                self._tr("extra_layer_dx_label", "Смещение X" if self._ui_language == "ru" else "Offset X")
            )
        if self.extra_layer_dy_label_widget is not None:
            self.extra_layer_dy_label_widget.setText(
                self._tr("extra_layer_dy_label", "Смещение Y" if self._ui_language == "ru" else "Offset Y")
            )
        for widget, tooltip_key in (
            (self.external_color_label_widget, "external_color"),
            (self.external_color_button, "external_color"),
            (self.hole_color_label_widget, "hole_color"),
            (self.hole_color_button, "hole_color"),
            (self.selected_color_label_widget, "selected_color"),
            (self.selected_color_button, "selected_color"),
            (self.vertex_color_label_widget, "vertex_color"),
            (self.vertex_color_button, "vertex_color"),
            (self.line_width_label_widget, "line_width"),
            (self.line_width_spin, "line_width"),
            (self.vertex_size_label_widget, "vertex_size"),
            (self.vertex_size_spin, "vertex_size"),
            (self.fill_opacity_label_widget, "fill_opacity"),
            (self.fill_opacity_spin, "fill_opacity"),
            (self.show_vertices_checkbox, "show_vertices"),
            (self.show_labels_checkbox, "show_labels"),
        ):
            self._set_common_tooltip(widget, tooltip_key)
        for widget, tooltip in (
            (
                self.random_object_colors_checkbox,
                "Раскрашивает каждый объект отдельным цветом. Это удобно, когда нужно видеть, какие контуры остались отдельными после правки."
                if self._ui_language == "ru"
                else "Colors each object separately. Useful for seeing which contours remain separate after edits.",
            ),
            (
                self.show_neighbor_frames_checkbox,
                "Показывает соседние изображения вокруг текущего кадра на фоне. Текущий кадр остается в центре и отмечается желтой рамкой."
                if self._ui_language == "ru"
                else "Shows neighboring images around the current frame in the background. The current frame stays centered and has a yellow border.",
            ),
            (
                self.neighbor_columns_spin,
                "Сколько кадров в одной строке исходной последовательности. Это нужно, чтобы правильно найти соседей сверху, снизу и по диагонали."
                if self._ui_language == "ru"
                else "How many frames are in one row of the source sequence. Used to locate top, bottom, and diagonal neighbors.",
            ),
            (
                self.neighbor_max_grid_spin,
                "Максимальный размер фоновой матрицы: 3, 5 или 7 кадров по стороне. При уменьшении масштаба сетка раскрывается до этого значения."
                if self._ui_language == "ru"
                else "Maximum background matrix size: 3, 5, or 7 frames per side. Zooming out expands the grid up to this value.",
            ),
            (
                self.neighbor_opacity_spin,
                "Прозрачность соседних кадров на фоне. Меньше значение делает их менее заметными относительно основного кадра."
                if self._ui_language == "ru"
                else "Opacity of neighboring background frames. Lower values make them less prominent than the main frame.",
            ),
            (
                self.neighbor_overlap_spin,
                "Сколько пикселей соседние кадры заходят друг на друга. Ноль размещает кадры вплотную без пересечения."
                if self._ui_language == "ru"
                else "How many pixels neighboring frames overlap. Zero places frames edge to edge without overlap.",
            ),
            (
                self.extra_layers_widget,
                "Список дополнительных JPG/PNG-слоев. Каждый слой можно включить, сделать прозрачнее и сдвинуть относительно основного изображения."
                if self._ui_language == "ru"
                else "List of additional JPG/PNG layers. Each layer can be shown, faded, and shifted relative to the main image.",
            ),
            (
                self.extra_layer_path_widget,
                "Путь к файлу выбранного слоя. Можно вставить или набрать путь вручную, затем нажать Enter или убрать фокус с поля."
                if self._ui_language == "ru"
                else "Path to the selected layer file. You can paste or type it manually, then press Enter or move focus away.",
            ),
            (
                self.extra_layer_path_browse_button,
                "Выбрать изображение для выбранного слоя."
                if self._ui_language == "ru"
                else "Choose an image for the selected layer.",
            ),
            (
                self.add_extra_layers_button,
                "Добавляет один или несколько дополнительных JPG/PNG-слоев поверх основного изображения."
                if self._ui_language == "ru"
                else "Adds one or more additional JPG/PNG layers over the main image.",
            ),
            (
                self.remove_extra_layer_button,
                "Удаляет выбранный дополнительный слой из просмотра."
                if self._ui_language == "ru"
                else "Removes the selected additional layer from the view.",
            ),
            (
                self.extra_layer_visible_checkbox,
                "Включает или выключает отображение выбранного дополнительного слоя."
                if self._ui_language == "ru"
                else "Shows or hides the selected additional layer.",
            ),
            (
                self.extra_layer_opacity_spin,
                "Прозрачность выбранного слоя: 0 делает его невидимым, 1 показывает полностью."
                if self._ui_language == "ru"
                else "Opacity of the selected layer: 0 hides it, 1 shows it fully.",
            ),
            (
                self.extra_layer_dx_spin,
                "Горизонтальное смещение выбранного слоя в пикселях относительно основного кадра."
                if self._ui_language == "ru"
                else "Horizontal offset of the selected layer in pixels relative to the main frame.",
            ),
            (
                self.extra_layer_dy_spin,
                "Вертикальное смещение выбранного слоя в пикселях относительно основного кадра."
                if self._ui_language == "ru"
                else "Vertical offset of the selected layer in pixels relative to the main frame.",
            ),
        ):
            widget.setToolTip(tooltip)
            widget.setStatusTip(tooltip)

        self.editor_group.setTitle(self._tr("editor_group_title"))
        self._update_tool_button_texts()
        self._update_action_button_texts()
        self.polygon_mode_label.setText("Полигон" if self._ui_language == "ru" else "Polygon")
        self.brush_mode_label.setText("Кисть" if self._ui_language == "ru" else "Brush")
        self.brush_size_label.setText("Толщина" if self._ui_language == "ru" else "Width")
        self.delete_vertex_mode_label.setText("Удаление" if self._ui_language == "ru" else "Delete")
        self.via_width_label.setText("Via W")
        self.via_height_label.setText("Via H")
        for widget, tooltip_key in (
            (self.polygon_mode_label, "polygon_mode"),
            (self.polygon_mode_combo, "polygon_mode"),
            (self.brush_mode_label, "brush_mode"),
            (self.brush_mode_combo, "brush_mode"),
            (self.brush_size_label, "brush_size"),
            (self.brush_size_spin, "brush_size"),
            (self.delete_vertex_mode_label, "delete_vertex_mode"),
            (self.delete_vertex_mode_combo, "delete_vertex_mode"),
            (self.via_width_label, "editor_via_width"),
            (self.via_width_spin, "editor_via_width"),
            (self.via_height_label, "editor_via_height"),
            (self.via_height_spin, "editor_via_height"),
        ):
            self._set_common_tooltip(widget, tooltip_key)
        self._on_editor_tool_changed(self.polygon_editor.current_tool)
        self._retranslate_editor_mode_combos()
        self.preview_busy_label.setText(self._busy_indicator_text())
        self._set_progress_status(self._progress_status_key, **self._progress_status_kwargs)

        self._populate_pipeline_operations()
        self._populate_pipeline_list()
        if selected_pipeline_row >= 0 and selected_pipeline_row < self.pipeline_list.count():
            self.pipeline_list.setCurrentRow(selected_pipeline_row)
        self._retranslate_contour_mode_combos()
        if selected_operation:
            target_item = self._find_operation_tree_item(selected_operation)
            if target_item is not None:
                self.operation_tree.setCurrentItem(target_item)
        self._update_pipeline_help_preview(self._selected_available_operation_name())
        self._refresh_help_menu()

    def _update_tool_button_texts(self) -> None:
        texts = {
            EditorTool.RULER: self._tr("tool_ruler", "Ruler"),
            EditorTool.ADD_VIA: self._tr("tool_add_via", "Via"),
            EditorTool.SELECT: self._tr("tool_select", "Выбор" if self._ui_language == "ru" else "Select"),
            EditorTool.SELECT_AREA: self._tr("tool_select_area", "Выбор рамкой" if self._ui_language == "ru" else "Area select"),
            EditorTool.PAN: self._tr("tool_pan", "Панорамирование" if self._ui_language == "ru" else "Pan"),
            EditorTool.ADD_POLYGON: self._tr("tool_add_polygon", "Полигон" if self._ui_language == "ru" else "Add polygon"),
            EditorTool.BRUSH: self._tr("tool_brush", "Кисть" if self._ui_language == "ru" else "Brush"),
            EditorTool.ADD_VERTEX: self._tr("tool_add_vertex", "Добавить вершину" if self._ui_language == "ru" else "Add vertex"),
            EditorTool.DELETE_VERTEX: self._tr("tool_delete_vertex", "Удалить вершину" if self._ui_language == "ru" else "Delete vertex"),
            EditorTool.MOVE_VERTEX: self._tr("tool_move_vertex", "Переместить вершину" if self._ui_language == "ru" else "Move vertex"),
            EditorTool.DELETE_POLYGON: self._tr("tool_delete_polygon", "Удалить полигон" if self._ui_language == "ru" else "Delete polygon"),
        }
        for tool, button in self._tool_buttons.items():
            label = texts.get(tool, tool.value)
            if tool == EditorTool.RULER:
                label = self._tr("tool_ruler", "Линейка" if self._ui_language == "ru" else "Ruler")
            tooltip_pair = EDITOR_TOOL_TOOLTIPS.get(tool)
            tooltip = (tooltip_pair[0] if self._ui_language == "ru" else tooltip_pair[1]) if tooltip_pair else label
            button.setToolTip(tooltip)
            button.setStatusTip(tooltip)
            button.setAccessibleName(label)

    def _update_action_button_texts(self) -> None:
        for button, label in [
            (self.undo_button, self._tr("undo_button", "Отменить" if self._ui_language == "ru" else "Undo")),
            (self.redo_button, self._tr("redo_button", "Повторить" if self._ui_language == "ru" else "Redo")),
            (self.zoom_in_button, self._tr("zoom_in_button", "Увеличить" if self._ui_language == "ru" else "Zoom in")),
            (self.zoom_out_button, self._tr("zoom_out_button", "Уменьшить" if self._ui_language == "ru" else "Zoom out")),
            (self.fit_button, self._tr("fit_button", "Подогнать" if self._ui_language == "ru" else "Fit")),
        ]:
            button.setToolTip(label)
            button.setStatusTip(label)
            button.setAccessibleName(label)
        for button, tooltip_key in (
            (self.undo_button, "undo_button"),
            (self.redo_button, "redo_button"),
            (self.zoom_in_button, "zoom_in_button"),
            (self.zoom_out_button, "zoom_out_button"),
            (self.fit_button, "fit_button"),
        ):
            tooltip = _localized_text(EDITOR_ACTION_TOOLTIPS, tooltip_key, self._ui_language)
            button.setToolTip(tooltip)
            button.setStatusTip(tooltip)

    def _on_editor_tool_changed(self, tool) -> None:
        is_ruler = tool == EditorTool.RULER
        self.ruler_status_label.setVisible(is_ruler)
        if is_ruler and not self.ruler_status_label.text():
            self.ruler_status_label.setText(
                self._tr(
                    "ruler_idle_label",
                    "Потяните на изображении для измерения" if self._ui_language == "ru" else "Drag on the image to measure",
                )
            )
        elif not is_ruler:
            self.ruler_status_label.clear()

    def _update_ruler_status(self, text: str) -> None:
        if not text:
            if self.polygon_editor.current_tool == EditorTool.RULER:
                self.ruler_status_label.setText(
                    self._tr(
                        "ruler_idle_label",
                        "Потяните на изображении для измерения" if self._ui_language == "ru" else "Drag on the image to measure",
                    )
                )
            else:
                self.ruler_status_label.clear()
            return
        self.ruler_status_label.setText(text)

    def _retranslate_editor_mode_combos(self) -> None:
        polygon_mode = self.polygon_mode_combo.currentData()
        brush_mode = self.brush_mode_combo.currentData()
        delete_mode = self.delete_vertex_mode_combo.currentData()

        self.polygon_mode_combo.setItemText(0, self._mode_text("polygon_points"))
        self.polygon_mode_combo.setItemText(1, self._mode_text("polygon_rectangle"))
        self.brush_mode_combo.setItemText(0, self._mode_text("brush_freeform"))
        self.brush_mode_combo.setItemText(1, self._mode_text("brush_45deg"))
        self.delete_vertex_mode_combo.setItemText(0, self._mode_text("delete_single"))
        self.delete_vertex_mode_combo.setItemText(1, self._mode_text("delete_area"))

        polygon_index = self.polygon_mode_combo.findData(polygon_mode)
        brush_index = self.brush_mode_combo.findData(brush_mode)
        delete_index = self.delete_vertex_mode_combo.findData(delete_mode)
        if polygon_index >= 0:
            self.polygon_mode_combo.setCurrentIndex(polygon_index)
        if brush_index >= 0:
            self.brush_mode_combo.setCurrentIndex(brush_index)
        if delete_index >= 0:
            self.delete_vertex_mode_combo.setCurrentIndex(delete_index)

    def _retranslate_contour_mode_combos(self) -> None:
        current_retrieval = self.retrieval_mode_combo.currentData()
        for index in range(self.retrieval_mode_combo.count()):
            mode_name = str(self.retrieval_mode_combo.itemData(index))
            self.retrieval_mode_combo.setItemText(index, self._tr(f"retrieval_mode.{mode_name}", default=mode_name))
        if current_retrieval is not None:
            self.retrieval_mode_combo.setCurrentIndex(self.retrieval_mode_combo.findData(current_retrieval))

        current_approximation = self.approximation_mode_combo.currentData()
        for index in range(self.approximation_mode_combo.count()):
            mode_name = str(self.approximation_mode_combo.itemData(index))
            self.approximation_mode_combo.setItemText(
                index,
                self._tr(f"approximation_mode.{mode_name}", default=mode_name),
            )
        if current_approximation is not None:
            self.approximation_mode_combo.setCurrentIndex(self.approximation_mode_combo.findData(current_approximation))

    def _wrap_group(self, title: str, widget: QWidget) -> QWidget:
        group = QGroupBox(title)
        layout = QVBoxLayout(group)
        layout.addWidget(widget)
        return group

    def _build_checkbox_spin_row(self, checkbox: QCheckBox, spinbox: QAbstractSpinBox) -> QWidget:
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        layout.addWidget(checkbox, 1)
        layout.addWidget(spinbox)
        return widget

    def _build_checkbox_range_row(
        self,
        checkbox: QCheckBox,
        min_spinbox: QAbstractSpinBox,
        max_spinbox: QAbstractSpinBox,
    ) -> QWidget:
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        layout.addWidget(checkbox, 1)
        layout.addWidget(min_spinbox)
        layout.addWidget(max_spinbox)
        return widget

    def _build_range_row(self, min_spinbox: QAbstractSpinBox, max_spinbox: QAbstractSpinBox) -> QWidget:
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        layout.addWidget(min_spinbox)
        layout.addWidget(max_spinbox)
        return widget

    def _configure_icon_only_button(self, button: QPushButton, icon: QIcon) -> None:
        button.setText("")
        button.setIcon(icon)
        button.setIconSize(QSize(20, 20))
        button.setFixedWidth(36)
        button.setMinimumHeight(30)

    def _refresh_files_icon(self) -> QIcon:
        pixmap = QPixmap(24, 24)
        pixmap.fill(Qt.GlobalColor.transparent)
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        pen = QPen(QColor("#22C55E"), 2.4)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(pen)
        painter.drawArc(5, 5, 14, 14, 35 * 16, 285 * 16)
        arrow = QPolygonF([QPointF(18.0, 5.0), QPointF(18.2, 11.0), QPointF(13.2, 8.0)])
        painter.setBrush(QColor("#22C55E"))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawPolygon(arrow)
        painter.end()
        return QIcon(pixmap)

    def _configure_compact_form(self, form: QFormLayout) -> None:
        form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
        form.setLabelAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)

    def _disable_spinbox_wheel_changes(self) -> None:
        for spinbox in self.findChildren(QAbstractSpinBox):
            spinbox.installEventFilter(self)

    def _register_spinbox(self, spinbox: QAbstractSpinBox) -> None:
        spinbox.installEventFilter(self)

    def eventFilter(self, watched, event) -> bool:
        if isinstance(watched, QAbstractSpinBox) and event.type() == QEvent.Type.Wheel:
            event.ignore()
            return True
        return super().eventFilter(watched, event)

    def _build_color_button(self, color: str, handler) -> QPushButton:
        button = QPushButton(color)
        button.clicked.connect(handler)
        self._update_color_button(button, color)
        return button

    def _update_color_button(self, button: QPushButton, color_value: str) -> None:
        button.setText(color_value)
        button.setStyleSheet(f"background-color: {color_value}; color: #111111;")

    def _populate_pipeline_operations(self) -> None:
        selected_operation = self._selected_available_operation_name()
        self.operation_tree.clear()
        for _group_key, labels, operations in PIPELINE_OPERATION_GROUPS:
            group_item = QTreeWidgetItem([labels[0] if self._ui_language == "ru" else labels[1]])
            group_item.setFlags(Qt.ItemFlag.ItemIsEnabled)
            for operation_name in operations:
                child_item = QTreeWidgetItem([get_operation_display_name(operation_name, self._ui_language)])
                child_item.setData(0, Qt.ItemDataRole.UserRole, operation_name)
                summary, use_case = self._operation_help_entry(operation_name)
                child_item.setToolTip(
                    0,
                    f"{summary}\n\n"
                    + (("Когда использовать: " if self._ui_language == "ru" else "When to use: ") + use_case),
                )
                group_item.addChild(child_item)
            group_item.setExpanded(True)
            self.operation_tree.addTopLevelItem(group_item)
        target_operation = selected_operation or self._all_operation_names()[0]
        target_item = self._find_operation_tree_item(target_operation)
        if target_item is not None:
            self.operation_tree.setCurrentItem(target_item)
            self._update_pipeline_help_preview(target_operation)

    def _populate_pipeline_list(self) -> None:
        self._ignore_pipeline_item_change = True
        self.pipeline_list.clear()
        for step in self._pipeline.steps:
            label = get_operation_display_name(step.operation, self._ui_language)
            item = QListWidgetItem(label)
            item.setFlags(
                item.flags()
                | Qt.ItemFlag.ItemIsUserCheckable
                | Qt.ItemFlag.ItemIsSelectable
                | Qt.ItemFlag.ItemIsEnabled
                | Qt.ItemFlag.ItemIsDragEnabled
            )
            item.setData(Qt.ItemDataRole.UserRole, self.pipeline_list.count())
            item.setData(Qt.ItemDataRole.UserRole + 1, step.operation)
            item.setCheckState(Qt.CheckState.Checked if step.enabled else Qt.CheckState.Unchecked)
            self.pipeline_list.addItem(item)
        self._ignore_pipeline_item_change = False
        if self.pipeline_list.count():
            self.pipeline_list.setCurrentRow(0)
            self._render_pipeline_parameters(0)
        else:
            self._clear_parameters_form()

    def _clear_parameters_form(self) -> None:
        while self.parameters_form.count():
            item = self.parameters_form.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self._parameter_widgets.clear()

    def _on_pipeline_step_selected(self, row: int) -> None:
        self._render_pipeline_parameters(row)

    def _render_pipeline_parameters(self, row: int) -> None:
        self._clear_parameters_form()
        if row < 0 or row >= len(self._pipeline.steps):
            self._set_color_pick_active(None)
            return
        step = self._pipeline.steps[row]
        descriptor = get_operation_descriptor(step.operation)
        for spec in descriptor.parameters:
            value = step.parameters.get(spec.name, spec.default)
            if spec.kind == "bool":
                widget = QCheckBox()
                widget.setChecked(bool(value))
                widget.stateChanged.connect(
                    lambda _state, name=spec.name, row_index=row, w=widget: self._update_step_parameter(row_index, name, w.isChecked())
                )
            elif spec.kind == "choice":
                widget = QComboBox()
                for option in spec.options:
                    widget.addItem(get_choice_display_label(spec.name, str(option), self._ui_language), option)
                selected_index = widget.findData(value)
                if selected_index >= 0:
                    widget.setCurrentIndex(selected_index)
                widget.currentIndexChanged.connect(
                    lambda _index, name=spec.name, row_index=row, w=widget: self._update_step_parameter(
                        row_index,
                        name,
                        w.currentData(),
                    )
                )
            elif spec.kind == "int":
                widget = QSpinBox()
                self._register_spinbox(widget)
                widget.setRange(int(spec.minimum or -1_000_000), int(spec.maximum or 1_000_000))
                widget.setSingleStep(int(spec.step or 1))
                widget.setValue(int(value))
                widget.valueChanged.connect(
                    lambda new_value, name=spec.name, row_index=row: self._update_step_parameter(row_index, name, int(new_value))
                )
            else:
                widget = QDoubleSpinBox()
                self._register_spinbox(widget)
                widget.setDecimals(spec.decimals)
                widget.setRange(float(spec.minimum or -1_000_000), float(spec.maximum or 1_000_000))
                widget.setSingleStep(float(spec.step or 0.1))
                widget.setValue(float(value))
                widget.valueChanged.connect(
                    lambda new_value, name=spec.name, row_index=row: self._update_step_parameter(row_index, name, float(new_value))
                )
            tooltip = spec.tooltip or self._pipeline_parameter_tooltip(step.operation, spec.name)
            widget.setToolTip(tooltip)
            self._parameter_widgets[spec.name] = widget
            label_widget = QLabel(get_parameter_display_label(spec, self._ui_language))
            label_widget.setToolTip(tooltip)
            self.parameters_form.addRow(label_widget, widget)
        if step.operation == "color_binarize":
            self._render_color_binarize_parameters(row)
        else:
            self._set_color_pick_active(None)

    def _update_step_parameter(self, row: int, parameter_name: str, value) -> None:
        if row < 0 or row >= len(self._pipeline.steps):
            return
        self._pipeline.steps[row].parameters[parameter_name] = value
        self._auto_apply_pipeline()

    def _color_selection_entries(self, row: int) -> list[dict[str, object]]:
        if row < 0 or row >= len(self._pipeline.steps):
            return []
        entries = self._pipeline.steps[row].parameters.get("selected_colors", [])
        if not isinstance(entries, list):
            entries = []
        normalized: list[dict[str, object]] = []
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            rgb = entry.get("rgb")
            if not isinstance(rgb, (list, tuple)) or len(rgb) != 3:
                continue
            try:
                parsed_rgb = [max(0, min(255, int(channel))) for channel in rgb]
            except (TypeError, ValueError):
                continue
            normalized.append({"rgb": parsed_rgb, "enabled": bool(entry.get("enabled", True))})
        self._pipeline.steps[row].parameters["selected_colors"] = normalized
        return normalized

    def _render_color_binarize_parameters(self, row: int) -> None:
        entries = self._color_selection_entries(row)
        group = QGroupBox(
            self._tr("color_binarize_group_title", "Цвета для бинаризации" if self._ui_language == "ru" else "Colors for binarization")
        )
        layout = QVBoxLayout(group)
        hint = QLabel(
            self._tr(
                "color_binarize_hint",
                "Включите выбор и кликните по изображению, чтобы добавить цвет. Галочкой можно временно отключить цвет."
                if self._ui_language == "ru"
                else "Enable picking and click the image to add a color. Uncheck an item to disable it temporarily.",
            )
        )
        hint.setWordWrap(True)
        hint.setToolTip(
            "Цвета из списка используются для построения бинарной маски; допуск задается параметром delta."
            if self._ui_language == "ru"
            else "Colors in the list are used to build the binary mask; tolerance is controlled by delta."
        )
        layout.addWidget(hint)
        color_list = QListWidget()
        color_list.setToolTip(
            "Отмеченные цвета участвуют в бинаризации. Снимите галочку, чтобы временно исключить цвет из маски."
            if self._ui_language == "ru"
            else "Checked colors participate in binarization. Uncheck a color to temporarily exclude it from the mask."
        )
        color_list.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        for entry in entries:
            rgb = entry["rgb"]
            item = QListWidgetItem(f"#{int(rgb[0]):02X}{int(rgb[1]):02X}{int(rgb[2]):02X}")
            item.setToolTip(
                "Этот цвет добавляет похожие пиксели в маску; галочка включает или выключает его."
                if self._ui_language == "ru"
                else "This color adds similar pixels to the mask; the checkbox enables or disables it."
            )
            item.setData(Qt.ItemDataRole.UserRole, list(rgb))
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsEnabled)
            item.setCheckState(Qt.CheckState.Checked if entry.get("enabled", True) else Qt.CheckState.Unchecked)
            item.setBackground(QColor(int(rgb[0]), int(rgb[1]), int(rgb[2])))
            brightness = int(rgb[0]) * 0.299 + int(rgb[1]) * 0.587 + int(rgb[2]) * 0.114
            item.setForeground(QColor("#111111" if brightness > 150 else "#F8FAFC"))
            color_list.addItem(item)
        color_list.itemChanged.connect(lambda item, row_index=row, widget=color_list: self._on_color_entry_changed(row_index, widget, item))
        layout.addWidget(color_list)

        buttons_row = QWidget()
        buttons_layout = QHBoxLayout(buttons_row)
        buttons_layout.setContentsMargins(0, 0, 0, 0)
        pick_button = QPushButton(
            self._tr("pick_colors_button", "Выбор с изображения" if self._ui_language == "ru" else "Pick from image")
        )
        pick_button.setCheckable(True)
        pick_button.setToolTip(
            "Включает выбор цвета с изображения: кликните по нужному пикселю, чтобы добавить его в список."
            if self._ui_language == "ru"
            else "Enables picking from the image: click a pixel to add its color to the list."
        )
        pick_button.setChecked(self._color_pick_pipeline_row == row)
        pick_button.toggled.connect(lambda checked, row_index=row: self._set_color_pick_active(row_index if checked else None))
        remove_button = QPushButton(
            self._tr("remove_selected_color_button", "Удалить выбранный" if self._ui_language == "ru" else "Remove selected")
        )
        remove_button.setToolTip(
            "Удаляет выбранный цвет из списка бинаризации."
            if self._ui_language == "ru"
            else "Removes the selected color from the binarization list."
        )
        remove_button.clicked.connect(lambda _checked=False, row_index=row, widget=color_list: self._remove_selected_color_entry(row_index, widget))
        clear_button = QPushButton(
            self._tr("clear_colors_button", "Очистить список" if self._ui_language == "ru" else "Clear list")
        )
        clear_button.setToolTip(
            "Очищает весь список цветов для этого шага бинаризации."
            if self._ui_language == "ru"
            else "Clears the whole color list for this binarization step."
        )
        clear_button.clicked.connect(lambda _checked=False, row_index=row: self._clear_color_entries(row_index))
        buttons_layout.addWidget(pick_button)
        buttons_layout.addWidget(remove_button)
        buttons_layout.addWidget(clear_button)
        layout.addWidget(buttons_row)
        self.parameters_form.addRow(group)

    def _on_color_entry_changed(self, row: int, color_list: QListWidget, item: QListWidgetItem) -> None:
        entries = self._color_selection_entries(row)
        index = color_list.row(item)
        if index < 0 or index >= len(entries):
            return
        entries[index]["enabled"] = item.checkState() == Qt.CheckState.Checked
        self._pipeline.steps[row].parameters["selected_colors"] = entries
        self._auto_apply_pipeline()

    def _remove_selected_color_entry(self, row: int, color_list: QListWidget) -> None:
        index = color_list.currentRow()
        if index < 0:
            return
        entries = self._color_selection_entries(row)
        if index >= len(entries):
            return
        entries.pop(index)
        self._pipeline.steps[row].parameters["selected_colors"] = entries
        self._render_pipeline_parameters(row)
        self._auto_apply_pipeline()

    def _clear_color_entries(self, row: int) -> None:
        if row < 0 or row >= len(self._pipeline.steps):
            return
        self._pipeline.steps[row].parameters["selected_colors"] = []
        self._render_pipeline_parameters(row)
        self._auto_apply_pipeline()

    def _set_color_pick_active(self, row: int | None) -> None:
        self._color_pick_pipeline_row = row
        if hasattr(self, "polygon_editor"):
            self.polygon_editor.set_image_click_mode(row is not None)

    def _set_via_template_pick_active(self, enabled: bool) -> None:
        if enabled:
            self._set_color_pick_active(None)
        if hasattr(self, "polygon_editor"):
            self.polygon_editor.set_image_region_selection_mode(enabled)

    def _refresh_via_template_list(self) -> None:
        if not hasattr(self, "via_template_list"):
            return
        self.via_template_list.clear()
        for index, template in enumerate(self._via_template_images, start=1):
            height, width = template.shape[:2]
            item = QListWidgetItem(f"{index}: {width}x{height}")
            preview_pixmap = QPixmap.fromImage(cv_to_qimage(template)).scaled(
                56,
                56,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            item.setIcon(QIcon(preview_pixmap))
            item.setToolTip(
                (
                    f"Шаблон via #{index}: {width}x{height} пикс."
                    if self._ui_language == "ru"
                    else f"Via template #{index}: {width}x{height} px"
                )
            )
            self.via_template_list.addItem(item)

    def _normalize_via_template_images(self, payload: list[object]) -> list[np.ndarray]:
        templates: list[np.ndarray] = []
        for item in payload:
            try:
                image = np.asarray(item, dtype=np.uint8)
            except (TypeError, ValueError):
                continue
            if image.ndim == 3:
                if image.shape[2] >= 3:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                else:
                    image = image[:, :, 0]
            if image.ndim != 2 or image.shape[0] < 2 or image.shape[1] < 2:
                continue
            templates.append(image.copy())
        return templates

    def _on_editor_image_region_selected(self, x_coord: float, y_coord: float, width: float, height: float) -> None:
        if hasattr(self, "add_via_template_button"):
            self.add_via_template_button.setChecked(False)
        if hasattr(self, "polygon_editor"):
            self.polygon_editor.set_image_region_selection_mode(False)
        image = self._workspace.current_display_image()
        if image is None:
            return
        data = np.asarray(image)
        if data.size == 0:
            return
        left = max(0, int(np.floor(x_coord)))
        top = max(0, int(np.floor(y_coord)))
        right = min(data.shape[1], int(np.ceil(x_coord + width)))
        bottom = min(data.shape[0], int(np.ceil(y_coord + height)))
        if right - left < 2 or bottom - top < 2:
            return
        template = data[top:bottom, left:right].copy()
        if template.ndim == 3:
            if template.shape[2] >= 3:
                template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
            else:
                template = template[:, :, 0]
        self._via_template_images.append(template.astype(np.uint8, copy=False))
        self._refresh_via_template_list()
        self._on_extraction_settings_changed()
        self._append_log(
            self._tr(
                "via_template_added_log",
                "Добавлен шаблон via {width}x{height}. Всего шаблонов: {count}."
                if self._ui_language == "ru"
                else "Added via template {width}x{height}. Total templates: {count}.",
                width=right - left,
                height=bottom - top,
                count=len(self._via_template_images),
            )
        )

    def _clear_via_templates(self, *_args) -> None:
        self._via_template_images.clear()
        self._refresh_via_template_list()
        self._on_extraction_settings_changed()

    def _remove_selected_via_template(self, *_args) -> None:
        row = self.via_template_list.currentRow() if hasattr(self, "via_template_list") else -1
        if row < 0 or row >= len(self._via_template_images):
            return
        self._via_template_images.pop(row)
        self._refresh_via_template_list()
        if self._via_template_images:
            self.via_template_list.setCurrentRow(min(row, len(self._via_template_images) - 1))
        self._on_extraction_settings_changed()

    def _built_in_via_presets(self) -> dict[str, dict[str, object]]:
        return {
            "Яркие via на дорожках" if self._ui_language == "ru" else "Bright vias on traces": self._noisy_traces_via_preset_payload(),
            "Слабые/размытые via" if self._ui_language == "ru" else "Weak/blurred vias": self._blurred_via_preset_payload(),
        }

    def _noisy_traces_via_preset_payload(self) -> dict[str, object]:
        return {
            "via_white_range_enabled": True,
            "via_white_range_min": 180,
            "via_white_range_max": 255,
            "via_black_range_enabled": False,
            "via_black_range_min": 0,
            "via_black_range_max": 30,
            "via_detector_gradient_enabled": True,
            "via_detector_spot_enabled": True,
            "via_detector_hough_enabled": False,
            "via_detector_components_enabled": False,
            "via_detector_contours_enabled": False,
            "via_detector_morphology_enabled": False,
            "via_detector_template_enabled": False,
            "via_detector_blob_enabled": False,
            "via_spot_min_contrast": 24.0,
            "via_spot_min_roundness": 55.0,
            "via_spot_line_suppression": 0.85,
            "via_gradient_min_strength": 14.0,
            "via_gradient_min_coverage": 0.28,
            "via_hough_edge_threshold": 100.0,
            "via_hough_accumulator_threshold": 20.0,
            "via_component_min_score": 0.0,
            "via_contour_min_score": 0.0,
            "via_morphology_peak_scale": 0.20,
            "via_template_min_score": 0.35,
            "via_blob_min_circularity": 0.60,
            "via_min_roundness": 40.0,
            "debug_enabled": True,
        }

    def _blurred_via_preset_payload(self) -> dict[str, object]:
        return {
            "via_white_range_enabled": True,
            "via_white_range_min": 135,
            "via_white_range_max": 255,
            "via_black_range_enabled": False,
            "via_black_range_min": 0,
            "via_black_range_max": 45,
            "via_detector_gradient_enabled": True,
            "via_detector_spot_enabled": True,
            "via_detector_hough_enabled": False,
            "via_detector_components_enabled": True,
            "via_detector_contours_enabled": False,
            "via_detector_morphology_enabled": True,
            "via_detector_template_enabled": False,
            "via_detector_blob_enabled": False,
            "via_spot_min_contrast": 12.0,
            "via_spot_min_roundness": 28.0,
            "via_spot_line_suppression": 0.55,
            "via_gradient_min_strength": 7.0,
            "via_gradient_min_coverage": 0.14,
            "via_hough_edge_threshold": 70.0,
            "via_hough_accumulator_threshold": 10.0,
            "via_component_min_score": 0.0,
            "via_contour_min_score": 0.0,
            "via_morphology_peak_scale": 0.12,
            "via_template_min_score": 0.30,
            "via_blob_min_circularity": 0.25,
            "via_min_roundness": 15.0,
            "debug_enabled": True,
        }

    def _load_user_via_presets(self) -> dict[str, dict[str, object]]:
        settings = QSettings("ViaLaNet", "PolygonWidget")
        raw_payload = settings.value(VIA_PRESETS_SETTINGS_KEY, "{}", type=str)
        try:
            payload = json.loads(str(raw_payload or "{}"))
        except json.JSONDecodeError:
            return {}
        if not isinstance(payload, dict):
            return {}
        return {
            str(name): dict(value)
            for name, value in payload.items()
            if isinstance(value, dict)
        }

    def _save_user_via_presets(self) -> None:
        settings = QSettings("ViaLaNet", "PolygonWidget")
        settings.setValue(VIA_PRESETS_SETTINGS_KEY, json.dumps(self._user_via_presets, ensure_ascii=False, sort_keys=True))
        settings.sync()

    def _refresh_via_preset_combo(self) -> None:
        if not hasattr(self, "via_preset_combo"):
            return
        current_name = self.via_preset_combo.currentText()
        self.via_preset_combo.clear()
        for name in self._built_in_via_presets():
            self.via_preset_combo.addItem(name, ("builtin", name))
        for name in sorted(self._user_via_presets):
            self.via_preset_combo.addItem(name, ("user", name))
        index = self.via_preset_combo.findText(current_name)
        if index >= 0:
            self.via_preset_combo.setCurrentIndex(index)

    def _current_via_preset_payload(self) -> dict[str, object]:
        payload = self._current_contour_settings().to_dict()
        excluded_keys = {
            "via_template_images",
            "fixed_via_widths",
            "fixed_via_heights",
            "min_via_width",
            "max_via_width",
            "min_via_height",
            "max_via_height",
            "via_size_mode",
        }
        return {key: value for key, value in payload.items() if key.startswith("via_") and key not in excluded_keys} | {
            "debug_enabled": self.debug_candidates_checkbox.isChecked()
        }

    def _apply_via_preset_payload(self, payload: dict[str, object]) -> None:
        blockers = [
            QSignalBlocker(self.via_white_range_checkbox),
            QSignalBlocker(self.via_white_range_min_spin),
            QSignalBlocker(self.via_white_range_max_spin),
            QSignalBlocker(self.via_black_range_checkbox),
            QSignalBlocker(self.via_black_range_min_spin),
            QSignalBlocker(self.via_black_range_max_spin),
            QSignalBlocker(self.via_detector_gradient_checkbox),
            QSignalBlocker(self.via_detector_hough_checkbox),
            QSignalBlocker(self.via_detector_components_checkbox),
            QSignalBlocker(self.via_detector_contours_checkbox),
            QSignalBlocker(self.via_detector_morphology_checkbox),
            QSignalBlocker(self.via_detector_template_checkbox),
            QSignalBlocker(self.via_detector_blob_checkbox),
            QSignalBlocker(self.via_detector_spot_checkbox),
            QSignalBlocker(self.via_spot_min_contrast_spin),
            QSignalBlocker(self.via_spot_min_roundness_spin),
            QSignalBlocker(self.via_spot_line_suppression_spin),
            QSignalBlocker(self.via_gradient_min_strength_spin),
            QSignalBlocker(self.via_gradient_min_coverage_spin),
            QSignalBlocker(self.via_hough_edge_threshold_spin),
            QSignalBlocker(self.via_hough_accumulator_threshold_spin),
            QSignalBlocker(self.via_component_min_score_spin),
            QSignalBlocker(self.via_contour_min_score_spin),
            QSignalBlocker(self.via_morphology_peak_scale_spin),
            QSignalBlocker(self.via_template_min_score_spin),
            QSignalBlocker(self.via_blob_min_circularity_spin),
            QSignalBlocker(self.debug_candidates_checkbox),
            QSignalBlocker(self.via_roundness_spin),
        ]
        try:
            self.via_white_range_checkbox.setChecked(bool(payload.get("via_white_range_enabled", self.via_white_range_checkbox.isChecked())))
            self.via_white_range_min_spin.setValue(int(payload.get("via_white_range_min", self.via_white_range_min_spin.value())))
            self.via_white_range_max_spin.setValue(int(payload.get("via_white_range_max", self.via_white_range_max_spin.value())))
            self.via_black_range_checkbox.setChecked(bool(payload.get("via_black_range_enabled", self.via_black_range_checkbox.isChecked())))
            self.via_black_range_min_spin.setValue(int(payload.get("via_black_range_min", self.via_black_range_min_spin.value())))
            self.via_black_range_max_spin.setValue(int(payload.get("via_black_range_max", self.via_black_range_max_spin.value())))
            self.via_detector_gradient_checkbox.setChecked(bool(payload.get("via_detector_gradient_enabled", self.via_detector_gradient_checkbox.isChecked())))
            self.via_detector_spot_checkbox.setChecked(bool(payload.get("via_detector_spot_enabled", self.via_detector_spot_checkbox.isChecked())))
            self.via_detector_hough_checkbox.setChecked(bool(payload.get("via_detector_hough_enabled", self.via_detector_hough_checkbox.isChecked())))
            self.via_detector_components_checkbox.setChecked(bool(payload.get("via_detector_components_enabled", self.via_detector_components_checkbox.isChecked())))
            self.via_detector_contours_checkbox.setChecked(bool(payload.get("via_detector_contours_enabled", self.via_detector_contours_checkbox.isChecked())))
            self.via_detector_morphology_checkbox.setChecked(bool(payload.get("via_detector_morphology_enabled", self.via_detector_morphology_checkbox.isChecked())))
            self.via_detector_template_checkbox.setChecked(bool(payload.get("via_detector_template_enabled", self.via_detector_template_checkbox.isChecked())))
            self.via_detector_blob_checkbox.setChecked(bool(payload.get("via_detector_blob_enabled", self.via_detector_blob_checkbox.isChecked())))
            self.via_spot_min_contrast_spin.setValue(float(payload.get("via_spot_min_contrast", self.via_spot_min_contrast_spin.value())))
            self.via_spot_min_roundness_spin.setValue(float(payload.get("via_spot_min_roundness", self.via_spot_min_roundness_spin.value())))
            self.via_spot_line_suppression_spin.setValue(float(payload.get("via_spot_line_suppression", self.via_spot_line_suppression_spin.value())))
            self.via_gradient_min_strength_spin.setValue(float(payload.get("via_gradient_min_strength", self.via_gradient_min_strength_spin.value())))
            self.via_gradient_min_coverage_spin.setValue(float(payload.get("via_gradient_min_coverage", self.via_gradient_min_coverage_spin.value())))
            self.via_hough_edge_threshold_spin.setValue(float(payload.get("via_hough_edge_threshold", self.via_hough_edge_threshold_spin.value())))
            self.via_hough_accumulator_threshold_spin.setValue(float(payload.get("via_hough_accumulator_threshold", self.via_hough_accumulator_threshold_spin.value())))
            self.via_component_min_score_spin.setValue(float(payload.get("via_component_min_score", self.via_component_min_score_spin.value())))
            self.via_contour_min_score_spin.setValue(float(payload.get("via_contour_min_score", self.via_contour_min_score_spin.value())))
            self.via_morphology_peak_scale_spin.setValue(float(payload.get("via_morphology_peak_scale", self.via_morphology_peak_scale_spin.value())))
            self.via_template_min_score_spin.setValue(float(payload.get("via_template_min_score", self.via_template_min_score_spin.value())))
            self.via_blob_min_circularity_spin.setValue(float(payload.get("via_blob_min_circularity", self.via_blob_min_circularity_spin.value())))
            self.via_roundness_spin.setValue(float(payload.get("via_min_roundness", self.via_roundness_spin.value())))
            self.debug_candidates_checkbox.setChecked(bool(payload.get("debug_enabled", self.debug_candidates_checkbox.isChecked())))
        finally:
            del blockers
        self._update_via_threshold_controls_state()
        self._on_extraction_settings_changed()

    def _apply_selected_via_preset(self) -> None:
        data = self.via_preset_combo.currentData()
        if not isinstance(data, tuple) or len(data) != 2:
            return
        preset_type, preset_name = data
        payload = self._built_in_via_presets().get(str(preset_name)) if preset_type == "builtin" else self._user_via_presets.get(str(preset_name))
        if payload:
            self._apply_via_preset_payload(payload)

    def _save_current_via_preset(self) -> None:
        name, ok = QInputDialog.getText(
            self,
            "Сохранить пресет" if self._ui_language == "ru" else "Save preset",
            "Имя пресета:" if self._ui_language == "ru" else "Preset name:",
        )
        name = str(name).strip()
        if not ok or not name:
            return
        self._user_via_presets[name] = self._current_via_preset_payload()
        self._save_user_via_presets()
        self._refresh_via_preset_combo()
        index = self.via_preset_combo.findText(name)
        if index >= 0:
            self.via_preset_combo.setCurrentIndex(index)

    def _delete_selected_via_preset(self) -> None:
        data = self.via_preset_combo.currentData()
        if not isinstance(data, tuple) or len(data) != 2 or data[0] != "user":
            return
        self._user_via_presets.pop(str(data[1]), None)
        self._save_user_via_presets()
        self._refresh_via_preset_combo()

    def _apply_noisy_traces_via_preset(self, *_args) -> None:
        self._apply_via_preset_payload(self._noisy_traces_via_preset_payload())

    def _apply_blurred_via_preset(self, *_args) -> None:
        self._apply_via_preset_payload(self._blurred_via_preset_payload())

    def _reset_via_search_parameters(self, *_args) -> None:
        blockers = [
            QSignalBlocker(self.via_detector_gradient_checkbox),
            QSignalBlocker(self.via_detector_hough_checkbox),
            QSignalBlocker(self.via_detector_components_checkbox),
            QSignalBlocker(self.via_detector_contours_checkbox),
            QSignalBlocker(self.via_detector_morphology_checkbox),
            QSignalBlocker(self.via_detector_template_checkbox),
            QSignalBlocker(self.via_detector_blob_checkbox),
            QSignalBlocker(self.via_detector_spot_checkbox),
            QSignalBlocker(self.via_spot_min_contrast_spin),
            QSignalBlocker(self.via_spot_min_roundness_spin),
            QSignalBlocker(self.via_spot_line_suppression_spin),
            QSignalBlocker(self.via_gradient_min_strength_spin),
            QSignalBlocker(self.via_gradient_min_coverage_spin),
            QSignalBlocker(self.via_hough_edge_threshold_spin),
            QSignalBlocker(self.via_hough_accumulator_threshold_spin),
            QSignalBlocker(self.via_component_min_score_spin),
            QSignalBlocker(self.via_contour_min_score_spin),
            QSignalBlocker(self.via_morphology_peak_scale_spin),
            QSignalBlocker(self.via_template_min_score_spin),
            QSignalBlocker(self.via_blob_min_circularity_spin),
        ]
        try:
            self.via_detector_gradient_checkbox.setChecked(True)
            self.via_detector_spot_checkbox.setChecked(True)
            self.via_detector_hough_checkbox.setChecked(True)
            self.via_detector_components_checkbox.setChecked(True)
            self.via_detector_contours_checkbox.setChecked(True)
            self.via_detector_morphology_checkbox.setChecked(True)
            self.via_detector_template_checkbox.setChecked(True)
            self.via_detector_blob_checkbox.setChecked(True)
            self.via_spot_min_contrast_spin.setValue(18.0)
            self.via_spot_min_roundness_spin.setValue(45.0)
            self.via_spot_line_suppression_spin.setValue(0.65)
            self.via_gradient_min_strength_spin.setValue(12.0)
            self.via_gradient_min_coverage_spin.setValue(0.24)
            self.via_hough_edge_threshold_spin.setValue(80.0)
            self.via_hough_accumulator_threshold_spin.setValue(10.0)
            self.via_component_min_score_spin.setValue(0.0)
            self.via_contour_min_score_spin.setValue(0.0)
            self.via_morphology_peak_scale_spin.setValue(0.18)
            self.via_template_min_score_spin.setValue(0.35)
            self.via_blob_min_circularity_spin.setValue(0.35)
        finally:
            del blockers
        self._update_via_threshold_controls_state()
        self._on_extraction_settings_changed()

    def _add_color_selection(self, row: int, rgb: tuple[int, int, int]) -> None:
        entries = self._color_selection_entries(row)
        for entry in entries:
            if tuple(entry["rgb"]) == tuple(rgb):
                entry["enabled"] = True
                self._pipeline.steps[row].parameters["selected_colors"] = entries
                self._render_pipeline_parameters(row)
                self._auto_apply_pipeline()
                return
        entries.append({"rgb": [int(rgb[0]), int(rgb[1]), int(rgb[2])], "enabled": True})
        self._pipeline.steps[row].parameters["selected_colors"] = entries
        self._render_pipeline_parameters(row)
        self._auto_apply_pipeline()

    def _on_editor_image_clicked(self, x_coord: float, y_coord: float) -> None:
        row = self._color_pick_pipeline_row
        if row is None or row < 0 or row >= len(self._pipeline.steps):
            return
        current_state = self._workspace.current_state
        if current_state is None or current_state.source_image is None:
            return
        image = np.asarray(current_state.source_image)
        x_index = int(round(x_coord))
        y_index = int(round(y_coord))
        if y_index < 0 or x_index < 0 or y_index >= image.shape[0] or x_index >= image.shape[1]:
            return
        if image.ndim == 2:
            value = int(image[y_index, x_index])
            rgb = (value, value, value)
        else:
            pixel = image[y_index, x_index]
            if image.shape[2] >= 3:
                rgb = (int(pixel[2]), int(pixel[1]), int(pixel[0]))
            else:
                value = int(pixel[0])
                rgb = (value, value, value)
        self._add_color_selection(row, rgb)
        self._append_log(
            self._tr(
                "color_picked_log",
                "Добавлен цвет {color}" if self._ui_language == "ru" else "Added color {color}",
                color=f"#{rgb[0]:02X}{rgb[1]:02X}{rgb[2]:02X}",
            )
        )

    def _on_via_debug_requested(self, polygon: PolygonData) -> None:
        current_state = self._workspace.current_state
        candidates = list(current_state.debug_candidates) if current_state is not None else []
        title = "Отладка via" if self._ui_language == "ru" else "Via debug"
        if not candidates:
            message = (
                "Для текущего кадра нет отладочных данных. Включите 'Проверять по клику' и дождитесь повторной обработки."
                if self._ui_language == "ru"
                else "There is no debug data for the current frame. Enable 'Inspect by click' and wait for processing to finish."
            )
            QMessageBox.information(self, title, message)
            return
        candidate = self._best_debug_candidate_for_polygon(polygon, candidates)
        if candidate is None:
            message = (
                "Для этой via не найден исходный кандидат распознавания. Вероятно, объект был создан или изменен вручную после обработки."
                if self._ui_language == "ru"
                else "No source recognition candidate was found for this via. The object was likely created or edited manually after processing."
            )
            QMessageBox.information(self, title, message)
            return
        source = self._debug_candidate_source(candidate)
        reason = str(getattr(candidate, "reason", "") or "")
        accepted = bool(getattr(candidate, "accepted", False))
        bbox = getattr(candidate, "bbox", (0, 0, 0, 0))
        status = "принята" if accepted and self._ui_language == "ru" else "accepted" if accepted else "отклонена" if self._ui_language == "ru" else "rejected"
        lines = [
            f"{'Статус' if self._ui_language == 'ru' else 'Status'}: {status}",
            f"{'Метод' if self._ui_language == 'ru' else 'Method'}: {self._debug_method_text(source)}",
            f"{'Критерий' if self._ui_language == 'ru' else 'Criterion'}: {self._debug_criterion_text(source, reason, accepted)}",
            f"{'Причина' if self._ui_language == 'ru' else 'Reason'}: {reason or '-'}",
            f"{'Оценка' if self._ui_language == 'ru' else 'Score'}: {float(getattr(candidate, 'score', 0.0)):.1f}",
            f"{'Округлость' if self._ui_language == 'ru' else 'Roundness'}: {float(getattr(candidate, 'roundness', 0.0)):.1f}",
            f"{'Размер кандидата' if self._ui_language == 'ru' else 'Candidate size'}: {int(bbox[2])} x {int(bbox[3])} px",
            f"{'Позиция' if self._ui_language == 'ru' else 'Position'}: x={int(bbox[0])}, y={int(bbox[1])}",
        ]
        message = "\n".join(lines)
        self._append_log(message.replace("\n", " | "))
        QMessageBox.information(self, title, message)

    def _best_debug_candidate_for_polygon(self, polygon: PolygonData, candidates: list[object]) -> object | None:
        polygon_rect = self._polygon_rect(polygon)
        if polygon_rect.isNull() or not candidates:
            return None
        polygon_center = polygon_rect.center()
        best_candidate: object | None = None
        best_rank: tuple[int, int, float, float] | None = None
        for index, candidate in enumerate(candidates):
            bbox = getattr(candidate, "bbox", None)
            if not bbox or len(bbox) != 4:
                continue
            candidate_rect = QRectF(float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])).normalized()
            if candidate_rect.isNull():
                continue
            overlap = self._rect_overlap_area(polygon_rect, candidate_rect)
            candidate_center = candidate_rect.center()
            dx = polygon_center.x() - candidate_center.x()
            dy = polygon_center.y() - candidate_center.y()
            distance_sq = dx * dx + dy * dy
            max_span = max(polygon_rect.width(), polygon_rect.height(), candidate_rect.width(), candidate_rect.height(), 1.0)
            if overlap <= 0.0 and distance_sq > (max_span * 1.5) * (max_span * 1.5):
                continue
            accepted_rank = 1 if bool(getattr(candidate, "accepted", False)) else 0
            rank = (accepted_rank, 1 if overlap > 0.0 else 0, overlap, -distance_sq - index * 1e-9)
            if best_rank is None or rank > best_rank:
                best_rank = rank
                best_candidate = candidate
        return best_candidate

    @staticmethod
    def _polygon_rect(polygon: PolygonData) -> QRectF:
        if polygon.points:
            x_values = [point[0] for point in polygon.points]
            y_values = [point[1] for point in polygon.points]
            return QRectF(
                min(x_values),
                min(y_values),
                max(x_values) - min(x_values),
                max(y_values) - min(y_values),
            ).normalized()
        x_coord, y_coord, width, height = polygon.bbox
        return QRectF(float(x_coord), float(y_coord), float(width), float(height)).normalized()

    @staticmethod
    def _rect_overlap_area(first: QRectF, second: QRectF) -> float:
        overlap = first.intersected(second)
        if overlap.isNull():
            return 0.0
        return max(0.0, overlap.width()) * max(0.0, overlap.height())

    @staticmethod
    def _debug_candidate_source(candidate: object) -> str:
        source = str(getattr(candidate, "source", "") or "")
        reason = str(getattr(candidate, "reason", "") or "")
        if not source and ":" in reason:
            source = reason.split(":", 1)[1]
        return source

    def _debug_method_text(self, source: str) -> str:
        source = source.lower()
        labels = {
            "range-components": ("Диапазон яркости + компоненты", "Intensity range + components"),
            "range-contours": ("Диапазон яркости + контуры", "Intensity range + contours"),
            "gradient": ("Градиент округлой формы", "Round gradient"),
            "spot": ("Локальная яркая/темная точка", "Local bright/dark spot"),
            "hough-gray": ("HoughCircles по grayscale", "HoughCircles on grayscale"),
            "hough": ("HoughCircles", "HoughCircles"),
            "components": ("Связанные компоненты", "Connected components"),
            "contours-response": ("Контуры по карте отклика", "Contours on response map"),
            "contours": ("Контуры", "Contours"),
            "morphology": ("Морфологические пики", "Morphology peaks"),
            "template": ("Шаблон", "Template"),
            "blob": ("Blob detector", "Blob detector"),
        }
        for prefix, pair in labels.items():
            if source.startswith(prefix):
                return pair[0] if self._ui_language == "ru" else pair[1]
        return source or ("Неизвестно" if self._ui_language == "ru" else "Unknown")

    def _debug_criterion_text(self, source: str, reason: str, accepted: bool) -> str:
        if not accepted:
            rejection_labels = {
                "duplicate": ("дубликат более сильного ближайшего кандидата", "duplicate of a stronger nearby candidate"),
                "component_score": ("отклик компоненты ниже заданного порога", "component response is below the configured threshold"),
                "contour_score": ("отклик контура ниже заданного порога", "contour response is below the configured threshold"),
                "min_via_width": ("ширина меньше допустимого минимума", "width is below the allowed minimum"),
                "max_via_width": ("ширина больше допустимого максимума", "width is above the allowed maximum"),
                "min_via_height": ("высота меньше допустимого минимума", "height is below the allowed minimum"),
                "max_via_height": ("высота больше допустимого максимума", "height is above the allowed maximum"),
                "min_aspect_ratio": ("соотношение сторон ниже допустимого", "aspect ratio is below the allowed minimum"),
                "max_aspect_ratio": ("соотношение сторон выше допустимого", "aspect ratio is above the allowed maximum"),
                "roundness": ("округлость ниже заданного порога", "roundness is below the configured threshold"),
                "empty_geometry": ("пустая геометрия кандидата", "candidate geometry is empty"),
            }
            pair = rejection_labels.get(reason)
            if pair is not None:
                return pair[0] if self._ui_language == "ru" else pair[1]
            return reason or ("кандидат не прошел фильтры" if self._ui_language == "ru" else "candidate did not pass filters")
        source = source.lower()
        accepted_labels = {
            "range-components": (
                "пиксели попали в заданный диапазон яркости, компонент прошел фильтры размера, формы и округлости",
                "pixels matched the configured intensity range, and the component passed size, shape, and roundness filters",
            ),
            "range-contours": (
                "контур из диапазона яркости прошел фильтры размера, формы и округлости",
                "an intensity-range contour passed size, shape, and roundness filters",
            ),
            "gradient": (
                "найден локальный круглый перепад яркости с достаточной силой и покрытием границы",
                "a local round brightness gradient had enough strength and edge coverage",
            ),
            "spot": (
                "найдена компактная локальная точка после подавления длинных дорожек",
                "a compact local spot was found after suppressing long traces",
            ),
            "hough-gray": (
                "HoughCircles нашел окружность на подготовленном grayscale-изображении",
                "HoughCircles found a circle on the prepared grayscale image",
            ),
            "hough": (
                "HoughCircles нашел окружность на карте отклика",
                "HoughCircles found a circle on the response map",
            ),
            "components": (
                "связанная компонента прошла порог отклика и геометрические фильтры",
                "a connected component passed the response threshold and geometry filters",
            ),
            "contours-response": (
                "контур на карте отклика прошел порог и геометрические фильтры",
                "a response-map contour passed the threshold and geometry filters",
            ),
            "contours": (
                "контур прошел порог отклика и геометрические фильтры",
                "a contour passed the response threshold and geometry filters",
            ),
            "morphology": (
                "морфологический пик прошел фильтры размера и формы",
                "a morphology peak passed size and shape filters",
            ),
            "template": (
                "область совпала с одним из сохраненных шаблонов выше заданного порога",
                "the area matched one of the saved templates above the configured threshold",
            ),
            "blob": (
                "Blob detector нашел компактное пятно с достаточной округлостью",
                "Blob detector found a compact spot with enough circularity",
            ),
        }
        for prefix, pair in accepted_labels.items():
            if source.startswith(prefix):
                return pair[0] if self._ui_language == "ru" else pair[1]
        return (
            "кандидат прошел фильтры размера, пропорций и округлости"
            if self._ui_language == "ru"
            else "candidate passed size, aspect, and roundness filters"
        )

    def _add_pipeline_step(self) -> None:
        operation_name = self._selected_available_operation_name()
        if not operation_name:
            return
        self._pipeline.steps.append(PreprocessingPipeline.create_step(operation_name))
        self._populate_pipeline_list()
        self.pipeline_list.setCurrentRow(len(self._pipeline.steps) - 1)
        self._auto_apply_pipeline()

    def _remove_pipeline_step(self) -> None:
        row = self.pipeline_list.currentRow()
        if row < 0:
            return
        self._pipeline.steps.pop(row)
        self._populate_pipeline_list()
        self._auto_apply_pipeline()

    def _move_pipeline_step_up(self) -> None:
        row = self.pipeline_list.currentRow()
        if row <= 0:
            return
        self._pipeline.steps[row - 1], self._pipeline.steps[row] = self._pipeline.steps[row], self._pipeline.steps[row - 1]
        self._populate_pipeline_list()
        self.pipeline_list.setCurrentRow(row - 1)
        self._auto_apply_pipeline()

    def _move_pipeline_step_down(self) -> None:
        row = self.pipeline_list.currentRow()
        if row < 0 or row >= len(self._pipeline.steps) - 1:
            return
        self._pipeline.steps[row + 1], self._pipeline.steps[row] = self._pipeline.steps[row], self._pipeline.steps[row + 1]
        self._populate_pipeline_list()
        self.pipeline_list.setCurrentRow(row + 1)
        self._auto_apply_pipeline()

    def _on_pipeline_item_changed(self, item: QListWidgetItem) -> None:
        if self._ignore_pipeline_item_change:
            return
        row = self.pipeline_list.row(item)
        if row < 0 or row >= len(self._pipeline.steps):
            return
        self._pipeline.steps[row].enabled = item.checkState() == Qt.CheckState.Checked
        self._auto_apply_pipeline()

    def _sync_pipeline_order_from_list(self) -> None:
        if self._ignore_pipeline_item_change:
            return
        old_steps = list(self._pipeline.steps)
        new_steps = []
        for row in range(self.pipeline_list.count()):
            item = self.pipeline_list.item(row)
            old_index = item.data(Qt.ItemDataRole.UserRole)
            if not isinstance(old_index, int) or old_index < 0 or old_index >= len(old_steps):
                return
            new_steps.append(old_steps[old_index])
        if len(new_steps) != len(old_steps) or all(first is second for first, second in zip(new_steps, old_steps, strict=False)):
            return
        self._pipeline.steps = new_steps
        for row in range(self.pipeline_list.count()):
            self.pipeline_list.item(row).setData(Qt.ItemDataRole.UserRole, row)
        self._render_pipeline_parameters(self.pipeline_list.currentRow())
        self._auto_apply_pipeline()

    def _on_extraction_settings_changed(self, *_args) -> None:
        if hasattr(self, "via_white_range_checkbox"):
            self._update_via_threshold_controls_state()
        self._store_active_extraction_profile_settings()
        if hasattr(self, "polygon_editor"):
            self.polygon_editor.set_debug_candidates([])
            self.polygon_editor.set_via_debug_inspection_enabled(self._via_debug_inspection_enabled())
        self._auto_apply_pipeline()

    def _on_via_size_mode_changed(self, *_args) -> None:
        self._update_via_size_controls_state()
        if normalize_via_size_mode(self.via_size_mode_combo.currentData()) == VIA_SIZE_MODE_FIXED and not self._fixed_via_rows:
            self._add_fixed_via_row(width=1, height=1)
            return
        self._on_extraction_settings_changed()

    def _on_extraction_profile_changed(self, *_args) -> None:
        if self._ignore_extraction_profile_change:
            return
        self._store_active_extraction_profile_settings()
        profile = str(self.extraction_profile_combo.currentData() or "conductors")
        self._active_extraction_profile = profile
        self._set_extraction_settings(self._contour_settings_profiles[profile])
        self._update_extraction_profile_controls_state()
        self._auto_apply_pipeline()

    def _store_active_extraction_profile_settings(self) -> None:
        if not hasattr(self, "extraction_profile_combo"):
            return
        profile = str(self.extraction_profile_combo.currentData() or self._active_extraction_profile or "conductors")
        self._contour_settings_profiles[profile] = self._current_contour_settings()

    def _save_pipeline_json(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self,
            self._tr("save_pipeline_dialog_title"),
            "",
            self._tr("json_file_filter"),
        )
        if not path:
            return
        Path(path).write_text(json.dumps(self.get_pipeline(), indent=2, ensure_ascii=False), encoding="utf-8")
        self._append_log(self._tr("pipeline_saved_log", path=path))

    def _load_pipeline_json(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            self._tr("load_pipeline_dialog_title"),
            "",
            self._tr("json_file_filter"),
        )
        if not path:
            return
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        self.set_pipeline(payload)
        self._append_log(self._tr("pipeline_loaded_log", path=path))

    def _on_image_item_changed(self, current: QListWidgetItem | None, previous: QListWidgetItem | None) -> None:
        if previous is not None:
            self._autosave_current_overlay_if_needed()
        if current is None:
            return
        image_path = current.data(Qt.ItemDataRole.UserRole)
        if image_path:
            try:
                self.load_image(str(image_path))
            except Exception as exc:
                self._append_log(self._tr("failed_to_load_image_log", image_path=image_path, error=exc))
                QMessageBox.warning(self, self._tr("image_load_error_title"), str(exc))

    def _select_input_directory(self) -> None:
        path = QFileDialog.getExistingDirectory(
            self,
            self._tr("select_input_directory_dialog"),
            self.input_dir_edit.text(),
        )
        if path:
            self.set_input_directory(path)

    def _select_cif_directory(self) -> None:
        path = QFileDialog.getExistingDirectory(
            self,
            self._tr("select_cif_directory_dialog"),
            self.cif_dir_edit.text(),
        )
        if path:
            self.set_cif_directory(path)

    def _select_output_directory(self) -> None:
        path = QFileDialog.getExistingDirectory(
            self,
            self._tr("select_output_directory_dialog"),
            self.output_dir_edit.text(),
        )
        if path:
            self.set_output_directory(path)

    def _select_dataset_directory(self) -> None:
        path = QFileDialog.getExistingDirectory(
            self,
            self._tr("select_dataset_directory_dialog"),
            self.dataset_dir_edit.text(),
        )
        if path:
            self.set_dataset_directory(path)

    def _apply_input_directory_edit(self) -> None:
        path = self.input_dir_edit.text().strip()
        if path:
            self.set_input_directory(path)
        else:
            self._workspace.replace_image_selection([], is_supported_image=is_image_path)
            self.image_list.clear()
            self._sync_current_state_views()
            self._save_persisted_paths()

    def _apply_cif_directory_edit(self) -> None:
        path = self.cif_dir_edit.text().strip()
        if path:
            self.set_cif_directory(path)
        else:
            self._workspace.clear_cif_index()
            self._save_persisted_paths()
            if self._workspace.current_image_path:
                try:
                    self.load_image(self._workspace.current_image_path)
                except Exception as exc:
                    self._append_log(self._tr("reload_with_cif_failed_log", error=exc))

    def _apply_output_directory_edit(self) -> None:
        self.set_output_directory(self.output_dir_edit.text().strip())

    def _apply_dataset_directory_edit(self) -> None:
        self.set_dataset_directory(self.dataset_dir_edit.text().strip())

    def _choose_external_color(self) -> None:
        self._choose_color("external_color", self.external_color_button)

    def _choose_hole_color(self) -> None:
        self._choose_color("hole_color", self.hole_color_button)

    def _choose_selected_color(self) -> None:
        self._choose_color("selected_color", self.selected_color_button)

    def _choose_vertex_color(self) -> None:
        self._choose_color("vertex_color", self.vertex_color_button)

    def _choose_color(self, attribute_name: str, button: QPushButton) -> None:
        initial = QColor(getattr(self._display_settings, attribute_name))
        color = QColorDialog.getColor(initial, self, self._tr("select_color_dialog_title"))
        if not color.isValid():
            return
        value = color.name(QColor.NameFormat.HexRgb)
        setattr(self._display_settings, attribute_name, value)
        self._update_color_button(button, value)
        self._apply_display_settings()

    def _apply_display_settings(self) -> None:
        if hasattr(self, "line_width_spin"):
            self._display_settings.line_width = float(self.line_width_spin.value())
            self._display_settings.vertex_size = float(self.vertex_size_spin.value())
            self._display_settings.fill_opacity = float(self.fill_opacity_spin.value())
            self._display_settings.show_vertices = bool(self.show_vertices_checkbox.isChecked())
            self._display_settings.show_labels = bool(self.show_labels_checkbox.isChecked())
        if hasattr(self, "polygon_editor"):
            self.polygon_editor.set_display_settings(self._display_settings)
            if hasattr(self, "random_object_colors_checkbox"):
                self.polygon_editor.set_random_object_colors_enabled(self.random_object_colors_checkbox.isChecked())
        self._save_persisted_display_settings()

    def _on_neighbor_display_settings_changed(self, *_args) -> None:
        self._sync_neighbor_frames()
        self._save_persisted_display_settings()

    def _refresh_extra_layers_list(self) -> None:
        if not hasattr(self, "extra_layers_list"):
            return
        current_row = self.extra_layers_list.currentRow()
        self.extra_layers_list.clear()
        for layer in self._extra_layers:
            name = str(layer.get("name", "Layer"))
            visible = bool(layer.get("visible", True))
            pixmap = layer.get("pixmap")
            loaded = isinstance(pixmap, QPixmap) and not pixmap.isNull()
            prefix = "[x] " if visible else "[ ] "
            if not loaded:
                prefix = "[!] "
            item = QListWidgetItem(prefix + name)
            item.setToolTip(str(layer.get("path", "")))
            self.extra_layers_list.addItem(item)
        if self._extra_layers:
            self.extra_layers_list.setCurrentRow(max(0, min(current_row, len(self._extra_layers) - 1)))
        else:
            self._on_extra_layer_selected(-1)

    def _sync_extra_layers(self) -> None:
        if hasattr(self, "polygon_editor"):
            self.polygon_editor.set_extra_layers(self._extra_layers)

    def _load_extra_layers(self) -> None:
        file_paths, _selected_filter = QFileDialog.getOpenFileNames(
            self,
            "Выберите изображения слоев" if self._ui_language == "ru" else "Select layer images",
            "",
            "Images (*.jpg *.jpeg *.png *.bmp *.tif *.tiff);;All files (*)",
        )
        for file_path in file_paths:
            layer = self._extra_layer_from_path(file_path)
            if layer is not None:
                self._extra_layers.append(layer)
        self._refresh_extra_layers_list()
        self._sync_extra_layers()

    def _browse_selected_extra_layer_path(self) -> None:
        current_path = ""
        row = self.extra_layers_list.currentRow() if hasattr(self, "extra_layers_list") else -1
        if 0 <= row < len(self._extra_layers):
            current_path = str(self._extra_layers[row].get("path", ""))
        file_path, _selected_filter = QFileDialog.getOpenFileName(
            self,
            "Выберите изображение слоя" if self._ui_language == "ru" else "Select layer image",
            str(Path(current_path).parent) if current_path else "",
            "Images (*.jpg *.jpeg *.png *.bmp *.tif *.tiff);;All files (*)",
        )
        if not file_path:
            return
        self.extra_layer_path_edit.setText(file_path)
        self._on_extra_layer_path_changed()

    def _on_extra_layer_path_changed(self) -> None:
        path = self.extra_layer_path_edit.text().strip()
        if not path:
            return
        layer = self._extra_layer_from_path(path)
        if layer is None:
            return
        row = self.extra_layers_list.currentRow() if hasattr(self, "extra_layers_list") else -1
        if row < 0 or row >= len(self._extra_layers):
            self._extra_layers.append(layer)
            row = len(self._extra_layers) - 1
        else:
            existing = self._extra_layers[row]
            existing["name"] = layer["name"]
            existing["path"] = layer["path"]
            existing["pixmap"] = layer["pixmap"]
        self._refresh_extra_layers_list()
        self.extra_layers_list.setCurrentRow(row)
        self._sync_extra_layers()

    def _extra_layer_from_path(self, file_path: str) -> dict[str, object] | None:
        path = Path(file_path.strip().strip("\"'")).expanduser()
        if not path.is_file():
            self._append_log(
                self._tr(
                    "extra_layer_missing_file_log",
                    "Файл слоя не найден: {path}" if self._ui_language == "ru" else "Layer file not found: {path}",
                    path=file_path,
                )
            )
            return None
        pixmap = QPixmap(str(path))
        if pixmap.isNull():
            self._append_log(
                self._tr(
                    "extra_layer_load_failed_log",
                    "Не удалось загрузить изображение слоя: {path}" if self._ui_language == "ru" else "Failed to load layer image: {path}",
                    path=str(path),
                )
            )
            return None
        return {
            "name": path.name,
            "path": str(path),
            "pixmap": pixmap,
            "visible": True,
            "opacity": 0.35,
            "dx": 0.0,
            "dy": 0.0,
        }

    def _remove_selected_extra_layer(self) -> None:
        row = self.extra_layers_list.currentRow() if hasattr(self, "extra_layers_list") else -1
        if row < 0 or row >= len(self._extra_layers):
            return
        self._extra_layers.pop(row)
        self._refresh_extra_layers_list()
        self._sync_extra_layers()

    def _on_extra_layer_selected(self, row: int) -> None:
        blockers = [
            QSignalBlocker(self.extra_layer_path_edit),
            QSignalBlocker(self.extra_layer_visible_checkbox),
            QSignalBlocker(self.extra_layer_opacity_spin),
            QSignalBlocker(self.extra_layer_dx_spin),
            QSignalBlocker(self.extra_layer_dy_spin),
        ]
        if row < 0 or row >= len(self._extra_layers):
            try:
                self.extra_layer_path_edit.clear()
                self.extra_layer_visible_checkbox.setChecked(False)
                self.extra_layer_opacity_spin.setValue(0.35)
                self.extra_layer_dx_spin.setValue(0.0)
                self.extra_layer_dy_spin.setValue(0.0)
            finally:
                del blockers
            return
        layer = self._extra_layers[row]
        try:
            self.extra_layer_path_edit.setText(str(layer.get("path", "")))
            self.extra_layer_visible_checkbox.setChecked(bool(layer.get("visible", True)))
            self.extra_layer_opacity_spin.setValue(float(layer.get("opacity", 0.35) or 0.35))
            self.extra_layer_dx_spin.setValue(float(layer.get("dx", 0.0) or 0.0))
            self.extra_layer_dy_spin.setValue(float(layer.get("dy", 0.0) or 0.0))
        finally:
            del blockers

    def _on_extra_layer_controls_changed(self, *_args) -> None:
        row = self.extra_layers_list.currentRow() if hasattr(self, "extra_layers_list") else -1
        if row < 0 or row >= len(self._extra_layers):
            return
        layer = self._extra_layers[row]
        layer["visible"] = self.extra_layer_visible_checkbox.isChecked()
        layer["opacity"] = float(self.extra_layer_opacity_spin.value())
        layer["dx"] = float(self.extra_layer_dx_spin.value())
        layer["dy"] = float(self.extra_layer_dy_spin.value())
        self._refresh_extra_layers_list()
        self.extra_layers_list.setCurrentRow(row)
        self._sync_extra_layers()

    def _auto_apply_pipeline(self) -> None:
        if self._workspace.current_image_path:
            self.process_current_image(debounced=True)

    def _start_auto_tune_from_reference(self) -> None:
        current_state = self._workspace.current_state
        current_image_path = self._workspace.current_image_path
        reference_polygons = self.get_polygons()

        if current_state is None or current_state.source_image is None or current_image_path is None:
            self._append_log(self._tr("no_image_selected_log", "Изображение не выбрано." if self._ui_language == "ru" else "No image selected."))
            return
        if not reference_polygons:
            self._append_log(
                self._tr(
                    "auto_tune_no_reference_log",
                    "Для автоподбора сначала нарисуйте эталонный полигон или область."
                    if self._ui_language == "ru"
                    else "Draw at least one reference polygon before running auto-fit.",
                )
            )
            return
        if self._auto_tune_running_request_id is not None:
            self._append_log(
                self._tr(
                    "auto_tune_already_running_log",
                    "Автоподбор уже выполняется."
                    if self._ui_language == "ru"
                    else "Auto-fit is already running.",
                )
            )
            return

        self._auto_tune_request_serial += 1
        request_id = self._auto_tune_request_serial
        self._auto_tune_running_request_id = request_id
        self._append_log(
            self._tr(
                "auto_tune_started_log",
                "Запущен автоподбор по {count} полигонам."
                if self._ui_language == "ru"
                else "Auto-fit started using {count} reference polygons.",
                count=len(reference_polygons),
            )
        )
        worker = AutoTuneRunnable(
            request_id=request_id,
            image_path=current_image_path,
            source_image=current_state.source_image,
            reference_polygons=reference_polygons,
        )
        worker.signals.result.connect(self._on_auto_tune_result)
        worker.signals.error.connect(self._on_auto_tune_error)
        worker.signals.finished.connect(self._on_auto_tune_finished)
        self._auto_tune_thread_pool.start(worker)
        self._refresh_busy_indicator()

    def _apply_auto_tune_result(self, result: AutoTuneResult) -> None:
        self._pipeline = PreprocessingPipeline.from_dict(result.pipeline_config)
        self._populate_pipeline_list()
        self._set_extraction_settings(result.contour_settings)
        self.process_current_image()

    def _set_extraction_settings(self, settings: ContourExtractionSettings) -> None:
        blockers = [
            QSignalBlocker(self.extraction_profile_combo),
            QSignalBlocker(self.retrieval_mode_combo),
            QSignalBlocker(self.approximation_mode_combo),
            QSignalBlocker(self.epsilon_spin),
            QSignalBlocker(self.epsilon_relative_checkbox),
            QSignalBlocker(self.min_area_spin),
            QSignalBlocker(self.max_area_spin),
            QSignalBlocker(self.min_perimeter_spin),
            QSignalBlocker(self.min_points_spin),
            QSignalBlocker(self.max_perimeter_spin),
            QSignalBlocker(self.min_bbox_width_spin),
            QSignalBlocker(self.max_bbox_width_spin),
            QSignalBlocker(self.min_bbox_height_spin),
            QSignalBlocker(self.max_bbox_height_spin),
            QSignalBlocker(self.min_aspect_ratio_spin),
            QSignalBlocker(self.max_aspect_ratio_spin),
            QSignalBlocker(self.exclude_border_touching_checkbox),
            QSignalBlocker(self.min_solidity_spin),
            QSignalBlocker(self.min_extent_spin),
            QSignalBlocker(self.min_polygon_angle_spin),
            QSignalBlocker(self.conductor_gradient_checkbox),
            QSignalBlocker(self.conductor_gradient_min_strength_spin),
            QSignalBlocker(self.conductor_gradient_band_radius_spin),
            QSignalBlocker(self.via_size_mode_combo),
            QSignalBlocker(self.via_white_range_checkbox),
            QSignalBlocker(self.via_white_range_min_spin),
            QSignalBlocker(self.via_white_range_max_spin),
            QSignalBlocker(self.via_black_range_checkbox),
            QSignalBlocker(self.via_black_range_min_spin),
            QSignalBlocker(self.via_black_range_max_spin),
            QSignalBlocker(self.via_detector_gradient_checkbox),
            QSignalBlocker(self.via_detector_hough_checkbox),
            QSignalBlocker(self.via_detector_components_checkbox),
            QSignalBlocker(self.via_detector_contours_checkbox),
            QSignalBlocker(self.via_detector_morphology_checkbox),
            QSignalBlocker(self.via_detector_template_checkbox),
            QSignalBlocker(self.via_detector_blob_checkbox),
            QSignalBlocker(self.via_detector_spot_checkbox),
            QSignalBlocker(self.via_spot_min_contrast_spin),
            QSignalBlocker(self.via_spot_min_roundness_spin),
            QSignalBlocker(self.via_spot_line_suppression_spin),
            QSignalBlocker(self.via_gradient_min_strength_spin),
            QSignalBlocker(self.via_gradient_min_coverage_spin),
            QSignalBlocker(self.via_hough_edge_threshold_spin),
            QSignalBlocker(self.via_hough_accumulator_threshold_spin),
            QSignalBlocker(self.via_component_min_score_spin),
            QSignalBlocker(self.via_contour_min_score_spin),
            QSignalBlocker(self.via_morphology_peak_scale_spin),
            QSignalBlocker(self.via_template_min_score_spin),
            QSignalBlocker(self.via_blob_min_circularity_spin),
            QSignalBlocker(self.debug_candidates_checkbox),
            QSignalBlocker(self.via_roundness_spin),
            QSignalBlocker(self.min_via_width_spin),
            QSignalBlocker(self.max_via_width_spin),
            QSignalBlocker(self.min_via_height_spin),
            QSignalBlocker(self.max_via_height_spin),
            QSignalBlocker(self.min_hierarchy_depth_spin),
            QSignalBlocker(self.max_hierarchy_depth_spin),
            QSignalBlocker(self.max_hole_area_ratio_spin),
        ]
        try:
            profile_index = self.extraction_profile_combo.findData(settings.extraction_profile)
            if profile_index >= 0:
                self._ignore_extraction_profile_change = True
                self.extraction_profile_combo.setCurrentIndex(profile_index)
                self._ignore_extraction_profile_change = False
            self._active_extraction_profile = str(settings.extraction_profile or self._active_extraction_profile)
            retrieval_index = self.retrieval_mode_combo.findData(settings.retrieval_mode)
            if retrieval_index >= 0:
                self.retrieval_mode_combo.setCurrentIndex(retrieval_index)
            approximation_index = self.approximation_mode_combo.findData(settings.approximation_mode)
            if approximation_index >= 0:
                self.approximation_mode_combo.setCurrentIndex(approximation_index)
            self.epsilon_spin.setValue(float(settings.epsilon))
            self.epsilon_relative_checkbox.setChecked(bool(settings.epsilon_relative))
            self.min_area_spin.setValue(float(settings.min_area))
            self.max_area_spin.setValue(0.0 if settings.max_area is None else float(settings.max_area))
            self.min_perimeter_spin.setValue(float(settings.min_perimeter))
            self.max_perimeter_spin.setValue(0.0 if settings.max_perimeter is None else float(settings.max_perimeter))
            self.min_points_spin.setValue(int(settings.min_points))
            self.min_bbox_width_spin.setValue(int(settings.min_bbox_width))
            self.max_bbox_width_spin.setValue(0 if settings.max_bbox_width is None else int(settings.max_bbox_width))
            self.min_bbox_height_spin.setValue(int(settings.min_bbox_height))
            self.max_bbox_height_spin.setValue(0 if settings.max_bbox_height is None else int(settings.max_bbox_height))
            self.min_aspect_ratio_spin.setValue(float(settings.min_aspect_ratio))
            self.max_aspect_ratio_spin.setValue(0.0 if settings.max_aspect_ratio is None else float(settings.max_aspect_ratio))
            self.exclude_border_touching_checkbox.setChecked(bool(settings.exclude_border_touching))
            self.min_solidity_spin.setValue(float(settings.min_solidity))
            self.min_extent_spin.setValue(float(settings.min_extent))
            self.min_polygon_angle_spin.setValue(float(settings.min_polygon_angle))
            self.conductor_gradient_checkbox.setChecked(bool(settings.conductor_gradient_enabled))
            self.conductor_gradient_min_strength_spin.setValue(float(settings.conductor_gradient_min_strength))
            self.conductor_gradient_band_radius_spin.setValue(int(settings.conductor_gradient_band_radius))
            via_size_mode_index = self.via_size_mode_combo.findData(normalize_via_size_mode(settings.via_size_mode))
            if via_size_mode_index >= 0:
                self.via_size_mode_combo.setCurrentIndex(via_size_mode_index)
            self.via_white_range_checkbox.setChecked(bool(settings.via_white_range_enabled))
            self.via_white_range_min_spin.setValue(int(settings.via_white_range_min))
            self.via_white_range_max_spin.setValue(int(settings.via_white_range_max))
            self.via_black_range_checkbox.setChecked(bool(settings.via_black_range_enabled))
            self.via_black_range_min_spin.setValue(int(settings.via_black_range_min))
            self.via_black_range_max_spin.setValue(int(settings.via_black_range_max))
            self.via_detector_gradient_checkbox.setChecked(bool(settings.via_detector_gradient_enabled))
            self.via_detector_spot_checkbox.setChecked(bool(settings.via_detector_spot_enabled))
            self.via_detector_hough_checkbox.setChecked(bool(settings.via_detector_hough_enabled))
            self.via_detector_components_checkbox.setChecked(bool(settings.via_detector_components_enabled))
            self.via_detector_contours_checkbox.setChecked(bool(settings.via_detector_contours_enabled))
            self.via_detector_morphology_checkbox.setChecked(bool(settings.via_detector_morphology_enabled))
            self.via_detector_template_checkbox.setChecked(bool(settings.via_detector_template_enabled))
            self.via_detector_blob_checkbox.setChecked(bool(settings.via_detector_blob_enabled))
            self.via_spot_min_contrast_spin.setValue(float(settings.via_spot_min_contrast))
            self.via_spot_min_roundness_spin.setValue(float(settings.via_spot_min_roundness))
            self.via_gradient_min_strength_spin.setValue(float(settings.via_gradient_min_strength))
            self.via_gradient_min_coverage_spin.setValue(float(settings.via_gradient_min_coverage))
            self.via_hough_edge_threshold_spin.setValue(float(settings.via_hough_edge_threshold))
            self.via_hough_accumulator_threshold_spin.setValue(float(settings.via_hough_accumulator_threshold))
            self.via_component_min_score_spin.setValue(float(settings.via_component_min_score))
            self.via_contour_min_score_spin.setValue(float(settings.via_contour_min_score))
            self.via_morphology_peak_scale_spin.setValue(float(settings.via_morphology_peak_scale))
            self.via_template_min_score_spin.setValue(float(settings.via_template_min_score))
            self.via_blob_min_circularity_spin.setValue(float(settings.via_blob_min_circularity))
            self.via_spot_line_suppression_spin.setValue(float(settings.via_spot_line_suppression))
            self._via_template_images = self._normalize_via_template_images(settings.via_template_images)
            self._refresh_via_template_list()
            self.debug_candidates_checkbox.setChecked(bool(settings.debug_enabled))
            self.via_roundness_spin.setValue(float(settings.via_min_roundness))
            self.min_via_width_spin.setValue(int(settings.min_via_width))
            self.max_via_width_spin.setValue(0 if settings.max_via_width is None else int(settings.max_via_width))
            self.min_via_height_spin.setValue(int(settings.min_via_height))
            self.max_via_height_spin.setValue(0 if settings.max_via_height is None else int(settings.max_via_height))
            self._suspend_fixed_via_updates = True
            self._clear_fixed_via_rows()
            for width, height in zip(settings.fixed_via_widths, settings.fixed_via_heights, strict=False):
                self._add_fixed_via_row(width=width, height=height)
            self._suspend_fixed_via_updates = False
            self.min_hierarchy_depth_spin.setValue(int(settings.min_hierarchy_depth))
            self.max_hierarchy_depth_spin.setValue(0 if settings.max_hierarchy_depth is None else int(settings.max_hierarchy_depth))
            self.max_hole_area_ratio_spin.setValue(0.0 if settings.max_hole_area_ratio is None else float(settings.max_hole_area_ratio))
            self._update_via_size_controls_state()
            self._update_via_threshold_controls_state()
            self._update_extraction_profile_controls_state()
        finally:
            self._suspend_fixed_via_updates = False
            self._ignore_extraction_profile_change = False
            del blockers

    def _current_contour_settings(self) -> ContourExtractionSettings:
        max_area = self.max_area_spin.value()
        max_perimeter = self.max_perimeter_spin.value()
        max_bbox_width = self.max_bbox_width_spin.value()
        max_bbox_height = self.max_bbox_height_spin.value()
        max_aspect_ratio = self.max_aspect_ratio_spin.value()
        max_via_width = self.max_via_width_spin.value()
        max_via_height = self.max_via_height_spin.value()
        via_size_mode = normalize_via_size_mode(self.via_size_mode_combo.currentData())
        fixed_via_pairs = self._fixed_via_pairs()
        fixed_via_widths = [width for width, _height in fixed_via_pairs]
        fixed_via_heights = [height for _width, height in fixed_via_pairs]
        max_hierarchy_depth = self.max_hierarchy_depth_spin.value()
        max_hole_area_ratio = self.max_hole_area_ratio_spin.value()
        extraction_profile = str(self.extraction_profile_combo.currentData() or self._active_extraction_profile or "conductors")
        object_type = "via" if extraction_profile == "vias" else "conductor"
        output_mode = "box" if extraction_profile == "vias" else "polygon"
        return ContourExtractionSettings(
            extraction_profile=extraction_profile,
            object_type=object_type,
            output_mode=output_mode,
            retrieval_mode=str(self.retrieval_mode_combo.currentData() or self.retrieval_mode_combo.currentText()),
            approximation_mode=str(self.approximation_mode_combo.currentData() or self.approximation_mode_combo.currentText()),
            epsilon=self.epsilon_spin.value(),
            epsilon_relative=self.epsilon_relative_checkbox.isChecked(),
            min_polygon_angle=self.min_polygon_angle_spin.value(),
            min_area=self.min_area_spin.value(),
            max_area=None if max_area <= 0 else max_area,
            min_perimeter=self.min_perimeter_spin.value(),
            min_points=self.min_points_spin.value(),
            max_perimeter=None if max_perimeter <= 0 else max_perimeter,
            min_bbox_width=self.min_bbox_width_spin.value(),
            max_bbox_width=None if max_bbox_width <= 0 else max_bbox_width,
            min_bbox_height=self.min_bbox_height_spin.value(),
            max_bbox_height=None if max_bbox_height <= 0 else max_bbox_height,
            min_aspect_ratio=self.min_aspect_ratio_spin.value(),
            max_aspect_ratio=None if max_aspect_ratio <= 0 else max_aspect_ratio,
            exclude_border_touching=self.exclude_border_touching_checkbox.isChecked(),
            min_solidity=self.min_solidity_spin.value(),
            min_extent=self.min_extent_spin.value(),
            conductor_gradient_enabled=self.conductor_gradient_checkbox.isChecked(),
            conductor_gradient_min_strength=self.conductor_gradient_min_strength_spin.value(),
            conductor_gradient_band_radius=self.conductor_gradient_band_radius_spin.value(),
            via_size_mode=via_size_mode,
            via_white_range_enabled=self.via_white_range_checkbox.isChecked(),
            via_white_range_min=self.via_white_range_min_spin.value(),
            via_white_range_max=self.via_white_range_max_spin.value(),
            via_black_range_enabled=self.via_black_range_checkbox.isChecked(),
            via_black_range_min=self.via_black_range_min_spin.value(),
            via_black_range_max=self.via_black_range_max_spin.value(),
            via_detector_gradient_enabled=self.via_detector_gradient_checkbox.isChecked(),
            via_detector_spot_enabled=self.via_detector_spot_checkbox.isChecked(),
            via_detector_hough_enabled=self.via_detector_hough_checkbox.isChecked(),
            via_detector_components_enabled=self.via_detector_components_checkbox.isChecked(),
            via_detector_contours_enabled=self.via_detector_contours_checkbox.isChecked(),
            via_detector_morphology_enabled=self.via_detector_morphology_checkbox.isChecked(),
            via_detector_template_enabled=self.via_detector_template_checkbox.isChecked(),
            via_detector_blob_enabled=self.via_detector_blob_checkbox.isChecked(),
            via_spot_min_contrast=self.via_spot_min_contrast_spin.value(),
            via_spot_min_roundness=self.via_spot_min_roundness_spin.value(),
            via_gradient_min_strength=self.via_gradient_min_strength_spin.value(),
            via_gradient_min_coverage=self.via_gradient_min_coverage_spin.value(),
            via_hough_edge_threshold=self.via_hough_edge_threshold_spin.value(),
            via_hough_accumulator_threshold=self.via_hough_accumulator_threshold_spin.value(),
            via_component_min_score=self.via_component_min_score_spin.value(),
            via_contour_min_score=self.via_contour_min_score_spin.value(),
            via_morphology_peak_scale=self.via_morphology_peak_scale_spin.value(),
            via_template_min_score=self.via_template_min_score_spin.value(),
            via_blob_min_circularity=self.via_blob_min_circularity_spin.value(),
            via_spot_line_suppression=self.via_spot_line_suppression_spin.value(),
            via_template_images=[template.copy() for template in self._via_template_images],
            debug_enabled=self.debug_candidates_checkbox.isChecked(),
            via_min_roundness=self.via_roundness_spin.value(),
            min_via_width=self.min_via_width_spin.value(),
            max_via_width=None if max_via_width <= 0 else max_via_width,
            min_via_height=self.min_via_height_spin.value(),
            max_via_height=None if max_via_height <= 0 else max_via_height,
            fixed_via_widths=fixed_via_widths,
            fixed_via_heights=fixed_via_heights,
            min_hierarchy_depth=self.min_hierarchy_depth_spin.value(),
            max_hierarchy_depth=None if max_hierarchy_depth <= 0 else max_hierarchy_depth,
            max_hole_area_ratio=None if max_hole_area_ratio <= 0 else max_hole_area_ratio,
        )

    def _current_save_options(self) -> SaveOptions:
        return SaveOptions(
            save_cif=self.save_cif_checkbox.isChecked(),
            save_csv=self.save_csv_checkbox.isChecked(),
            save_txt=self.save_txt_checkbox.isChecked(),
            save_svg=self.save_svg_checkbox.isChecked(),
            save_preview=self.save_preview_checkbox.isChecked(),
        )

    def _frame_status_for_image(self, image_path: str) -> str:
        if self._workspace.image_has_changes(image_path):
            return FRAME_STATUS_MODIFIED
        if str(Path(image_path)) in self._viewed_image_paths:
            return FRAME_STATUS_VIEWED
        return FRAME_STATUS_UNCHANGED

    def _frame_status_brush(self, status: str) -> QBrush:
        if status == FRAME_STATUS_MODIFIED:
            return QBrush(QColor("#86EFAC"))
        if status == FRAME_STATUS_VIEWED:
            return QBrush(QColor("#D1D5DB"))
        return QBrush(QColor("#D1D5DB"))

    def _apply_frame_status_to_item(self, item: QListWidgetItem, status: str) -> None:
        item.setData(FRAME_STATUS_ROLE, status)
        item.setBackground(self._frame_status_brush(status))

    def _find_image_list_item(self, image_path: str) -> QListWidgetItem | None:
        for index in range(self.image_list.count()):
            item = self.image_list.item(index)
            if item is not None and str(item.data(Qt.ItemDataRole.UserRole) or "") == image_path:
                return item
        return None

    def _update_frame_item_status(self, image_path: str | None) -> None:
        if not image_path:
            return
        item = self._find_image_list_item(image_path)
        if item is None:
            return
        self._apply_frame_status_to_item(item, self._frame_status_for_image(image_path))

    def _refresh_image_list_item_states(self) -> None:
        for index in range(self.image_list.count()):
            item = self.image_list.item(index)
            if item is None:
                continue
            image_path = str(item.data(Qt.ItemDataRole.UserRole) or "")
            self._apply_frame_status_to_item(item, self._frame_status_for_image(image_path))

    def _autosave_current_overlay_if_needed(self) -> None:
        current_state = self._workspace.current_state
        current_image_path = self._workspace.current_image_path
        if current_state is None or current_image_path is None:
            return
        current_polygons = self.get_polygons()
        self._workspace.update_current_polygons(current_polygons)
        current_has_changes = self._workspace.current_image_has_changes()
        if current_has_changes and self.dataset_mode_checkbox.isChecked():
            self._export_dataset_frame_for_state(current_image_path, current_state, current_polygons)
        if not current_state.loaded_cif_path or current_state.source_image is None:
            self._update_frame_item_status(current_image_path)
            return
        if not current_has_changes:
            self._update_frame_item_status(current_image_path)
            return
        image_size = (int(current_state.source_image.shape[1]), int(current_state.source_image.shape[0]))
        try:
            save_polygons_cif(
                current_state.loaded_cif_path,
                current_image_path,
                current_polygons,
                image_size=image_size,
            )
            self._append_log(
                self._tr(
                    "autosaved_cif_log",
                    "Автосохранен CIF: {path}" if self._ui_language == "ru" else "Autosaved CIF: {path}",
                    path=current_state.loaded_cif_path,
                )
            )
        except Exception as exc:
            self._append_log(
                self._tr(
                    "autosave_failed_log",
                    "Не удалось автосохранить CIF {path}: {error}"
                    if self._ui_language == "ru"
                    else "Failed to autosave CIF {path}: {error}",
                    path=current_state.loaded_cif_path,
                    error=exc,
                )
            )
        self._update_frame_item_status(current_image_path)

    def _sync_current_state_views(self) -> None:
        self._updating_views = True
        try:
            display_image = self._display_image_for_current_state()
            current_state = self._workspace.current_state
            polygons = current_state.polygons if current_state else []
            self.polygon_editor.set_image(display_image)
            self.polygon_editor.set_polygons(polygons)
            self.polygon_editor.set_debug_candidates([])
            self.polygon_editor.set_via_debug_inspection_enabled(self._via_debug_inspection_enabled())
            self._sync_neighbor_frames()
            self._sync_extra_layers()
        finally:
            self._updating_views = False

    def _display_image_for_current_state(self):
        return self._workspace.current_display_image()

    def _via_debug_inspection_enabled(self) -> bool:
        return bool(hasattr(self, "debug_candidates_checkbox") and self.debug_candidates_checkbox.isChecked())

    def _neighbor_frame_image(self, image_path: str):
        state = getattr(self._workspace, "_state_cache", {}).get(image_path)
        if state is not None:
            return state.preprocessed_image if state.preprocessed_image is not None else state.source_image
        cached = self._neighbor_image_cache.get(image_path)
        if cached is not None:
            return cached
        try:
            image = load_image_color(image_path)
        except Exception as exc:
            self._append_log(
                self._tr(
                    "neighbor_frame_load_failed_log",
                    "Не удалось загрузить соседний кадр {path}: {error}" if self._ui_language == "ru" else "Failed to load neighbor frame {path}: {error}",
                    path=image_path,
                    error=exc,
                )
            )
            return None
        self._neighbor_image_cache[image_path] = image
        return image

    def _odd_neighbor_grid_size(self, value: int) -> int:
        size = max(3, min(7, int(value)))
        return size if size % 2 else size - 1

    def _neighbor_grid_size_for_zoom(self) -> int:
        max_grid = self._odd_neighbor_grid_size(self.neighbor_max_grid_spin.value())
        zoom = self.polygon_editor.zoom_factor() if hasattr(self, "polygon_editor") else 1.0
        requested = 7 if zoom < 0.25 else 5 if zoom < 0.45 else 3
        return min(max_grid, requested)

    def _sync_neighbor_frames(self) -> None:
        if not hasattr(self, "polygon_editor"):
            return
        if not hasattr(self, "show_neighbor_frames_checkbox") or not self.show_neighbor_frames_checkbox.isChecked():
            self.polygon_editor.set_neighbor_frames([], 0.0, 0, False)
            return
        current_path = self._workspace.current_image_path
        image_paths = list(self._workspace.image_paths)
        if not current_path or current_path not in image_paths:
            self.polygon_editor.set_neighbor_frames([], 0.0, 0, False)
            return
        current_index = image_paths.index(current_path)
        columns = max(1, int(self.neighbor_columns_spin.value()))
        current_row = current_index // columns
        current_column = current_index % columns
        radius = self._neighbor_grid_size_for_zoom() // 2
        frames: list[tuple[int, int, object, str]] = []
        for row_offset in range(-radius, radius + 1):
            for column_offset in range(-radius, radius + 1):
                if row_offset == 0 and column_offset == 0:
                    continue
                row = current_row + row_offset
                column = current_column + column_offset
                if row < 0 or column < 0 or column >= columns:
                    continue
                index = row * columns + column
                if index < 0 or index >= len(image_paths):
                    continue
                image_path = image_paths[index]
                image = self._neighbor_frame_image(image_path)
                if image is None:
                    continue
                frames.append((column_offset, row_offset, image, image_path))
        self.polygon_editor.set_neighbor_frames(
            frames,
            float(self.neighbor_opacity_spin.value()),
            int(self.neighbor_overlap_spin.value()),
            True,
        )

    def _on_neighbor_frame_activated(self, image_path: str) -> None:
        if image_path in self._workspace.image_paths:
            self._autosave_current_overlay_if_needed()
            item = self._find_image_list_item(image_path)
            if item is not None:
                self.image_list.setCurrentItem(item)
            else:
                self.load_image(image_path)

    def _queue_prepared_image_update(self, image_path: str, source_image) -> None:
        request = PreparedImageRequest(
            image_path=image_path,
            source_image=source_image,
            pipeline_config=self.get_pipeline(),
        )
        signature = self._prepared_image_request_signature(request)
        if signature == self._prepared_image_running_signature or signature == self._prepared_image_pending_signature:
            self._refresh_busy_indicator()
            return
        self._prepared_image_pending_request = request
        self._prepared_image_pending_signature = signature
        self._refresh_busy_indicator()
        self._start_pending_prepared_image_update()

    def _start_pending_prepared_image_update(self) -> None:
        if self._prepared_image_running_request_id is not None or self._prepared_image_pending_request is None:
            return
        request = self._prepared_image_pending_request
        self._prepared_image_pending_request = None
        request_signature = self._prepared_image_pending_signature
        self._prepared_image_pending_signature = None
        self._prepared_image_request_serial += 1
        request_id = self._prepared_image_request_serial
        self._prepared_image_running_request_id = request_id
        self._prepared_image_running_signature = request_signature

        worker = PreparedImageRunnable(request_id=request_id, request=request)
        worker.signals.result.connect(self._on_prepared_image_result)
        worker.signals.error.connect(self._on_prepared_image_error)
        worker.signals.finished.connect(self._on_prepared_image_finished)
        self._prepared_image_thread_pool.start(worker)
        self._refresh_busy_indicator()

    def _build_preview_request(self) -> PreviewProcessingRequest | None:
        if not self._workspace.current_image_path:
            return None
        source_image = None
        preprocessed_image = None
        current_state = self._workspace.current_state
        pipeline_config = self.get_pipeline()
        if current_state is not None and current_state.image_path == self._workspace.current_image_path:
            source_image = current_state.source_image
            if current_state.preprocessed_image is not None and current_state.pipeline_config == pipeline_config:
                preprocessed_image = current_state.preprocessed_image
        return PreviewProcessingRequest(
            image_path=self._workspace.current_image_path,
            pipeline_config=pipeline_config,
            contour_settings=self._current_contour_settings(),
            source_image=source_image,
            preprocessed_image=preprocessed_image,
        )

    def _preview_request_signature(self, request: PreviewProcessingRequest) -> tuple[str, str, str]:
        return build_preview_request_signature(request)

    def _prepared_image_request_signature(self, request: PreparedImageRequest) -> tuple[str, str]:
        return build_prepared_image_signature(request)

    def _queue_preview_processing(self, *, debounced: bool) -> None:
        request = self._build_preview_request()
        if request is None:
            self._append_log(self._tr("no_image_selected_log"))
            return
        signature = self._preview_request_signature(request)
        if signature == self._preview_running_signature or signature == self._preview_pending_signature:
            self._refresh_busy_indicator()
            return
        self._preview_pending_request = request
        self._preview_pending_signature = signature
        self._refresh_busy_indicator()
        if debounced:
            self._preview_update_timer.start()
            return
        self._preview_update_timer.stop()
        self._start_pending_preview_processing()

    def _start_pending_preview_processing(self) -> None:
        if self._preview_running_request_id is not None or self._preview_pending_request is None:
            return
        request = self._preview_pending_request
        self._preview_pending_request = None
        request_signature = self._preview_pending_signature
        self._preview_pending_signature = None
        self._preview_request_serial += 1
        request_id = self._preview_request_serial
        self._preview_running_request_id = request_id
        self._preview_running_signature = request_signature

        worker = PreviewProcessingRunnable(request_id=request_id, request=request)
        worker.signals.result.connect(self._on_preview_processing_result)
        worker.signals.error.connect(self._on_preview_processing_error)
        worker.signals.finished.connect(self._on_preview_processing_finished)
        self._preview_thread_pool.start(worker)
        self._refresh_busy_indicator()

    def _append_log(self, message: str) -> None:
        self.logMessage.emit(message)

    def _refresh_busy_indicator(self) -> None:
        active = any(
            (
                self._preview_running_request_id is not None,
                self._preview_pending_request is not None,
                self._preview_update_timer.isActive(),
                self._prepared_image_running_request_id is not None,
                self._prepared_image_pending_request is not None,
                self._auto_tune_running_request_id is not None,
            )
        )
        if hasattr(self, "preview_busy_label"):
            self.preview_busy_label.setText(self._busy_indicator_text())
            self.preview_busy_label.setVisible(active)
        if hasattr(self, "preview_busy_progress"):
            self.preview_busy_progress.setVisible(active)
        if hasattr(self, "auto_tune_button"):
            self.auto_tune_button.setEnabled(self._auto_tune_running_request_id is None)

    def _on_prepared_image_result(self, request_id: int, image_path: str, preprocessed_image, pipeline_config: dict) -> None:
        if request_id != self._prepared_image_running_request_id:
            return
        if pipeline_config != self.get_pipeline():
            return
        if self._workspace.store_preprocessed_image(image_path, preprocessed_image, pipeline_config):
            self._sync_current_state_views()

    def _on_prepared_image_error(self, request_id: int, message: str) -> None:
        if request_id != self._prepared_image_running_request_id:
            return
        self._append_log(self._tr("processing_failed_log", error=message))

    def _on_prepared_image_finished(self, request_id: int) -> None:
        if request_id == self._prepared_image_running_request_id:
            self._prepared_image_running_request_id = None
            self._prepared_image_running_signature = None
        if self._prepared_image_pending_request is not None:
            self._start_pending_prepared_image_update()
        self._refresh_busy_indicator()

    def _on_auto_tune_result(self, request_id: int, result: AutoTuneResult) -> None:
        if request_id != self._auto_tune_running_request_id:
            return
        self._apply_auto_tune_result(result)
        roi_width = result.roi_bbox[2]
        roi_height = result.roi_bbox[3]
        self._append_log(
            self._tr(
                "auto_tune_finished_log",
                "Автоподбор завершён: score={score:.3f}, ROI={width}x{height}, проверок={evaluations}."
                if self._ui_language == "ru"
                else "Auto-fit completed: score={score:.3f}, ROI={width}x{height}, evaluations={evaluations}.",
                score=result.score,
                width=roi_width,
                height=roi_height,
                evaluations=result.evaluations,
            )
        )

    def _on_auto_tune_error(self, request_id: int, message: str) -> None:
        if request_id != self._auto_tune_running_request_id:
            return
        self._append_log(
            self._tr(
                "auto_tune_failed_log",
                "Ошибка автоподбора: {error}" if self._ui_language == "ru" else "Auto-fit failed: {error}",
                error=message,
            )
        )

    def _on_auto_tune_finished(self, request_id: int) -> None:
        if request_id == self._auto_tune_running_request_id:
            self._auto_tune_running_request_id = None
        self._refresh_busy_indicator()

    def _on_preview_processing_result(self, request_id: int, result) -> None:
        if request_id != self._preview_running_request_id:
            return
        if self._workspace.current_image_path != result.image_path:
            return

        if self._workspace.apply_processing_result(result):
            self._sync_current_state_views()
        self._update_frame_item_status(result.image_path)
        self._set_progress_status("current_image_processed_status")
        self._append_log(
            self._tr(
                "current_image_processed_log",
                image_name=Path(result.image_path).name,
                count=len(result.polygons),
            )
        )
        self.imageProcessed.emit(result.image_path, result.polygons)

    def _on_preview_processing_error(self, request_id: int, message: str) -> None:
        if request_id != self._preview_running_request_id:
            return
        self._append_log(self._tr("processing_failed_log", error=message))

    def _on_preview_processing_finished(self, request_id: int) -> None:
        if request_id == self._preview_running_request_id:
            self._preview_running_request_id = None
            self._preview_running_signature = None
        if self._preview_pending_request is not None and not self._preview_update_timer.isActive():
            self._start_pending_preview_processing()
        self._refresh_busy_indicator()

    def _show_batch_progress(self, total: int) -> None:
        if not self._batch_progress_enabled:
            self._hide_batch_progress()
            return
        self.batch_progress_bar.setRange(0, max(1, total))
        self.batch_progress_bar.setValue(0)
        self.batch_progress_bar.setVisible(True)

    def _hide_batch_progress(self) -> None:
        self.batch_progress_bar.setVisible(False)
        self.batch_progress_bar.setRange(0, 100)
        self.batch_progress_bar.setValue(0)

    def _on_polygons_edited(self) -> None:
        if self._updating_views:
            return
        if self._workspace.update_current_polygons(self.get_polygons()):
            self._update_frame_item_status(self._workspace.current_image_path)
            self.polygonsEdited.emit()

    def _on_batch_result(self, result) -> None:
        self.imageProcessed.emit(result.image_path, result.polygons)
        self._append_log(
            self._tr(
                "batch_result_log",
                image_name=Path(result.image_path).name,
                count=len(result.polygons),
            )
        )

    def _on_batch_progress(self, current: int, total: int) -> None:
        if self._batch_progress_enabled:
            self.batch_progress_bar.setRange(0, max(1, total))
            self.batch_progress_bar.setValue(current)
        self._set_progress_status("batch_progress_status", current=current, total=total)
        self.batchProgress.emit(current, total)

    def _on_batch_finished(self) -> None:
        self._batch_progress_enabled = False
        self._hide_batch_progress()
        self._set_progress_status("batch_finished_status")
        self.batchFinished.emit()

    def _on_batch_error(self, image_path: str, message: str) -> None:
        self._append_log(self._tr("batch_error_log", image_name=Path(image_path).name, message=message))

    def refresh_image_list(self) -> None:
        directory = self.input_dir_edit.text().strip()
        if not directory:
            self._append_log(self._tr("input_directory_empty_log"))
            return
        self.load_images(scan_image_files(directory))

    def set_input_directory(self, path: str) -> None:
        directory_state = load_input_directory(path, scan_images=scan_image_files)
        self.input_dir_edit.setText(directory_state.directory)
        self._save_persisted_paths()
        self.load_images(list(directory_state.image_paths))

    def set_cif_directory(self, path: str) -> None:
        directory_state = index_cif_directory(path)
        self.cif_dir_edit.setText(directory_state.directory)
        self._save_persisted_paths()
        self._workspace.set_cif_index(directory_state.indexed_paths)
        self._refresh_image_list_item_states()
        if directory_state.available:
            self._append_log(self._tr("cif_indexed_log", count=len(directory_state.indexed_paths)))
        else:
            self._append_log(self._tr("cif_directory_unavailable_log"))

        if self._workspace.current_image_path:
            try:
                self.load_image(self._workspace.current_image_path)
            except Exception as exc:
                self._append_log(self._tr("reload_with_cif_failed_log", error=exc))

    def set_output_directory(self, path: str) -> None:
        self.output_dir_edit.setText(path)
        self._save_persisted_paths()

    def set_dataset_directory(self, path: str) -> None:
        self.dataset_dir_edit.setText(path)
        self._save_persisted_paths()

    def load_images(self, paths: list[str]) -> None:
        normalized_paths = self._workspace.replace_image_selection(paths, is_supported_image=is_image_path)
        self._neighbor_image_cache.clear()
        self._viewed_image_paths.intersection_update(str(Path(path)) for path in normalized_paths)
        self._preview_update_timer.stop()
        self._preview_pending_request = None
        self._preview_pending_signature = None
        self._prepared_image_pending_request = None
        self._prepared_image_pending_signature = None
        self._refresh_busy_indicator()
        self.image_list.clear()
        for path in normalized_paths:
            item = QListWidgetItem(Path(path).name)
            item.setToolTip(
                (f"Путь к файлу: {path}" if self._ui_language == "ru" else f"File path: {path}")
            )
            item.setData(Qt.ItemDataRole.UserRole, path)
            self._apply_frame_status_to_item(item, self._frame_status_for_image(path))
            self.image_list.addItem(item)
        if normalized_paths:
            self.image_list.setCurrentRow(0)
        else:
            self._sync_current_state_views()

    def _find_matching_cif_path(self, image_path: str) -> str | None:
        return self._workspace.resolve_cif_path(image_path)

    def _load_cif_overlay_polygons(self, image_path: str) -> list[PolygonData]:
        cif_path = self._find_matching_cif_path(image_path)
        if not cif_path:
            return []
        try:
            referenced_image, image_size, polygons = load_polygons_cif(cif_path)
        except Exception as exc:
            self._append_log(self._tr("cif_load_failed_log", file_name=Path(cif_path).name, error=exc))
            return []
        if referenced_image and Path(referenced_image).stem.lower() != Path(image_path).stem.lower():
            self._append_log(
                self._tr(
                    "cif_reference_name_diff_log",
                    file_name=Path(cif_path).name,
                    referenced_image=referenced_image,
                )
            )
        if image_size is not None:
            self._append_log(
                self._tr(
                    "cif_overlay_loaded_with_size_log",
                    file_name=Path(cif_path).name,
                    width=image_size[0],
                    height=image_size[1],
                    count=len(polygons),
                )
            )
        else:
            self._append_log(self._tr("cif_overlay_loaded_log", file_name=Path(cif_path).name, count=len(polygons)))
        return polygons

    def load_image(self, path: str) -> None:
        self._preview_update_timer.stop()
        self._preview_pending_request = None
        self._preview_pending_signature = None
        self._prepared_image_pending_request = None
        self._prepared_image_pending_signature = None
        self._refresh_busy_indicator()
        image_result = self._workspace.load_image(
            path,
            load_source_image=load_image_color,
            load_cif_overlay=self._load_cif_overlay_polygons,
        )
        self._viewed_image_paths.add(str(Path(image_result.image_path)))
        if image_result.state is not None and not image_result.cache_hit and not image_result.reused_current_state:
            image_result.state.loaded_cif_path = self._find_matching_cif_path(image_result.image_path)
            image_result.state.reference_polygons = [polygon.clone() for polygon in image_result.state.polygons]
        if image_result.reused_current_state:
            self._update_frame_item_status(image_result.image_path)
            return
        self._sync_current_state_views()
        self._update_frame_item_status(image_result.image_path)
        if image_result.prepared_image_required and image_result.state is not None and image_result.state.source_image is not None:
            self._queue_prepared_image_update(image_result.image_path, image_result.state.source_image)
        if image_result.cache_hit:
            self._append_log(self._tr("loaded_cached_state_log", image_path=image_result.image_path))
        else:
            self._append_log(self._tr("loaded_image_log", image_path=image_result.image_path))

    def get_polygons(self) -> list[PolygonData]:
        return self.polygon_editor.get_polygons()

    def set_pipeline(self, config: dict) -> None:
        self._pipeline = PreprocessingPipeline.from_dict(config)
        self._populate_pipeline_list()
        self._auto_apply_pipeline()

    def get_pipeline(self) -> dict:
        return self._pipeline.to_dict()

    def process_current_image(self, *_args, debounced: bool = False) -> None:
        self._queue_preview_processing(debounced=debounced)

    def _export_dataset_frame_for_state(
        self,
        image_path: str,
        state: ImageProcessingState,
        polygons: list[PolygonData],
        dataset_directory: str | None = None,
    ) -> dict[str, str]:
        target_directory = dataset_directory or self.dataset_dir_edit.text().strip()
        if not target_directory:
            self._append_log(self._tr("dataset_directory_not_set_log"))
            return {}
        try:
            saved_files = export_dataset_frame(
                dataset_directory=target_directory,
                image_path=image_path,
                polygons=polygons,
                source_image=state.source_image,
            )
        except Exception as exc:
            self._append_log(self._tr("dataset_export_failed_log", image_name=Path(image_path).name, error=exc))
            return {}
        self._append_log(self._tr("dataset_exported_log", image_name=Path(image_path).name, saved_files=saved_files))
        return saved_files

    def export_current_frame_to_dataset(self, dataset_directory: str | None = None) -> dict[str, str]:
        current_state = self._workspace.current_state
        current_image_path = self._workspace.current_image_path
        if current_state is None or current_image_path is None:
            self._append_log(self._tr("nothing_to_save_log"))
            return {}
        current_polygons = self.get_polygons()
        self._workspace.update_current_polygons(current_polygons)
        self._update_frame_item_status(current_image_path)
        return self._export_dataset_frame_for_state(
            current_image_path,
            current_state,
            current_polygons,
            dataset_directory=dataset_directory,
        )

    def save_current_result(
        self,
        output_directory: str | None = None,
        save_options: SaveOptions | None = None,
    ) -> dict[str, str]:
        current_state = self._workspace.current_state
        current_image_path = self._workspace.current_image_path
        if current_state is None or current_image_path is None:
            self._append_log(self._tr("nothing_to_save_log"))
            return {}
        target_directory = output_directory or self.output_dir_edit.text().strip()
        if not target_directory:
            self._append_log(self._tr("output_directory_not_set_log"))
            return {}
        saved_files = save_result_bundle(
            output_directory=target_directory,
            image_path=current_image_path,
            polygons=self.get_polygons(),
            source_image=current_state.source_image,
            display_settings=self._display_settings,
            save_options=save_options or self._current_save_options(),
            metadata={
                "contour_settings": self._current_contour_settings().to_dict(),
                "pipeline": self.get_pipeline(),
            },
        )
        if saved_files:
            self._append_log(self._tr("saved_result_log", saved_files=saved_files))
        return saved_files

    def start_batch_processing(
        self,
        image_paths: list[str] | None = None,
        max_workers: int | None = None,
    ) -> None:
        if self._batch_processor.is_running:
            self._append_log(self._tr("batch_already_running_log"))
            return
        paths = image_paths or list(self._workspace.image_paths)
        if not paths:
            self._append_log(self._tr("batch_no_images_log"))
            return
        output_directory = self.output_dir_edit.text().strip() or None
        save_options = self._current_save_options()
        self._batch_progress_enabled = bool(output_directory and save_options.save_cif)
        self._show_batch_progress(len(paths))
        self._set_progress_status("batch_started_status")
        self._batch_processor.start(
            image_paths=paths,
            pipeline_config=self.get_pipeline(),
            contour_settings=self._current_contour_settings(),
            output_directory=output_directory,
            save_options=save_options,
            display_settings=self._display_settings,
            max_workers=max_workers or self.max_workers_spin.value(),
        )

    def stop_batch_processing(self) -> None:
        self._batch_processor.stop()
