"""Localized tooltip/help dictionaries for the polygon widget UI.

Moved out of ``contour.widget`` during the production-ready refactor to
keep the god-class module focused on widget behavior rather than i18n content.
"""

from __future__ import annotations
from contour.graphics_view import EditorTool

from .i18n_content_parts.module_state import (
    LocalizedTextMap,
    EXTRACTION_HELP_TEXTS,
    PIPELINE_PARAMETER_HELP_TEXTS,
    PIPELINE_CONTROL_TOOLTIPS,
    EDITOR_TOOL_TOOLTIPS,
    EDITOR_ACTION_TOOLTIPS,
    GENERAL_CONTROL_TOOLTIPS,
    PIPELINE_OPERATION_GROUPS,
    PIPELINE_OPERATION_HELP_TEXTS,
)
from .i18n_content_parts.localized_text import _localized_text

EXTRACTION_HELP_TEXTS.update(
    {
        "extraction_profile": (
            "Выбирает, что сейчас настраивается: проводники или переходные отверстия. У каждого профиля свои фильтры и свой результат векторизации.",
            "Selects what is being configured now: conductors or contacts. Each profile has its own filters and vectorization result.",
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
        "min_polygon_width": (
            "Минимальная допустимая ширина полигона в пикселях. "
            "Позволяет отсекать слишком тонкие линии, шумовые артефакты и ложные узкие сегменты. "
            "Если значение слишком маленькое — будет больше ложных полигонов. "
            "Если слишком большое — можно потерять реальные узкие дорожки или контакты.",
            "Minimum allowed polygon width in pixels; 0 disables the filter. Uses a robust local-thickness estimate.",
        ),
        "via_white_threshold": (
            "Добавляет в маску контактов пиксели светлее заданного порога. Включайте для поиска светлых контактов.",
            "Adds pixels brighter than the threshold to the contact mask. Enable it to detect bright contacts.",
        ),
        "via_white_range": (
            "Добавляет в маску контактов пиксели, яркость которых попадает в диапазон светлых контактов.",
            "Adds pixels whose brightness falls inside the bright-contact range to the contact mask.",
        ),
        "via_black_threshold": (
            "Добавляет в маску контактов пиксели темнее заданного порога. Включайте для поиска тёмных контактов.",
            "Adds pixels darker than the threshold to the contact mask. Enable it to detect dark contacts.",
        ),
        "via_black_range": (
            "Добавляет в маску контактов пиксели, яркость которых попадает в диапазон тёмных контактов.",
            "Adds pixels whose brightness falls inside the dark-contact range to the contact mask.",
        ),
        "via_min_score": (
            "Единый итоговый порог для контактов от 0 до 1. Комбинирует отклик, радиальный контраст и покрытие кромки. Увеличьте, чтобы оставить только самые уверенные кандидаты.",
            "Unified final contact score threshold from 0 to 1. Combines response, radial contrast and edge coverage. Increase to keep only the most confident candidates.",
        ),
        "via_min_contrast": (
            "Минимальный радиальный контраст центра контакта относительно кольца вокруг него (в уровнях 0-255). Увеличьте, чтобы отсечь слабые пятна.",
            "Minimum radial contrast of the contact center vs the surrounding ring (0-255 levels). Increase to reject weak spots.",
        ),
        "via_min_edge_coverage": (
            "Минимальная доля окружности, на которой есть заметная кромка. Увеличьте, чтобы требовать более замкнутую круглую границу.",
            "Minimum fraction of the circle that must show a visible edge. Increase to require a more complete round boundary.",
        ),
        "via_spot_line_suppression": (
            "Подавляет длинные горизонтальные и вертикальные дорожки перед поиском круглых точек. Увеличьте, если на дорожках появляются ложные контакты; уменьшите, если реальные контакты слишком сильно стираются.",
            "Suppresses long horizontal and vertical traces before round-dot detection. Increase it when traces create false contacts; decrease it if real contacts are erased.",
        ),
        "via_template_min_score": (
            "Минимальное совпадение с круглым шаблоном. Больше значение требует более похожую на круг область.",
            "Minimum circular-template match. Higher values require an area that looks more like a circle.",
        ),
        "via_templates": (
            "Список шаблонов контактов. Удерживайте Ctrl и протяните рамку по контакту на изображении; все шаблоны используются при поиске.",
            "List of contact templates. Hold Ctrl and drag a rectangle over a contact; all templates are used during detection.",
        ),
        "reset_via_search": (
            "Возвращает методы поиска контактов и их параметры к значениям по умолчанию. Сохранённые шаблоны не удаляются.",
            "Restores contact search methods and their parameters to defaults. Saved templates are not removed.",
        ),
        "via_noisy_traces_preset": (
            "Быстро настраивает поиск для кадров, где яркие круглые контакты находятся на фоне длинных дорожек и шума.",
            "Quick setup for frames where bright round contacts sit on long traces and noisy background.",
        ),
        "via_blurred_preset": (
            "Быстро настраивает поиск для слабых или размытых контактов.",
            "Quick setup for weak or blurred contacts.",
        ),
        "via_preset_selector": (
            "Список встроенных и сохранённых пользователем пресетов поиска контактов. Пресет меняет только экспертные параметры; шаблоны и размеры контактов остаются на месте.",
            "List of built-in and user-saved contact search presets. A preset changes expert parameters only; templates and contact sizes stay unchanged.",
        ),
        "debug_candidates": (
            "Включает инспектор распознавания контактов. После обработки кликните по найденному контакту, чтобы увидеть метод поиска, причину принятия и основные численные признаки.",
            "Enables the contact recognition inspector. After processing, click a detected contact to see the search method, acceptance reason, and main numeric features.",
        ),
        "via_threshold_range": (
            "Добавляет в маску контактов пиксели, яркость которых попадает в заданный диапазон. Удобно для контактов среднего тона.",
            "Adds pixels whose intensity falls inside the selected range to the contact mask. Useful for mid-tone contacts.",
        ),
        "via_min_roundness": (
            "Минимальная похожесть контакта на круг в процентах. Увеличьте, чтобы убрать вытянутые пятна; 0 отключает фильтр.",
            "Minimum contact circularity in percent. Increase it to reject elongated spots; 0 disables the filter.",
        ),
        "via_size_mode": (
            "Выбирает способ отбора контактов по размеру: общий диапазон ширины/высоты или список точных размеров.",
            "Chooses how contacts are filtered by size: a width/height range or a list of exact sizes.",
        ),
        "via_search_mode": (
            "Метод поиска контактов: эвристический, по шаблону или смешанный.",
            "Contact search method: heuristic, template, or mixed.",
        ),
        "min_via_width": (
            "Минимальная ширина переходного отверстия. Более узкие объекты будут отброшены.",
            "Minimum contact width. Narrower objects are rejected.",
        ),
        "max_via_width": (
            "Максимальная ширина переходного отверстия. Ноль означает без верхнего ограничения.",
            "Maximum contact width. Zero means no upper limit.",
        ),
        "min_via_height": (
            "Минимальная высота переходного отверстия. Более низкие объекты будут отброшены.",
            "Minimum contact height. Shorter objects are rejected.",
        ),
        "max_via_height": (
            "Максимальная высота переходного отверстия. Ноль означает без верхнего ограничения.",
            "Maximum contact height. Zero means no upper limit.",
        ),
        "fixed_via_widths": (
            "Ширина контакта из списка точных размеров. Пара X/Y в одной строке описывает один допустимый размер.",
            "Contact width in the exact-size list. The X/Y pair in one row describes one allowed size.",
        ),
        "fixed_via_heights": (
            "Высота контакта из списка точных размеров. Пара X/Y в одной строке описывает один допустимый размер.",
            "Contact height in the exact-size list. The X/Y pair in one row describes one allowed size.",
        ),
        "min_hierarchy_depth": (
            "Минимальная глубина вложенности контура. Ноль означает внешний контур; большие значения выбирают внутренние контуры.",
            "Minimum contour nesting depth. Zero means an outer contour; higher values select inner contours.",
        ),
        "min_inner_hole_area": (
            "Минимальная площадь отверстия для заливки. Меньшие отверстия удаляются при извлечении проводников и ручной постобработке.",
            "Minimum hole area to fill. Smaller holes are removed during conductor extraction and manual post-processing.",
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

# Imported so existing callers and tests can patch or import these names
# from this module after the implementation moved into focused units.
_FACADE_EXPORTS = (
    LocalizedTextMap,
    EXTRACTION_HELP_TEXTS,
    PIPELINE_PARAMETER_HELP_TEXTS,
    PIPELINE_CONTROL_TOOLTIPS,
    EDITOR_TOOL_TOOLTIPS,
    EDITOR_ACTION_TOOLTIPS,
    GENERAL_CONTROL_TOOLTIPS,
    PIPELINE_OPERATION_GROUPS,
    PIPELINE_OPERATION_HELP_TEXTS,
    _localized_text,
)
