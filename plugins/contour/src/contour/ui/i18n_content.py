"""Localized tooltip/help dictionaries for the polygon widget UI.

Moved out of ``contour.widget`` during the production-ready refactor to
keep the god-class module focused on widget behavior rather than i18n content.
"""

from __future__ import annotations

from contour.graphics_view import EditorTool

LocalizedTextMap = dict[str, tuple[str, str]]


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
        "min_polygon_width": (
            "Минимальная допустимая ширина полигона в пикселях. "
            "Позволяет отсекать слишком тонкие линии, шумовые артефакты и ложные узкие сегменты. "
            "Если значение слишком маленькое — будет больше ложных полигонов. "
            "Если слишком большое — можно потерять реальные узкие дорожки или контакты.",
            "Minimum allowed polygon width in pixels; 0 disables the filter. Uses a robust local-thickness estimate.",
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
        "via_min_score": (
            "Единый итоговый порог для via от 0 до 1. Комбинирует отклик, радиальный контраст и покрытие кромки. Увеличьте, чтобы оставить только самые уверенные кандидаты.",
            "Unified final via score threshold from 0 to 1. Combines response, radial contrast and edge coverage. Increase to keep only the most confident candidates.",
        ),
        "via_min_contrast": (
            "Минимальный радиальный контраст центра via относительно кольца вокруг него (в уровнях 0-255). Увеличьте, чтобы отсечь слабые пятна.",
            "Minimum radial contrast of the via center vs the surrounding ring (0-255 levels). Increase to reject weak spots.",
        ),
        "via_min_edge_coverage": (
            "Минимальная доля окружности, на которой есть заметная кромка. Увеличьте, чтобы требовать более замкнутую круглую границу.",
            "Minimum fraction of the circle that must show a visible edge. Increase to require a more complete round boundary.",
        ),
        "via_spot_line_suppression": (
            "\u041f\u043e\u0434\u0430\u0432\u043b\u044f\u0435\u0442 \u0434\u043b\u0438\u043d\u043d\u044b\u0435 \u0433\u043e\u0440\u0438\u0437\u043e\u043d\u0442\u0430\u043b\u044c\u043d\u044b\u0435 \u0438 \u0432\u0435\u0440\u0442\u0438\u043a\u0430\u043b\u044c\u043d\u044b\u0435 \u0434\u043e\u0440\u043e\u0436\u043a\u0438 \u043f\u0435\u0440\u0435\u0434 \u043f\u043e\u0438\u0441\u043a\u043e\u043c \u043a\u0440\u0443\u0433\u043b\u044b\u0445 \u0442\u043e\u0447\u0435\u043a. \u0423\u0432\u0435\u043b\u0438\u0447\u044c\u0442\u0435, \u0435\u0441\u043b\u0438 \u043d\u0430 \u0434\u043e\u0440\u043e\u0436\u043a\u0430\u0445 \u043f\u043e\u044f\u0432\u043b\u044f\u044e\u0442\u0441\u044f \u043b\u043e\u0436\u043d\u044b\u0435 via; \u0443\u043c\u0435\u043d\u044c\u0448\u0438\u0442\u0435, \u0435\u0441\u043b\u0438 via \u0441\u043b\u0438\u0448\u043a\u043e\u043c \u0432\u044b\u0442\u0435\u0440\u043b\u0438\u0441\u044c.",
            "Suppresses long horizontal and vertical traces before round-dot detection. Increase it when traces create false vias; decrease it if real vias are erased.",
        ),
        "via_template_min_score": (
            "Минимальное совпадение с круглым шаблоном. Больше значение требует более похожую на круг область.",
            "Minimum circular-template match. Higher values require an area that looks more like a circle.",
        ),
        "via_templates": (
            "Список шаблонов via. Нажмите выбор шаблона и протяните рамку по переходному отверстию на изображении; все шаблоны используются при поиске.",
            "List of via templates. Click pick template and drag a rectangle over a via in the image; all templates are used during detection.",
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
        "via_search_mode": (
            "Выбирает режим детекции via: гибрид (blob+шаблоны), только blob или только поиск по шаблонам.",
            "Selects the via detection mode: hybrid (blob+templates), blob only, or template-only matching.",
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
        "Создание нового полигона точками или прямоугольником.",
        "Create a new polygon with points or a rectangle.",
    ),
    EditorTool.BRUSH: (
        "Рисование или стирание области кистью. Круг под курсором показывает текущую толщину кисти.",
        "Draw or erase an area with the brush. The circle under the cursor shows the current brush width.",
    ),
    EditorTool.TRACE_PEN: (
        "Trace pen: drag from a trace centerline start to its end. Shift snaps to 45 degrees; right drag erases.",
        "Trace pen: drag from a trace centerline start to its end. Shift snaps to 45 degrees; right drag erases.",
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
    EditorTool.ANTIALIAS: (
        "Antialias polygon under cursor.",
        "Shows vertices for the polygon under the cursor and reduces its vertex count on click.",
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
        "Папка с векторной разметкой CIF/CV для наложения на изображения. Можно оставить пустой, если разметка не нужна.",
        "Folder with CIF/CV vector annotations to overlay on images. Leave empty if annotations are not needed.",
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
        "Выбрать папку с CIF/CV-разметкой для наложения.",
        "Choose the folder with CIF/CV annotations for overlay.",
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
    "pick_input_images": (
        "Выбрать отдельные файлы изображений (не всю папку).",
        "Pick individual image files instead of scanning the entire folder.",
    ),
    "merge_cif_files": (
        "Добавить в индекс отдельные .cif или .cv файлы (по имени сопоставляются с изображением).",
        "Add individual .cif or .cv files to the overlay index (matched by basename to images).",
    ),
    "sidebar_list_mode": (
        "Переключить список справа между кадрами и векторами из папки CIF.",
        "Switch the sidebar between image frames and CIF overlays from the overlay folder.",
    ),
    "vector_list_sidebar": (
        "Файлы CIF из текущего индекса. Выбор переходит на кадр с тем же именем.",
        "CIF entries from the current index. Choosing a row selects the matching image basename.",
    ),
    "reload_selected_cif_overlays": (
        "Сбросить кэш и перечитать CIF для строк, выделенных в этом списке.",
        "Clear caches and reload CIF overlays for the selected sidebar rows.",
    ),
    "reload_cif_for_selected_frames": (
        "Сбросить кэш и перечитать CIF для выделенных кадров (или текущего).",
        "Clear caches and reload CIF overlays for selected image frames (or the current one).",
    ),
    "frame_nav_previous": (
        "Перейти к предыдущему кадру списка.",
        "Jump to the previous frame in the list.",
    ),
    "frame_nav_next": (
        "Перейти к следующему кадру списка.",
        "Jump to the next frame in the list.",
    ),
    "frame_nav_jump": (
        "Выбрать кадр по номеру (1 … N в текущем списке изображений).",
        "Select a frame by index (1 … N within the loaded image list).",
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
    "save_cif": (
        "Сохранять векторную разметку в CIF.",
        "Save vector annotations as CIF.",
    ),
    "save_cv": (
        "Сохранять векторную разметку в текстовый .cv файл",
        "Save vector annotations as text .cv file.",
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
    "conductor_hover_highlight": (
        "Цвет контура проводника при наведении указателя на проводник, отверстие в нём или переходное отверстие (via).",
        "Outline color used for the trace when the pointer hovers the trace, a hole inside it, or a via.",
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
    (
        "advanced_edges",
        ("Современные детекторы границ", "Modern edge detectors"),
        (
            "scharr_edges",
            "auto_canny",
            "log_edges",
            "ridge_edges",
            "structured_edges",
            "phase_congruency",
            "combined_edges",
            "edge_method",
        ),
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
    "scharr_edges": {
        "summary": (
            "Градиент Шарра: резче Sobel на тонких линиях и мелких деталях.",
            "Scharr gradient: sharper than Sobel on thin lines and fine detail.",
        ),
        "use": (
            "Замените Sobel/Canny, когда нужно поймать очень тонкие или слабо-контрастные границы проводников.",
            "Use instead of Sobel/Canny to capture very thin or low-contrast conductor edges.",
        ),
    },
    "auto_canny": {
        "summary": (
            "Canny с автоматическими порогами по медиане яркости — работает без ручной настройки.",
            "Canny with automatic median-based thresholds — no manual tuning needed.",
        ),
        "use": (
            "Удобно для серий снимков с разной экспозицией: пороги подстраиваются под каждый кадр.",
            "Handy for image series with varying exposure: thresholds adapt to every frame.",
        ),
    },
    "log_edges": {
        "summary": (
            "Laplacian of Gaussian на нескольких масштабах — сильно откликается на пятна и точки.",
            "Multi-scale Laplacian of Gaussian — highly responsive to blobs and spots.",
        ),
        "use": (
            "Лучший выбор, когда важны переходные отверстия (via) и круглые контакты.",
            "Best choice when the goal is vias / round contacts (blob-like features).",
        ),
    },
    "ridge_edges": {
        "summary": (
            "Отклик Гессиана подсвечивает гребни — длинные вытянутые структуры.",
            "Hessian ridge response highlights long tubular structures and crests.",
        ),
        "use": (
            "Подходит для длинных проводников и тонких трасс, которые Sobel пропускает.",
            "Good for long conductors and thin traces that Sobel tends to miss.",
        ),
    },
    "structured_edges": {
        "summary": (
            "Обученный детектор Structured Random Forest (opencv-contrib). При отсутствии модели — усиленный Scharr с non-max suppression.",
            "Trained Structured Random Forest edge detector (opencv-contrib). Falls back to Scharr with non-max suppression when the model is missing.",
        ),
        "use": (
            "Наиболее стабильные границы на реальных снимках PCB; fallback тоже даёт чище результат, чем обычный Sobel.",
            "Yields the most stable edges on real PCB images; the fallback already beats a plain Sobel response.",
        ),
    },
    "phase_congruency": {
        "summary": (
            "Мера phase congruency: инвариантна к контрасту и освещению (log-Gabor в частотной области).",
            "Phase congruency: a contrast- and illumination-invariant edge feature (log-Gabor in the frequency domain).",
        ),
        "use": (
            "Лучший выбор при неравномерной подсветке и слабом локальном контрасте.",
            "Best choice for uneven lighting and low local contrast.",
        ),
    },
    "combined_edges": {
        "summary": (
            "Ансамбль нескольких детекторов (попиксельный максимум). Стабильнее любого отдельного оператора.",
            "Ensemble of several detectors (pixel-wise maximum). More robust than any single operator.",
        ),
        "use": (
            "Когда заранее неизвестно, какой оператор сработает лучше — выберите готовый пресет ('robust', 'fine_detail'…).",
            "When it is unclear which operator works best — pick a ready preset ('robust', 'fine_detail'…).",
        ),
    },
    "edge_method": {
        "summary": (
            "Диспетчер: выбирает, какой из современных детекторов применить (sobel / scharr / log / auto_canny / structured / ridge / phase_congruency / combined).",
            "Dispatcher: selects which modern detector to apply (sobel / scharr / log / auto_canny / structured / ridge / phase_congruency / combined).",
        ),
        "use": (
            "Добавьте один раз и переключайте метод без пересборки pipeline.",
            "Add it once and switch methods without rebuilding the pipeline.",
        ),
    },
}


def _localized_text(mapping: LocalizedTextMap, key: str, language: str) -> str:
    entry = mapping.get(key, ("", ""))
    return entry[0] if language == "ru" else entry[1]
