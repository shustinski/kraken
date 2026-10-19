"""PIPELINE_PARAMETER_HELP_TEXTS."""
from __future__ import annotations
from typing import TYPE_CHECKING
from contour.graphics_view import EditorTool

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

if TYPE_CHECKING:
    from contour.ui.i18n_content_parts.module_state_parts.module_state import LocalizedTextMap
