"""PIPELINE_OPERATION_HELP_TEXTS."""
from __future__ import annotations
from contour.graphics_view import EditorTool

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
    "brightness": {
        "summary": (
            "Линейно сдвигает яркость изображения без изменения множителя контраста.",
            "Linearly shifts image brightness without changing the contrast multiplier.",
        ),
        "use": (
            "Используйте, когда кадр нужно сделать светлее или темнее перед threshold.",
            "Use when a frame needs to be brightened or darkened before thresholding.",
        ),
    },
    "contrast": {
        "summary": (
            "Линейно усиливает или ослабляет различия яркостей без отдельного сдвига яркости.",
            "Linearly increases or reduces tonal separation without a separate brightness shift.",
        ),
        "use": (
            "Используйте, когда границы нужно сделать выразительнее без изменения базовой яркости.",
            "Use when edges need stronger separation without changing the base brightness level.",
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
            "Useful for finding small bright contacts or spots.",
        ),
    },
    "blackhat": {
        "summary": (
            "Выделяет маленькие тёмные детали на более светлом фоне.",
            "Highlights small dark details on a lighter background.",
        ),
        "use": (
            "Полезен для тёмных отверстий или канавок на светлом поле.",
            "Useful for dark contacts or grooves on a bright field.",
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
            "Лучший выбор, когда важны круглые контакты.",
            "Best choice when the goal is round contacts (blob-like features).",
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
