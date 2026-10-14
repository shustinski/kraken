"""_apply_bright_via_tooltips section 3."""
from __future__ import annotations

def __apply_bright_via_tooltips_section_3(self, label_widget, tt):
    if getattr(self, "metal_gap_bridge_spin", None) is not None:
        self.metal_gap_bridge_spin.setToolTip(
            tt(
                "Сшивка разрывов в маске (morphological close): соединяет обрывы дорожек.\n"
                "Слишком большое значение может слить соседние проводники.",
                "Gap bridging radius in pixels.",
            )
        )
    if getattr(self, "metal_speckle_removal_spin", None) is not None:
        self.metal_speckle_removal_spin.setToolTip(
            tt(
                "Удаление мелкого шума (morphological open): убирает одиночные пиксели и короткие отростки.",
                "Speckle removal radius in pixels.",
            )
        )
    if getattr(self, "metal_contour_smooth_spin", None) is not None:
        self.metal_contour_smooth_spin.setToolTip(
            tt(
                "Сглаживание контура перед векторизацией: убирает зубчатость от шума SEM.",
                "Contour smoothing before polygon simplification.",
            )
        )
    if getattr(self, "metal_min_width_spin", None) is not None:
        self.metal_min_width_spin.setToolTip(
            tt(
                "Оценка эффективной ширины по маске (медиальное ядро): объекты уже порога отбрасываются как шумовые царапины.\n"
                "Увеличение убирает тонкие ложные сегменты, но может отрезать реальные узкие дорожки; уменьшение спасает тонкие линии, но пропускает больше мусора.\n"
                "Стартуйте с 6–10 px для тонких технологий и 10–14 px для грубого SEM.",
                "Minimum conductor width in pixels.",
            )
        )
    if getattr(self, "metal_max_width_spin", None) is not None:
        self.metal_max_width_spin.setToolTip(
            tt(
                "Верхняя граница ширины: отсекает широкие заливки, контактные площадки и яркие «пятна», не являющиеся трассами.\n"
                "0 или пусто — без ограничения. Уменьшайте максимум, если в результат попадают крупные артефакты; увеличивайте, если режет широкие шины.\n"
                "Часто 40–120 px в зависимости от масштаба кадра.",
                "Maximum trace width; 0 = unlimited.",
            )
        )
    if getattr(self, "metal_min_length_spin", None) is not None:
        self.metal_min_length_spin.setToolTip(
            tt(
                "Минимальная длина по ограничивающему прямоугольнику: короткие фрагменты травления и одиночные засветы отсекаются.\n"
                "Увеличение сильнее чистит шум; уменьшение сохраняет короткие, но реальные сегменты (перемычки, стабы).\n"
                "Рабочий диапазон обычно 18–40 px.",
                "Minimum trace length.",
            )
        )
    if getattr(self, "metal_adaptive_block_spin", None) is not None:
        self.metal_adaptive_block_spin.setToolTip(
            tt(
                "Размер окна локального порога. 0 — автоматически от размера кадра (как раньше: нечётное окно около 15–63 px).\n"
                "Увеличение сглаживает неравномерную засветку, но может залить узкие зазоры.\n"
                "Уменьшение лучше держит тонкие щели, но сильнее реагирует на зерно SEM.",
                "Local-threshold window size. 0 = automatic from the frame size (odd 15–63 px).\n"
                "Larger windows flatten uneven illumination but may fill narrow gaps.\n"
                "Smaller windows keep thin seams but follow SEM grain more.",
            )
        )
    if getattr(self, "metal_adaptive_c_spin", None) is not None:
        self.metal_adaptive_c_spin.setToolTip(
            tt(
                "Смещение C в формуле адаптивного порога. Положительное значение делает маску строже (меньше металла).\n"
                "Отрицательное — мягче, больше заливки и риск слияний.",
                "Offset C in the adaptive-threshold formula. Positive values make the mask stricter.\n"
                "Negative values recover more fill and increase merge risk.",
            )
        )
    if getattr(self, "metal_adaptive_method_combo", None) is not None:
        self.metal_adaptive_method_combo.setToolTip(
            tt(
                "Как считается локальный фон: взвешенное гауссово среднее или простое среднее по окну.",
                "How the local background is estimated: Gaussian-weighted mean or plain window mean.",
            )
        )
    if getattr(self, "metal_ws_smoothing_spin", None) is not None:
        self.metal_ws_smoothing_spin.setToolTip(
            tt(
                "Сглаживание кадра перед выбором затравок водораздела. Убирает зерно SEM, чтобы ядра и зазоры не прыгали по шуму.\n"
                "Увеличение: границы спокойнее, меньше ложных затравок; узкие щели могут заплыть, соседние проводники — слиться.\n"
                "Уменьшение: тонкие зазоры и мелкие детали сохраняются лучше, но шум чаще даёт рваные контуры и лишние полигоны.",
                "Gaussian smoothing before watershed seeds.\n"
                "Increase: calmer borders, fewer noisy seeds; narrow gaps may close and neighbouring traces may merge.\n"
                "Decrease: keeps thin gaps and fine detail, but noise more often splits polygons.",
            )
        )
    if getattr(self, "metal_ws_core_margin_spin", None) is not None:
        self.metal_ws_core_margin_spin.setToolTip(
            tt(
                "На сколько единиц яркости ниже порога металла начинать ядро. Ядро — внутренняя затравка, от которой растёт полигон.\n"
                "Увеличение: ядра крупнее, бледная середина широких заливок заполняется надёжнее; слишком большое значение может перепрыгнуть узкий зазор и слить трассы.\n"
                "Уменьшение: ядра остаются только на самых ярких кромках, полигоны реже сливаются; бледная заливка может не заполниться или развалиться на фрагменты.",
                "How far below the metal Otsu limit a pixel still counts as a metal core seed.\n"
                "Increase: cores grow, pale fills recover better; too high and cores jump the gap and merge traces.\n"
                "Decrease: cores stay on the brightest rims, fewer merges; pale interiors may stay empty or fragment.",
            )
        )
    if getattr(self, "metal_ws_groove_margin_spin", None) is not None:
        self.metal_ws_groove_margin_spin.setToolTip(
            tt(
                "На сколько единиц яркости выше порога подложки считать тёмную точку затравкой зазора. Затравка зазора останавливает рост металла.\n"
                "Увеличение: серые канавки между трассами находятся увереннее; тёмная текстура внутри заливки чаще принимается за зазор и крошит широкий полигон.\n"
                "Уменьшение: затравками остаются только самые тёмные канавки, широкие заливки целее; узкие или светлые зазоры могут не разделить соседние проводники.",
                "How far above the substrate Otsu limit a pixel still counts as a gap seed that stops metal growth.\n"
                "Increase: grey channels between traces are found more reliably; texture inside a pour is more likely to split it.\n"
                "Decrease: only the darkest grooves remain seeds, pours stay whole; faint gaps may fail to separate neighbours.",
            )
        )
    if getattr(self, "metal_ws_rim_probe_spin", None) is not None:
        self.metal_ws_rim_probe_spin.setToolTip(
            tt(
                "Радиус кольца вокруг тёмной затравки. Если кольцо яркое (кромки соседей), это настоящий зазор; если вокруг заливка — пятно текстуры внутри металла, затравку отбрасываем.\n"
                "Увеличение: лучше отсекает тёмные пятна внутри широких заливок; слишком большое кольцо может «увидеть» далёкий металл и оставить ложную затравку.\n"
                "Уменьшение: требует, чтобы яркая кромка была вплотную к канавке — узкие зазоры сохраняются, но пятна внутри заливки чаще проходят как зазоры.",
                "How far around a dark seed we look for a bright rim. A real gap is lined with metal edges; a speck inside a pour is not.\n"
                "Increase: drops more texture specks inside wide fills; too large and a distant rim may keep a false gap seed.\n"
                "Decrease: only tightly rim-lined grooves survive, so narrow gaps stay; interior specks more often split a pour.",
            )
        )
    if getattr(self, "metal_ws_seed_speckle_spin", None) is not None:
        self.metal_ws_seed_speckle_spin.setToolTip(
            tt(
                "Морфологическое opening затравок ядер и зазоров: выкидывает одиночные пиксели до водораздела.\n"
                "Увеличение: меньше шумных затравок, контуры спокойнее; тонкие проводники и узкие канавки могут потерять затравку и пропасть либо слиться.\n"
                "Уменьшение (0 — выкл.): сохраняет мелкие ядра и щели, но шум даёт лишние полигоны и рваные края.",
                "Morphological opening of core and gap seeds before the watershed.\n"
                "Increase: fewer noisy seeds and calmer outlines; thin traces or narrow grooves may lose their seed and vanish or merge.\n"
                "Decrease (0 = off): keeps tiny cores and slits, but noise more often creates extra polygons.",
            )
        )
    if getattr(self, "metal_ws_valley_span_spin", None) is not None:
        self.metal_ws_valley_span_spin.setToolTip(
            tt(
                "Полуширина узкого шва между близкими проводниками. Такой шов не доходит до яркости подложки, поэтому обычная затравка зазора его пропускает и трассы слипаются.\n"
                "Увеличение: разделяются и более широкие промежутки; слишком большое значение начинает резать сами проводники по внутренней текстуре.\n"
                "Уменьшение (0 — выкл.): ищутся только самые тонкие швы; близко идущие трассы чаще остаются одним полигоном.",
                "Half-width of the thin seam between closely spaced conductors. Such a seam never reaches substrate brightness, so the ordinary gap seed misses it and traces fuse.\n"
                "Increase: wider separations are split too; too large and the detector starts cutting conductors along their own texture.\n"
                "Decrease (0 = off): only the thinnest seams are found; adjacent traces more often stay one polygon.",
            )
        )
    if getattr(self, "metal_ws_valley_depth_spin", None) is not None:
        self.metal_ws_valley_depth_spin.setToolTip(
            tt(
                "Насколько шов должен быть темнее окружающего металла, чтобы считаться разделителем. Это и отличает настоящий зазор от тёмной текстуры внутри заливки.\n"
                "Увеличение: разделителями считаются только явно тёмные швы, широкие заливки целее; слабые светлые швы не разделят соседние трассы.\n"
                "Уменьшение: находятся и бледные швы; тёмная текстура внутри проводника может ошибочно разрезать полигон.",
                "How much darker than the surrounding metal a seam must be to count as a separator. This is what distinguishes a real gap from dark texture inside a fill.\n"
                "Increase: only clearly dark seams separate, pours stay whole; faint seams no longer split neighbours.\n"
                "Decrease: pale seams are found too, but texture inside a conductor may wrongly cut the polygon.",
            )
        )
    if getattr(self, "metal_rw_beta_spin", None) is not None:
        self.metal_rw_beta_spin.setToolTip(
            tt(
                "Насколько Random Walker держится перепада яркости (шкала ~32 серых уровня). Больше β — граница садится на кромку и не перетекает через слабый зазор; слишком большое значение оставляет дыры в бледной заливке.\n"
                "Уменьшение: заливка восстанавливается смелее, но близкие проводники чаще сливаются.",
                "Random Walker boundary stiffness β, scaled so a ~32 grey-level edge is a barrier at the default value.\n"
                "Increase: the cut sits on the rim and leaks less through a faint gap; too high and pale fills stay hollow.\n"
                "Decrease: fills recover more readily, but neighbouring traces merge more often.",
            )
        )
        if (label_widget := getattr(self, "metal_rw_beta_label_widget", None)) is not None:
            label_widget.setToolTip(self.metal_rw_beta_spin.toolTip())
    if getattr(self, "metal_rw_iterations_spin", None) is not None:
        self.metal_rw_iterations_spin.setToolTip(
            tt(
                "Число итераций Якоби. Больше — границы стабильнее на широких кадрах; меньше — быстрее, но маска может не дойти до равновесия.",
                "Jacobi iterations for Random Walker. More iterations stabilize wide frames; fewer run faster but may stop short of equilibrium.",
            )
        )
        if (label_widget := getattr(self, "metal_rw_iterations_label_widget", None)) is not None:
            label_widget.setToolTip(self.metal_rw_iterations_spin.toolTip())
    if getattr(self, "metal_gc_iterations_spin", None) is not None:
        self.metal_gc_iterations_spin.setToolTip(
            tt(
                "Итерации OpenCV GrabCut. Обычно хватает 3–5; больше почти не меняет контур, но замедляет предпросмотр.",
                "OpenCV GrabCut iterations. 3–5 is typical; more rarely changes the contour and only slows the preview.",
            )
        )
        if (label_widget := getattr(self, "metal_gc_iterations_label_widget", None)) is not None:
            label_widget.setToolTip(self.metal_gc_iterations_spin.toolTip())
    if getattr(self, "metal_recon_erode_spin", None) is not None:
        self.metal_recon_erode_spin.setToolTip(
            tt(
                "Сжать ядра металла перед геодезической реконструкцией. 0 оставляет ядра как есть; увеличение уводит затравку от кромки и реже перепрыгивает узкий зазор.",
                "Erode metal cores before geodesic reconstruction. 0 keeps cores as-is; increasing pulls the seed off the rim and is less likely to jump a narrow gap.",
            )
        )
        if (label_widget := getattr(self, "metal_recon_erode_label_widget", None)) is not None:
            label_widget.setToolTip(self.metal_recon_erode_spin.toolTip())
    if getattr(self, "metal_boundary_relief_spin", None) is not None:
        self.metal_boundary_relief_spin.setToolTip(
            tt(
                "На сколько серых уровней край должен выступать над локальным фоном, чтобы считаться границей проводника. Алгоритм «Замкнутые границы» ищет пару «светлый гребень со стороны металла — тёмная канавка со стороны подложки» и заливает то, что этими стенками охвачено, поэтому работает там, где заливка не отличается по яркости от подложки.\n"
                "Увеличение: остаются только явные края, посторонние стенки исчезают; слабые границы рвутся, и области перетекают друг в друга.\n"
                "Уменьшение: ловятся бледные края, но шум даёт лишние стенки и дробит крупные заливки.",
                "How many grey levels an edge must rise above the local background to count as a conductor boundary. The closed-boundary algorithm looks for the pair \"bright ridge on the metal side, dark trench on the substrate side\" and fills whatever those walls enclose, so it works where the fill has the same brightness as the substrate.\n"
                "Increase: only distinct edges remain and spurious walls disappear, but weak boundaries break and regions leak into each other.\n"
                "Decrease: faint edges are caught, but noise adds walls and shatters large pours.",
            )
        )
        if (label_widget := getattr(self, "metal_boundary_relief_label_widget", None)) is not None:
            label_widget.setToolTip(
                self.metal_boundary_relief_spin.toolTip()
            )
    if getattr(self, "metal_boundary_background_spin", None) is not None:
        self.metal_boundary_background_spin.setToolTip(
            tt(
                "Размытие, задающее локальный фон, относительно которого меряется рельеф края. Оно определяет самую широкую структуру, которую алгоритм ещё считает границей, а не заливкой.\n"
                "Увеличение: границы широких проводников распознаются увереннее, но у соседних тонких трасс края начинают сливаться.\n"
                "Уменьшение: лучше разделяются плотные тонкие трассы; у широких заливок край может распасться на два отдельных вала.",
                "The blur that defines the local background against which the edge relief is measured. It sets the widest structure the algorithm still treats as a boundary rather than as fill.\n"
                "Increase: boundaries of wide conductors are found more reliably, but edges of neighbouring thin traces start to merge.\n"
                "Decrease: dense thin traces separate better, while the edge of a wide pour may split into two separate walls.",
            )
        )
        if (label_widget := getattr(self, "metal_boundary_background_label_widget", None)) is not None:
            label_widget.setToolTip(
                self.metal_boundary_background_spin.toolTip()
            )
    if getattr(self, "metal_show_conductors_checkbox", None) is not None:
        self.metal_show_conductors_checkbox.setToolTip(
            tt("Показывать принятые полигоны проводников на сцене редактора.", "Show accepted conductor polygons.")
        )
    if getattr(self, "metal_show_rejected_checkbox", None) is not None:
        self.metal_show_rejected_checkbox.setToolTip(
            tt(
                "Красным контуром показать отклонённые компоненты (после фильтров). Полезно понять, что алгоритм отбрасывает.",
                "Draw rejected candidates in red.",
            )
        )
    if getattr(self, "metal_show_suspicious_checkbox", None) is not None:
        self.metal_show_suspicious_checkbox.setToolTip(
            tt(
                "Жёлтым — объекты, прошедшие фильтр, но с пограничными углами или прямолинейностью; проверьте вручную.",
                "Highlight borderline accepted traces in yellow.",
            )
        )
    if getattr(self, "metal_show_border_checkbox", None) is not None:
        self.metal_show_border_checkbox.setToolTip(
            tt(
                "Синим — проводники, касающиеся края кадра (часто обрезаны SEM). Не ошибка, но требует осторожности при метриках.",
                "Highlight border-touching traces in blue.",
            )
        )
    if getattr(self, "metal_show_mask_checkbox", None) is not None:
        self.metal_show_mask_checkbox.setToolTip(
            tt(
                "Включить цветное наложение поверх изображения по выбранному режиму отладки (маска, контуры, фильтр и т.д.).",
                "Enable debug / mask overlay on the image.",
            )
        )
    if getattr(self, "metal_debug_visual_combo", None) is not None:
        self.metal_debug_visual_combo.setToolTip(
            tt(
                "Что именно рисуется в оверлее: итоговая смесь, градиенты по X/Y, градиентное поле, сырая маска, контуры или этапы фильтрации.",
                "Which debug channel is shown in the overlay, including X/Y gradients and the gradient field.",
            )
        )
    if getattr(self, "metal_gradient_3d_button", None) is not None:
        self.metal_gradient_3d_button.setToolTip(
            tt(
                "Открыть отдельное окно с 3D-поверхностью модуля градиента и линиями тока по направлению Sobel.",
                "Open a separate window with a 3D magnitude surface and Sobel streamlines.",
            )
        )
    if getattr(self, "metal_overlay_opacity_spin", None) is not None:
        self.metal_overlay_opacity_spin.setToolTip(
            tt("Прозрачность оверлея отладки/маски (0.05–1.0).", "Overlay opacity.")
        )
    if getattr(self, "metal_min_area_spin", None) is not None:
        self.metal_min_area_spin.setToolTip(
            tt(
                "Минимальная площадь компонента в px² после бинаризации; отсекает мелкие засветы.\n"
                "Увеличение — меньше шумовых островков; уменьшение — спасает тонкие, но короткие фрагменты.\n"
                "Часто 40–120.",
                "Minimum area filter.",
            )
        )
    if getattr(self, "metal_max_area_spin", None) is not None:
        self.metal_max_area_spin.setToolTip(
            tt("Максимальная площадь (0 = нет лимита); режет крупные заливки.", "Maximum area, 0 = off.")
        )
    if getattr(self, "metal_min_perimeter_spin", None) is not None:
        self.metal_min_perimeter_spin.setToolTip(
            tt(
                "Минимальный периметр контура; дополнительный отсев «крошки» вокруг реальных трасс.",
                "Minimum perimeter.",
            )
        )
    if getattr(self, "metal_max_perimeter_spin", None) is not None:
        self.metal_max_perimeter_spin.setToolTip(
            tt(
                "Максимальный периметр (0 = нет); для отсечения огромных некорректных компонентов.",
                "Maximum perimeter.",
            )
        )
    if getattr(self, "metal_epsilon_spin", None) is not None:
        self.metal_epsilon_spin.setToolTip(
            tt(
                "Epsilon для Douglas–Peucker при упрощении цепочки контура перед проверками углов и топологии.\n"
                "Больше — меньше вершин, устойчивее к зубцам; меньше — точнее геометрия, но шумнее углы.",
                "Contour simplify epsilon.",
            )
        )
    if getattr(self, "metal_min_points_spin", None) is not None:
        self.metal_min_points_spin.setToolTip(
            tt("Минимальное число вершин упрощённого полигона для принятия.", "Minimum vertex count.")
        )
    if getattr(self, "metal_min_angle_spin", None) is not None:
        self.metal_min_angle_spin.setToolTip(
            tt(
                "Подавляет острые «шипы» на контуре: вершины с меньшим внутренним углом выкидываются при упрощении.",
                "Minimum interior angle at simplified vertices.",
            )
        )
