"""_apply_bright_via_tooltips section 2."""
from __future__ import annotations

def _apply_bright_via_tooltips_section_2(self, tt):
    via_help: list[tuple[str, str, str]] = [
        (
            "heuristic_background_sigma_spin",
            "Размер размытия для оценки фона перед поиском локальных пиков.\n"
            "Увеличение убирает крупный фон и помогает на плавной засветке, но может ослабить близкие контакты.\n"
            "Уменьшение делает поиск локальнее и быстрее реагирует на мелкие перепады, но чаще принимает шум.",
            "Background blur sigma. Higher removes broad illumination; lower is more local and noisier.",
        ),
        (
            "heuristic_analysis_window_scale_spin",
            "Размер окна анализа вокруг найденного пика в долях диаметра контакта.\n"
            "Увеличение даёт больше контекста для формы и кольца, но медленнее и может захватить соседние дорожки.\n"
            "Уменьшение ускоряет проверку и лучше для плотных контактов, но хуже оценивает окружение.",
            "Analysis window in contact diameters. Higher = more context/slower; lower = faster/tighter.",
        ),
        (
            "heuristic_min_center_brightness_spin",
            "Минимальная допустимая яркость пикселя в центре контакта по шкале от 0 до 255.\n"
            "Кандидат с яркостью центра ниже порога отклоняется. Увеличение оставляет только более светлые центры, "
            "уменьшение допускает более тёмные. Значение 0 отключает этот фильтр.",
            "Minimum allowed center-pixel brightness from 0 to 255. Candidates below it are rejected; zero disables the filter.",
        ),
        (
            "heuristic_min_center_contrast_spin",
            "Минимальная разница яркости центра и окружения.\n"
            "Увеличение уменьшает ложные срабатывания на слабой текстуре, но пропускает тусклые контакты.\n"
            "Уменьшение повышает полноту, но добавляет шумовые кандидаты.",
            "Minimum center-vs-surround contrast. Higher = cleaner; lower = more recall.",
        ),
        (
            "heuristic_min_peak_prominence_spin",
            "Насколько пик должен выделяться внутри локального окна.\n"
            "Увеличение отсекает плоские пятна и шум, но может потерять размытые контакты.\n"
            "Уменьшение принимает слабые пики и увеличивает число проверяемых кандидатов.",
            "Minimum local peak prominence. Higher = fewer candidates; lower = more recall/slower.",
        ),
        (
            "heuristic_min_compactness_spin",
            "Минимальная компактность локального компонента.\n"
            "Увеличение строже требует круглую/плотную форму контакта и режет вытянутые артефакты.\n"
            "Уменьшение допускает деформированные контакты, но чаще пропускает куски дорожек.",
            "Minimum component compactness. Higher = rounder; lower = more tolerant.",
        ),
        (
            "heuristic_min_circularity_spin",
            "Минимальная округлость контура от 0 до 1: 1 соответствует идеальной окружности, 0 — сильно неровной или вытянутой форме.\n"
            "Кандидат со значением ниже порога отклоняется. Увеличение оставляет только более круглые контакты, уменьшение допускает повреждённые и размытые края.\n"
            "Значение 0 отключает обязательную фильтрацию, но округлость продолжает участвовать в итоговой оценке через параметр «Вес округлости».",
            "Minimum contour circularity from 0 to 1. Values below it are rejected; 0 disables the hard filter.",
        ),
        (
            "heuristic_max_elongation_spin",
            "Максимальная вытянутость компонента.\n"
            "Уменьшение сильнее отбрасывает линии и края дорожек.\n"
            "Увеличение допускает вытянутые/размытые контакты, но растит ложные на трассах.",
            "Maximum elongation. Lower rejects lines; higher tolerates stretched candidates.",
        ),
        (
            "heuristic_line_penalty_spin",
            "Штраф за похожесть на линию.\n"
            "Увеличение сильнее снижает балл кандидатов на дорожках.\n"
            "Уменьшение помогает контактам, слитым с проводником, но добавляет ложные вдоль линий.",
            "Line penalty. Higher suppresses trace-like detections; lower is more permissive.",
        ),
        (
            "heuristic_border_penalty_spin",
            "Штраф для кандидатов у края окна/кадра.\n"
            "Увеличение убирает неполные объекты у границ.\n"
            "Уменьшение сохраняет контакты около края, но может принять обрезанные артефакты.",
            "Border penalty. Higher rejects edge candidates; lower keeps edge contacts.",
        ),
        (
            "heuristic_local_binarize_percentile_spin",
            "Процентиль локальной бинаризации компонента.\n"
            "Увеличение делает компонент меньше и строже по яркости.\n"
            "Уменьшение расширяет компонент и помогает слабым контактам, но может слить его с дорожкой.",
            "Local binarization percentile. Higher = stricter/smaller; lower = larger/more tolerant.",
        ),
        (
            "heuristic_min_abs_peak_spin",
            "Абсолютный минимум отклика пика до детальной проверки.\n"
            "Увеличение ускоряет поиск на шумных кадрах, потому что проверяется меньше seed-пиков.\n"
            "Уменьшение ищет слабые контакты, но может сильно замедлить обработку.",
            "Absolute seed floor. Higher is faster/stricter; lower finds weak contacts but can be slower.",
        ),
        (
            "heuristic_use_bilateral_checkbox",
            "Билатеральное шумоподавление вместо медианного.\n"
            "Включение лучше сохраняет края, но обычно медленнее.\n"
            "Выключение быстрее и достаточно для большинства SEM-кадров.",
            "Use bilateral denoise. Preserves edges but is slower than median filtering.",
        ),
        (
            "via_template_min_score_spin",
            "Минимальная корреляция с шаблоном.\n"
            "Увеличение уменьшает ложные совпадения, но пропускает отличающиеся контакты.\n"
            "Уменьшение повышает полноту и число кандидатов.",
            "Template correlation threshold. Higher = cleaner; lower = more matches.",
        ),
        (
            "via_template_nms_distance_spin",
            "Расстояние подавления дублей для шаблонного поиска.\n"
            "Увеличение сильнее сливает близкие совпадения.\n"
            "Уменьшение сохраняет соседние контакты, но может давать дубли одного контакта.",
            "Template NMS distance. Higher merges duplicates; lower keeps close matches.",
        ),
        (
            "via_template_scale_min_spin",
            "Минимальный масштаб шаблона.\n"
            "Уменьшение позволяет находить контакты меньше сохранённого шаблона, но добавляет лишние масштабы и замедляет поиск.\n"
            "Увеличение сужает поиск и ускоряет, но может пропустить маленькие контакты.",
            "Minimum template scale. Lower finds smaller contacts but is slower.",
        ),
        (
            "via_template_scale_max_spin",
            "Максимальный масштаб шаблона.\n"
            "Увеличение позволяет находить контакты крупнее шаблона, но добавляет вычисления и ложные совпадения.\n"
            "Уменьшение ускоряет и делает поиск строже, но может пропустить крупные контакты.",
            "Maximum template scale. Higher finds larger contacts but is slower/noisier.",
        ),
        (
            "via_template_scale_step_spin",
            "Шаг перебора масштаба шаблона.\n"
            "Увеличение ускоряет поиск, но может промахнуться по размеру.\n"
            "Уменьшение точнее, но заметно медленнее.",
            "Template scale step. Higher is faster; lower is more accurate but slower.",
        ),
    ]
    via_help.extend(
        [
            ("heuristic_size_tolerance_range_spin", "Допустимое относительное отличие найденного диаметра от проверяемого при поиске в диапазоне. 0,36 означает 36 %. Большее значение сохраняет сильнее искажённые контакты, но чаще принимает объекты неверного размера; меньшее — строже фильтрует размер.", "Allowed relative diameter error in range mode. Higher is more tolerant; lower is stricter."),
            ("heuristic_size_tolerance_fixed_spin", "Допустимое относительное отличие найденного диаметра от заданного фиксированного размера. 0,26 означает 26 %. Кандидат принимается, пока ошибка не превышает порог. Увеличение добавляет допуск, уменьшение требует точного размера.", "Allowed relative diameter error in fixed mode. Candidates above the threshold are rejected."),
            ("heuristic_max_center_drift_ratio_spin", "Максимальное смещение уточнённого центра от исходного пика, в долях диаметра. Смещение выше порога отклоняет кандидата. Большее значение терпимее к асимметрии, меньшее лучше отсекает соседние компоненты.", "Maximum center drift as a diameter fraction. Candidates above it are rejected."),
            ("heuristic_max_line_coherence_spin", "Максимальная направленность градиентов от 0 до 1: 0 — направления распределены равномерно, 1 — почти одна прямая линия. Значение выше порога отклоняется. Уменьшение сильнее подавляет дорожки, увеличение сохраняет контакты на линиях.", "Maximum gradient direction coherence, 0..1. Values above it are rejected."),
            ("heuristic_min_edge_sharpness_spin", "Минимальная резкость границы контакта относительно его контраста. Значение ниже порога отклоняется как размытое пятно. Увеличение требует резкой кромки, уменьшение помогает размытым контактам.", "Minimum edge sharpness. Values below it are rejected as diffuse spots."),
            ("heuristic_contrast_score_min_spin", "Контраст, соответствующий нулевому вкладу контраста в оценку, в уровнях яркости 0–255. Вместе с верхней границей задаёт шкалу. Увеличение делает слабые контакты менее значимыми.", "Contrast mapped to zero score contribution; must not exceed the upper bound."),
            ("heuristic_contrast_score_max_spin", "Контраст, после которого вклад контраста считается максимальным, в уровнях яркости 0–255. Должен быть не меньше нижней границы. Уменьшение быстрее насыщает оценку, увеличение лучше различает очень контрастные контакты.", "Contrast mapped to full score contribution; must be at least the lower bound."),
            ("heuristic_prominence_score_min_spin", "Минимальная выраженность локального пика для начала роста её вклада в оценку, в уровнях яркости. Это отличие пика от типичного значения в окне. Увеличение снижает балл слабых пиков.", "Local peak prominence mapped to zero contribution; must not exceed the upper bound."),
            ("heuristic_prominence_score_max_spin", "Выраженность локального пика, дающая максимальный вклад в оценку, в уровнях яркости. Должна быть не меньше нижней границы. Уменьшение быстрее насыщает вклад.", "Local peak prominence mapped to full contribution; must be at least the lower bound."),
            ("heuristic_edge_snr_score_min_spin", "Нижняя граница отношения силы края к фоновому шуму. На ней качество края даёт только минимальный вклад. Увеличение сильнее наказывает шумные и слабые границы.", "Lower edge-to-noise ratio used for score normalization."),
            ("heuristic_edge_snr_score_max_spin", "Верхняя граница отношения силы края к фоновому шуму, после которой качество края считается максимальным. Должна быть не меньше нижней границы.", "Upper edge-to-noise ratio used for score normalization."),
            ("heuristic_edge_quality_floor_spin", "Минимальная доля от 0 до 1, сохраняемая у вклада выраженности пика даже при шумном крае. Увеличение меньше учитывает качество края, уменьшение сильнее штрафует шум.", "Minimum retained peak contribution when edge quality is poor, 0..1."),
            ("heuristic_border_balance_scale_spin", "Чувствительность оценки к разнице яркости слева/справа и сверху/снизу. Увеличение сильнее снижает балл асимметричных кандидатов; уменьшение терпимее к неравномерному фону.", "Sensitivity to left/right and top/bottom intensity imbalance."),
            ("heuristic_seed_percentile_spin", "Порог отбора локальных пиков как процентиль карты отклика, 0–100 %. Принимаются пики не ниже выбранного процентиля. Увеличение оставляет меньше сильных кандидатов и ускоряет поиск, но повышает риск пропусков; уменьшение добавляет слабые кандидаты, ложные срабатывания и увеличивает время обработки.", "Response-map percentile for candidate peaks, 0..100%. Higher is stricter and faster; lower finds more weak candidates and costs more time."),
            ("heuristic_w_contrast_spin", "Вес контраста в итоговой оценке 0–100. Большее значение усиливает влияние отличия центра от окружения; меньшее делает контраст менее важным.", "Final-score contrast weight. Higher increases its influence."),
            ("heuristic_w_prominence_spin", "Вес выраженности локального пика в итоговой оценке 0–100. Большее значение усиливает влияние отчётливых пиков.", "Final-score peak prominence weight. Higher increases its influence."),
            ("heuristic_w_size_spin", "Вес соответствия найденного диаметра ожидаемому в итоговой оценке 0–100. Большее значение сильнее предпочитает заданный размер.", "Final-score expected-size weight. Higher increases its influence."),
            ("heuristic_w_compact_spin", "Вес плотности и заполненности формы в итоговой оценке 0–100. Большее значение усиливает влияние компактности.", "Final-score compactness weight. Higher increases its influence."),
            ("heuristic_w_round_spin", "Вес округлости контура в итоговой оценке 0–100. Большее значение сильнее предпочитает круглые контакты.", "Final-score roundness weight. Higher increases its influence."),
            ("heuristic_w_balance_spin", "Вес симметрии яркости вокруг центра в итоговой оценке 0–100. Большее значение усиливает влияние баланса.", "Final-score intensity-balance weight. Higher increases its influence."),
            ("heuristic_w_line_spin", "Вес вычитаемого штрафа за похожесть на прямую линию, 0–100. Большее значение сильнее подавляет кандидаты на дорожках; 0 отключает этот штраф.", "Line-penalty weight. Higher subtracts more; zero disables it."),
            ("heuristic_w_border_spin", "Вес вычитаемого штрафа за дисбаланс границы, 0–100. Большее значение сильнее снижает оценку неполных и асимметричных кандидатов; 0 отключает штраф.", "Border-imbalance penalty weight. Higher subtracts more; zero disables it."),
        ]
    )
    for attr, ru_text, en_text in via_help:
        widget = getattr(self, attr, None)
        if widget is not None:
            tooltip = tt(ru_text, en_text)
            widget.setToolTip(tooltip)
            for form_name in ("bright_via_form", "bright_via_basics_form", "bright_via_quality_form"):
                form = getattr(self, form_name, None)
                if form is None:
                    continue
                label = form.labelForField(widget)
                if label is not None:
                    label.setToolTip(tooltip)
    if getattr(self, "metal_preset_combo", None) is not None:
        self.metal_preset_combo.setToolTip(
            tt(
                "Пресет распознавания проводников.\n"
                "Встроен только «Стандартный»; остальные пресеты — пользовательские "
                "(сохранение, выгрузка и загрузка из JSON).",
                "Conductor recognition preset.\n"
                "Only «Standard» is built-in; other presets are user-created "
                "(save, export, and import from JSON).",
            )
        )
    if getattr(self, "metal_noise_suppression_slider", None) is not None:
        self.metal_noise_suppression_slider.setToolTip(
            tt(
                "Подавление зернистости SEM: выравнивание освещения и шумоподавление перед сегментацией.\n"
                "Выше — меньше шума в маске, но возможна потеря тонких деталей.",
                "SEM noise suppression before segmentation.",
            )
        )
    if getattr(self, "metal_min_contrast_slider", None) is not None:
        self.metal_min_contrast_slider.setToolTip(
            tt(
                "Минимальная разница яркости проводника и фона (1–255). "
                "Распознаются области с таким или большим контрастом. При уменьшении порога "
                "соседние фрагменты могут объединиться, поэтому площадь маски растёт, а число контуров иногда уменьшается.",
                "Minimum conductor-to-background brightness difference (1–255). "
                "Areas at or above the threshold are recognized. Lowering the threshold can merge adjacent fragments, "
                "so mask area grows even though the contour count can decrease.",
            )
        )
    if getattr(self, "metal_min_object_source_contrast_spin", None) is not None:
        self.metal_min_object_source_contrast_spin.setToolTip(
            tt(
                "Минимальная разница медианной яркости объекта и локального фона. "
                "Повышайте, если распознаются тёмные тени рядом с проводниками.",
                "Minimum median brightness difference between an object and its local background. "
                "Increase it when dark shadows beside conductors are detected.",
            )
        )
    if getattr(self, "metal_min_object_rim_contrast_spin", None) is not None:
        self.metal_min_object_rim_contrast_spin.setToolTip(
            tt(
                "\u041c\u0438\u043d\u0438\u043c\u0430\u043b\u044c\u043d\u044b\u0439 \u043a\u043e\u043d\u0442\u0440\u0430\u0441\u0442 \u044f\u0440\u043a\u043e\u0439 \u043a\u0440\u043e\u043c\u043a\u0438, \u043f\u043e\u0434\u0442\u0432\u0435\u0440\u0436\u0434\u0430\u044e\u0449\u0435\u0439 \u0441\u043b\u0430\u0431\u043e\u043a\u043e\u043d\u0442\u0440\u0430\u0441\u0442\u043d\u044b\u0439 \u043f\u0440\u043e\u0432\u043e\u0434\u043d\u0438\u043a.",
                "Minimum bright-rim contrast used to confirm a low-contrast conductor.",
            )
        )
    if getattr(self, "metal_min_object_rim_area_fraction_spin", None) is not None:
        self.metal_min_object_rim_area_fraction_spin.setToolTip(
            tt(
                "\u041c\u0438\u043d\u0438\u043c\u0430\u043b\u044c\u043d\u0430\u044f \u0434\u043e\u043b\u044f \u043a\u0430\u0434\u0440\u0430, \u043f\u0440\u0438 \u043a\u043e\u0442\u043e\u0440\u043e\u0439 \u043e\u0431\u044a\u0435\u043a\u0442 \u0441\u0447\u0438\u0442\u0430\u0435\u0442\u0441\u044f \u0434\u043e\u0441\u0442\u0430\u0442\u043e\u0447\u043d\u043e \u043a\u0440\u0443\u043f\u043d\u044b\u043c \u0434\u043b\u044f \u043a\u0440\u043e\u043c\u043e\u0447\u043d\u043e\u0439 \u043f\u0440\u043e\u0432\u0435\u0440\u043a\u0438.",
                "Minimum image-area fraction for rim-backed object filtering.",
            )
        )
    if getattr(self, "metal_min_hole_source_contrast_spin", None) is not None:
        self.metal_min_hole_source_contrast_spin.setToolTip(
            tt(
                "Минимальная разница яркости между проводником и внутренним отверстием.",
                "Minimum brightness difference between a conductor and an inner hole.",
            )
        )
    if getattr(self, "metal_min_hole_source_contrast_fraction_spin", None) is not None:
        self.metal_min_hole_source_contrast_fraction_spin.setToolTip(
            tt(
                "Минимальный контраст отверстия как доля общего разделения светлого и тёмного классов.",
                "Minimum hole contrast as a fraction of the source class separation.",
            )
        )
    if getattr(self, "metal_segmentation_strategy_combo", None) is not None:
        self.metal_segmentation_strategy_combo.setToolTip(
            tt(
                "Алгоритм сегментации проводников.\n"
                "«Авто» сравнивает Otsu и Watershed по количеству, плотности и связности объектов.\n"
                "«Порог Otsu» — классическая бинаризация по яркости.\n"
                "«Адаптивный порог» считает локальный порог в окрестности каждого пикселя "
                "и лучше держит неравномерную засветку SEM.\n"
                "Watershed, Random Walker, Graph Cut и Reconstruction растят ядра металла "
                "до затравок в зазорах и лучше держат бледную заливку SEM с яркими кромками.\n"
                "«Замкнутые границы» вообще не опирается на яркость заливки: он замыкает края "
                "в стенки и заливает охваченные ими области. Это единственный вариант, когда "
                "проводник по яркости неотличим от подложки и виден только его контур.",
                "Conductor segmentation algorithm.\n"
                "Auto compares Otsu and Watershed using object count, density, and topology.\n"
                "Otsu threshold is classical intensity binarization.\n"
                "Adaptive threshold uses a local neighbourhood of each pixel "
                "and holds uneven SEM illumination better.\n"
                "Watershed, Random Walker, Graph Cut and Reconstruction grow metal cores "
                "until they meet gap seeds and recover pale SEM fills outlined by bright rims.\n"
                "Closed boundary ignores the fill brightness altogether: it seals edges into "
                "walls and fills whatever they enclose. It is the only option when a conductor "
                "is indistinguishable from the substrate and only its outline is visible.",
            )
        )
        if (label_widget := getattr(self, "metal_segmentation_strategy_label_widget", None)) is not None:
            label_widget.setToolTip(
                self.metal_segmentation_strategy_combo.toolTip()
            )
    if getattr(self, "metal_auto_contrast_step_spin", None) is not None:
        self.metal_auto_contrast_step_spin.setToolTip(
            tt(
                "Дополнительное увеличение минимального контраста, которое режим Auto проверяет при признаках слияния проводников. "
                "Уточнённый результат принимается только при стабильном числе объектов и лучшем согласии с Watershed. "
                "Значение 0 отключает эту проверку.",
                "Additional minimum-contrast increase tested by Auto when conductor merges are suspected. "
                "The refined result is accepted only when object count stays stable and agreement with Watershed improves. "
                "Set to 0 to disable this check.",
            )
        )
        if (label_widget := getattr(self, "metal_auto_contrast_step_label_widget", None)) is not None:
            label_widget.setToolTip(
                self.metal_auto_contrast_step_spin.toolTip()
            )
    if getattr(self, "metal_auto_source_contrast_step_spin", None) is not None:
        self.metal_auto_source_contrast_step_spin.setToolTip(
            tt(
                "Шаг повышения порога контраста объектов для дополнительной проверки локальной сегментации в режиме Auto. "
                "Результат принимается только после стабилизации числа объектов и при близкой маске. "
                "Значение 0 отключает проверку.",
                "Source-contrast increment used to validate a local-segmentation candidate in Auto mode. "
                "The result is accepted only after object count stabilizes and the mask remains close. "
                "Set to 0 to disable this check.",
            )
        )
        if (label_widget := getattr(self, "metal_auto_source_contrast_step_label_widget", None)) is not None:
            label_widget.setToolTip(
                self.metal_auto_source_contrast_step_spin.toolTip()
            )
    if getattr(self, "metal_auto_directional_gap_bridge_spin", None) is not None:
        self.metal_auto_directional_gap_bridge_spin.setToolTip(
            tt(
                "Радиус направленной сшивки горизонтальных разрывов в режиме Auto. "
                "Сшивка применяется только при горизонтальной структуре кадра и подтверждении исходной яркостью. "
                "Значение 0 отключает сшивку.",
                "Maximum directional bridge radius for horizontal gaps in Auto mode. "
                "It is applied only to horizontally structured frames and requires source-intensity evidence. "
                "Set to 0 to disable bridging.",
            )
        )
        if (label_widget := getattr(self, "metal_auto_directional_gap_bridge_label_widget", None)) is not None:
            label_widget.setToolTip(
                self.metal_auto_directional_gap_bridge_spin.toolTip()
            )
    if getattr(self, "metal_auto_directional_gap_min_source_spin", None) is not None:
        self.metal_auto_directional_gap_min_source_spin.setToolTip(
            tt(
                "Минимальная медианная яркость исходных пикселей внутри добавляемого мостика. "
                "Повышение значения уменьшает риск слияния соседних проводников.",
                "Minimum median source intensity inside an added bridge. "
                "Increase it to reduce the risk of merging neighboring conductors.",
            )
        )
        if (label_widget := getattr(self, "metal_auto_directional_gap_min_source_label_widget", None)) is not None:
            label_widget.setToolTip(
                self.metal_auto_directional_gap_min_source_spin.toolTip()
            )
    return label_widget
