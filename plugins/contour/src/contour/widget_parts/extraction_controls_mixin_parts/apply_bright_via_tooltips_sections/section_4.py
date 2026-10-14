"""_apply_bright_via_tooltips section 4."""
from __future__ import annotations

def __apply_bright_via_tooltips_section_4(self, tt):
    if getattr(self, "metal_approximation_checkbox", None) is not None:
        self.metal_approximation_checkbox.setToolTip(
            tt(
                "Включить упрощение контура (approxPolyDP); выключите только для отладки сырой цепочки.",
                "Enable DP simplify.",
            )
        )
    if getattr(self, "metal_hierarchy_combo", None) is not None:
        self.metal_hierarchy_combo.setToolTip(
            tt(
                "Полная иерархия (RETR_TREE) учитывает вложенность контуров; только внешние — быстрее и проще, если дырки не нужны.",
                "Contour hierarchy retrieval mode.",
            )
        )
    if getattr(self, "metal_allowed_angles_combo", None) is not None:
        self.metal_allowed_angles_combo.setToolTip(
            tt(
                "Ограничение на углы трассировки после упрощения: ортогональ, 45°/90° или без ограничений.\n"
                "Жёстче режим — меньше ложных изломанных контуров, но риск отсечь слегка «кривую» реальную дорожку.",
                "Allowed routing angles.",
            )
        )
    if getattr(self, "metal_angle_tolerance_spin", None) is not None:
        self.metal_angle_tolerance_spin.setToolTip(
            tt(
                "На сколько градусов можно отклониться от идеальных 0/45/90°, чтобы угол всё ещё считался допустимым.\n"
                "Увеличьте при шумном крае; уменьшите, если просачиваются диагональные артефакты. Типично 5–10°.",
                "Angular tolerance in degrees.",
            )
        )
    if getattr(self, "metal_straightness_spin", None) is not None:
        self.metal_straightness_spin.setToolTip(
            tt(
                "Отношение «длина по minAreaRect» к периметру: низкие значения характерны для рыхлых, извилистых шумовых масок.\n"
                "Повышение отсекает пятна и ветвистый мусор; понижение спасает сложные, но реальные формы. Старт 0.55–0.7.",
                "Minimum straightness metric.",
            )
        )
    if getattr(self, "metal_t_junction_checkbox", None) is not None:
        self.metal_t_junction_checkbox.setToolTip(
            tt(
                "Разрешать T-образные соединения в растровой маске (один связный компонент с разветвлением).\n"
                "Выключение слегка ужесточает отбор по выпуклым дефектам — полезно, если шум даёт ложные «тройники» внутри одного контура.",
                "Allow T-junction topology in mask components.",
            )
        )
    if getattr(self, "metal_border_handling_combo", None) is not None:
        self.metal_border_handling_combo.setToolTip(
            tt(
                "«Игнорировать» — отбрасывать всё, что касается края кадра; «Принимать» — не отличать; "
                "«Помечать» — принять, но выделить отдельно (часто обрезанные проводники).",
                "How to treat image-border-touching components.",
            )
        )
    if getattr(self, "metal_validity_checkbox", None) is not None:
        self.metal_validity_checkbox.setToolTip(
            tt(
                "Проверка простого замкнутого контура без самопересечений и лишних самокасаний на упрощённой цепочке.\n"
                "Отключайте только временно для отладки сырой векторизации — иначе в выдачу могут попасть некорректные полигоны.",
                "Validate simplified ring geometry.",
            )
        )
    if getattr(self, "metal_morph_close_spin", None) is not None:
        self.metal_morph_close_spin.setToolTip(
            tt(
                "Радиус морфологического closing после порога: склеивает мелкие разрывы маски.\n"
                "Держите низким (2–4), иначе сливаются близкие несвязанные объекты.",
                "Closing radius; keep small.",
            )
        )
    if getattr(self, "metal_morph_open_spin", None) is not None:
        self.metal_morph_open_spin.setToolTip(
            tt("Opening для удаления тонкого соли-and-pepper шума; 0 — отключено.", "Opening radius, 0 = off.")
        )
    if getattr(self, "metal_preview_mask_button", None) is not None:
        self.metal_preview_mask_button.setToolTip(
            tt("Переключить оверлей на бинарную маску и включить показ.", "Jump to binary mask overlay.")
        )
    if getattr(self, "metal_reset_params_button", None) is not None:
        self.metal_reset_params_button.setToolTip(
            tt("Сбросить параметры восстановления к значениям по умолчанию.", "Reset metal parameters to defaults.")
        )
