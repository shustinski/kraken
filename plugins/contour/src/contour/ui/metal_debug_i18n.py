"""Bilingual labels for the metal debug visual combo."""

from __future__ import annotations

from PyQt6.QtWidgets import QComboBox

METAL_DEBUG_VISUAL_LABELS: dict[str, tuple[str, str]] = {
    "overlay": ("Итоговое наложение", "Final overlay"),
    "metal_source_gray": ("Исходное (серое)", "Source (grayscale)"),
    "metal_gradient_x": ("Градиент по X", "Gradient X"),
    "metal_gradient_y": ("Градиент по Y", "Gradient Y"),
    "metal_gradient_field": ("Градиентное поле", "Gradient field"),
    "metal_preprocessed": ("После подготовки", "After preprocessing"),
    "metal_raw_segmentation": ("Сегментация (сырая)", "Raw segmentation"),
    "metal_after_topology": ("После топологии", "After topology"),
    "metal_binary_mask": ("Бинарная маска", "Binary mask"),
    "metal_threshold_mask": ("Порог Otsu", "Otsu threshold"),
    "metal_contours_raw": ("Контуры (сырые)", "Raw contours"),
    "metal_filtered_mask": ("После фильтрации", "After filtering"),
    "metal_width_check": ("Проверка ширины", "Width check"),
    "metal_structural_denoised": ("Структура: денойз", "Structure: denoise"),
    "metal_structural_gx": ("Структура: Gx", "Structure: Gx"),
    "metal_structural_gy": ("Структура: Gy", "Structure: Gy"),
    "metal_structural_gradient_magnitude": ("Структура: |G|", "Structure: |G|"),
    "metal_structural_orientation": ("Структура: ориентация", "Structure: orientation"),
    "metal_structural_coherence": ("Структура: когерентность", "Structure: coherence"),
    "metal_structural_ridge_response": ("Структура: отклик гребня", "Structure: ridge response"),
    "metal_structural_ridge_markers": ("Структура: маркеры гребня", "Structure: ridge markers"),
    "metal_structural_ridge_fragments": ("Структура: фрагменты гребня", "Structure: ridge fragments"),
    "metal_structural_ridge_links_accepted": (
        "Структура: принятые связи гребня",
        "Structure: accepted ridge links",
    ),
    "metal_structural_ridge_links_rejected": (
        "Структура: отклонённые связи гребня",
        "Structure: rejected ridge links",
    ),
    "metal_structural_ridge_links_boundary_veto": (
        "Структура: запрет пересечения границы",
        "Structure: boundary-crossing veto",
    ),
    "metal_structural_logical_ridge": ("Структура: объединённый гребень", "Structure: logical ridge"),
    "metal_structural_wide_interior_markers": (
        "Структура: маркеры широкой области",
        "Structure: wide-interior markers",
    ),
    "metal_structural_wide_fragments": ("Структура: широкие фрагменты", "Structure: wide fragments"),
    "metal_structural_logical_wide": (
        "Структура: объединённая широкая область",
        "Structure: logical wide region",
    ),
    "metal_structural_logical_markers": (
        "Структура: объединённые маркеры",
        "Structure: logical markers",
    ),
    "metal_structural_conductor_bands": ("Структура: полосы проводников", "Structure: conductor bands"),
    "metal_structural_transverse_samples": (
        "Структура: поперечные отсчёты",
        "Structure: transverse samples",
    ),
    "metal_structural_band_groups_accepted": (
        "Структура: принятые группы полос",
        "Structure: accepted band groups",
    ),
    "metal_structural_band_groups_rejected": (
        "Структура: отклонённые группы полос",
        "Structure: rejected band groups",
    ),
    "metal_structural_foreground_markers": (
        "Структура: маркеры переднего плана",
        "Structure: foreground markers",
    ),
    "metal_structural_background_markers": ("Структура: маркеры фона", "Structure: background markers"),
    "metal_structural_boundary_cost": ("Структура: стоимость границы", "Structure: boundary cost"),
    "metal_structural_watershed_labels": ("Структура: метки водораздела", "Structure: watershed labels"),
    "metal_structural_instance_labels": ("Структура: метки экземпляров", "Structure: instance labels"),
    "metal_structural_label_boundary": ("Структура: границы меток", "Structure: label boundaries"),
    "metal_structural_final_mask": ("Структура: итоговая маска", "Structure: final mask"),
    "metal_owt_oriented_boundaries": ("OWT-UCM: ориентированные границы", "OWT-UCM: oriented boundaries"),
    "metal_owt_initial_watershed": ("OWT-UCM: исходный водораздел", "OWT-UCM: initial watershed"),
    "metal_owt_ucm": ("OWT-UCM: карта иерархии", "OWT-UCM: hierarchy map"),
    "metal_owt_selected_hierarchy": ("OWT-UCM: выбранная иерархия", "OWT-UCM: selected hierarchy"),
    "metal_msp_separator_cost": (
        "Мульти-разделитель: стоимость разделителя",
        "Multi-separator: separator cost",
    ),
    "metal_msp_selected_separators": (
        "Мульти-разделитель: выбранные разделители",
        "Multi-separator: selected separators",
    ),
    "metal_msp_regions": ("Мульти-разделитель: области", "Multi-separator: regions"),
    "metal_gasp_attractive_affinity": ("GASP: сходство притяжения", "GASP: attractive affinity"),
    "metal_gasp_repulsive_affinity": ("GASP: сходство отталкивания", "GASP: repulsive affinity"),
    "metal_gasp_final_labels": ("GASP: итоговые метки", "GASP: final labels"),
    "metal_mutex_watershed_attractive_affinity": ("MWS: сходство притяжения", "MWS: attractive affinity"),
    "metal_mutex_watershed_repulsive_affinity": ("MWS: сходство отталкивания", "MWS: repulsive affinity"),
    "metal_mutex_watershed_long_range_mutex": (
        "MWS: дальние взаимоисключения",
        "MWS: long-range mutex",
    ),
    "metal_mutex_watershed_final_labels": ("MWS: итоговые метки", "MWS: final labels"),
    "metal_multicut_attractive_affinity": (
        "Мультиразрез: сходство притяжения",
        "Multicut: attractive affinity",
    ),
    "metal_multicut_repulsive_affinity": (
        "Мультиразрез: сходство отталкивания",
        "Multicut: repulsive affinity",
    ),
    "metal_multicut_final_labels": ("Мультиразрез: итоговые метки", "Multicut: final labels"),
    "metal_lifted_multicut_lifted_relations": (
        "Расширенный мультиразрез: дальние связи",
        "Lifted multicut: long-range relations",
    ),
    "metal_lifted_multicut_final_labels": (
        "Расширенный мультиразрез: итоговые метки",
        "Lifted multicut: final labels",
    ),
    "metal_material_confidence": ("Материал: уверенность", "Material: confidence"),
    "metal_material_core_evidence": ("Материал: признак ядра", "Material: core evidence"),
    "metal_material_substrate_evidence": ("Материал: признак подложки", "Material: substrate evidence"),
}


def metal_debug_visual_label(item_data: str, language: str) -> str | None:
    pair = METAL_DEBUG_VISUAL_LABELS.get(item_data)
    if pair is None:
        return None
    return pair[0] if language == "ru" else pair[1]


def retranslate_metal_debug_visual_combo(combo: QComboBox, language: str) -> None:
    for index in range(combo.count()):
        localized = metal_debug_visual_label(str(combo.itemData(index) or ""), language)
        if localized is not None:
            combo.setItemText(index, localized)
