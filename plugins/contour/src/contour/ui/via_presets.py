"""Built-in presets for expert heuristic via-recognition parameters."""

from __future__ import annotations

from ..application.processing import ContourExtractionSettings


def _expert_defaults(**overrides: object) -> dict[str, object]:
    payload = {
        key: value
        for key, value in ContourExtractionSettings().to_dict().items()
        if key.startswith("heuristic_")
    }
    payload.update(overrides)
    return payload


def noisy_traces_via_preset_payload() -> dict[str, object]:
    return _expert_defaults(
        heuristic_min_center_contrast=55.0,
        heuristic_min_compactness=0.16,
        heuristic_min_circularity=0.20,
        heuristic_max_elongation=1.35,
        heuristic_max_line_coherence=0.68,
        heuristic_min_edge_sharpness=0.48,
        heuristic_w_line=30.0,
        heuristic_line_penalty_scale=4.0,
    )


def blurred_via_preset_payload() -> dict[str, object]:
    return _expert_defaults(
        heuristic_min_center_contrast=35.0,
        heuristic_min_peak_prominence=1.0,
        heuristic_min_compactness=0.08,
        heuristic_max_elongation=1.80,
        heuristic_min_edge_sharpness=0.22,
        heuristic_edge_quality_floor=0.35,
        heuristic_seed_percentile=86.0,
    )


def built_in_via_presets(language: str) -> dict[str, dict[str, object]]:
    standard = _expert_defaults()
    strict = _expert_defaults(
        heuristic_min_center_contrast=60.0,
        heuristic_min_peak_prominence=3.0,
        heuristic_min_compactness=0.18,
        heuristic_min_circularity=0.25,
        heuristic_max_elongation=1.35,
        heuristic_max_line_coherence=0.72,
        heuristic_min_edge_sharpness=0.50,
        heuristic_seed_percentile=96.0,
    )
    sensitive = _expert_defaults(
        heuristic_min_center_contrast=35.0,
        heuristic_min_peak_prominence=1.0,
        heuristic_min_compactness=0.08,
        heuristic_max_elongation=1.80,
        heuristic_min_edge_sharpness=0.25,
        heuristic_seed_percentile=84.0,
    )
    noisy = noisy_traces_via_preset_payload()
    blurred = blurred_via_preset_payload()
    if language == "ru":
        return {
            "Стандартный": standard,
            "Строгий": strict,
            "Чувствительный": sensitive,
            "Шумное изображение": noisy,
            "Размытые контакты": blurred,
        }
    return {
        "Standard": standard,
        "Strict": strict,
        "Sensitive": sensitive,
        "Noisy image": noisy,
        "Blurred contacts": blurred,
    }


__all__ = [
    "blurred_via_preset_payload",
    "built_in_via_presets",
    "noisy_traces_via_preset_payload",
]
