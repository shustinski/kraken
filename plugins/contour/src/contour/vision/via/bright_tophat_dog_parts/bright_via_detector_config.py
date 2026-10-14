"""bright_tophat_dog: bright via detector config."""
from __future__ import annotations
from dataclasses import dataclass, replace
from typing import Any

from contour.vision.via.bright_tophat_dog_parts.build_isolation_map import _odd_kernel_size

from contour.vision.via.bright_tophat_dog_parts.draw_overlay import _normalize_metal_constraint_mode

@dataclass(frozen=True, slots=True)
class BrightViaDetectorConfig:
    diameter_min: int = 6
    diameter_max: int = 8
    clahe_clip_limit: float = 2.0
    clahe_tile_grid_size: int = 8
    median_blur_kernel: int = 3
    tophat_kernel_size: int = 11
    dog_sigma_small: float = 0.8
    dog_sigma_large: float = 2.0
    threshold_percentile: float = 99.0
    mask_combine_mode: str = "OR"
    min_area_factor: float = 0.45
    max_area_factor: float = 1.8
    min_circularity: float = 0.30
    min_aspect: float = 0.45
    max_aspect: float = 2.2
    bright_center_min_score: float = 6.0
    bright_range_enabled: bool = False
    bright_range_min: float = 140.0
    bright_range_max: float = 255.0
    dark_range_enabled: bool = False
    dark_range_min: float = 0.0
    dark_range_max: float = 60.0
    metal_constraint_mode: str = "soft"
    use_metal_mask: bool = True
    metal_fraction_min: float = 0.3
    max_radial_asymmetry: float = 32.0
    max_edge_likeness: float = 35.0
    max_line_likeness: float = 65.0
    nms_distance: int = 5
    min_final_score: float = 45.0
    show_rejected_candidates: bool = True
    hard_reject_on_asymmetry: bool = False
    hard_reject_on_edge: bool = False
    hard_reject_on_line: bool = False
    min_annular_contrast: float = 6.0
    max_annular_contrast: float = 0.0
    min_isolation_score: float = 0.4
    tiled_threshold_size: int = 256
    scale_kernels_from_diameter: bool = True

    @classmethod
    def from_legacy_settings(cls, settings: Any) -> BrightViaDetectorConfig:
        defaults = cls()

        def _attr(legacy: str, field: str) -> Any:
            return getattr(settings, legacy, getattr(defaults, field))

        return cls(
            diameter_min=int(_attr("bright_via_diameter_min", "diameter_min")),
            diameter_max=int(_attr("bright_via_diameter_max", "diameter_max")),
            clahe_clip_limit=float(_attr("bright_via_clahe_clip_limit", "clahe_clip_limit")),
            clahe_tile_grid_size=int(_attr("bright_via_clahe_tile_grid_size", "clahe_tile_grid_size")),
            median_blur_kernel=int(_attr("bright_via_median_blur_kernel", "median_blur_kernel")),
            tophat_kernel_size=int(_attr("bright_via_tophat_kernel_size", "tophat_kernel_size")),
            dog_sigma_small=float(_attr("bright_via_dog_sigma_small", "dog_sigma_small")),
            dog_sigma_large=float(_attr("bright_via_dog_sigma_large", "dog_sigma_large")),
            threshold_percentile=float(_attr("bright_via_threshold_percentile", "threshold_percentile")),
            mask_combine_mode=str(_attr("bright_via_mask_combine_mode", "mask_combine_mode")),
            min_area_factor=float(_attr("bright_via_min_area_factor", "min_area_factor")),
            max_area_factor=float(_attr("bright_via_max_area_factor", "max_area_factor")),
            min_circularity=float(_attr("bright_via_min_circularity", "min_circularity")),
            min_aspect=float(_attr("bright_via_min_aspect", "min_aspect")),
            max_aspect=float(_attr("bright_via_max_aspect", "max_aspect")),
            bright_center_min_score=float(_attr("bright_via_bright_center_min_score", "bright_center_min_score")),
            bright_range_enabled=bool(_attr("via_white_range_enabled", "bright_range_enabled")),
            bright_range_min=float(_attr("via_white_range_min", "bright_range_min")),
            bright_range_max=float(_attr("via_white_range_max", "bright_range_max")),
            dark_range_enabled=bool(_attr("via_black_range_enabled", "dark_range_enabled")),
            dark_range_min=float(_attr("via_black_range_min", "dark_range_min")),
            dark_range_max=float(_attr("via_black_range_max", "dark_range_max")),
            metal_constraint_mode=_normalize_metal_constraint_mode(
                _attr(
                    "bright_via_metal_constraint_mode",
                    "metal_constraint_mode",
                )
                if hasattr(settings, "bright_via_metal_constraint_mode")
                else ("soft" if bool(_attr("bright_via_use_metal_mask", "use_metal_mask")) else "disabled")
            ),
            use_metal_mask=bool(_attr("bright_via_use_metal_mask", "use_metal_mask")),
            metal_fraction_min=float(_attr("bright_via_metal_fraction_min", "metal_fraction_min")),
            max_radial_asymmetry=float(_attr("bright_via_max_radial_asymmetry", "max_radial_asymmetry")),
            max_edge_likeness=float(_attr("bright_via_max_edge_likeness", "max_edge_likeness")),
            max_line_likeness=float(_attr("bright_via_max_line_likeness", "max_line_likeness")),
            nms_distance=int(_attr("bright_via_nms_distance", "nms_distance")),
            min_final_score=float(_attr("bright_via_min_final_score", "min_final_score")),
            show_rejected_candidates=bool(_attr("bright_via_show_rejected", "show_rejected_candidates")),
            hard_reject_on_asymmetry=bool(_attr("bright_via_hard_reject_on_asymmetry", "hard_reject_on_asymmetry")),
            hard_reject_on_edge=bool(_attr("bright_via_hard_reject_on_edge", "hard_reject_on_edge")),
            hard_reject_on_line=bool(_attr("bright_via_hard_reject_on_line", "hard_reject_on_line")),
            min_annular_contrast=float(_attr("bright_via_min_annular_contrast", "min_annular_contrast")),
            min_isolation_score=float(_attr("bright_via_min_isolation_score", "min_isolation_score")),
            max_annular_contrast=float(_attr("bright_via_max_annular_contrast", "max_annular_contrast")),
            tiled_threshold_size=int(_attr("bright_via_tiled_threshold_size", "tiled_threshold_size")),
            scale_kernels_from_diameter=bool(
                _attr("bright_via_scale_kernels_from_diameter", "scale_kernels_from_diameter")
            ),
        ).validated()

    def validated(self) -> BrightViaDetectorConfig:
        errors: list[str] = []
        if self.diameter_min <= 0 or self.diameter_max <= 0:
            errors.append("diameters must be positive")
        if self.diameter_min > self.diameter_max:
            errors.append("diameter_min must be <= diameter_max")
        if self.median_blur_kernel < 1 or self.median_blur_kernel % 2 == 0:
            errors.append("median_blur_kernel must be odd and >= 1")
        if self.tophat_kernel_size < 3 or self.tophat_kernel_size % 2 == 0:
            errors.append("tophat_kernel_size must be odd and >= 3")
        if self.clahe_tile_grid_size < 1:
            errors.append("clahe_tile_grid_size must be >= 1")
        if self.clahe_clip_limit <= 0.0:
            errors.append("clahe_clip_limit must be positive")
        if self.dog_sigma_small <= 0.0 or self.dog_sigma_large <= 0.0:
            errors.append("DoG sigmas must be positive")
        if self.dog_sigma_small >= self.dog_sigma_large:
            errors.append("dogSigmaSmall must be < dogSigmaLarge")
        if not 90.0 <= self.threshold_percentile <= 99.9:
            errors.append("thresholdPercentile must be in range 90-99.9")
        if str(self.mask_combine_mode).upper() not in {"OR", "AND"}:
            errors.append("maskCombineMode must be OR or AND")
        if _normalize_metal_constraint_mode(self.metal_constraint_mode) not in {"disabled", "soft", "strict"}:
            errors.append("metalConstraintMode must be disabled, soft, or strict")
        for name, value in (
            ("min_area_factor", self.min_area_factor),
            ("max_area_factor", self.max_area_factor),
            ("min_circularity", self.min_circularity),
            ("min_aspect", self.min_aspect),
            ("max_aspect", self.max_aspect),
            ("metal_fraction_min", self.metal_fraction_min),
            ("max_radial_asymmetry", self.max_radial_asymmetry),
            ("max_edge_likeness", self.max_edge_likeness),
            ("max_line_likeness", self.max_line_likeness),
        ):
            if float(value) < 0.0:
                errors.append(f"{name} must be non-negative")
        if not 0.0 <= float(self.min_final_score) <= 100.0:
            errors.append("minFinalScore must be 0-100")
        if self.min_area_factor <= 0.0 or self.max_area_factor <= 0.0:
            errors.append("area factors must be positive")
        if self.min_aspect <= 0.0 or self.max_aspect <= 0.0:
            errors.append("aspect thresholds must be positive")
        if self.min_aspect > self.max_aspect:
            errors.append("minAspect must be <= maxAspect")
        if self.nms_distance < 0:
            errors.append("nmsDistance must be >= 0")
        if float(self.min_annular_contrast) < 0.0:
            errors.append("min_annular_contrast must be non-negative")
        if self.tiled_threshold_size < 32:
            errors.append("tiled_threshold_size must be >= 32")
        if errors:
            raise ValueError("; ".join(errors))
        return self

    def with_diameter_scaled_kernels(self) -> BrightViaDetectorConfig:
        if not self.scale_kernels_from_diameter:
            return self
        nominal = (float(self.diameter_min) + float(self.diameter_max)) * 0.5
        tophat = _odd_kernel_size(max(7, round(nominal * 1.4)))
        dog_large = max(float(self.dog_sigma_small) + 0.1, nominal * 0.25)
        return replace(
            self,
            tophat_kernel_size=tophat,
            dog_sigma_large=dog_large,
            scale_kernels_from_diameter=False,
        ).validated()
