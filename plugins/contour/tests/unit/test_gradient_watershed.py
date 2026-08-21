from __future__ import annotations

import cv2
import numpy as np
import pytest

import contour.vision.metal_recovery.seeded_segmentation as seeded_module
from contour.vision.metal_recovery.gradient_watershed import (
    ConductorSeeds,
    GradientWatershedConfig,
    analyze_metal_presence,
    build_conductor_seeds,
    gradient_watershed_mask,
    intensity_class_limits,
    keep_rim_lined_seeds,
    narrow_valley_seeds,
)
from contour.vision.metal_recovery.seeded_segmentation import (
    _prepare_working_image,
    random_walker_from_seeds,
    seeded_segmentation_mask,
)
from contour.vision.metal_recovery.segmentation import (
    normalize_metal_segmentation_strategy,
    resolve_metal_segmentation_strategy,
)

SUBSTRATE = 40
FILL = 75
RIM = 220


def _rim_lit_trace(canvas: np.ndarray, x0: int, x1: int) -> None:
    """Draw a conductor whose centre is barely above the substrate but whose edges glow."""
    canvas[10:-10, x0:x1] = FILL
    canvas[10:-10, x0 : x0 + 3] = RIM
    canvas[10:-10, x1 - 3 : x1] = RIM
    canvas[10:13, x0:x1] = RIM
    canvas[-13:-10, x0:x1] = RIM


def _component_count(mask: np.ndarray) -> int:
    count, _labels = cv2.connectedComponents((mask > 0).astype(np.uint8), connectivity=8)
    return count - 1


def test_strategy_token_is_recognised() -> None:
    assert normalize_metal_segmentation_strategy("gradient_watershed") == "gradient_watershed"
    assert normalize_metal_segmentation_strategy("Водораздел") == "gradient_watershed"
    assert normalize_metal_segmentation_strategy("Random Walker") == "random_walker"
    assert normalize_metal_segmentation_strategy("графовый разрез") == "graph_cut"
    assert normalize_metal_segmentation_strategy("Реконструкция") == "reconstruction"
    assert normalize_metal_segmentation_strategy("Замкнутые границы") == "closed_boundary"


def test_legacy_watershed_flag_resolves_when_strategy_is_not_seeded() -> None:
    assert resolve_metal_segmentation_strategy("legacy_otsu", use_wide_conductor_gradient=True) == (
        "gradient_watershed"
    )
    assert resolve_metal_segmentation_strategy("random_walker", use_wide_conductor_gradient=True) == ("random_walker")


def test_uniform_traces_without_rims_are_recovered_beside_rim_lit() -> None:
    """Mid-grey traces with no bright outline must not vanish next to rim-lit metal."""
    image = np.full((180, 420), SUBSTRATE, np.uint8)
    for x0 in range(8, 160, 28):
        _rim_lit_trace(image, x0, x0 + 18)
    image[:, 220:234] = 82
    image[:, 260:320] = 78
    image[:, 350:400] = 90
    image = cv2.GaussianBlur(image, (0, 0), 1.0)

    mask = gradient_watershed_mask(image, GradientWatershedConfig())

    assert mask[90, 227] > 0, "a 14 px uniform trace without a rim must still be metal"
    assert mask[90, 290] > 0, "a wide uniform conductor must not be classified as substrate"
    assert mask[90, 375] > 0
    assert mask[90, 17] > 0, "rim-lit neighbours must remain"
    assert mask[90, 245] == 0, "substrate between the uniform traces must stay empty"


def test_narrow_rim_lit_trace_touching_the_border_is_kept() -> None:
    """A ~12 px rim-lit trace must not be swallowed by morphological valley seeds."""
    image = np.full((120, 200), SUBSTRATE, np.uint8)
    image[:, 20:32] = FILL
    image[:, 20:23] = RIM
    image[:, 29:32] = RIM
    _rim_lit_trace(image, 70, 160)
    image = cv2.GaussianBlur(image, (0, 0), 1.0)

    mask = gradient_watershed_mask(image, GradientWatershedConfig())

    assert mask[60, 26] > 0, "narrow rim-lit fill must remain metal, not a gap seed"
    assert mask[60, 115] > 0
    assert mask[60, 50] == 0
    assert _component_count(mask) == 2


def test_pale_centred_traces_are_filled_and_kept_apart() -> None:
    image = np.full((140, 300), SUBSTRATE, np.uint8)
    _rim_lit_trace(image, 30, 110)
    _rim_lit_trace(image, 170, 260)
    image = cv2.GaussianBlur(image, (0, 0), 1.0)

    mask = gradient_watershed_mask(image, GradientWatershedConfig())

    assert mask[70, 70] > 0, "pale conductor centre must be recovered"
    assert mask[70, 215] > 0
    assert mask[70, 140] == 0, "the substrate gap must stay empty"
    assert _component_count(mask) == 2, "neighbouring conductors must not merge into one polygon"


def test_dark_texture_inside_a_pour_does_not_split_it() -> None:
    image = np.full((200, 200), SUBSTRATE, np.uint8)
    _rim_lit_trace(image, 20, 180)
    cv2.circle(image, (100, 100), 9, SUBSTRATE, -1)
    image = cv2.GaussianBlur(image, (0, 0), 1.0)

    mask = gradient_watershed_mask(image, GradientWatershedConfig())

    assert _component_count(mask) == 1, "a dark speck inside metal must not fragment the pour"


def test_wide_dark_conductor_center_is_not_a_background_seed() -> None:
    image = np.full((180, 280), SUBSTRATE, np.uint8)
    image[15:165, 25:255] = 100
    image[15:165, 25:30] = RIM
    image[15:165, 250:255] = RIM
    image[15:20, 25:255] = RIM
    image[160:165, 25:255] = RIM
    image[42:138, 64:216] = 49
    image = cv2.GaussianBlur(image, (0, 0), 1.0)

    seeds = build_conductor_seeds(image, GradientWatershedConfig())
    mask = gradient_watershed_mask(image, GradientWatershedConfig())

    assert seeds is not None
    assert seeds.groove_seeds[90, 140] == 0
    assert mask[90, 140] > 0
    assert _component_count(mask) == 1


def test_rim_filter_drops_seeds_surrounded_by_metal() -> None:
    smoothed = np.full((80, 160), FILL, np.uint8)
    smoothed[:, 74:78] = RIM
    smoothed[:, 78:82] = SUBSTRATE
    smoothed[:, 82:86] = RIM
    cv2.circle(smoothed, (30, 40), 6, SUBSTRATE, -1)
    seeds = (smoothed <= SUBSTRATE).astype(np.uint8) * 255

    kept = keep_rim_lined_seeds(seeds, smoothed, rim_level=150.0, probe_px=6)

    assert kept[40, 80] > 0, "the gap lined with bright rims is a real seed"
    assert kept[40, 30] == 0, "a dark speck surrounded by fill is not a gap"


def test_class_limits_bracket_the_conductor_fill() -> None:
    image = np.full((100, 200), SUBSTRATE, np.uint8)
    _rim_lit_trace(image, 60, 140)
    smoothed = cv2.GaussianBlur(image, (0, 0), 1.0)

    substrate_limit, metal_limit = intensity_class_limits(smoothed)

    assert substrate_limit < FILL < metal_limit


def _pair_split_by_a_grey_seam() -> np.ndarray:
    """Two conductors parted by a seam that stays well above the substrate level."""
    image = np.full((120, 240), SUBSTRATE, np.uint8)
    image[:, 60:180] = RIM
    image[:, 116:124] = 130
    return cv2.GaussianBlur(image, (0, 0), 1.0)


def test_grey_seam_between_close_traces_separates_them() -> None:
    image = _pair_split_by_a_grey_seam()

    mask = gradient_watershed_mask(image, GradientWatershedConfig())

    assert _component_count(mask) == 2, "a seam darker than the metal must part the neighbours"


@pytest.mark.parametrize("gap_width", [1, 2, 3, 4])
def test_selective_recovery_preserves_resolved_close_conductor_gaps(gap_width: int) -> None:
    image = np.full((160, 260), SUBSTRATE, np.uint8)
    split = 130
    image[10:150, 25:split] = RIM
    image[10:150, split + gap_width : 235] = RIM
    image[10:150, split : split + gap_width] = 105

    mask = gradient_watershed_mask(image, GradientWatershedConfig())

    assert not np.any(mask[20:140, split : split + gap_width])
    assert _component_count(mask) == 2


def test_selective_recovery_does_not_bridge_a_weak_bright_gap() -> None:
    image = np.full((180, 300), SUBSTRATE, np.uint8)
    image[10:170, 20:140] = RIM
    image[10:170, 145:280] = RIM
    image[10:170, 140:145] = 125

    mask = gradient_watershed_mask(image, GradientWatershedConfig())

    assert not np.any(mask[20:160, 140:145])
    assert _component_count(mask) == 2


def test_selective_recovery_keeps_wide_dark_conductor_touching_two_borders() -> None:
    image = np.full((180, 320), SUBSTRATE, np.uint8)
    image[:, 70:250] = 95
    image[:, 70:75] = RIM
    image[:, 245:250] = RIM
    image[:, 105:215] = 48
    image = cv2.GaussianBlur(image, (0, 0), 1.0)

    mask = gradient_watershed_mask(image, GradientWatershedConfig())

    assert mask[90, 160] > 0
    assert mask[90, 40] == 0
    assert mask[90, 280] == 0


def test_seam_separation_can_be_switched_off() -> None:
    image = _pair_split_by_a_grey_seam()

    mask = gradient_watershed_mask(image, GradientWatershedConfig(valley_span_px=0))

    assert _component_count(mask) == 1, "without valley seeding the grey seam is invisible"


def test_valley_seeds_ignore_shallow_texture() -> None:
    smoothed = np.full((80, 200), RIM, np.uint8)
    smoothed[:, 90:96] = 130
    smoothed[:, 150:156] = RIM - 20

    seeds = narrow_valley_seeds(smoothed, span_px=5, depth=45.0)

    assert seeds[40, 93] > 0, "a deep seam is a separator"
    assert seeds[40, 153] == 0, "a shallow dip is surface texture, not a gap"


def test_uniform_image_yields_empty_mask_instead_of_failing() -> None:
    mask = gradient_watershed_mask(np.full((40, 40), 128, np.uint8), GradientWatershedConfig())

    assert mask.shape == (40, 40)
    assert int(np.count_nonzero(mask)) == 0


def test_texture_only_frame_is_rejected_before_watershed() -> None:
    rng = np.random.default_rng(1234)
    texture = rng.normal(128.0, 5.0, size=(320, 320)).astype(np.float32)
    texture = cv2.GaussianBlur(texture, (0, 0), 1.2)
    image = np.clip(texture, 0.0, 255.0).astype(np.uint8)

    presence = analyze_metal_presence(image)
    mask = gradient_watershed_mask(image, GradientWatershedConfig())

    assert not presence.has_metal
    assert not np.any(mask)


def test_low_area_rim_lit_trace_is_not_rejected_as_empty() -> None:
    rng = np.random.default_rng(4321)
    image = np.clip(
        rng.normal(120.0, 4.0, size=(400, 400)),
        0.0,
        255.0,
    ).astype(np.uint8)
    image[180:220, 190:210] = 175
    image[180:183, 190:210] = 235
    image[217:220, 190:210] = 235
    image[180:220, 190:193] = 235
    image[180:220, 207:210] = 235
    image = cv2.GaussianBlur(image, (0, 0), 1.0)

    presence = analyze_metal_presence(image)
    mask = gradient_watershed_mask(image, GradientWatershedConfig())

    assert presence.has_metal
    assert np.any(mask[180:220, 190:210])


@pytest.mark.parametrize("strategy", ["random_walker", "graph_cut", "reconstruction"])
def test_seeded_algorithms_fill_pale_traces_and_keep_them_apart(strategy: str) -> None:
    image = np.full((140, 300), SUBSTRATE, np.uint8)
    _rim_lit_trace(image, 30, 110)
    _rim_lit_trace(image, 170, 260)
    image = cv2.GaussianBlur(image, (0, 0), 1.0)

    mask = seeded_segmentation_mask(image, strategy, GradientWatershedConfig())

    assert mask[70, 70] > 0, f"{strategy}: pale conductor centre must be recovered"
    assert mask[70, 215] > 0
    assert mask[70, 140] == 0, f"{strategy}: the substrate gap must stay empty"
    assert _component_count(mask) == 2, f"{strategy}: neighbouring conductors must not merge"


@pytest.mark.parametrize("width_px", [1, 2, 3])
def test_downsampling_preserves_thin_full_resolution_gap_markers(width_px: int) -> None:
    image = np.full((64, 2000), 128, np.uint8)
    cores = np.full_like(image, 255)
    grooves = np.zeros_like(image)
    grooves[:, 1001 : 1001 + width_px] = 255
    cores[grooves > 0] = 0
    seeds = ConductorSeeds(
        smoothed=image,
        core_seeds=cores,
        groove_seeds=grooves,
        metal_limit=128.0,
    )

    _working, working_cores, working_grooves = _prepare_working_image(image, seeds)

    assert np.any(working_grooves), f"{width_px}px groove vanished at solver resolution"
    assert not np.any((working_cores > 0) & (working_grooves > 0))


def test_random_walker_beta_and_iterations_change_the_mask() -> None:
    image = np.full((80, 220), 40, dtype=np.uint8)
    image[:, :110] = 200
    image[:, 104:116] = np.linspace(200, 40, 12, dtype=np.uint8)
    cores = np.zeros_like(image)
    grooves = np.zeros_like(image)
    cores[:, :108] = 255
    grooves[:, 112:] = 255

    soft = random_walker_from_seeds(image, cores, grooves, beta=1.0, max_iterations=160)
    hard = random_walker_from_seeds(image, cores, grooves, beta=400.0, max_iterations=160)

    assert not np.array_equal(soft, hard), "β must move the cut when a brightness edge sits between seeds"

    wide = np.full((40, 180), 80, dtype=np.uint8)
    wide[:, :90] = 180
    left = np.zeros_like(wide)
    right = np.zeros_like(wide)
    left[:, :20] = 255
    right[:, 160:] = 255
    few = random_walker_from_seeds(wide, left, right, beta=90.0, max_iterations=8)
    many = random_walker_from_seeds(wide, left, right, beta=90.0, max_iterations=400)

    assert not np.array_equal(few, many), "iteration count must change an unfinished solve"


def test_random_walker_config_reaches_the_solver_on_partitioned_seeds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image = np.full((140, 300), SUBSTRATE, np.uint8)
    _rim_lit_trace(image, 30, 110)
    _rim_lit_trace(image, 170, 260)
    image = cv2.GaussianBlur(image, (0, 0), 1.0)

    observed_beta: list[float] = []

    def _recording_solver(
        _gray: np.ndarray,
        core_seeds: np.ndarray,
        _groove_seeds: np.ndarray,
        *,
        beta: float,
        max_iterations: int,
    ) -> np.ndarray:
        observed_beta.append(beta)
        assert max_iterations == 160
        return core_seeds

    monkeypatch.setattr(seeded_module, "random_walker_from_seeds", _recording_solver)

    seeded_segmentation_mask(image, "random_walker", GradientWatershedConfig(random_walker_beta=1.0))
    seeded_segmentation_mask(image, "random_walker", GradientWatershedConfig(random_walker_beta=400.0))

    assert observed_beta == [1.0, 400.0]
