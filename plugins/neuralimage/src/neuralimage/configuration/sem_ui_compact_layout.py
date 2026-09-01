from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from neuralimage.targets.config import BASIC_TARGET_NAMES, GEOMETRY_TARGET_NAMES


RowKind = Literal['effect', 'bool_weight', 'labeled', 'inline']


@dataclass(frozen=True)
class CompactRow:
    kind: RowKind
    enable_key: str | None = None
    probability_key: str | None = None
    weight_key: str | None = None
    strength_keys: tuple[str, ...] = ()
    field_keys: tuple[str, ...] = ()


@dataclass(frozen=True)
class CompactSectionLayout:
    master_key: str | None = None
    checkable: bool = False
    rows: tuple[CompactRow, ...] = ()


def _field_row(*keys: str) -> CompactRow:
    if len(keys) == 1:
        return CompactRow('labeled', field_keys=keys)
    return CompactRow('inline', field_keys=keys)


def _target_rows(names: tuple[str, ...]) -> tuple[CompactRow, ...]:
    return tuple(
        CompactRow('bool_weight', enable_key=f'target_{name}', weight_key=f'weight_{name}')
        for name in names
    )


SEM_AUGMENTATION_EFFECT_ROWS: tuple[dict[str, object], ...] = (
    {
        'enable_key': 'aug_charging',
        'probability_key': 'aug_charging_probability',
        'strength_keys': ('aug_charging_strength',),
    },
    {
        'enable_key': 'aug_drift',
        'probability_key': 'aug_drift_probability',
        'strength_keys': ('aug_drift_pixels',),
    },
    {
        'enable_key': 'aug_focus',
        'probability_key': 'aug_focus_probability',
        'strength_keys': ('aug_focus_sigma',),
    },
    {
        'enable_key': 'aug_noise',
        'probability_key': 'aug_noise_probability',
        'strength_keys': ('aug_peak_electrons', 'aug_read_noise'),
    },
    {
        'enable_key': 'aug_gradient',
        'probability_key': 'aug_gradient_probability',
        'strength_keys': ('aug_gain_strength',),
    },
    {
        'enable_key': 'aug_defects',
        'probability_key': 'aug_defects_probability',
        'strength_keys': (),
    },
)


def _augmentation_rows() -> tuple[CompactRow, ...]:
    rows: list[CompactRow] = [_field_row('aug_plan')]
    for effect in SEM_AUGMENTATION_EFFECT_ROWS:
        rows.append(
            CompactRow(
                'effect',
                enable_key=str(effect['enable_key']),
                probability_key=str(effect['probability_key']),
                strength_keys=tuple(str(key) for key in effect.get('strength_keys', ())),
            )
        )
    return tuple(rows)


SEM_UI_COMPACT_LAYOUTS: dict[str, CompactSectionLayout] = {
    'preprocessing': CompactSectionLayout(checkable=True, rows=(_field_row('pre_mode'),)),
    'augmentation': CompactSectionLayout(master_key='aug_enabled', rows=_augmentation_rows()),
    'basic_targets': CompactSectionLayout(
        rows=_target_rows(BASIC_TARGET_NAMES)
        + (
            _field_row('target_boundary_kernel', 'target_skeleton_iterations'),
            _field_row('target_sdf_clip', 'target_distance_clip'),
            _field_row('target_thickness_max', 'target_border_ignore'),
            _field_row('target_cldice_iterations', 'target_distance_boundary_weight'),
            _field_row('target_cache', 'target_cache_size'),
        )
    ),
    'geometry_targets': CompactSectionLayout(
        rows=_target_rows(GEOMETRY_TARGET_NAMES)
        + (
            _field_row('geometry_corner_sigma', 'geometry_junction_degree'),
            _field_row('geometry_orientation_bins', 'geometry_orientation_radius'),
            _field_row('geometry_border_ignore',),
        )
    ),
    'losses': CompactSectionLayout(rows=(_field_row('loss_strategy', 'loss_mask_floor'),)),
    'hard_mining': CompactSectionLayout(
        rows=(
            _field_row('hard_mode',),
            _field_row('hard_geometry_weight', 'hard_loss_weight'),
            _field_row('hard_exploration', 'hard_ema'),
            _field_row('hard_clip', 'hard_refresh'),
            _field_row('hard_manifest',),
        )
    ),
    'context': CompactSectionLayout(
        master_key='context_enabled',
        rows=(
            _field_row('context_fusion',),
            CompactRow('labeled', field_keys=('context_attention',)),
            _field_row('context_dim', 'context_heads'),
            _field_row('context_tokens',),
        ),
    ),
    'confidence_training': CompactSectionLayout(
        master_key='confidence_training_enabled',
        rows=(_field_row('confidence_training_loss_weight',),),
    ),
    'inference_uncertainty': CompactSectionLayout(
        master_key='uncertainty_enabled',
        rows=(
            _field_row('uncertainty_method',),
            _field_row('uncertainty_samples', 'uncertainty_rate'),
            _field_row('uncertainty_tta_flips', 'uncertainty_tta_rotations'),
            CompactRow('labeled', field_keys=('uncertainty_export',)),
        ),
    ),
    'active_learning': CompactSectionLayout(
        master_key='al_enabled',
        rows=(
            _field_row('al_export_dir',),
            _field_row('al_low_confidence', 'al_entropy'),
            _field_row('al_instability', 'al_disagreement'),
            _field_row('al_max_exports', 'al_max_rois'),
            _field_row('al_min_area', 'al_padding'),
            _field_row('al_merge',),
        ),
    ),
    'validation': CompactSectionLayout(
        rows=(
            CompactRow('labeled', field_keys=('validation_full_frame',)),
            _field_row('validation_tolerance', 'validation_bins'),
            CompactRow('labeled', field_keys=('validation_hd95',)),
        ),
    ),
    'experiment': CompactSectionLayout(
        rows=(
            CompactRow('labeled', field_keys=('experiment_topology_first',)),
            _field_row('experiment_seed_1', 'experiment_seed_2', 'experiment_seed_3'),
            _field_row('experiment_manifest',),
        )
    ),
}
