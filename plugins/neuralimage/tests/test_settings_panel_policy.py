import pytest

from neuralimage.lib.data_interfaces import WorkMode
from neuralimage.view.settings_panel_policy import (
    DEFAULT_PATCH_BATCH_SYNC_MODE,
    build_patch_batch_sync_plan,
    normalize_patch_batch_mode_safe,
    resolve_work_mode_applicability,
)


@pytest.mark.parametrize(
    ('mode', 'expected'),
    [
        (WorkMode.train_only.value, (True, False, True)),
        (WorkMode.train_and_recognition.value, (True, True, True)),
        (WorkMode.recognition_only.value, (False, True, False)),
        (WorkMode.further_training.value, (True, True, False)),
        (None, (True, True, True)),
        ('unknown', (True, True, True)),
    ],
)
def test_resolve_work_mode_applicability(mode, expected):
    applicability = resolve_work_mode_applicability(mode)
    assert (
        applicability.training,
        applicability.recognition,
        applicability.model_selector,
    ) == expected
    assert applicability.batch_related == (applicability.training or applicability.recognition)


def test_normalize_patch_batch_mode_safe_uses_fallback_on_exception():
    def _broken_normalizer(_value: str) -> str:
        raise RuntimeError('boom')

    assert (
        normalize_patch_batch_mode_safe(
            'invalid',
            normalizer=_broken_normalizer,
        )
        == DEFAULT_PATCH_BATCH_SYNC_MODE
    )


@pytest.mark.parametrize(
    ('normalizer_result', 'expected'),
    [
        ('patch', 'patch'),
        ('', DEFAULT_PATCH_BATCH_SYNC_MODE),
    ],
)
def test_normalize_patch_batch_mode_safe_handles_normalizer_result(normalizer_result, expected):
    assert (
        normalize_patch_batch_mode_safe(
            'anything',
            normalizer=lambda _value: normalizer_result,
        )
        == expected
    )


@pytest.mark.parametrize(
    ('raw_mode', 'expected_patch_sync', 'expected_batch_sync', 'expected_patch_targets', 'expected_batch_target'),
    [
        ('patch_and_batch', True, True, (128, 256), 16),
        ('patch', True, False, (128, 256), None),
        ('batch', False, True, (None, None), 16),
        ('off', False, False, (None, None), None),
    ],
)
def test_build_patch_batch_sync_plan(raw_mode, expected_patch_sync, expected_batch_sync, expected_patch_targets, expected_batch_target):
    plan = build_patch_batch_sync_plan(
        raw_mode,
        train_patch_x=128,
        train_patch_y=256,
        train_batch=16,
    )

    assert plan.patch_sync is expected_patch_sync
    assert plan.batch_sync is expected_batch_sync
    assert (plan.recognition_patch_x_target, plan.recognition_patch_y_target) == expected_patch_targets
    assert plan.recognition_batch_target == expected_batch_target


def test_build_patch_batch_sync_plan_uses_safe_fallback_when_normalizer_fails():
    plan = build_patch_batch_sync_plan(
        'broken',
        train_patch_x=64,
        train_patch_y=96,
        train_batch=8,
        normalizer=lambda _value: (_ for _ in ()).throw(ValueError('bad mode')),
    )

    assert plan.normalized_mode == DEFAULT_PATCH_BATCH_SYNC_MODE
    assert plan.patch_sync is True
    assert plan.batch_sync is True
    assert plan.recognition_patch_x_target == 64
    assert plan.recognition_patch_y_target == 96
    assert plan.recognition_batch_target == 8
