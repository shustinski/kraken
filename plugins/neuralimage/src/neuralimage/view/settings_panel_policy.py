from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from neuralimage.lib.data_interfaces import WorkMode, normalize_patch_batch_sync_mode

PATCH_SYNC_MODES = frozenset({'patch', 'patch_and_batch'})
BATCH_SYNC_MODES = frozenset({'batch', 'patch_and_batch'})
DEFAULT_PATCH_BATCH_SYNC_MODE = 'patch_and_batch'


@dataclass(frozen=True)
class WorkModeApplicability:
    training: bool
    recognition: bool
    model_selector: bool

    @property
    def batch_related(self) -> bool:
        return self.training or self.recognition


@dataclass(frozen=True)
class PatchBatchSyncPlan:
    normalized_mode: str
    patch_sync: bool
    batch_sync: bool
    recognition_patch_x_target: int | None
    recognition_patch_y_target: int | None
    recognition_batch_target: int | None


def resolve_work_mode_applicability(work_mode: str | None) -> WorkModeApplicability:
    """Return which settings groups are applicable for a work mode."""
    mode = str(work_mode or '')
    training_modes = {
        WorkMode.train_only.value,
        WorkMode.train_and_recognition.value,
        WorkMode.further_training.value,
    }
    recognition_modes = {
        WorkMode.train_and_recognition.value,
        WorkMode.recognition_only.value,
        WorkMode.further_training.value,
    }
    model_selector_modes = {
        WorkMode.train_only.value,
        WorkMode.train_and_recognition.value,
    }
    known_modes = training_modes | recognition_modes
    if mode not in known_modes:
        return WorkModeApplicability(True, True, True)
    return WorkModeApplicability(
        training=mode in training_modes,
        recognition=mode in recognition_modes,
        model_selector=mode in model_selector_modes,
    )


def normalize_patch_batch_mode_safe(
    raw_mode: str | None,
    *,
    normalizer: Callable[[str], str] = normalize_patch_batch_sync_mode,
    fallback: str = DEFAULT_PATCH_BATCH_SYNC_MODE,
) -> str:
    """Normalize patch/batch sync mode and fall back to a safe default."""
    try:
        normalized = str(normalizer(str(raw_mode or '')))
    except Exception:
        return fallback
    return normalized or fallback


def build_patch_batch_sync_plan(
    raw_mode: str | None,
    *,
    train_patch_x: int,
    train_patch_y: int,
    train_batch: int,
    normalizer: Callable[[str], str] = normalize_patch_batch_sync_mode,
) -> PatchBatchSyncPlan:
    """Build a pure synchronization plan for recognition patch/batch fields."""
    normalized_mode = normalize_patch_batch_mode_safe(raw_mode, normalizer=normalizer)
    patch_sync = normalized_mode in PATCH_SYNC_MODES
    batch_sync = normalized_mode in BATCH_SYNC_MODES
    return PatchBatchSyncPlan(
        normalized_mode=normalized_mode,
        patch_sync=patch_sync,
        batch_sync=batch_sync,
        recognition_patch_x_target=int(train_patch_x) if patch_sync else None,
        recognition_patch_y_target=int(train_patch_y) if patch_sync else None,
        recognition_batch_target=int(train_batch) if batch_sync else None,
    )
