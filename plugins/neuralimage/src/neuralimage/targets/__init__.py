"""Automatic supervision target generation from binary polygon masks."""

from neuralimage.targets.config import SupervisionTargetConfig, GeometrySupervisionConfig
from neuralimage.targets.registry import TargetGeneratorRegistry, generate_supervision_targets
from neuralimage.targets.batch import collate_supervision_targets, extract_mask_from_target

__all__ = [
    'SupervisionTargetConfig',
    'GeometrySupervisionConfig',
    'TargetGeneratorRegistry',
    'generate_supervision_targets',
    'collate_supervision_targets',
    'extract_mask_from_target',
]
