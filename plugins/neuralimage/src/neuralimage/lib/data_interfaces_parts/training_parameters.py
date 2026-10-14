"""data_interfaces: training parameters."""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from neuralimage.active_learning.config import ActiveLearningConfig
from neuralimage.augmentations.sem_config import SemAugmentationConfig
from neuralimage.preprocessing.config import PreprocessingConfig
from neuralimage.targets.config import SupervisionTargetsParameters
from neuralimage.uncertainty.config import UncertaintyConfig

from neuralimage.lib.data_interfaces_parts.ic_defect_parameters import SyntheticDefectGeneratorParameters

from neuralimage.lib.data_interfaces_parts.pcb_defect_parameters import (
    CutoutParameters,
    HardMiningParameters,
    MixupParameters,
    PCBDefectParameters,
    RandomArtifactsParameters,
)

from neuralimage.lib.data_interfaces_parts.random_patch_size_parameters import (
    EarlyStoppingParameters,
    MixedPrecisionMode,
    OptimizerParameters,
    RandomPatchSizeParameters,
    SchedulerParameters,
    ValidationSource,
    WarmupParameters,
    _default_active_learning_config,
    _default_preprocessing_config,
    _default_sem_augmentation_config,
    _default_supervision_targets,
    _default_uncertainty_config,
)

@dataclass
class TrainingParameters:
    image_path:Path
    label_path:Path
    shuffle:bool
    validation:bool
    validation_percent:int
    batch_size: int
    cut_mode:SampleCutMode
    colors: int
    epochs: int
    generation:SampleGenerationSettings
    prepare:SamplePrepareSettings
    validation_source: str = ValidationSource.split.value
    validation_image_path: Path | None = None
    validation_label_path: Path | None = None
    save_validation_binary_images: bool = False
    optimizer: OptimizerParameters = field(default_factory=OptimizerParameters)
    mixed_precision: MixedPrecisionMode = MixedPrecisionMode.off
    loss_function: str = 'bce'
    loss_term_weights: dict[str, float] = field(default_factory=dict)
    dice_loss_weight: float = 0.5
    iou_loss_weight: float = 0.5
    early_stopping: EarlyStoppingParameters = field(default_factory=EarlyStoppingParameters)
    warmup: WarmupParameters = field(default_factory=WarmupParameters)
    scheduler: SchedulerParameters = field(default_factory=SchedulerParameters)
    hard_mining: HardMiningParameters = field(default_factory=HardMiningParameters)
    random_patch_size: RandomPatchSizeParameters = field(default_factory=RandomPatchSizeParameters)
    cutout: CutoutParameters = field(default_factory=CutoutParameters)
    random_artifacts: RandomArtifactsParameters = field(default_factory=RandomArtifactsParameters)
    mixup: MixupParameters = field(default_factory=MixupParameters)
    skip_uniform_labels: bool = False
    rare_patch_oversampling_enabled: bool = False
    rare_patch_oversampling_factor: int = 2
    use_multi_gpu: bool = False
    multi_gpu_mode: str = 'off'
    show_batch_preview: bool = True
    log_update_frequency: int = 0
    local_crop_size: tuple[int, int] | None = None
    context_crop_size: tuple[int, int] | None = None
    context_input_size: tuple[int, int] | None = None
    context_branch_channels: tuple[int, ...] = (16, 32, 64, 128)
    fusion_type: str = 'concat'
    use_context_branch: bool | None = None
    use_cross_attention: bool = True
    attention_dim: int = 128
    attention_heads: int = 4
    attention_max_global_tokens: int = 1024
    deep_supervision: bool = False
    artifact_dir: Path | None = None
    resume_from_checkpoint: bool = False
    dataloader_num_workers: int = -1
    recursive_file_search: bool = False
    pcb_defects: PCBDefectParameters = field(default_factory=PCBDefectParameters)
    synthetic_defect_generator: SyntheticDefectGeneratorParameters = field(
        default_factory=SyntheticDefectGeneratorParameters
    )
    supervision_targets: SupervisionTargetsParameters = field(default_factory=_default_supervision_targets)
    preprocessing: PreprocessingConfig = field(default_factory=_default_preprocessing_config)
    sem_augmentation: SemAugmentationConfig = field(default_factory=_default_sem_augmentation_config)
    uncertainty: UncertaintyConfig = field(default_factory=_default_uncertainty_config)
    active_learning: ActiveLearningConfig = field(default_factory=_default_active_learning_config)
    advanced_validation: bool = True
    advanced_validation_full_frame: bool = True
    advanced_validation_boundary_tolerance: int = 2
    advanced_validation_include_hd95: bool = True
    advanced_validation_confidence_bins: int = 10
    loss_weighting_strategy: str = 'static'
    mask_loss_weight_floor: float = 0.25
    topograph_enabled: bool = False
    topograph_loss_weight: float = 0.1
    topograph_debug_viz: bool = False
    topograph_num_processes: int = 1
    topograph_use_c: bool = False
