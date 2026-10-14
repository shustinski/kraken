"""data_interfaces: recognition parameters."""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from neuralimage.active_learning.config import ActiveLearningConfig
from neuralimage.preprocessing.config import PreprocessingConfig
from neuralimage.uncertainty.config import UncertaintyConfig

from neuralimage.lib.data_interfaces_parts.random_patch_size_parameters import (
    ConfidenceSaveMode,
    _default_active_learning_config,
    _default_preprocessing_config,
    _default_uncertainty_config,
)

@dataclass
class RecognitionParameters:
    source_files: list[Path]
    result_folder: Path
    model: str | Path | Any
    part_size: tuple[int,int]
    batch_size: int
    overlap: int
    source_folder: Path | None = None
    jpeg_quality: int = 95
    recognition_multiprocessing_enabled: bool = True
    binarize_output: bool = True
    lossless_binary_png: bool = False
    use_auto_threshold: bool = True
    threshold: float = 0.5
    postprocess_enabled: bool = False
    postprocess_kernel_size: int = 3
    recognition_tta_enabled: bool = False
    confidence_tta_enabled: bool = False
    confidence_save_mode: str = ConfidenceSaveMode.off.value
    resume_manifest_path: Path | None = None
    resume_from_manifest: bool = False
    use_context_branch: bool | None = None
    use_cross_attention: bool | None = None
    context_crop_size: tuple[int, int] | None = None
    context_input_size: tuple[int, int] | None = None
    recursive_file_search: bool = False
    compression_factor: int = 1
    preprocessing: PreprocessingConfig = field(default_factory=_default_preprocessing_config)
    uncertainty: UncertaintyConfig = field(default_factory=_default_uncertainty_config)
    active_learning: ActiveLearningConfig = field(default_factory=_default_active_learning_config)
    sem_config_hash: str | None = None

@dataclass
class CutSettings:
    vertical_rotation: bool
    horizontal_rotation: bool
    step: int
    color_mode:str
    x_size: int
    y_size: int
    model: str
    flip_x: bool = False
    flip_y: bool = False
    additional_augmentation: bool = False
    augmentation_gamma_strength: float = 0.15
    augmentation_blur_probability: float = 0.25
    augmentation_blur_radius: float = 1.0
    random_crop: bool = False
    crops_per_image: int = 64
    scale_augmentation: bool = False
    scale_augmentation_strength: float = 0.2
    recursive_file_search: bool = False

@dataclass
class NeuralThreadConfig:
    source_folder: str
    result_folder: str
    ready_model: bool
    model_path: str
    model: Any #temporal while not
    sample_image: str
    sample_label: str
    epochs: int
    train_params: CutSettings
