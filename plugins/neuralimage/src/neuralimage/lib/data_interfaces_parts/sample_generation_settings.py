"""data_interfaces: sample generation settings."""
from __future__ import annotations
from dataclasses import dataclass, field

from neuralimage.lib.data_interfaces_parts.pcb_defect_parameters import TechAugmentationParameters

@dataclass
class SampleGenerationSettings:
    step: int
    segment_size: tuple[int,int]
    vertical_rotation: bool
    horizontal_rotation: bool
    channels: int
    flip_x: bool = False
    flip_y: bool = False
    horizontal_rotation_probability: float = 1.0
    vertical_rotation_probability: float = 1.0
    flip_x_probability: float = 1.0
    flip_y_probability: float = 1.0
    additional_augmentation: bool = False
    augmentation_multiplier: float = 0.0
    augmentation_brightness_strength: float = 0.1
    augmentation_brightness_enabled: bool = True
    augmentation_brightness_probability: float = 1.0
    augmentation_contrast_strength: float = 0.1
    augmentation_contrast_enabled: bool = True
    augmentation_contrast_probability: float = 1.0
    augmentation_gamma_strength: float = 0.15
    augmentation_gamma_enabled: bool = True
    augmentation_gamma_probability: float = 1.0
    augmentation_noise_enabled: bool = True
    augmentation_noise_probability: float = 0.5
    augmentation_noise_sigma: float = 0.01
    augmentation_blur_probability: float = 0.25
    augmentation_blur_enabled: bool = True
    augmentation_blur_radius: float = 1.0
    shuffle_patches_in_frame: bool = True
    random_crop: bool = False
    crops_per_image: int = 64
    scale_augmentation: bool = False
    scale_augmentation_probability: float = 1.0
    scale_augmentation_strength: float = 0.2
    recursive_file_search: bool = False
    tech_aug: TechAugmentationParameters = field(default_factory=TechAugmentationParameters)

@dataclass
class SamplePrepareSettings:
    enable_crop: bool = False
    enable_resize: bool = False
    edge_cut:tuple[int,int]|None = None
    target_size:tuple[int,int]|None = None
    compression_factor: int = 1
