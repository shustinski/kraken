from __future__ import annotations
import copy
import random
import zlib
from contextlib import contextmanager
from dataclasses import replace
import numpy as np
import torch
from neuralimage.augmentations import (
    ICDefectAugmentor,
    PCBDefectAugmentor,
    SyntheticTopologyGenerator,
    SyntheticTopologyParameters,
)
from neuralimage.lib.data_interfaces import (
    build_ic_defect_parameters,
    build_pcb_defect_parameters,
    build_synthetic_defect_generator_parameters,
)
from neuralimage.lib.images import SampleFastCutter
from neuralimage.lib.random_artifacts import generate_random_artifact_patch
from neuralimage.configuration import build_sem_segmentation_config
from neuralimage.targets.dataset_hooks import apply_dataset_preprocessing, apply_dataset_sem_augmentation_preview

MIN_AUGMENTATION_MULTIPLIER = 0.0
MAX_AUGMENTATION_MULTIPLIER = 20.0


@contextmanager
def _seeded_random(seed: int):
    random_state = random.getstate()
    np_random_state = np.random.get_state()
    torch_random_state = torch.random.get_rng_state()
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    try:
        yield
    finally:
        random.setstate(random_state)
        np.random.set_state(np_random_state)
        torch.random.set_rng_state(torch_random_state)


class PreviewEngine:
    def __init__(self, training, options, pairs, arrays, sample_index, variant, salt, cutter_item):
        self._training_parameters = training
        self.options = options
        self._sample_pairs = pairs
        self.arrays = arrays
        self._current_sample_index = sample_index
        self._variant_serial = variant
        self._resample_salt = salt
        self._cutter_item_index = cutter_item

    def _load_prepared_arrays(self, sample_index):
        return self.arrays[sample_index]

    def _current_sem_config(self):
        return build_sem_segmentation_config(self.options["get_sem_segmentation_config"])

    def _build_apply_pcb_defects_config(self):
        if not self.options["synthetic_defect_generator_check_box"]:
            config = build_pcb_defect_parameters(None)
            config.enabled = False
            return config
        synthetic_generator = build_synthetic_defect_generator_parameters(
            getattr(self._training_parameters, "synthetic_defect_generator", None)
        )
        config = copy.deepcopy(synthetic_generator.pcb_defects)
        min_defects = int(self.options["pcb_defects_min_count_spinbox"])
        max_defects = int(self.options["pcb_defects_max_count_spinbox"])
        if min_defects > max_defects:
            min_defects, max_defects = (max_defects, min_defects)
        config.enabled = self.options["pcb_defects_check_box"]
        config.defect_probability = float(self.options["pcb_defects_probability_spinbox"])
        config.min_defects = min_defects
        config.max_defects = max_defects
        for defect_name in self.options["pcb_defect_type_checkboxes"]:
            config.defect_probabilities[defect_name] = (
                1.0 if self.options["pcb_defect_type_checkboxes"][defect_name] else 0.0
            )
            config.defect_severities[defect_name] = (
                float(self.options["pcb_defect_type_spinboxes"][defect_name]) / 100.0
            )
        return config

    def _build_apply_ic_defects_config(self):
        if not self.options["synthetic_defect_generator_check_box"]:
            config = build_ic_defect_parameters(None)
            config.enabled = False
            return config
        synthetic_generator = build_synthetic_defect_generator_parameters(
            getattr(self._training_parameters, "synthetic_defect_generator", None)
        )
        config = copy.deepcopy(synthetic_generator.ic_defects)
        min_defects = int(self.options["pcb_defects_min_count_spinbox"])
        max_defects = int(self.options["pcb_defects_max_count_spinbox"])
        if min_defects > max_defects:
            min_defects, max_defects = (max_defects, min_defects)
        config.enabled = self.options["pcb_defects_check_box"]
        config.defect_probability = float(self.options["pcb_defects_probability_spinbox"])
        config.min_defects = min_defects
        config.max_defects = max_defects
        for defect_name in self.options["ic_defect_type_checkboxes"]:
            config.defect_probabilities[defect_name] = (
                1.0 if self.options["ic_defect_type_checkboxes"][defect_name] else 0.0
            )
            config.defect_severities[defect_name] = float(self.options["ic_defect_type_spinboxes"][defect_name]) / 100.0
        return config

    def _augmentation_multiplier_value(self) -> float:
        return max(
            MIN_AUGMENTATION_MULTIPLIER,
            min(MAX_AUGMENTATION_MULTIPLIER, float(self.options["augmentation_multiplier_spinbox"])),
        )

    def _generation_settings_for_cutter(self) -> object:
        generation = self._training_parameters.generation
        operations = self.options["get_training_augmentation_config"]
        return replace(
            generation,
            horizontal_rotation=False,
            vertical_rotation=False,
            flip_x=False,
            flip_y=False,
            additional_augmentation=False,
            augmentation_multiplier=0.0,
            augmentation_brightness_strength=float(self.options["augmentation_brightness_spinbox"]),
            augmentation_brightness_enabled=bool(operations["brightness"]["enabled"]),
            augmentation_brightness_probability=float(operations["brightness"]["probability"]),
            augmentation_contrast_strength=float(self.options["augmentation_contrast_spinbox"]),
            augmentation_contrast_enabled=bool(operations["contrast"]["enabled"]),
            augmentation_contrast_probability=float(operations["contrast"]["probability"]),
            augmentation_gamma_strength=float(self.options["augmentation_gamma_spinbox"]),
            augmentation_gamma_enabled=bool(operations["gamma"]["enabled"]),
            augmentation_gamma_probability=float(operations["gamma"]["probability"]),
            augmentation_noise_enabled=bool(operations["noise"]["enabled"]),
            augmentation_noise_probability=float(self.options["augmentation_noise_probability_spinbox"]),
            augmentation_noise_sigma=float(self.options["augmentation_noise_sigma_spinbox"]),
            augmentation_blur_probability=float(self.options["augmentation_blur_probability_spinbox"]),
            augmentation_blur_enabled=bool(operations["blur"]["enabled"]),
            augmentation_blur_radius=float(self.options["augmentation_blur_radius_spinbox"]),
            random_crop=bool(self.options["random_crop_check_box"]),
            crops_per_image=int(self.options["crops_per_image_spinbox"]),
            scale_augmentation=bool(self.options["scale_augmentation_check_box"]),
            scale_augmentation_probability=float(operations["scale"]["probability"]),
            scale_augmentation_strength=float(self.options["scale_augmentation_strength_spinbox"]),
        )

    def _build_preview_arrays(self, sample_index: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        synthetic = self.options["synthetic_defect_generator_check_box"]
        raw_image, raw_label = (
            self._build_synthetic_base_arrays() if synthetic else self._load_prepared_arrays(sample_index)
        )
        sem_config = self._current_sem_config()
        original_base = apply_dataset_preprocessing(raw_image, sem_config.preprocessing)
        with _seeded_random(self._seed_for(sample_index, f"sem:{self._variant_serial}")):
            augmented_base, augmented_base_label = apply_dataset_sem_augmentation_preview(
                raw_image, raw_label, sem_config.augmentation
            )
        augmented_base = apply_dataset_preprocessing(augmented_base, sem_config.preprocessing)
        full_image = self.options["full_image"]
        if full_image:
            original_image, original_label = self._build_original_full_image(original_base, raw_label)
            augmented_image, augmented_label = self._build_augmented_full_image(
                sample_index, augmented_base, augmented_base_label, include_mixup=not synthetic
            )
        else:
            original_image, original_label = self._build_original_patch(sample_index, original_base, raw_label)
            augmented_image, augmented_label = self._build_augmented_patch(
                sample_index, augmented_base, augmented_base_label, include_mixup=not synthetic
            )
        return (
            self._to_display_array(original_image),
            self._to_display_array(original_label),
            self._to_display_array(augmented_image),
            self._to_display_array(augmented_label),
        )

    def _build_synthetic_base_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        patch_height, patch_width = tuple(getattr(self._training_parameters.generation, "segment_size", (256, 256)))
        size_hw = (
            max(int(patch_height), int(self.options["synthetic_image_height_spinbox"])),
            max(int(patch_width), int(self.options["synthetic_image_width_spinbox"])),
        )
        trace_count = self._sample_preview_int_range(
            self.options["synthetic_trace_count_min_spinbox"],
            self.options["synthetic_trace_count_max_spinbox"],
            salt="synthetic_trace_count",
        )
        background_noise_sigma = self._sample_preview_float_range(
            self.options["synthetic_background_noise_sigma_min_spinbox"],
            self.options["synthetic_background_noise_sigma_max_spinbox"],
            salt="synthetic_background_noise_sigma",
        )
        trace_noise_sigma = self._sample_preview_float_range(
            self.options["synthetic_trace_noise_sigma_min_spinbox"],
            self.options["synthetic_trace_noise_sigma_max_spinbox"],
            salt="synthetic_trace_noise_sigma",
        )
        params = SyntheticTopologyParameters(
            trace_count=trace_count,
            segment_count_range=tuple(
                sorted(
                    (
                        int(self.options["synthetic_segment_count_min_spinbox"]),
                        int(self.options["synthetic_segment_count_max_spinbox"]),
                    )
                )
            ),
            trace_half_width_range=tuple(
                sorted(
                    (
                        int(self.options["synthetic_trace_half_width_min_spinbox"]),
                        int(self.options["synthetic_trace_half_width_max_spinbox"]),
                    )
                )
            ),
            topology_domain=self._get_synthetic_topology_domain(),
            topology_family=self._get_synthetic_topology_family(),
            via_count_range=(1, max(1, min(6, int(round(trace_count / 3.0))))),
            background_noise_sigma=background_noise_sigma,
            trace_noise_sigma=trace_noise_sigma,
        )
        generator = SyntheticTopologyGenerator(params)
        synthetic_channels = (
            3 if self._get_synthetic_topology_domain() == "pcb" else int(self._training_parameters.colors)
        )
        image_array, label_array = generator.generate(
            size_hw=size_hw,
            channels=synthetic_channels,
            seed=self._seed_for(self._current_sample_index, f"synthetic:{self._variant_serial}"),
        )
        return (image_array.astype(np.float32, copy=False), label_array.astype(np.float32, copy=False))

    def _build_original_patch(
        self, sample_index: int, image_matrix: np.ndarray, label_matrix: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        return self._extract_patch(sample_index, image_matrix, label_matrix, random_crop=False, scale=False)

    def _build_original_full_image(
        self, image_matrix: np.ndarray, label_matrix: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        return (image_matrix.astype(np.float32, copy=True), label_matrix.astype(np.float32, copy=True))

    def _build_augmented_patch(
        self, sample_index: int, image_matrix: np.ndarray, label_matrix: np.ndarray, *, include_mixup: bool
    ) -> tuple[np.ndarray, np.ndarray]:
        image_patch, label_patch = self._build_pre_batch_patch(sample_index, image_matrix, label_matrix)
        if include_mixup and self.options["mixup_check_box"]:
            image_patch, label_patch = self._apply_mixup(sample_index, image_patch, label_patch)
        image_patch = self._apply_cutout(sample_index, image_patch)
        image_patch = self._apply_random_artifacts(sample_index, image_patch)
        return (image_patch, label_patch)

    def _build_augmented_full_image(
        self, sample_index: int, image_matrix: np.ndarray, label_matrix: np.ndarray, *, include_mixup: bool
    ) -> tuple[np.ndarray, np.ndarray]:
        image_full, label_full = self._build_pre_batch_full_image(sample_index, image_matrix, label_matrix)
        if include_mixup and self.options["mixup_check_box"]:
            image_full, label_full = self._apply_mixup(sample_index, image_full, label_full, full_image=True)
        image_full = self._apply_cutout(sample_index, image_full)
        image_full = self._apply_random_artifacts(sample_index, image_full)
        return (image_full, label_full)

    def _build_pre_batch_patch(
        self, sample_index: int, image_matrix: np.ndarray, label_matrix: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        augmented_image = image_matrix.astype(np.float32, copy=True)
        augmented_label = label_matrix.astype(np.float32, copy=True)
        image_patch, label_patch = self._extract_patch(
            sample_index,
            augmented_image,
            augmented_label,
            random_crop=self.options["random_crop_check_box"],
            scale=self.options["scale_augmentation_check_box"],
        )
        image_patch, label_patch = self._apply_rotations(image_patch, label_patch)
        image_patch = self._apply_photometric_augmentations(sample_index, image_patch)
        image_patch, label_patch = self._apply_pcb_defects(sample_index, image_patch, label_patch)
        return (image_patch, label_patch)

    def _build_pre_batch_full_image(
        self, sample_index: int, image_matrix: np.ndarray, label_matrix: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        image_full = image_matrix.astype(np.float32, copy=True)
        label_full = label_matrix.astype(np.float32, copy=True)
        image_full, label_full = self._apply_rotations(image_full, label_full)
        image_full = self._apply_photometric_augmentations(sample_index, image_full)
        image_full, label_full = self._apply_pcb_defects(sample_index, image_full, label_full)
        return (image_full, label_full)

    def _extract_patch(
        self, sample_index: int, image_matrix: np.ndarray, label_matrix: np.ndarray, *, random_crop: bool, scale: bool
    ) -> tuple[np.ndarray, np.ndarray]:
        generation = self._generation_settings_for_cutter()
        with _seeded_random(self._seed_for(sample_index, f"extract:{int(random_crop)}:{int(scale)}")):
            cutter = SampleFastCutter((image_matrix, label_matrix), generation, shuffle=False)
        if len(cutter) <= 0:
            return (image_matrix.copy(), label_matrix.copy())
        item_index = min(max(0, int(getattr(self, "_cutter_item_index", 0))), len(cutter) - 1)
        image_patch, label_patch = cutter[item_index]
        return (np.asarray(image_patch, dtype=np.float32).copy(), np.asarray(label_patch, dtype=np.float32).copy())

    def _apply_rotations(self, image_patch: np.ndarray, label_patch: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        image = image_patch.copy()
        label = label_patch.copy()
        operations = self.options["get_training_augmentation_config"]

        def selected(key: str) -> bool:
            item = operations[key]
            return bool(item["enabled"]) and random.random() < float(item["probability"])

        if selected("rotate_90") and image.shape[1] == image.shape[2]:
            image = np.rot90(image, k=-1, axes=(1, 2)).copy()
            label = np.rot90(label, k=-1, axes=(1, 2)).copy()
        if selected("rotate_180"):
            image = image[:, ::-1, ::-1].copy()
            label = label[:, ::-1, ::-1].copy()
        if selected("flip_x"):
            image = image[:, ::-1, :].copy()
            label = label[:, ::-1, :].copy()
        if selected("flip_y"):
            image = image[:, :, ::-1].copy()
            label = label[:, :, ::-1].copy()
        return (image, label)

    def _apply_photometric_augmentations(self, sample_index: int, image_patch: np.ndarray) -> np.ndarray:
        multiplier = self._augmentation_multiplier_value()
        if multiplier > 0.0 and self._variant_serial <= 0:
            return image_patch.astype(np.float32, copy=False)
        image = image_patch.astype(np.float32, copy=True)
        with _seeded_random(self._seed_for(sample_index, "photometric")):
            operations = self.options["get_training_augmentation_config"]

            def selected(key: str) -> bool:
                item = operations[key]
                return bool(item["enabled"]) and random.random() < float(item["probability"])

            if selected("blur"):
                blur_radius = max(0.0, float(self.options["augmentation_blur_radius_spinbox"]))
                if blur_radius > 0.0:
                    image = SampleFastCutter._apply_gaussian_blur(image, blur_radius)
            if selected("brightness"):
                strength = max(0.0, float(self.options["augmentation_brightness_spinbox"]))
                brightness = 1.0 + strength
                image *= float(brightness)
            if selected("contrast"):
                strength = max(0.0, float(self.options["augmentation_contrast_spinbox"]))
                contrast = 1.0 + strength
                mean = image.mean(axis=(1, 2), keepdims=True)
                image = (image - mean) * float(contrast) + mean
            if selected("gamma"):
                gamma_strength = max(0.0, float(self.options["augmentation_gamma_spinbox"]))
                gamma = max(0.1, 1.0 - min(gamma_strength, 0.9))
                image = np.power(np.clip(image, 0.0, 1.0), float(gamma)).astype(np.float32, copy=False)
            if selected("noise"):
                sigma = max(0.0, float(self.options["augmentation_noise_sigma_spinbox"]))
                if sigma > 0.0:
                    image += np.random.normal(0.0, sigma, size=image.shape).astype(np.float32)
        np.clip(image, 0.0, 1.0, out=image)
        return image.astype(np.float32, copy=False)

    def _apply_pcb_defects(
        self, sample_index: int, image_patch: np.ndarray, label_patch: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        if not self.options["synthetic_defect_generator_check_box"]:
            return (image_patch, label_patch)
        if not self.options["pcb_defects_check_box"]:
            return (image_patch, label_patch)
        current_domain = self._get_synthetic_topology_domain()
        if current_domain == "ic":
            selected_probabilities = {
                "line_break": 1.0 if self.options["ic_defect_type_checkboxes"]["line_break"] else 0.0,
                "bridge": 1.0 if self.options["ic_defect_type_checkboxes"]["bridge"] else 0.0,
                "necking": 1.0 if self.options["ic_defect_type_checkboxes"]["necking"] else 0.0,
                "missing_metal": 1.0 if self.options["ic_defect_type_checkboxes"]["missing_metal"] else 0.0,
                "spur": 1.0 if self.options["ic_defect_type_checkboxes"]["spur"] else 0.0,
                "pinhole": 1.0 if self.options["ic_defect_type_checkboxes"]["pinhole"] else 0.0,
                "via_open": 1.0 if self.options["ic_defect_type_checkboxes"]["via_open"] else 0.0,
                "line_shift": 1.0 if self.options["ic_defect_type_checkboxes"]["line_shift"] else 0.0,
            }
            selected_severities = {
                defect_name: float(self.options["ic_defect_type_spinboxes"][defect_name]) / 100.0
                for defect_name in selected_probabilities
            }
            config = self._build_apply_ic_defects_config()
            augmentor_cls = ICDefectAugmentor
        else:
            selected_probabilities = {
                "break": 1.0 if self.options["pcb_defect_type_checkboxes"]["break"] else 0.0,
                "short": 1.0 if self.options["pcb_defect_type_checkboxes"]["short"] else 0.0,
                "missing_copper": 1.0 if self.options["pcb_defect_type_checkboxes"]["missing_copper"] else 0.0,
                "excess_copper": 1.0 if self.options["pcb_defect_type_checkboxes"]["excess_copper"] else 0.0,
                "pinhole": 1.0 if self.options["pcb_defect_type_checkboxes"]["pinhole"] else 0.0,
                "spurious_copper": 1.0 if self.options["pcb_defect_type_checkboxes"]["spurious_copper"] else 0.0,
                "via": 1.0 if self.options["pcb_defect_type_checkboxes"]["via"] else 0.0,
                "misalignment": 1.0 if self.options["pcb_defect_type_checkboxes"]["misalignment"] else 0.0,
            }
            selected_severities = {
                defect_name: float(self.options["pcb_defect_type_spinboxes"][defect_name]) / 100.0
                for defect_name in selected_probabilities
            }
            config = self._build_apply_pcb_defects_config()
            augmentor_cls = PCBDefectAugmentor
        active_count = sum((1 for probability in selected_probabilities.values() if probability > 0.0))
        if active_count <= 0:
            return (image_patch, label_patch)
        config.defect_probability = 1.0
        config.min_defects = min(active_count, max(1, int(self.options["pcb_defects_min_count_spinbox"])))
        config.max_defects = min(
            active_count, max(config.min_defects, int(self.options["pcb_defects_max_count_spinbox"]))
        )
        for defect_name in tuple(config.defect_probabilities.keys()):
            config.defect_probabilities[defect_name] = float(selected_probabilities.get(defect_name, 0.0))
            config.defect_severities[defect_name] = float(selected_severities.get(defect_name, 0.5))
        augmentor = augmentor_cls(config)
        source_image = image_patch.astype(np.float32, copy=False)
        source_label = label_patch.astype(np.float32, copy=False)
        preview_attempts = max(1, min(16, int(getattr(config, "max_attempts_per_defect", 8))))
        for attempt_index in range(preview_attempts):
            augmented_image, defect_mask, _augmented_mask = augmentor(
                source_image,
                source_label,
                seed=self._seed_for(sample_index, f"pcb_defects_{attempt_index}"),
                return_augmented_mask=True,
            )
            defect_mask_array = np.asarray(defect_mask)
            if np.count_nonzero(defect_mask_array) <= 0 and np.array_equal(augmented_image, source_image):
                continue
            return (augmented_image, source_label)
        return (source_image, source_label)

    def _apply_mixup(
        self, sample_index: int, image_patch: np.ndarray, label_patch: np.ndarray, *, full_image: bool = False
    ) -> tuple[np.ndarray, np.ndarray]:
        if len(self._sample_pairs) <= 1:
            return (image_patch, label_patch)
        alpha = max(0.0, float(self.options["mixup_alpha_spinbox"]))
        if alpha <= 0.0:
            return (image_patch, label_patch)
        partner_index = (sample_index + 1 + self._variant_serial) % len(self._sample_pairs)
        if partner_index == sample_index:
            partner_index = (partner_index + 1) % len(self._sample_pairs)
        partner_base_image, partner_base_label = self._load_prepared_arrays(partner_index)
        if full_image:
            partner_image, partner_label = self._build_pre_batch_full_image(
                partner_index, partner_base_image, partner_base_label
            )
        else:
            partner_image, partner_label = self._build_pre_batch_patch(
                partner_index, partner_base_image, partner_base_label
            )
        if partner_image.shape != image_patch.shape or partner_label.shape != label_patch.shape:
            return (image_patch, label_patch)
        with _seeded_random(self._seed_for(sample_index, "mixup")):
            lambda_value = float(np.random.beta(alpha, alpha))
        lambda_value = float(min(max(lambda_value, 0.0), 1.0))
        mixed_image = lambda_value * image_patch + (1.0 - lambda_value) * partner_image
        mixed_label = lambda_value * label_patch + (1.0 - lambda_value) * partner_label
        return (mixed_image.astype(np.float32, copy=False), mixed_label.astype(np.float32, copy=False))

    def _apply_cutout(self, sample_index: int, image_patch: np.ndarray) -> np.ndarray:
        if not self.options["cutout_check_box"]:
            return image_patch
        holes = max(1, int(self.options["cutout_holes_spinbox"]))
        size_ratio = float(self.options["cutout_size_ratio_spinbox"])
        if size_ratio <= 0.0:
            return image_patch
        image = torch.from_numpy(np.ascontiguousarray(image_patch[None, ...])).float()
        with _seeded_random(self._seed_for(sample_index, "cutout")):
            _batch, channels, height, width = image.shape
            max_cutout_height = max(1, min(int(height), int(round(int(height) * size_ratio))))
            max_cutout_width = max(1, min(int(width), int(round(int(width) * size_ratio))))
            if max_cutout_height <= 0 or max_cutout_width <= 0:
                return image_patch
            for _ in range(holes):
                cutout_height = (
                    1 if max_cutout_height == 1 else int(torch.randint(1, max_cutout_height + 1, (1,)).item())
                )
                cutout_width = 1 if max_cutout_width == 1 else int(torch.randint(1, max_cutout_width + 1, (1,)).item())
                max_top = max(0, height - cutout_height)
                max_left = max(0, width - cutout_width)
                top = 0 if max_top == 0 else int(torch.randint(0, max_top + 1, (1,)).item())
                left = 0 if max_left == 0 else int(torch.randint(0, max_left + 1, (1,)).item())
                fill_color = torch.rand((channels, 1, 1), dtype=image.dtype)
                image[0, :, top : top + cutout_height, left : left + cutout_width] = fill_color
        return image[0].numpy().astype(np.float32, copy=False)

    def _apply_random_artifacts(self, sample_index: int, image_patch: np.ndarray) -> np.ndarray:
        if not self.options["random_artifacts_check_box"]:
            return image_patch
        artifact_types = self._selected_artifact_types()
        if not artifact_types:
            return image_patch
        count = max(1, int(self.options["random_artifacts_count_spinbox"]))
        size_ratio = float(self.options["random_artifacts_size_ratio_spinbox"])
        if size_ratio <= 0.0:
            return image_patch
        image = torch.from_numpy(np.ascontiguousarray(image_patch[None, ...])).float()
        _, channels, height, width = image.shape
        min_h, max_h, min_w, max_w = self._artifact_size_bounds(height, width, size_ratio)
        with _seeded_random(self._seed_for(sample_index, "random_artifacts")):
            for _ in range(count):
                artifact_height = int(min_h if max_h == min_h else np.random.randint(min_h, max_h + 1))
                artifact_width = int(min_w if max_w == min_w else np.random.randint(min_w, max_w + 1))
                max_top = max(0, height - artifact_height)
                max_left = max(0, width - artifact_width)
                top = 0 if max_top == 0 else int(np.random.randint(0, max_top + 1))
                left = 0 if max_left == 0 else int(np.random.randint(0, max_left + 1))
                overlay, alpha = generate_random_artifact_patch(
                    int(channels),
                    int(artifact_height),
                    int(artifact_width),
                    device=torch.device("cpu"),
                    dtype=torch.float32,
                    artifact_types=artifact_types,
                    seed=int(np.random.randint(0, 2**31 - 1)),
                )
                patch = image[0, :, top : top + artifact_height, left : left + artifact_width]
                image[0, :, top : top + artifact_height, left : left + artifact_width] = torch.clamp(
                    patch * (1.0 - alpha) + overlay * alpha, min=0.0, max=1.0
                )
        return image[0].numpy().astype(np.float32, copy=False)

    def _selected_artifact_types(self) -> tuple[str, ...]:
        return tuple(
            (
                artifact_name
                for artifact_name, checkbox in self.options["random_artifact_type_checkboxes"].items()
                if checkbox
            )
        )

    def _sample_preview_int_range(self, min_widget: float, max_widget: float, *, salt: str) -> int:
        lower = int(min(min_widget, max_widget))
        upper = int(max(min_widget, max_widget))
        with _seeded_random(self._seed_for(self._current_sample_index, salt)):
            return int(np.random.randint(lower, upper + 1))

    def _sample_preview_float_range(self, min_widget: float, max_widget: float, *, salt: str) -> float:
        lower = float(min(min_widget, max_widget))
        upper = float(max(min_widget, max_widget))
        if upper <= lower:
            return lower
        with _seeded_random(self._seed_for(self._current_sample_index, salt)):
            return float(np.random.uniform(lower, upper))

    def _get_synthetic_topology_domain(self) -> str:
        return (
            str(
                self.options["synthetic_topology_domain_combo"]
                or self.options["synthetic_topology_domain_combo"]
                or "pcb"
            )
            .strip()
            .lower()
        )

    def _get_synthetic_topology_family(self) -> str:
        combo = (
            self.options["ic_topology_family_combo"]
            if self._get_synthetic_topology_domain() == "ic"
            else self.options["pcb_topology_family_combo"]
        )
        return str(combo or combo or "").strip().lower()

    @staticmethod
    def _artifact_size_bounds(image_height: int, image_width: int, size_ratio: float) -> tuple[int, int, int, int]:
        max_artifact_height = max(1, min(int(image_height), int(round(int(image_height) * float(size_ratio)))))
        max_artifact_width = max(1, min(int(image_width), int(round(int(image_width) * float(size_ratio)))))
        min_artifact_height = 1 if max_artifact_height <= 2 else max(2, int(round(max_artifact_height * 0.35)))
        min_artifact_width = 1 if max_artifact_width <= 2 else max(2, int(round(max_artifact_width * 0.35)))
        min_artifact_height = min(max_artifact_height, min_artifact_height)
        min_artifact_width = min(max_artifact_width, min_artifact_width)
        return (int(min_artifact_height), int(max_artifact_height), int(min_artifact_width), int(max_artifact_width))

    def _seed_for(self, sample_index: int, salt: str) -> int:
        if 0 <= int(sample_index) < len(self._sample_pairs):
            sample_key = self._sample_pairs[sample_index][0].as_posix()
        else:
            sample_key = "synthetic"
        payload = f"{sample_key}|{sample_index}|{getattr(self, '_cutter_item_index', 0)}|{self._variant_serial}|{self._resample_salt}|{salt}"
        return int(zlib.crc32(payload.encode("utf-8")) & 4294967295)

    @staticmethod
    def _to_display_array(image_array: np.ndarray) -> np.ndarray:
        array = np.asarray(image_array, dtype=np.float32)
        finite = array[np.isfinite(array)]
        if finite.size and (float(finite.min()) < 0.0 or float(finite.max()) > 1.0):
            low, high = np.percentile(finite, (1.0, 99.0))
            if float(high - low) > 1e-06:
                array = (array - float(low)) / float(high - low)
            else:
                array = np.zeros_like(array, dtype=np.float32)
        array = np.nan_to_num(array, nan=0.0, posinf=1.0, neginf=0.0)
        if array.ndim == 2:
            return np.clip(np.round(array * 255.0), 0.0, 255.0).astype(np.uint8)
        if array.ndim == 3 and array.shape[0] == 1:
            return np.clip(np.round(array[0] * 255.0), 0.0, 255.0).astype(np.uint8)
        if array.ndim == 3 and array.shape[0] >= 3:
            rgb = np.transpose(array[:3], (1, 2, 0))
            return np.clip(np.round(rgb * 255.0), 0.0, 255.0).astype(np.uint8)
        return np.zeros((16, 16), dtype=np.uint8)
