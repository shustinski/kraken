"""SEM-specific image augmentations for IC layout microscopy."""

from __future__ import annotations

import random

import cv2
import numpy as np

from neuralimage.augmentations.sem_config import SemAugmentationConfig


class SemAugmentor:
    """Additional SEM-domain augmentations that complement existing tech/IC augmentors."""

    def __init__(self, config: SemAugmentationConfig | None = None):
        self.config = config or SemAugmentationConfig()

    @property
    def enabled(self) -> bool:
        return self.config.enabled

    def __call__(self, image: np.ndarray, label: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray | None]:
        return self.apply(image, label)

    def apply(self, image: np.ndarray, label: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray | None]:
        if not self.config.enabled:
            array = np.asarray(image, dtype=np.float32)
            if array.max() > 1.0:
                array = array / 255.0
            return array, label
        augmented = np.asarray(image, dtype=np.float32)
        if augmented.max() <= 1.0:
            augmented = augmented * 255.0
        augmented = augmented.astype(np.uint8)

        if self.config.charging_artifacts and random.random() < self.config.charging_probability:
            augmented = self._charging_artifacts(augmented)
        if self.config.scan_drift and random.random() < self.config.scan_drift_probability:
            augmented = self._scan_drift(augmented)
        if self.config.local_focus_variation and random.random() < self.config.focus_variation_probability:
            augmented = self._local_focus_variation(augmented)
        if self.config.detector_noise and random.random() < self.config.detector_noise_probability:
            augmented = self._detector_noise(augmented)
        if self.config.brightness_gradients and random.random() < self.config.brightness_gradient_probability:
            augmented = self._brightness_gradients(augmented)
        if self.config.realistic_defects and random.random() < self.config.realistic_defect_probability:
            augmented = self._realistic_defects(augmented)

        if augmented.max() > 1.0:
            augmented = augmented.astype(np.float32) / 255.0
        else:
            augmented = augmented.astype(np.float32)
        return augmented, label

    def _charging_artifacts(self, image: np.ndarray) -> np.ndarray:
        result = image.copy()
        height, width = result.shape[:2]
        streak_count = random.randint(1, 3)
        for _ in range(streak_count):
            x = random.randint(0, max(0, width - 1))
            thickness = random.randint(1, 4)
            brighten = random.randint(20, 80)
            current = int(np.clip(float(result.max()) + brighten, 0, 255))
            cv2.line(result, (x, 0), (x, height - 1), current, thickness)
        return result

    def _scan_drift(self, image: np.ndarray) -> np.ndarray:
        shift_y = random.randint(-3, 3)
        shift_x = random.randint(-3, 3)
        matrix = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
        return cv2.warpAffine(image, matrix, (image.shape[1], image.shape[0]), borderMode=cv2.BORDER_REFLECT)

    def _local_focus_variation(self, image: np.ndarray) -> np.ndarray:
        height, width = image.shape[:2]
        center_x = random.randint(width // 4, 3 * width // 4)
        center_y = random.randint(height // 4, 3 * height // 4)
        radius = random.randint(min(height, width) // 8, min(height, width) // 4)
        mask = np.zeros((height, width), dtype=np.float32)
        cv2.circle(mask, (center_x, center_y), radius, 1.0, -1)
        blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=random.uniform(0.8, 2.5))
        mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=radius / 3.0)[..., None]
        if image.ndim == 2:
            mask = mask[..., 0]
        combined = image.astype(np.float32) * (1.0 - mask) + blurred.astype(np.float32) * mask
        return np.clip(combined, 0, 255).astype(np.uint8)

    def _detector_noise(self, image: np.ndarray) -> np.ndarray:
        sigma = random.uniform(2.0, 12.0)
        noise = np.random.normal(0.0, sigma, size=image.shape).astype(np.float32)
        return np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    def _brightness_gradients(self, image: np.ndarray) -> np.ndarray:
        height, width = image.shape[:2]
        axis = random.choice(['horizontal', 'vertical'])
        gradient = np.linspace(0.85, 1.15, width if axis == 'horizontal' else height, dtype=np.float32)
        if axis == 'horizontal':
            gradient = gradient[None, :]
        else:
            gradient = gradient[:, None]
        return np.clip(image.astype(np.float32) * gradient, 0, 255).astype(np.uint8)

    def _realistic_defects(self, image: np.ndarray) -> np.ndarray:
        result = image.copy()
        height, width = result.shape[:2]
        for _ in range(random.randint(1, 4)):
            x = random.randint(0, width - 1)
            y = random.randint(0, height - 1)
            radius = random.randint(1, max(2, min(height, width) // 64))
            intensity = random.randint(-40, 40)
            cv2.circle(result, (x, y), radius, int(np.clip(result[y, x] + intensity, 0, 255)), -1)
        return result
