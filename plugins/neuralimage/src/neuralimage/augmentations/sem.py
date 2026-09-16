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

    def apply_preview(self, image: np.ndarray, label: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray | None]:
        """Apply all enabled SEM effects for UI preview (no probability gating)."""
        if not self.config.enabled:
            array = np.asarray(image, dtype=np.float32)
            if array.max() > 1.0:
                array = array / 255.0
            return array, label
        augmented = np.asarray(image, dtype=np.float32)
        if self.config.plan == 'sem_v2':
            if augmented.max(initial=0.0) > 1.0:
                augmented = augmented / (65535.0 if augmented.max() > 255.0 else 255.0)
            if self.config.charging_artifacts:
                augmented = self._charging_bloom(augmented)
            if self.config.scan_drift:
                augmented = self._row_dependent_drift(augmented)
            if self.config.local_focus_variation:
                augmented = self._continuous_focus_variation(augmented)
            if self.config.detector_noise:
                augmented = self._poisson_read_noise(augmented)
            if self.config.brightness_gradients:
                augmented = self._smooth_gain_field(augmented)
            if self.config.realistic_defects:
                augmented = self._contamination_and_scan_defects(augmented)
            return np.clip(augmented, 0.0, 1.0).astype(np.float32), label
        if augmented.max() <= 1.0:
            augmented = augmented * 255.0
        augmented = augmented.astype(np.uint8)
        if self.config.charging_artifacts:
            augmented = self._charging_artifacts(augmented)
        if self.config.scan_drift:
            augmented = self._scan_drift(augmented)
        if self.config.local_focus_variation:
            augmented = self._local_focus_variation(augmented)
        if self.config.detector_noise:
            augmented = self._detector_noise(augmented)
        if self.config.brightness_gradients:
            augmented = self._brightness_gradients(augmented)
        if self.config.realistic_defects:
            augmented = self._realistic_defects(augmented)
        if augmented.max() > 1.0:
            augmented = augmented.astype(np.float32) / 255.0
        else:
            augmented = augmented.astype(np.float32)
        return augmented, label

    def apply(self, image: np.ndarray, label: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray | None]:
        if not self.config.enabled:
            array = np.asarray(image, dtype=np.float32)
            if array.max() > 1.0:
                array = array / 255.0
            return array, label
        augmented = np.asarray(image, dtype=np.float32)
        if self.config.plan == 'sem_v2':
            if augmented.max(initial=0.0) > 1.0:
                augmented = augmented / (65535.0 if augmented.max() > 255.0 else 255.0)
            if self.config.charging_artifacts and random.random() < self.config.charging_probability:
                augmented = self._charging_bloom(augmented)
            if self.config.scan_drift and random.random() < self.config.scan_drift_probability:
                augmented = self._row_dependent_drift(augmented)
            if self.config.local_focus_variation and random.random() < self.config.focus_variation_probability:
                augmented = self._continuous_focus_variation(augmented)
            if self.config.detector_noise and random.random() < self.config.detector_noise_probability:
                augmented = self._poisson_read_noise(augmented)
            if self.config.brightness_gradients and random.random() < self.config.brightness_gradient_probability:
                augmented = self._smooth_gain_field(augmented)
            if self.config.realistic_defects and random.random() < self.config.realistic_defect_probability:
                augmented = self._contamination_and_scan_defects(augmented)
            return np.clip(augmented, 0.0, 1.0).astype(np.float32), label
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

    def _charging_bloom(self, image: np.ndarray) -> np.ndarray:
        height, width = image.shape[:2]
        yy, xx = np.mgrid[:height, :width]
        field = np.zeros((height, width), dtype=np.float32)
        for _ in range(random.randint(1, 3)):
            cx, cy = random.uniform(0, width), random.uniform(0, height)
            sigma = random.uniform(max(2.0, min(height, width) * 0.03), max(3.0, min(height, width) * 0.15))
            field += np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2.0 * sigma * sigma)).astype(np.float32)
        field = np.clip(field, 0.0, 1.0)
        strength = max(0.0, float(self.config.charging_strength))
        return np.clip(image + strength * field * (1.0 - image), 0.0, 1.0)

    def _row_dependent_drift(self, image: np.ndarray) -> np.ndarray:
        height, width = image.shape[:2]
        increments = np.random.normal(0.0, 0.12, size=height).astype(np.float32)
        shifts = np.cumsum(increments)
        shifts -= shifts.mean()
        maximum = float(np.max(np.abs(shifts), initial=0.0))
        if maximum > 0.0:
            shifts *= max(0.0, float(self.config.drift_max_pixels)) / maximum
        map_x = np.broadcast_to(np.arange(width, dtype=np.float32), (height, width)).copy()
        map_x -= shifts[:, None]
        map_y = np.broadcast_to(np.arange(height, dtype=np.float32)[:, None], (height, width)).copy()
        return cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)

    def _continuous_focus_variation(self, image: np.ndarray) -> np.ndarray:
        height, width = image.shape[:2]
        center = (random.randrange(max(1, width)), random.randrange(max(1, height)))
        radius = random.uniform(max(3.0, min(height, width) * 0.08), max(4.0, min(height, width) * 0.35))
        mask = np.zeros((height, width), dtype=np.float32)
        cv2.circle(mask, center, max(1, int(radius)), 1.0, -1)
        mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=max(1.0, radius * 0.4))
        blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=random.uniform(0.5, max(0.5, self.config.focus_sigma_max)))
        return image * (1.0 - mask) + blurred * mask

    def _poisson_read_noise(self, image: np.ndarray) -> np.ndarray:
        peak = max(1.0, float(self.config.detector_peak_electrons))
        shot = np.random.poisson(np.clip(image, 0.0, 1.0) * peak).astype(np.float32) / peak
        read = np.random.normal(0.0, max(0.0, self.config.read_noise_sigma), image.shape).astype(np.float32)
        return np.clip(shot + read, 0.0, 1.0)

    def _smooth_gain_field(self, image: np.ndarray) -> np.ndarray:
        height, width = image.shape[:2]
        coarse = np.random.normal(0.0, 1.0, size=(3, 3)).astype(np.float32)
        field = cv2.resize(coarse, (width, height), interpolation=cv2.INTER_CUBIC)
        field /= max(float(np.max(np.abs(field), initial=0.0)), 1e-6)
        gain = 1.0 + max(0.0, float(self.config.gain_field_strength)) * field
        return np.clip(image * gain, 0.0, 1.0)

    def _contamination_and_scan_defects(self, image: np.ndarray) -> np.ndarray:
        result = image.copy()
        height, width = result.shape[:2]
        if random.random() < 0.5:
            y = random.randrange(max(1, height))
            thickness = random.randint(1, max(1, min(4, height)))
            attenuation = random.uniform(0.55, 0.9)
            result[y:min(height, y + thickness)] *= attenuation
        else:
            center = (random.randrange(max(1, width)), random.randrange(max(1, height)))
            radius = random.randint(1, max(2, min(height, width) // 32))
            overlay = result.copy()
            cv2.circle(overlay, center, radius, random.uniform(0.0, 0.4), -1)
            result = cv2.addWeighted(result, 0.65, overlay, 0.35, 0.0)
        return np.clip(result, 0.0, 1.0)

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
            value = int(result[y, x]) + intensity
            cv2.circle(result, (x, y), radius, int(np.clip(value, 0, 255)), -1)
        return result
