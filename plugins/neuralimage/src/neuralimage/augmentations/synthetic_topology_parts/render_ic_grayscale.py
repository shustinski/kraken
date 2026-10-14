"""synthetic_topology: render ic grayscale."""
from __future__ import annotations
import cv2
import numpy as np


class RenderIcGrayscaleMixin:
    def _render_ic_grayscale(self, mask_u8: np.ndarray, np_rng: np.random.Generator) -> np.ndarray:
        height, width = mask_u8.shape
        background_level = 0.18
        copper_base = 0.62
        blur_sigma = 0.55
        low_freq_scale = 0.022
        background = np.full((height, width), background_level, dtype=np.float32)
        low_freq = cv2.resize(
            np_rng.normal(0.0, 1.0, size=(max(2, height // 16), max(2, width // 16))).astype(np.float32),
            (width, height),
            interpolation=cv2.INTER_CUBIC,
        )
        low_freq = cv2.GaussianBlur(low_freq, (0, 0), sigmaX=2.0)
        low_freq /= max(1e-6, float(np.max(np.abs(low_freq))))
        background += low_freq * low_freq_scale

        copper = (mask_u8 > 0).astype(np.uint8)
        distance = cv2.distanceTransform(copper, cv2.DIST_L2, 5).astype(np.float32)
        if float(distance.max()) > 0.0:
            distance /= float(distance.max())
        copper_tone = copper_base + (distance * (0.14 if self._domain() == 'ic' else 0.18))
        image = background + copper.astype(np.float32) * copper_tone
        image = cv2.GaussianBlur(image, (0, 0), sigmaX=blur_sigma)

        background_noise = np_rng.normal(
            0.0,
            float(self.params.background_noise_sigma),
            size=image.shape,
        ).astype(np.float32)
        trace_noise = np_rng.normal(
            0.0,
            float(self.params.trace_noise_sigma),
            size=image.shape,
        ).astype(np.float32) * copper.astype(np.float32)
        image += background_noise
        image += trace_noise
        return np.clip(image, 0.0, 1.0, out=image)
