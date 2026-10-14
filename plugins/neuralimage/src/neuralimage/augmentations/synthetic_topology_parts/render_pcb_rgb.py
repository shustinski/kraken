"""synthetic_topology: render pcb rgb."""
from __future__ import annotations
import cv2
import numpy as np


class RenderPcbRgbMixin:
    def _render_pcb_rgb(self, mask_u8: np.ndarray, np_rng: np.random.Generator) -> np.ndarray:
        height, width = mask_u8.shape
        background_palette = np.asarray(
            np_rng.choice(
                np.asarray(
                    [
                        [0.34, 0.28, 0.14],
                        [0.29, 0.22, 0.10],
                        [0.20, 0.31, 0.13],
                        [0.25, 0.36, 0.16],
                    ],
                    dtype=np.float32,
                ),
                axis=0,
            ),
            dtype=np.float32,
        )
        copper_mode = str(np_rng.choice(['copper', 'dark', 'bright', 'through_mask']))
        copper_palette = {
            'copper': np.asarray([0.80, 0.47, 0.23], dtype=np.float32),
            'dark': np.asarray([0.50, 0.33, 0.18], dtype=np.float32),
            'bright': np.asarray([0.88, 0.70, 0.36], dtype=np.float32),
            'through_mask': (background_palette * 0.60) + np.asarray([0.20, 0.18, 0.08], dtype=np.float32),
        }
        copper_base = np.asarray(copper_palette[copper_mode], dtype=np.float32)

        background = np.ones((height, width, 3), dtype=np.float32) * background_palette[None, None, :]
        for channel_index in range(3):
            low_freq = cv2.resize(
                np_rng.normal(0.0, 1.0, size=(max(2, height // 18), max(2, width // 18))).astype(np.float32),
                (width, height),
                interpolation=cv2.INTER_CUBIC,
            )
            low_freq = cv2.GaussianBlur(low_freq, (0, 0), sigmaX=2.4)
            low_freq /= max(1e-6, float(np.max(np.abs(low_freq))))
            background[..., channel_index] += low_freq * float(np_rng.uniform(0.018, 0.045))

        copper = (mask_u8 > 0).astype(np.uint8)
        distance = cv2.distanceTransform(copper, cv2.DIST_L2, 5).astype(np.float32)
        if float(distance.max()) > 0.0:
            distance /= float(distance.max())
        copper_shade = 0.90 + (distance * 0.22)
        copper_rgb = copper_base[None, None, :] * copper_shade[..., None]
        image = background.copy()
        image[copper > 0] = copper_rgb[copper > 0]
        image = cv2.GaussianBlur(image, (0, 0), sigmaX=0.75)

        background_noise = np_rng.normal(
            0.0,
            float(self.params.background_noise_sigma),
            size=image.shape,
        ).astype(np.float32)
        trace_noise = np_rng.normal(
            0.0,
            float(self.params.trace_noise_sigma),
            size=image.shape,
        ).astype(np.float32) * copper[..., None].astype(np.float32)
        image += background_noise
        image += trace_noise
        return np.clip(image, 0.0, 1.0, out=image)
