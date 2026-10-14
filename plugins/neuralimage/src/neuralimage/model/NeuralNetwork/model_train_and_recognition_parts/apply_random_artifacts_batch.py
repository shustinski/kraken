"""model_train_and_recognition: apply random artifacts batch."""
from __future__ import annotations
import numpy as np
import torch
from neuralimage.lib.random_artifacts import generate_random_artifact_patch


class ApplyRandomArtifactsBatchMixin:
    def _apply_random_artifacts_to_batch(self, image: torch.Tensor) -> torch.Tensor:
        enabled, probability, count, size_ratio, artifact_types = self._resolved_random_artifact_parameters()
        if (not enabled) or probability <= 0.0 or size_ratio <= 0.0 or not artifact_types:
            return image

        batch_size, channels, height, width = image.shape
        min_artifact_height, max_artifact_height, min_artifact_width, max_artifact_width = (
            self._random_artifact_size_bounds(
                image_height=int(height),
                image_width=int(width),
                size_ratio=size_ratio,
            )
        )
        if max_artifact_height <= 0 or max_artifact_width <= 0:
            return image

        artifact_bank = self._ensure_random_artifact_bank(image=image, artifact_types=artifact_types)
        artifact_image = image.clone()
        for sample_index in range(batch_size):
            if float(np.random.random()) > probability:
                continue
            for _ in range(count):
                artifact_height = (
                    int(min_artifact_height)
                    if max_artifact_height == min_artifact_height
                    else int(np.random.randint(min_artifact_height, max_artifact_height + 1))
                )
                artifact_width = (
                    int(min_artifact_width)
                    if max_artifact_width == min_artifact_width
                    else int(np.random.randint(min_artifact_width, max_artifact_width + 1))
                )
                max_top = max(0, height - artifact_height)
                max_left = max(0, width - artifact_width)
                top = 0 if max_top == 0 else int(np.random.randint(0, max_top + 1))
                left = 0 if max_left == 0 else int(np.random.randint(0, max_left + 1))

                if artifact_bank is not None:
                    overlay, alpha = artifact_bank.acquire(
                        height=int(artifact_height),
                        width=int(artifact_width),
                    )
                else:
                    overlay, alpha = generate_random_artifact_patch(
                        int(channels),
                        int(artifact_height),
                        int(artifact_width),
                        device=torch.device('cpu'),
                        dtype=torch.float32,
                        artifact_types=artifact_types,
                    )
                overlay = overlay.to(device=image.device, dtype=image.dtype, non_blocking=True)
                alpha = alpha.to(device=image.device, dtype=image.dtype, non_blocking=True)
                patch = artifact_image[sample_index, :, top:top + artifact_height, left:left + artifact_width]
                artifact_image[sample_index, :, top:top + artifact_height, left:left + artifact_width] = torch.clamp(
                    (patch * (1.0 - alpha)) + (overlay * alpha),
                    min=0.0,
                    max=1.0,
                )
        return artifact_image
