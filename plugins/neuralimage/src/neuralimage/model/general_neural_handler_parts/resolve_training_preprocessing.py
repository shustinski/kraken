"""general_neural_handler: resolve training preprocessing."""
from __future__ import annotations
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable
from neuralimage.lib.images import ImagePreparator
from neuralimage.preprocessing.config import PreprocessingConfig
from neuralimage.preprocessing.pipeline import image_to_channel_first_float01
from neuralimage.preprocessing.statistics import compute_dataset_statistics

from neuralimage.model.general_neural_handler_parts.deterministic_validation_split import _deterministic_validation_split


class ResolveTrainingPreprocessingMixin:
    def _resolve_training_preprocessing(self, train_samples: list[tuple[Path, Path]]) -> None:
        config = getattr(self.tranining_parameters, 'preprocessing', None) or PreprocessingConfig()
        if config.mode != 'dataset_zscore':
            return
        if config.has_dataset_statistics():
            return
        channels = int(self.tranining_parameters.generation.channels)

        def training_images() -> Iterable[Any]:
            for image_path, _label_path in train_samples:
                image = ImagePreparator(image_path, self.tranining_parameters.prepare, asset_source=self.dataset_source).image
                yield image_to_channel_first_float01(image, channels)

        statistics = compute_dataset_statistics(training_images())
        resolved = replace(
            config,
            dataset_mean=statistics.mean,
            dataset_std=statistics.std,
        )
        self.tranining_parameters.preprocessing = resolved
        self.message_bus.publish(
            'logging',
            (
                'Dataset z-score statistics calculated from train split only: '
                f'mean={statistics.mean:.8f}, std={statistics.std:.8f}, '
                f'pixels={statistics.pixel_count}.'
            ),
        )

    def _split_validation_samples(
        self,
        samples: list[tuple[Path, Path]],
    ) -> tuple[list[tuple[Path, Path]], list[tuple[Path, Path]]]:
        total_count = len(samples)
        val_count = int(total_count * self.tranining_parameters.validation_percent / 100)
        if self.tranining_parameters.validation_percent > 0 and total_count > 1 and val_count == 0:
            val_count = 1
        if val_count >= total_count:
            val_count = max(total_count - 1, 0)
        train_samples, val_samples = _deterministic_validation_split(
            samples,
            val_count=val_count,
            asset_source=self.dataset_source,
        )
        self.message_bus.publish(
            'logging',
            (
                'Validation split: deterministic stratified split by label coverage '
                f'(train={len(train_samples)}, val={len(val_samples)}).'
            ),
        )
        return train_samples, val_samples
