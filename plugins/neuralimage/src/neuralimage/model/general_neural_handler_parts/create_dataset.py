"""general_neural_handler: create dataset."""
from __future__ import annotations
from dataclasses import replace
from pathlib import Path
from neuralimage.lib.data_interfaces import (
    build_pcb_defect_parameters,
    build_synthetic_defect_generator_parameters,
    build_tech_augmentation_config,
)
from neuralimage.model.NeuralNetwork.dataset import (
    NoCutDataset,
    SyntheticDefectDataset,
)

from neuralimage.model.general_neural_handler_parts.composite_dataset import CompositeDataset

from neuralimage.model.general_neural_handler_parts.estimate_label_foreground_ratio import _validation_grid_step


class CreateDatasetMixin:
    def _create_dataset(
        self,
        image_folder: Path,
        label_folder: Path,
        *,
        validation_image_folder: Path | None = None,
        validation_label_folder: Path | None = None,
    ):
        if self._need_stop:
            return None, None
        train_samples = self._collect_matched_samples(image_folder, label_folder)
        if self._need_stop or train_samples is None:
            return None, None

        val_samples: list[tuple[Path, Path]] | None = None
        if self._uses_external_validation():
            if validation_image_folder is None or validation_label_folder is None:
                self.message_bus.publish('error', 'External validation folders are not configured.')
                self._need_stop = True
                return None, None
            val_samples = self._collect_matched_samples(validation_image_folder, validation_label_folder)
            if self._need_stop or val_samples is None:
                return None, None
            self.message_bus.publish(
                'logging',
                (
                    'Validation source: external dataset '
                    f'(train={len(train_samples)}, val={len(val_samples)}).'
                ),
            )
        elif self.tranining_parameters.validation:
            train_samples, val_samples = self._split_validation_samples(train_samples)
            if self._need_stop:
                return None, None

        self._resolve_training_preprocessing(train_samples)

        training_without_tech_aug = replace(
            self.tranining_parameters,
            generation=replace(
                self.tranining_parameters.generation,
                tech_aug=build_tech_augmentation_config(None),
            ),
            pcb_defects=build_pcb_defect_parameters(None),
        )
        evaluation_settings = replace(
            training_without_tech_aug,
            shuffle=False,
            generation=replace(
                training_without_tech_aug.generation,
                # Validation evaluates a deterministic covering grid. Reusing
                # the small training shift here can multiply work by hundreds
                # (for example, 2000px frames, 256px patches, step=12).
                step=_validation_grid_step(training_without_tech_aug.generation.segment_size),
                vertical_rotation=False,
                horizontal_rotation=False,
                flip_x=False,
                flip_y=False,
                additional_augmentation=False,
                random_crop=False,
                scale_augmentation=False,
            ),
            cutout=replace(training_without_tech_aug.cutout, enabled=False),
            random_artifacts=replace(training_without_tech_aug.random_artifacts, enabled=False),
            mixup=replace(training_without_tech_aug.mixup, enabled=False),
        )
        synthetic_generator = build_synthetic_defect_generator_parameters(
            getattr(self.tranining_parameters, 'synthetic_defect_generator', None)
        )

        train_dataset = NoCutDataset(
            train_samples,
            training_without_tech_aug,
            apply_train_only_transforms=True,
            asset_source=self.dataset_source,
        )
        val_dataset = (
            NoCutDataset(
                val_samples,
                evaluation_settings,
                apply_train_only_transforms=False,
                asset_source=self.dataset_source,
            )
            if val_samples
            else None
        )

        self._train_control_dataset = None
        if bool(getattr(self.tranining_parameters.early_stopping, 'enabled', False)) and not val_samples:
            self._train_control_dataset = NoCutDataset(
                train_samples,
                evaluation_settings,
                apply_train_only_transforms=False,
                asset_source=self.dataset_source,
            )
            self.message_bus.publish(
                'logging',
                'Early stopping without validation uses a fixed train-control set. '
                'This mode cannot objectively determine overfitting.',
            )

        if (
            synthetic_generator.enabled
            and float(synthetic_generator.epoch_size_factor) > 0.0
            and train_dataset is not None
            and bool(len(train_samples))
        ):
            synthetic_frame_count = max(
                1,
                int(round(len(train_samples) * float(synthetic_generator.epoch_size_factor))),
            )
            synthetic_settings = replace(
                training_without_tech_aug,
                synthetic_defect_generator=synthetic_generator,
            )
            synthetic_dataset = SyntheticDefectDataset(
                synthetic_frame_count,
                synthetic_settings,
                apply_train_only_transforms=True,
            )
            train_dataset = CompositeDataset(train_dataset, synthetic_dataset)
            self.message_bus.publish(
                'logging',
                (
                    'Synthetic defect dataset generator enabled '
                    f'(real_frames={len(train_samples)}, synthetic_frames={synthetic_frame_count}, '
                    f'synthetic_samples={len(synthetic_dataset)}).'
                ),
            )

        return train_dataset, val_dataset
