"""general_neural_handler: create dataloader."""
from __future__ import annotations
import sys
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from neuralimage.lib.data_interfaces import (
    SampleCutMode,
)

from neuralimage.model.general_neural_handler_parts.hard_frame_sampler import HardFrameSampler

from neuralimage.model.general_neural_handler_parts.indexed_dataset import IndexedDataset

from neuralimage.model.general_neural_handler_parts.random_patch_batch_sampler import (
    FrameBatchSampler,
    RandomPatchBatchSampler,
)


class CreateDataloaderMixin:
    def _create_dataloader(self, train_dataset, val_dataset):
        from neuralimage.model.general_neural_handler import DataLoader

        self._hard_mining_active = False
        if self._need_stop or train_dataset is None:
            return

        shuffle = False
        workers = self._resolve_dataloader_workers()
        pin_memory = bool(torch.cuda.is_available())
        train_persistent_workers = workers > 0 and not self._dataset_requires_worker_restart(train_dataset)
        val_persistent_workers = workers > 0 and not self._dataset_requires_worker_restart(val_dataset)
        train_loader_kwargs = {
            'batch_size': self.tranining_parameters.batch_size,
            'num_workers': workers,
            'pin_memory': pin_memory,
            'persistent_workers': train_persistent_workers,
        }
        val_loader_kwargs = {
            'batch_size': self.tranining_parameters.batch_size,
            'shuffle': False,
            'num_workers': workers,
            'pin_memory': pin_memory,
            'persistent_workers': val_persistent_workers,
        }
        if workers > 0:
            # Each prefetched Windows batch is transferred through a named
            # shared-memory file mapping.  Keeping only one batch queued per
            # explicitly requested worker substantially reduces mapping/pagefile
            # pressure while preserving the opt-in multiprocessing path.
            prefetch_factor = 1 if sys.platform.startswith('win') else 2
            train_loader_kwargs['prefetch_factor'] = prefetch_factor
            val_loader_kwargs['prefetch_factor'] = prefetch_factor

        hard_mining = self.tranining_parameters.hard_mining
        hard_mining_enabled = bool(hard_mining.enabled)
        self._hard_mining_active = hard_mining_enabled
        if hard_mining_enabled:
            train_dataset = IndexedDataset(train_dataset)
            hard_sampler = HardFrameSampler(
                train_dataset,
                shuffle=bool(shuffle),
                parameters=hard_mining,
            )
            geometry_resolver = getattr(train_dataset, 'geometry_score', None)
            if callable(geometry_resolver) and float(getattr(hard_mining, 'geometry_weight', 0.0)) > 0.0:
                geometry_scores = {
                    hard_sampler._sample_key(index): float(geometry_resolver(index))
                    for index in range(len(train_dataset))
                }
                hard_sampler.set_geometry_scores(geometry_scores)
            train_loader_kwargs['sampler'] = hard_sampler
            train_loader_kwargs['shuffle'] = False
            self.message_bus.publish(
                'logging',
                'Hard-frame sampling enabled: frames with RMS loss above population sigma '
                'will be repeated once in the next epoch.',
            )
        else:
            train_loader_kwargs['shuffle'] = shuffle
        random_patch_size = getattr(self.tranining_parameters, 'random_patch_size', None)
        random_patch_enabled = bool(
            random_patch_size is not None
            and getattr(random_patch_size, 'enabled', False)
            and self.tranining_parameters.cut_mode == SampleCutMode.online
        )
        if random_patch_enabled:
            base_sampler = train_loader_kwargs.pop('sampler', None)
            if base_sampler is None:
                base_sampler = RandomSampler(train_dataset) if shuffle else SequentialSampler(train_dataset)
            train_loader_kwargs.pop('shuffle', None)
            train_loader_kwargs.pop('batch_size', None)
            train_loader_kwargs['batch_sampler'] = RandomPatchBatchSampler(
                base_sampler,
                batch_size=self.tranining_parameters.batch_size,
                min_size=tuple(random_patch_size.min_size),
                max_size=tuple(random_patch_size.max_size),
            )
            self.message_bus.publish(
                'logging',
                (
                    'Random physical patch size enabled for online cutting: '
                    f'min={tuple(random_patch_size.min_size)}, max={tuple(random_patch_size.max_size)}, '
                    'alignment=32, one size per batch.'
                ),
            )
        if val_dataset is not None:
            val_dataset = IndexedDataset(val_dataset)
            base_val_dataset = getattr(val_dataset, '_base_dataset', None)
            frame_lengths_fn = getattr(base_val_dataset, 'validation_frame_lengths', None)
            if callable(frame_lengths_fn):
                val_loader_kwargs.pop('batch_size', None)
                val_loader_kwargs.pop('shuffle', None)
                val_loader_kwargs['batch_sampler'] = FrameBatchSampler(
                    frame_lengths_fn(),
                    batch_size=self.tranining_parameters.batch_size,
                )
        try:
            self.train_loader = DataLoader(
                train_dataset,
                **train_loader_kwargs,
            )
            self.val_loader = (
                DataLoader(
                    val_dataset,
                    **val_loader_kwargs,
                )
                if val_dataset
                else None
            )
            if self.val_loader is not None:
                try:
                    validation_batches = len(self.val_loader)
                except TypeError:
                    validation_batches = 'unknown'
                self.message_bus.publish(
                    'logging',
                    (
                        'Validation loader ready: '
                        f'frames={len(getattr(getattr(val_dataset, "_base_dataset", None), "samples", ()))}, '
                        f'patches={len(val_dataset)}, batches={validation_batches}, '
                        f'batch_size={self.tranining_parameters.batch_size}.'
                    ),
                )
            self.control_loader = self._create_train_control_loader()
        except Exception as error:
            self.message_bus.publish(
                'logging',
                f'Ошибка DataLoader (workers={workers}, pin_memory={pin_memory}): {error}. '
                f'Используется безопасный fallback workers=0.',
            )
            fallback_train_kwargs = {'num_workers': 0, 'pin_memory': False}
            if 'batch_sampler' in train_loader_kwargs:
                fallback_train_kwargs['batch_sampler'] = train_loader_kwargs['batch_sampler']
            elif 'sampler' in train_loader_kwargs:
                fallback_train_kwargs['batch_size'] = self.tranining_parameters.batch_size
                fallback_train_kwargs['sampler'] = train_loader_kwargs['sampler']
                fallback_train_kwargs['shuffle'] = False
            else:
                fallback_train_kwargs['batch_size'] = self.tranining_parameters.batch_size
                fallback_train_kwargs['shuffle'] = shuffle
            self.train_loader = DataLoader(train_dataset, **fallback_train_kwargs)
            fallback_val_kwargs = {'num_workers': 0, 'pin_memory': False}
            if 'batch_sampler' in val_loader_kwargs:
                fallback_val_kwargs['batch_sampler'] = val_loader_kwargs['batch_sampler']
            else:
                fallback_val_kwargs['batch_size'] = self.tranining_parameters.batch_size
                fallback_val_kwargs['shuffle'] = False
            self.val_loader = DataLoader(val_dataset, **fallback_val_kwargs) if val_dataset else None
            self.control_loader = self._create_train_control_loader()
