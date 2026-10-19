"""general_neural_handler: resolve dataloader workers."""
from __future__ import annotations
import os
import sys
import math
import torch
from torch.utils.data import DataLoader, Subset
from neuralimage.lib.data_interfaces import (
    SampleCutMode,
)

from neuralimage.model.general_neural_handler_parts.module_state import _VALIDATION_SPLIT_SEED

from neuralimage.model.general_neural_handler_parts.estimate_label_foreground_ratio import _is_debugger_attached


class ResolveDataloaderWorkersMixin:
    def _create_train_control_loader(self):
        from neuralimage.model.general_neural_handler import DataLoader

        dataset = getattr(self, '_train_control_dataset', None)
        if dataset is None:
            return None
        train_size = int(len(dataset))
        if train_size <= 0:
            return None
        control_size = min(train_size, 1024, max(32, math.ceil(0.05 * train_size)))
        generator = torch.Generator().manual_seed(_VALIDATION_SPLIT_SEED)
        indices = torch.randperm(train_size, generator=generator)[:control_size].tolist()
        subset = Subset(dataset, [int(index) for index in indices])
        self.message_bus.publish(
            'logging',
            f'Fixed train-control set created: seed={_VALIDATION_SPLIT_SEED}, size={control_size}/{train_size}.',
        )
        return DataLoader(
            subset,
            batch_size=self.tranining_parameters.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=bool(torch.cuda.is_available()),
        )

    def _resolve_dataloader_workers(self) -> int:
        from neuralimage.model.general_neural_handler import _is_debugger_attached

        if _is_debugger_attached():
            return 0
        try:
            configured_workers = int(getattr(self.tranining_parameters, 'dataloader_num_workers', -1))
        except (TypeError, ValueError):
            configured_workers = -1
        if configured_workers >= 0:
            return configured_workers
        # PyTorch's spawn-based Windows workers transfer collated tensors via
        # named shared-memory file mappings.  Large image batches can exhaust
        # those mappings only while iterating (after DataLoader construction),
        # so the constructor fallback above cannot recover.  Make automatic
        # mode reliable on Windows; users can still explicitly opt into workers.
        if sys.platform.startswith('win'):
            return 0
        cpu_count = os.cpu_count() or 1
        max_workers = 16
        workers = max(0, min(max_workers, cpu_count - 1))
        if self.tranining_parameters.cut_mode == SampleCutMode.online:
            workers = min(workers, 8)
        if self.tranining_parameters.batch_size <= 4:
            workers = min(workers, 2)
        return workers

    @staticmethod
    def _dataset_requires_worker_restart(dataset) -> bool:
        if dataset is None:
            return False
        return callable(getattr(dataset, 'set_epoch', None))
