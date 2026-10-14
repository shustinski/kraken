"""general_neural_handler: composite dataset."""
from __future__ import annotations
import torch
from torch.utils.data import Dataset
from neuralimage.model.NeuralNetwork.dataset import (
    NoCutDataset,
    PatchSampleRequest,
    SyntheticDefectDataset,
)
from neuralimage.training.hard_mining import compute_geometry_difficulty_score

class CompositeDataset(Dataset):
    def __init__(self, *datasets: Dataset):
        self._datasets = [dataset for dataset in datasets if dataset is not None]
        self._lengths = [int(len(dataset)) for dataset in self._datasets]
        self._offsets: list[int] = []
        total = 0
        for length in self._lengths:
            self._offsets.append(total)
            total += int(length)
        self._total_length = total

    def __len__(self) -> int:
        return self._total_length

    def __getitem__(self, index: int):
        requested_size: tuple[int, int] | None = None
        if isinstance(index, PatchSampleRequest):
            requested_size = tuple(index.size_xy)
            index = int(index.index)
        if index < 0 or index >= self._total_length:
            raise IndexError('dataset index out of range')
        for dataset, offset, length in zip(self._datasets, self._offsets, self._lengths):
            if index < offset + length:
                local_index = int(index - offset)
                if requested_size is not None and isinstance(dataset, (NoCutDataset, SyntheticDefectDataset)):
                    return dataset[PatchSampleRequest(local_index, requested_size)]
                return dataset[local_index]
        raise IndexError('dataset index out of range')

    def describe_sample(self, index: int) -> str:
        if index < 0 or index >= self._total_length:
            raise IndexError('dataset index out of range')
        for dataset, offset, length in zip(self._datasets, self._offsets, self._lengths):
            if index >= offset + length:
                continue
            describe_fn = getattr(dataset, 'describe_sample', None)
            if callable(describe_fn):
                return str(describe_fn(index - offset))
            return f'sample_{int(index):06d}'
        return f'sample_{int(index):06d}'

    def set_epoch(self) -> None:
        for dataset in self._datasets:
            set_epoch_fn = getattr(dataset, 'set_epoch', None)
            if callable(set_epoch_fn):
                set_epoch_fn()

    def _resolve_dataset_index(self, index: int) -> tuple[Dataset, int, int]:
        if index < 0 or index >= self._total_length:
            raise IndexError('dataset index out of range')
        for dataset_number, (dataset, offset, length) in enumerate(zip(self._datasets, self._offsets, self._lengths)):
            if index < offset + length:
                return dataset, int(index - offset), int(dataset_number)
        raise IndexError('dataset index out of range')

    def frame_key(self, index: int) -> str:
        dataset, local_index, dataset_number = self._resolve_dataset_index(int(index))
        resolver = getattr(dataset, 'frame_key', None)
        value = resolver(local_index) if callable(resolver) else self.describe_sample(int(index))
        return f'{dataset_number}::{value}'

    def sample_key(self, index: int) -> str:
        dataset, local_index, dataset_number = self._resolve_dataset_index(int(index))
        resolver = getattr(dataset, 'sample_key', None)
        value = resolver(local_index) if callable(resolver) else self.describe_sample(int(index))
        return f'{dataset_number}::{value}'

    def geometry_score(self, index: int) -> float:
        dataset, local_index, _dataset_number = self._resolve_dataset_index(int(index))
        resolver = getattr(dataset, 'geometry_score', None)
        if callable(resolver):
            return float(resolver(local_index))
        _image, label = dataset[local_index]
        if isinstance(label, dict):
            label = label['mask']
        if torch.is_tensor(label):
            label = label.detach().cpu().numpy()
        return compute_geometry_difficulty_score(label)
