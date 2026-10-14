"""general_neural_handler: indexed dataset."""
from __future__ import annotations
import torch
from torch.utils.data import Dataset
from neuralimage.model.NeuralNetwork.dataset import (
    PatchSampleRequest,
)
from neuralimage.training.hard_mining import compute_geometry_difficulty_score

class IndexedDataset(Dataset):
    def __init__(self, base_dataset: Dataset):
        self._base_dataset = base_dataset

    def __len__(self) -> int:
        return int(len(self._base_dataset))

    def __getitem__(self, index: int):
        resolved_index = int(index.index) if isinstance(index, PatchSampleRequest) else int(index)
        image, label = self._base_dataset[index]
        return image, label, resolved_index

    def describe_sample(self, index: int) -> str:
        describe_fn = getattr(self._base_dataset, 'describe_sample', None)
        if callable(describe_fn):
            return str(describe_fn(int(index)))
        return f'sample_{int(index):06d}'

    def set_epoch(self) -> None:
        set_epoch_fn = getattr(self._base_dataset, 'set_epoch', None)
        if callable(set_epoch_fn):
            set_epoch_fn()

    def frame_key(self, index: int) -> str:
        resolver = getattr(self._base_dataset, 'frame_key', None)
        return str(resolver(int(index))) if callable(resolver) else self.describe_sample(int(index))

    def sample_key(self, index: int) -> str:
        resolver = getattr(self._base_dataset, 'sample_key', None)
        return str(resolver(int(index))) if callable(resolver) else self.describe_sample(int(index))

    def geometry_score(self, index: int) -> float:
        resolver = getattr(self._base_dataset, 'geometry_score', None)
        if callable(resolver):
            return float(resolver(int(index)))
        _image, label = self._base_dataset[int(index)]
        if isinstance(label, dict):
            label = label['mask']
        if torch.is_tensor(label):
            label = label.detach().cpu().numpy()
        return compute_geometry_difficulty_score(label)
