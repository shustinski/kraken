import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.distributed import DistributedSampler

from neuralimage.model.NeuralNetwork.model_train_and_recognition import TrainerProcess


def _make_loader(size: int, *, batch_size: int, shuffle: bool) -> DataLoader:
    dataset = TensorDataset(
        torch.arange(size, dtype=torch.float32).unsqueeze(1),
        torch.zeros(size, 1, dtype=torch.float32),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)


def test_configure_ddp_dataloaders_keeps_full_validation_on_main_rank():
    trainer = TrainerProcess.__new__(TrainerProcess)
    trainer._train_dataloader = _make_loader(8, batch_size=2, shuffle=True)
    trainer._val_dataloader = _make_loader(6, batch_size=2, shuffle=False)

    original_val_loader = trainer._val_dataloader
    TrainerProcess._configure_ddp_dataloaders(trainer, rank=0, world_size=2)

    assert isinstance(trainer._train_dataloader.sampler, DistributedSampler)
    assert trainer._val_dataloader is original_val_loader
    assert not isinstance(trainer._val_dataloader.sampler, DistributedSampler)


def test_configure_ddp_dataloaders_disables_validation_on_non_main_rank():
    trainer = TrainerProcess.__new__(TrainerProcess)
    trainer._train_dataloader = _make_loader(8, batch_size=2, shuffle=True)
    trainer._val_dataloader = _make_loader(6, batch_size=2, shuffle=False)

    TrainerProcess._configure_ddp_dataloaders(trainer, rank=1, world_size=2)

    assert isinstance(trainer._train_dataloader.sampler, DistributedSampler)
    assert trainer._val_dataloader is None
