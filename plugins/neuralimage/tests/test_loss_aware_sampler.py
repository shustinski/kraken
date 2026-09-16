from conftest import safe_import_or_skip

safe_import_or_skip('torch')
safe_import_or_skip('PIL')

import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset, SequentialSampler

from neuralimage.lib.data_interfaces import HardMiningParameters, SampleGenerationSettings
from neuralimage.lib.images import SampleFastCutter
from neuralimage.model.general_neural_handler import HardFrameSampler, RandomPatchBatchSampler
from neuralimage.model.NeuralNetwork.model_train_and_recognition import _DistributedLossAwareSampler


class _FrameDataset(Dataset):
    def __len__(self):
        return 4

    def __getitem__(self, index):
        return index

    @staticmethod
    def frame_key(index):
        return 'frame-a' if index < 2 else 'frame-b'

    @staticmethod
    def sample_key(index):
        return f'sample-{index}'


def test_hard_frame_sampler_repeats_selected_frame_once_next_epoch():
    dataset = _FrameDataset()
    sampler = HardFrameSampler(dataset, shuffle=False)
    sampler.update_batch_losses(
        torch.tensor([0, 1, 2, 3]),
        torch.tensor([4.0, 4.0, 1.0, 1.0]),
    )

    selected = sampler.finalize_epoch()

    assert selected == {'frame-a'}
    assert list(sampler) == [0, 1, 2, 3, 0, 1]


def test_hard_frame_sampler_does_not_count_duplicate_patch_twice():
    dataset = _FrameDataset()
    sampler = HardFrameSampler(dataset, shuffle=False)
    sampler.update_batch_losses(torch.tensor([0]), torch.tensor([4.0]))
    sampler.update_batch_losses(torch.tensor([0]), torch.tensor([100.0]))
    sampler.update_batch_losses(torch.tensor([1, 2, 3]), torch.tensor([4.0, 1.0, 1.0]))

    sampler.finalize_epoch()

    assert sampler.last_frame_losses['frame-a'] == 4.0


def test_hard_frame_sampler_restores_partial_epoch_without_double_counting():
    dataset = _FrameDataset()
    original = HardFrameSampler(dataset, shuffle=False)
    original.update_batch_losses(torch.tensor([0, 1]), torch.tensor([4.0, 4.0]))

    restored = HardFrameSampler(dataset, shuffle=False)
    restored.load_state_dict(original.state_dict())
    restored.update_batch_losses(torch.tensor([0, 2, 3]), torch.tensor([100.0, 1.0, 1.0]))
    restored.finalize_epoch()

    assert restored.last_frame_losses['frame-a'] == 4.0
    assert restored.last_frame_losses['frame-b'] == 1.0


def test_geometry_and_ema_sampler_is_deterministic_and_keeps_exploration():
    parameters = HardMiningParameters(
        enabled=True,
        geometry_weight=0.5,
        loss_weight=0.5,
        exploration_floor=0.1,
    )
    first = HardFrameSampler(_FrameDataset(), shuffle=True, parameters=parameters)
    first.set_geometry_scores({'sample-0': 1.0, 'sample-1': 0.8, 'sample-2': 0.0, 'sample-3': 0.0})
    first.update_batch_losses(torch.arange(4), torch.tensor([4.0, 3.0, 1.0, 1.0]))
    second = HardFrameSampler(_FrameDataset(), shuffle=True, parameters=parameters)
    second.load_state_dict(first.state_dict())
    assert list(first) == list(second)
    assert set(range(4)).issubset(list(first))


def test_distributed_loss_aware_sampler_shards_one_deterministic_plan():
    rank_zero_base = HardFrameSampler(_FrameDataset(), shuffle=False)
    rank_one_base = HardFrameSampler(_FrameDataset(), shuffle=False)
    rank_zero = _DistributedLossAwareSampler(rank_zero_base, rank=0, world_size=2, drop_last=False)
    rank_one = _DistributedLossAwareSampler(rank_one_base, rank=1, world_size=2, drop_last=False)

    assert list(rank_zero) == [0, 2]
    assert list(rank_one) == [1, 3]
    rank_zero.update_batch_losses(torch.tensor([0, 2]), torch.tensor([4.0, 1.0]))
    assert rank_zero_base.state_dict()['epoch_losses']


def test_random_patch_batch_sampler_uses_one_aligned_size_per_batch():
    dataset = _FrameDataset()
    sampler = RandomPatchBatchSampler(
        SequentialSampler(dataset),
        batch_size=3,
        min_size=(65, 97),
        max_size=(128, 160),
    )

    batches = list(sampler)

    assert [len(batch) for batch in batches] == [3, 1]
    for batch in batches:
        assert len({request.size_xy for request in batch}) == 1
        width, height = batch[0].size_xy
        assert 65 <= width <= 128 and width % 32 == 0
        assert 97 <= height <= 160 and height % 32 == 0


class _VariableTensorDataset(Dataset):
    def __len__(self):
        return 6

    def __getitem__(self, request):
        width, height = request.size_xy
        return torch.zeros(1, height, width), torch.zeros(1, height, width)


def test_random_patch_dataloader_continues_after_first_batch():
    dataset = _VariableTensorDataset()
    loader = DataLoader(
        dataset,
        batch_sampler=RandomPatchBatchSampler(
            SequentialSampler(dataset),
            batch_size=2,
            min_size=(64, 64),
            max_size=(128, 128),
        ),
        num_workers=0,
    )

    batches = list(loader)

    assert len(batches) == 3
    assert all(images.shape == labels.shape for images, labels in batches)


def test_sample_fast_cutter_extracts_requested_rectangular_size_after_rotation():
    image = np.arange(96 * 128, dtype=np.float32).reshape(1, 96, 128)
    label = (image > image.mean()).astype(np.float32)
    settings = SampleGenerationSettings(
        step=32,
        segment_size=(64, 64),
        vertical_rotation=True,
        horizontal_rotation=False,
        channels=1,
    )
    cutter = SampleFastCutter((image, label), settings, shuffle=False)

    rotated_item = next(
        index
        for index in range(len(cutter))
        if cutter._decode_part_index(index)[1] != 'identity'
    )
    patch, patch_label, _bounds = cutter.get_sized_item(rotated_item, size_xy=(96, 64))

    assert patch.shape == (1, 64, 96)
    assert patch_label.shape == patch.shape
