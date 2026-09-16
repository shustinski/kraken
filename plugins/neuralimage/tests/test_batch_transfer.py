import torch

from neuralimage.training.batch_transfer import move_batch_to_device


def test_move_batch_to_device_recurses_through_target_mappings():
    batch = {
        'mask': torch.ones((1, 1, 2, 2)),
        'geometry': {'orientation': torch.zeros((1, 2, 2, 2))},
        'metadata': ('frame', [torch.tensor([1])]),
    }

    moved = move_batch_to_device(batch, torch.device('cpu'))

    assert moved['geometry']['orientation'].device.type == 'cpu'
    assert moved['metadata'][0] == 'frame'
    assert moved['metadata'][1][0].device.type == 'cpu'
