import torch

from neuralimage.lib.data_interfaces import OptimizerParameters
from neuralimage.model.NeuralNetwork.model_train_and_recognition import TrainerProcess


class _Scaler:
    def state_dict(self):
        return {'scale': 1.0}


def test_training_checkpoint_v2_is_written_atomically(tmp_path):
    trainer = TrainerProcess.__new__(TrainerProcess)
    trainer._model = torch.nn.Conv2d(1, 1, 1)
    trainer._save_path = tmp_path / 'model.pth'
    trainer._epochs = 4
    trainer._optimizer_params = OptimizerParameters()
    trainer._recommended_inference_threshold = 0.5
    optimizer = torch.optim.AdamW(trainer._model.parameters())

    trainer._save_checkpoint(0, optimizer, _Scaler(), None, None, current_epoch=0, next_batch=7)

    checkpoint_path = tmp_path / 'model.ckpt'
    payload = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    assert payload['version'] == 2
    assert payload['completed_epoch'] == 0
    assert payload['current_epoch'] == 0
    assert payload['next_batch'] == 7
    assert payload['torch_rng_state'] is not None
    assert not (tmp_path / 'model.ckpt.tmp').exists()
