from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

torch = pytest.importorskip('torch')

from neuralimage.lib.data_interfaces import RecognitionParameters, SampleCutMode, SampleGenerationSettings, SamplePrepareSettings, TrainingParameters, WorkMode
from neuralimage.model.NeuralNetwork import create_model
from neuralimage.model.NeuralNetwork.dataset import NoCutDataset
from neuralimage.model.NeuralNetwork.recognition_pipeline import cut_image_prepare, gpu_predict
from neuralimage.model.general_neural_handler import GeneralNeuralHandler
from tests.helpers import make_test_dir


class _Bus:
    def __init__(self) -> None:
        self.messages: list[tuple[str, str]] = []

    def publish(self, topic: str, payload) -> None:
        self.messages.append((str(topic), str(payload)))


class _ContextAwareModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.seen_batches: list[tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]] = []

    def forward(self, batch):
        assert isinstance(batch, dict)
        local = batch['local_image']
        context = batch['context_image']
        coords = batch['patch_coords_norm']
        self.seen_batches.append((tuple(local.shape), tuple(context.shape), tuple(coords.shape)))
        return local.mean(dim=1, keepdim=True)


def _save_grayscale(path: Path, matrix: np.ndarray) -> None:
    Image.fromarray(matrix.astype(np.uint8), mode='L').save(path)


def test_quasi_dual_scale_unet_forward_returns_local_mask_shape():
    model = create_model(
        'quasi_dual_scale_unet',
        1,
        local_crop_size=(64, 64),
        context_crop_size=(128, 128),
        context_input_size=(64, 64),
        context_branch_channels=(16, 32, 64, 128),
        fusion_type='concat',
        use_context_branch=True,
    )

    outputs = model(
        {
            'local_image': torch.randn(2, 1, 64, 64),
            'context_image': torch.randn(2, 1, 64, 64),
        }
    )

    assert tuple(outputs['mask'].shape) == (2, 1, 64, 64)
    assert tuple(outputs['confidence'].shape) == (2, 1, 64, 64)


def test_no_cut_dataset_returns_context_input_when_enabled():
    root = make_test_dir('quasi_dual_scale_dataset')
    image_path = root / 'frame.png'
    label_path = root / 'frame_mask.png'
    _save_grayscale(image_path, np.full((6, 6), 255, dtype=np.uint8))
    _save_grayscale(label_path, np.full((6, 6), 255, dtype=np.uint8))

    generation = SampleGenerationSettings(
        step=4,
        segment_size=(4, 4),
        vertical_rotation=False,
        horizontal_rotation=False,
        channels=1,
    )
    settings = TrainingParameters(
        image_path=root,
        label_path=root,
        shuffle=False,
        validation=False,
        validation_percent=0,
        batch_size=1,
        cut_mode=SampleCutMode.online,
        colors=1,
        epochs=1,
        generation=generation,
        prepare=SamplePrepareSettings(),
        local_crop_size=(4, 4),
        context_crop_size=(8, 8),
        context_input_size=(4, 4),
        use_context_branch=True,
    )

    dataset = NoCutDataset([(image_path, label_path)], settings)
    batch_input, label = dataset[0]

    assert isinstance(batch_input, dict)
    assert tuple(batch_input['local_image'].shape) == (1, 4, 4)
    assert tuple(batch_input['global_image'].shape) == (1, 4, 4)
    assert tuple(batch_input['context_image'].shape) == (1, 4, 4)
    assert tuple(batch_input['patch_coords_norm'].shape) == (4,)
    assert tuple(batch_input['patch_coords_px'].shape) == (4,)
    assert np.allclose(batch_input['patch_coords_px'], np.asarray([0, 0, 4, 4], dtype=np.float32))
    assert np.allclose(batch_input['patch_coords_norm'], np.asarray([0.0, 0.0, 4.0 / 6.0, 4.0 / 6.0], dtype=np.float32))
    assert np.allclose(batch_input['source_size_hw'], np.asarray([6, 6], dtype=np.float32))
    assert tuple(label.shape) == (1, 4, 4)


def test_general_handler_passes_quasi_dual_scale_model_kwargs(monkeypatch):
    import neuralimage.model.general_neural_handler as target
    captured: dict[str, object] = {}

    class _DummyModel:
        pass

    def _fake_create_model(model_name, input_channels, **kwargs):
        captured['model_name'] = model_name
        captured['input_channels'] = input_channels
        captured['kwargs'] = kwargs
        return _DummyModel()

    monkeypatch.setattr(target, 'create_model', _fake_create_model)

    handler = GeneralNeuralHandler.__new__(GeneralNeuralHandler)
    handler.work_mode = WorkMode.train_only
    handler.message_bus = _Bus()
    handler.recognition_parameters = SimpleNamespace(model='quasi_dual_scale_unet')
    handler.tranining_parameters = SimpleNamespace(
        colors=1,
        epochs=1,
        cut_mode=SampleCutMode.online,
        image_path=Path('tmp_train') / 'images',
        generation=SimpleNamespace(
            step=32,
            segment_size=(256, 256),
            vertical_rotation=False,
            horizontal_rotation=False,
        ),
        local_crop_size=(256, 256),
        context_crop_size=(512, 512),
        context_input_size=(192, 192),
        context_branch_channels=(16, 32, 64, 128),
        fusion_type='concat',
        use_context_branch=True,
    )

    model, _save_path = target.GeneralNeuralHandler._resolve_training_model(handler)

    assert captured == {
        'model_name': 'quasi_dual_scale_unet',
        'input_channels': 1,
        'kwargs': {
            'local_crop_size': (256, 256),
            'context_crop_size': (512, 512),
            'context_input_size': (192, 192),
            'context_branch_channels': (16, 32, 64, 128),
            'fusion_type': 'concat',
            'use_context_branch': True,
            'use_cross_attention': True,
            'attention_dim': 128,
            'attention_heads': 4,
            'attention_max_global_tokens': 1024,
            'deep_supervision': True,
        },
    }
    assert getattr(model, '_neuralimage_model_kwargs') == captured['kwargs']


def test_gpu_predict_uses_context_batch_when_present():
    root = make_test_dir('quasi_dual_scale_predict')
    image_path = root / 'frame.png'
    _save_grayscale(image_path, np.arange(12 * 12, dtype=np.uint8).reshape(12, 12))

    payload = cut_image_prepare(
        image_path,
        segment_size=(1, 4, 4),
        overlap=0,
        use_context_branch=True,
        context_crop_size=(8, 8),
        context_input_size=(4, 4),
    )
    model = _ContextAwareModel()

    predicted = gpu_predict(payload, model, torch.device('cpu'), batch_size=2)

    assert payload['context_image'] is not None
    assert payload['global_image'] is not None
    assert payload['patch_coords_norm'] is not None
    assert payload['context_image'].shape == payload['cutted_image'].shape
    assert tuple(payload['global_image'].shape) == (1, 4, 4)
    assert tuple(payload['patch_coords_norm'].shape) == (16, 4)
    assert predicted['predicted_image'].shape == payload['cutted_image'].shape
    assert predicted['confidence_image'].shape == payload['cutted_image'].shape
    assert model.seen_batches
    assert model.seen_batches[0][0][-2:] == (4, 4)
    assert model.seen_batches[0][1][-2:] == (4, 4)
    assert model.seen_batches[0][2][-1] == 4
