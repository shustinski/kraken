from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

torch = pytest.importorskip('torch')

from lib.data_interfaces import RecognitionParameters, SampleCutMode, SampleGenerationSettings, SamplePrepareSettings, TrainingParameters, WorkMode
from model.NeuralNetwork import create_model
from model.NeuralNetwork.dataset import NoCutDataset
from model.NeuralNetwork.recognition_pipeline import cut_image_prepare, gpu_predict
from model.general_neural_handler import GeneralNeuralHandler
from tests.helpers import make_test_dir


class _Bus:
    def __init__(self) -> None:
        self.messages: list[tuple[str, str]] = []

    def publish(self, topic: str, payload) -> None:
        self.messages.append((str(topic), str(payload)))


class _ContextAwareModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.seen_batches: list[tuple[tuple[int, ...], tuple[int, ...]]] = []

    def forward(self, batch):
        assert isinstance(batch, dict)
        local = batch['local_image']
        context = batch['context_image']
        self.seen_batches.append((tuple(local.shape), tuple(context.shape)))
        return local.mean(dim=1, keepdim=True)


def _save_grayscale(path: Path, matrix: np.ndarray) -> None:
    Image.fromarray(matrix.astype(np.uint8), mode='L').save(path)


def test_frame_unet_forward_returns_local_mask_shape():
    model = create_model(
        'FrameUnet',
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
    root = make_test_dir('frame_unet_dataset')
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
    assert tuple(batch_input['context_image'].shape) == (1, 4, 4)
    assert tuple(label.shape) == (1, 4, 4)
    assert float(batch_input['context_image'].min()) < float(batch_input['local_image'].min())


def test_no_cut_dataset_applies_same_geometric_transform_to_context_input():
    root = make_test_dir('frame_unet_rotated_context')
    image_path = root / 'frame.png'
    label_path = root / 'frame_mask.png'
    _save_grayscale(image_path, np.arange(36, dtype=np.uint8).reshape(6, 6))
    _save_grayscale(label_path, np.full((6, 6), 255, dtype=np.uint8))

    generation = SampleGenerationSettings(
        step=4,
        segment_size=(4, 4),
        vertical_rotation=False,
        horizontal_rotation=True,
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
        context_crop_size=(4, 4),
        context_input_size=(4, 4),
        use_context_branch=True,
    )

    dataset = NoCutDataset([(image_path, label_path)], settings)
    batch_input, _label = dataset[1]

    assert np.allclose(batch_input['context_image'], batch_input['local_image'])


def test_general_handler_passes_frame_unet_model_kwargs(monkeypatch):
    import model.general_neural_handler as target

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
    handler.recognition_parameters = SimpleNamespace(model='FrameUnet')
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
        'model_name': 'FrameUnet',
        'input_channels': 1,
        'kwargs': {
            'local_crop_size': (256, 256),
            'context_crop_size': (512, 512),
            'context_input_size': (192, 192),
            'context_branch_channels': (16, 32, 64, 128),
            'fusion_type': 'concat',
            'use_context_branch': True,
            'deep_supervision': True,
        },
    }
    assert getattr(model, '_neuralimage_model_kwargs') == captured['kwargs']


def test_gpu_predict_uses_context_batch_when_present():
    root = make_test_dir('frame_unet_predict')
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
    assert payload['context_image'].shape == payload['cutted_image'].shape
    assert predicted['predicted_image'].shape == payload['cutted_image'].shape
    assert predicted['confidence_image'].shape == payload['cutted_image'].shape
    assert model.seen_batches
    assert model.seen_batches[0][0][-2:] == (4, 4)
    assert model.seen_batches[0][1][-2:] == (4, 4)


def test_gpu_predict_multiscale_keeps_context_and_local_shapes_aligned(monkeypatch):
    root = make_test_dir('frame_unet_predict_multiscale')
    image_path = root / 'frame.png'
    _save_grayscale(image_path, np.arange(64 * 64, dtype=np.uint8).reshape(64, 64))

    payload = cut_image_prepare(
        image_path,
        segment_size=(1, 32, 32),
        overlap=0,
        use_context_branch=True,
        context_crop_size=(48, 48),
        context_input_size=(32, 32),
    )
    model = _ContextAwareModel()
    monkeypatch.setenv('NEURALIMAGE_MS_SCALES', '1.0,0.5')

    gpu_predict(payload, model, torch.device('cpu'), batch_size=2)

    assert model.seen_batches
    assert all(local[-2:] == context[-2:] for local, context in model.seen_batches)


def test_frame_unet_requires_context_input_when_enabled():
    model = create_model(
        'FrameUnet',
        1,
        local_crop_size=(64, 64),
        context_crop_size=(128, 128),
        context_input_size=(64, 64),
        use_context_branch=True,
    )

    with pytest.raises(ValueError, match='context_image'):
        model(torch.randn(1, 1, 64, 64))


def test_context_injection_uses_actual_source_crop_size():
    dataset = NoCutDataset.__new__(NoCutDataset)
    dataset._local_crop_size = (4, 4)
    dataset._context_crop_size = (8, 8)
    dataset._context_input_size = (4, 4)

    context_image = np.zeros((1, 4, 4), dtype=np.float32)
    local_image = np.ones((1, 4, 4), dtype=np.float32)

    injected = dataset._inject_local_patch_into_context(
        context_image,
        local_image,
        source_crop_size_xy=(2, 2),
    )

    assert int(np.count_nonzero(injected)) == 1
