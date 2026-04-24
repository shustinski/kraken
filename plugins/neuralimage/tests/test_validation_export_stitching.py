from pathlib import Path
from types import SimpleNamespace

import torch

import neuralimage.model.NeuralNetwork.model_train_and_recognition as target
from neuralimage.model.NeuralNetwork.dataset import NoCutDataset
from tests.helpers import make_test_dir


class _Bus:
    def __init__(self) -> None:
        self.messages: list[list[object]] = []

    def put(self, message) -> None:
        self.messages.append(message)


def test_validation_binary_export_uses_recognition_style_stitching(monkeypatch):
    root = make_test_dir('validation_export_stitching')
    image_a = root / 'frame_a.jpg'
    image_b = root / 'frame_b.jpg'
    image_a.write_bytes(b'a')
    image_b.write_bytes(b'b')

    dataset = NoCutDataset.__new__(NoCutDataset)
    dataset.samples = [(image_a, root / 'frame_a_mask.jpg'), (image_b, root / 'frame_b_mask.jpg')]
    dataset._cut_settings = SimpleNamespace(segment_size=(256, 256), step=192)
    dataset.colors = 1
    dataset._use_context_branch = False
    dataset._context_crop_size = None
    dataset._context_input_size = None

    trainer = target.TrainerProcess.__new__(target.TrainerProcess)
    trainer._save_validation_binary_images = True
    trainer._val_dataloader = SimpleNamespace(batch_size=3, dataset=SimpleNamespace(_base_dataset=dataset))
    trainer._save_path = root / 'model.pth'
    trainer._bus = _Bus()
    trainer._model = object()

    cut_calls: list[tuple[Path, tuple[int, int, int], int]] = []
    predict_calls: list[tuple[object, torch.device, int]] = []
    sew_calls: list[tuple[Path, dict, float]] = []

    def _fake_cut(image_path, segment_shape, overlap, **kwargs):
        cut_calls.append((Path(image_path), tuple(segment_shape), int(overlap)))
        return {
            'name': Path(image_path).name,
            'baseim_size': (256, 256),
            'overlap': int(overlap),
            'cutted_image': torch.zeros((1, 1, 256, 256), dtype=torch.float32).numpy(),
        }

    def _fake_predict(prepared, model, device, batch_size):
        predict_calls.append((model, device, int(batch_size)))
        prepared['predicted_image'] = torch.zeros((1, 1, 256, 256), dtype=torch.float32).numpy()
        return prepared

    def _fake_sew(output_dir, item, jpeg_quality=95, *, threshold=None, postprocess_kernel_size=0):
        sew_calls.append((Path(output_dir), dict(item), float(threshold)))

    monkeypatch.setattr(target, '_cut_image_prepare', _fake_cut)
    monkeypatch.setattr(target, '_gpu_predict', _fake_predict)
    monkeypatch.setattr(target, '_sew', _fake_sew)

    trainer._save_validation_binary_predictions(
        epoch=1,
        device=torch.device('cpu'),
        autocast_ctx=lambda: target.nullcontext(),
        threshold=0.65,
    )

    assert [call[0].name for call in cut_calls] == ['frame_a.jpg', 'frame_b.jpg']
    assert all(call[1] == (1, 256, 256) for call in cut_calls)
    assert all(call[2] == 64 for call in cut_calls)
    assert len(predict_calls) == 2
    assert all(call[2] == 3 for call in predict_calls)
    assert len(sew_calls) == 2
    assert all(call[0].name == 'epoch_0002' for call in sew_calls)
    assert all(call[2] == 0.65 for call in sew_calls)
    assert trainer._bus.messages


def test_validation_binary_export_uses_cached_predictions_without_second_inference(monkeypatch):
    root = make_test_dir('validation_export_cached')
    trainer = target.TrainerProcess.__new__(target.TrainerProcess)
    trainer._save_validation_binary_images = True
    trainer._val_dataloader = SimpleNamespace(batch_size=2, dataset=object())
    trainer._save_path = root / 'model.pth'
    trainer._bus = _Bus()

    frame_a = target._ValidationNoCutFrameExportCache(
        frame_index=0,
        image_path=root / 'frame_a.jpg',
        baseim_size=(256, 256),
        overlap=64,
        parts_count=1,
        part_lookup=None,
        patches={0: torch.zeros((1, 256, 256), dtype=torch.float16).numpy()},
    )
    frame_b = target._ValidationNoCutFrameExportCache(
        frame_index=1,
        image_path=root / 'frame_b.jpg',
        baseim_size=(256, 256),
        overlap=64,
        parts_count=1,
        part_lookup=None,
        patches={0: torch.ones((1, 256, 256), dtype=torch.float16).numpy()},
    )
    export_cache = target._ValidationExportCache(
        mode='no_cut',
        dataset=object(),
        frame_predictions={0: frame_a, 1: frame_b},
    )

    monkeypatch.setattr(
        target,
        '_cut_image_prepare',
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError('unexpected second cut pass')),
    )
    monkeypatch.setattr(
        target,
        '_gpu_predict',
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError('unexpected second inference pass')),
    )

    sew_calls: list[tuple[Path, dict, float]] = []

    def _fake_sew(output_dir, item, jpeg_quality=95, *, threshold=None, postprocess_kernel_size=0):
        sew_calls.append((Path(output_dir), dict(item), float(threshold)))

    monkeypatch.setattr(target, '_sew', _fake_sew)

    trainer._save_validation_binary_predictions(
        epoch=1,
        device=torch.device('cpu'),
        autocast_ctx=lambda: target.nullcontext(),
        threshold=0.65,
        export_cache=export_cache,
    )

    assert len(sew_calls) == 2
    assert all(call[0].name == 'epoch_0002' for call in sew_calls)
    assert sorted(call[1]['name'] for call in sew_calls) == ['frame_a.jpg', 'frame_b.jpg']
    assert all(call[2] == 0.65 for call in sew_calls)
