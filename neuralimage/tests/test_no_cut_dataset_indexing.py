from conftest import safe_import_or_skip

safe_import_or_skip('torch')
safe_import_or_skip('torchvision')
safe_import_or_skip('PIL')

import numpy as np
from PIL import Image

import lib.images as images
from lib.data_interfaces import SampleCutMode, SampleGenerationSettings, SamplePrepareSettings, TrainingParameters
from lib.images import ImagePreparator, SampleFastCutter
from lib.rare_patch_masks import save_rare_patch_mask
from model.NeuralNetwork.dataset import NoCutDataset
from tests.helpers import make_test_dir


def test_no_cut_dataset_non_square_segments_do_not_overrun_parts_list(monkeypatch):
    monkeypatch.setattr(images, '_sample_fast_cutter_getitem', None, raising=False)
    monkeypatch.setattr(images, '_sample_fast_cutter_accelerated', False, raising=False)

    root = make_test_dir('no_cut_dataset_non_square')
    image_dir = root / 'images'
    label_dir = root / 'labels'
    image_dir.mkdir()
    label_dir.mkdir()

    image_path = image_dir / 'sample.png'
    label_path = label_dir / 'sample.png'
    payload = np.zeros((500, 1000), dtype=np.uint8)
    Image.fromarray(payload, mode='L').save(image_path)
    Image.fromarray(payload, mode='L').save(label_path)

    generation = SampleGenerationSettings(
        step=100,
        segment_size=(256, 512),
        vertical_rotation=False,
        horizontal_rotation=False,
        channels=1,
    )
    prepare = SamplePrepareSettings(enable_crop=False, enable_resize=False)
    settings = TrainingParameters(
        image_path=image_dir,
        label_path=label_dir,
        shuffle=False,
        validation=False,
        validation_percent=0,
        batch_size=1,
        cut_mode=SampleCutMode.online,
        colors=1,
        epochs=1,
        generation=generation,
        prepare=prepare,
    )

    dataset = NoCutDataset([(image_path, label_path)], settings)

    prepared_image = ImagePreparator(image_path, prepare).image.convert('L')
    prepared_label = ImagePreparator(label_path, prepare).image.convert('L')
    expected_parts = len(SampleFastCutter.from_image((prepared_image, prepared_label), generation, shuffle=False))
    assert len(dataset) == expected_parts

    for index in range(len(dataset)):
        image_part, label_part = dataset[index]
        assert image_part.shape == label_part.shape


def test_image_preparator_size_matches_actual_cropped_size():
    root = make_test_dir('image_preparator_crop_size')
    image_path = root / 'sample.png'
    payload = np.zeros((80, 100), dtype=np.uint8)
    Image.fromarray(payload, mode='L').save(image_path)

    prepare = SamplePrepareSettings(
        enable_crop=True,
        enable_resize=False,
        edge_cut=(10, 5),
        target_size=None,
    )

    preparator = ImagePreparator(image_path, prepare)
    assert preparator.size == (80, 70)
    assert preparator.image.size == (80, 70)


def test_no_cut_dataset_length_respects_skip_uniform_labels():
    root = make_test_dir('no_cut_dataset_skip_uniform_labels')
    image_dir = root / 'images'
    label_dir = root / 'labels'
    image_dir.mkdir()
    label_dir.mkdir()

    image_path = image_dir / 'sample.png'
    label_path = label_dir / 'sample.png'

    image_payload = np.arange(16, dtype=np.uint8).reshape(4, 4)
    label_payload = np.zeros((4, 4), dtype=np.uint8)
    label_payload[2:, 2:] = 255
    label_payload[0:2, 2:] = 255
    label_payload[2, 1] = 255

    Image.fromarray(image_payload, mode='L').save(image_path)
    Image.fromarray(label_payload, mode='L').save(label_path)

    generation = SampleGenerationSettings(
        step=2,
        segment_size=(2, 2),
        vertical_rotation=False,
        horizontal_rotation=False,
        channels=1,
        shuffle_patches_in_frame=False,
    )
    prepare = SamplePrepareSettings(enable_crop=False, enable_resize=False)
    settings = TrainingParameters(
        image_path=image_dir,
        label_path=label_dir,
        shuffle=False,
        validation=False,
        validation_percent=0,
        batch_size=1,
        cut_mode=SampleCutMode.online,
        colors=1,
        epochs=1,
        generation=generation,
        prepare=prepare,
        skip_uniform_labels=True,
    )

    dataset = NoCutDataset([(image_path, label_path)], settings)

    assert len(dataset) == 1
    _, label_part = dataset[0]
    assert np.any(label_part > 0.5)
    assert np.any(label_part <= 0.5)


def test_no_cut_dataset_length_respects_rare_patch_oversampling():
    root = make_test_dir('no_cut_dataset_rare_patch_oversampling')
    image_dir = root / 'images'
    label_dir = root / 'labels'
    image_dir.mkdir()
    label_dir.mkdir()

    image_path = image_dir / 'sample.png'
    label_path = label_dir / 'sample.png'

    image_payload = np.arange(16, dtype=np.uint8).reshape(4, 4)
    label_payload = np.zeros((4, 4), dtype=np.uint8)
    label_payload[1:3, 1:3] = 255
    rare_mask_payload = np.zeros((4, 4), dtype=np.uint8)
    rare_mask_payload[0:2, 0:2] = 255

    Image.fromarray(image_payload, mode='L').save(image_path)
    Image.fromarray(label_payload, mode='L').save(label_path)
    save_rare_patch_mask(image_dir, image_path.stem, rare_mask_payload)

    generation = SampleGenerationSettings(
        step=2,
        segment_size=(2, 2),
        vertical_rotation=False,
        horizontal_rotation=False,
        channels=1,
        shuffle_patches_in_frame=False,
    )
    prepare = SamplePrepareSettings(enable_crop=False, enable_resize=False)
    settings = TrainingParameters(
        image_path=image_dir,
        label_path=label_dir,
        shuffle=False,
        validation=False,
        validation_percent=0,
        batch_size=1,
        cut_mode=SampleCutMode.online,
        colors=1,
        epochs=1,
        generation=generation,
        prepare=prepare,
        rare_patch_oversampling_enabled=True,
        rare_patch_oversampling_factor=3,
    )

    dataset = NoCutDataset([(image_path, label_path)], settings)

    assert len(dataset) == 6
    first_image, first_label = dataset[0]
    repeated_image, repeated_label = dataset[2]
    assert np.array_equal(first_image, repeated_image)
    assert np.array_equal(first_label, repeated_label)


def test_no_cut_dataset_set_epoch_rebuilds_lookup_after_frame_shuffle(monkeypatch):
    root = make_test_dir('no_cut_dataset_set_epoch_shuffle')
    image_dir = root / 'images'
    label_dir = root / 'labels'
    image_dir.mkdir()
    label_dir.mkdir()

    first_image_path = image_dir / 'first.png'
    first_label_path = label_dir / 'first.png'
    second_image_path = image_dir / 'second.png'
    second_label_path = label_dir / 'second.png'

    first_payload = np.zeros((4, 4), dtype=np.uint8)
    second_payload = np.zeros((6, 6), dtype=np.uint8)
    Image.fromarray(first_payload, mode='L').save(first_image_path)
    Image.fromarray(first_payload, mode='L').save(first_label_path)
    Image.fromarray(second_payload, mode='L').save(second_image_path)
    Image.fromarray(second_payload, mode='L').save(second_label_path)

    generation = SampleGenerationSettings(
        step=2,
        segment_size=(2, 2),
        vertical_rotation=False,
        horizontal_rotation=False,
        channels=1,
        shuffle_patches_in_frame=False,
    )
    prepare = SamplePrepareSettings(enable_crop=False, enable_resize=False)
    settings = TrainingParameters(
        image_path=image_dir,
        label_path=label_dir,
        shuffle=True,
        validation=False,
        validation_percent=0,
        batch_size=1,
        cut_mode=SampleCutMode.online,
        colors=1,
        epochs=1,
        generation=generation,
        prepare=prepare,
    )

    dataset = NoCutDataset(
        [(first_image_path, first_label_path), (second_image_path, second_label_path)],
        settings,
    )
    total_length = len(dataset)
    original_build_frame_cutter = dataset._build_frame_cutter

    def reverse_in_place(items):
        items.reverse()

    monkeypatch.setattr('model.NeuralNetwork.dataset.random.shuffle', reverse_in_place)
    monkeypatch.setattr(
        dataset,
        '_build_frame_cutter',
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError('_build_frame_cutter called during set_epoch')),
    )
    dataset.set_epoch()
    monkeypatch.setattr(dataset, '_build_frame_cutter', original_build_frame_cutter)

    assert len(dataset) == total_length
    for index in range(len(dataset)):
        image_part, label_part = dataset[index]
        assert image_part.shape == label_part.shape


def test_no_cut_dataset_reuses_cached_frame_cutter(monkeypatch):
    root = make_test_dir('no_cut_dataset_frame_cache')
    image_dir = root / 'images'
    label_dir = root / 'labels'
    image_dir.mkdir()
    label_dir.mkdir()

    first_image_path = image_dir / 'first.png'
    first_label_path = label_dir / 'first.png'
    second_image_path = image_dir / 'second.png'
    second_label_path = label_dir / 'second.png'

    first_payload = np.zeros((4, 4), dtype=np.uint8)
    second_payload = np.ones((4, 4), dtype=np.uint8) * 255
    Image.fromarray(first_payload, mode='L').save(first_image_path)
    Image.fromarray(first_payload, mode='L').save(first_label_path)
    Image.fromarray(second_payload, mode='L').save(second_image_path)
    Image.fromarray(second_payload, mode='L').save(second_label_path)

    generation = SampleGenerationSettings(
        step=2,
        segment_size=(2, 2),
        vertical_rotation=False,
        horizontal_rotation=False,
        channels=1,
        shuffle_patches_in_frame=False,
    )
    settings = TrainingParameters(
        image_path=image_dir,
        label_path=label_dir,
        shuffle=False,
        validation=False,
        validation_percent=0,
        batch_size=1,
        cut_mode=SampleCutMode.online,
        colors=1,
        epochs=1,
        generation=generation,
        prepare=SamplePrepareSettings(enable_crop=False, enable_resize=False),
    )
    dataset = NoCutDataset(
        [(first_image_path, first_label_path), (second_image_path, second_label_path)],
        settings,
    )
    dataset._frame_cache_limit = 2

    calls: list[int] = []
    original_build_frame_cutter = dataset._build_frame_cutter

    def wrapped_build_frame_cutter(frame_index: int, *, shuffle: bool):
        calls.append(frame_index)
        return original_build_frame_cutter(frame_index, shuffle=shuffle)

    monkeypatch.setattr(dataset, '_build_frame_cutter', wrapped_build_frame_cutter)

    dataset[0]
    dataset[4]
    dataset[0]

    assert calls == [0, 1]


def test_no_cut_dataset_length_uses_image_size_fast_path(monkeypatch):
    root = make_test_dir('no_cut_dataset_length_fast_path')
    image_dir = root / 'images'
    label_dir = root / 'labels'
    image_dir.mkdir()
    label_dir.mkdir()

    image_path = image_dir / 'sample.png'
    label_path = label_dir / 'sample.png'
    payload = np.zeros((8, 8), dtype=np.uint8)
    Image.fromarray(payload, mode='L').save(image_path)
    Image.fromarray(payload, mode='L').save(label_path)

    generation = SampleGenerationSettings(
        step=2,
        segment_size=(2, 2),
        vertical_rotation=False,
        horizontal_rotation=False,
        channels=1,
        shuffle_patches_in_frame=False,
    )
    settings = TrainingParameters(
        image_path=image_dir,
        label_path=label_dir,
        shuffle=False,
        validation=False,
        validation_percent=0,
        batch_size=1,
        cut_mode=SampleCutMode.online,
        colors=1,
        epochs=1,
        generation=generation,
        prepare=SamplePrepareSettings(enable_crop=False, enable_resize=False),
    )

    called = {'count': 0}
    original_build_frame_cutter = NoCutDataset._build_frame_cutter

    def fail_if_called(self, frame_index: int, *, shuffle: bool):
        called['count'] += 1
        return original_build_frame_cutter(self, frame_index, shuffle=shuffle)

    monkeypatch.setattr(NoCutDataset, '_build_frame_cutter', fail_if_called)

    dataset = NoCutDataset([(image_path, label_path)], settings)

    assert called['count'] == 0
    assert len(dataset) == 16


def test_no_cut_dataset_set_epoch_recomputes_dynamic_frame_lengths(monkeypatch):
    root = make_test_dir('no_cut_dataset_dynamic_epoch_length')
    image_dir = root / 'images'
    label_dir = root / 'labels'
    image_dir.mkdir()
    label_dir.mkdir()

    image_path = image_dir / 'sample.png'
    label_path = label_dir / 'sample.png'
    payload = np.zeros((8, 8), dtype=np.uint8)
    Image.fromarray(payload, mode='L').save(image_path)
    Image.fromarray(payload, mode='L').save(label_path)

    generation = SampleGenerationSettings(
        step=2,
        segment_size=(2, 2),
        vertical_rotation=False,
        horizontal_rotation=False,
        channels=1,
        shuffle_patches_in_frame=False,
        random_crop=True,
    )
    settings = TrainingParameters(
        image_path=image_dir,
        label_path=label_dir,
        shuffle=False,
        validation=False,
        validation_percent=0,
        batch_size=1,
        cut_mode=SampleCutMode.online,
        colors=1,
        epochs=1,
        generation=generation,
        prepare=SamplePrepareSettings(enable_crop=False, enable_resize=False),
        skip_uniform_labels=True,
    )
    dataset = NoCutDataset([(image_path, label_path)], settings)

    class _FakeCutter:
        def __init__(self, length: int):
            self._length = int(length)

        def __len__(self):
            return self._length

        def __getitem__(self, index: int):
            if index < 0 or index >= self._length:
                raise IndexError('fake cutter index out of range')
            image = np.zeros((1, 2, 2), dtype=np.float32)
            label = np.zeros((1, 2, 2), dtype=np.float32)
            return image, label

    monkeypatch.setattr(
        dataset,
        '_build_frame_cutter',
        lambda frame_index, *, shuffle: _FakeCutter(1 if dataset._epoch_index > 0 else 2),
    )

    dataset.set_epoch()

    assert dataset._dynamic_frame_lengths is True
    assert len(dataset) == 1
    image_part, label_part = dataset[0]
    assert image_part.shape == (1, 2, 2)
    assert label_part.shape == (1, 2, 2)
