import random
from collections import OrderedDict
from bisect import bisect_right
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset, get_worker_info
from torchvision.transforms import ToTensor

from lib.data_interfaces import TrainingParameters, SampleGenerationSettings, SamplePrepareSettings
from lib.images import ImagePreparator, SampleCalculator, SampleFastCutter
from lib.rare_patch_masks import resolve_rare_patch_mask_path


class NoCutDataset(Dataset):
    def __init__(self, samples, settings: TrainingParameters):
        self.samples = samples
        self._sample_folder = Path(settings.image_path)
        self.colors = settings.colors
        self.shuffle_frames = bool(getattr(settings, 'shuffle', True))
        # Backward-compatible alias used by old code/tests.
        self.shuffle = self.shuffle_frames
        self._prep_settings: SamplePrepareSettings = settings.prepare
        self._cut_settings: SampleGenerationSettings = settings.generation
        self._skip_uniform_labels = bool(getattr(settings, 'skip_uniform_labels', False))
        self._rare_patch_oversampling_enabled = bool(
            getattr(settings, 'rare_patch_oversampling_enabled', False)
        )
        self._rare_patch_oversampling_factor = max(
            1,
            int(getattr(settings, 'rare_patch_oversampling_factor', 2)),
        )
        self.shuffle_patches_in_frame = bool(
            getattr(self._cut_settings, 'shuffle_patches_in_frame', self.shuffle_frames)
        )
        self._samples_amount: int = 0
        self._frame_lengths: list[int] = []
        self._lookup_len_list: list[int] = []
        self._epoch_index: int = 0

        self._current_frame_index: int | None = None
        self._current_image_cutter: SampleFastCutter | None = None
        self._frame_cache: OrderedDict[tuple[int, int, bool], SampleFastCutter] = OrderedDict()
        self._frame_cache_limit = self._resolve_frame_cache_limit()

        self._create_files_list()
        self._calculate_len()

    def set_epoch(self):
        self._epoch_index += 1
        if self.shuffle_frames:
            self._shuffle_samples_and_lengths()
            self._rebuild_lookup()
        self._current_frame_index = None
        self._current_image_cutter = None
        self._frame_cache.clear()

    def __getitem__(self, index):
        if index < 0 or index >= self._samples_amount:
            raise IndexError('dataset index out of range')
        frame, part = index_in_list(index, self._lookup_len_list)
        if self._current_frame_index != frame:
            self._current_frame_index = frame
            self._current_image_cutter = self._get_frame_cutter(
                frame,
                shuffle=self.shuffle_patches_in_frame,
            )

        return self._current_image_cutter[part]

    def __len__(self):
        return self._samples_amount

    def _create_files_list(self):
        if self.shuffle_frames:
            random.shuffle(self.samples)

    def _calculate_len(self):
        len_list: list[int] = []
        for frame_index in range(len(self.samples)):
            len_list.append(self._calculate_frame_len(frame_index))

        self._frame_lengths = len_list
        self._rebuild_lookup()

    def _rebuild_lookup(self):
        self._lookup_len_list = summarise_list(self._frame_lengths.copy())
        self._samples_amount = sum(self._frame_lengths)

    def _shuffle_samples_and_lengths(self):
        if not self.samples:
            return
        order = list(range(len(self.samples)))
        random.shuffle(order)
        self.samples = [self.samples[index] for index in order]
        if self._frame_lengths:
            self._frame_lengths = [self._frame_lengths[index] for index in order]

    def _prepare_frame_images(self, frame_index: int):
        image_path, label_path = self.samples[frame_index]
        prepared_image = ImagePreparator(image_path, self._prep_settings).image
        prepared_label = ImagePreparator(label_path, self._prep_settings).image

        if self.colors == 1:
            prepared_image = prepared_image.convert('L')
        else:
            prepared_image = prepared_image.convert('RGB')

        prepared_label = prepared_label.convert('L')
        prepared_rare_mask = self._load_prepared_rare_mask(image_path, prepared_image.size)
        return image_path, prepared_image, prepared_label, prepared_rare_mask

    def _frame_seed(self, frame_index: int) -> int:
        return hash((self._epoch_index, frame_index, len(self.samples))) & 0xFFFFFFFF

    def _build_frame_cutter(self, frame_index: int, *, shuffle: bool) -> SampleFastCutter:
        _image_path, prepared_image, prepared_label, prepared_rare_mask = self._prepare_frame_images(frame_index)
        random_state = random.getstate()
        random.seed(self._frame_seed(frame_index))
        try:
            return SampleFastCutter.from_image(
                (prepared_image, prepared_label),
                self._cut_settings,
                shuffle=shuffle,
                skip_uniform_labels=self._skip_uniform_labels,
                rare_mask=prepared_rare_mask,
                rare_patch_oversampling_factor=self._rare_patch_oversampling_factor,
            )
        finally:
            random.setstate(random_state)

    def _calculate_frame_len(self, frame_index: int) -> int:
        if self._skip_uniform_labels or self._rare_patch_oversampling_enabled:
            return len(self._build_frame_cutter(frame_index, shuffle=False))
        image_path, _label_path = self.samples[frame_index]
        prepared_size = ImagePreparator(image_path, self._prep_settings).size
        return len(SampleCalculator((prepared_size[1], prepared_size[0]), self._cut_settings))

    @staticmethod
    def _resolve_frame_cache_limit() -> int:
        worker_info = get_worker_info()
        if worker_info is None:
            return 4
        worker_count = max(1, int(worker_info.num_workers))
        if worker_count >= 6:
            return 1
        if worker_count >= 3:
            return 2
        return 4

    def _get_frame_cutter(self, frame_index: int, *, shuffle: bool) -> SampleFastCutter:
        cache_limit = max(1, int(self._frame_cache_limit))
        cache_key = (self._epoch_index, int(frame_index), bool(shuffle))
        cached = self._frame_cache.get(cache_key)
        if cached is not None:
            self._frame_cache.move_to_end(cache_key)
            return cached

        cutter = self._build_frame_cutter(frame_index, shuffle=shuffle)
        self._frame_cache[cache_key] = cutter
        self._frame_cache.move_to_end(cache_key)
        while len(self._frame_cache) > cache_limit:
            self._frame_cache.popitem(last=False)
        return cutter

    def _load_prepared_rare_mask(self, image_path: Path, image_size: tuple[int, int]) -> Image.Image | None:
        if not self._rare_patch_oversampling_enabled or self._rare_patch_oversampling_factor <= 1:
            return None
        rare_mask_path = resolve_rare_patch_mask_path(self._sample_folder, image_path.stem)
        if not rare_mask_path.exists():
            return None
        rare_mask = ImagePreparator(rare_mask_path, self._prep_settings).image.convert('L')
        if rare_mask.size != image_size:
            rare_mask = rare_mask.resize(image_size, resample=Image.Resampling.NEAREST)
        if rare_mask.getbbox() is None:
            return None
        return rare_mask


def summarise_list(datalist: list[int]):
    for i in range(len(datalist)):
        if i == 0:
            continue
        datalist[i] += datalist[i - 1]
    return datalist


def index_in_list(index: int, datalist: list[int]):
    """Map a global sample index to (frame_index, local_part_index).

    ``datalist`` is expected to be a cumulative-length lookup list.
    """
    if not datalist:
        raise ValueError('datalist must not be empty')
    if index < 0:
        raise IndexError('dataset index out of range')

    frame = bisect_right(datalist, index)
    if frame == 0:
        return 0, index

    if frame < len(datalist):
        return frame, index - datalist[frame - 1]

    raise IndexError('dataset index out of range')


class CustomDataset(Dataset):
    def __init__(self, samples, channels: int, transform=None):
        self.samples: list[tuple[Path, Path]] = samples
        self.channels = channels
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image = Image.open(self.samples[idx][0]).convert("RGB" if self.channels == 3 else "L")

        # Convert to tensor and normalize to [0., 1.]
        image_tensor = ToTensor()(image)

        # If needed, ensure the data type is float32
        image_tensor = image_tensor.float()

        label = Image.open(self.samples[idx][1]).convert("L")

        # Convert to tensor and normalize to [0., 1.]
        label_tensor = ToTensor()(label)

        # If needed, ensure the data type is float32
        label_tensor = label_tensor.float()

        # if self.transform:
        #     image = self.transform(image)
        #     label = self.transform(label)

        return image_tensor, label_tensor
