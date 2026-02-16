import random
from bisect import bisect_right
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor

from lib.data_interfaces import TrainingParameters, SampleGenerationSettings, SamplePrepareSettings
from lib.images import SampleCalculator, ImagePreparator, SampleFastCutter


class NoCutDataset(Dataset):
    def __init__(self, samples, settings:TrainingParameters):
        self.samples = samples
        self.colors = settings.colors
        self.shuffle = settings.shuffle
        self._prep_settings: SamplePrepareSettings = settings.prepare
        self._cut_settings:SampleGenerationSettings = settings.generation
        self._samples_amount: int = 0
        self._lookup_len_list:list[int] = []

        self._current_frame_index: int | None = None
        self._current_image_cutter: SampleFastCutter | None = None

        self._create_files_list()
        self._calculate_len()

    def set_epoch(self):
        if self.shuffle:
            random.shuffle(self.samples)

    def __getitem__(self, index):
        frame, part = index_in_list(index, self._lookup_len_list)
        if self._current_frame_index != frame:
            self._current_frame_index = frame
            image_path,label_path = self.samples[frame]
            prepared_image = ImagePreparator(image_path,self._prep_settings).image
            prepared_label = ImagePreparator(label_path, self._prep_settings).image
            if self.colors == 1:
                prepared_image = prepared_image.convert('L')  # 1‑канальное
                prepared_label = prepared_label.convert('L')
            else:
                prepared_image = prepared_image.convert('RGB') # 3‑канального
                prepared_label = prepared_label.convert('RGB')


            self._current_image_cutter = SampleFastCutter.from_image((prepared_image,prepared_label), self._cut_settings, shuffle=self.shuffle)

        return self._current_image_cutter[part]

    def __len__(self):
        return self._samples_amount


    def _create_files_list(self):
        if self.shuffle:
            random.shuffle(self.samples)

    def _calculate_len(self):

        len_list = [ len(SampleCalculator(ImagePreparator(image, self._prep_settings).size,self._cut_settings))
            for image,_ in self.samples]

        self._lookup_len_list = summarise_list(len_list.copy())
        self._samples_amount = sum(len_list.copy())


def summarise_list(datalist:list[int]):
    for i in range(len(datalist)):
        if i == 0:
            continue
        datalist[i] += datalist[i-1]
    return datalist

def index_in_list(index: int, datalist: list[int]):
    """Map a global sample index to (frame_index, local_part_index).

    ``datalist`` is expected to be a cumulative-length lookup list.
    """
    if not datalist:
        raise ValueError('datalist must not be empty')

    frame = bisect_right(datalist, index)
    if frame == 0:
        return 0, index

    last_index = len(datalist) - 1
    last_total = datalist[-1]

    # Keep exact-tail behavior used by existing callers/tests.
    if index == last_total:
        return len(datalist), 0

    if frame <= last_index:
        return frame, index - datalist[frame - 1]

    # For overflow, clamp frame to the last valid one and return overflow part.
    return last_index, index - last_total


class CustomDataset(Dataset):
    def __init__(self, samples, channels:int, transform=None):
        self.samples:list[tuple[Path,Path]] = samples
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
