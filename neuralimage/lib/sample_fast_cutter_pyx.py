import numpy as np


def sample_fast_cutter_getitem(
    parts_list,
    item,
    vertical_rotation,
    horizontal_rotation,
    width_steps,
    step,
    sample_x,
    sample_y,
    base_w,
    base_h,
    image_matrix,
    label_matrix,
):
    """
    Python fallback shim for environments where the compiled Cython module
    is unavailable. Keeps the same call signature as the Cython implementation.
    """
    loc = parts_list[item]

    if vertical_rotation and horizontal_rotation:
        location = loc // 4
        rotation_index = loc % 4
    elif horizontal_rotation:
        location = loc // 3
        rotation_index = 2 - (loc % 3)
    elif vertical_rotation:
        location = loc // 2
        rotation_index = 2 * (loc % 2)
    else:
        location = loc
        rotation_index = 0

    row, col = divmod(location, width_steps)
    left = col * step
    top = row * step
    right = left + sample_x
    bottom = top + sample_y

    if right <= base_w and bottom <= base_h:
        image = image_matrix[:, top:bottom, left:right].copy()
        label = label_matrix[:, top:bottom, left:right].copy()
    elif right > base_w and bottom > base_h:
        image = image_matrix[:, -sample_y:, -sample_x:].copy()
        label = label_matrix[:, -sample_y:, -sample_x:].copy()
    elif right > base_w:
        image = image_matrix[:, top:bottom, -sample_x:].copy()
        label = label_matrix[:, top:bottom, -sample_x:].copy()
    else:
        image = image_matrix[:, -sample_y:, left:right].copy()
        label = label_matrix[:, -sample_y:, left:right].copy()

    if rotation_index != 0:
        if rotation_index == 1:
            image[0] = np.flipud(image[0].T)
            label[0] = np.flipud(label[0].T)
        elif rotation_index == 2:
            image[0] = image[0, ::-1, ::-1]
            label[0] = label[0, ::-1, ::-1]
        else:
            image[0] = np.fliplr(image[0].T)
            label[0] = np.fliplr(label[0].T)

    return image, label


class SampleFastCutterCython:
    """
    Python shim with the same public API as the Cython cdef class.
    Used when compiled extension is unavailable.
    """

    def __init__(
        self,
        parts_list,
        vertical_rotation,
        horizontal_rotation,
        width_steps,
        step,
        sample_x,
        sample_y,
        base_w,
        base_h,
        image_matrix,
        label_matrix,
    ):
        self._parts = np.asarray(parts_list, dtype=np.int32)
        self._vertical_rotation = bool(vertical_rotation)
        self._horizontal_rotation = bool(horizontal_rotation)
        self._width_steps = int(width_steps)
        self._step = int(step)
        self._sample_x = int(sample_x)
        self._sample_y = int(sample_y)
        self._base_w = int(base_w)
        self._base_h = int(base_h)
        self._image_matrix = image_matrix
        self._label_matrix = label_matrix

    def __len__(self):
        return int(self._parts.shape[0])

    def __getitem__(self, item):
        return sample_fast_cutter_getitem(
            self._parts,
            item,
            self._vertical_rotation,
            self._horizontal_rotation,
            self._width_steps,
            self._step,
            self._sample_x,
            self._sample_y,
            self._base_w,
            self._base_h,
            self._image_matrix,
            self._label_matrix,
        )
