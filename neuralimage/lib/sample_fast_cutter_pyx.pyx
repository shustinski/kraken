# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: initializedcheck=False
# cython: cdivision=True

import numpy as np
cimport numpy as cnp
from libc.string cimport memcpy


ctypedef cnp.float32_t float32_t
ctypedef cnp.int32_t int32_t


cdef tuple _sample_fast_cutter_getitem_impl(
    cnp.ndarray[int32_t, ndim=1] parts,
    Py_ssize_t item,
    bint vertical_rotation,
    bint horizontal_rotation,
    int width_steps,
    int step,
    int sample_x,
    int sample_y,
    int base_w,
    int base_h,
    cnp.ndarray[float32_t, ndim=3] image_matrix,
    cnp.ndarray[float32_t, ndim=3] label_matrix,
):
    cdef int loc = parts[item]
    cdef int location
    cdef int rotation_index
    cdef int row
    cdef int col
    cdef int left
    cdef int top
    cdef int right
    cdef int bottom
    cdef int src_top
    cdef int src_left
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

    row = location // width_steps
    col = location - row * width_steps
    left = col * step
    top = row * step
    right = left + sample_x
    bottom = top + sample_y

    if right <= base_w and bottom <= base_h:
        src_top = top
        src_left = left
    elif right > base_w and bottom > base_h:
        src_top = base_h - sample_y
        src_left = base_w - sample_x
    elif right > base_w:
        src_top = top
        src_left = base_w - sample_x
    else:
        src_top = base_h - sample_y
        src_left = left

    cdef cnp.ndarray[float32_t, ndim=3] image
    cdef cnp.ndarray[float32_t, ndim=3] label

    image = image_matrix[:, src_top:src_top + sample_y, src_left:src_left + sample_x].copy()
    label = label_matrix[:, src_top:src_top + sample_y, src_left:src_left + sample_x].copy()

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


cpdef tuple sample_fast_cutter_getitem(
    object parts_list,
    Py_ssize_t item,
    bint vertical_rotation,
    bint horizontal_rotation,
    int width_steps,
    int step,
    int sample_x,
    int sample_y,
    int base_w,
    int base_h,
    cnp.ndarray[float32_t, ndim=3] image_matrix,
    cnp.ndarray[float32_t, ndim=3] label_matrix,
):
    cdef cnp.ndarray[int32_t, ndim=1] parts = np.asarray(parts_list, dtype=np.int32)
    return _sample_fast_cutter_getitem_impl(
        parts,
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
    )


cdef class SampleFastCutterCython:
    cdef object _parts
    cdef bint _vertical_rotation
    cdef bint _horizontal_rotation
    cdef int _width_steps
    cdef int _step
    cdef int _sample_x
    cdef int _sample_y
    cdef int _base_w
    cdef int _base_h
    cdef object _image_matrix
    cdef object _label_matrix

    def __init__(
        self,
        object parts_list,
        bint vertical_rotation,
        bint horizontal_rotation,
        int width_steps,
        int step,
        int sample_x,
        int sample_y,
        int base_w,
        int base_h,
        cnp.ndarray[float32_t, ndim=3] image_matrix,
        cnp.ndarray[float32_t, ndim=3] label_matrix,
    ):
        self._parts = np.asarray(parts_list, dtype=np.int32)
        self._vertical_rotation = vertical_rotation
        self._horizontal_rotation = horizontal_rotation
        self._width_steps = width_steps
        self._step = step
        self._sample_x = sample_x
        self._sample_y = sample_y
        self._base_w = base_w
        self._base_h = base_h
        self._image_matrix = image_matrix
        self._label_matrix = label_matrix

    def __len__(self):
        return self._parts.shape[0]

    def __getitem__(self, Py_ssize_t item):
        cdef cnp.ndarray[int32_t, ndim=1] parts = self._parts
        cdef cnp.ndarray[float32_t, ndim=3] image_matrix = self._image_matrix
        cdef cnp.ndarray[float32_t, ndim=3] label_matrix = self._label_matrix
        cdef float32_t[:, :, ::1] image_src = image_matrix
        cdef float32_t[:, :, ::1] label_src = label_matrix
        cdef int loc = parts[item]
        cdef int location
        cdef int rotation_index
        cdef int row
        cdef int col
        cdef int left
        cdef int top
        cdef int right
        cdef int bottom
        cdef int src_top
        cdef int src_left
        cdef int channels
        cdef int ch
        cdef int y
        cdef int x
        cdef cnp.ndarray[float32_t, ndim=3] image
        cdef cnp.ndarray[float32_t, ndim=3] label
        cdef float32_t[:, :, ::1] image_out
        cdef float32_t[:, :, ::1] label_out
        cdef bint square_sample

        if self._vertical_rotation and self._horizontal_rotation:
            location = loc // 4
            rotation_index = loc % 4
        elif self._horizontal_rotation:
            location = loc // 3
            rotation_index = 2 - (loc % 3)
        elif self._vertical_rotation:
            location = loc // 2
            rotation_index = 2 * (loc % 2)
        else:
            location = loc
            rotation_index = 0

        row = location // self._width_steps
        col = location - row * self._width_steps
        left = col * self._step
        top = row * self._step
        right = left + self._sample_x
        bottom = top + self._sample_y

        if right <= self._base_w and bottom <= self._base_h:
            src_top = top
            src_left = left
        elif right > self._base_w and bottom > self._base_h:
            src_top = self._base_h - self._sample_y
            src_left = self._base_w - self._sample_x
        elif right > self._base_w:
            src_top = top
            src_left = self._base_w - self._sample_x
        else:
            src_top = self._base_h - self._sample_y
            src_left = left

        channels = image_src.shape[0]
        image = np.empty((channels, self._sample_y, self._sample_x), dtype=np.float32)
        label = np.empty((channels, self._sample_y, self._sample_x), dtype=np.float32)
        image_out = image
        label_out = label
        square_sample = self._sample_x == self._sample_y

        if rotation_index == 0:
            for ch in range(channels):
                for y in range(self._sample_y):
                    memcpy(&image_out[ch, y, 0], &image_src[ch, src_top + y, src_left], self._sample_x * sizeof(float32_t))
                    memcpy(&label_out[ch, y, 0], &label_src[ch, src_top + y, src_left], self._sample_x * sizeof(float32_t))
            return image, label

        for ch in range(channels):
            if ch != 0:
                for y in range(self._sample_y):
                    for x in range(self._sample_x):
                        image_out[ch, y, x] = image_src[ch, src_top + y, src_left + x]
                        label_out[ch, y, x] = label_src[ch, src_top + y, src_left + x]
                continue

            if rotation_index == 2:
                for y in range(self._sample_y):
                    for x in range(self._sample_x):
                        image_out[0, y, x] = image_src[0, src_top + (self._sample_y - 1 - y), src_left + (self._sample_x - 1 - x)]
                        label_out[0, y, x] = label_src[0, src_top + (self._sample_y - 1 - y), src_left + (self._sample_x - 1 - x)]
                continue

            if square_sample and rotation_index == 1:
                for y in range(self._sample_y):
                    for x in range(self._sample_x):
                        image_out[0, y, x] = image_src[0, src_top + x, src_left + (self._sample_x - 1 - y)]
                        label_out[0, y, x] = label_src[0, src_top + x, src_left + (self._sample_x - 1 - y)]
                continue

            if square_sample and rotation_index == 3:
                for y in range(self._sample_y):
                    for x in range(self._sample_x):
                        image_out[0, y, x] = image_src[0, src_top + (self._sample_y - 1 - x), src_left + y]
                        label_out[0, y, x] = label_src[0, src_top + (self._sample_y - 1 - x), src_left + y]
                continue

            # Preserve legacy behavior for non-square samples on 90/270 rotations.
            for y in range(self._sample_y):
                for x in range(self._sample_x):
                    image_out[0, y, x] = image_src[0, src_top + y, src_left + x]
                    label_out[0, y, x] = label_src[0, src_top + y, src_left + x]
            if rotation_index == 1:
                image[0] = np.flipud(image[0].T)
                label[0] = np.flipud(label[0].T)
            else:
                image[0] = np.fliplr(image[0].T)
                label[0] = np.fliplr(label[0].T)

        return image, label
