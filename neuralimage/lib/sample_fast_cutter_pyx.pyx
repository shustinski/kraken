# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: initializedcheck=False
# cython: cdivision=True

import numpy as np
cimport numpy as cnp


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
        if rotation_index == 2:
            image = image[:, ::-1, ::-1].copy()
            label = label[:, ::-1, ::-1].copy()
        elif sample_x == sample_y:
            if rotation_index == 1:
                image = np.rot90(image, k=1, axes=(1, 2)).copy()
                label = np.rot90(label, k=1, axes=(1, 2)).copy()
            else:
                image = np.rot90(image, k=-1, axes=(1, 2)).copy()
                label = np.rot90(label, k=-1, axes=(1, 2)).copy()

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
        return _sample_fast_cutter_getitem_impl(
            parts,
            item,
            self._vertical_rotation,
            self._horizontal_rotation,
            self._width_steps,
            self._step,
            self._sample_x,
            self._sample_y,
            self._base_w,
            self._base_h,
            image_matrix,
            label_matrix,
        )
