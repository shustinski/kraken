import numpy as np
import cv2
from skimage.morphology import skeletonize as skimage_skeletonize

from neuralimage.Validation_gradient_widget_lite.core.repository import _neighbor_count, skeletonize


def test_validation_skeletonize_delegates_to_scikit_image_zhang():
    mask = np.zeros((32, 32), dtype=bool)
    mask[4:27, 7:13] = True
    mask[21:27, 7:25] = True

    assert np.array_equal(
        skeletonize(mask),
        skimage_skeletonize(mask, method="zhang"),
    )


def test_validation_skeletonize_preserves_empty_shape_and_boolean_dtype():
    result = skeletonize(np.zeros((0, 4), dtype=np.uint8))

    assert result.shape == (0, 4)
    assert result.dtype == np.bool_


def test_validation_neighbor_count_matches_opencv_filter():
    mask = np.zeros((9, 11), dtype=np.uint8)
    mask[0:5, 3:8] = 1
    kernel = np.ones((3, 3), dtype=np.uint8)
    kernel[1, 1] = 0

    expected = cv2.filter2D(mask, cv2.CV_8U, kernel, borderType=cv2.BORDER_CONSTANT)
    assert np.array_equal(_neighbor_count(mask), expected)
