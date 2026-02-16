import numpy as np
import pytest

pytest.importorskip('PIL')

from PIL import Image

from lib.image_processing import get_names_from_file, reshape_imgs, cut_image, img_crop_border, sew_image
from tests.helpers import make_test_dir


def test_get_names_from_file():
    tmp_path = make_test_dir("image_processing")
    names_file = tmp_path / 'names.txt'
    names_file.write_text('a\nb\n', encoding='utf-8')

    assert list(get_names_from_file(names_file)) == ['a', 'b']


def test_reshape_imgs():
    imgs = np.arange(2 * 3 * 4).reshape(1, 2, 3, 4)
    reshaped = reshape_imgs(imgs)
    assert reshaped.shape == (1, 2, 3, 1)


def test_cut_and_sew_image_smoke():
    base = np.arange(1 * 4 * 4, dtype=np.float32).reshape(1, 4, 4)
    parts = cut_image(base, (1, 2, 2), overlap=0)
    assert parts.shape[1:] == (1, 2, 2)

    pred = np.ones((parts.shape[0], 1, 2, 2), dtype=np.float32)
    res = sew_image((4, 4), pred, overlap=0)
    assert res.size == (4, 4)


def test_img_crop_border_crops_to_black_frame():
    img = Image.fromarray(np.full((5, 5), 255, dtype=np.uint8), mode='L')
    cropped = img_crop_border(1, img)
    arr = np.array(cropped)
    assert arr[0, 0] == 0
    assert arr[2, 2] == 255

