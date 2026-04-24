import pytest

pytest.importorskip('PIL')
pytest.importorskip('numpy')

from neuralimage.lib.backend import (
    make_lines_splitted,
    check_and_get_size,
    find_size,
    convert_polygon_to_polygon,
    convert_box_to_polygon,
    convert_box_to_ellipse,
)


def test_make_lines_splitted():
    lines = ['a b', '1 2']
    assert make_lines_splitted(lines, ' ') == [['a', 'b'], ['1', '2']]


def test_check_and_get_size_valid_and_invalid():
    ok, size = check_and_get_size(['X', 'S', '10', '20', ';'])
    assert ok is True
    assert size == (10, 20)

    ok, size = check_and_get_size(['X', 'S', 'a', '20'])
    assert ok is False
    assert size == [0, 0]


def test_find_size():
    ok, size = find_size([['P', '1'], ['A', 'S', '30', '40', ';']])
    assert ok is True
    assert size == (30, 40)


def test_convert_polygon_to_polygon():
    status, polygon = convert_polygon_to_polygon(['1', '2', '3', '4'])
    assert status == 1
    assert polygon == [(1, 2), (3, 4)]

    status, polygon = convert_polygon_to_polygon(['1', '2', '3', '1204;\n'])
    assert status == 1
    assert polygon == [(1, 2), (3, 1204)]


def test_check_and_get_size_accepts_trailing_markers():
    ok, size = check_and_get_size(['DS', 'S', '10', '20;\n'])
    assert ok is True
    assert size == (10, 20)


def test_convert_box_to_polygon_and_ellipse():
    status, polygon = convert_box_to_polygon(['4', '2', '10', '20;;'])
    assert status == 1
    assert polygon == [(10, 20), (14, 20), (14, 22), (10, 22)]

    status, ellipse = convert_box_to_ellipse(['4', '2', '10', '20;;'], 100)
    assert status == 1
    assert ellipse == (8, 79, 12, 81)

