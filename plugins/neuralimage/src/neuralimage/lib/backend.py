import io
import json
import os
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image, ImageDraw

ERROR_CIF_SIZE_NOT_FOUND = 'Не могу найти строку с размером кадра в cif файле'
ERROR_CIF_PARSE = 'Ошибка в cif файле'


def make_lines_splitted(lines: list[str], sep: str) -> list[list[str]]:
    if sep.isspace() or sep == ' ':
        return [line.split() for line in lines]
    return [line.split(sep) for line in lines]


def find_size(cif_lines: list[list[str]]) -> tuple[bool, tuple[int, int] | list[int]]:
    for line in cif_lines:
        is_size_line, size = check_and_get_size(line)
        if is_size_line:
            return True, size
    return False, [0, 0]


def cif_to_jpg(cif_file) -> Image.Image | tuple[int, str]:
    with open(cif_file, 'r') as file:
        cif_lines = file.readlines()

    cif_splitted = make_lines_splitted(cif_lines, ' ')
    is_size_line_contains, size = find_size(cif_splitted)
    if not is_size_line_contains:
        return 0, ERROR_CIF_SIZE_NOT_FOUND

    x_size, y_size = size
    image = Image.new('1', size, 0)
    draw = ImageDraw.Draw(image)

    for line in cif_splitted:
        if len(line) < 5:
            continue

        vector_type = line[0]
        vector_properties = line[1:]
        if vector_type not in ('P', 'B'):
            continue

        if vector_type == 'P':
            status, polygon = convert_polygon_to_polygon(vector_properties)
            if status == 0:
                return 0, polygon
            if len(polygon) < 3:
                continue
            draw.polygon([(x, y_size - y) for x, y in polygon], outline=1, fill=1)
        else:
            status, ellipse = convert_box_to_ellipse(vector_properties, y_size)
            if status == 0:
                return 0, ellipse
            draw.ellipse(ellipse, outline=1, fill=1)

    return image


def check_and_get_size(cif_line: list[str]) -> tuple[bool, tuple[int, int] | list[int]]:
    size = [0, 0]
    if len(cif_line) < 4:
        return False, size
    if 'S' not in cif_line:
        return False, size

    size_index = cif_line.index('S')
    if len(cif_line) < size_index + 3:
        return False, size

    try:
        x_size = _parse_cif_int(cif_line[size_index + 1])
        y_size = _parse_cif_int(cif_line[size_index + 2])
    except (TypeError, ValueError):
        return False, size
    return True, (x_size, y_size)


def _strip_trailing_markers(value: str) -> str:
    return value.rstrip(';\n\r\t ')


def _parse_cif_int(value: str) -> int:
    normalized = _strip_trailing_markers(str(value or ''))
    if not normalized:
        raise ValueError('empty CIF integer token')
    return int(normalized)


def convert_polygon_to_polygon(polygon):
    image_polygon = []
    for i in range(0, len(polygon), 2):
        try:
            x_coord = _parse_cif_int(polygon[i])
            y_coord = _parse_cif_int(polygon[i + 1])
        except (TypeError, ValueError, IndexError):
            return 0, ERROR_CIF_PARSE
        image_polygon.append((x_coord, y_coord))

    return 1, image_polygon


def convert_box_to_polygon(box):
    try:
        x_box_size = _parse_cif_int(box[0])
        y_box_size = _parse_cif_int(box[1])
        x_box_coord = _parse_cif_int(box[2])
        y_box_coord = _parse_cif_int(box[3])
        box_polygon = [
            (x_box_coord, y_box_coord),
            (x_box_coord + x_box_size, y_box_coord),
            (x_box_coord + x_box_size, y_box_coord + y_box_size),
            (x_box_coord, y_box_coord + y_box_size),
        ]
        return 1, box_polygon
    except (TypeError, IndexError, ValueError):
        return 0, ERROR_CIF_PARSE


def convert_box_to_ellipse(box, y_size):
    try:
        x_box_size = _parse_cif_int(box[0])
        y_box_size = _parse_cif_int(box[1])
        x_box_coord = _parse_cif_int(box[2])
        y_box_coord = _parse_cif_int(box[3])
        box_polygon = (
            int(x_box_coord - x_box_size / 2),
            y_size - int(y_box_coord + y_box_size / 2),
            int(x_box_coord + x_box_size / 2),
            y_size - int(y_box_coord - y_box_size / 2),
        )
        return 1, box_polygon
    except (TypeError, IndexError, ValueError):
        return 0, ERROR_CIF_PARSE


def frame_cut(
    frame_path: Path,
    save_path: Path,
    segment_dimension: tuple[int, int],
    horizontal_rotation: bool,
    vertical_rotation: bool,
    flip_x: bool,
    flip_y: bool,
    cut_step: int,
):
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    with Image.open(frame_path) as frame:
        im_width, im_height = frame.size
        segment_width, segment_height = segment_dimension
        column_iterations = int(im_width / cut_step) + 1
        row_iterations = int(im_height / cut_step) + 1

        row = col = 0
        segment_counter = 0

        while row < row_iterations:
            while col < column_iterations:
                segment_counter += 1

                left = col * cut_step
                right = left + segment_width
                top = row * cut_step
                bottom = top + segment_height

                if right >= im_width and bottom >= im_height:
                    crop_coordinates = (
                        im_width - segment_width,
                        im_height - segment_height,
                        im_width,
                        im_height,
                    )
                    col = column_iterations
                    row = row_iterations
                elif right > im_width:
                    crop_coordinates = (im_width - segment_width, top, im_width, bottom)
                    row += 1
                    col = 0
                elif bottom > im_height:
                    crop_coordinates = (left, im_height - segment_height, right, im_height)
                    col += 1
                else:
                    crop_coordinates = (left, top, right, bottom)
                    col += 1

                file_name = f'{Path(frame_path).stem}_{segment_counter}'
                cropped_im = frame.crop(crop_coordinates)
                cropped_im.save(save_path / f'{file_name}.jpg', 'JPEG')

                if horizontal_rotation:
                    cropped_im.rotate(90).save(save_path / f'{file_name}_R90.jpg', 'JPEG')
                    cropped_im.rotate(270).save(save_path / f'{file_name}_R270.jpg', 'JPEG')
                if vertical_rotation:
                    cropped_im.rotate(180).save(save_path / f'{file_name}_R180.jpg', 'JPEG')
                if flip_x:
                    cropped_im.transpose(Image.Transpose.FLIP_TOP_BOTTOM).save(
                        save_path / f'{file_name}_FX.jpg',
                        'JPEG',
                    )
                if flip_y:
                    cropped_im.transpose(Image.Transpose.FLIP_LEFT_RIGHT).save(
                        save_path / f'{file_name}_FY.jpg',
                        'JPEG',
                    )
            col = 0
            row += 1


def get_images(directory):
    directory_path = Path(directory)
    imgs = []
    for file in directory_path.iterdir():
        if file.suffix.upper() in ('.JPG', '.BMP'):
            with Image.open(file) as im:
                imgs.append(np.array(im).astype('float32'))
    return np.true_divide(imgs, 255)


def create_json():
    try:
        with open('data.json', 'r', encoding='utf-8') as fh:
            json.load(fh)
    except (io.UnsupportedOperation, TypeError, FileNotFoundError, json.JSONDecodeError):
        data = {
            'work_mode': 'cif',
            'source_path': '',
            'result_path': '',
            'use_trained_model': False,
            'sample_path': '',
            'sample_jpg_path': '',
            'sample_cif_path': '',
            'enable_90_degree_rotation': False,
            'enable_180_degree_rotation': False,
            'crop_shift': 256,
            'number_of_epochs': 25,
        }
        with open('data.json', 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)


def calculate_image_parts(
    path,
    crop_size,
    step,
    enable_vertical_rotation=False,
    enable_horizontal_rotation=False,
    enable_flip_x=False,
    enable_flip_y=False,
):
    frames_total = 0
    path = Path(path)
    files = [p for p in path.iterdir() if p.is_file()]
    frames_by_frame = np.zeros(len(files))

    for i, image_path in enumerate(files):
        with Image.open(image_path) as frame:
            im_width, im_height = frame.size

        width_steps = int(im_width / crop_size[0]) + 1
        height_steps = int(im_height / crop_size[1]) + 1
        frames_in_frame = width_steps * height_steps

        transform_multiplier = 1
        if enable_vertical_rotation:
            transform_multiplier += 1
        if enable_horizontal_rotation and int(crop_size[0]) == int(crop_size[1]):
            transform_multiplier += 2
        if enable_flip_x:
            transform_multiplier += 1
        if enable_flip_y:
            transform_multiplier += 1
        frames_in_frame *= transform_multiplier

        frames_by_frame[i] = frames_in_frame
        frames_total += frames_in_frame

    return frames_total, frames_by_frame
