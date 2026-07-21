import os
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from PIL import Image
import torch
import torch.nn.functional as F

from neuralimage.lib.tiling import TilePlan, build_tile_plan


# Считывание файла с именами изображений
def get_names_from_file(path):
    ##    with open(path) as file:
    ##        img_names = [row.strip() for row in file]
    ##    return img_names

    f = open(path, 'r')
    for line in f:
        filename = line.rstrip('\n')
        yield filename
    f.close()


# Загрузка изображения
def get_imges(fname, dirname, img_rows, img_cols, ext):
    img_names = get_names_from_file(fname)
    imgs = []
    for img_name in img_names:
        path = os.path.join(dirname, img_name + ext)
        im = Image.open(path)  # Загрузка изображения
        if (im.size[0] != img_cols or im.size[1] != img_rows):  # Изменение размера, если неообходимо
            im.thumbnail([img_cols, img_rows])
        imgs.append(np.array(im).astype('float32'))  # Перобразование в формат np.array
    imgs = np.true_divide(imgs, 255)  # Нормализация
    return imgs


# Изменение числа каналов
def reshape_imgs(imgs):
    imgs = imgs.reshape((imgs.shape[0], imgs.shape[1], imgs.shape[2], -1, 1))
    # Значение "-1" означает, что по этому напрвлению размерность итоговой матрицы будет расчитываться исходя из исходной
    # в данном случае это позволяет одинаково обрабатывать как многоканальные RGB, так и одноканаольные L изображения
    imgs = imgs[:, :, :, 0]
    return imgs


# Разрезать большое входное изображение на массив маленьких входных картинок для НС
def cut_image(base_image, segment_size, overlap, *, tile_plan: TilePlan | None = None):
    # segment_size(width,height,channels)
    channels, segment_width, segment_height = segment_size
    base_height = base_image.shape[1]
    base_width = base_image.shape[2]
    plan = tile_plan or build_tile_plan(
        (base_height, base_width),
        (segment_width, segment_height),
        overlap,
    )
    if plan.base_shape_hw != (base_height, base_width):
        raise ValueError(f'Tile plan image shape {plan.base_shape_hw!r} does not match {(base_height, base_width)!r}.')
    if plan.tile_shape_hw != (segment_height, segment_width):
        raise ValueError(f'Tile plan tile shape {plan.tile_shape_hw!r} does not match {(segment_height, segment_width)!r}.')

    fragments = plan.tile_count
    # Tensor layout is (N, C, H, W)
    images = np.zeros((fragments, channels, segment_height, segment_width), dtype=base_image.dtype)

    for image_index, window in enumerate(plan.windows):
        patch = np.zeros((channels, segment_height, segment_width), dtype=base_image.dtype)
        patch[:, :window.height, :window.width] = base_image[
            :, window.top:window.bottom, window.left:window.right
        ]
        images[image_index] = patch

    return images / 255


# Срезать рамку с одноканального изображения (залить чёрным)
def img_crop_border(cropBorder, img):
    im_width = img.size[0]
    im_height = img.size[1]
    imgPix = img.load()  # Выгружаются значения пикселей

    result = np.zeros((im_height, im_width))

    for i in range(cropBorder, im_width - cropBorder, 1):
        for j in range(cropBorder, im_height - cropBorder, 1):
            result[j, i] = imgPix[i, j]

    img = Image.fromarray(result.astype('uint8'), mode=img.mode)
    return img


def _normalize_morph_kernel_size(kernel_size: int) -> int:
    size = max(1, int(kernel_size))
    if size % 2 == 0:
        size += 1
    return size


def _binary_dilate(mask: torch.Tensor, kernel_size: int) -> torch.Tensor:
    if kernel_size <= 1:
        return mask
    padding = kernel_size // 2
    return F.max_pool2d(mask, kernel_size=kernel_size, stride=1, padding=padding)


def _binary_erode(mask: torch.Tensor, kernel_size: int) -> torch.Tensor:
    if kernel_size <= 1:
        return mask
    padding = kernel_size // 2
    return 1.0 - F.max_pool2d(1.0 - mask, kernel_size=kernel_size, stride=1, padding=padding)


def _apply_binary_postprocess(probabilities: np.ndarray, *, threshold: float, kernel_size: int) -> np.ndarray:
    binary = (probabilities >= float(threshold)).astype(np.float32, copy=False)
    normalized_kernel = _normalize_morph_kernel_size(kernel_size)
    if normalized_kernel <= 1:
        return binary

    mask_tensor = torch.from_numpy(binary[None, None, :, :])
    closed = _binary_erode(_binary_dilate(mask_tensor, normalized_kernel), normalized_kernel)
    opened = _binary_dilate(_binary_erode(closed, normalized_kernel), normalized_kernel)
    return opened.squeeze(0).squeeze(0).detach().cpu().numpy().astype(np.float32, copy=False)


# base_image, segment_size, overlap
def _sew_image_legacy(
    base_image,
    predictions: npt.ArrayLike,
    overlap,
    *,
    threshold: float | None = None,
    postprocess_kernel_size: int = 0,
) -> Image:
    """

    :param base_image:  (width,height)
    :param predictions:
    :param overlap:
    :return: sewed image
    """
    base_width = int(base_image[0])
    base_height = int(base_image[1])
    result = np.zeros((base_height, base_width), dtype=np.float32)
    weight_sum = np.zeros((base_height, base_width), dtype=np.float32)

    # Predictions layout is (N, C, H, W)
    segment_height = int(predictions.shape[2])
    segment_width = int(predictions.shape[3])
    stride_height = max(1, int(segment_height - overlap))
    stride_width = max(1, int(segment_width - overlap))
    row_steps = int(base_height / stride_height) + 1
    column_steps = int(base_width / stride_width) + 1

    # If overlap is too large we still keep at least one pixel in each direction.
    raw_crop = int(overlap / 2) if overlap % 2 == 0 else int(overlap / 2) + 1
    crop_height = min(raw_crop, max(0, segment_height // 2))
    crop_width = min(raw_crop, max(0, segment_width // 2))
    parts_count = int(predictions.shape[0])

    base_weight = np.ones((segment_height, segment_width), dtype=np.float32)
    if overlap > 0:
        taper_h = min(segment_height // 2, max(1, int(overlap)))
        taper_w = min(segment_width // 2, max(1, int(overlap)))
        if taper_h > 0:
            ramp_h = np.linspace(0.2, 1.0, taper_h, dtype=np.float32)
            base_weight[:taper_h, :] *= ramp_h[:, None]
            base_weight[-taper_h:, :] *= ramp_h[::-1][:, None]
        if taper_w > 0:
            ramp_w = np.linspace(0.2, 1.0, taper_w, dtype=np.float32)
            base_weight[:, :taper_w] *= ramp_w[None, :]
            base_weight[:, -taper_w:] *= ramp_w[::-1][None, :]
    base_weight = np.clip(base_weight, 1e-4, 1.0)

    for row in range(row_steps):
        for col in range(column_steps):
            sewed_part_index = row * column_steps + col
            if sewed_part_index >= parts_count:
                continue

            left = col * stride_width
            right = left + segment_width
            top = row * stride_height
            bottom = top + segment_height

            src_top = top if bottom <= base_height else max(0, base_height - segment_height)
            src_left = left if right <= base_width else max(0, base_width - segment_width)

            top_crop = 0 if row == 0 else crop_height
            bottom_crop = 0 if row == (row_steps - 1) else crop_height
            left_crop = 0 if col == 0 else crop_width
            right_crop = 0 if col == (column_steps - 1) else crop_width

            dst_top = src_top + top_crop
            dst_bottom = src_top + segment_height - bottom_crop
            dst_left = src_left + left_crop
            dst_right = src_left + segment_width - right_crop

            dst_top = max(0, min(base_height, dst_top))
            dst_bottom = max(0, min(base_height, dst_bottom))
            dst_left = max(0, min(base_width, dst_left))
            dst_right = max(0, min(base_width, dst_right))

            if dst_bottom <= dst_top or dst_right <= dst_left:
                continue

            src_patch_top = dst_top - src_top
            src_patch_left = dst_left - src_left
            src_patch_bottom = src_patch_top + (dst_bottom - dst_top)
            src_patch_right = src_patch_left + (dst_right - dst_left)

            patch = predictions[sewed_part_index, 0, :, :]
            patch_crop = patch[src_patch_top:src_patch_bottom, src_patch_left:src_patch_right]
            weight_crop = base_weight[src_patch_top:src_patch_bottom, src_patch_left:src_patch_right]
            result[dst_top:dst_bottom, dst_left:dst_right] += patch_crop * weight_crop
            weight_sum[dst_top:dst_bottom, dst_left:dst_right] += weight_crop
    weight_sum = np.where(weight_sum <= 0.0, 1.0, weight_sum)
    result = result / weight_sum
    result = np.nan_to_num(result, nan=0.0, posinf=1.0, neginf=0.0)
    result = np.clip(result, 0.0, 1.0)
    if threshold is not None:
        result = _apply_binary_postprocess(
            result,
            threshold=float(min(max(threshold, 0.0), 1.0)),
            kernel_size=int(max(0, postprocess_kernel_size)),
        )
    result = result * 255
    # result = result.reshape(base_height, base_width)
    resimg = Image.fromarray(result.astype('uint8'), mode='L')
    return resimg


@dataclass(frozen=True)
class StitchResult:
    probabilities: np.ndarray
    tile_count: int
    min_weight: float
    max_weight: float
    min_overlap: int
    max_overlap: int


def _raised_cosine_axis_weights(
    length: int,
    *,
    start_overlap: int,
    end_overlap: int,
) -> np.ndarray:
    weights = np.ones(int(length), dtype=np.float32)
    start_taper = min(max(0, int(start_overlap)), int(length))
    end_taper = min(max(0, int(end_overlap)), int(length))
    if start_taper > 0:
        phase = np.arange(1, start_taper + 1, dtype=np.float32) / float(start_taper + 1)
        weights[:start_taper] *= 0.5 - (0.5 * np.cos(np.pi * phase))
    if end_taper > 0:
        phase = np.arange(1, end_taper + 1, dtype=np.float32) / float(end_taper + 1)
        weights[-end_taper:] *= (0.5 - (0.5 * np.cos(np.pi * phase)))[::-1]
    return np.clip(weights, 1e-6, 1.0)


def _axis_overlap_lookup(bounds: set[tuple[int, int]]) -> tuple[dict[int, tuple[int, int]], list[int]]:
    ordered = sorted(bounds)
    lookup: dict[int, tuple[int, int]] = {}
    overlaps: list[int] = []
    for index, (start, end) in enumerate(ordered):
        start_overlap = max(0, ordered[index - 1][1] - start) if index > 0 else 0
        end_overlap = max(0, end - ordered[index + 1][0]) if index + 1 < len(ordered) else 0
        lookup[start] = (start_overlap, end_overlap)
        if end_overlap > 0:
            overlaps.append(end_overlap)
    return lookup, overlaps


def stitch_probability_map(
    base_image: tuple[int, int],
    predictions: npt.ArrayLike,
    overlap: int,
    *,
    tile_plan: TilePlan | None = None,
) -> StitchResult:
    """Stitch binary-segmentation probabilities without discarding overlap pixels."""

    prediction_array = np.asarray(predictions)
    if prediction_array.ndim != 4 or int(prediction_array.shape[1]) != 1:
        raise ValueError(
            'Binary segmentation predictions must have shape (N, 1, H, W), '
            f'got {prediction_array.shape!r}.'
        )
    base_width, base_height = int(base_image[0]), int(base_image[1])
    tile_height, tile_width = int(prediction_array.shape[2]), int(prediction_array.shape[3])
    plan = tile_plan or build_tile_plan(
        (base_height, base_width),
        (tile_width, tile_height),
        overlap,
    )
    if plan.base_shape_hw != (base_height, base_width):
        raise ValueError(f'Tile plan image shape {plan.base_shape_hw!r} does not match {(base_height, base_width)!r}.')
    if plan.tile_shape_hw != (tile_height, tile_width):
        raise ValueError(f'Tile plan tile shape {plan.tile_shape_hw!r} does not match {(tile_height, tile_width)!r}.')
    if int(plan.overlap) != int(overlap):
        raise ValueError(f'Tile plan overlap {plan.overlap} does not match {overlap}.')
    if int(prediction_array.shape[0]) != plan.tile_count:
        raise ValueError(
            f'Prediction count {prediction_array.shape[0]} does not match tile plan count {plan.tile_count}.'
        )

    prediction_array = np.nan_to_num(
        prediction_array.astype(np.float32, copy=False),
        nan=0.0,
        posinf=1.0,
        neginf=0.0,
    )
    result = np.zeros((base_height, base_width), dtype=np.float32)
    weight_sum = np.zeros_like(result)
    x_overlap_lookup, x_overlaps = _axis_overlap_lookup({(window.left, window.right) for window in plan.windows})
    y_overlap_lookup, y_overlaps = _axis_overlap_lookup({(window.top, window.bottom) for window in plan.windows})
    for index, window in enumerate(plan.windows):
        start_y_overlap, end_y_overlap = y_overlap_lookup[window.top]
        y_weights = _raised_cosine_axis_weights(
            window.height,
            start_overlap=start_y_overlap,
            end_overlap=end_y_overlap,
        )
        start_x_overlap, end_x_overlap = x_overlap_lookup[window.left]
        x_weights = _raised_cosine_axis_weights(
            window.width,
            start_overlap=start_x_overlap,
            end_overlap=end_x_overlap,
        )
        weights = y_weights[:, None] * x_weights[None, :]
        patch = prediction_array[index, 0, :window.height, :window.width]
        result[window.top:window.bottom, window.left:window.right] += patch * weights
        weight_sum[window.top:window.bottom, window.left:window.right] += weights

    uncovered = weight_sum <= 0.0
    if bool(np.any(uncovered)):
        raise RuntimeError(f'Tile plan left {int(np.count_nonzero(uncovered))} output pixels uncovered.')
    probabilities = np.clip(result / weight_sum, 0.0, 1.0).astype(np.float32, copy=False)
    actual_overlaps = x_overlaps + y_overlaps
    return StitchResult(
        probabilities=probabilities,
        tile_count=plan.tile_count,
        min_weight=float(weight_sum.min()),
        max_weight=float(weight_sum.max()),
        min_overlap=min(actual_overlaps, default=0),
        max_overlap=max(actual_overlaps, default=0),
    )


def probability_map_to_image(
    probabilities: np.ndarray,
    *,
    threshold: float | None = None,
    postprocess_kernel_size: int = 0,
) -> Image.Image:
    result = np.nan_to_num(np.asarray(probabilities, dtype=np.float32), nan=0.0, posinf=1.0, neginf=0.0)
    result = np.clip(result, 0.0, 1.0)
    if threshold is not None:
        result = _apply_binary_postprocess(
            result,
            threshold=float(min(max(threshold, 0.0), 1.0)),
            kernel_size=int(max(0, postprocess_kernel_size)),
        )
    return Image.fromarray(np.rint(result * 255.0).astype(np.uint8), mode='L')


def sew_image(
    base_image,
    predictions: npt.ArrayLike,
    overlap,
    *,
    threshold: float | None = None,
    postprocess_kernel_size: int = 0,
    tile_plan: TilePlan | None = None,
    pipeline_version: str = 'v2',
) -> Image.Image:
    if str(pipeline_version).strip().lower() in {'legacy', 'legacy_v1', 'v1', '1'}:
        return _sew_image_legacy(
            base_image,
            predictions,
            overlap,
            threshold=threshold,
            postprocess_kernel_size=postprocess_kernel_size,
        )
    stitched = stitch_probability_map(
        base_image,
        predictions,
        overlap,
        tile_plan=tile_plan,
    )
    return probability_map_to_image(
        stitched.probabilities,
        threshold=threshold,
        postprocess_kernel_size=postprocess_kernel_size,
    )
