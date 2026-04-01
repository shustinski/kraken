from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image

from lib import backend
from lib.file_func import filter_files
from lib.file_func import filter_images

RARE_PATCH_MASKS_DIRNAME = '.neuralimage_rare_patch_masks'


def resolve_rare_patch_masks_dir(sample_folder: str | Path) -> Path:
    return Path(sample_folder) / RARE_PATCH_MASKS_DIRNAME


def resolve_rare_patch_mask_path(sample_folder: str | Path, sample_name: str | Path) -> Path:
    stem = Path(str(sample_name)).stem
    return resolve_rare_patch_masks_dir(sample_folder) / f'{stem}.png'


def prepare_label_folder_for_rare_patch_editor(
    label_folder: str | Path,
    *,
    log_callback: Callable[[str], None] | None = None,
) -> tuple[Path, str | None]:
    label_root = Path(label_folder)
    if not label_root.is_dir():
        return label_root, f'Label folder does not exist: {label_root}'

    cif_files = sorted(filter_files(label_root, ('.cif',)))
    if not cif_files:
        return label_root, None

    binary_label_root = label_root.parent / 'binary_cif'
    binary_label_root.mkdir(parents=True, exist_ok=True)

    if log_callback is not None:
        log_callback('Обнаружены CIF-метки. Начинаю преобразование в JPG для редактора редких областей.')

    for cif_path in cif_files:
        if log_callback is not None:
            log_callback(f'Преобразую в jpg файл {cif_path.stem}')

        converted = backend.cif_to_jpg(cif_path)
        if isinstance(converted, tuple) and converted[0] == 0:
            return binary_label_root, f'Ошибка в {cif_path.name}: {converted[1]}'

        converted.save(binary_label_root / f'{cif_path.stem}.jpg')

    return binary_label_root, None


def collect_matching_sample_label_pairs(
    sample_folder: str | Path,
    label_folder: str | Path,
    *,
    strict: bool = True,
) -> tuple[list[tuple[Path, Path]], str | None]:
    sample_root = Path(sample_folder)
    label_root = Path(label_folder)
    if not sample_root.is_dir():
        return [], f'Sample folder does not exist: {sample_root}'
    if not label_root.is_dir():
        return [], f'Label folder does not exist: {label_root}'

    sample_files = sorted(filter_images(sample_root))
    label_files = sorted(filter_images(label_root))
    if not sample_files:
        return [], 'No sample images were found.'
    if not label_files:
        return [], 'No label images were found.'

    sample_map, sample_error = _build_stem_map(sample_files, 'sample')
    if sample_error is not None:
        return [], sample_error
    label_map, label_error = _build_stem_map(label_files, 'label')
    if label_error is not None:
        return [], label_error

    sample_stems = set(sample_map.keys())
    label_stems = set(label_map.keys())
    missing_labels = sorted(sample_stems - label_stems)
    missing_samples = sorted(label_stems - sample_stems)
    if strict and (missing_labels or missing_samples):
        missing_labels_preview = ', '.join(missing_labels[:10]) if missing_labels else '-'
        missing_samples_preview = ', '.join(missing_samples[:10]) if missing_samples else '-'
        return (
            [],
            (
                'Image/label mismatch detected. '
                f'Missing labels for samples: {missing_labels_preview}. '
                f'Missing samples for labels: {missing_samples_preview}.'
            ),
        )

    matched_stems = sorted(sample_stems & label_stems)
    pairs = [(sample_map[stem], label_map[stem]) for stem in matched_stems]
    if not pairs:
        return [], 'No matched sample/label pairs were found.'
    return pairs, None


def load_rare_patch_mask(
    sample_folder: str | Path,
    sample_name: str | Path,
    image_size: tuple[int, int],
) -> Image.Image:
    mask_path = resolve_rare_patch_mask_path(sample_folder, sample_name)
    if not mask_path.exists():
        return Image.new('L', image_size, 0)

    with Image.open(mask_path) as image:
        image.load()
        mask = image.convert('L').copy()
    if mask.size != image_size:
        mask = mask.resize(image_size, resample=Image.Resampling.NEAREST)
    return mask


def save_rare_patch_mask(
    sample_folder: str | Path,
    sample_name: str | Path,
    mask: Image.Image | np.ndarray,
) -> Path:
    mask_path = resolve_rare_patch_mask_path(sample_folder, sample_name)
    binary_mask = _normalize_mask_array(mask)
    if binary_mask.size == 0 or not bool(np.any(binary_mask)):
        if mask_path.exists():
            mask_path.unlink()
        _cleanup_empty_masks_dir(mask_path.parent)
        return mask_path

    mask_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(binary_mask, mode='L').save(mask_path, format='PNG')
    return mask_path


def _build_stem_map(files: list[Path], kind: str) -> tuple[dict[str, Path], str | None]:
    mapping: dict[str, Path] = {}
    duplicates: list[str] = []
    for file_path in files:
        stem = file_path.stem
        if stem in mapping:
            duplicates.append(stem)
            continue
        mapping[stem] = file_path
    if duplicates:
        duplicates_preview = ', '.join(sorted(set(duplicates))[:10])
        return {}, f'Duplicate {kind} stems detected: {duplicates_preview}.'
    return mapping, None


def _normalize_mask_array(mask: Image.Image | np.ndarray) -> np.ndarray:
    if isinstance(mask, Image.Image):
        array = np.asarray(mask.convert('L'), dtype=np.uint8)
    else:
        array = np.asarray(mask)
        if array.ndim == 3:
            array = array[..., 0]
        array = array.astype(np.uint8, copy=False)
    return np.where(array > 0, 255, 0).astype(np.uint8, copy=False)


def _cleanup_empty_masks_dir(mask_dir: Path) -> None:
    try:
        if not mask_dir.exists():
            return
        next(mask_dir.iterdir())
    except StopIteration:
        try:
            mask_dir.rmdir()
        except OSError:
            return
    except OSError:
        return
