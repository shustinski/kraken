from pathlib import Path
from typing import Iterable, Any

try:
    from torch import nn
except ModuleNotFoundError:  # allows using non-ML helpers without torch installed
    nn = None


IMAGE_EXTENSIONS = ('.jpg', '.bmp', '.png')
VECTOR_EXTENSIONS = ('.cif',)


def get_input_channels(model: Any) -> int:
    """Return input channels of the first convolution layer in *model*."""
    if nn is None:
        raise ModuleNotFoundError('torch is required for get_input_channels')
    for module in model.modules():
        if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            return module.in_channels
    raise ValueError('Model has no convolution layers')


def _normalize_extensions(extensions: Iterable[str]) -> tuple[str, ...]:
    normalized = []
    for ext in extensions:
        ext = ext.strip().lower()
        if not ext:
            continue
        if not ext.startswith('.'):
            ext = f'.{ext}'
        normalized.append(ext)
    return tuple(normalized)


def _collect_stems(folder: str | Path, extensions: Iterable[str]) -> list[str]:
    folder_path = Path(folder)
    if not folder_path.exists() or not folder_path.is_dir():
        return []

    valid_exts = set(_normalize_extensions(extensions))
    stems: list[str] = []
    for file in folder_path.iterdir():
        if file.is_file() and file.suffix.lower() in valid_exts:
            stems.append(file.stem)
    return stems


def compare_filenames_in_folders(folder1, folder2, extencion=None):
    """
    Compare file stems in image folder (folder1) and vector folder (folder2).

    Returns:
        1 on success, otherwise tuple[int, str] with error details.
    """
    image_extensions = IMAGE_EXTENSIONS if extencion is None else _normalize_extensions((extencion,))

    image_stems = _collect_stems(folder1, image_extensions)
    vector_stems = _collect_stems(folder2, VECTOR_EXTENSIONS)

    if not image_stems:
        return 0, 'В папке нет картинок'

    image_set = set(image_stems)
    vector_set = set(vector_stems)

    if image_set != vector_set:
        vector_absent_in_images = vector_set - image_set
        image_absent_in_vectors = image_set - vector_set

        if vector_absent_in_images:
            return 0, 'Этих векторов нет в папке с кадрами ' + str(vector_absent_in_images)
        if image_absent_in_vectors:
            return 0, 'Этих кадров нет в папке с векторами ' + str(image_absent_in_vectors)

    return 1


def get_names_of_files(files):
    return [Path(file).stem for file in files]
