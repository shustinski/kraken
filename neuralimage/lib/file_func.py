from pathlib import Path
from typing import Iterable, List, Tuple, Union

SUPPORTED_IMAGES = ('.jpg', '.bmp')

def filter_images(folder: Union[str, Path]) -> List[Path]:
   return filter_files(folder, SUPPORTED_IMAGES)

def filter_files(folder: Union[str, Path],
                 valid_extensions: Tuple[str, ...]) -> List[Path]:
    """
    Возвращает список файлов из *folder*, чьи расширения (без учёта регистра)
    присутствуют в *valid_extensions*.

    Parameters
    ----------
    folder : str | Path
        Папка, в которой производится поиск.
    valid_extensions : tuple[str, ...]
        Кортеж (или любой итерируемый контейнер) допустимых расширений.
        Должны включать точку, например ('.txt', '.csv').

    Returns
    -------
    list[Path]
        Список найденных файлов в виде объектов pathlib.Path.
    """
    folder_path = Path(folder)

    # Приводим все допустимые расширения к нижнему регистру один раз,
    # чтобы сравнение было быстрее.
    lowered_exts = {ext.lower() for ext in valid_extensions}

    # Генерируем список файлов, отфильтрованных по расширению.
    return [
        file_path
        for file_path in folder_path.iterdir()
        if file_path.is_file() and file_path.suffix.lower() in lowered_exts
    ]
