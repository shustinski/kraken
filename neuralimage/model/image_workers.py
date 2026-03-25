import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional

from PIL import Image

from lib import backend
from lib.data_interfaces import SampleGenerationSettings
from lib.file_func import filter_images
from lib.message_bus import AbstractMessageBus


_PROGRESS_LOG_STEP = 25


class ConvertCifThread(threading.Thread):
    def __init__(
        self,
        source: Path,
        savepath: Path,
        message_bus: AbstractMessageBus,
        callback: Callable[..., None] | None = None,
    ):
        super().__init__()
        self.path = source
        self.savepath = savepath
        self.bus = message_bus
        self.callback = callback
        self._stop_event = threading.Event()

    def run(self):
        self.bus.publish('logging', 'Начинаю преобразование cif в бинарные изображения')
        self.savepath.mkdir(parents=True, exist_ok=True)

        cif_files = [
            file for file in self.path.iterdir()
            if file.is_file() and file.suffix.lower() == '.cif'
        ]
        total_files = len(cif_files)
        for index, file in enumerate(cif_files, start=1):
            if self._stop_event.is_set():
                break

            if index == 1 or index == total_files or index % _PROGRESS_LOG_STEP == 0:
                self.bus.publish('logging', f'Преобразование CIF: {index}/{total_files} ({file.stem})')
            converted = backend.cif_to_jpg(file)
            if isinstance(converted, tuple) and converted[0] == 0:
                self.bus.publish('logging', f'Ошибка в {file.name}: {converted[1]}')
                continue

            save_path = self.savepath / f'{file.stem}.jpg'
            converted.save(save_path)

        if self.callback is not None:
            self.callback()

    def stop(self):
        self._stop_event.set()


class CutImageThread(threading.Thread):
    def __init__(
        self,
        source: Path,
        target: Path,
        sample_generation_settings: SampleGenerationSettings,
        message_bus: AbstractMessageBus,
        callback: Callable[..., None] | None = None,
    ):
        super().__init__()
        self.setting = sample_generation_settings
        self.bus = message_bus
        self.callback = callback
        self.source = source
        self.target = target
        self._stop_event = threading.Event()

    def run(self):
        self.bus.publish('logging', f'Начинаю производить нарезку кадров из {self.source}')
        image_files = sorted(filter_images(self.source))
        total_files = len(image_files)
        for index, file in enumerate(image_files, start=1):
            if self._stop_event.is_set():
                break

            if index == 1 or index == total_files or index % _PROGRESS_LOG_STEP == 0:
                self.bus.publish('logging', f'Нарезка кадров: {index}/{total_files} ({file.stem})')
            backend.frame_cut(
                file,
                self.target,
                self.setting.segment_size,
                self.setting.horizontal_rotation,
                self.setting.vertical_rotation,
                self.setting.step,
            )

        if self.callback is not None:
            self.callback()

    def stop(self):
        self._stop_event.set()


class CifProccesor:
    """Backward-compatible CIF processor wrapper around lib.backend."""

    x_size: int = 0
    y_size: int = 0
    image: Image.Image | None = None

    def __init__(self, path: str | Path | None = None):
        self.path = str(path) if path is not None else None
        self.errors = ParseErrorCollector()

        if self.path is None:
            self.errors.add('Путь к CIF файлу не указан')
            return

        result = backend.cif_to_jpg(self.path)
        if isinstance(result, tuple) and result[0] == 0:
            self.errors.add(result[1])
            return

        self.image = result
        self.x_size, self.y_size = result.size

    def has_error(self):
        return self.errors.has_errors()

    def get_errors(self):
        return self.errors

    def get_image(self):
        return self.image

    @staticmethod
    def convert_polygon_to_polygon(polygon):
        return backend.convert_polygon_to_polygon(polygon)

    @staticmethod
    def convert_box_to_polygon(box):
        return backend.convert_box_to_polygon(box)

    @staticmethod
    def convert_box_to_ellipse(box, y_size):
        return backend.convert_box_to_ellipse(box, y_size)


@dataclass
class ParseError:
    message: str
    line: Optional[int] = None
    column: Optional[int] = None
    fragment: Optional[str] = None
    cause: Optional[BaseException] = None

    def __str__(self) -> str:
        loc = []
        if self.line is not None:
            loc.append(f'строка {self.line}')
        if self.column is not None:
            loc.append(f'колонка {self.column}')
        location = ', '.join(loc)

        base = self.message
        if location:
            base = f'{base} ({location})'
        if self.fragment:
            base = f'{base}: "{self.fragment}"'
        return base


class ParseErrorCollector:
    def __init__(self):
        self.errors: List[ParseError] = []

    def add(
        self,
        message: str,
        line: Optional[int] = None,
        column: Optional[int] = None,
        fragment: Optional[str] = None,
        cause: Optional[BaseException] = None,
    ) -> None:
        self.errors.append(ParseError(message, line, column, fragment, cause))

    def has_errors(self) -> bool:
        return bool(self.errors)

    def __len__(self) -> int:
        return len(self.errors)

    def __iter__(self):
        return iter(self.errors)

    def __str__(self) -> str:
        if not self.errors:
            return 'Ошибок парсинга не обнаружено.'
        return '\n'.join(f'{i + 1}. {err}' for i, err in enumerate(self.errors))


def safe_int(value: Any, *, base: int = 10, default=None, raise_on_error: bool = False):
    try:
        return int(value, base) if isinstance(value, (str, bytes, bytearray)) else int(value)
    except (ValueError, TypeError):
        if raise_on_error:
            raise
        return default
