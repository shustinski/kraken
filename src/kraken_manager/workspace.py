"""Two-root local project workspaces and microscope layer source contracts."""

from __future__ import annotations

import os
import re
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any, Mapping


_FORBIDDEN_NAME_CHARS = frozenset('<>:"/\\|?*')
_WINDOWS_DEVICES = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    *(f"COM{index}" for index in range(1, 10)),
    *(f"LPT{index}" for index in range(1, 10)),
}
_FRAME_NUMBER = re.compile(r"(\d+)")


class WorkspaceValidationError(ValueError):
    """Raised when a workspace path or file set is not safe to use."""


class LayerSourceMode(StrEnum):
    MANAGED_COPY = "managed_copy"
    EXTERNAL = "external"


class DerivedRunKind(StrEnum):
    DATASET = "dataset"
    RESULT = "result"
    VECTOR = "vector"


class DerivedRunState(StrEnum):
    DRAFT = "draft"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class ImageConversionSettings:
    target_format: str = "jpg"
    flip_horizontal: bool = False
    flip_vertical: bool = False
    jpeg_quality: int = 95
    jpeg_subsampling: str = "4:4:4"
    jpeg_optimize: bool = False
    jpeg_progressive: bool = False
    png_compression: int = 6
    png_optimize: bool = False

    def __post_init__(self) -> None:
        target = str(self.target_format).lower()
        if target not in {"jpg", "png"}:
            raise WorkspaceValidationError("Формат результата должен быть JPG или PNG.")
        if not 1 <= int(self.jpeg_quality) <= 95:
            raise WorkspaceValidationError("Качество JPEG должно быть от 1 до 95.")
        if self.jpeg_subsampling not in {"4:4:4", "4:2:2", "4:2:0"}:
            raise WorkspaceValidationError("Выбрана неподдерживаемая субдискретизация JPEG.")
        if not 0 <= int(self.png_compression) <= 9:
            raise WorkspaceValidationError("Сжатие PNG должно быть от 0 до 9.")
        object.__setattr__(self, "target_format", target)


@dataclass(frozen=True, slots=True)
class ProjectWorkspaceBinding:
    project_id: str
    project_name: str
    source_root: str
    derived_root: str
    source_project_dir: str
    derived_project_dir: str
    schema_version: int = 1

    @property
    def available(self) -> bool:
        return Path(self.source_project_dir).is_dir() and Path(self.derived_project_dir).is_dir()


@dataclass(frozen=True, slots=True)
class LayerFileBinding:
    layer_id: str
    layer_name: str
    mode: LayerSourceMode
    image_directory: str
    ssc_directory: str = ""
    prv_directory: str = ""
    aux_directory: str = ""
    import_root: str = ""
    conversion: ImageConversionSettings = field(default_factory=ImageConversionSettings)
    frame_positions: Mapping[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "mode", LayerSourceMode(self.mode))
        object.__setattr__(
            self,
            "frame_positions",
            {str(name): int(position) for name, position in self.frame_positions.items()},
        )


@dataclass(frozen=True, slots=True)
class DerivedRun:
    run_id: str
    layer_id: str
    kind: DerivedRunKind
    state: DerivedRunState
    path: str
    plugin_id: str
    operation: str
    created_at: str
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", DerivedRunKind(self.kind))
        object.__setattr__(self, "state", DerivedRunState(self.state))
        object.__setattr__(self, "provenance", dict(self.provenance))


@dataclass(frozen=True, slots=True)
class LayerSourceScan:
    selected_root: str
    working_directory: str
    jpg_files: tuple[str, ...]
    bmp_files: tuple[str, ...]
    ssc_files: tuple[str, ...]
    prv_files: tuple[str, ...]
    frame_positions: Mapping[str, int]
    total_files: int
    total_bytes: int
    issues: tuple[str, ...] = ()
    file_fingerprints: Mapping[str, tuple[int, int]] = field(default_factory=dict)

    @property
    def image_files(self) -> tuple[str, ...]:
        return self.jpg_files or self.bmp_files

    @property
    def ready(self) -> bool:
        return bool(self.image_files) and not self.issues


def validate_workspace_name(value: str, *, field_name: str) -> str:
    """Return a Windows-safe, exact folder name without slugging it."""

    name = str(value).strip()
    if not name:
        raise WorkspaceValidationError(f"{field_name}: значение обязательно.")
    if name != value or name.endswith((" ", ".")):
        raise WorkspaceValidationError(
            f"{field_name}: пробелы и точки в начале или конце недопустимы."
        )
    if any(character in _FORBIDDEN_NAME_CHARS or ord(character) < 32 for character in name):
        raise WorkspaceValidationError(
            f"{field_name}: имя содержит символ, запрещённый в папках Windows."
        )
    stem = name.split(".", 1)[0].upper()
    if stem in _WINDOWS_DEVICES:
        raise WorkspaceValidationError(
            f"{field_name}: это зарезервированное имя устройства Windows."
        )
    return name


def frame_number(path: Path | str) -> int:
    numbers = _FRAME_NUMBER.findall(Path(path).stem)
    if not numbers:
        raise WorkspaceValidationError(
            f"В имени изображения нет номера кадра: {Path(path).name}"
        )
    return int(numbers[-1])


def map_frame_positions(paths: tuple[Path, ...], *, maximum: int) -> dict[str, int]:
    """Map zero-based filename numbers to Kraken's one-based matrix slots."""

    positions: dict[str, int] = {}
    occupied: dict[int, str] = {}
    for path in paths:
        external_number = frame_number(path)
        if external_number >= maximum:
            raise WorkspaceValidationError(
                f"Кадр {external_number} из файла {path.name} выходит за пределы "
                f"матрицы проекта (допустимы номера от 0 до {maximum - 1})."
            )
        previous = occupied.get(external_number)
        if previous is not None:
            raise WorkspaceValidationError(
                f"Номер кадра {external_number} повторяется в файлах {previous} и {path.name}."
            )
        occupied[external_number] = path.name
        positions[path.name] = external_number + 1
    return positions


def _is_reparse_or_symlink(path: Path) -> bool:
    if path.is_symlink():
        return True
    try:
        attributes = int(getattr(path.stat(follow_symlinks=False), "st_file_attributes", 0))
    except OSError:
        return True
    return bool(attributes & int(getattr(os.stat_result, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)))


def scan_layer_source(root: Path | str, *, maximum_frames: int) -> LayerSourceScan:
    """Find the densest image-containing directory in one filesystem pass."""

    selected = Path(root).expanduser().resolve(strict=True)
    if not selected.is_dir() or _is_reparse_or_symlink(selected):
        raise WorkspaceValidationError("Выбранный источник должен быть обычной папкой.")

    candidates: list[tuple[int, int, str, Path, tuple[Path, ...]]] = []
    total_files = 0
    total_bytes = 0
    fingerprints: dict[str, tuple[int, int]] = {}
    for current_text, directory_names, file_names in os.walk(selected, followlinks=False):
        current = Path(current_text)
        safe_directories: list[str] = []
        for name in directory_names:
            child = current / name
            if _is_reparse_or_symlink(child):
                raise WorkspaceValidationError(
                    f"Символические ссылки и junction-папки недопустимы: {child}"
                )
            safe_directories.append(name)
        directory_names[:] = safe_directories
        files: list[Path] = []
        for name in file_names:
            path = current / name
            if _is_reparse_or_symlink(path) or not path.is_file():
                raise WorkspaceValidationError(
                    f"Источник содержит не обычный файл: {path}"
                )
            files.append(path)
            metadata = path.stat()
            total_files += 1
            total_bytes += metadata.st_size
            fingerprints[str(path.resolve())] = (metadata.st_size, metadata.st_mtime_ns)
        image_count = sum(path.suffix.casefold() in {".jpg", ".jpeg", ".bmp"} for path in files)
        if image_count:
            depth = len(current.relative_to(selected).parts)
            candidates.append((-len(files), depth, str(current).casefold(), current, tuple(files)))

    if not candidates:
        return LayerSourceScan(
            str(selected), "", (), (), (), (), {}, total_files, total_bytes,
            ("В выбранном дереве не найдены изображения JPG/JPEG или BMP.",),
            fingerprints,
        )

    _negative_count, _depth, _key, working, files = min(candidates)
    jpg = tuple(path for path in files if path.suffix.casefold() in {".jpg", ".jpeg"})
    bmp = tuple(path for path in files if path.suffix.casefold() == ".bmp")
    ssc = tuple(path for path in files if path.suffix.casefold() == ".ssc")
    prv = tuple(path for path in files if path.suffix.casefold() == ".prv")
    issues: list[str] = []
    if jpg and bmp:
        issues.append(
            "В рабочей папке одновременно найдены JPG и BMP. "
            "Оставьте изображения только одного формата."
        )
    positions: dict[str, int] = {}
    if not issues:
        try:
            positions = map_frame_positions(jpg or bmp, maximum=maximum_frames)
        except WorkspaceValidationError as exc:
            issues.append(str(exc))
    natural = lambda values: tuple(sorted((str(path) for path in values), key=str.casefold))
    return LayerSourceScan(
        selected_root=str(selected),
        working_directory=str(working),
        jpg_files=natural(jpg),
        bmp_files=natural(bmp),
        ssc_files=natural(ssc),
        prv_files=natural(prv),
        frame_positions=positions,
        total_files=total_files,
        total_bytes=total_bytes,
        issues=tuple(issues),
        file_fingerprints=fingerprints,
    )


def project_workspace_to_dict(binding: ProjectWorkspaceBinding) -> dict[str, Any]:
    return asdict(binding)


def layer_binding_to_dict(binding: LayerFileBinding) -> dict[str, Any]:
    value = asdict(binding)
    value["mode"] = binding.mode.value
    return value


__all__ = [
    "DerivedRun",
    "DerivedRunKind",
    "DerivedRunState",
    "ImageConversionSettings",
    "LayerFileBinding",
    "LayerSourceMode",
    "LayerSourceScan",
    "ProjectWorkspaceBinding",
    "WorkspaceValidationError",
    "frame_number",
    "layer_binding_to_dict",
    "map_frame_positions",
    "project_workspace_to_dict",
    "scan_layer_source",
    "validate_workspace_name",
]
