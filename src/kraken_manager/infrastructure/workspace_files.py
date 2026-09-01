"""Filesystem implementation of two-root project workspaces."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from threading import Event, Lock
from typing import Any, Callable
from uuid import uuid4

from kraken_manager.workspace import (
    DerivedRun,
    DerivedRunKind,
    DerivedRunState,
    ImageConversionSettings,
    LayerFileBinding,
    LayerSourceMode,
    LayerSourceScan,
    ProjectWorkspaceBinding,
    WorkspaceValidationError,
    map_frame_positions,
    scan_layer_source,
    validate_workspace_name,
)


ProgressCallback = Callable[[int, int, str], None]


@dataclass(frozen=True, slots=True)
class LayerDeletionStage:
    delete_id: str
    moves: tuple[tuple[Path, Path], ...]
    trash_roots: tuple[Path, ...]


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, sort_keys=True, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _contained(root: Path, candidate: Path) -> Path:
    resolved = candidate.resolve(strict=False)
    if resolved != root and root not in resolved.parents:
        raise WorkspaceValidationError(f"Путь выходит за пределы папки проекта: {candidate}")
    return resolved


def _same_or_nested(first: Path, second: Path) -> bool:
    return first == second or first in second.parents or second in first.parents


def validate_workspace_roots(source_root: Path | str, derived_root: Path | str) -> tuple[Path, Path]:
    source = Path(source_root).expanduser().resolve(strict=True)
    derived = Path(derived_root).expanduser().resolve(strict=True)
    if not source.is_dir() or not derived.is_dir():
        raise WorkspaceValidationError("Оба корня хранилищ должны быть существующими папками.")
    if _same_or_nested(source, derived):
        raise WorkspaceValidationError(
            "Корни хранилищ должны различаться и не должны быть вложены друг в друга."
        )
    for label, root in (("source", source), ("derived", derived)):
        try:
            descriptor, probe_name = tempfile.mkstemp(prefix=".kraken-write-test-", dir=root)
            os.close(descriptor)
            Path(probe_name).unlink()
        except OSError as exc:
            title = "исходных" if label == "source" else "производных"
            raise WorkspaceValidationError(
                f"Нет доступа на запись в корень {title} данных: {root}"
            ) from exc
    return source, derived


class WorkspaceRegistry:
    """Atomic local projection for workspace paths; canonical IDs stay in Kraken."""

    def __init__(self, catalog_root: Path | str) -> None:
        self.catalog_root = Path(catalog_root).resolve()

    def _path(self, project_id: str) -> Path:
        return self.catalog_root / "projects" / str(project_id) / "workspace.json"

    def _read(self, project_id: str) -> dict[str, Any]:
        path = self._path(project_id)
        if not path.is_file():
            return {"schema_version": 1, "project": None, "layers": {}, "runs": []}
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or int(payload.get("schema_version", 0)) != 1:
            raise WorkspaceValidationError(f"Неподдерживаемая версия метаданных: {path}")
        payload.setdefault("layers", {})
        payload.setdefault("runs", [])
        return payload

    def save_project(self, binding: ProjectWorkspaceBinding) -> None:
        payload = self._read(binding.project_id)
        payload["project"] = asdict(binding)
        _atomic_json(self._path(binding.project_id), payload)

    def remove_project(self, project_id: str) -> None:
        self._path(project_id).unlink(missing_ok=True)

    def get_project(self, project_id: str) -> ProjectWorkspaceBinding | None:
        raw = self._read(project_id).get("project")
        return ProjectWorkspaceBinding(**raw) if isinstance(raw, dict) else None

    def save_layer(self, project_id: str, binding: LayerFileBinding) -> None:
        payload = self._read(project_id)
        value = asdict(binding)
        value["mode"] = binding.mode.value
        payload["layers"][binding.layer_id] = value
        _atomic_json(self._path(project_id), payload)

    def get_layer(self, project_id: str, layer_id: str) -> LayerFileBinding | None:
        raw = self._read(project_id).get("layers", {}).get(str(layer_id))
        if not isinstance(raw, dict):
            return None
        conversion = raw.get("conversion", {})
        return LayerFileBinding(
            **{
                **raw,
                "conversion": ImageConversionSettings(**conversion),
                "mode": LayerSourceMode(raw["mode"]),
            }
        )

    def remove_layer(self, project_id: str, layer_id: str) -> None:
        payload = self._read(project_id)
        payload["layers"].pop(str(layer_id), None)
        payload["runs"] = [
            value for value in payload["runs"] if str(value.get("layer_id", "")) != str(layer_id)
        ]
        _atomic_json(self._path(project_id), payload)

    def save_run(self, project_id: str, run: DerivedRun) -> None:
        payload = self._read(project_id)
        encoded = asdict(run)
        encoded["kind"] = run.kind.value
        encoded["state"] = run.state.value
        payload["runs"] = [
            value for value in payload["runs"] if str(value.get("run_id", "")) != run.run_id
        ]
        payload["runs"].append(encoded)
        _atomic_json(self._path(project_id), payload)

    def list_runs(self, project_id: str, layer_id: str = "") -> tuple[DerivedRun, ...]:
        values = []
        for raw in self._read(project_id).get("runs", []):
            if layer_id and str(raw.get("layer_id", "")) != str(layer_id):
                continue
            values.append(DerivedRun(**raw))
        return tuple(values)

    def get_run(self, project_id: str, run_id: str) -> DerivedRun | None:
        return next(
            (
                value
                for value in self.list_runs(project_id)
                if value.run_id == str(run_id)
            ),
            None,
        )


class WorkspaceFileService:
    def __init__(self, registry: WorkspaceRegistry) -> None:
        self.registry = registry

    def create_project(
        self,
        *,
        project_id: str,
        project_name: str,
        source_root: Path | str,
        derived_root: Path | str,
    ) -> ProjectWorkspaceBinding:
        name = validate_workspace_name(project_name, field_name="Название проекта")
        source, derived = validate_workspace_roots(source_root, derived_root)
        source_final = source / name
        derived_final = derived / name
        if source_final.exists() or derived_final.exists():
            raise FileExistsError(f"Папка проекта «{name}» уже существует.")
        token = uuid4().hex
        source_stage = source / f".{name}.kraken-{token}"
        derived_stage = derived / f".{name}.kraken-{token}"
        promoted: list[tuple[Path, Path]] = []
        try:
            for category in ("img", "ssc", "prv", "aux"):
                (source_stage / category).mkdir(parents=True, exist_ok=False)
            for category in ("dataset", "result", "vector"):
                (derived_stage / category).mkdir(parents=True, exist_ok=False)
            os.replace(source_stage, source_final)
            promoted.append((source_final, source_stage))
            os.replace(derived_stage, derived_final)
            promoted.append((derived_final, derived_stage))
        except Exception:
            for final, stage in reversed(promoted):
                if final.exists() and not stage.exists():
                    os.replace(final, stage)
            shutil.rmtree(source_stage, ignore_errors=True)
            shutil.rmtree(derived_stage, ignore_errors=True)
            raise
        binding = ProjectWorkspaceBinding(
            project_id=str(project_id),
            project_name=name,
            source_root=str(source),
            derived_root=str(derived),
            source_project_dir=str(source_final),
            derived_project_dir=str(derived_final),
        )
        try:
            self.registry.save_project(binding)
        except Exception:
            self.remove_project_layout(binding)
            raise
        return binding

    @staticmethod
    def remove_project_layout(binding: ProjectWorkspaceBinding) -> None:
        for path in (Path(binding.source_project_dir), Path(binding.derived_project_dir)):
            if path.is_dir() and not path.is_symlink():
                shutil.rmtree(path)

    @staticmethod
    def remove_managed_layer_layout(binding: LayerFileBinding, project: ProjectWorkspaceBinding) -> None:
        if binding.mode is not LayerSourceMode.MANAGED_COPY:
            return
        source_root = Path(project.source_project_dir).resolve()
        for value in (
            binding.image_directory,
            binding.ssc_directory,
            binding.prv_directory,
            binding.aux_directory,
        ):
            if not value:
                continue
            candidate = _contained(source_root, Path(value))
            if candidate.is_dir() and not candidate.is_symlink():
                shutil.rmtree(candidate)

    def scan(self, root: Path | str, *, maximum_frames: int) -> LayerSourceScan:
        return scan_layer_source(root, maximum_frames=maximum_frames)

    def stage_layer_deletion(
        self,
        *,
        project: ProjectWorkspaceBinding,
        binding: LayerFileBinding,
        delete_id: str,
    ) -> LayerDeletionStage:
        source_root = Path(project.source_project_dir).resolve(strict=True)
        derived_root = Path(project.derived_project_dir).resolve(strict=True)
        candidates: list[tuple[Path, Path]] = []
        if binding.mode is LayerSourceMode.MANAGED_COPY:
            for value in (
                binding.image_directory,
                binding.ssc_directory,
                binding.prv_directory,
                binding.aux_directory,
            ):
                if value:
                    source = _contained(source_root, Path(value))
                    relative = source.relative_to(source_root)
                    candidates.append(
                        (source, source_root / "_trash" / delete_id / relative)
                    )
        for kind in DerivedRunKind:
            source = _contained(
                derived_root,
                derived_root / kind.value / binding.layer_name,
            )
            candidates.append(
                (
                    source,
                    derived_root
                    / "_trash"
                    / delete_id
                    / kind.value
                    / binding.layer_name,
                )
            )

        moved: list[tuple[Path, Path]] = []
        try:
            for source, trash in candidates:
                if not source.exists():
                    continue
                if source.is_symlink():
                    raise WorkspaceValidationError(
                        f"При удалении обнаружена недопустимая символическая ссылка: {source}"
                    )
                trash.parent.mkdir(parents=True, exist_ok=True)
                os.replace(source, trash)
                moved.append((source, trash))
        except Exception:
            for source, trash in reversed(moved):
                if trash.exists() and not source.exists():
                    source.parent.mkdir(parents=True, exist_ok=True)
                    os.replace(trash, source)
            raise
        return LayerDeletionStage(
            delete_id=delete_id,
            moves=tuple(moved),
            trash_roots=(
                source_root / "_trash" / delete_id,
                derived_root / "_trash" / delete_id,
            ),
        )

    @staticmethod
    def rollback_layer_deletion(stage: LayerDeletionStage) -> None:
        for source, trash in reversed(stage.moves):
            if trash.exists() and not source.exists():
                source.parent.mkdir(parents=True, exist_ok=True)
                os.replace(trash, source)
        for root in stage.trash_roots:
            shutil.rmtree(root, ignore_errors=True)

    @staticmethod
    def purge_layer_deletion(stage: LayerDeletionStage) -> None:
        for root in stage.trash_roots:
            shutil.rmtree(root, ignore_errors=True)

    def bind_external_layer(
        self,
        *,
        project_id: str,
        layer_id: str,
        layer_name: str,
        image_directory: Path | str,
        ssc_directory: Path | str | None,
        prv_directory: Path | str | None,
        maximum_frames: int,
    ) -> LayerFileBinding:
        name = validate_workspace_name(layer_name, field_name="Название слоя")
        images = Path(image_directory).expanduser().resolve(strict=True)
        if not images.is_dir():
            raise WorkspaceValidationError("Обязательно укажите папку изображений.")
        image_paths = tuple(
            path for path in images.iterdir()
            if path.is_file() and path.suffix.casefold() in {".jpg", ".jpeg", ".bmp", ".png"}
        )
        if not image_paths:
            raise WorkspaceValidationError(
                "В папке нет поддерживаемых изображений JPG, PNG или BMP."
            )
        positions = map_frame_positions(image_paths, maximum=maximum_frames)

        def optional_directory(value: Path | str | None) -> str:
            if value is None or not str(value).strip():
                return ""
            resolved = Path(value).expanduser().resolve(strict=True)
            if not resolved.is_dir():
                raise WorkspaceValidationError(
                    f"Необязательный путь не является папкой: {resolved}"
                )
            return str(resolved)

        binding = LayerFileBinding(
            layer_id=str(layer_id),
            layer_name=name,
            mode=LayerSourceMode.EXTERNAL,
            image_directory=str(images),
            ssc_directory=optional_directory(ssc_directory),
            prv_directory=optional_directory(prv_directory),
            frame_positions=positions,
        )
        self.registry.save_layer(project_id, binding)
        return binding

    def import_layer(
        self,
        *,
        project: ProjectWorkspaceBinding,
        layer_id: str,
        layer_name: str,
        scan: LayerSourceScan,
        conversion: ImageConversionSettings,
        progress: ProgressCallback | None = None,
        cancelled: Event | None = None,
    ) -> LayerFileBinding:
        if not scan.ready:
            raise WorkspaceValidationError(
                "; ".join(scan.issues) or "Сканирование источника не завершено."
            )
        name = validate_workspace_name(layer_name, field_name="Название слоя")
        source_project = Path(project.source_project_dir).resolve(strict=True)
        selected_root = Path(scan.selected_root).resolve(strict=True)
        if _same_or_nested(source_project, selected_root):
            raise WorkspaceValidationError(
                "Источник импорта и папка проекта не могут содержать друг друга."
            )
        destinations = {
            category: _contained(source_project, source_project / category / name)
            for category in ("img", "ssc", "prv", "aux")
        }
        if any(path.exists() for path in destinations.values()):
            raise FileExistsError(f"Папки слоя «{name}» уже существуют в проекте.")
        estimated_bytes = scan.total_bytes + sum(
            scan.file_fingerprints.get(str(Path(value).resolve()), (Path(value).stat().st_size, 0))[0] * 4
            for value in scan.bmp_files
        )
        free_bytes = shutil.disk_usage(source_project).free
        if estimated_bytes > free_bytes:
            raise WorkspaceValidationError(
                f"Недостаточно свободного места: требуется примерно {estimated_bytes:n} байт, "
                f"доступно {free_bytes:n} байт."
            )
        for filename, expected in scan.file_fingerprints.items():
            path = Path(filename)
            try:
                metadata = path.stat()
            except OSError as exc:
                raise WorkspaceValidationError(
                    f"После сканирования исходный файл исчез или недоступен: {path}"
                ) from exc
            if (metadata.st_size, metadata.st_mtime_ns) != tuple(expected):
                raise WorkspaceValidationError(
                    f"После сканирования изменились размер или время файла: {path}"
                )
        stage = source_project / f".import-{uuid4().hex}"
        stage.mkdir()
        for category in destinations:
            (stage / category).mkdir()
        recognized_images = {Path(value).resolve() for value in scan.image_files}
        recognized_ssc = {Path(value).resolve() for value in scan.ssc_files}
        recognized_prv = {Path(value).resolve() for value in scan.prv_files}
        skip_aux = recognized_ssc | recognized_prv | (recognized_images if scan.jpg_files else set())

        tasks: list[tuple[Callable[..., None], tuple[Any, ...], str]] = []
        for raw in scan.jpg_files:
            source = Path(raw)
            target = stage / "img" / source.name
            if conversion.flip_horizontal or conversion.flip_vertical:
                tasks.append((self._convert_image, (source, target, "jpg", conversion), source.name))
            else:
                tasks.append((self._copy_file, (source, target), source.name))
        for raw in scan.bmp_files:
            source = Path(raw)
            extension = ".jpg" if conversion.target_format == "jpg" else ".png"
            tasks.append(
                (self._convert_image, (source, stage / "img" / f"{source.stem}{extension}", conversion.target_format, conversion), source.name)
            )
        for category, values in (("ssc", scan.ssc_files), ("prv", scan.prv_files)):
            for raw in values:
                source = Path(raw)
                tasks.append((self._copy_file, (source, stage / category / source.name), source.name))
        for current_text, directory_names, file_names in os.walk(selected_root, followlinks=False):
            current = Path(current_text)
            for directory_name in directory_names:
                candidate = current / directory_name
                if candidate.is_symlink():
                    raise WorkspaceValidationError(
                        f"Символические ссылки недопустимы: {candidate}"
                    )
                relative_directory = candidate.relative_to(selected_root)
                (stage / "aux" / relative_directory).mkdir(parents=True, exist_ok=True)
            for file_name in file_names:
                source = (current / file_name).resolve()
                if source in skip_aux:
                    continue
                relative = source.relative_to(selected_root)
                tasks.append((self._copy_file, (source, stage / "aux" / relative), str(relative)))
        completed = 0
        lock = Lock()

        def run_task(task: tuple[Callable[..., None], tuple[Any, ...], str]) -> None:
            nonlocal completed
            if cancelled is not None and cancelled.is_set():
                raise InterruptedError("Импорт слоя отменён.")
            function, arguments, label = task
            source = Path(arguments[0]).resolve()
            expected = scan.file_fingerprints.get(str(source))
            if expected is None:
                raise WorkspaceValidationError(
                    f"Файл не входил в результат сканирования: {source}"
                )
            before = source.stat()
            if (before.st_size, before.st_mtime_ns) != tuple(expected):
                raise WorkspaceValidationError(f"Файл изменился во время импорта: {source}")
            function(*arguments)
            after = source.stat()
            if (after.st_size, after.st_mtime_ns) != tuple(expected):
                raise WorkspaceValidationError(f"Файл изменился во время импорта: {source}")
            with lock:
                completed += 1
                if progress is not None:
                    progress(completed, len(tasks), label)

        promoted: list[tuple[Path, Path]] = []
        try:
            with ThreadPoolExecutor(max_workers=min(8, max(2, os.cpu_count() or 2))) as pool:
                futures = [pool.submit(run_task, task) for task in tasks]
                for future in futures:
                    future.result()
            for category, final in destinations.items():
                staged = stage / category
                os.replace(staged, final)
                promoted.append((final, staged))
        except Exception:
            for final, staged in reversed(promoted):
                if final.exists() and not staged.exists():
                    os.replace(final, staged)
            raise
        finally:
            shutil.rmtree(stage, ignore_errors=True)

        binding = LayerFileBinding(
            layer_id=str(layer_id),
            layer_name=name,
            mode=LayerSourceMode.MANAGED_COPY,
            image_directory=str(destinations["img"]),
            ssc_directory=str(destinations["ssc"]),
            prv_directory=str(destinations["prv"]),
            aux_directory=str(destinations["aux"]),
            import_root=scan.selected_root,
            conversion=conversion,
            frame_positions={
                (f"{Path(filename).stem}.jpg" if scan.bmp_files and conversion.target_format == "jpg" else
                 f"{Path(filename).stem}.png" if scan.bmp_files else Path(filename).name): position
                for filename, position in scan.frame_positions.items()
            },
        )
        try:
            self.registry.save_layer(project.project_id, binding)
        except Exception:
            self.remove_managed_layer_layout(binding, project)
            raise
        return binding

    @staticmethod
    def _copy_file(source: Path, target: Path) -> None:
        target.parent.mkdir(parents=True, exist_ok=True)
        before = source.stat()
        shutil.copy2(source, target)
        after = source.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise WorkspaceValidationError(f"Файл изменился во время импорта: {source}")
        if target.stat().st_size != after.st_size:
            raise IOError(f"Размер скопированного файла не совпадает: {source}")

    @staticmethod
    def _convert_image(
        source: Path,
        target: Path,
        target_format: str,
        settings: ImageConversionSettings,
    ) -> None:
        from PIL import Image, ImageOps

        before = source.stat()
        target.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(source) as opened:
            image = ImageOps.exif_transpose(opened)
            if settings.flip_horizontal:
                image = ImageOps.mirror(image)
            if settings.flip_vertical:
                image = ImageOps.flip(image)
            metadata: dict[str, Any] = {}
            if "dpi" in opened.info:
                metadata["dpi"] = opened.info["dpi"]
            if "icc_profile" in opened.info:
                metadata["icc_profile"] = opened.info["icc_profile"]
            temporary = target.with_name(f".{target.name}.{uuid4().hex}.tmp")
            try:
                if target_format == "jpg":
                    if image.mode not in {"L", "RGB"}:
                        image = image.convert("RGB")
                    image.save(
                        temporary,
                        format="JPEG",
                        quality=settings.jpeg_quality,
                        subsampling=settings.jpeg_subsampling,
                        optimize=settings.jpeg_optimize,
                        progressive=settings.jpeg_progressive,
                        **metadata,
                    )
                else:
                    image.save(
                        temporary,
                        format="PNG",
                        compress_level=settings.png_compression,
                        optimize=settings.png_optimize,
                        **metadata,
                    )
                os.replace(temporary, target)
            finally:
                temporary.unlink(missing_ok=True)
        after = source.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise WorkspaceValidationError(f"Файл изменился во время преобразования: {source}")

    def begin_run(
        self,
        *,
        project_id: str,
        layer_id: str,
        layer_name: str,
        kind: DerivedRunKind,
        plugin_id: str,
        operation: str,
    ) -> DerivedRun:
        project = self.registry.get_project(project_id)
        if project is None:
            raise WorkspaceValidationError("Проект не привязан к двухдисковому хранилищу.")
        name = validate_workspace_name(layer_name, field_name="Название слоя")
        run_id = str(uuid4())
        timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        path = Path(project.derived_project_dir) / kind.value / name / f"{timestamp}_{run_id[:8]}"
        path.mkdir(parents=True, exist_ok=False)
        run = DerivedRun(
            run_id=run_id,
            layer_id=layer_id,
            kind=kind,
            state=DerivedRunState.DRAFT,
            path=str(path),
            plugin_id=plugin_id,
            operation=operation,
            created_at=datetime.now(UTC).isoformat(),
        )
        self.registry.save_run(project_id, run)
        return run

    def publish_run(
        self,
        *,
        project_id: str,
        run_id: str,
        output_directory: Path | str,
        provenance: dict[str, Any] | None = None,
    ) -> DerivedRun:
        run = self.registry.get_run(project_id, run_id)
        if run is None:
            raise WorkspaceValidationError("Запуск производных данных не найден.")
        source = Path(output_directory).expanduser()
        if source.is_symlink():
            raise WorkspaceValidationError(
                "Результат плагина не может быть символической ссылкой."
            )
        source = source.resolve(strict=True)
        if not source.is_dir():
            raise WorkspaceValidationError("Результат плагина должен быть папкой.")
        destination = Path(run.path).resolve(strict=True)
        if source != destination:
            stage = destination.parent / f".publish-{run.run_id}"
            if stage.exists():
                raise FileExistsError(f"Временная папка публикации уже существует: {stage}")
            for current_text, directory_names, file_names in os.walk(
                source, followlinks=False
            ):
                current = Path(current_text)
                for name in directory_names:
                    if (current / name).is_symlink():
                        raise WorkspaceValidationError(
                            f"Результат плагина содержит символическую ссылку: {current / name}"
                        )
                for name in file_names:
                    if (current / name).is_symlink():
                        raise WorkspaceValidationError(
                            f"Результат плагина содержит символическую ссылку: {current / name}"
                        )
            shutil.copytree(source, stage, copy_function=shutil.copy2)
            previous = destination.parent / f".draft-{run.run_id}"
            try:
                os.replace(destination, previous)
                os.replace(stage, destination)
            except Exception:
                if previous.exists() and not destination.exists():
                    os.replace(previous, destination)
                raise
            finally:
                shutil.rmtree(stage, ignore_errors=True)
                shutil.rmtree(previous, ignore_errors=True)
        published = replace(
            run,
            state=DerivedRunState.SUCCEEDED,
            path=str(destination),
            provenance=dict(provenance or {}),
        )
        self.registry.save_run(project_id, published)
        return published

    def fail_run(
        self,
        *,
        project_id: str,
        run_id: str,
        error: str,
    ) -> DerivedRun:
        run = self.registry.get_run(project_id, run_id)
        if run is None:
            raise WorkspaceValidationError("Запуск производных данных не найден.")
        failed = replace(
            run,
            state=DerivedRunState.FAILED,
            provenance={**dict(run.provenance), "error": str(error)[:10_000]},
        )
        self.registry.save_run(project_id, failed)
        return failed


__all__ = [
    "WorkspaceFileService",
    "WorkspaceRegistry",
    "validate_workspace_roots",
]
