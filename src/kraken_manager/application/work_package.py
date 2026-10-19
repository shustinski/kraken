"""Place an issued work package on removable media or in a server folder."""

from __future__ import annotations

import hashlib
import io
import json
import mimetypes
import zipfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from kraken_core.plugin_protocol import PluginFrameInput, PluginJobManifest
from kraken_manager.domain.artifacts import deterministic_frame_series_id
from kraken_manager.domain.common import PerformerId, new_uuid
from kraken_manager.domain.project import RepresentationKind
from kraken_manager.domain.selection import FrameRowRange, FrameSelectionV1
from kraken_manager.domain.work_orders import WorkKind, WorkOrder, WorkPlacement


_KIND_DIRECTORY = {
    WorkKind.VECTOR_REVIEW: "Проверка",
    WorkKind.VECTORIZE: "Векторизация",
    WorkKind.TRAIN: "Обучение",
    WorkKind.STITCH: "Сшивка",
}
_IGNORED_RESULT_NAMES = {"kraken-work.json", "frames.json", "job.json"}


@dataclass(frozen=True, slots=True)
class WorkFrameFile:
    x: int
    y: int
    frame_id: str
    version_id: str
    sha256: str
    filename: str
    media_type: str
    relative_path: str
    content: bytes | None


def safe_folder_name(value: str) -> str:
    cleaned = "".join(character if character.isalnum() or character in "-_" else "_" for character in value)
    return cleaned.strip("_") or "project"


def external_package_directory(parent: Path, *, project_name: str, kind: WorkKind, now: datetime) -> Path:
    if not parent.is_dir():
        raise ValueError("Выберите существующий каталог внешнего накопителя")
    destination = parent / (
        f"{_KIND_DIRECTORY[kind]}_{safe_folder_name(project_name)}_{now.astimezone().strftime('%Y%m%d-%H%M%S')}"
    )
    destination.mkdir(parents=False, exist_ok=False)
    return destination


def write_exclusive(path: Path, content: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    with path.open("xb") as stream:
        stream.write(content)
    return hashlib.sha256(content).hexdigest()


def write_json(path: Path, payload: object) -> None:
    write_exclusive(path, json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8"))


def extract_dataset_archive(content: bytes, destination: Path) -> None:
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        for info in archive.infolist():
            if info.is_dir():
                continue
            relative = Path(info.filename)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("Архив выборки содержит небезопасный путь")
            write_exclusive(destination / relative, archive.read(info))


def media_type_for(filename: str, declared: str = "") -> str:
    if declared and declared != "application/octet-stream":
        return declared
    return mimetypes.guess_type(filename)[0] or "application/octet-stream"


def selection_from_coordinates(coordinates: tuple[tuple[int, int], ...]) -> FrameSelectionV1:
    if not coordinates:
        raise ValueError("Выберите хотя бы один кадр")
    ordered = sorted({(int(x), int(y)) for x, y in coordinates})
    return FrameSelectionV1(row_ranges=tuple(FrameRowRange(y=y, x_start=x, x_end=x) for x, y in ordered))


def lookup_frame_files(
    *,
    project,
    representation_id: str,
    selection: FrameSelectionV1,
    active_version,
    role: str,
) -> tuple[WorkFrameFile, ...]:
    files: list[WorkFrameFile] = []
    missing: list[str] = []
    folder = "vectors" if role == "vector" else "images" if role == "image" else "sources"
    for coordinate in selection.iter_coordinates():
        frame_id = coordinate.frame_id(project.id)
        series_id = deterministic_frame_series_id(representation_id, frame_id)
        version = active_version(series_id)
        if version is None:
            missing.append(f"({coordinate.x}, {coordinate.y})")
            continue
        suffix = Path(version.filename).suffix or (".cif" if role == "vector" else ".png")
        filename = f"{coordinate.x}_{coordinate.y}{suffix}"
        files.append(
            WorkFrameFile(
                x=coordinate.x,
                y=coordinate.y,
                frame_id=str(frame_id),
                version_id=str(version.id),
                sha256=version.sha256,
                filename=version.filename,
                media_type=media_type_for(version.filename, version.media_type),
                relative_path=f"{folder}/{filename}",
                content=None,
            )
        )
    if missing:
        label = "CIF" if role == "vector" else "изображение"
        raise ValueError(f"Нет файла ({label}) для кадров: " + ", ".join(missing))
    return tuple(files)


def materialize_files(root: Path, frames: tuple[WorkFrameFile, ...], read_bytes) -> tuple[WorkFrameFile, ...]:
    written: list[WorkFrameFile] = []
    for frame in frames:
        content = read_bytes(frame.version_id)
        digest = write_exclusive(root / frame.relative_path, content)
        if digest != frame.sha256:
            raise ValueError(f"Контрольная сумма не совпала: {frame.relative_path}")
        written.append(frame)
    return tuple(written)


def write_contour_job(
    root: Path,
    *,
    order_id: str,
    project,
    layer_id: str,
    actor_id: str,
    target_representation_id: str,
    frames: tuple[WorkFrameFile, ...],
    operation: str,
) -> None:
    manifest = PluginJobManifest(
        job_id=order_id,
        operation=operation,
        project_id=str(project.id),
        layer_id=str(layer_id),
        actor_id=str(actor_id),
        target_representation_id=str(target_representation_id),
        inputs=tuple(
            PluginFrameInput(
                frame_id=frame.frame_id,
                x=frame.x,
                y=frame.y,
                artifact_version_id=frame.version_id,
                sha256=frame.sha256,
                media_type=frame.media_type,
                relative_path=frame.relative_path,
            )
            for frame in frames
        ),
        parameters={"project_width": project.width, "input_transport": "files"},
    )
    write_json(root / "job.json", manifest.to_dict())


def write_work_manifest(root: Path, order: WorkOrder, frames: tuple[WorkFrameFile, ...]) -> None:
    write_json(
        root / "kraken-work.json",
        {
            "schema": 1,
            "id": order.id,
            "kind": order.kind.value,
            "kind_title": order.kind.title,
            "placement": order.placement.value,
            "placement_title": order.placement.title,
            "transport": order.transport,
            "project_id": str(order.project_id),
            "layer_id": str(order.layer_id),
            "assignee_id": str(order.assignee_id),
            "instructions": order.instructions,
            "capability": order.capability,
            "launch_mode": order.launch_mode,
            "frames": [
                {"x": frame.x, "y": frame.y, "path": frame.relative_path, "frame_id": frame.frame_id}
                for frame in frames
            ],
        },
    )
    write_json(
        root / "frames.json",
        [{"x": frame.x, "y": frame.y, "path": frame.relative_path} for frame in frames],
    )


def package_has_result(destination: Path, kind: WorkKind) -> bool:
    if not destination.is_dir():
        return False
    if kind is WorkKind.VECTORIZE:
        return any((destination / "outputs").glob("*.cif"))
    if kind is WorkKind.TRAIN:
        dataset = destination / "dataset"
        return any(path.is_file() and path.suffix.casefold() in {".onnx", ".pt", ".pth", ".h5", ".keras"} for path in dataset.rglob("*"))
    if kind is WorkKind.STITCH:
        return any(
            path.is_file()
            and path.name not in _IGNORED_RESULT_NAMES
            and "images" not in path.parts
            and "sources" not in path.parts
            and "vectors" not in path.parts
            for path in destination.rglob("*")
        )
    return False


def resolve_transport(*, placement: WorkPlacement, kind: WorkKind, remote: bool, requested: str | None) -> str:
    # Server storage is allocated automatically. Callers never pick its directory and reach it by HTTP.
    if placement is WorkPlacement.SERVER and (remote or kind is not WorkKind.VECTOR_REVIEW):
        return "http"
    if requested is not None:
        if requested not in {"files", "http"}:
            raise ValueError("Способ доступа должен быть files или http")
        return requested
    return "files"


def new_order_id() -> str:
    return new_uuid()


def require_active_performer(performer, assignee_id: str) -> PerformerId:
    if performer is None or not performer.active:
        raise ValueError("Исполнитель не найден или архивирован")
    return PerformerId(str(assignee_id))


def representation_of(representations, representation_id: str, kind: RepresentationKind):
    found = next((item for item in representations if str(item.id) == str(representation_id)), None)
    if found is None or found.kind is not kind:
        return None
    return found


@dataclass(frozen=True, slots=True)
class WorkIssueRequest:
    project: object
    layer: object
    kind: WorkKind
    placement: WorkPlacement
    assignee_id: str
    created_by: str
    coordinates: tuple[tuple[int, int], ...]
    image_representation_id: str
    vector_representation_id: str
    instructions: str
    due_at: datetime | None
    destination_parent: Path | None
    server_directory_for: object
    idempotency_key: str
    transport: str | None
    remote: bool
    now: datetime


def latest_dataset_version(series_list, versions_of):
    candidates = []
    for series in series_list:
        for version in versions_of(series.id):
            parameters = version.parameters or {}
            if version.media_type == "application/zip" and parameters.get("plugin_output_role") == "dataset":
                candidates.append(version)
    if not candidates:
        return None
    return max(candidates, key=lambda item: item.created_at)


def place_work_order(request: WorkIssueRequest, *, active_version, read_bytes, dataset_version) -> WorkOrder:
    """Create the package directory and the order describing it. Persistence is the caller's job."""

    kind = WorkKind(request.kind)
    placement = WorkPlacement(request.placement)
    transport = resolve_transport(
        placement=placement, kind=kind, remote=request.remote, requested=request.transport
    )
    if kind is WorkKind.TRAIN:
        selection = (
            selection_from_coordinates(request.coordinates)
            if request.coordinates
            else FrameSelectionV1(row_ranges=(FrameRowRange(y=1, x_start=1, x_end=1),))
        )
    else:
        selection = selection_from_coordinates(request.coordinates)
    selection.validate_bounds(width=request.project.width, height=request.project.height)
    order_id = new_order_id()
    if placement is WorkPlacement.EXTERNAL:
        if request.destination_parent is None:
            raise ValueError("Выберите каталог внешнего накопителя")
        destination = external_package_directory(
            request.destination_parent,
            project_name=request.project.name,
            kind=kind,
            now=request.now,
        )
    else:
        destination = Path(request.server_directory_for(order_id))
        destination.mkdir(parents=True, exist_ok=False)

    image_frames: tuple[WorkFrameFile, ...] = ()
    vector_frames: tuple[WorkFrameFile, ...] = ()
    dataset_id = ""
    capability = ""
    launch_mode = "external-tool"
    if kind is WorkKind.TRAIN:
        version = dataset_version()
        if version is None:
            raise ValueError("Сначала опубликуйте выборку с изображениями и CIF")
        dataset_id = str(version.id)
        capability = "dataset.model.train.v1"
        launch_mode = "plugin-http" if transport == "http" else "neural-dir"
        if transport == "files":
            extract_dataset_archive(read_bytes(version.id), destination / "dataset")
            images = destination / "dataset" / "images"
            labels = destination / "dataset" / "cif"
            if not images.is_dir() or not labels.is_dir() or not any(images.iterdir()) or not any(labels.iterdir()):
                raise ValueError("Архив выборки не содержит папки images и cif")
    else:
        if not request.image_representation_id:
            raise ValueError("Выберите репрезентацию изображений")
        image_frames = lookup_frame_files(
            project=request.project,
            representation_id=request.image_representation_id,
            selection=selection,
            active_version=active_version,
            role="image" if kind is not WorkKind.STITCH else "source",
        )
        if kind is WorkKind.VECTOR_REVIEW:
            if not request.vector_representation_id:
                raise ValueError("Выберите связанные CIF")
            vector_frames = lookup_frame_files(
                project=request.project,
                representation_id=request.vector_representation_id,
                selection=selection,
                active_version=active_version,
                role="vector",
            )
            capability = "frames.vectorize.v1"
            launch_mode = "contour-job"
        elif kind is WorkKind.VECTORIZE:
            capability = "frames.vectorize.v1"
            launch_mode = "plugin-http" if transport == "http" else "contour-job"
        else:
            launch_mode = "external-tool"
        if transport == "files":
            materialize_files(destination, image_frames, read_bytes)
            if vector_frames:
                materialize_files(destination, vector_frames, read_bytes)
            if kind is WorkKind.VECTORIZE:
                write_contour_job(
                    destination,
                    order_id=order_id,
                    project=request.project,
                    layer_id=str(request.layer.id),
                    actor_id=request.created_by,
                    target_representation_id=request.vector_representation_id or request.image_representation_id,
                    frames=image_frames,
                    operation=capability,
                )
                (destination / "outputs").mkdir(exist_ok=True)

    order = WorkOrder.create(
        project_id=request.project.id,
        layer_id=request.layer.id,
        kind=kind,
        placement=placement,
        assignee_id=PerformerId(str(request.assignee_id)),
        created_by=request.created_by,
        destination=str(destination),
        frame_count=0 if kind is WorkKind.TRAIN else len(tuple(selection.iter_coordinates())),
        selection=selection,
        instructions=request.instructions,
        due_at=request.due_at,
        transport=transport,
        image_representation_id=request.image_representation_id,
        vector_representation_id=request.vector_representation_id,
        dataset_version_id=dataset_id,
        capability=capability,
        launch_mode=launch_mode,
        idempotency_key=request.idempotency_key,
        order_id=order_id,
        created_at=request.now,
    )
    write_work_manifest(destination, order, image_frames if kind is not WorkKind.TRAIN else ())
    return order
