"""remote_client: extract review archive."""
from __future__ import annotations
from typing import TYPE_CHECKING
import shutil
import stat
import tempfile
import zipfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from kraken_manager.domain.identity import (
    Principal,
)

def _layer_file_binding(payload: Mapping[str, object]):
    from kraken_manager.workspace import ImageConversionSettings, LayerFileBinding

    conversion = payload.get("conversion") or {}
    positions = payload.get("frame_positions") or {}
    return LayerFileBinding(
        layer_id=str(payload["layer_id"]),
        layer_name=str(payload["layer_name"]),
        mode=str(payload["mode"]),
        image_directory=str(payload.get("image_directory") or ""),
        ssc_directory=str(payload.get("ssc_directory") or ""),
        prv_directory=str(payload.get("prv_directory") or ""),
        aux_directory=str(payload.get("aux_directory") or ""),
        import_root=str(payload.get("import_root") or ""),
        conversion=ImageConversionSettings(
            **(dict(conversion) if isinstance(conversion, Mapping) else {})
        ),
        frame_positions=dict(positions) if isinstance(positions, Mapping) else {},
    )

def _workspace_layer_result(service: RemoteServerProjectService, project_id: object, payload: object):
    if not isinstance(payload, Mapping):
        raise TypeError("Server returned an invalid layer")
    layer_payload = payload.get("layer")
    representation_payload = payload.get("representation")
    binding_payload = payload.get("binding")
    if not isinstance(layer_payload, Mapping) or not isinstance(binding_payload, Mapping):
        raise TypeError("Server returned an invalid layer")
    layer = service._layer(project_id, layer_payload)
    representation = (
        service._representation(project_id, layer.id, representation_payload)
        if isinstance(representation_payload, Mapping)
        else None
    )
    return layer, _layer_file_binding(binding_payload), representation

def _review_archive(source: Path, destination: Path) -> None:
    with zipfile.ZipFile(
        destination, "x", compression=zipfile.ZIP_DEFLATED, allowZip64=True
    ) as archive:
        for path in sorted(source.rglob("*")):
            metadata = path.lstat()
            attributes = int(getattr(metadata, "st_file_attributes", 0))
            if stat.S_ISLNK(metadata.st_mode) or attributes & 0x400:
                raise ValueError(f"Review package contains a link: {path}")
            if stat.S_ISREG(metadata.st_mode):
                archive.write(path, path.relative_to(source).as_posix())

def _extract_review_archive(archive: Path, destination: Path) -> None:
    staging = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}-", dir=destination.parent)
    )
    try:
        total = 0
        with zipfile.ZipFile(archive) as package:
            entries = package.infolist()
            if len(entries) > 250_000:
                raise ValueError("Review package has too many entries")
            for entry in entries:
                parts = tuple(part for part in entry.filename.split("/") if part)
                if (
                    not parts
                    or entry.filename.startswith(("/", "\\"))
                    or "\\" in entry.filename
                    or any(part in {".", ".."} or ":" in part for part in parts)
                    or stat.S_ISLNK(entry.external_attr >> 16)
                ):
                    raise ValueError(f"Unsafe review archive member: {entry.filename}")
                total += entry.file_size
                if entry.file_size > 2 * 1024**3 or total > 16 * 1024**3:
                    raise ValueError("Review package exceeds the size limit")
                if (
                    entry.file_size >= 1024 * 1024
                    and entry.file_size / max(1, entry.compress_size) > 200
                ):
                    raise ValueError("Review package compression ratio is unsafe")
                target = staging.joinpath(*parts)
                if entry.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                observed = 0
                with package.open(entry) as input_stream, target.open("xb") as output:
                    while chunk := input_stream.read(1024 * 1024):
                        observed += len(chunk)
                        if observed > entry.file_size:
                            raise ValueError(
                                "Review archive member expanded unexpectedly"
                            )
                        output.write(chunk)
                if observed != entry.file_size:
                    raise ValueError("Review archive member size differs")
        if destination.exists():
            raise FileExistsError(destination)
        staging.replace(destination)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise

class RemoteServerError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        status: int | None = None,
        body: str = "",
        problem: Mapping[str, object] | None = None,
    ) -> None:
        super().__init__(message)
        self.status = status
        self.body = body
        self.problem = dict(problem or {})
        self.code = str(self.problem.get("code", ""))

@dataclass(slots=True)
class RemoteAuth:
    access_token: str
    principal: Principal

if TYPE_CHECKING:
    from kraken_hub.infrastructure.remote.client import RemoteServerProjectService
