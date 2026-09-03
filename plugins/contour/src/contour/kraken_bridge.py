"""Kraken Agent bridge for the interactive Contour application.

The bridge intentionally knows nothing about Kraken projects or storage. It
can only resolve files contained by the per-job staging directory supplied by
Kraken Agent.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from kraken_core.plugin_protocol import (
    PluginFrameOutput,
    PluginJobManifest,
    PluginJobOutcome,
    PluginOperation,
    PluginResultManifest,
    WorkspacePluginContextV1,
    WorkspacePluginResultV1,
    safe_relative_path,
)

from .__version__ import __version__
from .infrastructure.runtime_config import config_string

_LOGGER = logging.getLogger(__name__)


JOB_ENV = "KRAKEN_JOB_MANIFEST"
RESULT_ENV = "KRAKEN_RESULT_MANIFEST"
STAGING_ENV = "KRAKEN_STAGING_ROOT"
_MAX_MANIFEST_BYTES = 16 * 1024 * 1024


class KrakenBridgeError(RuntimeError):
    """Raised when a managed plugin job is unsafe or malformed."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_workspace_root(value: str | os.PathLike[str]) -> Path:
    raw = Path(value)
    if raw.is_symlink():
        raise KrakenBridgeError("Kraken staging root must not be a symbolic link")
    try:
        root = raw.resolve(strict=True)
    except OSError as exc:
        raise KrakenBridgeError(f"Kraken staging root is unavailable: {raw}") from exc
    if not root.is_dir():
        raise KrakenBridgeError("Kraken staging root is not a directory")
    return root


def _resolve_direct_child(
    root: Path,
    value: str | os.PathLike[str],
    *,
    must_exist: bool,
) -> Path:
    raw = Path(value)
    if raw.is_symlink():
        raise KrakenBridgeError(f"Symbolic links are not accepted: {raw}")
    try:
        candidate = raw.resolve(strict=must_exist)
    except OSError as exc:
        raise KrakenBridgeError(f"Kraken bridge path is unavailable: {raw}") from exc
    if candidate.parent != root:
        raise KrakenBridgeError(f"Kraken bridge path must be a direct staging child: {raw}")
    return candidate


def _resolve_relative_file(root: Path, relative_path: str) -> Path:
    normalized = safe_relative_path(relative_path)
    cursor = root
    for part in normalized.split("/"):
        cursor = cursor / part
        if cursor.is_symlink():
            raise KrakenBridgeError(f"Staged path contains a symbolic link: {relative_path}")
    try:
        candidate = cursor.resolve(strict=True)
    except OSError as exc:
        raise KrakenBridgeError(f"Staged input is missing: {relative_path}") from exc
    if root not in candidate.parents or not candidate.is_file():
        raise KrakenBridgeError(f"Staged input is not a regular contained file: {relative_path}")
    return candidate


def _atomic_write(path: Path, payload: str) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    if temporary.exists() or temporary.is_symlink():
        raise KrakenBridgeError(f"Temporary result path already exists: {temporary.name}")
    try:
        with temporary.open("xb") as stream:
            stream.write(payload.encode("utf-8"))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


@dataclass(frozen=True, slots=True)
class ContourKrakenSession:
    """Validated, staging-only view of one interactive Contour job."""

    manifest: PluginJobManifest
    staging_root: Path
    job_manifest_path: Path
    result_manifest_path: Path
    input_paths: tuple[Path, ...]
    output_directory: Path

    @classmethod
    def load(
        cls,
        *,
        job_manifest: str | os.PathLike[str],
        result_manifest: str | os.PathLike[str],
        staging_root: str | os.PathLike[str],
    ) -> "ContourKrakenSession":
        root = _resolve_workspace_root(staging_root)
        manifest_path = _resolve_direct_child(root, job_manifest, must_exist=True)
        result_path = _resolve_direct_child(root, result_manifest, must_exist=False)
        if result_path.exists() or result_path.is_symlink():
            raise KrakenBridgeError("Kraken result manifest already exists")
        if not manifest_path.is_file() or manifest_path.stat().st_size > _MAX_MANIFEST_BYTES:
            raise KrakenBridgeError("Kraken job manifest is not a regular, reasonably sized file")
        try:
            manifest = PluginJobManifest.from_json(manifest_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, ValueError) as exc:
            raise KrakenBridgeError(f"Invalid Kraken job manifest: {exc}") from exc
        if manifest.operation not in {
            PluginOperation.VECTORIZE_FRAMES.value,
            PluginOperation.PREPARE_DATASET.value,
        }:
            raise KrakenBridgeError(f"Contour does not support operation {manifest.operation!r}")
        if len(manifest.inputs) > 10_000:
            raise KrakenBridgeError("Contour protocol v1 accepts at most 10000 frames per job")

        inputs: list[Path] = []
        output_stems: set[str] = set()
        for item in manifest.inputs:
            if item.media_type not in {"image/png", "image/tiff", "image/jpeg"}:
                raise KrakenBridgeError(f"Unsupported Contour input media type: {item.media_type}")
            candidate = _resolve_relative_file(root, item.relative_path)
            if _sha256(candidate) != item.sha256:
                raise KrakenBridgeError(f"Staged input checksum mismatch: {item.relative_path}")
            stem_key = candidate.stem.casefold()
            if stem_key in output_stems:
                raise KrakenBridgeError(
                    "Contour job contains duplicate input stems; CIF output mapping would be ambiguous"
                )
            output_stems.add(stem_key)
            inputs.append(candidate)

        output_directory = root / "outputs"
        if output_directory.is_symlink():
            raise KrakenBridgeError("Kraken output directory must not be a symbolic link")
        output_directory.mkdir(exist_ok=True)
        output_directory = output_directory.resolve(strict=True)
        if output_directory.parent != root or not output_directory.is_dir():
            raise KrakenBridgeError("Kraken output directory escapes staging")
        return cls(
            manifest=manifest,
            staging_root=root,
            job_manifest_path=manifest_path,
            result_manifest_path=result_path,
            input_paths=tuple(inputs),
            output_directory=output_directory,
        )

    def build_result(self) -> PluginResultManifest:
        outputs: list[PluginFrameOutput] = []
        missing: list[str] = []
        for index, (frame, source_path) in enumerate(zip(self.manifest.inputs, self.input_paths, strict=True), start=1):
            output_path = self.output_directory / f"{source_path.stem}.cif"
            if not output_path.exists():
                missing.append(frame.frame_id)
                continue
            if output_path.is_symlink():
                raise KrakenBridgeError(f"Contour output is a symbolic link: {output_path.name}")
            resolved = output_path.resolve(strict=True)
            if resolved.parent != self.output_directory or not resolved.is_file():
                raise KrakenBridgeError(f"Contour output escapes staging: {output_path.name}")
            outputs.append(
                PluginFrameOutput(
                    output_id=f"{self.manifest.job_id}:contour:{index}",
                    frame_id=frame.frame_id,
                    relative_path=resolved.relative_to(self.staging_root).as_posix(),
                    sha256=_sha256(resolved),
                    media_type="application/x-cif",
                    role="vector",
                )
            )
        if not outputs:
            raise KrakenBridgeError("No CIF results were saved; nothing can be returned to Kraken")
        outcome = PluginJobOutcome.SUCCEEDED.value if not missing else PluginJobOutcome.PARTIAL.value
        errors = () if not missing else (f"Missing CIF results for {len(missing)} frame(s): {', '.join(missing)}",)
        return PluginResultManifest(
            job_id=self.manifest.job_id,
            outcome=outcome,
            plugin_id="contour",
            plugin_version=__version__,
            outputs=tuple(outputs),
            applied_parameters=dict(self.manifest.parameters),
            errors=errors,
        )

    def write_result(self, result: PluginResultManifest | None = None) -> PluginResultManifest:
        resolved = result or self.build_result()
        if resolved.job_id != self.manifest.job_id:
            raise KrakenBridgeError("Refusing to write a result for another Kraken job")
        if self.result_manifest_path.exists() or self.result_manifest_path.is_symlink():
            raise KrakenBridgeError("Kraken result manifest already exists")
        _atomic_write(self.result_manifest_path, resolved.to_json())
        return resolved

    def attach_return_action(self, window: object) -> None:
        """Install an explicit UI callback without coupling to project storage."""

        from PyQt6.QtGui import QAction, QKeySequence
        from PyQt6.QtWidgets import QMainWindow, QMessageBox

        if not isinstance(window, QMainWindow):
            raise TypeError("Contour Kraken bridge requires a QMainWindow")
        menu = window.menuBar().addMenu("Kraken")
        action = QAction("Вернуть результаты в Kraken", window)
        action.setShortcut(QKeySequence(config_string("shortcuts", "return_to_kraken", "Ctrl+Shift+Return")))

        def return_results() -> None:
            try:
                result = self.build_result()
                if result.outcome == PluginJobOutcome.PARTIAL.value:
                    answer = QMessageBox.question(
                        window,
                        "Kraken",
                        "Сохранены не все CIF. Вернуть частичный результат в Kraken?",
                    )
                    if answer != QMessageBox.StandardButton.Yes:
                        return
                self.write_result(result)
            except Exception as exc:
                _LOGGER.exception("Failed to return the interactive result to Kraken")
                QMessageBox.critical(window, "Kraken", f"Не удалось вернуть результат:\n{exc}")
                return
            QMessageBox.information(
                window,
                "Kraken",
                f"Результат задания {self.manifest.job_id} записан. Contour можно закрыть.",
            )
            window.close()

        action.triggered.connect(return_results)
        menu.addAction(action)
        setattr(window, "_kraken_return_action", action)


@dataclass(frozen=True, slots=True)
class ContourWorkspaceSession:
    """Direct local-filesystem session using the two-root Kraken workspace."""

    context: WorkspacePluginContextV1
    context_path: Path
    input_directory: Path
    output_directory: Path
    result_manifest_path: Path

    @classmethod
    def load(cls, path: str | os.PathLike[str]) -> "ContourWorkspaceSession":
        raw = Path(path)
        if raw.is_symlink():
            raise KrakenBridgeError("Workspace context must not be a symbolic link")
        try:
            context_path = raw.resolve(strict=True)
            context = WorkspacePluginContextV1.read(context_path)
        except (OSError, UnicodeError, ValueError) as exc:
            raise KrakenBridgeError(f"Invalid Kraken workspace context: {exc}") from exc
        if context.plugin_id != "contour":
            raise KrakenBridgeError("Workspace context is not intended for Contour")
        if context.operation not in {
            PluginOperation.VECTORIZE_FRAMES.value,
            PluginOperation.PREPARE_DATASET.value,
        }:
            raise KrakenBridgeError(
                f"Contour does not support operation {context.operation!r}"
            )
        try:
            input_directory = Path(context.input_directories["images"]).resolve(
                strict=True
            )
            output_directory = Path(context.proposed_output_directory).resolve(
                strict=True
            )
        except (KeyError, OSError) as exc:
            raise KrakenBridgeError("Workspace input or output directory is unavailable") from exc
        if (
            input_directory.is_symlink()
            or output_directory.is_symlink()
            or not input_directory.is_dir()
            or not output_directory.is_dir()
        ):
            raise KrakenBridgeError("Workspace paths must be regular directories")
        result_manifest_path = Path(context.result_manifest_path).resolve(strict=False)
        if result_manifest_path.exists() or result_manifest_path.is_symlink():
            raise KrakenBridgeError("Workspace result manifest already exists")
        return cls(
            context,
            context_path,
            input_directory,
            output_directory,
            result_manifest_path,
        )

    def _selected_output_directory(self, window: object) -> Path:
        widget = getattr(window, "_widget", None)
        edit_name = (
            "dataset_dir_edit"
            if self.context.operation == PluginOperation.PREPARE_DATASET.value
            else "output_dir_edit"
        )
        edit = getattr(widget, edit_name, None)
        value = str(edit.text()).strip() if edit is not None else str(self.output_directory)
        candidate = Path(value).expanduser()
        if not candidate.is_absolute() or candidate.is_symlink():
            raise KrakenBridgeError("Result directory must be an absolute regular directory")
        candidate.mkdir(parents=True, exist_ok=True)
        resolved = candidate.resolve(strict=True)
        if not resolved.is_dir():
            raise KrakenBridgeError("Result directory is unavailable")
        return resolved

    def attach_return_action(self, window: object) -> None:
        from PyQt6.QtGui import QAction, QKeySequence
        from PyQt6.QtWidgets import QMainWindow, QMessageBox

        if not isinstance(window, QMainWindow):
            raise TypeError("Contour Kraken bridge requires a QMainWindow")
        menu = window.menuBar().addMenu("Kraken")
        action = QAction("Вернуть результаты в Kraken", window)
        action.setShortcut(QKeySequence(config_string("shortcuts", "return_to_kraken", "Ctrl+Shift+Return")))

        def return_results() -> None:
            try:
                output = self._selected_output_directory(window)
                if self.context.operation == PluginOperation.PREPARE_DATASET.value:
                    outputs = tuple(
                        path
                        for directory in (output / "images", output / "cif")
                        if directory.is_dir()
                        for path in directory.iterdir()
                        if path.is_file()
                    )
                else:
                    outputs = tuple(output.glob("*.cif"))
                if not outputs:
                    raise KrakenBridgeError(
                        "В выбранной папке нет результатов для возврата в Kraken"
                    )
                WorkspacePluginResultV1(
                    run_id=self.context.run_id,
                    plugin_id="contour",
                    operation=self.context.operation,
                    outcome=PluginJobOutcome.SUCCEEDED.value,
                    output_directory=str(output),
                    provenance={
                        "plugin_version": __version__,
                        "file_count": len(outputs),
                    },
                ).write(self.result_manifest_path)
            except Exception as exc:
                _LOGGER.exception("Failed to write the Contour result manifest")
                QMessageBox.critical(
                    window,
                    "Kraken",
                    f"Не удалось вернуть результат:\n{exc}",
                )
                return
            QMessageBox.information(
                window,
                "Kraken",
                "Результаты переданы Kraken. Окно Contour можно закрыть.",
            )
            window.close()

        action.triggered.connect(return_results)
        menu.addAction(action)
        setattr(window, "_kraken_return_action", action)


def _validated_ui_arguments(arguments: Sequence[str]) -> list[str]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--language", choices=("ru", "en"))
    parser.add_argument("--width", type=int)
    parser.add_argument("--height", type=int)
    parser.add_argument("--no-qss", action="store_true")
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--log-file")
    try:
        _, unknown = parser.parse_known_args(list(arguments))
    except SystemExit as exc:
        raise KrakenBridgeError("Invalid Contour UI option in managed mode") from exc
    if unknown:
        raise KrakenBridgeError(
            "Managed Contour jobs accept only display/logging options; unexpected arguments: "
            + " ".join(unknown)
        )
    return list(arguments)


def prepare_contour_launch(
    argv: Sequence[str],
    *,
    environ: Mapping[str, str] | None = None,
) -> tuple[ContourKrakenSession | None, list[str]]:
    """Extract bridge options and return the controlled Contour argv."""

    environment = os.environ if environ is None else environ
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--kraken-job-manifest")
    parser.add_argument("--kraken-result-manifest")
    parser.add_argument("--kraken-staging-root")
    parser.add_argument("--kraken-workspace-context")
    namespace, remaining = parser.parse_known_args(list(argv))
    values = {
        "job_manifest": namespace.kraken_job_manifest or environment.get(JOB_ENV),
        "result_manifest": namespace.kraken_result_manifest or environment.get(RESULT_ENV),
        "staging_root": namespace.kraken_staging_root or environment.get(STAGING_ENV),
    }
    if namespace.kraken_workspace_context:
        if any(values.values()):
            raise KrakenBridgeError(
                "Agent staging and direct workspace modes cannot be combined"
            )
        safe_ui_arguments = _validated_ui_arguments(remaining)
        workspace_session = ContourWorkspaceSession.load(
            namespace.kraken_workspace_context
        )
        destination_option = (
            "--dataset-dir"
            if workspace_session.context.operation
            == PluginOperation.PREPARE_DATASET.value
            else "--output-dir"
        )
        controlled = [
            *safe_ui_arguments,
            "--input-dir",
            str(workspace_session.input_directory),
            destination_option,
            str(workspace_session.output_directory),
        ]
        return workspace_session, controlled
    if not any(values.values()):
        return None, list(argv)
    missing = [name for name, value in values.items() if not value]
    if missing:
        raise KrakenBridgeError("Incomplete Kraken bridge configuration: " + ", ".join(missing))
    safe_ui_arguments = _validated_ui_arguments(remaining)
    session = ContourKrakenSession.load(
        job_manifest=str(values["job_manifest"]),
        result_manifest=str(values["result_manifest"]),
        staging_root=str(values["staging_root"]),
    )
    destination_option = (
        "--dataset-dir"
        if session.manifest.operation == PluginOperation.PREPARE_DATASET.value
        else "--output-dir"
    )
    controlled = [
        *safe_ui_arguments,
        destination_option,
        str(session.output_directory),
        *(str(path) for path in session.input_paths),
    ]
    return session, controlled


__all__ = [
    "ContourKrakenSession",
    "ContourWorkspaceSession",
    "KrakenBridgeError",
    "prepare_contour_launch",
]
