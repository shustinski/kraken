from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence


@dataclass(slots=True)
class StartupConfiguration:
    input_dir: str | None = None
    output_dir: str | None = None
    cif_dir: str | None = None
    pipeline_json: str | None = None
    file_paths: list[str] = field(default_factory=list)
    directory_paths: list[str] = field(default_factory=list)

    @classmethod
    def from_cli(
        cls,
        *,
        input_dir: str | None = None,
        output_dir: str | None = None,
        cif_dir: str | None = None,
        pipeline_json: str | None = None,
        paths: Sequence[str] | None = None,
    ) -> "StartupConfiguration":
        normalized_paths = [str(Path(path)) for path in (paths or [])]
        return cls(
            input_dir=input_dir,
            output_dir=output_dir,
            cif_dir=cif_dir,
            pipeline_json=pipeline_json,
            file_paths=[path for path in normalized_paths if Path(path).is_file()],
            directory_paths=[path for path in normalized_paths if Path(path).is_dir()],
        )

    @property
    def fallback_input_directory(self) -> str | None:
        if self.input_dir:
            return self.input_dir
        if self.directory_paths:
            return self.directory_paths[0]
        return None


@dataclass(slots=True)
class PolygonWidgetApplicationModel:
    language: str | None = None
    width: int = 1680
    height: int = 980
    window_title: str = "Polygon Extraction"
    startup: StartupConfiguration = field(default_factory=StartupConfiguration)

    @property
    def initial_size(self) -> tuple[int, int]:
        return max(640, self.width), max(480, self.height)
