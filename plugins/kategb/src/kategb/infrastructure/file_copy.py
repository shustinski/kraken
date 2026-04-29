from __future__ import annotations

import re
import shutil
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from kategb.domain.models import CopyPlan

ProgressCallback = Callable[[int, int, Path], None]


@dataclass(frozen=True, slots=True)
class CopyReport:
    copied_files: tuple[Path, ...]
    missing_frames: tuple[int, ...]


class FrameFileCopier:
    def copy(self, plan: CopyPlan, progress: ProgressCallback | None = None) -> CopyReport:
        copied: list[Path] = []
        missing: set[int] = set()
        total = len(plan.frames) * len(plan.sources)
        done = 0
        for source_index, source in enumerate(plan.sources):
            target_dir = plan.destination / f"{source.role}_{plan.check_name}" if source_index == 0 else plan.destination / f"{source.role}_{plan.check_name}_{source_index}"
            target_dir.mkdir(parents=True, exist_ok=True)
            for sample_index, frame in enumerate(plan.frames):
                src = self._find_frame_file(source.folder, frame)
                done += 1
                if src is None:
                    missing.add(frame)
                    continue
                dst = target_dir / f"{plan.check_name}_{sample_index:0{len(str(len(plan.frames)))}}{src.suffix}"
                shutil.copy2(src, dst)
                if plan.rewrite_cif_references and dst.suffix.lower() == ".cif":
                    self._rewrite_cif_reference(dst, dst.stem)
                copied.append(dst)
                if progress is not None:
                    progress(done, total, src)
        return CopyReport(copied_files=tuple(copied), missing_frames=tuple(sorted(missing)))

    def _find_frame_file(self, folder: Path, frame: int) -> Path | None:
        pattern = re.compile(rf"(^|[^0-9])0*{frame}([^0-9]|$)")
        matches = [path for path in folder.iterdir() if path.is_file() and pattern.search(path.stem)]
        return sorted(matches)[0] if matches else None

    def _rewrite_cif_reference(self, path: Path, new_stem: str) -> None:
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
        rewritten = []
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("( R ") or stripped.startswith("(R "):
                rewritten.append(f"( R {new_stem}.jpg );")
            else:
                rewritten.append(line)
        path.write_text("\n".join(rewritten) + "\n", encoding="utf-8")
