"""Real-data export tree + wizard step screenshots for the delivery report."""
from __future__ import annotations

import os
import sys
from dataclasses import replace
from pathlib import Path

# Ensure plugin src is importable when run as a script.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from PyQt6.QtWidgets import QApplication, QWizard

from karakal.app.presenter import KarakalPresenter
from karakal.core.grid_anomaly import analyze_grid_frame_path
from karakal.core.grid_error_export import (
    GridErrorExportFrame,
    GridExportLayerSpec,
    export_grid_error_wizard_run,
)
from karakal.ui.grid_export_wizard import ExportLayerOffer, GridErrorExportWizard
from karakal.ui.i18n import Translator
from karakal.ui.ui_constants import GRID_INSPECTION_DEFAULT_ERROR_TYPES, GRID_INSPECTION_ERROR_TYPE_COLORS

DATA_ROOT = Path(os.environ.get("KARAKAL_REAL_DATA") or r"F:\test_test_test")
OUT_ROOT = Path(os.environ.get("KARAKAL_EXPORT_DEMO") or (ROOT / "docs" / "export_wizard_demo"))
FRAMES = (9, 12, 39)
ROLE_DIRS = {
    "1": ("NN_res/cells/Direct_result_test", "Test по единицам"),
    "0": ("NN_res/cells_inv/inv_result_test", "Test по нулям"),
}


def _t(key: str, **kwargs) -> str:
    return Translator("ru").tr(key, **kwargs)


def _analyze(frame: int, role_key: str):
    name = f"TEST_DIRECT_{int(frame):04d}.jpg"
    mask_rel, title = ROLE_DIRS[role_key]
    mask = DATA_ROOT / mask_rel / name
    config = replace(KarakalPresenter._grid_damage_config_from_payload({}), cell_representation="binary")
    result = analyze_grid_frame_path(
        mask,
        frame_id=f"{role_key}:{frame:04d}",
        config=config,
        use_cache=False,
        read_cache=False,
        write_cache=False,
    )
    return name, title, result


def capture_wizard_screens(shot_dir: Path) -> None:
    app = QApplication.instance() or QApplication([])
    offers = (
        ExportLayerOffer("1", "Test по единицам", error_count=42, frame_count=3),
        ExportLayerOffer("0", "Test по нулям", error_count=18, frame_count=3),
    )
    types = tuple(GRID_INSPECTION_DEFAULT_ERROR_TYPES)
    wizard = GridErrorExportWizard(
        _t,
        layers=offers,
        selected_types=types,
        has_selection=False,
        output_dir=OUT_ROOT / "export_cyrillic_папка",
        preview_builder=lambda w: (
            "Karakal_export_YYYY-MM-DD_HHMM\\\n"
            "  PNG_filter-colors\\\n"
            "    Test по единицам\\  (3 файлов)\n"
            "    Test по нулям\\  (3 файлов)"
        ),
    )
    wizard.apply_saved_choices(
        {
            "image_format": "png",
            "color_mode": "filter",
            "layer_keys": ["1", "0"],
            "frame_scope": "all",
            "skip_empty": False,
            "output_dir": str(OUT_ROOT / "export_cyrillic_папка"),
        }
    )
    wizard.show()
    app.processEvents()
    shot_dir.mkdir(parents=True, exist_ok=True)
    names = ("01_format", "02_color", "03_layers", "04_destination")
    for index, name in enumerate(names):
        wizard.setCurrentId(index)
        app.processEvents()
        page = wizard.currentPage()
        target = page if page is not None else wizard
        pix = wizard.grab()
        path = shot_dir / f"{name}.png"
        pix.save(str(path), "PNG")
        print(f"wrote {path}")
    wizard.close()


def run_real_export(export_dir: Path) -> Path:
    selected = tuple(GRID_INSPECTION_DEFAULT_ERROR_TYPES)
    frames: list[GridErrorExportFrame] = []
    layers: list[GridExportLayerSpec] = []
    for role_key, (_rel, title) in ROLE_DIRS.items():
        layers.append(
            GridExportLayerSpec(
                key=role_key,
                title=title,
                folder_name=title,
                source_path=str(DATA_ROOT / ROLE_DIRS[role_key][0]),
                analysis_layer="binary",
            )
        )
        for frame_id in FRAMES:
            name, _title, result = _analyze(frame_id, role_key)
            frames.append(
                GridErrorExportFrame(
                    file_name=name,
                    mask_name=role_key,
                    layer_name="binary",
                    result=result,
                )
            )
    export_dir.mkdir(parents=True, exist_ok=True)
    report = export_grid_error_wizard_run(
        frames,
        export_dir,
        selected,
        layers,
        image_format="png",
        color_mode="filter",
        skip_empty=False,
        type_colors=dict(GRID_INSPECTION_ERROR_TYPE_COLORS),
        algorithm_version="demo",
    )
    assert report.run_dir is not None
    print(f"run_dir={report.run_dir}")
    print(f"written={len(report.written)}")
    for path in report.written:
        print(f"  {path.relative_to(report.run_dir)}")
    return report.run_dir


def main() -> int:
    if not DATA_ROOT.is_dir():
        print(f"missing real data: {DATA_ROOT}")
        return 1
    shots = OUT_ROOT / "screenshots"
    export_parent = OUT_ROOT / "export_cyrillic_папка"
    capture_wizard_screens(shots)
    run_dir = run_real_export(export_parent)
    tree_path = OUT_ROOT / "tree.txt"
    lines = [str(run_dir)]
    for path in sorted(run_dir.rglob("*")):
        rel = path.relative_to(run_dir)
        mark = "/" if path.is_dir() else f" ({path.stat().st_size} bytes)"
        lines.append(f"  {rel}{mark if path.is_dir() else mark}")
    tree_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"tree -> {tree_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
