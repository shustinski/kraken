"""Masks and confidence maps of one frame told apart by file name suffixes."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PyQt6.QtCore import QSettings

from karakal.core.analytics import collect_frame_records
from karakal.core.domain import BuildOptions, FolderSpec, ModelSpec
from karakal.core.frame_naming import FrameNaming, validate_frame_naming
from karakal.core.image_io import _grayscale_array_to_qimage


def _image(path: Path, value: int = 255) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    assert _grayscale_array_to_qimage(np.full((8, 8), value, dtype=np.uint8)).save(str(path))


def _frames(folder: Path, names) -> None:
    for name in names:
        _image(folder / name)


def _collect(mask_folder: Path, prob_folder: Path | None = None, **naming) -> dict[str, tuple[int, str]]:
    result = collect_frame_records((ModelSpec("m", "Model", mask_folder, prob_folder=prob_folder),), BuildOptions(**naming))
    return {
        record.key: (record.identity.frame_id, Path(record.model_prob_paths["m"]).name if record.model_prob_paths["m"] else "")
        for record in result.records
    }


def test_mixed_folder_gives_masks_as_frames_and_confidence_beside_them(tmp_path) -> None:
    folder = tmp_path / "net"
    _frames(folder, ["TEST2_01800.jpg", "TEST2_01800_confidence.jpg", "TEST2_01801.jpg", "TEST2_01801_CONFIDENCE.jpg", "TEST2_01802.jpg"])
    expected = {
        "TEST2_01800.jpg": (1800, "TEST2_01800_confidence.jpg"),
        "TEST2_01801.jpg": (1801, "TEST2_01801_CONFIDENCE.jpg"),
        "TEST2_01802.jpg": (1802, ""),
    }
    # Without a confidence folder and with the same folder chosen as confidence folder.
    assert _collect(folder) == expected
    assert _collect(folder, folder) == expected


def test_separate_confidence_folder_with_or_without_suffix(tmp_path) -> None:
    masks, suffixed, plain = tmp_path / "masks", tmp_path / "suffixed", tmp_path / "plain"
    _frames(masks, ["A_0001.jpg", "A_0002.jpg"])
    _frames(suffixed, ["A_0001_confidence.jpg", "A_0002_confidence.jpg"])
    _frames(plain, ["A_0001.jpg", "A_0002.jpg"])
    assert _collect(masks, suffixed) == {"A_0001.jpg": (1, "A_0001_confidence.jpg"), "A_0002.jpg": (2, "A_0002_confidence.jpg")}
    assert _collect(masks, plain) == {"A_0001.jpg": (1, "A_0001.jpg"), "A_0002.jpg": (2, "A_0002.jpg")}


def test_mask_suffix_and_originals_without_it(tmp_path) -> None:
    folder, originals = tmp_path / "net", tmp_path / "src"
    _frames(folder, ["B_0007_mask.jpg", "B_0007_prob.jpg", "notes.jpg"])
    _frames(originals, ["B_0007.jpg"])
    result = collect_frame_records(
        (ModelSpec("m", "Model", folder),),
        BuildOptions(mask_suffix="_mask", confidence_suffix="_prob"),
        original_folder=FolderSpec(path=originals, label="src"),
    )
    assert [record.key for record in result.records] == ["B_0007_mask.jpg"]
    record = result.records[0]
    assert record.identity.frame_id == 7
    assert Path(record.model_prob_paths["m"]).name == "B_0007_prob.jpg"
    assert Path(record.original_path).name == "B_0007.jpg"


def test_empty_confidence_suffix_keeps_every_file_a_frame(tmp_path) -> None:
    folder = tmp_path / "net"
    _frames(folder, ["C_0001.jpg", "C_0001_confidence.jpg"])
    assert set(_collect(folder, confidence_suffix="")) == {"C_0001.jpg", "C_0001_confidence.jpg"}


def test_suffix_validation() -> None:
    assert validate_frame_naming(FrameNaming("", "_confidence")) == ""
    assert validate_frame_naming(FrameNaming("_x", "_X")) == "same"
    assert validate_frame_naming(FrameNaming("", "_conf.jpg")) == "characters"


def test_settings_keep_suffixes(tmp_path) -> None:
    from karakal.infra.services import KarakalSettingsService

    settings = QSettings(str(tmp_path / "k.ini"), QSettings.Format.IniFormat)
    service = KarakalSettingsService(settings)
    assert service.load_frame_naming() == FrameNaming("", "_confidence")
    service.save_frame_naming(FrameNaming(" _mask ", "_prob"))
    assert KarakalSettingsService(settings).load_frame_naming() == FrameNaming("_mask", "_prob")


@pytest.mark.parametrize("mixed", (True, False))
def test_adding_a_mixed_folder_fills_its_confidence_folder(tmp_path, qtbot, monkeypatch, mixed) -> None:
    from PyQt6.QtWidgets import QFileDialog

    from karakal.app.main_window import KarakalWidget
    from karakal.ui.ui_constants import FOLDER_CONFIDENCE_ROLE

    folder = tmp_path / "net"
    _frames(folder, ["D_0001.jpg"] + (["D_0001_confidence.jpg"] if mixed else []))
    widget = KarakalWidget(settings=QSettings(str(tmp_path / "k.ini"), QSettings.Format.IniFormat))
    qtbot.addWidget(widget)
    monkeypatch.setattr(QFileDialog, "getExistingDirectory", lambda *_args, **_kwargs: str(folder))
    widget._presenter._add_folder()
    item = widget.folder_list.item(widget.folder_list.count() - 1)
    assert str(item.data(FOLDER_CONFIDENCE_ROLE) or "") == (str(folder) if mixed else "")
