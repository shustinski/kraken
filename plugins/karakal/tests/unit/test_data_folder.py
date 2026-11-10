"""Karakal's data folder: projects, profiles and exports in one chosen place."""

from __future__ import annotations

from pathlib import Path

from karakal.core import data_folder
from karakal.core.profiles import list_profiles, read_profile, save_profile


def test_the_data_folder_holds_projects_profiles_and_exports(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("KARAKAL_PROJECTS_DIR", raising=False)
    data_folder.set_data_root(tmp_path / "Каракал")
    try:
        from karakal.core.project_store import default_projects_dir

        assert default_projects_dir() == tmp_path / "Каракал" / data_folder.PROJECTS
        assert data_folder.profiles_dir() == tmp_path / "Каракал" / data_folder.PROFILES
        assert data_folder.exports_dir() == tmp_path / "Каракал" / data_folder.EXPORTS
        project = tmp_path / "Каракал" / data_folder.PROJECTS / "run.karakal"
        assert data_folder.exports_dir(project) == project / data_folder.EXPORTS
    finally:
        data_folder.set_data_root(None)


def test_projects_of_older_versions_are_moved(tmp_path) -> None:
    old = tmp_path / "projects"
    (old / "run.karakal").mkdir(parents=True)
    (old / "run.karakal" / "project.json").write_text("{}", encoding="utf-8")
    target = tmp_path / "data" / "Проекты"
    (target / "run.karakal").mkdir(parents=True)
    (target / "run.karakal" / "project.json").write_text("{}", encoding="utf-8")
    moved = data_folder.move_projects(old, target)
    # A project of the same name is not overwritten: the moved one gets a number.
    assert moved == {str(old / "run.karakal"): str(target / "run (2).karakal")}
    assert (target / "run (2).karakal" / "project.json").is_file() and not (old / "run.karakal").exists()


def test_a_profile_keeps_its_settings(tmp_path) -> None:
    path = save_profile("Ячейки: строгий", {"geometry_sensitivity": 80, "errors_format": "jpg"}, folder=tmp_path)
    assert path.parent == tmp_path and path.suffix == ".json"
    assert [item.stem for item in list_profiles(tmp_path)] == [path.stem]
    assert read_profile(path) == {"geometry_sensitivity": 80, "errors_format": "jpg"}


def test_profile_round_trip_through_the_presenter(tmp_path, qtbot) -> None:
    from PyQt6.QtCore import QSettings

    from karakal.app.main_window import KarakalWidget

    widget = KarakalWidget(settings=QSettings(str(tmp_path / "k.ini"), QSettings.Format.IniFormat))
    qtbot.addWidget(widget)
    presenter = widget._presenter
    presenter._apply_grid_sparse_reach(False, (7, 2.5))
    presenter._batch_controller._page.set_errors_format("tif")
    settings = presenter._profile_settings()
    assert settings["sparse_reach_auto"] is False and (settings["sparse_reach_x"], settings["sparse_reach_y"]) == (7, 2.5)
    assert settings["errors_format"] == "tif"
    path = save_profile("Мой", settings)
    presenter._apply_grid_sparse_reach(True)
    presenter._batch_controller._page.set_errors_format("png")
    assert presenter.load_profile(path)
    assert presenter._grid_sparse_reach_effective() == (7, 2.5)
    assert not widget.grid_sparse_reach_auto_check.isChecked() and widget.grid_sparse_reach_y_spin.value() == 2.5
    assert presenter._batch_controller._page.errors_format() == "tif"
    assert widget.home_page.profile_combo.currentText() == "Мой"


def test_the_home_page_lists_every_project_of_the_data_folder(tmp_path, qtbot) -> None:
    import os

    from PyQt6.QtCore import QSettings

    from karakal.app.main_window import KarakalWidget
    from karakal.core.project_store import find_projects, write_manifest

    root = Path(os.environ["KARAKAL_PROJECTS_DIR"])
    top = root / "прогон.karakal"
    nested = root / "Изделие А" / "2026-11" / "копия.karakal"
    for folder, frames in ((top, 10), (nested, 92)):
        folder.mkdir(parents=True)
        write_manifest(folder, {"frames": frames, "layers": [{"label": "binary_cif"}]})
    (root / "пустая папка").mkdir()
    (root / "без файла.karakal").mkdir()
    # Not looked into: a project's own folder, and nothing below the depth limit.
    assert set(find_projects(root, depth=4)) == {top, nested}
    assert find_projects(root) == [top]

    widget = KarakalWidget(settings=QSettings(str(tmp_path / "k.ini"), QSettings.Format.IniFormat))
    qtbot.addWidget(widget)
    controller = widget._presenter._batch_controller
    controller._show_projects()
    page = widget.home_page
    texts = [page.projects_list.item(row).text() for row in range(page.projects_list.count())]
    assert len(texts) == 2
    nested_text = next(text for text in texts if text.startswith("копия"))
    assert "Изделие А" in nested_text and "92" in nested_text
    # A project copied in later shows up the next time the page is shown.
    later = root / "новый.karakal"
    later.mkdir()
    write_manifest(later, {"frames": 1, "layers": []})
    widget.show_home_page()
    assert page.projects_list.count() == 3
