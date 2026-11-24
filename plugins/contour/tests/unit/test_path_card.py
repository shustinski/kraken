from __future__ import annotations

from pathlib import Path


from contour.application.services.path_settings import MAX_RECENT_PATHS, PathSettingsController, merge_recent_paths
from contour.i18n import tr
from contour.ui.path_card import (
    ROLE_CIF,
    ROLE_INPUT,
    ROLE_OUTPUT,
    STATUS_OK,
    PathCard,
    split_path_for_display,
)


def _tr(key: str, default: str = "", **kwargs: object) -> str:
    return tr(key, default=default, language="en", **kwargs)


def _card(role: str = ROLE_INPUT, *, allow_files: bool = False) -> PathCard:
    return PathCard(role, _tr, allow_files=allow_files)


def _menu_texts(card: PathCard) -> list[str]:
    card._rebuild_menu()
    return [action.text() for action in card.menu.actions() if action.text()]


def test_split_path_for_display_separates_name_and_parent(tmp_path: Path) -> None:
    folder = tmp_path / "frames_lot42"
    assert split_path_for_display(str(folder)) == ("frames_lot42", str(tmp_path))
    assert split_path_for_display("") == ("", "")


def test_set_text_updates_display_without_committing(tmp_path: Path) -> None:
    card = _card()
    committed: list[str] = []
    card.pathCommitted.connect(committed.append)

    card.setText(str(tmp_path / "frames"))

    assert card.text() == str(tmp_path / "frames")
    assert card.name_label.full_text() == "frames"
    assert card.parent_label.full_text() == str(tmp_path)
    assert str(tmp_path / "frames") in card.toolTip()
    assert committed == []


def test_empty_card_shows_call_to_action() -> None:
    card = _card(ROLE_OUTPUT)
    assert card.name_label.full_text() == "＋ Result folder is not set"
    assert card.property("empty") is True


def test_manual_edit_commits_once_and_escape_reverts(tmp_path: Path) -> None:
    card = _card()
    card.setText(str(tmp_path))
    committed: list[str] = []
    card.pathCommitted.connect(committed.append)

    card.begin_edit()
    card.editor.setText(str(tmp_path / "other"))
    card.commit_edit()
    card.commit_edit()
    assert committed == [str(tmp_path / "other")]
    assert card.text() == str(tmp_path / "other")

    card.begin_edit()
    card.editor.setText("ignored")
    card.cancel_edit()
    assert card.text() == str(tmp_path / "other")
    assert committed == [str(tmp_path / "other")]

    card.begin_edit()
    card.commit_edit()  # unchanged value does not commit again
    assert committed == [str(tmp_path / "other")]


def test_dropping_a_folder_sets_the_path_and_files_are_forwarded(tmp_path: Path) -> None:
    image = tmp_path / "frame.png"
    image.write_bytes(b"")
    card = _card(allow_files=True)
    committed: list[str] = []
    dropped: list[list[str]] = []
    card.pathCommitted.connect(committed.append)
    card.filesDropped.connect(dropped.append)

    card.handle_dropped_paths([str(tmp_path)])
    card.handle_dropped_paths([str(image)])

    assert committed == [str(tmp_path)]
    assert dropped == [[str(image)]]


def test_folder_only_card_ignores_dropped_files(tmp_path: Path) -> None:
    image = tmp_path / "frame.png"
    image.write_bytes(b"")
    card = _card(ROLE_OUTPUT)
    dropped: list[list[str]] = []
    card.filesDropped.connect(dropped.append)

    card.handle_dropped_paths([str(image)])

    assert dropped == []
    assert card.text() == ""


def test_locked_card_is_read_only(tmp_path: Path) -> None:
    card = _card(allow_files=True)
    card.setText(str(tmp_path))
    committed: list[str] = []
    card.pathCommitted.connect(committed.append)

    card.set_locked(True, "Set by Kraken")
    card.begin_edit()
    card.handle_dropped_paths([str(tmp_path.parent)])

    assert not card.is_editing()
    assert committed == []
    assert not card.acceptDrops()
    assert _menu_texts(card) == ["Open in file manager", "Copy path"]
    assert "Set by Kraken" in card.toolTip()


def test_menu_lists_role_actions_and_recent_paths(tmp_path: Path) -> None:
    card = _card(ROLE_CIF, allow_files=True)
    card.set_recent_provider(lambda: [str(tmp_path)])
    texts = _menu_texts(card)

    assert texts[:3] == ["Choose folder…", "Vector files…", "Recent"]
    recent_menu = next(action.menu() for action in card.menu.actions() if action.text() == "Recent")
    assert [action.text() for action in recent_menu.actions()] == [str(tmp_path)]


def test_status_line_uses_level_marks() -> None:
    card = _card()
    card.set_status("Frames: 3", STATUS_OK)
    assert card.status_label.full_text() == "✓ Frames: 3"
    assert card.status_label.property("level") == STATUS_OK
    card.set_status("")
    assert card.status_label.isHidden()


def test_merge_recent_paths_deduplicates_and_limits(tmp_path: Path) -> None:
    existing = [str(tmp_path / f"d{index}") for index in range(MAX_RECENT_PATHS)]
    merged = merge_recent_paths(existing, str(tmp_path / "d3"))
    assert merged[0] == str(tmp_path / "d3")
    assert len(merged) == MAX_RECENT_PATHS
    assert merged.count(str(tmp_path / "d3")) == 1

    merged = merge_recent_paths(existing, str(tmp_path / "new"))
    assert merged[0] == str(tmp_path / "new")
    assert len(merged) == MAX_RECENT_PATHS
    assert str(tmp_path / f"d{MAX_RECENT_PATHS - 1}") not in merged


def test_widget_path_store_persists_recent_paths_per_role(tmp_path: Path) -> None:
    from contour.infrastructure.settings_store import WidgetPathSettingsStore

    controller = PathSettingsController(WidgetPathSettingsStore())
    controller.remember_recent_path(ROLE_INPUT, tmp_path / "a")
    controller.remember_recent_path(ROLE_INPUT, tmp_path / "b")
    controller.remember_recent_path(ROLE_OUTPUT, tmp_path / "out")

    reloaded = PathSettingsController(WidgetPathSettingsStore())
    assert reloaded.recent_paths(ROLE_INPUT) == [str(tmp_path / "b"), str(tmp_path / "a")]
    assert reloaded.recent_paths(ROLE_OUTPUT) == [str(tmp_path / "out")]
