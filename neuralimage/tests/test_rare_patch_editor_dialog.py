import numpy as np
import pytest

pytest.importorskip('PyQt6')

from PIL import Image
from PyQt6.QtGui import QImage
from PyQt6.QtWidgets import QApplication

from lib.rare_patch_masks import (
    collect_matching_sample_label_pairs,
    load_rare_patch_mask,
    prepare_label_folder_for_rare_patch_editor,
    resolve_rare_patch_mask_path,
)
from tests.helpers import make_test_dir
from view.rare_patch_editor_dialog import RarePatchCanvas, RarePatchEditorDialog


@pytest.fixture(scope='module')
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_collect_matching_sample_label_pairs_reports_missing_pairs():
    root = make_test_dir('rare_patch_editor_pairs')
    sample_dir = root / 'samples'
    label_dir = root / 'labels'
    sample_dir.mkdir()
    label_dir.mkdir()

    Image.fromarray(np.zeros((4, 4), dtype=np.uint8), mode='L').save(sample_dir / 'frame_a.png')
    Image.fromarray(np.zeros((4, 4), dtype=np.uint8), mode='L').save(label_dir / 'frame_b.png')

    pairs, error_message = collect_matching_sample_label_pairs(sample_dir, label_dir)

    assert pairs == []
    assert error_message is not None
    assert 'mismatch' in error_message.lower()


def test_prepare_label_folder_for_rare_patch_editor_converts_cif_to_binary_cif():
    root = make_test_dir('rare_patch_editor_cif_labels')
    sample_dir = root / 'samples'
    label_dir = root / 'labels'
    sample_dir.mkdir()
    label_dir.mkdir()

    Image.fromarray(np.zeros((8, 8), dtype=np.uint8), mode='L').save(sample_dir / 'frame.png')
    (label_dir / 'frame.cif').write_text('0 0 S 8 8 \nB 4 4 4 4;;\n', encoding='utf-8')

    resolved_label_dir, error_message = prepare_label_folder_for_rare_patch_editor(label_dir)
    pairs, pair_error = collect_matching_sample_label_pairs(sample_dir, resolved_label_dir)

    assert error_message is None
    assert resolved_label_dir == root / 'binary_cif'
    assert (resolved_label_dir / 'frame.jpg').exists()
    assert pair_error is None
    assert len(pairs) == 1


def test_rare_patch_editor_dialog_saves_mask(qapp):
    root = make_test_dir('rare_patch_editor_dialog')
    sample_dir = root / 'samples'
    label_dir = root / 'labels'
    sample_dir.mkdir()
    label_dir.mkdir()

    image_payload = np.zeros((8, 8, 3), dtype=np.uint8)
    image_payload[..., 1] = 180
    label_payload = np.zeros((8, 8), dtype=np.uint8)
    label_payload[2:6, 2:6] = 255

    Image.fromarray(image_payload, mode='RGB').save(sample_dir / 'frame.png')
    Image.fromarray(label_payload, mode='L').save(label_dir / 'frame.png')

    dialog = RarePatchEditorDialog(sample_dir, label_dir)
    rare_mask = np.zeros((8, 8), dtype=np.uint8)
    rare_mask[1:5, 1:5] = 255
    dialog.canvas.set_rare_mask(rare_mask)
    dialog._save_current_mask()

    saved_path = resolve_rare_patch_mask_path(sample_dir, 'frame')
    loaded_mask = np.asarray(load_rare_patch_mask(sample_dir, 'frame', (8, 8)), dtype=np.uint8)

    assert saved_path.exists()
    assert 'frame.png' in dialog.image_label.text()
    assert int(np.count_nonzero(loaded_mask)) == 16

    dialog.close()


def test_rare_patch_canvas_applies_rectangular_selection_and_erase(qapp):
    widget = RarePatchCanvas()
    image = QImage(10, 10, QImage.Format.Format_RGB888)
    image.fill(0)
    widget.set_base_image(image)

    widget.apply_selection_rect(2, 3, 6, 8)
    assert int(np.count_nonzero(widget.rare_mask())) == 20

    widget.apply_selection_rect(4, 5, 6, 7, erase=True)
    assert int(np.count_nonzero(widget.rare_mask())) == 16


def test_rare_patch_canvas_undoes_last_change(qapp):
    widget = RarePatchCanvas()
    image = QImage(10, 10, QImage.Format.Format_RGB888)
    image.fill(0)
    widget.set_base_image(image)

    widget.apply_selection_rect(1, 1, 5, 5)
    widget.apply_selection_rect(6, 6, 9, 9)
    assert int(np.count_nonzero(widget.rare_mask())) == 25

    widget.undo_last_action()
    assert int(np.count_nonzero(widget.rare_mask())) == 16


def test_rare_patch_editor_dialog_undo_shortcut_reverts_last_change(qapp):
    root = make_test_dir('rare_patch_editor_dialog_undo')
    sample_dir = root / 'samples'
    label_dir = root / 'labels'
    sample_dir.mkdir()
    label_dir.mkdir()

    image_payload = np.zeros((8, 8, 3), dtype=np.uint8)
    label_payload = np.zeros((8, 8), dtype=np.uint8)

    Image.fromarray(image_payload, mode='RGB').save(sample_dir / 'frame.png')
    Image.fromarray(label_payload, mode='L').save(label_dir / 'frame.png')

    dialog = RarePatchEditorDialog(sample_dir, label_dir)
    dialog.canvas.apply_selection_rect(1, 1, 4, 4)
    dialog.canvas.apply_selection_rect(5, 5, 7, 7)
    assert int(np.count_nonzero(dialog.canvas.rare_mask())) == 13

    dialog._undo_shortcut.activated.emit()
    assert int(np.count_nonzero(dialog.canvas.rare_mask())) == 9

    dialog.close()
