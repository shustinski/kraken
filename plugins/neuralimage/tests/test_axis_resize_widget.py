import pytest

pytest.importorskip("PyQt6")

from PyQt6.QtCore import QPoint, QPointF, QSize, Qt
from PyQt6.QtGui import QWheelEvent
from PyQt6.QtWidgets import QApplication, QSpinBox

from neuralimage.view.axis_resize_widget import AxisResizeWidget
from neuralimage.view.settings_panel import SettingsPanel


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def test_defaults_and_grid_positions(qapp):
    widget = AxisResizeWidget()

    assert widget.target_size() == QSize(256, 256)
    assert widget.size_lock_button.isChecked()
    assert widget.size_lock_button.text() == ""
    assert widget.width_spinbox.isEnabled()
    assert not widget.height_spinbox.isEnabled()
    assert widget.grid_layout.itemAtPosition(0, 0).widget() is widget.preview
    assert widget.grid_layout.itemAtPosition(0, 1).widget() is widget.height_container
    assert widget.grid_layout.itemAtPosition(1, 0).widget() is widget.width_container
    assert widget.grid_layout.itemAtPosition(1, 1).widget() is widget.size_lock_button


def test_spinboxes_are_buttonless_and_badges_are_present(qapp):
    widget = AxisResizeWidget()

    assert widget.width_spinbox.buttonSymbols() == QSpinBox.ButtonSymbols.NoButtons
    assert widget.height_spinbox.buttonSymbols() == QSpinBox.ButtonSymbols.NoButtons
    badge_texts = {label.text() for label in widget.findChildren(type(widget.preview))}
    assert {"↔", "↕", "px"}.issubset(badge_texts)


def test_spinbox_values_do_not_change_with_mouse_wheel(qapp):
    widget = AxisResizeWidget()
    widget.size_lock_button.setChecked(False)
    widget.width_spinbox.setValue(320)
    widget.height_spinbox.setValue(192)

    for spinbox in (widget.width_spinbox, widget.height_spinbox):
        before = spinbox.value()
        event = QWheelEvent(
            QPointF(5, 5),
            QPointF(5, 5),
            QPoint(),
            QPoint(0, 120),
            Qt.MouseButton.NoButton,
            Qt.KeyboardModifier.NoModifier,
            Qt.ScrollPhase.ScrollUpdate,
            False,
        )
        QApplication.sendEvent(spinbox, event)
        assert spinbox.value() == before


def test_widget_applies_localized_texts_without_changing_values(qapp):
    widget = AxisResizeWidget()
    widget.apply_texts(
        {
            "preview": "Предпросмотр",
            "width_input": "Ширина",
            "height_input": "Высота",
            "width_axis": "Горизонталь",
            "height_axis": "Вертикаль",
            "pixels": "Пиксели",
            "unit": "пкс",
            "linked": "Размеры связаны",
            "unlinked": "Размеры независимы",
            "height_linked": "Высота равна ширине",
        }
    )

    assert widget.target_size() == QSize(256, 256)
    assert widget.preview.accessibleName() == "Предпросмотр"
    assert widget.width_spinbox.toolTip() == "Ширина"
    assert widget.height_spinbox.toolTip() == "Высота равна ширине"
    assert widget.width_axis_badge.toolTip() == "Горизонталь"
    assert widget.height_unit_badge.text() == "пкс"
    assert widget.size_lock_button.toolTip() == "Размеры связаны"


def test_settings_panel_retranslates_axis_widgets(qapp):
    panel = SettingsPanel()

    panel.set_ui_language("en")
    assert panel.train_patch_size_widget.width_spinbox.toolTip() == "Width in pixels"
    assert "linked" in panel.train_patch_size_widget.size_lock_button.toolTip()

    panel.set_ui_language("ru")
    assert panel.train_patch_size_widget.width_spinbox.toolTip() == "Ширина в пикселях"
    assert "связаны" in panel.train_patch_size_widget.size_lock_button.toolTip()


def test_linked_values_stay_equal_without_recursive_emission(qapp):
    widget = AxisResizeWidget()
    emissions: list[tuple[int, int]] = []
    widget.sizeChanged.connect(lambda width, height: emissions.append((width, height)))

    widget.width_spinbox.setValue(512)
    assert widget.target_size() == QSize(512, 512)
    assert emissions == [(512, 512)]

    emissions.clear()
    widget.height_spinbox.setValue(320)
    assert widget.target_size() == QSize(320, 320)
    assert emissions == [(320, 320)]


def test_set_target_size_unlocks_unequal_dimensions(qapp):
    widget = AxisResizeWidget()
    emissions: list[tuple[int, int]] = []
    widget.sizeChanged.connect(lambda width, height: emissions.append((width, height)))

    widget.set_target_size(1024, 768)

    assert widget.target_size() == QSize(1024, 768)
    assert not widget.size_lock_button.isChecked()
    assert widget.height_spinbox.isEnabled()
    assert emissions == [(1024, 768)]
    widget.width_spinbox.setValue(512)
    assert widget.target_size() == QSize(512, 768)


def test_unlocked_values_are_independent(qapp):
    widget = AxisResizeWidget()
    widget.size_lock_button.setChecked(False)

    assert widget.height_spinbox.isEnabled()

    widget.width_spinbox.setValue(640)
    widget.height_spinbox.setValue(360)

    assert widget.target_size() == QSize(640, 360)

    widget.size_lock_button.setChecked(True)

    assert widget.target_size() == QSize(640, 640)
    assert not widget.height_spinbox.isEnabled()


def test_preview_responds_to_square_landscape_and_portrait_values(qapp):
    widget = AxisResizeWidget()
    widget.size_lock_button.setChecked(False)

    square = widget.preview.size()
    widget.width_spinbox.setValue(512)
    widget.height_spinbox.setValue(256)
    landscape = widget.preview.size()
    widget.width_spinbox.setValue(256)
    widget.height_spinbox.setValue(512)
    portrait = widget.preview.size()

    assert square.width() == square.height()
    assert landscape.width() > landscape.height()
    assert portrait.height() > portrait.width()
    assert widget.width_container.width() == portrait.width()
    assert widget.height_container.height() == portrait.height()


@pytest.mark.parametrize("width,height", [(0, 256), (256, 0), (-1, 256)])
def test_target_dimensions_must_be_positive(qapp, width, height):
    with pytest.raises(ValueError, match="target dimensions must be positive"):
        AxisResizeWidget(target_width=width, target_height=height)


def test_settings_panel_uses_axis_widget_and_preserves_control_aliases(qapp):
    panel = SettingsPanel()

    assert isinstance(panel.synthetic_image_size_widget, AxisResizeWidget)
    assert panel.synthetic_image_width_spinbox is panel.synthetic_image_size_widget.width_spinbox
    assert panel.synthetic_image_height_spinbox is panel.synthetic_image_size_widget.height_spinbox
    assert panel.synthetic_image_size_widget.target_size() == QSize(256, 256)
    assert isinstance(panel.train_patch_size_widget, AxisResizeWidget)
    assert panel.train_patch_x_size is panel.train_patch_size_widget.width_spinbox
    assert panel.train_patch_y_size is panel.train_patch_size_widget.height_spinbox
    assert panel._field_rows[panel.train_patch_size_widget] is panel.train_patch_size_groupbox
    assert isinstance(panel.recognition_patch_size_widget, AxisResizeWidget)
    assert panel.recognition_patch_x_size is panel.recognition_patch_size_widget.width_spinbox
    assert panel.recognition_patch_y_size is panel.recognition_patch_size_widget.height_spinbox
    assert panel._field_rows[panel.recognition_patch_size_widget] is panel.recognition_patch_size_groupbox
    assert 'train_patch_size' not in panel._desc_labels
    assert 'recognition_patch_size' not in panel._desc_labels
    assert panel.train_patch_size_groupbox.title().strip()
    assert panel.recognition_patch_size_groupbox.title().strip()
    assert isinstance(panel.recognition_tab_patch_size_widget, AxisResizeWidget)
    assert panel.recognition_page_layout.indexOf(panel.recognition_tab_patch_size_groupbox) != -1
    assert panel.recognition_tab_patch_size_groupbox.isAncestorOf(
        panel.recognition_tab_patch_size_widget
    )
    assert panel.recognition_tab_patch_size_groupbox.title() == panel.recognition_patch_size_groupbox.title()
    assert isinstance(panel.random_patch_min_size_widget, AxisResizeWidget)
    assert isinstance(panel.random_patch_max_size_widget, AxisResizeWidget)


def test_recognition_tab_patch_size_copy_stays_synchronized(qapp):
    panel = SettingsPanel()

    panel.recognition_tab_patch_size_widget.set_target_size(384, 192)
    assert panel.recognition_patch_size_widget.target_size() == QSize(384, 192)
    assert panel.train_patch_size_widget.target_size() == QSize(384, 192)

    panel.sync_patch_sizes_check_box.setChecked(False)
    panel.recognition_tab_patch_size_widget.set_target_size(640, 320)
    assert panel.recognition_patch_size_widget.target_size() == QSize(640, 320)
    assert panel.train_patch_size_widget.target_size() == QSize(384, 192)

    panel.recognition_patch_size_widget.set_target_size(512, 256)
    assert panel.recognition_tab_patch_size_widget.target_size() == QSize(512, 256)


def test_patch_group_layout_follows_sync_and_random_modes(qapp):
    panel = SettingsPanel()
    panel.connect_internal_signals()

    assert panel.sync_patch_sizes_check_box.isChecked()
    assert panel.recognition_patch_size_groupbox.isHidden()
    assert panel.train_patch_size_widget.isHidden() is False
    assert panel.random_patch_widgets_container.isHidden()

    panel.sync_patch_sizes_check_box.setChecked(False)
    train_position = panel.patch_size_groups_layout.getItemPosition(
        panel.patch_size_groups_layout.indexOf(panel.train_patch_size_groupbox)
    )
    recognition_position = panel.patch_size_groups_layout.getItemPosition(
        panel.patch_size_groups_layout.indexOf(panel.recognition_patch_size_groupbox)
    )
    assert train_position[:2] == (0, 0)
    assert recognition_position[:2] == (0, 1)
    assert panel.recognition_patch_size_groupbox.isHidden() is False

    panel.random_patch_size_check_box.setChecked(True)
    train_position = panel.patch_size_groups_layout.getItemPosition(
        panel.patch_size_groups_layout.indexOf(panel.train_patch_size_groupbox)
    )
    recognition_position = panel.patch_size_groups_layout.getItemPosition(
        panel.patch_size_groups_layout.indexOf(panel.recognition_patch_size_groupbox)
    )
    assert not panel.sync_patch_sizes_check_box.isChecked()
    assert not panel.sync_patch_sizes_check_box.isEnabled()
    assert panel.train_patch_size_widget.isHidden()
    assert panel.random_patch_widgets_container.isHidden() is False
    assert train_position == (0, 0, 1, 2)
    assert recognition_position == (1, 0, 1, 2)


def test_settings_panel_loads_non_square_size_as_independent_dimensions(qapp):
    panel = SettingsPanel()
    panel.set_synthetic_defect_generator_config({"image_size_xy": [1024, 768]})

    assert panel.synthetic_image_size_widget.target_size() == QSize(1024, 768)
    assert not panel.synthetic_image_size_widget.size_lock_button.isChecked()
    panel.synthetic_image_width_spinbox.setValue(512)
    assert panel.synthetic_image_size_widget.target_size() == QSize(512, 768)


def test_visible_patch_widgets_keep_independent_dimensions_and_sync(qapp):
    panel = SettingsPanel()
    panel.connect_internal_signals()
    panel.train_patch_size_widget.set_target_size(320, 160)

    panel.train_patch_x_size.setValue(640)

    assert panel.train_patch_size_widget.target_size() == QSize(640, 160)
    assert panel.recognition_patch_size_widget.target_size() == QSize(640, 160)
