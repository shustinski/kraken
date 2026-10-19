"""Runtime references must remain available after splitting desktop modules."""

import numpy as np
import pytest
from PyQt6.QtWidgets import QLabel

from neuralimage.application.dto import MainWindowState
from neuralimage.infrastructure.config.state_store import QSettingsStateStore
from neuralimage.lib.data_interfaces import (
    normalize_confidence_save_mode,
    normalize_multi_gpu_mode,
    normalize_patch_batch_sync_mode,
    normalize_scheduler_name,
    normalize_work_mode,
)
from neuralimage.view.settings_panel_parts.optional_config_value import OptionalConfigValueMixin
from neuralimage.view.main_window_parts.update_memory_runtime_label import UpdateMemoryRuntimeLabelMixin


def test_normalization_aliases_are_available():
    assert normalize_work_mode("recognintion_only") == "recognition_only"
    assert normalize_confidence_save_mode("grayscale") == "separate_grayscale"
    assert normalize_scheduler_name("steplr") == "step_lr"
    assert normalize_multi_gpu_mode("dp") == "dataparallel"
    assert normalize_patch_batch_sync_mode(None) == "patch_and_batch"


def test_main_window_state_roundtrip_after_module_split(tmp_path, monkeypatch):
    monkeypatch.setenv("NEURALIMAGE_SETTINGS_DIR", str(tmp_path))
    store = QSettingsStateStore()
    state = MainWindowState()
    state.work_mode = "recognition_only"
    store.save_main_window_state(state)
    assert store.load_main_window_state().work_mode == state.work_mode


def test_optional_settings_can_copy_payload_and_remove_defaults():
    original = {"nested": {"values": [1, 2]}}
    copied = OptionalConfigValueMixin._clone_plain_payload(original)
    copied["nested"]["values"].append(3)
    assert original == {"nested": {"values": [1, 2]}}
    OptionalConfigValueMixin._set_optional_config_value(copied, ("nested", "enabled"), True, False)
    assert copied["nested"]["enabled"] is True
    OptionalConfigValueMixin._set_optional_config_value(copied, ("nested", "enabled"), False, False)
    assert "enabled" not in copied["nested"]


@pytest.mark.parametrize("channels", [1, 3])
def test_preview_accepts_non_contiguous_numpy_buffers(qtbot, channels):
    pixels = np.full((8, 12, channels), 47, dtype=np.uint8)
    pixels = pixels[:, ::2]
    if channels == 1:
        pixels = pixels[:, :, 0]
    assert not pixels.flags.c_contiguous
    label = QLabel()
    qtbot.addWidget(label)
    UpdateMemoryRuntimeLabelMixin._set_preview_image(label, pixels)
    image = label.pixmap().toImage()
    assert not image.isNull()
    assert image.pixelColor(0, 0).red() == 47
