from __future__ import annotations
import numpy as np
from PIL import Image
from fastapi.testclient import TestClient
from PyQt6.QtWidgets import QApplication
from neuralimage.application.dto import MainWindowState, SettingsState
from neuralimage.application.services.workflow_mapper import build_workflow_parameters
from neuralimage.remote.preview import compute_preview, decode_array
from neuralimage.remote.preview_qt import snapshot
from neuralimage.remote.server import create_app
from neuralimage.view.settings_panel import SettingsPanel
from neuralimage.view.augmentation_preview_dialog import AugmentationPreviewDialog


def test_http_preview_matches_local_and_renders(tmp_path, monkeypatch):
    monkeypatch.setenv("NEURALIMAGE_SETTINGS_DIR", str(tmp_path / "settings"))
    app = QApplication.instance() or QApplication([])
    images, labels = tmp_path / "images", tmp_path / "labels"
    images.mkdir()
    labels.mkdir()
    values = np.zeros((48, 48), dtype=np.uint8)
    values[10:30, 12:32] = 200
    Image.fromarray(values).save(images / "a.png")
    Image.fromarray((values > 0).astype(np.uint8) * 255).save(labels / "a.png")
    _, training, _ = build_workflow_parameters(
        MainWindowState(work_mode="train_only", sample_folder=str(images), label_folder=str(labels)),
        SettingsState(color_mode="L", sample_size=(32, 32), train_patch_size=(32, 32)),
    )
    panel = SettingsPanel()
    dialog = AugmentationPreviewDialog(training, panel)
    panel.synthetic_defect_generator_check_box.setChecked(False)
    dialog.show()
    app.processEvents()
    payload = snapshot(dialog)
    local = compute_preview(payload)
    with TestClient(create_app(tmp_path / "server", require_cuda=False, execute=False)) as client:
        response = client.post("/api/v1/preview", json=payload)
        assert response.status_code == 200, response.text
        for index, (expected, received) in enumerate(zip(local, response.json(), strict=True)):
            assert np.array_equal(decode_array(expected), decode_array(received)), (
                index,
                np.max(np.abs(decode_array(expected).astype(float) - decode_array(received))),
            )
    assert dialog.image_preview.pixmap() is not None
    assert not dialog.image_preview.pixmap().isNull()
    assert dialog._original_image_array.size > 0
    dialog.close()


def test_stale_preview_cannot_replace_newer_settings(tmp_path, monkeypatch):
    import threading
    import time
    from neuralimage.remote.preview import encode_array
    from neuralimage.remote import preview_qt
    from neuralimage.remote.client import RemoteClient
    from PyQt6.QtWidgets import QLabel
    app = QApplication.instance() or QApplication([])
    first_started = threading.Event()
    release_first = threading.Event()
    calls = []
    class PreviewView:
        _remote_preview_serial = 0
        _preview_thread = None
        _original_image_array = None
        _panel = None
        def __init__(self):
            self.resample_button = QLabel()
            self.status_label = QLabel()
            self.displays = []
        def _update_preview_mode_label(self):
            pass
        def _update_visible_preview(self):
            self.displays.append(int(self._original_image_array[0, 0]))
    view = PreviewView()
    monkeypatch.setattr(preview_qt, "snapshot", lambda dialog: {"serial": dialog._remote_preview_serial})
    monkeypatch.setattr(preview_qt, "remote_url", lambda: "http://localhost:8765")
    def request(client, method, route, payload):
        calls.append(payload["serial"])
        if len(calls) == 1:
            first_started.set()
            assert release_first.wait(5)
        array = encode_array(np.full((8,8), payload["serial"], dtype=np.uint8))
        return [array] * 4
    monkeypatch.setattr(RemoteClient, "request", request)
    preview_qt.request_preview(view)
    assert first_started.wait(5)
    preview_qt.request_preview(view)
    release_first.set()
    deadline = time.monotonic() + 5
    while view._preview_thread is not None and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(0.01)
    assert calls == [1, 2]
    assert view.displays == [2]
