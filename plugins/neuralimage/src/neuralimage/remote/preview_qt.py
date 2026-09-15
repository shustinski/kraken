"""Capture widget values on the GUI thread; calculate on a worker."""

from __future__ import annotations
from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtWidgets import QApplication
from .codec import encode
from .preview import encode_array, decode_array
from .preview_options import OPTION_NAMES
from .client import RemoteClient
from .ui import remote_url


def snapshot(dialog):
    def scalar(widget):
        if isinstance(widget, dict):
            return {k: scalar(v) for k, v in widget.items()}
        for method in ("isChecked", "value", "currentData", "currentText"):
            getter = getattr(widget, method, None)
            if callable(getter):
                return getter()
        raise TypeError(f"Unsupported preview control: {type(widget).__name__}")

    options = {}
    for name in OPTION_NAMES:
        value = getattr(dialog._panel, name)
        options[name] = value() if name.startswith("get_") else scalar(value)
    options["full_image"] = dialog.full_image_check_box.isChecked()
    index = dialog._current_sample_index
    indices = {index} if dialog._sample_pairs else set()
    if len(dialog._sample_pairs) > 1 and dialog._panel.mixup_check_box.isChecked():
        partner = (index + 1 + dialog._variant_serial) % len(dialog._sample_pairs)
        indices.add((partner + 1) % len(dialog._sample_pairs) if partner == index else partner)
    return {
        "training": encode(dialog._training_parameters),
        "options": options,
        "names": [p[0].as_posix() for p in dialog._sample_pairs],
        "arrays": {str(i): [encode_array(a) for a in dialog._load_prepared_arrays(i)] for i in indices},
        "index": index,
        "variant": dialog._variant_serial,
        "salt": dialog._resample_salt,
        "cutter_item": getattr(dialog, "_cutter_item_index", 0),
    }


class PreviewThread(QThread):
    ready = pyqtSignal(int, object)
    failed = pyqtSignal(int, str)

    def __init__(self, serial, payload, url):
        super().__init__()
        self.serial, self.payload, self.url = serial, payload, url

    def run(self):
        try:
            if self.url is not None:
                result = RemoteClient(self.url, timeout=10).request("POST", "/preview", self.payload)
            else:
                from .preview import compute_preview

                result = compute_preview(self.payload)
            self.ready.emit(self.serial, tuple(decode_array(a) for a in result))
        except Exception as error:
            self.failed.emit(self.serial, str(error))


def request_preview(dialog):
    serial = getattr(dialog, "_remote_preview_serial", 0) + 1
    dialog._remote_preview_serial = serial
    payload = snapshot(dialog)
    # Coalesce rapid parameter changes; one request is active per dialog.
    dialog._pending_preview = (serial, payload, remote_url())
    if getattr(dialog, "_preview_thread", None) is not None:
        return
    start_pending(dialog)


def start_pending(dialog):
    pending = getattr(dialog, "_pending_preview", None)
    if pending is None:
        return
    dialog._pending_preview = None
    thread = PreviewThread(*pending)
    app = QApplication.instance()
    thread.setParent(app)
    def shutdown():
        thread.requestInterruption()
        thread.wait()
    app.aboutToQuit.connect(shutdown)
    dialog._preview_thread = thread

    def ready(serial, arrays):
        if serial != dialog._remote_preview_serial:
            return
        (
            dialog._original_image_array,
            dialog._original_label_array,
            dialog._augmented_image_array,
            dialog._augmented_label_array,
        ) = arrays
        dialog.resample_button.setEnabled(True)
        dialog._update_preview_mode_label()
        dialog._update_visible_preview()

    def failed(serial, error):
        if serial == dialog._remote_preview_serial:
            dialog.status_label.setText(error)

    def finished():
        app.aboutToQuit.disconnect(shutdown)
        dialog._preview_thread = None
        start_pending(dialog)

    thread.ready.connect(ready)
    thread.failed.connect(failed)
    thread.finished.connect(finished)
    thread.finished.connect(thread.deleteLater)
    thread.start()
