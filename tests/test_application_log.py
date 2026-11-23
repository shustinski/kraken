from __future__ import annotations

import logging

from kraken_core import application_log


def test_application_log_keeps_the_traceback(tmp_path, monkeypatch):
    root = logging.getLogger()
    previous_handlers = list(root.handlers)
    previous_level = root.level
    for handler in previous_handlers:
        root.removeHandler(handler)
    monkeypatch.setattr(application_log.sys, "excepthook", application_log.sys.__excepthook__)
    try:
        path = application_log.configure(tmp_path)
        try:
            raise RuntimeError("matrix mosaic failed")
        except RuntimeError:
            logging.getLogger("kraken.matrix").exception("Matrix background error")
    finally:
        for handler in list(root.handlers):
            handler.close()
            root.removeHandler(handler)
        for handler in previous_handlers:
            root.addHandler(handler)
        root.setLevel(previous_level)

    report = path.read_text(encoding="utf-8")
    assert "matrix mosaic failed" in report
    assert "Traceback" in report
    assert "RuntimeError" in report
