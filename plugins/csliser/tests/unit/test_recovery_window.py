from uuid import uuid4

import pytest
from PyQt6.QtCore import QSettings
from PyQt6.QtWidgets import QFileDialog, QMessageBox

from csliser.application.file_transfer import _rename_new
from csliser.domain.models import FileOperation, ProcessingConfig, SelectionMode, SourceFolder
from csliser.domain.planner import build_operation_plan
from csliser.infrastructure.removal_receipt import RemovalReceipt
from csliser.infrastructure.settings_store import WindowSettingsStore
from csliser.presentation.qt.window import CSliserWindow


@pytest.mark.parametrize('confirmed', [True, False], ids=['restore-with-qt-worker', 'decline-preserves-quarantine'])
def test_window_recovery_action_uses_confirmation_and_actual_worker(qtbot, tmp_path, monkeypatch, confirmed):
    folder = tmp_path/'source'
    folder.mkdir()
    source = folder/'frame_1.jpg'
    source.write_bytes(b'real recovery payload')
    item, = build_operation_plan(ProcessingConfig((SourceFolder(folder, ('.jpg',)),), '1',
                  selection_mode=SelectionMode.FULL_RANGE, operation=FileOperation.DELETE)).operations
    quarantine = folder/('.'+source.name+'.csliser-remove-'+uuid4().hex)
    with pytest.raises(RuntimeError):
        with RemovalReceipt(item, quarantine):
            _rename_new(source, quarantine)
            raise RuntimeError('controlled interrupted removal')
    window = CSliserWindow(settings_store=WindowSettingsStore(
        lambda: QSettings(str(tmp_path/'settings.ini'), QSettings.Format.IniFormat)))
    qtbot.addWidget(window)
    notices = []
    monkeypatch.setattr(QFileDialog, 'getExistingDirectory', lambda *_args: str(folder))
    monkeypatch.setattr(QMessageBox, 'question', lambda *_args:
        QMessageBox.StandardButton.Yes if confirmed else QMessageBox.StandardButton.No)
    monkeypatch.setattr(QMessageBox, 'information', lambda *_args: notices.append(_args[-1]))
    monkeypatch.setattr(QMessageBox, 'warning', lambda *_args: pytest.fail(str(_args)))
    action, = [a for a in window.menuBar().actions()[0].menu().actions() if 'карантина' in a.text()]
    action.trigger()
    if confirmed:
        qtbot.waitUntil(lambda: window._worker_thread is None, timeout=5000)
        assert window._last_result.ok and window._last_result.completed == 1
        assert source.read_bytes() == b'real recovery payload' and not quarantine.exists()
        assert notices and 'карантина' in notices[0] and not window._busy
    else:
        assert window._worker_thread is None and not source.exists() and quarantine.read_bytes() == b'real recovery payload'
        assert not notices
    window.close()
