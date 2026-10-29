"""Administration I/O must leave Qt responsive and survive owner destruction."""
import threading
import os
from pathlib import Path
import subprocess
import sys

import pytest

from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import QFileDialog, QMessageBox, QPushButton

from kraken_hub.presentation.qt.parts.administration_panel import _administration_panel
from tests.test_work_orders import _service, _project_with_frames


def test_backup_runs_outside_gui_and_close_waits_without_destroying_worker(tmp_path, qtbot, monkeypatch):
    service = _service(tmp_path)
    owner, *_ = _project_with_frames(service, tmp_path / "inputs", with_vectors=False)
    started, release = threading.Event(), threading.Event()
    threads, notices, errors, ticks = [], [], [], []
    original = service.export_backup

    def slow(*args, **kwargs):
        threads.append(threading.get_ident())
        started.set()
        release.wait(2)
        return original(*args, **kwargs)

    monkeypatch.setattr(service, "export_backup", slow)
    monkeypatch.setattr(QFileDialog, "getExistingDirectory", lambda *_: str(tmp_path / "backups"))
    monkeypatch.setattr(QMessageBox, "information", lambda *args: notices.append(args))
    monkeypatch.setattr(QMessageBox, "warning", lambda *args: errors.append(args))
    panel = _administration_panel(service, owner)
    qtbot.addWidget(panel)
    panel.show()
    qtbot.waitUntil(lambda: getattr(panel, "_admin_tasks", None) is None or not panel._admin_tasks.busy)
    button = next(b for b in panel.findChildren(QPushButton) if b.text() == "Создать резервную копию")
    try:
        button.click()
        qtbot.waitUntil(started.is_set)
        assert threads == [threads[0]] and threads[0] != threading.get_ident()
        QTimer.singleShot(0, lambda: ticks.append(True))
        qtbot.waitUntil(lambda: bool(ticks))
        assert not button.isEnabled()
        assert not panel.close()
        assert panel.isVisible()
        assert not notices and not errors
    finally:
        release.set()
        runner = getattr(panel, "_admin_tasks", None)
        if runner is not None:
            qtbot.waitUntil(lambda: not runner.busy, timeout=10000)
    assert len(notices) == 1 and not errors and button.isEnabled()
    panel.close()


@pytest.mark.parametrize("method,label", [
    ("scan_integrity", "Проверить целостность"),
    ("import_backup", "Восстановить резервную копию"),
    ("attach_project", "Подключить проект из локального каталога…"),
])
def test_administration_failures_are_reported_on_gui_and_allow_retry(tmp_path, qtbot, monkeypatch, method, label):
    service = _service(tmp_path)
    owner, *_ = _project_with_frames(service, tmp_path / "inputs", with_vectors=False)
    calls, errors, notices = [], [], []

    def failed(*_args, **_kwargs):
        calls.append(threading.get_ident())
        raise OSError("own diagnostic I/O failure")

    monkeypatch.setattr(service, method, failed)
    monkeypatch.setattr(QFileDialog, "getExistingDirectory", lambda *_: str(tmp_path))
    monkeypatch.setattr(QMessageBox, "question", lambda *_: QMessageBox.StandardButton.Yes)
    monkeypatch.setattr(QMessageBox, "warning", lambda *args: errors.append((threading.get_ident(), args)))
    monkeypatch.setattr(QMessageBox, "information", lambda *args: notices.append(args))
    panel = _administration_panel(service, owner)
    qtbot.addWidget(panel)
    qtbot.waitUntil(lambda: not panel._admin_tasks.busy)
    button = next(b for b in panel.findChildren(QPushButton) if b.text() == label)
    for attempt in range(2):
        button.click()
        qtbot.waitUntil(lambda: not panel._admin_tasks.busy)
        assert len(errors) == attempt + 1 and button.isEnabled()
    assert len(calls) == 2 and all(t != threading.get_ident() for t in calls)
    assert all(t == threading.get_ident() and "own diagnostic" in args[2] for t, args in errors)
    assert not notices


def test_restore_button_imports_into_empty_catalog_and_refreshes_it(tmp_path, qtbot, monkeypatch):
    source = _service(tmp_path / "source")
    owner, _performer, project, *_ = _project_with_frames(source, tmp_path / "inputs", with_vectors=False)
    backup = tmp_path / "backup"
    source.export_backup(project.id, backup, principal=owner.principal)
    restored = _service(tmp_path / "restored")
    new_owner = restored.create_initial_account("restore", "Restore", "")
    notices, errors = [], []
    monkeypatch.setattr(QFileDialog, "getExistingDirectory", lambda *_: str(backup))
    monkeypatch.setattr(QMessageBox, "question", lambda *_: QMessageBox.StandardButton.Yes)
    monkeypatch.setattr(QMessageBox, "warning", lambda *args: errors.append(args))
    monkeypatch.setattr(QMessageBox, "information", lambda *args: notices.append(args))
    panel = _administration_panel(restored, new_owner)
    qtbot.addWidget(panel)
    qtbot.waitUntil(lambda: not panel._admin_tasks.busy)
    button = next(b for b in panel.findChildren(QPushButton) if b.text() == "Восстановить резервную копию")
    button.click()
    qtbot.waitUntil(lambda: not panel._admin_tasks.busy, timeout=10000)
    assert not errors and len(notices) == 1
    assert restored.get_project(project.id).name == project.name
    assert len(restored.list_artifact_series(project.id)) == 2
    from PyQt6.QtWidgets import QComboBox
    assert panel.findChild(QComboBox).currentData() == str(project.id)


def test_catalog_failure_can_be_retried_with_refresh_button(tmp_path, qtbot, monkeypatch):
    service = _service(tmp_path)
    owner, *_ = _project_with_frames(service, tmp_path / "inputs", with_vectors=False)
    original = service.list_projects
    calls, errors = [], []

    def load(*args, **kwargs):
        calls.append(threading.get_ident())
        if len(calls) == 1:
            raise OSError("own catalog failure")
        return original(*args, **kwargs)

    monkeypatch.setattr(service, "list_projects", load)
    monkeypatch.setattr(QMessageBox, "warning", lambda *args: errors.append(args))
    panel = _administration_panel(service, owner)
    qtbot.addWidget(panel)
    qtbot.waitUntil(lambda: not panel._admin_tasks.busy)
    assert len(errors) == 1
    button = next(b for b in panel.findChildren(QPushButton) if b.text() == "Обновить каталог")
    button.click()
    qtbot.waitUntil(lambda: not panel._admin_tasks.busy)
    assert len(calls) == 2 and all(t != threading.get_ident() for t in calls)
    from PyQt6.QtWidgets import QComboBox
    assert panel.findChild(QComboBox).count() == 1


@pytest.mark.parametrize("mode", ["delete", "quit"])
def test_running_task_survives_parent_deletion_and_application_quit(tmp_path, mode):
    script = r'''
import threading,time,sys
from PyQt6 import sip
from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import QApplication,QWidget,QLabel
from kraken_hub.presentation.qt.administration_tasks import AdministrationTasks,_RUNNING
app=QApplication([]);owner=QWidget();status=QLabel(owner);received=[]
entered=threading.Event();release=threading.Event();done=threading.Event()
def work():
 entered.set();assert release.wait(5);done.set();return 'done'
runner=AdministrationTasks(owner,status,lambda busy:None)
runner.start('diagnostic',work,lambda result:received.append(result),lambda error:received.append(error))
assert entered.wait(3)
if sys.argv[1]=='delete':
 sip.delete(owner);release.set();deadline=time.monotonic()+5
 while _RUNNING and time.monotonic()<deadline:app.processEvents();time.sleep(.005)
 assert not _RUNNING and done.is_set() and not received
else:
 threading.Timer(.15,release.set).start();QTimer.singleShot(0,app.quit)
 app.exec();assert done.is_set();app.processEvents()
print('retained-worker-passed')
'''
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run([sys.executable, "-c", script, mode], capture_output=True, timeout=15,
                            env=dict(os.environ, PYTHONPATH=str(root / "src"), QT_QPA_PLATFORM="offscreen"))
    assert result.returncode == 0, result.stderr
    assert b"retained-worker-passed" in result.stdout
