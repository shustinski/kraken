"""Bounded own-profile Qt backup/scan/restore acceptance, without native input."""
import threading
import time


def probe_administration(service, session, project, run, app, expected):
    from PyQt6.QtCore import QTimer
    from PyQt6.QtWidgets import QFileDialog, QMessageBox, QPushButton
    from kraken_hub.composition import EmbeddedProjectService
    from kraken_hub.presentation.probes.deployment_matrix_probe import verify_project
    from kraken_hub.presentation.qt.parts.administration_panel import _administration_panel
    from kraken_manager.infrastructure.migration import BundleVerifier

    directory = run / "administration-backups"
    directory.mkdir()
    chosen = [str(directory)]
    entered, release = threading.Event(), threading.Event()
    thread_ids, messages, errors, ticks, panels = [], [], [], [], []
    original = service.export_backup
    saved = (QFileDialog.getExistingDirectory, QMessageBox.information, QMessageBox.warning, QMessageBox.question)

    def export(*args, **kwargs):
        thread_ids.append(threading.get_ident())
        entered.set()
        if not release.wait(10):
            raise TimeoutError("Administration backup occupied the Qt thread")
        return original(*args, **kwargs)

    def wait_until(predicate):
        deadline = time.monotonic() + 20
        while not predicate() and time.monotonic() < deadline:
            app.processEvents()
            time.sleep(.005)
        if not predicate():
            raise TimeoutError("Administration acceptance did not reach its expected state")

    def button(panel, text):
        return next(value for value in panel.findChildren(QPushButton) if value.text() == text)

    class ProbeSecrets:
        def __init__(self):
            self.values = {}

        def get(self, key):
            return self.values.get(key)

        def set(self, key, value):
            self.values[key] = value

    try:
        # Only fixture path selection and message dismissal are substituted.
        # The real buttons, runner, service, files and restore/replay execute.
        QFileDialog.getExistingDirectory = lambda *_args: chosen[0]
        QMessageBox.information = lambda *_args: messages.append(_args[1:])
        QMessageBox.warning = lambda *_args: errors.append(_args[1:])
        QMessageBox.question = lambda *_args: QMessageBox.StandardButton.Yes
        service.export_backup = export
        panel = _administration_panel(service, session)
        panels.append(panel)
        panel.show()
        wait_until(lambda: not panel._admin_tasks.busy)
        button(panel, "Создать резервную копию").click()
        wait_until(entered.is_set)
        if thread_ids != [thread_ids[0]] or thread_ids[0] == threading.get_ident():
            raise RuntimeError("Administration export did not run outside Qt")
        QTimer.singleShot(0, lambda: ticks.append(True))
        wait_until(lambda: bool(ticks))
        if panel.close() or not panel.isVisible() or messages or errors:
            raise RuntimeError("Administration did not guard a busy close")
        if not panel.grab().save(str(run / "administration-backup-busy.png")):
            raise RuntimeError("Administration busy state did not render")
        release.set()
        wait_until(lambda: not panel._admin_tasks.busy)
        archive, = directory.iterdir()
        BundleVerifier().verify(archive).raise_for_errors()
        button(panel, "Проверить целостность").click()
        wait_until(lambda: not panel._admin_tasks.busy)
        if errors or len(messages) != 2:
            raise RuntimeError(f"Administration export/scan failed: {errors}")

        restored = EmbeddedProjectService(run / "administration-restored", secret_store=ProbeSecrets())
        restored_session = restored.create_initial_account("restore-probe", "Restore diagnostic", "")
        chosen[0] = str(archive)
        restore_panel = _administration_panel(restored, restored_session)
        panels.append(restore_panel)
        restore_panel.show()
        wait_until(lambda: not restore_panel._admin_tasks.busy)
        button(restore_panel, "Восстановить резервную копию").click()
        wait_until(lambda: not restore_panel._admin_tasks.busy)
        if errors or len(messages) != 3:
            raise RuntimeError(f"Administration restore failed: {errors}")
        verify_project(restored, expected)
        if restored.get_project(project.id).name != project.name:
            raise RuntimeError("Administration restore did not preserve its project")
        return dict(status="passed", gui_heartbeat=True, export_outside_gui=True,
                    busy_close_guarded=True, scan_passed=True, restored_managed_pairs=len(expected["expected"]),
                    fixture_dialog_selection=True, ordinary_native_input=False)
    finally:
        release.set()
        for panel in panels:
            worker = panel._admin_tasks._worker
            if worker is not None:
                worker.wait()
                app.processEvents()
            panel.close()
            panel.deleteLater()
        del service.export_backup
        (QFileDialog.getExistingDirectory, QMessageBox.information,
         QMessageBox.warning, QMessageBox.question) = saved
        app.processEvents()
