import os
from pathlib import Path
import subprocess
import sys

from PyQt6.QtWidgets import QPushButton
from PyQt6.QtCore import Qt
import pytest

from kraken_hub.legacy_work_order_dialog import LegacyWorkOrderMigrationDialog
from kraken_hub.manager_app_parts.administration_panel import _administration_panel

pytest_plugins = ["tests.test_legacy_work_order_migration"]


def test_reviewed_dialog_uses_real_background_migration(legacy, qtbot):
    service, owner, project, order, _projection = legacy
    dialog = LegacyWorkOrderMigrationDialog(service, owner.principal, project.id)
    dialog.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, False)
    qtbot.addWidget(dialog)
    dialog.show()
    qtbot.waitUntil(lambda: dialog._worker is None, timeout=10000)
    assert order.instructions in dialog.preview.toPlainText()
    assert str(order.destination) in dialog.preview.toPlainText()
    assert dialog.apply_button.isEnabled()
    assert service._projection(project.id).get_work_order(order.id) == order
    with service._uow(str(project.id))() as unit:
        assert unit.event_store.load_stream(f'work-order:{order.id}') == ()
    dialog.apply_button.click()
    qtbot.waitUntil(lambda: dialog._worker is None, timeout=10000)
    assert '1' in dialog.status.text() and not dialog.apply_button.isEnabled()
    with service._uow(str(project.id))() as unit:
        event, = unit.event_store.load_stream(f'work-order:{order.id}')
    assert event.payload['work_order']['revision'] == 7
    dialog.close()


def test_administration_exposes_migration_for_catalog_project(legacy, qtbot):
    service, owner, project, _order, _projection = legacy
    panel = _administration_panel(service, owner)
    qtbot.addWidget(panel)
    panel.show()
    qtbot.waitUntil(lambda: not panel._admin_tasks.busy)
    button = panel.findChild(QPushButton, 'migrateLegacyWorkOrders')
    assert button is not None
    button.click()
    dialog = panel._legacy_migration_dialog
    qtbot.waitUntil(lambda: dialog._worker is None, timeout=10000)
    assert dialog.project_id == str(project.id)
    dialog.reject()
    panel.close()


def test_parent_deletion_does_not_destroy_running_migration_thread(tmp_path):
    root = Path(__file__).resolve().parents[1]
    script = r'''
import time
from threading import Event
from PyQt6 import sip
from PyQt6.QtCore import QCoreApplication,QEvent
from PyQt6.QtWidgets import QApplication,QWidget
from kraken_hub.legacy_work_order_dialog import LegacyWorkOrderMigrationDialog,_RUNNING
entered=Event(); release=Event()
class Service:
 def plan_legacy_work_order_migration(self,**kwargs):
  entered.set()
  assert release.wait(10)
  return {'legacy':[],'digest':'a'*64}
app=QApplication([]); parent=QWidget()
dialog=LegacyWorkOrderMigrationDialog(Service(),object(),'project-id',parent)
deadline=time.monotonic()+5
while not entered.is_set() and time.monotonic()<deadline:app.processEvents();time.sleep(.005)
assert entered.is_set()
parent.deleteLater();QCoreApplication.sendPostedEvents(None,QEvent.Type.DeferredDelete)
assert sip.isdeleted(dialog) and len(_RUNNING)==1
release.set();deadline=time.monotonic()+5
while _RUNNING and time.monotonic()<deadline:app.processEvents();time.sleep(.005)
assert not _RUNNING
QCoreApplication.sendPostedEvents(None,QEvent.Type.DeferredDelete)
print('owner-deletion-safe')
'''
    result = subprocess.run([sys.executable, '-c', script], env=dict(os.environ, PYTHONPATH=str(root/'src')),
                            capture_output=True, timeout=20)
    assert result.returncode == 0, result.stderr
    assert b'owner-deletion-safe' in result.stdout


@pytest.mark.parametrize('plan', [None, {'legacy': [{}], 'digest': 'a'*64}])
def test_unrenderable_plan_cannot_enable_approval(qtbot, plan):
    class Service:
        def plan_legacy_work_order_migration(self, **_kwargs):
            return plan
    dialog = LegacyWorkOrderMigrationDialog(Service(), object(), 'diagnostic-project')
    dialog.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, False)
    qtbot.addWidget(dialog)
    qtbot.waitUntil(lambda: dialog._worker is None, timeout=5000)
    assert dialog._plan is None and not dialog.apply_button.isEnabled()
    assert 'Ошибка' in dialog.status.text()
    dialog.close()
