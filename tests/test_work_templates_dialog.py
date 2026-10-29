"""Shown Qt dialogs exercise actual template authoring and work generation."""

from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtWidgets import QApplication, QDialog

from kraken_hub.presentation.qt.work_planning_dialog import WorkPlanningDialog
from kraken_hub.presentation.qt.work_template_editor import WorkTemplateEditor
from tests.test_work_planning_dialog import click, wait_done, check_layer
from tests.test_work_planning_integration import setup_project
from tests.test_work_templates import definition, recipe


def test_shown_template_editor_validates_steps_and_preserves_acceptance(qtbot):
    editor = WorkTemplateEditor(language="English")
    qtbot.addWidget(editor)
    editor.show()
    assert editor.isVisible()
    editor.accept()
    assert editor.result() != QDialog.DialogCode.Accepted
    assert editor.status.text()
    editor.title.setText("Controlled reconstruction")
    editor.step_title.setText("Prepare")
    editor.criterion.setText("Input versions recorded")
    qtbot.mouseClick(editor.add_button, Qt.MouseButton.LeftButton)
    assert editor.table.rowCount() == 1
    editor.step_title.setText("Segment")
    editor.multiplicity.setCurrentIndex(editor.multiplicity.findData("frame"))
    editor.predecessor.setCurrentIndex(1)
    qtbot.mouseClick(editor.add_button, Qt.MouseButton.LeftButton)
    assert editor.table.rowCount() == 2
    assert editor.definition["dependencies"] == [{"predecessor_key": "step_1", "successor_key": "step_2"}]
    assert editor.definition["tasks"][0]["criteria"][0]["description"] == "Input versions recorded"
    editor.accept()
    assert editor.result() == QDialog.DialogCode.Accepted


def test_shown_hub_dialog_generates_version_pinned_work_and_authors_new_template_version(qtbot, tmp_path, monkeypatch):
    local, principal, project, layer, performer = setup_project(tmp_path)
    template = local.save_work_template(
        project_id=project.id, principal=principal, definition=definition(recipe(project.id)), idempotency_key="t"
    )
    dialog = WorkPlanningDialog(local, principal, project.id, language="English")
    qtbot.addWidget(dialog)
    dialog.show()
    wait_done(qtbot, dialog)
    assert dialog.templates.currentData() == template
    dialog.plan_title.setText("Recovery")
    click(qtbot, dialog, dialog.create_button)
    check_layer(dialog, layer.id)
    dialog.template_frames.setText("1,1; 2,1")
    dialog.performer.setCurrentIndex(dialog.performer.findText(performer.name))
    click(qtbot, dialog, dialog.apply_template_button)
    assert dialog.plan.revision == 1 and dialog.items.rowCount() == 7
    assert all(item.assignment.performer_id == performer.id for item in dialog.plan.items)
    assert dialog.plan.template_applications[0].template_version == 1
    plan_id = dialog.plan.id
    original_exec = WorkTemplateEditor.exec
    observed = []

    def exec_and_edit(editor):
        def fill_visible_editor():
            observed.append(editor.isVisible())
            editor.title.setText("Second controlled process")
            editor.step_title.setText("Final acceptance")
            editor.multiplicity.setCurrentIndex(editor.multiplicity.findData("project"))
            editor.predecessor.setCurrentIndex(editor.predecessor.findData("register"))
            editor.add_step()
            editor.accept()

        QTimer.singleShot(0, fill_visible_editor)
        return original_exec(editor)

    monkeypatch.setattr(WorkTemplateEditor, "exec", exec_and_edit)
    click(qtbot, dialog, dialog.edit_template_button)
    assert observed == [True]
    assert dialog.templates.currentData().version == 2
    assert local.get_work_template(project.id, template.id, version=1) == template
    # Publishing a definition leaves previously generated work unchanged.
    assert local.get_work_plan(project.id, plan_id).revision == 1
    assert dialog.plan.template_applications[0].template_version == 1
    click(qtbot, dialog, dialog.apply_template_button)
    assert dialog.plan.revision == 2 and dialog.items.rowCount() == 15
    assert [a.template_version for a in dialog.plan.template_applications] == [1, 2]
    dialog.close()
    QApplication.processEvents()


def test_shown_editor_rejects_parent_acceptance_deadlock(qtbot):
    editor = WorkTemplateEditor(language="English")
    qtbot.addWidget(editor)
    editor.show()
    editor.title.setText("Safe hierarchy")
    editor.step_title.setText("Parent")
    editor.add_step()
    editor.step_title.setText("Child")
    editor.parent_step.setCurrentIndex(1)
    editor.predecessor.setCurrentIndex(1)
    editor.add_step()
    assert editor.table.rowCount() == 1
    assert "deadlock" in editor.status.text()
    editor.close()
