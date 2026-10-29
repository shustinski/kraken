from __future__ import annotations

from types import SimpleNamespace
from uuid import uuid4
from datetime import UTC, datetime

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import QDialogButtonBox

from kraken_hub.presentation.qt.work_planning_dialog import WorkPlanningDialog
from kraken_manager.domain.planning import (
    AcceptanceCriterion, WorkAssignment, WorkItem, WorkPlan, WorkScope,
)


def uid():
    return str(uuid4())


class PlanningAdapter:
    """Contract adapter exercising the real immutable aggregate, without a server."""

    def __init__(self):
        self.plans = {}
        self.calls = []
        self.fail_next = False
        self.performers = [SimpleNamespace(id=uid(), name="Review engineer", active=True)]
        self.layers = [SimpleNamespace(id=uid(), name="Metal 1", state="active")]
        self.versions = [SimpleNamespace(id=uid(), filename="Reviewed mask.png", created_at=datetime.now(UTC), sha256="e"*64)]

    def list_performers(self, **kwargs):
        return self.performers

    def list_layers(self, project_id, **kwargs):
        return self.layers

    def get_project(self, project_id):
        return SimpleNamespace(width=2, height=2)

    def quality_frame_versions(self, project_id, layer_id, x, y):
        return self.versions

    def list_work_plans(self, project_id):
        return [p for p in self.plans.values() if p.project_id == project_id]

    def get_work_plan(self, project_id, plan_id):
        plan = self.plans[plan_id]
        assert plan.project_id == project_id
        return plan

    def create_work_plan(self, project_id, title, principal, idempotency_key):
        plan = WorkPlan.create(project_id, title)
        self.plans[plan.id] = plan
        return plan

    def mutate_work_plan(self, project_id, plan_id, principal, expected_revision, operation, arguments, idempotency_key):
        self.calls.append((operation, arguments, expected_revision, idempotency_key, principal))
        if self.fail_next:
            self.fail_next = False
            raise ConnectionError("Disconnected before saving")
        plan = self.get_work_plan(project_id, plan_id)
        args = dict(arguments)
        if operation == "add_item":
            item = dict(args["item"])
            item["scope"] = WorkScope(**item["scope"])
            if item.get("assignment"):
                item["assignment"] = WorkAssignment(**item["assignment"])
            item["criteria"] = tuple(AcceptanceCriterion(uid(), **c) for c in item.get("criteria", ()))
            args["item"] = WorkItem.create(**item)
        elif operation == "assign":
            args["assignment"] = WorkAssignment(**args["assignment"])
        elif operation == "satisfy_criterion":
            args["reviewer_id"] = principal.id
        changed = getattr(plan, operation)(expected_revision=expected_revision, **args)
        # Use the existing wire format when returning data to the widget.
        self.plans[plan.id] = WorkPlan.from_dict(changed.to_dict())
        return self.plans[plan.id]


def wait_done(qtbot, dialog):
    qtbot.waitUntil(lambda: dialog._worker is None, timeout=10000)


def click(qtbot, dialog, button):
    assert button.isEnabled()
    qtbot.mouseClick(button, Qt.MouseButton.LeftButton)
    wait_done(qtbot, dialog)


def create_dialog(qtbot, service=None, *, language="English"):
    service = service or PlanningAdapter()
    dialog = WorkPlanningDialog(service, SimpleNamespace(id=uid()), uid(), language=language)
    qtbot.addWidget(dialog)
    dialog.show()
    wait_done(qtbot, dialog)
    return dialog, service


def create_plan_and_item(qtbot, dialog):
    dialog.plan_title.setText("Layer reconstruction")
    click(qtbot, dialog, dialog.create_button)
    dialog.item_title.setText("Review mask")
    dialog.performer.setCurrentIndex(1)
    dialog.criterion_description.setText("Reviewed against versioned artifact")
    click(qtbot, dialog, dialog.add_button)


def move(qtbot, dialog, state):
    index = dialog.target_state.findData(state)
    assert index >= 0
    dialog.target_state.setCurrentIndex(index)
    click(qtbot, dialog, dialog.move_button)


def choose_evidence(qtbot, dialog, monkeypatch, identifier=None):
    from kraken_hub.presentation.qt.planning_evidence_picker import PlanningEvidencePicker
    original = PlanningEvidencePicker.exec

    def execute(picker):
        def select():
            try:
                assert picker.isVisible()
                qtbot.waitUntil(lambda: picker._worker is None, timeout=10000)
                qtbot.mouseClick(picker.load_button, Qt.MouseButton.LeftButton)
                qtbot.waitUntil(lambda: picker._worker is None, timeout=10000)
                candidates = [picker.versions.item(i) for i in range(picker.versions.count())]
                row = next(item for item in candidates if identifier is None or item.data(Qt.ItemDataRole.UserRole) == str(identifier))
                row.setCheckState(Qt.CheckState.Checked)
                qtbot.mouseClick(picker.buttons.button(QDialogButtonBox.StandardButton.Ok), Qt.MouseButton.LeftButton)
            finally:
                if picker.isVisible():
                    picker.reject()
        QTimer.singleShot(0, select)
        return original(picker)

    with monkeypatch.context() as patch:
        patch.setattr(PlanningEvidencePicker, "exec", execute)
        click(qtbot, dialog, dialog.evidence_button)


def check_layer(dialog, identifier):
    for i in range(dialog.layer_ids.count()):
        item = dialog.layer_ids.item(i)
        if item.data(Qt.ItemDataRole.UserRole) == str(identifier):
            item.setCheckState(Qt.CheckState.Checked)
            return
    raise AssertionError("Layer is absent from the named selector")


def test_shown_dialog_full_acceptance_baseline_completion_and_reopen(qtbot, monkeypatch):
    dialog, service = create_dialog(qtbot)
    assert dialog.isVisible()
    create_plan_and_item(qtbot, dialog)
    assert dialog.items.rowCount() == 1
    click(qtbot, dialog, dialog.activate_button)
    move(qtbot, dialog, "running")
    dialog.percent.setValue(100)
    dialog.effort.setValue(2.5)
    click(qtbot, dialog, dialog.progress_button)
    move(qtbot, dialog, "in_review")
    # Formal acceptance must fail while the mandatory criterion is unsatisfied.
    move(qtbot, dialog, "accepted")
    assert dialog.plan.items[0].state == "in_review"
    assert "Error:" in dialog.status.text()
    choose_evidence(qtbot, dialog, monkeypatch)
    click(qtbot, dialog, dialog.satisfy_button)
    move(qtbot, dialog, "accepted")
    assert dialog.plan.items[0].criteria[0].reviewed_by == dialog.principal.id
    assert dialog.plan.items[0].progress.actual_effort_hours == 2.5
    click(qtbot, dialog, dialog.baseline_button)
    assert len(dialog.plan.baselines) == 1
    click(qtbot, dialog, dialog.complete_button)
    assert dialog.plan.state == "completed"
    assert not dialog.add_button.isEnabled()
    plan_id = dialog.plan.id
    project_id = dialog.project_id
    dialog.close()
    reopened = WorkPlanningDialog(service, dialog.principal, project_id, language="English")
    qtbot.addWidget(reopened)
    reopened.show()
    wait_done(qtbot, reopened)
    assert reopened.plan.id == plan_id
    assert reopened.plan.state == "completed"
    assert reopened.items.rowCount() == 1
    assert len(reopened.plan.baselines) == 1
    reopened.close()


def test_failure_preserves_state_and_reuses_idempotency_key_for_same_retry(qtbot):
    dialog, service = create_dialog(qtbot, language="Русский")
    create_plan_and_item(qtbot, dialog)
    revision = dialog.plan.revision
    service.fail_next = True
    click(qtbot, dialog, dialog.activate_button)
    assert dialog.plan.state == "draft"
    assert dialog.plan.revision == revision
    assert "Disconnected" in dialog.status.text()
    assert dialog.controls.isEnabled()
    click(qtbot, dialog, dialog.activate_button)
    assert dialog.plan.state == "active"
    assert service.calls[-1][3] == service.calls[-2][3]
    assert service.calls[-1][4] is dialog.principal
    dialog.close()


def test_stale_revision_rejected_then_explicit_refresh_recovers(qtbot):
    dialog, service = create_dialog(qtbot)
    create_plan_and_item(qtbot, dialog)
    stale = dialog.plan
    service.plans[stale.id] = stale.capture_baseline(expected_revision=stale.revision)
    click(qtbot, dialog, dialog.activate_button)
    assert dialog.plan == stale
    assert "revision" in dialog.status.text()
    click(qtbot, dialog, dialog.refresh_button)
    assert dialog.plan.revision == stale.revision + 1
    assert len(dialog.plan.baselines) == 1
    click(qtbot, dialog, dialog.activate_button)
    assert dialog.plan.state == "active"
    dialog.close()


def test_real_embedded_service_accepts_versioned_evidence_and_reopens_durable_plan(qtbot, tmp_path, monkeypatch):
    from tests.test_work_orders import _project_with_frames, _service

    monkeypatch.setenv("KRAKEN_PROJECT_ACCESS_MODE", "acl")
    service = _service(tmp_path)
    owner, performer, project, layer, images, _vectors = _project_with_frames(
        service, tmp_path / "source", with_vectors=False,
    )
    series = next(
        item for item in service.list_artifact_series(project.id, layer_id=layer.id)
        if item.representation_id == images.id and item.frame_id == project.frame_id_at(1, 1)
    )
    evidence = service.artifact_versions(project.id, series.id)[0]
    dialog = WorkPlanningDialog(service, owner.principal, project.id, language="English")
    qtbot.addWidget(dialog)
    dialog.show()
    wait_done(qtbot, dialog)
    dialog.plan_title.setText("Persistent layer acceptance")
    click(qtbot, dialog, dialog.create_button)
    assert dialog.plan is not None, dialog.status.text()
    dialog.item_title.setText("Review imported artifact")
    dialog.performer.setCurrentIndex(dialog.performer.findText(performer.name))
    check_layer(dialog, layer.id)
    dialog.criterion_description.setText("Engineer reviewed the imported revision")
    click(qtbot, dialog, dialog.add_button)
    assert dialog.items.rowCount() == 1, dialog.status.text()
    click(qtbot, dialog, dialog.activate_button)
    assert dialog.plan.state == "active", dialog.status.text()
    move(qtbot, dialog, "running")
    dialog.percent.setValue(100)
    dialog.effort.setValue(1.25)
    click(qtbot, dialog, dialog.progress_button)
    move(qtbot, dialog, "in_review")
    assert dialog.plan.items[0].state == "in_review", dialog.status.text()
    choose_evidence(qtbot, dialog, monkeypatch, evidence.id)
    click(qtbot, dialog, dialog.satisfy_button)
    assert dialog.plan.items[0].criteria[0].satisfied, dialog.status.text()
    move(qtbot, dialog, "accepted")
    assert dialog.plan.items[0].state == "accepted", dialog.status.text()
    click(qtbot, dialog, dialog.baseline_button)
    click(qtbot, dialog, dialog.complete_button)
    assert dialog.plan.state == "completed", dialog.status.text()
    saved = dialog.plan
    dialog.close()

    # A fresh composition reads the event log, not the first service's objects.
    reopened_service = _service(tmp_path)
    reopened = WorkPlanningDialog(reopened_service, owner.principal, project.id, language="English")
    qtbot.addWidget(reopened)
    reopened.show()
    wait_done(qtbot, reopened)
    assert reopened.plan == saved, reopened.status.text()
    accepted = reopened.plan.items[0]
    assert accepted.progress.actual_effort_hours == 1.25
    assert accepted.criteria[0].evidence_version_ids == (str(evidence.id),)
    assert accepted.criteria[0].reviewed_by == owner.principal.id
    assert reopened.items.rowCount() == 1
    assert len(reopened.plan.baselines) == 1
    reopened.close()

def test_plan_operation_participates_in_hub_close_guard(qtbot):
    from threading import Event
    from types import SimpleNamespace
    from uuid import uuid4
    from kraken_hub.presentation.qt.work_planning_dialog import WorkPlanningDialog

    release = Event()
    workers = set()

    class SlowService:
        def list_work_plans(self, _project):
            assert release.wait(timeout=5)
            return ()

    dialog = WorkPlanningDialog(
        SlowService(), SimpleNamespace(id=str(uuid4())), str(uuid4()), background_workers=workers,
    )
    qtbot.addWidget(dialog)
    try:
        qtbot.waitUntil(lambda: bool(workers))
        assert any(worker.isRunning() for worker in workers)
        assert dialog.close() is False
    finally:
        release.set()
        qtbot.waitUntil(lambda: dialog._worker is None)
    assert workers == set()
    assert dialog.close() is True
