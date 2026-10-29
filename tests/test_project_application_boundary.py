"""Run application policies without Qt, HTTP, SQL, or filesystem storage."""

from types import SimpleNamespace
from contextlib import nullcontext

import pytest

from kraken_manager.application.dto import CommandContext, DeleteLayerCommand, RenameLayerCommand
from kraken_manager.application.errors import ConflictError
from kraken_manager.application.layer_deletion import DeleteLayerHandler, StagedLayerDeletion
from kraken_manager.application.project_application import ProjectApplication
from kraken_manager.application.project_commands import ProjectCommandService
from kraken_manager.application.workspace_rename import RenameWorkspaceHandler
from kraken_manager.domain.project import StructureState
from test_plugin_job_application import Clock, Fixture


def memory_application(shared):
    fixture = Fixture(shared=shared)
    projections = fixture.factory.uow.projections
    projections.save_layer = lambda layer: projections.layers.__setitem__(layer.id, layer)
    projections.list_plugin_jobs = lambda _project_id: ()
    context = CommandContext(actor=fixture.actor, idempotency_key="command", gitlab_identity_verified=True)
    return fixture, context


@pytest.mark.parametrize("shared", [False, True])
def test_same_command_service_uses_only_injected_ports(shared):
    fixture, context = memory_application(shared)
    service = ProjectCommandService(fixture.factory, fixture.profiles, Clock())
    result = service.execute(
        RenameLayerCommand(
            context=context,
            project_id=fixture.project.id,
            layer_id=fixture.layer.id,
            name="Renamed",
            expected_revision=fixture.layer.revision,
        )
    )
    assert result.name == "Renamed"
    assert fixture.factory.uow.projections.get_layer(fixture.layer.id) == result
    assert fixture.factory.uow.event_store.load_stream(f"layer:{fixture.layer.id}")[-1].event_type == "LayerRenamed"


class MemoryDeletionStorage:
    def __init__(self):
        self.bound = True
        self.operations = []

    def lock(self, _project_id):
        return nullcontext()

    def is_bound(self, *_args):
        return self.bound

    def has_running_runs(self, *_args):
        return False

    def stage(self, *_args):
        self.operations.append("stage")
        return StagedLayerDeletion("opaque-memory-token", True)

    def rollback(self, _stage):
        self.operations.append("rollback")

    def complete(self, *_args):
        self.operations.append("complete")
        self.bound = False


@pytest.mark.parametrize("shared", [False, True])
def test_deletion_policy_runs_against_memory_and_replays(shared):
    fixture, context = memory_application(shared)
    storage = MemoryDeletionStorage()
    handler = DeleteLayerHandler(fixture.factory, fixture.profiles, Clock(), storage)
    command = DeleteLayerCommand(
        context=context,
        project_id=fixture.project.id,
        layer_id=fixture.layer.id,
        confirmation_name=fixture.layer.name,
        expected_revision=fixture.layer.revision,
    )
    deleted = handler(command)
    assert deleted.state is StructureState.ARCHIVED
    assert storage.operations == ["stage", "complete"]
    assert handler(command) == deleted
    assert storage.operations == ["stage", "complete"]


def test_failed_metadata_commit_rolls_back_through_storage_port():
    fixture, context = memory_application(False)
    storage = MemoryDeletionStorage()

    def fail(_command):
        raise ConflictError("Concurrent change")

    handler = DeleteLayerHandler(fixture.factory, fixture.profiles, Clock(), storage, archive=fail)
    with pytest.raises(ConflictError):
        handler(
            DeleteLayerCommand(
                context=context,
                project_id=fixture.project.id,
                layer_id=fixture.layer.id,
                confirmation_name=fixture.layer.name,
                expected_revision=fixture.layer.revision,
            )
        )
    assert storage.operations == ["stage", "rollback"]
    assert fixture.factory.uow.projections.get_layer(fixture.layer.id).state is StructureState.ACTIVE


def test_workspace_rename_rolls_back_without_knowing_storage_technology():
    fixture, context = memory_application(False)
    calls = []
    storage = SimpleNamespace(
        lock=lambda _project_id: nullcontext(),
        has_running_runs=lambda *_args: False,
        stage_layer=lambda *_args: calls.append("stage") or "opaque-stage",
        rollback=lambda _token: calls.append("rollback"),
        complete=lambda _token: calls.append("complete"),
    )

    def fail(_command):
        raise ConflictError("Metadata changed")

    handler = RenameWorkspaceHandler(fixture.factory, fixture.profiles, Clock(), storage, fail, layer=True)
    with pytest.raises(ConflictError):
        handler(
            RenameLayerCommand(
                context=context,
                project_id=fixture.project.id,
                layer_id=fixture.layer.id,
                name="Changed",
                expected_revision=fixture.layer.revision,
            )
        )
    assert calls == ["stage", "rollback"]


def test_project_application_accepts_an_injected_gateway_without_backend_branches():
    calls = []
    gateway = SimpleNamespace(get_artifact_version=lambda *args: calls.append(args) or "version")
    application = ProjectApplication(lambda _project_id: gateway)
    assert application.get_artifact_version("project", "artifact") == "version"
    assert calls == [("project", "artifact")]


def test_unknown_command_is_rejected_without_opening_storage():
    def forbidden_unit():
        pytest.fail("An unknown command must not access storage")

    service = ProjectCommandService(forbidden_unit, SimpleNamespace(), Clock())
    with pytest.raises(TypeError, match="Unknown project command"):
        service.execute(object())


def test_protocol_stubs_cannot_masquerade_as_a_complete_adapter():
    from kraken_manager.application.backend_contract import BackendContractError, validate_project_operations
    from kraken_manager.application.project_operations import ProjectOperations

    class Unimplemented(ProjectOperations):
        pass

    with pytest.raises(BackendContractError, match="has no implementation"):
        validate_project_operations(Unimplemented())


def test_an_adapter_cannot_make_an_optional_contract_parameter_required():
    from kraken_manager.application.backend_contract import BackendContractError, validate_project_operations
    from kraken_hub.infrastructure.remote.client import RemoteServerProjectService

    class Incompatible(RemoteServerProjectService):
        def list_layers(self, project_id, *, include_archived, as_of=None):
            return ()

    with pytest.raises(BackendContractError, match="optional parameter include_archived"):
        validate_project_operations(object.__new__(Incompatible))


def test_an_adapter_cannot_swap_positional_arguments():
    from kraken_manager.application.backend_contract import BackendContractError, validate_project_operations
    from kraken_hub.infrastructure.remote.client import RemoteServerProjectService

    class Incompatible(RemoteServerProjectService):
        def get_artifact_version(self, artifact_version_id, project_id):
            pytest.fail("Contract validation must not invoke the adapter")

    with pytest.raises(BackendContractError, match="positional parameter order"):
        validate_project_operations(object.__new__(Incompatible))


def test_an_adapter_must_accept_keyword_calls_declared_by_the_contract():
    from kraken_manager.application.backend_contract import BackendContractError, validate_project_operations
    from kraken_hub.infrastructure.remote.client import RemoteServerProjectService

    class Incompatible(RemoteServerProjectService):
        def get_artifact_version(self, project_id, artifact_version_id, /):
            pytest.fail("Contract validation must not invoke the adapter")

    with pytest.raises(BackendContractError, match="keyword parameter project_id"):
        validate_project_operations(object.__new__(Incompatible))


def test_statistics_are_calculated_in_the_core_without_export_adapters():
    from datetime import timedelta
    from kraken_manager.application.reporting import ActivityRecord, ReportFilters, ReportService

    timestamp = Clock().now()
    records = (ActivityRecord("event", timestamp, "ArtifactImported", "project", bytes_count=5),)
    service = ReportService()
    metrics = service.aggregate(
        records, ReportFilters(timestamp - timedelta(seconds=1), timestamp + timedelta(seconds=1))
    )
    assert metrics.values["imported_files"] == 1
    assert metrics.values["imported_bytes"] == 5
    assert not hasattr(service, "write_csv")
