"""Assembly validates adapters; project operations have explicit routes."""

import pytest
from kraken_hub.composition import EmbeddedProjectService
from kraken_hub.composition.catalog import DualCatalogService
from kraken_hub.infrastructure.remote.client import RemoteAuth, RemoteServerProjectService
from kraken_manager.application.backend_contract import BackendContractError
from kraken_manager.domain.identity import Principal


class _Project:
    def __init__(self, project_id):
        self.id = project_id


class _Local(EmbeddedProjectService):
    def __init__(self, directory):
        super().__init__(directory)
        self.calls = []

    def get_artifact_version(self, project_id, artifact_version_id):
        self.calls.append("get_artifact_version")
        return "local-version"

    def plan_import_directory(self, *, project, directory, mode=None, regex=None):
        self.calls.append("plan_import_directory")
        return "local-plan"


class _Remote(RemoteServerProjectService):
    def __init__(self, directory, projects=()):
        self._projects = projects
        super().__init__(
            "http://example.test",
            data_dir=directory,
            auth=RemoteAuth("test", Principal.local(subject="test", display_name="Test")),
        )
        self.calls = []

    def list_projects(self, *, include_archived=False):
        return self._projects

    def get_artifact_version(self, project_id, artifact_version_id):
        self.calls.append("get_artifact_version")
        return "remote-version"

    def plan_import_directory(self, *, project, directory, mode=None, regex=None):
        self.calls.append("plan_import_directory")
        return "remote-plan"


def test_known_server_project_is_labeled_and_routed_to_the_server(tmp_path):
    local = _Local(tmp_path / "local")
    remote = _Remote(tmp_path / "remote", (_Project("remote-1"),))
    catalog = DualCatalogService(local, remote)
    assert catalog.is_remote_project("remote-1")
    assert not catalog.is_remote_project("local-1")
    assert catalog.get_artifact_version("remote-1", "version") == "remote-version"
    assert remote.calls == ["get_artifact_version"]
    assert not local.calls


def test_directory_plan_uses_server_files_for_a_server_project(tmp_path):
    local = _Local(tmp_path / "local")
    remote = _Remote(tmp_path / "remote", (_Project("remote-1"),))
    catalog = DualCatalogService(local, remote)
    assert catalog.plan_import_directory(project=_Project("remote-1"), directory="server-path") == "remote-plan"
    assert not local.calls
    assert remote.calls == ["plan_import_directory"]


def test_missing_operation_is_rejected_before_any_project_action(tmp_path):
    local = _Local(tmp_path / "local")
    remote = _Remote(tmp_path / "remote")
    remote.delete_layer = None
    with pytest.raises(BackendContractError, match="delete_layer"):
        DualCatalogService(local, remote)
    assert not local.calls


def test_unknown_operation_cannot_fall_back_to_the_local_catalog(tmp_path):
    catalog = DualCatalogService(_Local(tmp_path / "local"))
    with pytest.raises(AttributeError):
        catalog.missing_operation(project_id="local-1")


def test_local_project_uses_the_local_adapter(tmp_path):
    local = _Local(tmp_path / "local")
    catalog = DualCatalogService(local)
    assert catalog.get_artifact_version("local-1", "version") == "local-version"
    assert local.calls == ["get_artifact_version"]


def test_disconnected_server_project_never_falls_back_to_local_storage(tmp_path):
    from kraken_hub.application.remote_error import RemoteServerError

    local = _Local(tmp_path / "local")
    catalog = DualCatalogService(local)
    catalog._remote_ids = {"remote-1"}
    with pytest.raises(RemoteServerError):
        catalog.get_artifact_version("remote-1", "version")
    with pytest.raises(RemoteServerError):
        catalog.get_project("remote-1")
    assert not local.calls
