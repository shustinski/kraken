from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest
import sqlalchemy as sa

from kraken_admin.local_admin import open_project_services
from kraken_admin.project_deletion import (
    ProjectDeletion,
    directories_outside_project,
    project_storage_paths,
    safe_path,
)
from kraken_admin.project_roles import all_principals, set_roles, snapshot
from kraken_manager.domain.common import ProjectId
from kraken_manager.domain.identity import Principal, ProjectRole
from kraken_manager.infrastructure.auth.local import ScryptPasswordHasher
from kraken_manager.infrastructure.postgres.account_store import PostgresAccountStore
from kraken_manager.infrastructure.workspace_files import WorkspaceFileService, WorkspaceRegistry
from kraken_server.project_deletion import DeletionRequests, operation
from kraken_server.services import CommandContext, ConflictError, ForbiddenError


@pytest.fixture
def database(tmp_path, monkeypatch):
    url = os.environ.get("KRAKEN_TEST_POSTGRES_URL")
    if not url:
        pytest.skip("KRAKEN_TEST_POSTGRES_URL is not configured")
    from alembic import command
    from alembic.config import Config

    base = sa.create_engine(url)
    schema = "admin_test_" + uuid4().hex
    with base.begin() as connection:
        connection.execute(sa.text(f'CREATE SCHEMA "{schema}"'))
    scoped = sa.engine.make_url(url).update_query_dict({"options": f"-csearch_path={schema}"})
    monkeypatch.setenv("KRAKEN_DATABASE_URL", scoped.render_as_string(hide_password=False))
    engine = sa.create_engine(scoped)
    try:
        command.upgrade(Config("alembic.ini"), "head")
        services = open_project_services(str(scoped), tmp_path / "blobs", engine=engine)
        services.workspace_files = WorkspaceFileService(WorkspaceRegistry(tmp_path / "blobs" / "workspaces"))
        services.source_root = str(tmp_path / "source")
        services.derived_root = str(tmp_path / "derived")
        Path(services.source_root).mkdir()
        Path(services.derived_root).mkdir()
        accounts = PostgresAccountStore(engine, ScryptPasswordHasher())
        admin = accounts.create_account("admin", "Administrator", "password")
        accounts.grant_global_role(admin.account_id, "server_admin")
        person = Principal.local(subject="admin", display_name="Administrator", principal_id=admin.account_id)
        services.identities.save(person)
        config = SimpleNamespace(blob_root=tmp_path / "blobs", source_root=Path(services.source_root),
                                 derived_root=Path(services.derived_root))
        yield services, accounts, person, config
    finally:
        engine.dispose()
        with base.begin() as connection:
            connection.execute(sa.text(f'DROP SCHEMA "{schema}" CASCADE'))
        base.dispose()


def project(services, person, name="Example"):
    return services.create_project({"name": name, "width": 2, "height": 2},
                                   CommandContext(str(person.id), str(uuid4()), None))


def test_full_deletion_retains_audit_and_other_project(database):
    services, accounts, person, config = database
    target = project(services, person)
    other = project(services, person, "Other")
    binding = services.workspace_files.registry.get_project(target["project_id"])
    image = Path(binding.source_project_dir) / "image.jpg"
    image.write_bytes(b"image")
    request = services.request_project_deletion(target["project_id"], str(person.id))
    assert services.request_project_deletion(target["project_id"], str(person.id))["request_id"] == request["request_id"]
    cleaner = ProjectDeletion(services, accounts, config)
    plan = cleaner.preview(request["request_id"])
    cleaner.execute(request["request_id"], target["name"], plan)
    assert not image.exists()
    assert [row["project_id"] for row in services.list_projects()] == [other["project_id"]]
    assert DeletionRequests(services.engine).list(target["project_id"])[0]["state"] == "deleted"
    assert any(row["action"] == "project.deleted" for row in accounts.administration_audit())


def test_deletion_without_workspace_registry_removes_project_folders(database):
    services, accounts, person, config = database
    target = project(services, person, "Legacy")
    binding = services.workspace_files.registry.get_project(target["project_id"])
    image = Path(binding.source_project_dir) / "image.jpg"
    image.write_bytes(b"image")
    services.workspace_files.registry.remove_project(target["project_id"])
    request = services.request_project_deletion(target["project_id"], str(person.id))
    cleaner = ProjectDeletion(services, accounts, config)
    plan = cleaner.preview(request["request_id"])
    assert str(image.parent) in [entry["path"] for entry in plan["paths"]]
    cleaner.execute(request["request_id"], target["name"], plan)
    assert not image.exists()
    assert target["project_id"] not in [row["project_id"] for row in services.list_projects()]


def test_deletion_without_folders_or_registry_removes_database_project(database):
    import shutil

    services, accounts, person, config = database
    target = project(services, person, "Empty")
    binding = services.workspace_files.registry.get_project(target["project_id"])
    shutil.rmtree(binding.source_project_dir)
    shutil.rmtree(binding.derived_project_dir)
    services.workspace_files.registry.remove_project(target["project_id"])
    request = services.request_project_deletion(target["project_id"], str(person.id))
    cleaner = ProjectDeletion(services, accounts, config)
    cleaner.execute(request["request_id"], target["name"], cleaner.preview(request["request_id"]))
    assert target["project_id"] not in [row["project_id"] for row in services.list_projects()]


def test_request_permissions_rejection_and_live_operation(database):
    services, accounts, person, config = database
    target = project(services, person)
    outsider = Principal.local(subject="outsider", display_name="Outsider")
    services.identities.save(outsider)
    with pytest.raises(ForbiddenError):
        services.request_project_deletion(target["project_id"], str(outsider.id))
    request = services.request_project_deletion(target["project_id"], str(person.id))
    cleaner = ProjectDeletion(services, accounts, config)
    plan = cleaner.preview(request["request_id"])
    with operation(services.engine, target["project_id"]):
        with pytest.raises(ConflictError, match="операция"):
            cleaner.execute(request["request_id"], target["name"], plan)
    cleaner.reject(request["request_id"])
    assert DeletionRequests(services.engine).list()[0]["state"] == "rejected"
    assert services.request_project_deletion(target["project_id"], str(person.id))["request_id"] != request["request_id"]


def test_failed_cleanup_blocks_writes_and_can_resume(database, monkeypatch):
    services, accounts, person, config = database
    target = project(services, person)
    request = services.request_project_deletion(target["project_id"], str(person.id))
    cleaner = ProjectDeletion(services, accounts, config)
    import kraken_admin.deletion_confirmation as module
    remove = module.shutil.rmtree
    monkeypatch.setattr(module.shutil, "rmtree", lambda *_: (_ for _ in ()).throw(OSError("disk failure")))
    with pytest.raises(OSError, match="disk failure"):
        cleaner.execute(request["request_id"], target["name"], cleaner.preview(request["request_id"]))
    with pytest.raises(ConflictError):
        with operation(services.engine, target["project_id"]):
            pass
    with services.engine.begin() as connection:
        with pytest.raises(sa.exc.DBAPIError):
            connection.execute(sa.text("UPDATE projects SET name = 'changed' WHERE project_id = :id"),
                               {"id": target["project_id"]})
    monkeypatch.setattr(module.shutil, "rmtree", remove)
    cleaner.execute(request["request_id"], target["name"], cleaner.preview(request["request_id"]))
    assert DeletionRequests(services.engine).list()[0]["state"] == "deleted"


def test_atomic_roles_include_disabled_users_and_detect_conflicts(database):
    services, accounts, person, _config = database
    target = project(services, person)
    account = accounts.create_account("disabled", "Disabled", "password")
    accounts.set_enabled(account.account_id, False)
    user = next(item for item in all_principals(services, accounts) if str(item.id) == account.account_id)
    assert not user.active
    identifier = ProjectId(target["project_id"])
    before = snapshot(services.identities.assignments_for(identifier))
    desired = frozenset({ProjectRole.VIEWER, ProjectRole.CORRECTOR})
    set_roles(services, accounts, project_id=str(identifier), principal=user, desired=desired,
              expected_revision=0, expected_snapshot=before)
    assert services.identities.roles_for(identifier, user.id) == desired
    from kraken_manager.application.errors import ConcurrencyError
    with pytest.raises(ConcurrencyError):
        set_roles(services, accounts, project_id=str(identifier), principal=user, desired=frozenset(),
                  expected_revision=0, expected_snapshot=before)


def test_external_layer_outside_the_project_is_left_in_place(tmp_path: Path) -> None:
    source = tmp_path / "source" / "Chip"
    derived = tmp_path / "derived" / "Chip"
    external = tmp_path / "Temp" / "kraken-e2e-hub" / "images"
    inside = source / "img" / "Metal"
    workspace = {
        "layers": {
            "layer-1": {
                "image_directory": str(external),
                "ssc_directory": "",
                "import_root": str(inside),
            }
        }
    }

    untouched = directories_outside_project(workspace, [source, derived])

    assert untouched == [str(external.resolve())]


def test_missing_registry_uses_conventional_project_folders(tmp_path: Path) -> None:
    source = tmp_path / "source"
    derived = tmp_path / "derived"
    (source / "Chip").mkdir(parents=True)
    (derived / "Chip").mkdir(parents=True)

    owned, untouched = project_storage_paths(
        project_name="Chip",
        source_root=source,
        derived_root=derived,
        binding=None,
        recorded_source=None,
        recorded_derived=None,
        claimed_directories=[],
    )

    assert owned == [(source / "Chip", source), (derived / "Chip", derived)]
    assert untouched == []


def test_missing_registry_without_folders_owns_nothing(tmp_path: Path) -> None:
    owned, untouched = project_storage_paths(
        project_name="Chip",
        source_root=tmp_path / "source",
        derived_root=tmp_path / "derived",
        binding=None,
        recorded_source=None,
        recorded_derived=None,
        claimed_directories=[],
    )

    assert owned == []
    assert untouched == []


def test_recorded_directory_outside_roots_is_left_in_place(tmp_path: Path) -> None:
    external = tmp_path / "external" / "Chip"

    owned, untouched = project_storage_paths(
        project_name="Chip",
        source_root=tmp_path / "source",
        derived_root=tmp_path / "derived",
        binding=None,
        recorded_source=str(external),
        recorded_derived=None,
        claimed_directories=[],
    )

    assert owned == []
    assert untouched == [str(external)]


def test_recorded_directory_inside_root_is_owned_after_rename(tmp_path: Path) -> None:
    source = tmp_path / "source"
    folder = source / "old-name"
    folder.mkdir(parents=True)

    owned, untouched = project_storage_paths(
        project_name="Renamed",
        source_root=source,
        derived_root=tmp_path / "derived",
        binding=None,
        recorded_source=str(folder),
        recorded_derived=None,
        claimed_directories=[],
    )

    assert owned == [(folder, source)]
    assert untouched == []


def test_conventional_folder_claimed_by_another_project_is_left_in_place(tmp_path: Path) -> None:
    source = tmp_path / "source"
    folder = source / "Chip"
    folder.mkdir(parents=True)

    owned, untouched = project_storage_paths(
        project_name="Chip",
        source_root=source,
        derived_root=tmp_path / "derived",
        binding=None,
        recorded_source=None,
        recorded_derived=None,
        claimed_directories=[str(folder)],
    )

    assert owned == []
    assert untouched == [str(folder.resolve())]


def test_recorded_directory_claimed_by_another_project_is_refused(tmp_path: Path) -> None:
    source = tmp_path / "source"
    folder = source / "Chip"
    folder.mkdir(parents=True)

    with pytest.raises(ValueError, match="другим проектом"):
        project_storage_paths(
            project_name="Chip",
            source_root=source,
            derived_root=tmp_path / "derived",
            binding=None,
            recorded_source=str(folder),
            recorded_derived=None,
            claimed_directories=[str(folder)],
        )


def test_safe_path_rejects_root_and_outside(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    with pytest.raises(ValueError):
        safe_path(root, root)
    with pytest.raises(ValueError):
        safe_path(tmp_path / "outside", root)
    assert safe_path(root / "project", root) == root / "project"


def test_shared_blobs_and_unregistered_transfers(database):
    import hashlib

    services, accounts, person, config = database
    target = project(services, person)
    other = project(services, person, "Other")
    requests = DeletionRequests(services.engine)
    shared, owned = hashlib.sha256(b"shared").hexdigest(), hashlib.sha256(b"owned").hexdigest()
    for digest in (shared, owned):
        path = services.uow_factory.blobs._path(digest)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"test")
        requests.record_transfer(digest, {"project": target["project_id"]})
    requests.record_transfer(shared, {"project": other["project_id"]})
    row = services.request_project_deletion(target["project_id"], str(person.id))
    cleaner = ProjectDeletion(services, accounts, config)
    cleaner.execute(row["request_id"], target["name"], cleaner.preview(row["request_id"]))
    assert services.uow_factory.blobs._path(shared).exists()
    assert not services.uow_factory.blobs._path(owned).exists()


def test_gateway_transfer_and_shared_directory_block_deletion(database):
    from dataclasses import replace

    services, accounts, person, config = database
    target = project(services, person)
    row = services.request_project_deletion(target["project_id"], str(person.id))
    cleaner = ProjectDeletion(services, accounts, config)
    lease = config.blob_root / ".transfers" / target["project_id"] / "lease"
    lease.parent.mkdir(parents=True)
    lease.touch()
    with pytest.raises(ConflictError, match="передачи"):
        cleaner.execute(row["request_id"], target["name"], cleaner.preview(row["request_id"]))
    assert not (config.blob_root / ".deleting" / target["project_id"]).exists()
    other = project(services, person, "Other")
    registry = services.workspace_files.registry
    registry.save_project(replace(registry.get_project(other["project_id"]),
                                  source_project_dir=registry.get_project(target["project_id"]).source_project_dir))
    with pytest.raises(ValueError, match="другим проектом"):
        cleaner.preview(row["request_id"])


def test_http_request_and_no_remote_approval(database):
    from fastapi.testclient import TestClient
    from kraken_server.app import SessionPrincipal, create_app

    services, accounts, person, _config = database
    target = project(services, person)
    app = create_app(services=services, account_store=accounts,
                     session_resolver=lambda token: SessionPrincipal(str(person.id), "local", token))
    with TestClient(app) as client:
        url = f"/api/v1/projects/{target['project_id']}"
        assert client.post(url + "/deletion-requests").status_code == 401
        headers = {"Authorization": "Bearer admin"}
        response = client.post(url + "/deletion-requests", headers=headers)
        assert response.status_code == 202, response.text
        assert response.json()["reason"] == ""
        with_reason = client.post(url + "/deletion-requests", headers=headers, json={"reason": "  duplicate scan  "})
        assert with_reason.status_code == 202, with_reason.text
        assert with_reason.json()["request_id"] == response.json()["request_id"]
        assert with_reason.json()["reason"] == "duplicate scan"
        assert client.get(url + "/deletion-requests", headers=headers).json()["items"][0]["state"] == "pending"
        too_long = client.post(url + "/deletion-requests", headers=headers, json={"reason": "x" * 501})
        assert too_long.status_code == 422
        assert client.delete(url, headers=headers).status_code == 405
        assert client.post(url + "/deletion-requests/approve", headers=headers).status_code == 404


def test_role_batch_rolls_back_on_second_failure(database, monkeypatch):
    from kraken_manager.application.acl import AssignProjectRoleHandler

    services, accounts, person, _config = database
    target = project(services, person)
    account = accounts.create_account("user", "User", "password")
    user = next(item for item in all_principals(services, accounts) if str(item.id) == account.account_id)
    identifier = ProjectId(target["project_id"])
    original = AssignProjectRoleHandler._apply
    def fail_second(handler, uow, command, now):
        if command.role is ProjectRole.VIEWER:
            raise RuntimeError("second assignment failed")
        original(handler, uow, command, now)
    monkeypatch.setattr(AssignProjectRoleHandler, "_apply", fail_second)
    with pytest.raises(RuntimeError, match="second assignment"):
        set_roles(services, accounts, project_id=str(identifier), principal=user,
                  desired=frozenset({ProjectRole.CORRECTOR, ProjectRole.VIEWER}), expected_revision=0,
                  expected_snapshot=snapshot(services.identities.assignments_for(identifier)))
    assert not services.identities.roles_for(identifier, user.id)
    assert services.events.current_revision(f"acl:{identifier}:{user.id}") == 0


def test_cascade_preview_matches_applied_revocations(database):
    from kraken_admin.project_roles import cascade_preview

    services, accounts, person, _config = database
    target = project(services, person)
    identifier = ProjectId(target["project_id"])
    for name in ("maintainer", "viewer"):
        account = accounts.create_account(name, name, "password")
        services.identities.save(Principal.local(subject=name, display_name=name, principal_id=account.account_id))
    people = {item.subject: item for item in all_principals(services, accounts)}
    maintainer, viewer = people["maintainer"], people["viewer"]
    set_roles(services, accounts, project_id=str(identifier), principal=maintainer,
              desired=frozenset({ProjectRole.MAINTAINER}), expected_revision=0,
              expected_snapshot=snapshot(services.identities.assignments_for(identifier)))
    services.assign_project_role(str(identifier), str(viewer.id), "viewer",
                                 CommandContext(str(maintainer.id), str(uuid4()), 0))
    assignments = snapshot(services.identities.assignments_for(identifier))
    assert cascade_preview(assignments, maintainer, frozenset()) == ((str(viewer.id), "viewer"),)
    set_roles(services, accounts, project_id=str(identifier), principal=maintainer, desired=frozenset(),
              expected_revision=1, expected_snapshot=assignments)
    assert not services.identities.roles_for(identifier, viewer.id)


def test_file_deletion_requires_confirmation_or_enabled_auto_confirm(database):
    from kraken_admin.deletion_confirmation import DeletionPolicy

    services, accounts, person, config = database
    target = project(services, person)
    image = Path(services.workspace_files.registry.get_project(target["project_id"]).source_project_dir) / "image.jpg"
    image.write_bytes(b"image")
    request = services.request_project_deletion(target["project_id"], str(person.id), "  old scan  ")
    assert request["reason"] == "old scan"
    cleaner = ProjectDeletion(services, accounts, config)
    plan = cleaner.preview(request["request_id"])
    with pytest.raises(ConflictError, match="подтверждения"):
        cleaner.execute(request["request_id"], "wrong", plan)
    with pytest.raises(ConflictError, match="выключено"):
        cleaner.execute(request["request_id"], "", plan, auto_confirm=True)
    assert image.exists()
    DeletionPolicy(services.engine).set_enabled(True, accounts)
    cleaner.execute(request["request_id"], "", plan, auto_confirm=True)
    assert not image.exists()
    audit = accounts.administration_audit()
    assert any(row["action"] == "deletion.auto_confirm_enabled" for row in audit)
    deleted = next(row for row in audit if row["action"] == "project.deleted")
    assert deleted["details"]["reason"] == "old scan"
    assert deleted["details"]["auto_confirmed"] is True


def test_admin_matrix_lists_all_users_and_projects(database, qtbot, tmp_path):
    from kraken_admin.admin_app import AdminWindow

    services, accounts, person, _config = database
    project(services, person)
    second = project(services, person, "Archived")
    services.archive_project(second["project_id"], CommandContext(str(person.id), str(uuid4()), second["revision"]))
    disabled = accounts.create_account("disabled", "Disabled", "password")
    accounts.set_enabled(disabled.account_id, False)
    window = AdminWindow()
    qtbot.addWidget(window.window)
    window._accounts, window._services = accounts, services
    window.reload()
    table = window.roles_panel.table
    assert table.rowCount() == 2
    assert table.columnCount() == 3
    assert table.item(1, 0).text().endswith("отключён")
    assert table.item(1, 1).text() == "Нет ролей"
    assert "Сопровождающий" in table.item(0, 1).text()
