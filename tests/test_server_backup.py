"""Native PostgreSQL tools against UUID databases owned only by each fixture."""
import json
import os
import socket
import subprocess
import sys
import time
from uuid import uuid4
from pathlib import Path

import pytest
from psycopg import sql
from sqlalchemy.engine import make_url

from kraken_admin.server_backup import (
    PostgresTools, ServerBackupError, _connect, create_server_backup,
    restore_server_backup, verify_server_backup,
)
from kraken_manager.infrastructure.blob import FilesystemBlobStore
from kraken_server.configuration import run_migrations


@pytest.fixture
def recovery(tmp_path):
    url = os.environ.get("KRAKEN_TEST_POSTGRES_URL")
    binaries = os.environ.get("KRAKEN_TEST_PG_BIN")
    if not url or not binaries:
        pytest.skip("KRAKEN_TEST_POSTGRES_URL and KRAKEN_TEST_PG_BIN required for native backup/restore")
    names = ["kraken_backup_test_" + uuid4().hex for _ in range(2)]
    urls = [make_url(url).set(database=name).render_as_string(hide_password=False) for name in names]
    created = []
    with _connect(url) as administrator:
        try:
            for name in names:
                administrator.execute(sql.SQL("CREATE DATABASE {} TEMPLATE template0").format(sql.Identifier(name)))
                created.append(name)
            run_migrations(urls[0])
            blobs = FilesystemBlobStore(tmp_path / "blobs")
            blob = blobs.put(b"Own recovery payload").blob
            with _connect(urls[0]) as source:
                source.execute("CREATE TABLE backup_fixture (id integer PRIMARY KEY,payload jsonb NOT NULL)")
                source.execute("INSERT INTO backup_fixture VALUES (1,%s::jsonb)",
                    (json.dumps({"blob": {"sha256": blob.sha256, "size_bytes": blob.size_bytes}, "value": "before"}),))
            yield dict(source=urls[0], target=urls[1], blobs=blobs.root, blob=blob,
                       archive=tmp_path / "archive", restored=tmp_path / "restored", tools=PostgresTools(binaries))
        finally:
            for name in created:
                assert name.startswith("kraken_backup_test_") and len(name) == len("kraken_backup_test_") + 32
                administrator.execute(sql.SQL("DROP DATABASE {} WITH (FORCE)").format(sql.Identifier(name)))


def test_native_consistent_backup_restore_and_repeat(recovery):
    r = recovery
    for folder in (".transfers", ".gateway-staging"):
        (r["blobs"] / folder).mkdir()
        (r["blobs"] / folder / "own-crashed-transfer").write_bytes(b"unfinished operational bytes")
    manifest = create_server_backup(r["source"], r["blobs"], r["archive"], tools=r["tools"])
    assert not any(entry.get("storage_path", "").startswith((".transfers/", ".gateway-staging/"))
                   for entry in manifest["entries"])
    assert (r["blobs"] / ".gateway-staging" / "own-crashed-transfer").is_file()
    assert verify_server_backup(r["archive"])["backup_id"] == manifest["backup_id"]
    result = restore_server_backup(r["archive"], r["target"], r["restored"], tools=r["tools"])
    assert result["status"] == "passed"
    with _connect(r["target"]) as target:
        assert target.execute("SELECT payload->>'value' FROM backup_fixture WHERE id=1").fetchone()[0] == "before"
    assert FilesystemBlobStore(result["blob_root"]).read(r["blob"]) == b"Own recovery payload"
    with pytest.raises(ServerBackupError):
        restore_server_backup(r["archive"], r["target"], r["restored"], tools=r["tools"])
    assert restore_server_backup(r["archive"], r["target"], r["restored"], tools=r["tools"], resume=True)["resumed"]


def test_configured_workspace_trees_relocate_and_preserve_empty_folders(recovery, tmp_path):
    from kraken_manager.infrastructure.workspace_files import WorkspaceFileService, WorkspaceRegistry
    from kraken_manager.workspace import DerivedRun, DerivedRunKind, DerivedRunState

    r = recovery
    source, derived = tmp_path / "source", tmp_path / "derived"
    source.mkdir()
    derived.mkdir()
    workspace = WorkspaceFileService(WorkspaceRegistry(r["blobs"] / "workspaces"))
    pid = str(uuid4())
    binding = workspace.create_server_project(project_id=pid, project_name="Own fixture", source_root=source, derived_root=derived)
    original_file = Path(binding.source_project_dir) / "img" / "original.jpg"
    original_file.write_bytes(b"own source bytes")
    run_path = Path(binding.derived_project_dir) / "result" / "run"
    run_path.mkdir()
    run_id = str(uuid4())
    provenance = {"path": str(original_file), "source_root": str(source)}
    workspace.registry.save_run(pid, DerivedRun(run_id=run_id, layer_id=str(uuid4()), kind=DerivedRunKind.RESULT,
        state=DerivedRunState.SUCCEEDED, path=str(run_path), plugin_id="own", operation="own", created_at="2026-10-28", provenance=provenance))
    create_server_backup(r["source"], r["blobs"], r["archive"], source_root=source, derived_root=derived, tools=r["tools"])
    result = restore_server_backup(r["archive"], r["target"], r["restored"], tools=r["tools"])
    registry = WorkspaceRegistry(Path(result["blob_root"]) / "workspaces")
    current = registry.get_project(pid)
    assert current.source_root == result["storage_roots"]["source"]
    assert current.derived_root == result["storage_roots"]["derived"]
    assert (Path(current.source_project_dir) / "img" / "original.jpg").read_bytes() == b"own source bytes"
    assert (Path(current.derived_project_dir) / "vector").is_dir()
    assert original_file.read_bytes() == b"own source bytes"
    run = registry.get_run(pid, run_id)
    assert run.path == str(Path(current.derived_project_dir) / "result" / "run")
    assert run.provenance == provenance
    restore_server_backup(r["archive"], r["target"], r["restored"], tools=r["tools"], resume=True)


def test_exported_snapshot_excludes_a_later_database_commit(recovery):
    r = recovery
    original = r["tools"].run

    def late_commit(name, *args, **kwargs):
        if name == "pg_dump":
            with _connect(r["source"]) as connection:
                connection.execute("INSERT INTO backup_fixture VALUES (2,'{\"value\":\"later\"}')")
        return original(name, *args, **kwargs)

    r["tools"].run = late_commit
    create_server_backup(r["source"], r["blobs"], r["archive"], tools=r["tools"])
    result = restore_server_backup(r["archive"], r["target"], r["restored"], tools=r["tools"])
    assert result["status"] == "passed"
    with _connect(r["source"]) as source, _connect(r["target"]) as target:
        assert source.execute("SELECT count(*) FROM backup_fixture").fetchone()[0] == 2
        assert target.execute("SELECT count(*) FROM backup_fixture").fetchone()[0] == 1


@pytest.mark.parametrize("committed", [False, True])
def test_restore_recovers_rollback_or_a_committed_reply_loss(recovery, committed):
    r = recovery
    create_server_backup(r["source"], r["blobs"], r["archive"], tools=r["tools"])
    original = r["tools"].run

    def fail(name, arguments, *args, **kwargs):
        if name == "psql":
            if committed:
                original(name, arguments, *args, **kwargs)
                raise OSError("own discarded committed reply")
            arguments = [*arguments, "--command", "SELECT 1/0"]
        return original(name, arguments, *args, **kwargs)

    r["tools"].run = fail
    with pytest.raises((OSError, ServerBackupError)):
        restore_server_backup(r["archive"], r["target"], r["restored"], tools=r["tools"])
    with _connect(r["target"]) as target:
        assert (target.execute("SELECT to_regclass('public.backup_fixture')").fetchone()[0] is not None) == committed
    calls = []

    def retry(name, *args, **kwargs):
        calls.append(name)
        return original(name, *args, **kwargs)

    r["tools"].run = retry
    restore_server_backup(r["archive"], r["target"], r["restored"], tools=r["tools"], resume=True)
    assert ("psql" in calls) != committed


def test_nonempty_database_refused_before_creating_target(recovery):
    r = recovery
    create_server_backup(r["source"], r["blobs"], r["archive"], tools=r["tools"])
    with _connect(r["target"]) as target:
        target.execute("CREATE TABLE user_data (value text)")
        target.execute("INSERT INTO user_data VALUES ('keep')")
    with pytest.raises(ServerBackupError, match="empty"):
        restore_server_backup(r["archive"], r["target"], r["restored"], tools=r["tools"])
    assert not r["restored"].exists()
    with _connect(r["target"]) as target:
        assert target.execute("SELECT value FROM user_data").fetchone()[0] == "keep"


@pytest.mark.parametrize("kind", ["checksum", "storage-traversal", "directory-traversal", "duplicate", "row-count",
                                 "relative-root", "incorrect-alias", "duplicate-directory", "unexpected"])
def test_bad_archives_fail_before_target_mutation(recovery, kind):
    r = recovery
    manifest = create_server_backup(r["source"], r["blobs"], r["archive"], tools=r["tools"])
    entry = next(value for value in manifest["entries"] if value["path"] != "database.dump")
    if kind == "checksum":
        (r["archive"] / entry["path"]).write_bytes(b"own corrupted archive")
    elif kind == "storage-traversal":
        entry["storage_path"] = "../escape"
    elif kind == "directory-traversal":
        manifest["directories"].append(dict(store="blobs", path="../escape"))
    elif kind == "duplicate":
        manifest["entries"].append(dict(entry))
    elif kind == "row-count":
        manifest["tables"][0]["rows"] = -1
    elif kind == "relative-root":
        manifest["roots"]["blobs"]["original"] = "relative-root"
    elif kind == "incorrect-alias":
        manifest["roots"]["source"] = dict(original=manifest["roots"]["blobs"]["original"], store="blobs", relative="wrong")
    elif kind == "duplicate-directory":
        manifest["directories"] += [dict(store="blobs", path="same"), dict(store="blobs", path="SAME")]
    else:
        (r["archive"] / "unexpected.txt").write_bytes(b"keep")
    (r["archive"] / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises((ServerBackupError, ValueError)):
        restore_server_backup(r["archive"], r["target"], r["restored"], tools=r["tools"])
    assert not r["restored"].exists()
    with _connect(r["target"]) as target:
        assert target.execute("SELECT to_regclass('public.backup_fixture')").fetchone()[0] is None


@pytest.mark.parametrize("nested", [False, True])
def test_equal_and_nested_storage_roots_keep_their_aliases(recovery, tmp_path, nested):
    r = recovery
    shared = tmp_path / "shared"
    shared.mkdir()
    blobs = FilesystemBlobStore(shared / "blobs")
    blobs.put(b"Own recovery payload")
    derived = shared / "derived" if nested else shared
    derived.mkdir(exist_ok=True)
    (derived / "own.txt").write_bytes(b"derived data")
    manifest = create_server_backup(r["source"], blobs.root, r["archive"], source_root=shared,
                                    derived_root=derived, tools=r["tools"])
    assert len({value["store"] for value in manifest["roots"].values()}) == 1
    result = restore_server_backup(r["archive"], r["target"], r["restored"], tools=r["tools"])
    roots = result["storage_roots"]
    assert Path(roots["blobs"]) == Path(roots["source"]) / "blobs"
    assert Path(roots["derived"]) == (Path(roots["source"]) / "derived" if nested else Path(roots["source"]))
    assert (Path(roots["derived"]) / "own.txt").read_bytes() == b"derived data"
    assert FilesystemBlobStore(roots["blobs"]).read(r["blob"]) == b"Own recovery payload"


def test_changed_restore_bytes_and_existing_function_are_preserved(recovery):
    r = recovery
    create_server_backup(r["source"], r["blobs"], r["archive"], tools=r["tools"])
    with _connect(r["target"]) as target:
        target.execute("CREATE FUNCTION own_keep() RETURNS integer LANGUAGE sql AS 'SELECT 42'")
    with pytest.raises(ServerBackupError, match="empty"):
        restore_server_backup(r["archive"], r["target"], r["restored"], tools=r["tools"])
    assert not r["restored"].exists()
    with _connect(r["target"]) as target:
        assert target.execute("SELECT own_keep()").fetchone()[0] == 42
        target.execute("DROP FUNCTION own_keep()")
    result = restore_server_backup(r["archive"], r["target"], r["restored"], tools=r["tools"])
    blob = r["blob"]
    path = Path(result["blob_root"]) / blob.sha256[:2] / blob.sha256[2:4] / blob.sha256
    path.write_bytes(b"later user bytes")
    with pytest.raises(ServerBackupError, match="target bytes differ"):
        restore_server_backup(r["archive"], r["target"], r["restored"], tools=r["tools"], resume=True)
    assert path.read_bytes() == b"later user bytes"


def test_mutating_source_does_not_publish_completed_manifest(recovery):
    r = recovery
    note = r["blobs"] / "own-note.txt"
    note.write_bytes(b"before")
    original = r["tools"].run

    def mutate(name, *args, **kwargs):
        result = original(name, *args, **kwargs)
        if name == "pg_dump":
            note.write_bytes(b"after")
        return result
    r["tools"].run = mutate
    with pytest.raises(ServerBackupError, match="changed"):
        create_server_backup(r["source"], r["blobs"], r["archive"], tools=r["tools"])
    assert not (r["archive"] / "manifest.json").exists()
    assert (r["archive"] / ".incomplete.json").is_file()


def test_second_generation_recovery_composes_historic_roots(recovery, tmp_path):
    from kraken_manager.infrastructure.root_relocations import RootRelocations

    r = recovery
    name = "kraken_backup_chain_" + uuid4().hex
    third_url = make_url(r["source"]).set(database=name).render_as_string(hide_password=False)
    with _connect(r["source"]) as administrator:
        administrator.execute(sql.SQL("CREATE DATABASE {} TEMPLATE template0").format(sql.Identifier(name)))
        try:
            create_server_backup(r["source"], r["blobs"], r["archive"], tools=r["tools"])
            first = restore_server_backup(r["archive"], r["target"], r["restored"], tools=r["tools"])
            second_archive = tmp_path / "archive-second"
            create_server_backup(r["target"], first["blob_root"], second_archive, tools=r["tools"])
            second = restore_server_backup(second_archive, third_url, tmp_path / "restored-second", tools=r["tools"])
            blob = r["blob"]
            suffix = Path(blob.sha256[:2]) / blob.sha256[2:4] / blob.sha256
            aliases = RootRelocations.load(second["blob_root"], [second["blob_root"]], required=True)
            expected = Path(second["blob_root"]) / suffix
            assert aliases.resolve(r["blobs"] / suffix) == expected
            assert aliases.resolve(Path(first["blob_root"]) / suffix) == expected
            assert aliases.resolve(expected) == expected
            assert expected.read_bytes() == b"Own recovery payload"
        finally:
            assert name.startswith("kraken_backup_chain_") and len(name) == len("kraken_backup_chain_") + 32
            administrator.execute(sql.SQL("DROP DATABASE {} WITH (FORCE)").format(sql.Identifier(name)))


def test_restored_pending_cleanup_never_deletes_original_trees(recovery, tmp_path, monkeypatch):
    import sqlalchemy as sa
    from types import SimpleNamespace
    from kraken_admin.local_admin import open_project_services
    from kraken_admin.project_deletion import ProjectDeletion
    from kraken_admin import project_deletion
    from kraken_manager.domain.identity import Principal
    from kraken_manager.infrastructure.auth.local import ScryptPasswordHasher
    from kraken_manager.infrastructure.postgres.account_store import PostgresAccountStore
    from kraken_manager.infrastructure.workspace_files import WorkspaceFileService, WorkspaceRegistry
    from kraken_server.services import CommandContext

    r = recovery
    source_root, derived_root = tmp_path / "source-files", tmp_path / "derived-files"
    source_root.mkdir()
    derived_root.mkdir()
    source_engine, target_engine = sa.create_engine(r["source"]), sa.create_engine(r["target"])
    try:
        accounts = PostgresAccountStore(source_engine, ScryptPasswordHasher())
        account = accounts.create_account("cleanup-owner", "Cleanup owner", "own password")
        accounts.grant_global_role(account.account_id, "server_admin")
        services = open_project_services(r["source"], r["blobs"], engine=source_engine)
        services.workspace_files = WorkspaceFileService(WorkspaceRegistry(r["blobs"] / "workspaces"))
        services.source_root, services.derived_root = str(source_root), str(derived_root)
        owner = Principal.local(subject="cleanup-owner", display_name="Cleanup owner", principal_id=account.account_id)
        services.identities.save(owner)
        created = services.create_project({"name": "Pending cleanup", "width": 1, "height": 1}, CommandContext(owner.id, "create", None))
        pid = created["project_id"]
        binding = services.workspace_files.registry.get_project(pid)
        original_file = Path(binding.source_project_dir) / "img" / "own.txt"
        original_file.write_bytes(b"original pending-cleanup bytes")
        request = services.request_project_deletion(pid, str(owner.id))
        source_config = SimpleNamespace(blob_root=r["blobs"], source_root=source_root, derived_root=derived_root)
        cleaner = ProjectDeletion(services, accounts, source_config)
        plan = cleaner.preview(request["request_id"])
        with monkeypatch.context() as context:
            def interrupted(*_args):
                raise OSError("own cleanup interruption after staging")
            context.setattr(project_deletion, "remove_stored_path", interrupted)
            with pytest.raises(OSError, match="interruption"):
                cleaner.execute(request["request_id"], "Pending cleanup", plan)
        staged_file = source_root / ".kraken-trash" / request["request_id"] / "0" / "img" / "own.txt"
        assert staged_file.read_bytes() == b"original pending-cleanup bytes"
        create_server_backup(r["source"], r["blobs"], r["archive"], source_root=source_root, derived_root=derived_root, tools=r["tools"])
        result = restore_server_backup(r["archive"], r["target"], r["restored"], tools=r["tools"])
        restored = open_project_services(r["target"], Path(result["blob_root"]), engine=target_engine)
        restored.workspace_files = WorkspaceFileService(WorkspaceRegistry(Path(result["blob_root"]) / "workspaces"))
        roots = result["storage_roots"]
        target_config = SimpleNamespace(blob_root=Path(roots["blobs"]), source_root=Path(roots["source"]),
            derived_root=Path(roots["derived"]), recovery_roots=Path(roots["blobs"]) / ".server/recovery-roots.json")
        target_accounts = PostgresAccountStore(target_engine, ScryptPasswordHasher())
        target_cleaner = ProjectDeletion(restored, target_accounts, target_config)
        target_plan = target_cleaner.preview(request["request_id"])
        assert all(Path(entry["path"]).is_relative_to(r["restored"]) for entry in target_plan["paths"])
        target_cleaner.execute(request["request_id"], "Pending cleanup", target_plan)
        assert staged_file.read_bytes() == b"original pending-cleanup bytes"
        assert cleaner.requests.list(pid)[0]["state"] == "failed"
        assert target_cleaner.requests.list(pid)[0]["state"] == "deleted"
    finally:
        source_engine.dispose()
        target_engine.dispose()


def test_busy_storage_and_missing_managed_object_are_fail_closed(recovery):
    r = recovery
    with _connect(r["source"]) as busy:
        busy.execute("SELECT pg_advisory_lock_shared(hashtextextended('kraken:storage',0))")
        with pytest.raises(ServerBackupError, match="busy"):
            create_server_backup(r["source"], r["blobs"], r["archive"], tools=r["tools"])
    assert not r["archive"].exists()
    blob = r["blob"]
    path = r["blobs"] / blob.sha256[:2] / blob.sha256[2:4] / blob.sha256
    path.rename(path.with_suffix(".own-hidden"))
    with pytest.raises(ServerBackupError, match="missing managed"):
        create_server_backup(r["source"], r["blobs"], r["archive"], tools=r["tools"])
    assert not r["archive"].exists()


def test_direct_storage_command_is_fenced_before_filesystem_or_database_mutation(recovery):
    import sqlalchemy as sa
    from kraken_server.project_deletion import guarded_storage
    from kraken_server.services import ConflictError

    r = recovery
    engine = sa.create_engine(r["source"], pool_size=1, max_overflow=0)
    path = r["blobs"] / "direct-command.txt"

    class DirectService:
        @guarded_storage
        def command(self):
            path.write_bytes(b"own direct command")
            with self.engine.begin() as connection:
                connection.execute(sa.text("INSERT INTO backup_fixture VALUES (2,'{}')"))

    service = DirectService()
    service.engine = engine
    try:
        with _connect(r["source"]) as exclusive:
            exclusive.execute("SELECT pg_advisory_lock(hashtextextended('kraken:storage',0))")
            with pytest.raises(ConflictError):
                service.command()
            assert not path.exists()
            assert exclusive.execute("SELECT count(*) FROM backup_fixture").fetchone()[0] == 1
        service.command()
        assert path.read_bytes() == b"own direct command"
        with engine.connect() as connection:
            assert connection.execute(sa.text("SELECT count(*) FROM backup_fixture")).scalar() == 2
    finally:
        engine.dispose()


def test_cli_restores_protected_configuration_and_authenticated_http(recovery, tmp_path, monkeypatch, capsys):
    import httpx
    import sqlalchemy as sa
    from kraken_admin.cli import _execute, _parser
    from kraken_admin.local_admin import open_project_services
    from kraken_manager.domain.identity import Principal
    from kraken_manager.infrastructure.auth.local import Argon2PasswordHasher
    from kraken_manager.infrastructure.postgres.account_store import PostgresAccountStore
    from kraken_manager.infrastructure.workspace_files import WorkspaceFileService, WorkspaceRegistry
    from kraken_server.configuration import ServerConfig, write_config
    from kraken_server.agent_resources import QueueResourceBudget
    from kraken_manager.domain.pipeline import PipelineResources
    from kraken_server.services import CommandContext

    r = recovery
    source_config = write_config(tmp_path / "source-config" / "server.toml", database_url=r["source"], blob_root=r["blobs"],
        source_root=tmp_path / "source-files", derived_root=tmp_path / "derived-files",
        queue_budget=QueueResourceBudget(2, PipelineResources(cpu=2, gpu=0, memory_mb=4096)))
    engine = sa.create_engine(r["source"])
    try:
        accounts = PostgresAccountStore(engine, Argon2PasswordHasher())
        account = accounts.create_account("recovery-owner", "Recovery owner", "own functional password")
        accounts.grant_global_role(account.account_id, "server_admin")
        services = open_project_services(r["source"], r["blobs"], engine=engine)
        services.accounts = accounts
        services.workspace_files = WorkspaceFileService(WorkspaceRegistry(r["blobs"] / "workspaces"))
        services.source_root, services.derived_root = str(source_config.source_root), str(source_config.derived_root)
        owner = Principal.local(subject="recovery-owner", display_name="Recovery owner", principal_id=account.account_id)
        services.identities.save(owner)
        project = services.create_project({"name": "Restored project", "width": 2, "height": 2}, CommandContext(owner.id, "create", None))
        binding = services.workspace_files.registry.get_project(project["project_id"])
        original = Path(binding.source_project_dir) / "img" / "исходник.jpg"
        original.write_bytes(b"own image payload")
        from PIL import Image
        images = tmp_path / "layer-inputs"
        images.mkdir()
        Image.new("RGB", (24, 20), "red").save(images / "0000.jpg")
        linked = services.create_external_layer(project["project_id"],
            dict(name="M1", type="metal", order=0, image_directory=str(images)), CommandContext(owner.id, "layer", None))
        binary_dir = str(r["tools"].directory)
        assert _execute(_parser().parse_args(["backup-server", "--config", str(source_config.path),
            "--destination", str(r["archive"]), "--pg-bin-dir", binary_dir])) == 0
        assert _execute(_parser().parse_args(["backup-verify", "--backup", str(r["archive"])])) == 0
        monkeypatch.setenv("KRAKEN_RESTORE_DATABASE_URL", r["target"])
        with socket.socket() as reserved:
            reserved.bind(("127.0.0.1", 0))
            port = reserved.getsockname()[1]
        arguments = ["restore-server", "--backup", str(r["archive"]), "--destination", str(r["restored"]),
            "--config-dir", str(tmp_path / "recovery-config"), "--pg-bin-dir", binary_dir, "--port", str(port)]
        assert _execute(_parser().parse_args(arguments)) == 0
        assert _execute(_parser().parse_args([*arguments, "--resume"])) == 0
        result = json.loads(capsys.readouterr().out.splitlines()[-1])
        restored_config = ServerConfig.load(result["config"])
        assert restored_config.database_url == r["target"]
        assert restored_config.queue_budget == source_config.queue_budget
        assert "own functional password" not in restored_config.path.read_text(encoding="utf-8")
        restored_binding = WorkspaceRegistry(restored_config.blob_root / "workspaces").get_project(project["project_id"])
        restored_image = Path(restored_binding.source_project_dir) / "img" / "исходник.jpg"
        assert restored_image.read_bytes() == original.read_bytes()
        assert restored_image != original
        environment = dict(os.environ, PYTHONIOENCODING="utf-8")
        environment.pop("KRAKEN_SERVER_DEVELOPMENT", None)
        with (tmp_path / "restored-server.log").open("w", encoding="utf-8") as log:
            process = subprocess.Popen([sys.executable, "-m", "kraken_server", "--config", result["config"], "--admin-control"],
                stdin=subprocess.PIPE, stdout=log, stderr=subprocess.STDOUT, text=True, env=environment)
            try:
                with httpx.Client(base_url=f"http://127.0.0.1:{port}", trust_env=False, timeout=3) as client:
                    deadline = time.monotonic() + 30
                    while time.monotonic() < deadline:
                        assert process.poll() is None, "Restored server exited; inspect its log"
                        try:
                            if client.get("/api/v1/health").status_code == 200:
                                break
                        except httpx.TransportError:
                            pass
                        time.sleep(.05)
                    else:
                        pytest.fail("Restored server did not reach health")
                    response = client.post("/api/v1/auth/sessions", json={"username": "recovery-owner", "password": "own functional password"})
                    assert response.status_code == 200
                    client.headers["Authorization"] = "Bearer " + response.json()["access_token"]
                    restored_project = client.get(f"/api/v1/projects/{project['project_id']}")
                    assert restored_project.status_code == 200 and restored_project.json()["name"] == "Restored project"
                    workspace = client.get(f"/api/v1/projects/{project['project_id']}/workspace")
                    assert workspace.status_code == 200
                    assert workspace.json()["source_project_dir"] == restored_binding.source_project_dir
                    layer_id = linked["layer"]["layer_id"]
                    representation_id = linked["representation"]["representation_id"]
                    prepared = client.post(f"/api/v1/projects/{project['project_id']}/layers/{layer_id}/representations/{representation_id}/prepare-inputs",
                        headers={"Idempotency-Key": "restored-materialize"})
                    assert prepared.status_code == 200, prepared.text
                    assert prepared.json() == {"prepared_frames": 1}
            finally:
                if process.poll() is None:
                    process.stdin.write("stop\n")
                    process.stdin.flush()
                    try:
                        process.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=5)
                        pytest.fail("Restored server did not stop gracefully")
            assert process.returncode == 0
        assert original.read_bytes() == b"own image payload"
    finally:
        engine.dispose()
