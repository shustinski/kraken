"""Consistent PostgreSQL/BlobStore archives and recovery into a fresh target.

Native PostgreSQL tools keep the schema, triggers and sequences intact. A
storage barrier protects managed files while pg_dump consumes an exported
snapshot. Restore records its receipt in the same transaction as the SQL,
so a lost client reply can be resolved without replaying a committed dump.
"""
from contextlib import contextmanager
from datetime import UTC, datetime
from functools import partial
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
import shutil
import stat
import subprocess
import tempfile
from uuid import UUID, uuid4

from kraken_core.safe_files import (
    contained_path, ensure_regular_directory, is_link_or_reparse,
    make_contained_directories, open_exclusive_write, open_regular_read,
)
from kraken_manager.infrastructure.filesystem._atomic import atomic_write_bytes, atomic_write_json, fsync_directory
from kraken_manager.infrastructure.filesystem.locking import ProjectFileLock, ProjectLockTimeout, kernel_file_gate
from kraken_manager.infrastructure.migration.safe_paths import validate_bundle_path

SCHEMA = "kraken-server-backup/v1"
MANIFEST = "manifest.json"
PENDING = ".incomplete.json"
RECOVERY_SCHEMA = "_kraken_recovery"
_SHA = re.compile(r"[0-9a-f]{64}\Z")
_MAX_MANIFEST = 64 * 1024 * 1024
_EMPTY_DATABASE = """NOT (
    EXISTS (SELECT 1 FROM pg_class c JOIN pg_namespace n ON n.oid=c.relnamespace
        WHERE c.relkind IN ('r','p','v','m','S','f') AND n.nspname !~ '^pg_'
          AND n.nspname <> 'information_schema')
 OR EXISTS (SELECT 1 FROM pg_proc p JOIN pg_namespace n ON n.oid=p.pronamespace
        WHERE n.nspname !~ '^pg_' AND n.nspname <> 'information_schema')
 OR EXISTS (SELECT 1 FROM pg_type t JOIN pg_namespace n ON n.oid=t.typnamespace
        WHERE n.nspname !~ '^pg_' AND n.nspname <> 'information_schema')
 OR EXISTS (SELECT 1 FROM pg_namespace WHERE nspname !~ '^pg_'
        AND nspname NOT IN ('public','information_schema'))
 OR EXISTS (SELECT 1 FROM pg_extension WHERE extname <> 'plpgsql')
)"""


class ServerBackupError(RuntimeError):
    pass


def _directory(path):
    original = Path(path).absolute()
    for part in (*reversed(original.parents), original):
        if is_link_or_reparse(part):
            raise ServerBackupError(f"Link/reparse directory is forbidden: {part}")
    return ensure_regular_directory(original)


def _relative(value):
    if not isinstance(value, str) or not value or len(value) > 4096 or any(c in value for c in "\\:\x00"):
        raise ServerBackupError("Invalid storage path")
    path = PurePosixPath(value)
    if path.is_absolute() or path.as_posix() != value or any(
        part in {".", "..", ""} or part.endswith((" ", ".")) for part in path.parts
    ):
        raise ServerBackupError("Non-canonical archive path")
    return path.parts


def _archive_path(store, relative):
    # Storage has a legitimate fixed `aux` directory. Encode file locations
    # instead of interpreting DOS device names as archive paths on Windows.
    digest = hashlib.sha256(relative.encode("utf-8")).hexdigest()
    return f"stores/{store}/{digest[:2]}/{digest[2:4]}/{digest}"


def _files(root, *, excluded=(), excluded_trees=()):
    root = _directory(root)
    result = []
    for parent, directories, files in os.walk(root, followlinks=False):
        for name in sorted((*directories, *files)):
            path = Path(parent) / name
            if is_link_or_reparse(path):
                raise ServerBackupError(f"Link/reparse entry is forbidden: {path}")
            value = path.lstat()
            if stat.S_ISREG(value.st_mode):
                if path in excluded or any(path.is_relative_to(tree) for tree in excluded_trees):
                    continue
                relative = path.relative_to(root).as_posix()
                _relative(relative)
                result.append(relative)
            elif not stat.S_ISDIR(value.st_mode):
                raise ServerBackupError(f"Special file is forbidden: {path}")
    return sorted(result)


def _hash(path, root):
    digest, size = hashlib.sha256(), 0
    with open_regular_read(path, root=root) as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
            size += len(block)
    return digest.hexdigest(), size


def _entry(root, relative):
    digest, size = _hash(contained_path(root, _relative(relative)), root)
    return {"path": relative, "sha256": digest, "size": size}


def _copy(source_root, source_relative, target_root, target_relative, work, *, expected_entry=None, transform=None):
    parts = _relative(target_relative)
    make_contained_directories(target_root, parts[:-1])
    target = contained_path(target_root, parts)
    expected = _entry(source_root, source_relative)
    if expected_entry is not None and (expected["sha256"], expected["size"]) != (expected_entry["sha256"], expected_entry["size"]):
        raise ServerBackupError(f"Archive changed before copy: {source_relative}")
    transformed = None
    if transform is not None:
        if expected["size"] > _MAX_MANIFEST:
            raise ServerBackupError("Workspace registry exceeds relocation limit")
        with open_regular_read(contained_path(source_root, _relative(source_relative)), root=source_root) as stream:
            original = stream.read()
        if (hashlib.sha256(original).hexdigest(), len(original)) != (expected["sha256"], expected["size"]):
            raise ServerBackupError("Archive registry changed during read")
        transformed = transform(original)
        expected = dict(path=source_relative, sha256=hashlib.sha256(transformed).hexdigest(), size=len(transformed))
    if target.exists():
        if _hash(target, target_root) != (expected["sha256"], expected["size"]):
            raise ServerBackupError(f"Existing target bytes differ: {target_relative}")
        return expected
    descriptor, temporary_name = tempfile.mkstemp(prefix="copy-", dir=work)
    temporary = Path(temporary_name)
    try:
        digest, size = hashlib.sha256(), 0
        input_stream = io.BytesIO(transformed) if transformed is not None else open_regular_read(
            contained_path(source_root, _relative(source_relative)), root=source_root,
        )
        with os.fdopen(descriptor, "wb") as output, input_stream as source:
            while block := source.read(1024 * 1024):
                output.write(block)
                digest.update(block)
                size += len(block)
            output.flush()
            os.fsync(output.fileno())
        if (digest.hexdigest(), size) != (expected["sha256"], expected["size"]):
            raise ServerBackupError(f"Source changed during copy: {source_relative}")
        try:
            os.link(temporary, target)
        except FileExistsError:
            raise ServerBackupError(f"Target appeared during copy: {target_relative}") from None
        except OSError:
            if os.name != "nt":
                raise
            os.rename(temporary, target)  # Windows no-replace publication.
        fsync_directory(target.parent)
        return expected
    finally:
        temporary.unlink(missing_ok=True)


def _connection_parameters(database_url):
    from sqlalchemy.engine import make_url

    url = make_url(database_url)
    if url.get_backend_name() != "postgresql":
        raise ServerBackupError("PostgreSQL is required")
    parameters = url.translate_connect_args(username="user", database="dbname")
    parameters.update(url.query)
    return parameters


class PostgresTools:
    def __init__(self, directory=None, *, timeout=3600):
        self.directory = None if directory is None else _directory(directory)
        if not isinstance(timeout, (int, float)) or timeout <= 0:
            raise ValueError("Tool timeout must be positive")
        self.timeout = timeout

    @staticmethod
    def _argument(value):
        if os.name != "nt" or not isinstance(value, os.PathLike):
            return str(value)
        text = str(Path(value).absolute())
        if text.startswith("\\\\?\\"):
            return text
        return "\\\\?\\UNC\\" + text[2:] if text.startswith("\\\\") else "\\\\?\\" + text

    def executable(self, name):
        filename = name + (".exe" if os.name == "nt" else "")
        path = (self.directory / filename) if self.directory else shutil.which(filename)
        if path is None or not Path(path).is_file():
            raise ServerBackupError(f"PostgreSQL tool {name} not found; specify --pg-bin-dir")
        return str(Path(path).absolute())

    def run(self, name, arguments, database_url=None, *, stdin=None, stdout=None):
        from psycopg.conninfo import make_conninfo

        environment = dict(os.environ)
        # The DSN is authoritative; inherited PGOPTIONS/PGSERVICE must not
        # silently alter the target or transaction behaviour.
        for key in tuple(environment):
            if key.startswith("PG"):
                environment.pop(key)
        secrets = []
        if database_url is not None:
            parameters = _connection_parameters(database_url)
            password = parameters.pop("password", None)
            if password is not None:
                environment["PGPASSWORD"] = password
                secrets.append(password)
            arguments = [*arguments, "--dbname", make_conninfo(**parameters)]
        startup = None
        if os.name == "nt":
            startup = subprocess.STARTUPINFO()
            startup.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            startup.wShowWindow = subprocess.SW_HIDE
        result = subprocess.run(
            [self._argument(Path(self.executable(name))), *map(self._argument, arguments)], env=environment,
            stderr=subprocess.PIPE, stdout=subprocess.PIPE if stdout is None else stdout,
            timeout=self.timeout, startupinfo=startup, stdin=stdin,
        )
        if result.returncode:
            message = result.stderr.decode("utf-8", errors="replace")[-4000:]
            for secret in secrets:
                if secret:
                    message = message.replace(secret, "<redacted>")
            raise ServerBackupError(f"{name} failed ({result.returncode}): {message}")
        return (result.stdout or b"").decode("utf-8", errors="replace").strip()


def _connect(database_url):
    import psycopg

    return psycopg.connect(**_connection_parameters(database_url), autocommit=True)


def _identity(connection):
    row = connection.execute("SELECT current_database(), oid FROM pg_database WHERE datname=current_database()").fetchone()
    address = connection.execute("SELECT inet_server_addr()::text, inet_server_port()").fetchone()
    return dict(database=row[0], oid=row[1], address=address[0], port=address[1])


@contextmanager
def _barrier(connection, name):
    acquired = connection.execute("SELECT pg_try_advisory_lock(hashtextextended(%s,0))", (name,)).fetchone()[0]
    if not acquired:
        raise ServerBackupError("Storage is busy; retry after active project operations finish")
    try:
        yield
    finally:
        connection.execute("SELECT pg_advisory_unlock(hashtextextended(%s,0))", (name,))


def _gate_paths(blob_root):
    return (blob_root / ".server" / "storage-backup.lock.gate",)


def _transient_trees(blob_root):
    return (blob_root / ".transfers", blob_root / ".gateway-staging")


@contextmanager
def _gateway_fence(blob_root):
    make_contained_directories(blob_root, (".server",))
    gate, = _gate_paths(blob_root)
    contained_path(blob_root, gate.relative_to(blob_root).parts)
    try:
        with kernel_file_gate(gate):
            yield
    except ProjectLockTimeout as exc:
        raise ServerBackupError("Blob Gateway storage is busy; retry after uploads finish") from exc


def _tables(connection):
    return connection.execute("""
        SELECT n.nspname,c.relname FROM pg_class c JOIN pg_namespace n ON n.oid=c.relnamespace
        WHERE c.relkind IN ('r','p') AND n.nspname !~ '^pg_'
          AND n.nspname NOT IN ('information_schema',%s) ORDER BY 1,2
    """, (RECOVERY_SCHEMA,)).fetchall()


def _database_inventory(connection):
    from psycopg import sql

    tables, blobs, external = [], {}, {}

    def reference(digest, size=None):
        if not isinstance(digest, str) or not _SHA.fullmatch(digest):
            raise ServerBackupError("Invalid managed blob reference in database")
        if size is not None and (type(size) is not int or size < 0):
            raise ServerBackupError("Invalid managed blob size in database")
        previous = blobs.get(digest)
        if previous is not None and size is not None and previous != size:
            raise ServerBackupError("Conflicting sizes for a managed blob")
        if digest not in blobs or size is not None:
            blobs[digest] = size

    def walk(value, depth=0):
        if depth > 100:
            raise ServerBackupError("Database JSON nesting exceeds backup limit")
        if isinstance(value, dict):
            if "blob" in value and isinstance(value["blob"], dict):
                blob = value["blob"]
                if "sha256" in blob:
                    reference(blob["sha256"], blob.get("size_bytes"))
            for key in ("blob_sha256", "bundle_sha256"):
                if isinstance(value.get(key), str):
                    reference(value[key])
            if "uri" in value and "fingerprint_sha256" in value:
                external[str(value["uri"])] = {
                    key: value[key] for key in ("uri", "fingerprint_sha256", "observed_size_bytes") if key in value
                }
            for item in value.values():
                walk(item, depth + 1)
        elif isinstance(value, list):
            for item in value:
                walk(item, depth + 1)

    for schema, table in _tables(connection):
        count = 0
        query = sql.SQL("SELECT row_to_json(t) FROM {}.{} t").format(sql.Identifier(schema), sql.Identifier(table))
        with connection.cursor(name="backup_" + uuid4().hex) as cursor:
            cursor.execute(query)
            for (row,) in cursor:
                count += 1
                walk(row)
                if table == "agent_job_outputs":
                    reference(row["sha256"], row.get("size_bytes"))
        tables.append(dict(schema=schema, name=table, rows=count))
    return tables, blobs, list(external.values())


def _storage_roots(blob_root, source_root, derived_root):
    originals = {"blobs": _directory(blob_root)}
    if (source_root is None) != (derived_root is None):
        raise ServerBackupError("Both source and derived roots are required together")
    if source_root is not None:
        originals.update(source=_directory(source_root), derived=_directory(derived_root))
    stores, roots = {}, {}
    for name, root in sorted(originals.items(), key=lambda pair: len(pair[1].parts)):
        store = next((key for key, base in stores.items() if root.is_relative_to(base)), None)
        if store is None:
            stores[name], store = root, name
        relative = root.relative_to(stores[store]).as_posix()
        roots[name] = dict(original=str(root), store=store, relative="" if relative == "." else relative)
    return stores, roots


def _root_prefix(root):
    return "/".join(part for part in ("stores", root["store"], root["relative"]) if part)


def _relocate_registry(encoded, roots, target):
    value = json.loads(encoded)
    changed = False

    def relocate(text):
        for root in sorted(roots.values(), key=lambda item: len(item["original"]), reverse=True):
            cls = PureWindowsPath if re.match(r"^(?:[A-Za-z]:|\\\\)", root["original"]) else PurePosixPath
            try:
                relative = cls(text).relative_to(cls(root["original"]))
            except ValueError:
                continue
            if relative.parts:
                _relative(relative.as_posix())
            return str(target.joinpath(*_relative(_root_prefix(root)), *relative.parts))
        return text

    def fields(item, names):
        nonlocal changed
        if item is None:
            return
        if not isinstance(item, dict):
            raise ServerBackupError("Invalid workspace registry record")
        for key in names:
            child = item.get(key)
            if isinstance(child, str) and child:
                new = relocate(child)
                changed |= new != child
                item[key] = new
    if (not isinstance(value, dict) or value.get("schema_version", 1) != 1 or not isinstance(value.get("layers", {}), dict)
            or not isinstance(value.get("runs", []), list)):
        raise ServerBackupError("Invalid workspace registry")
    fields(value.get("project"), ("source_root", "derived_root", "source_project_dir", "derived_project_dir"))
    for layer in value.get("layers", {}).values():
        fields(layer, ("image_directory", "ssc_directory", "prv_directory", "aux_directory", "import_root"))
    for run in value.get("runs", []):
        fields(run, ("path",))
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode() if changed else encoded


def _recovery_metadata(archive, manifest, target):
    from kraken_manager.infrastructure.root_relocations import SCHEMA as ROOT_SCHEMA, original_path

    roots = manifest["roots"]
    blob = roots["blobs"]
    original_relative = "/".join(part for part in (blob["relative"], ".server/recovery-roots.json") if part)
    archived = _archive_path(blob["store"], original_relative)
    previous = []
    entry = next((entry for entry in manifest["entries"] if entry["path"] == archived), None)
    if entry is not None:
        if entry["size"] > 1024 * 1024:
            raise ServerBackupError("Prior relocation metadata exceeds size limit")
        with open_regular_read(contained_path(archive, _relative(archived)), root=archive) as stream:
            encoded = stream.read()
        if (hashlib.sha256(encoded).hexdigest(), len(encoded)) != (entry["sha256"], entry["size"]):
            raise ServerBackupError("Prior relocation metadata changed")
        value = json.loads(encoded)
        if not isinstance(value, dict) or value.get("schema") != ROOT_SCHEMA or not isinstance(value.get("mappings"), list):
            raise ServerBackupError("Unsupported prior storage relocations")
        previous = value["mappings"]
        if len(previous) > 10000:
            raise ServerBackupError("Too many prior storage relocations")
    mappings = {}
    for item in previous:
        old_target = original_path(item["target"])
        for root in sorted(roots.values(), key=lambda item: len(item["original"]), reverse=True):
            try:
                relative = old_target.relative_to(original_path(root["original"]))
            except (ValueError, TypeError):
                continue
            if relative.parts:
                _relative(relative.as_posix())
            mappings[original_path(item["original"])] = target.joinpath(*_relative(_root_prefix(root)), *relative.parts)
            break
        else:
            raise ServerBackupError("Prior relocation points outside backed-up storage roots")
    for root in roots.values():
        mappings[original_path(root["original"])] = target.joinpath(*_relative(_root_prefix(root)))
    encoded = (json.dumps(dict(schema=ROOT_SCHEMA, mappings=[dict(original=str(source), target=str(destination))
        for source, destination in sorted(mappings.items(), key=lambda pair: str(pair[0]))]),
        ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
    return _root_prefix(blob) + "/.server/recovery-roots.json", encoded


def _registry_path(relative):
    parts = PurePosixPath(relative).parts
    if len(parts) != 3 or parts[0] != "projects" or parts[-1] != "workspace.json":
        return False
    try:
        UUID(parts[1])
    except ValueError:
        return False
    return True


def create_server_backup(database_url, blob_root, destination, *, source_root=None, derived_root=None, tools=None, configuration=None):
    tools = tools or PostgresTools()
    tools.executable("pg_dump")
    source = _directory(blob_root)
    transient = _gate_paths(source)
    transient_trees = _transient_trees(source)
    stores, roots = _storage_roots(blob_root, source_root, derived_root)
    destination = Path(destination).absolute()
    parent = _directory(destination.parent)
    destination = parent / destination.name
    if destination.exists() or destination.is_symlink():
        raise ServerBackupError("Backup requires a new directory; completed/incomplete archives are preserved")
    if any(destination.is_relative_to(root) or root.is_relative_to(destination) for root in stores.values()):
        raise ServerBackupError("Backup destination must be separate from storage roots")
    with _gateway_fence(source), _connect(database_url) as connection, _barrier(connection, "kraken:storage"):
        connection.execute("SET idle_in_transaction_session_timeout=0")
        connection.execute("SET idle_session_timeout=0")
        with connection.transaction():
            connection.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY")
            snapshot = connection.execute("SELECT pg_export_snapshot()").fetchone()[0]
            identity = _identity(connection)
            version = int(connection.execute("SHOW server_version_num").fetchone()[0])
            tables, references, external = _database_inventory(connection)
            if not any(t["name"] == "alembic_version" for t in tables):
                raise ServerBackupError("Database has no Kraken migration version")
            inventories = {name: _files(root, excluded=transient, excluded_trees=transient_trees) for name, root in stores.items()}
            fingerprints = {
                (name, relative): _entry(stores[name], relative)
                for name, files_ in inventories.items() for relative in files_
            }
            files = _files(source, excluded=transient, excluded_trees=transient_trees)
            file_names = set(files)
            for digest, size in references.items():
                relative = f"{digest[:2]}/{digest[2:4]}/{digest}"
                if relative not in file_names:
                    raise ServerBackupError(f"Database references missing managed blob: {digest}")
                actual, actual_size = _hash(contained_path(source, _relative(relative)), source)
                if actual != digest or (size is not None and actual_size != size):
                    raise ServerBackupError(f"Database references invalid managed blob: {digest}")
            destination.mkdir(mode=0o700, exist_ok=False)
            archive_id = str(uuid4())
            atomic_write_json(destination / PENDING, dict(schema=SCHEMA, backup_id=archive_id), overwrite=False)
            work = destination / ".work"
            work.mkdir(mode=0o700)
            dump = work / "database.dump"
            with open_exclusive_write(dump) as stream:
                tools.run("pg_dump", ["--no-password", "--format=custom", "--no-owner", "--no-acl",
                    "--exclude-schema=" + RECOVERY_SCHEMA, "--snapshot=" + snapshot], database_url, stdout=stream)
                stream.flush()
                os.fsync(stream.fileno())
            os.rename(dump, destination / "database.dump")
            entries = [_entry(destination, "database.dump")]
            directories = []
            for name, root in stores.items():
                for parent_dir, names, _files_ignored in os.walk(root, followlinks=False):
                    for child in names:
                        relative = (Path(parent_dir) / child).relative_to(root).as_posix()
                        _relative(relative)
                        directories.append(dict(store=name, path=relative))
                for relative in inventories[name]:
                    archived = _archive_path(name, relative)
                    expected = _copy(root, relative, destination, archived, work,
                                     expected_entry=fingerprints[name, relative])
                    entries.append({**expected, "path": archived, "store": name, "storage_path": relative})
            # New uploads may add unreferenced immutable objects. Existing
            # files must remain identical; all their bytes belong to this cut.
            for entry in entries[1:]:
                name, relative = entry["store"], entry["storage_path"]
                root = stores[name]
                if _hash(contained_path(root, _relative(relative)), root) != (entry["sha256"], entry["size"]):
                    raise ServerBackupError(f"Source changed during backup: {relative}")
            for name, root in stores.items():
                for relative in set(_files(root, excluded=transient, excluded_trees=transient_trees)) - set(inventories[name]):
                    path = contained_path(root, _relative(relative))
                    try:
                        parts = path.relative_to(source).parts
                    except ValueError:
                        parts = ()
                    if not (len(parts) == 3 and _SHA.fullmatch(parts[-1]) and parts[:2] == (parts[-1][:2], parts[-1][2:4])):
                        raise ServerBackupError("Mutable storage inventory changed during backup")
            connection.execute("SELECT 1")  # A lost barrier connection must not publish a completed cut.
            work.rmdir()
            _validate_managed_entries(entries, roots, references)
            manifest = dict(schema=SCHEMA, backup_id=archive_id, created_at=datetime.now(UTC).isoformat(),
                source=identity, postgres_version_num=version, tables=tables, entries=entries,
                roots=roots, directories=directories, required_blobs=references, external_references=external,
                configuration=configuration,
                excluded_operational_paths=[".server/storage-backup.lock.gate", ".transfers/", ".gateway-staging/"],
                excluded_operational_schema=RECOVERY_SCHEMA,
                scope="PostgreSQL and configured storage roots; external URI bytes outside those roots are not included")
            atomic_write_json(destination / MANIFEST, manifest, overwrite=False)
            (destination / PENDING).unlink()
            fsync_directory(destination)
    verify_server_backup(destination)
    return manifest


def verify_server_backup(directory):
    root = _directory(directory)
    path = contained_path(root, (MANIFEST,))
    if path.stat().st_size > _MAX_MANIFEST:
        raise ServerBackupError("Backup manifest exceeds size limit")
    with open_regular_read(path, root=root) as stream:
        manifest = json.load(stream)
    if not isinstance(manifest, dict) or manifest.get("schema") != SCHEMA:
        raise ServerBackupError("Unsupported server backup manifest")
    UUID(manifest["backup_id"])
    if type(manifest.get("postgres_version_num")) is not int or manifest["postgres_version_num"] < 100000:
        raise ServerBackupError("Invalid PostgreSQL source version")
    roots = manifest.get("roots")
    if not isinstance(roots, dict) or "blobs" not in roots or set(roots) - {"blobs", "source", "derived"}:
        raise ServerBackupError("Invalid storage root inventory")
    from kraken_manager.infrastructure.root_relocations import original_path

    for storage in roots.values():
        if (not isinstance(storage, dict) or storage.get("store") not in roots
                or not isinstance(storage.get("original"), str) or not storage["original"]
                or not isinstance(storage.get("relative"), str)):
            raise ServerBackupError("Invalid storage root")
        if storage.get("relative"):
            _relative(storage["relative"])
        _relative(_root_prefix(storage))
        base = roots[storage["store"]]
        if base.get("relative") != "" or base.get("store") != storage["store"]:
            raise ServerBackupError("Invalid storage alias")
        source, base_path = original_path(storage["original"]), original_path(base["original"])
        if source != base_path.joinpath(*PurePosixPath(storage["relative"]).parts):
            raise ServerBackupError("Storage alias disagrees with original root")
    stores = {name for name, storage in roots.items() if storage["store"] == name}
    for name in stores:
        for other in stores - {name}:
            if original_path(roots[name]["original"]).is_relative_to(original_path(roots[other]["original"])):
                raise ServerBackupError("Overlapping physical storage roots")
    tables = manifest.get("tables")
    if not isinstance(tables, list) or not tables or len(tables) > 100_000:
        raise ServerBackupError("Invalid database table inventory")
    table_keys = set()
    for table in tables:
        if (not isinstance(table, dict) or not isinstance(table.get("schema"), str) or not table["schema"]
                or not isinstance(table.get("name"), str) or not table["name"]
                or type(table.get("rows")) is not int or table["rows"] < 0):
            raise ServerBackupError("Invalid database table")
        key = (table["schema"], table["name"])
        if key in table_keys or table["schema"].startswith("pg_") or table["schema"] in {RECOVERY_SCHEMA, "information_schema"}:
            raise ServerBackupError("Duplicate/reserved database table")
        table_keys.add(key)
    directories = manifest.get("directories")
    if not isinstance(directories, list) or len(directories) > 1_000_000:
        raise ServerBackupError("Invalid directory inventory")
    directory_names = set()
    for item in directories:
        if not isinstance(item, dict) or item.get("store") not in stores:
            raise ServerBackupError("Unknown directory store")
        _relative(item["path"])
        key = (item["store"], item["path"].casefold())
        if key in directory_names:
            raise ServerBackupError("Duplicate/case-colliding directory path")
        directory_names.add(key)
    entries = manifest.get("entries")
    if not isinstance(entries, list) or not entries or len(entries) > 1_000_000:
        raise ServerBackupError("Invalid backup inventory")
    names, folded, storage_names = set(), set(), set()
    for entry in entries:
        relative, digest, size = entry["path"], entry["sha256"], entry["size"]
        validate_bundle_path(relative)
        _relative(relative)
        if relative != "database.dump" and not (
            relative.startswith("stores/") and len(relative.split("/")) >= 3 and relative.split("/")[1] in roots
        ):
            raise ServerBackupError("Unexpected backup entry kind")
        if relative != "database.dump":
            store, storage_path = entry.get("store"), entry.get("storage_path")
            if store not in stores:
                raise ServerBackupError("Unknown file store")
            _relative(storage_path)
            if relative != _archive_path(store, storage_path):
                raise ServerBackupError("Archive location does not match storage path")
            key = (store, storage_path.casefold())
            if key in storage_names:
                raise ServerBackupError("Duplicate/case-colliding storage path")
            storage_names.add(key)
        if relative in names or relative.casefold() in folded:
            raise ServerBackupError("Duplicate/case-colliding backup path")
        names.add(relative)
        folded.add(relative.casefold())
        if not isinstance(digest, str) or not _SHA.fullmatch(digest) or type(size) is not int or size < 0:
            raise ServerBackupError("Invalid backup checksum/size")
        if _hash(contained_path(root, _relative(relative)), root) != (digest, size):
            raise ServerBackupError(f"Backup checksum mismatch: {relative}")
    if "database.dump" not in names or set(_files(root)) - {MANIFEST, PENDING} != names:
        raise ServerBackupError("Missing/unexpected files in backup")
    _validate_managed_entries(entries, roots, manifest.get("required_blobs", {}))
    return manifest


def _validate_managed_entries(entries, roots, references):
    by_path = {entry["path"]: entry for entry in entries}
    blob_root = roots["blobs"]
    for entry in entries:
        if entry.get("store") == blob_root["store"]:
            try:
                relative = PurePosixPath(entry["storage_path"]).relative_to(PurePosixPath(blob_root["relative"]))
            except ValueError:
                continue
            parts = list(relative.parts)
            if len(parts) == 3 and _SHA.fullmatch(parts[-1]) and parts[:2] == [parts[-1][:2], parts[-1][2:4]]:
                if entry["sha256"] != parts[-1]:
                    raise ServerBackupError("Corrupt content-addressed object in backup")
    if not isinstance(references, dict):
        raise ServerBackupError("Invalid managed blob inventory")
    for digest, size in references.items():
        if not _SHA.fullmatch(digest):
            raise ServerBackupError("Invalid managed reference")
        if size is not None and (type(size) is not int or size < 0):
            raise ServerBackupError("Invalid managed blob size")
        storage_path = "/".join(p for p in (blob_root["relative"], digest[:2], digest[2:4], digest) if p)
        relative = _archive_path(blob_root["store"], storage_path)
        entry = by_path.get(relative)
        if entry is None or entry["sha256"] != digest or (size is not None and entry["size"] != size):
            raise ServerBackupError("Backup does not cover a required managed blob")


def _restore_marker(connection, backup_id):
    if connection.execute("SELECT to_regclass('_kraken_recovery.restore_receipts')").fetchone()[0] is None:
        return None
    return connection.execute(
        "SELECT manifest_sha256,target_root FROM _kraken_recovery.restore_receipts WHERE backup_id=%s", (backup_id,),
    ).fetchone()


def _check_restored_tables(connection, expected):
    from psycopg import sql

    actual = _tables(connection)
    if set(actual) != {(table["schema"], table["name"]) for table in expected}:
        raise ServerBackupError("Restored database table inventory differs")
    for table in expected:
        count = connection.execute(sql.SQL("SELECT count(*) FROM {}.{}").format(
            sql.Identifier(table["schema"]), sql.Identifier(table["name"]),
        )).fetchone()[0]
        if count != table["rows"]:
            raise ServerBackupError(f"Restored table row count differs: {table['schema']}.{table['name']}")


def restore_server_backup(directory, database_url, blob_root, *, tools=None, resume=False):
    from psycopg import sql

    tools = tools or PostgresTools()
    for tool in ("pg_restore", "psql"):
        tools.executable(tool)
    archive = _directory(directory)
    manifest = verify_server_backup(archive)
    manifest_sha, _ = _hash(archive / MANIFEST, archive)
    target = Path(blob_root).absolute()
    parent = _directory(target.parent)
    target = parent / target.name
    if target == archive or target.is_relative_to(archive) or archive.is_relative_to(target):
        raise ServerBackupError("Restore destination must be separate from archive")
    relocation_relative, relocation_bytes = _recovery_metadata(archive, manifest, target)
    receipt_path = contained_path(parent, (f".{target.name}.kraken-restore.json",))
    lock_path = contained_path(parent, (f".{target.name}.kraken-restore.lock",))
    contained_path(parent, (lock_path.name + ".gate",))
    with ProjectFileLock(lock_path).hold(0), _connect(database_url) as connection, _barrier(connection, "kraken:server-restore"):
        identity = _identity(connection)
        version = int(connection.execute("SHOW server_version_num").fetchone()[0])
        if version // 10000 < manifest["postgres_version_num"] // 10000:
            raise ServerBackupError("Restore PostgreSQL major must not be older than the backup")
        pairing = dict(backup_id=manifest["backup_id"], manifest_sha256=manifest_sha, target_root=str(target), database=identity)
        if receipt_path.exists():
            with open_regular_read(receipt_path, root=parent) as stream:
                receipt = json.load(stream)
            if not resume or any(receipt.get(key) != value for key, value in pairing.items()):
                raise ServerBackupError("Existing restore receipt requires --resume and the same archive/target/database")
            if (receipt.get("schema") != "kraken-server-restore/v1" or not isinstance(receipt.get("work_directory"), str)
                    or not re.fullmatch(re.escape(f".{target.name}.kraken-restore-") + r"[0-9a-f]{32}", receipt["work_directory"])):
                raise ServerBackupError("Invalid restore working directory receipt")
            work = contained_path(parent, (receipt["work_directory"],))
            _directory(work)
            _directory(target)
        else:
            if target.exists() or target.is_symlink():
                raise ServerBackupError("Restore BlobStore must be a new directory")
            if not connection.execute("SELECT " + _EMPTY_DATABASE).fetchone()[0]:
                raise ServerBackupError("Restore database must be empty; existing data is never overwritten")
            work = parent / f".{target.name}.kraken-restore-{uuid4().hex}"
            work.mkdir(mode=0o700, exist_ok=False)
            target.mkdir(mode=0o700, exist_ok=False)
            receipt = dict(**pairing, schema="kraken-server-restore/v1", work_directory=work.name, state="copying")
            atomic_write_json(receipt_path, receipt, overwrite=False)
        marker = _restore_marker(connection, manifest["backup_id"])
        if marker is not None and tuple(marker) != (manifest_sha, str(target)):
            raise ServerBackupError("Database restore receipt belongs to another target")
        if marker is None and not connection.execute("SELECT " + _EMPTY_DATABASE).fetchone()[0]:
            raise ServerBackupError("Target database is nonempty without this restore receipt")
        expected_files = set()
        for root in manifest["roots"].values():
            make_contained_directories(target, _relative(_root_prefix(root)))
        for item in manifest["directories"]:
            make_contained_directories(target, ("stores", item["store"], *_relative(item["path"])))
        registry_prefix = _root_prefix(manifest["roots"]["blobs"]) + "/workspaces/"
        for entry in manifest["entries"]:
            if entry["path"].startswith("stores/"):
                relative = f"stores/{entry['store']}/{entry['storage_path']}"
                expected_files.add(relative)
                transform = None
                if relative.startswith(registry_prefix) and _registry_path(relative[len(registry_prefix):]):
                    transform = partial(_relocate_registry, roots=manifest["roots"], target=target)
                if relative == relocation_relative:
                    def transform(_data):
                        return relocation_bytes
                _copy(archive, entry["path"], target, relative, work, expected_entry=entry, transform=transform)
        make_contained_directories(target, _relative(relocation_relative)[:-1])
        relocation_path = contained_path(target, _relative(relocation_relative))
        if relocation_path.exists():
            if _hash(relocation_path, target) != (hashlib.sha256(relocation_bytes).hexdigest(), len(relocation_bytes)):
                raise ServerBackupError("Existing relocation metadata differs")
        else:
            atomic_write_bytes(relocation_path, relocation_bytes, overwrite=False)
        expected_files.add(relocation_relative)
        target_blob = target.joinpath(*_relative(_root_prefix(manifest["roots"]["blobs"])))
        if set(_files(target, excluded=_gate_paths(target_blob), excluded_trees=_transient_trees(target_blob))) != expected_files:
            raise ServerBackupError("Restore target contains unexpected files")
        if marker is None:
            script = work / f"restore-{uuid4().hex}.sql"
            dump_entry = next(entry for entry in manifest["entries"] if entry["path"] == "database.dump")
            with open_regular_read(archive / "database.dump", root=archive) as stream:
                digest = hashlib.sha256()
                size = 0
                while block := stream.read(1024 * 1024):
                    digest.update(block)
                    size += len(block)
                if (digest.hexdigest(), size) != (dump_entry["sha256"], dump_entry["size"]):
                    raise ServerBackupError("Database archive changed before restore")
                stream.seek(0)
                with open_exclusive_write(script) as output:
                    tools.run("pg_restore", ["--no-owner", "--no-acl", "--file", "-"], stdin=stream, stdout=output)
                    output.flush()
                    os.fsync(output.fileno())
            receipt_sql = sql.SQL("""
                CREATE SCHEMA IF NOT EXISTS _kraken_recovery;
                CREATE TABLE IF NOT EXISTS _kraken_recovery.restore_receipts (
                    backup_id uuid PRIMARY KEY,manifest_sha256 text NOT NULL,target_root text NOT NULL,
                    restored_at timestamptz NOT NULL DEFAULT now());
                INSERT INTO _kraken_recovery.restore_receipts(backup_id,manifest_sha256,target_root)
                VALUES ({},{},{});
            """).format(sql.Literal(manifest["backup_id"]), sql.Literal(manifest_sha), sql.Literal(str(target))).as_string(connection)
            precondition = "DO $$ BEGIN IF NOT (" + _EMPTY_DATABASE + ") THEN " \
                "RAISE EXCEPTION 'Restore requires an empty database'; END IF; END $$;"
            try:
                with open_regular_read(script, root=work) as script_input:
                    tools.run("psql", ["--no-password", "--no-psqlrc", "--single-transaction", "--set", "ON_ERROR_STOP=1",
                        "--command", precondition, "--file", "-", "--command", receipt_sql], database_url, stdin=script_input)
            finally:
                script.unlink(missing_ok=True)
        marker = _restore_marker(connection, manifest["backup_id"])
        if marker is None or tuple(marker) != (manifest_sha, str(target)):
            raise ServerBackupError("Committed restore receipt is missing")
        _check_restored_tables(connection, manifest["tables"])
        receipt["state"] = "complete"
        receipt["completed_at"] = datetime.now(UTC).isoformat()
        atomic_write_json(receipt_path, receipt)
        restored_roots = {name: str(target.joinpath(*_relative(_root_prefix(root)))) for name, root in manifest["roots"].items()}
        return dict(status="passed", backup_id=manifest["backup_id"], blob_root=restored_roots["blobs"],
                    storage_roots=restored_roots, tables=len(manifest["tables"]), receipt=str(receipt_path), resumed=resume)
