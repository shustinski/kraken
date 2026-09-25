"""Local-only, resumable physical cleanup after an administrator's approval."""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from pathlib import Path

import sqlalchemy as sa

from kraken_server.project_deletion import DeletionRequests, lock
from kraken_server.services import ConflictError

from .deletion_confirmation import (
    DeletionAuthorization,
    DeletionPolicy,
    remove_stored_path,
    stage_stored_path,
)


def safe_path(path: Path, root: Path) -> Path:
    root = root.absolute()
    path = path.absolute()
    if path == root or not path.is_relative_to(root):
        raise ValueError(f"Путь вне разрешённого корня: {path}")
    for part in (path, *path.parents):
        if part.is_symlink() or part.is_junction():
            raise ValueError(f"Ссылка в пути удаления: {part}")
    resolved = path.resolve()
    if resolved == root.resolve() or not resolved.is_relative_to(root.resolve()):
        raise ValueError(f"Небезопасный путь: {path}")
    if path.is_dir():
        for parent, dirs, files in path.walk(follow_symlinks=False):
            for name in (*dirs, *files):
                child = parent / name
                if child.is_symlink() or child.is_junction():
                    raise ValueError(f"Ссылка внутри удаляемого каталога: {child}")
    return resolved


def _digests(value: object) -> set[str]:
    if isinstance(value, dict):
        return set().union(*(_digests(item) for item in value.values())) if value else set()
    if isinstance(value, (list, tuple)):
        return set().union(*(_digests(item) for item in value)) if value else set()
    return {value} if isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) else set()


def project_predicate(table, project_id: str, *, other: bool = False, visited=frozenset()):
    """Include dependent rows such as job outputs and analysis partitions."""
    if "project_id" in table.c:
        return table.c.project_id != project_id if other else table.c.project_id == project_id
    if table.name in visited:
        return None
    conditions = []
    for foreign_key in table.foreign_keys:
        parent = foreign_key.column.table
        predicate = project_predicate(parent, project_id, other=other, visited=visited | {table.name})
        if predicate is not None:
            conditions.append(foreign_key.parent.in_(sa.select(foreign_key.column).where(predicate)))
    return sa.or_(*conditions) if conditions else None


def assert_unshared(registry, project_id: str, paths: list[Path]) -> None:
    for manifest in (registry.catalog_root / "projects").glob("*/workspace.json"):
        other = json.loads(manifest.read_text(encoding="utf-8"))
        if str((other.get("project") or {}).get("project_id", "")) == project_id:
            continue
        def check(value):
            if isinstance(value, dict):
                for key, item in value.items():
                    if key not in {"source_root", "derived_root"}:
                        check(item)
            elif isinstance(value, list):
                for item in value:
                    check(item)
            elif isinstance(value, str) and Path(value).is_absolute():
                resolved = Path(value).resolve()
                for candidate in paths:
                    target = candidate.resolve()
                    if resolved == target or resolved.is_relative_to(target) or target.is_relative_to(resolved):
                        raise ValueError(f"Папка используется другим проектом: {resolved}")
        check(other)


class ProjectDeletion:
    def __init__(self, services, accounts, config) -> None:
        self.services = services
        self.accounts = accounts
        self.config = config
        self.requests = DeletionRequests(services.engine)
        self.table = self.requests.table

    def _schema(self, connection):
        metadata = sa.MetaData()
        metadata.reflect(connection)
        return metadata

    def preview(self, request_id: str) -> dict:
        rows = [row for row in self.requests.list() if row["request_id"] == request_id]
        if not rows:
            raise ValueError("Заявка не найдена")
        row = rows[0]
        if row["state"] in {"deleting", "failed"}:
            return row["plan"]
        if row["state"] != "pending":
            raise ConflictError("Заявка уже обработана")
        project_id = str(row["project_id"])
        project = self.services.get_project(project_id)
        registry = self.services.workspace_files.registry
        binding = registry.get_project(project_id)
        if binding is None:
            raise ValueError("Нет реестра папок проекта; принадлежность файлов нельзя проверить")
        paths = [
            (Path(binding.source_project_dir), self.config.source_root),
            (Path(binding.derived_project_dir), self.config.derived_root),
            (registry.catalog_root / "projects" / project_id, registry.catalog_root / "projects"),
        ]
        for candidate, root in paths:
            safe_path(candidate, root)
        assert_unshared(registry, project_id, [path for path, _ in paths])
        # External layer directories must also be owned by this project's two trees.
        own = json.loads(registry._path(project_id).read_text(encoding="utf-8"))
        for layer in own.get("layers", {}).values():
            for key in ("image_directory", "ssc_directory", "prv_directory", "aux_directory", "import_root"):
                if layer.get(key):
                    directory = Path(layer[key]).resolve()
                    if not any(directory.is_relative_to(path.resolve()) for path, _ in paths[:2]):
                        raise ValueError(f"Внешняя папка не принадлежит проекту: {directory}")
        return {"project_id": project_id, "name": project["name"],
                "paths": [{"path": str(path), "root": str(root)} for path, root in paths]}

    def reject(self, request_id: str) -> None:
        self.requests.reject(request_id)
        with self.services.engine.begin() as connection:
            self.accounts._record_audit(connection, None, "project.deletion_rejected", None,
                                        {"request_id": request_id})

    def execute(
        self,
        request_id: str,
        confirmation_name: str,
        expected_plan: dict,
        *,
        auto_confirm: bool = False,
        reason: str | None = None,
    ) -> None:
        from .project_roles import administrator

        administrator(self.accounts)
        if reason is not None:
            self.requests.update_reason(request_id, reason)
        engine = self.services.engine
        # A session lock serializes two Admin windows across all cleanup steps.
        with engine.connect() as owner:
            if owner.dialect.name == "postgresql":
                acquired = owner.execute(sa.text("SELECT pg_try_advisory_lock(hashtextextended(:key,0))"),
                                         {"key": f"delete:{request_id}"}).scalar()
                owner.commit()
                if not acquired:
                    raise ConflictError("Удаление уже выполняется в другом окне")
            try:
                self._execute(request_id, confirmation_name, expected_plan, auto_confirm=auto_confirm)
            except Exception:
                rows = [row for row in self.requests.list() if row["request_id"] == request_id]
                if rows and rows[0]["state"] == "pending":
                    (self.config.blob_root / ".deleting" / str(rows[0]["project_id"])).unlink(missing_ok=True)
                raise
            finally:
                if owner.dialect.name == "postgresql":
                    owner.execute(sa.text("SELECT pg_advisory_unlock(hashtextextended(:key,0))"),
                                  {"key": f"delete:{request_id}"})
                    owner.commit()

    def _execute(
        self,
        request_id: str,
        confirmation_name: str,
        expected_plan: dict,
        *,
        auto_confirm: bool,
    ) -> None:
        engine = self.services.engine
        with engine.begin() as connection:
            row = connection.execute(sa.select(self.table).where(
                self.table.c.request_id == request_id).with_for_update()).mappings().one()
            project_id = str(row["project_id"])
            if row["state"] not in {"pending", "deleting", "failed"}:
                raise ConflictError("Заявка уже обработана")
            if not lock(connection, f"project:{project_id}", attempt=True):
                raise ConflictError("В проекте выполняется операция. Повторите после её завершения.")
            schema = self._schema(connection)
            for name, terminal in (("background_jobs", ("succeeded", "completed", "failed", "cancelled")),
                                   ("analysis_runs", ("completed", "partial", "failed", "cancelled"))):
                table = schema.tables.get(name)
                if table is not None and connection.execute(sa.select(table.c.project_id).where(
                    table.c.project_id == project_id, table.c.state.not_in(terminal),
                )).first():
                    raise ConflictError("В проекте есть незавершённые задания. Завершите или отмените их.")
            plan = self.preview(request_id)
            if plan != expected_plan:
                raise ConflictError("Проект изменился. Повторно просмотрите и подтвердите удаление.")
            policy_enabled = DeletionPolicy(engine).enabled(connection)
            authorization = DeletionAuthorization.grant(
                name_matches=confirmation_name == plan["name"],
                auto_confirm=auto_confirm,
                policy_enabled=policy_enabled,
            )
            confirmed_automatically = bool(auto_confirm and policy_enabled)
            deletion_reason = str(row["reason"] or "")
            marker = self.config.blob_root / ".deleting" / project_id
            marker.parent.mkdir(parents=True, exist_ok=True)
            marker.touch()
            transfers = self.config.blob_root / ".transfers" / project_id
            if transfers.exists() and any(transfers.iterdir()):
                if row["state"] == "pending":
                    marker.unlink()
                raise ConflictError("В проекте есть активные передачи Blob Gateway")
            if row["state"] == "pending":
                candidates = set()
                for table in schema.tables.values():
                    predicate = project_predicate(table, project_id)
                    if predicate is None or table.name == self.table.name:
                        continue
                    for record in connection.execute(sa.select(table).where(predicate)).mappings():
                        candidates.update(_digests(dict(record)))
                plan = {**plan, "digests": sorted(candidates)}
            connection.execute(self.table.update().where(self.table.c.request_id == request_id).values(
                state="deleting", plan=plan, error="", updated_at=datetime.now(UTC),
            ))
            self.accounts._record_audit(connection, None, "project.deletion_started", None, {
                "request_id": request_id, "project_id": project_id, "reason": deletion_reason,
                "auto_confirmed": confirmed_automatically,
            })
            completed = list(row["completed"])
        try:
            for index, entry in enumerate(plan["paths"]):
                path = safe_path(Path(entry["path"]), Path(entry["root"]))
                step = str(path)
                if step not in completed:
                    root = Path(entry["root"])
                    trash = safe_path(root / ".kraken-trash" / request_id / str(index), root)
                    staged = f"staged:{step}"
                    if staged not in completed:
                        with engine.begin() as connection:
                            if not lock(connection, "kraken:storage", attempt=True):
                                raise ConflictError("Хранилище занято. Повторите перенос папок для удаления.")
                            if not trash.exists():
                                assert_unshared(self.services.workspace_files.registry, project_id, [path])
                                if path.exists():
                                    stage_stored_path(path, trash, authorization)
                            completed.append(staged)
                            connection.execute(self.table.update().where(self.table.c.request_id == request_id)
                                               .values(completed=completed, updated_at=datetime.now(UTC)))
                    remove_stored_path(trash, authorization)
                    completed.append(step)
                    self._progress(request_id, completed)
            with engine.begin() as connection:
                # Brief global barrier protects deduplication/reference registration.
                if not lock(connection, "kraken:storage", attempt=True):
                    raise ConflictError("Хранилище занято передачей данных. Повторите очистку blobs.")
                schema = self._schema(connection)
                shared = set()
                for table in schema.tables.values():
                    predicate = project_predicate(table, project_id, other=True)
                    if predicate is not None and table.name != self.table.name:
                        for record in connection.execute(sa.select(table).where(
                            predicate,
                        )).mappings():
                            shared.update(_digests(dict(record)))
                for digest in set(plan.get("digests", ())) - shared:
                    path = safe_path(self.services.uow_factory.blobs._path(digest), self.config.blob_root)
                    remove_stored_path(path, authorization)
                for table in reversed(schema.sorted_tables):
                    predicate = project_predicate(table, project_id)
                    if predicate is not None and table.name != self.table.name:
                        connection.execute(table.delete().where(predicate))
                connection.execute(self.table.update().where(self.table.c.request_id == request_id).values(
                    state="deleted", error="", completed=[*completed, "database"], updated_at=datetime.now(UTC),
                ))
                self.accounts._record_audit(connection, None, "project.deleted", None, {
                    "request_id": request_id, "project_id": project_id, "reason": deletion_reason,
                    "auto_confirmed": confirmed_automatically,
                })
        except Exception as exc:
            with engine.begin() as connection:
                connection.execute(self.table.update().where(self.table.c.request_id == request_id).values(
                    state="failed", error=str(exc), updated_at=datetime.now(UTC),
                ))
            raise

    def _progress(self, request_id: str, completed: list[str]) -> None:
        with self.services.engine.begin() as connection:
            connection.execute(self.table.update().where(self.table.c.request_id == request_id).values(
                completed=completed, updated_at=datetime.now(UTC),
            ))
