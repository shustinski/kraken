from __future__ import annotations

from collections.abc import Iterable, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

from kraken_manager.domain.events import EventEnvelope

from kraken_manager.infrastructure.blob import FilesystemBlobStore

from .event_store import EventStreamConflict, FilesystemEventStore
from .projection_store import SQLiteProjectionStore


class _StagedEventStore:
    def __init__(self, backing: FilesystemEventStore) -> None:
        self.backing = backing
        self._pending: list[EventEnvelope] = []

    def current_revision(self, stream_id: str) -> int:
        return self.backing.current_revision(stream_id) + sum(
            event.stream_id == stream_id for event in self._pending
        )

    def append(
        self,
        stream_id: str,
        *,
        expected_revision: int,
        events: Sequence[EventEnvelope],
    ) -> int:
        actual = self.current_revision(stream_id)
        if actual != expected_revision:
            raise EventStreamConflict(stream_id, expected_revision, actual)
        for offset, event in enumerate(events, start=1):
            if event.stream_id != stream_id or str(event.project_id) != self.backing.project_id:
                raise ValueError("staged event stream/project does not match its local store")
            if event.revision != expected_revision + offset:
                raise ValueError("staged event revisions must be contiguous")
        self._pending.extend(events)
        return expected_revision + len(events)

    def load_stream(
        self,
        stream_id: str,
        *,
        after_revision: int = 0,
        as_of: datetime | None = None,
    ) -> tuple[EventEnvelope, ...]:
        persisted = self.backing.load_stream(stream_id, after_revision=after_revision, as_of=as_of)
        pending = tuple(
            event
            for event in self._pending
            if event.stream_id == stream_id
            and event.revision > after_revision
            and (as_of is None or event.recorded_at <= as_of)
        )
        return persisted + pending

    def find_by_idempotency_key(self, project_id: Any, idempotency_key: str) -> tuple[EventEnvelope, ...]:
        persisted = self.backing.find_by_idempotency_key(project_id, idempotency_key)
        if project_id is not None and str(project_id) != self.backing.project_id:
            return persisted
        return persisted + tuple(
            event for event in self._pending if event.idempotency_key == idempotency_key
        )

    def commit(self) -> None:
        if self._pending:
            self.backing.append_preserved(self._pending)
            self._pending.clear()

    def rollback(self) -> None:
        self._pending.clear()


class _StagedProjectionStore:
    _MODEL_KEYS = {
        "project": "project",
        "layer": "layer",
        "representation": "representation",
        "artifact_series": "artifact_series",
        "artifact_version": "artifact_version",
        "plugin_job": "plugin_job",
        "review_batch": "review_batch",
        "note": "note",
    }

    def __init__(self, backing: SQLiteProjectionStore, events: _StagedEventStore) -> None:
        self.backing = backing
        self.events = events
        self._current: dict[tuple[str, str], tuple[Any, dict[str, Any]]] = {}
        self._operations: list[tuple[str, Any, dict[str, Any]]] = []

    def _save(self, kind: str, model: Any, **options: Any) -> None:
        if self.events._pending:
            options["recorded_at"] = self.events._pending[-1].recorded_at
        identifier = str(getattr(model, "id", getattr(model, "note_id", "")))
        if not identifier:
            raise ValueError(f"{kind} projection has no identity")
        self._current[(kind, identifier)] = (model, options)
        self._operations.append((kind, model, options))

    def _get(self, kind: str, entity_id: Any, *, as_of: datetime | None = None) -> Any | None:
        if as_of is None:
            staged = self._current.get((kind, str(entity_id)))
            if staged is not None:
                return staged[0]
        method = getattr(self.backing, f"get_{self._MODEL_KEYS[kind]}")
        return method(entity_id, as_of=as_of)

    def _merge_list(
        self,
        kind: str,
        persisted: Iterable[Any],
        predicate: Any,
        *,
        include_archived: bool,
        as_of: datetime | None = None,
    ) -> tuple[Any, ...]:
        if as_of is not None:
            return tuple(
                model
                for model in persisted
                if include_archived or not self.backing._is_archived(model)
            )
        values = {str(model.id): model for model in persisted}
        for (saved_kind, entity_id), (model, _) in self._current.items():
            if saved_kind == kind and predicate(model):
                values[entity_id] = model
        if not include_archived:
            values = {
                key: model
                for key, model in values.items()
                if not self.backing._is_archived(model)
            }
        return tuple(sorted(values.values(), key=lambda model: (getattr(model, "order", 0), str(model.id))))

    def get_project(self, project_id: Any, *, as_of: datetime | None = None) -> Any | None:
        return self._get("project", project_id, as_of=as_of)

    def save_project(self, project: Any) -> None:
        self._save("project", project)

    def get_layer(self, layer_id: Any, *, as_of: datetime | None = None) -> Any | None:
        return self._get("layer", layer_id, as_of=as_of)

    def list_layers(
        self, project_id: Any, *, include_archived: bool = False, as_of: datetime | None = None
    ) -> tuple[Any, ...]:
        return self._merge_list(
            "layer",
            self.backing.list_layers(project_id, include_archived=True, as_of=as_of),
            lambda model: str(model.project_id) == str(project_id),
            include_archived=include_archived,
            as_of=as_of,
        )

    def save_layer(self, layer: Any) -> None:
        self._save("layer", layer)

    def get_representation(self, representation_id: Any, *, as_of: datetime | None = None) -> Any | None:
        return self._get("representation", representation_id, as_of=as_of)

    def list_representations(
        self, layer_id: Any, *, include_archived: bool = False, as_of: datetime | None = None
    ) -> tuple[Any, ...]:
        return self._merge_list(
            "representation",
            self.backing.list_representations(layer_id, include_archived=True, as_of=as_of),
            lambda model: str(model.layer_id) == str(layer_id),
            include_archived=include_archived,
            as_of=as_of,
        )

    def save_representation(self, representation: Any) -> None:
        self._save("representation", representation)

    def get_artifact_series(self, series_id: Any, *, as_of: datetime | None = None) -> Any | None:
        return self._get("artifact_series", series_id, as_of=as_of)

    def save_artifact_series(self, series: Any) -> None:
        self._save("artifact_series", series)

    def get_artifact_version(self, version_id: Any, *, as_of: datetime | None = None) -> Any | None:
        return self._get("artifact_version", version_id, as_of=as_of)

    def get_active_artifact_version(self, series_id: Any, *, as_of: datetime | None = None) -> Any | None:
        if as_of is None:
            for kind, model, options in reversed(self._operations):
                if kind == "artifact_version" and str(model.series_id) == str(series_id) and options.get("activate"):
                    return model
        return self.backing.get_active_artifact_version(series_id, as_of=as_of)

    def list_artifact_versions(
        self, series_id: Any, *, as_of: datetime | None = None
    ) -> tuple[Any, ...]:
        return self._merge_list(
            "artifact_version",
            self.backing.list_artifact_versions(series_id, as_of=as_of),
            lambda model: str(model.series_id) == str(series_id),
            include_archived=True,
            as_of=as_of,
        )

    def save_artifact_version(self, version: Any, *, activate: bool) -> None:
        self._save("artifact_version", version, activate=activate)

    def save_plugin_job(self, job: Any) -> None:
        self._save("plugin_job", job)

    def get_plugin_job(self, job_id: Any, *, as_of: datetime | None = None) -> Any | None:
        return self._get("plugin_job", job_id, as_of=as_of)

    def list_plugin_jobs(
        self, project_id: Any, *, as_of: datetime | None = None
    ) -> tuple[Any, ...]:
        return self._merge_list(
            "plugin_job",
            self.backing.list_plugin_jobs(project_id, as_of=as_of),
            lambda model: str(model.project_id) == str(project_id),
            include_archived=True,
            as_of=as_of,
        )

    def get_review_batch(self, batch_id: Any, *, as_of: datetime | None = None) -> Any | None:
        return self._get("review_batch", batch_id, as_of=as_of)

    def save_review_batch(self, batch: Any) -> None:
        self._save("review_batch", batch)

    def list_active_review_batches(
        self, project_id: Any, layer_id: Any, *, as_of: datetime | None = None
    ) -> tuple[Any, ...]:
        values = self._merge_list(
            "review_batch",
            self.backing.list_active_review_batches(project_id, layer_id, as_of=as_of),
            lambda model: str(model.project_id) == str(project_id) and str(model.layer_id) == str(layer_id),
            include_archived=True,
            as_of=as_of,
        )
        return tuple(
            model
            for model in values
            if getattr(model.state, "value", model.state) not in {"completed", "cancelled"}
        )

    def list_review_batches(
        self,
        project_id: Any,
        *,
        layer_id: Any | None = None,
        as_of: datetime | None = None,
    ) -> tuple[Any, ...]:
        return self._merge_list(
            "review_batch",
            self.backing.list_review_batches(
                project_id,
                layer_id=layer_id,
                as_of=as_of,
            ),
            lambda model: (
                str(model.project_id) == str(project_id)
                and (layer_id is None or str(model.layer_id) == str(layer_id))
            ),
            include_archived=True,
            as_of=as_of,
        )

    def get_note(self, note_id: str, *, as_of: datetime | None = None) -> Any | None:
        return self._get("note", note_id, as_of=as_of)

    def list_notes(
        self,
        project_id: Any,
        *,
        layer_id: Any | None = None,
        frame_id: str | None = None,
        as_of: datetime | None = None,
    ) -> tuple[Any, ...]:
        return self._merge_list(
            "note",
            self.backing.list_notes(
                project_id,
                layer_id=layer_id,
                frame_id=frame_id,
                as_of=as_of,
            ),
            lambda model: (
                str(model.project_id) == str(project_id)
                and (layer_id is None or str(model.layer_id) == str(layer_id))
                and (frame_id is None or str(model.frame_id) == str(frame_id))
            ),
            include_archived=True,
            as_of=as_of,
        )

    def save_note(self, note: Any) -> None:
        self._save("note", note)

    def commit(self) -> None:
        for kind, model, options in self._operations:
            if "activate" in options:
                options = {**options, "active": options["activate"]}
                del options["activate"]
            self.backing._save_typed(kind, model, **options)
        self.rollback()

    def rollback(self) -> None:
        self._current.clear()
        self._operations.clear()


class _StagedAclStore:
    def __init__(self, backing: Any) -> None:
        self.backing = backing
        self._operations: list[tuple[str, tuple[Any, ...]]] = []

    def roles_for(self, project_id: Any, principal_id: Any) -> frozenset[Any]:
        roles = set(self.backing.roles_for(project_id, principal_id))
        for operation, arguments in self._operations:
            if operation == "assign":
                assignment = arguments[0]
                if assignment.project_id == project_id and assignment.principal_id == principal_id:
                    roles.add(assignment.role)
            elif arguments[:2] == (project_id, principal_id):
                roles.discard(arguments[2])
        return frozenset(roles)

    def assign(self, assignment: Any) -> None:
        self._operations.append(("assign", (assignment,)))

    def revoke(self, project_id: Any, principal_id: Any, role: Any) -> None:
        self._operations.append(("revoke", (project_id, principal_id, role)))

    def commit(self) -> None:
        for operation, arguments in self._operations:
            getattr(self.backing, operation)(*arguments)
        self._operations.clear()

    def rollback(self) -> None:
        self._operations.clear()


class LocalProjectUnitOfWork:
    """Project-scoped local UoW with staged authoritative events and cache writes."""

    def __init__(
        self,
        event_store: FilesystemEventStore,
        projections: SQLiteProjectionStore,
        blobs: FilesystemBlobStore,
        identities: Any,
        acl: Any,
    ) -> None:
        self._backing_event_store = event_store
        self._backing_projections = projections
        self.event_store = _StagedEventStore(event_store)
        self.projections = _StagedProjectionStore(projections, self.event_store)
        self.blobs = blobs
        self.identities = identities
        self.acl = _StagedAclStore(acl)
        self._entered = False
        self._committed = False

    def __enter__(self) -> LocalProjectUnitOfWork:
        if self._entered:
            raise RuntimeError("unit of work cannot be entered twice")
        self._backing_event_store.lock.acquire(self._backing_event_store.lock_timeout)
        self._entered = True
        return self

    def commit(self) -> None:
        if not self._entered:
            raise RuntimeError("unit of work is not active")
        previous_position = self._backing_event_store.last_global_position()
        self.event_store.commit()
        # ACL is authoritative state and must complete before advancing the
        # rebuildable projection checkpoint. A crash/failure here is detected
        # on the next open and recovered from the committed event.
        self.acl.commit()
        self.projections.commit()
        # Advance the durable index checkpoint only after every typed write.
        # Any earlier failure leaves a detectable lag and triggers full replay.
        for stored in self._backing_event_store.iter_project(after_global_position=previous_position):
            self._backing_projections.apply(stored)
        project = self._backing_projections.get_project(self._backing_event_store.project_id)
        if project is not None and not self._backing_event_store.layout.descriptor_path.exists():
            self._backing_event_store.layout.initialize(
                {
                    "schema_version": 1,
                    "project_id": str(project.id),
                    "name": project.name,
                    "width": project.width,
                    "height": project.height,
                    "orientation": project.orientation.value,
                    "storage_profile": project.storage_profile,
                }
            )
        self._committed = True

    def rollback(self) -> None:
        self.event_store.rollback()
        self.projections.rollback()
        self.acl.rollback()

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> bool:
        try:
            if exc_type is not None or not self._committed:
                self.rollback()
        finally:
            self._backing_event_store.lock.release()
            self._entered = False
        return False


class LocalProjectUnitOfWorkFactory:
    def __init__(
        self,
        catalog_root: str | Path,
        project_id: str,
        *,
        identities: Any,
        acl: Any,
    ) -> None:
        self.catalog_root = Path(catalog_root)
        self.project_id = project_id
        self.identities = identities
        self.acl = acl

    def __call__(self) -> LocalProjectUnitOfWork:
        return LocalProjectUnitOfWork(
            event_store=FilesystemEventStore(self.catalog_root, self.project_id),
            projections=SQLiteProjectionStore(self.catalog_root, self.project_id),
            blobs=FilesystemBlobStore.for_project(self.catalog_root, self.project_id),
            identities=self.identities,
            acl=self.acl,
        )


__all__ = ["LocalProjectUnitOfWork", "LocalProjectUnitOfWorkFactory"]
