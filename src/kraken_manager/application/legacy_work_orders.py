"""Capture existing projection-only work as an explicitly reviewed baseline.

The import preserves the observed snapshot, never invents earlier events or
executes a package's recorded destination. Codec and commit strategy are ports.
"""
import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import replace

from kraken_manager.application.authorization import AuthorizationPolicy
from kraken_manager.application.errors import AuthorizationError, ConcurrencyError, NotFoundError
from kraken_manager.domain.common import require_non_empty, utc_now
from kraken_manager.domain.events import ActorSnapshot, EventEnvelope, ProgramSnapshot
from kraken_manager.domain.identity import Permission, parse_project_role

IMPORTED = "WorkOrderImportedFromProjectionV1"
RECEIPT = "LegacyWorkOrdersMigrationRecordedV1"
MAX_ORDERS = 10000


def snapshot_digest(value, encode) -> str:
    data = json.dumps(encode(value), sort_keys=True, ensure_ascii=True,
                      separators=(",", ":"), allow_nan=False).encode("ascii")
    return hashlib.sha256(data).hexdigest()


def validate_legacy_observations(observations, project_id, order_id):
    if not isinstance(observations, (tuple, list)) or len(observations) > 10000:
        raise ValueError("Legacy cached observations must be a bounded list")
    for observation in observations:
        if not isinstance(observation, Mapping) or set(observation) != {"recorded_at", "valid_to", "snapshot"}:
            raise ValueError("Invalid legacy cached observation")
        snapshot = observation["snapshot"]
        if (not isinstance(snapshot, Mapping) or str(snapshot.get("id")) != str(order_id)
                or str(snapshot.get("project_id")) != str(project_id)):
            raise ValueError("Legacy cached observation belongs to another work order/project")
        if not isinstance(observation["recorded_at"], str) or not isinstance(observation["valid_to"], (str, type(None))):
            raise ValueError("Invalid legacy cached observation timestamp")


def legacy_work_orders(projections, events, project_id):
    orders = tuple(projections.list_work_orders(project_id))
    legacy = []
    for order in sorted(orders, key=lambda item: str(item.id)):
        if str(order.project_id) != str(project_id):
            raise ValueError("Work order projection belongs to another project")
        history = events.load_stream(f"work-order:{order.id}")
        if any(getattr(event, "schema_version", None) != 1 for event in history):
            raise ConcurrencyError("Unsupported work order event schema; upgrade before migration")
        if not history:
            legacy.append(order)
            if len(legacy) > MAX_ORDERS:
                raise ValueError(f"Legacy migration supports at most {MAX_ORDERS} unrecorded work orders per project")
        elif not any(e.event_type in {"WorkOrderRecordedV1", IMPORTED} for e in history):
            raise ConcurrencyError(f"Unknown canonical work order stream; preserve the cache: {order.id}")
    return tuple(legacy), len(orders)-len(legacy)


def _authorize(unit, profiles, principal, project_id, *, gitlab_identity_verified):
    current = unit.identities.get(principal.id)
    if current is None or not current.active:
        raise AuthorizationError("The account is inactive or unavailable")
    # Refresh revoked/inactive identities without reintroducing console roles
    # which the network composition deliberately removed from its caller.
    principal = replace(current, system_roles=current.system_roles & principal.system_roles)
    project = unit.projections.get_project(project_id)
    if project is None:
        raise NotFoundError("Project was not found")
    roles = set(unit.acl.roles_for(project.id, principal.id))
    # Migration can open a lagging cache which ordinary reads refuse to rebuild.
    # A durable revocation must still override its stale ACL materialization.
    for event in unit.event_store.load_stream(f"acl:{project.id}:{principal.id}"):
        if event.schema_version != 1 or event.event_type not in {"ProjectRoleAssigned", "ProjectRoleRevoked"}:
            raise ConcurrencyError("Unsupported ACL event schema/type; upgrade before migration")
        if str(event.project_id) != str(project.id) or str(event.payload.get("principal_id")) != str(principal.id):
            raise ConcurrencyError("ACL event identity differs from its stream")
        if event.event_type == "ProjectRoleAssigned":
            roles.add(parse_project_role(event.payload["role"]))
        elif event.event_type == "ProjectRoleRevoked":
            roles.discard(parse_project_role(event.payload["role"]))
    storage_profile = project.storage_profile
    for event in unit.event_store.load_stream(f"project:{project.id}"):
        if event.schema_version != 1:
            raise ConcurrencyError("Unsupported project event schema; upgrade before migration")
        if event.event_type == "ProjectCreated":
            storage_profile = event.payload["storage_profile"]
        elif event.event_type in {"ProjectRenamed", "ProjectArchived", "ProjectRestored"}:
            storage_profile = event.payload["project"]["storage_profile"]
    AuthorizationPolicy().require(principal=principal, storage=profiles.get(storage_profile),
        permission=Permission.MIGRATE_PROJECT, roles=roles,
        gitlab_identity_verified=gitlab_identity_verified)
    return project, principal


def _plan(unit, project_id, encode):
    legacy, recorded = legacy_work_orders(unit.projections, unit.event_store, project_id)
    rows = []
    captured_bytes = 0
    for order in legacy:
        observations = unit.projections.list_work_order_observations(project_id, order.id)
        validate_legacy_observations(observations, project_id, order.id)
        captured_bytes += len(json.dumps(encode((order, observations)), ensure_ascii=True))
        if captured_bytes > 64 * 1024 * 1024:
            raise ValueError("Legacy work order observation batch exceeds 64 MiB; old cache preserved")
        rows.append(dict(id=str(order.id), revision=order.revision, state=order.state.value,
                 kind=order.kind.value, destination=order.destination,
                 snapshot_sha256=snapshot_digest(order, encode), snapshot=encode(order),
                 observations=encode(observations), observations_sha256=snapshot_digest(observations, encode)))
    fingerprint = dict(project_id=str(project_id), legacy=[(row["id"], row["snapshot_sha256"], row["observations_sha256"]) for row in rows])
    return dict(project_id=str(project_id), legacy=rows, already_recorded=recorded,
                prior_history="unknown", digest=snapshot_digest(fingerprint, encode)), legacy


def plan_legacy_work_order_migration(*, uow_factory, profiles, principal, project_id, encode,
                                     gitlab_identity_verified=False):
    with uow_factory() as unit:
        project, _ = _authorize(unit, profiles, principal, project_id, gitlab_identity_verified=gitlab_identity_verified)
        plan, _ = _plan(unit, project.id, encode)
        return plan


def migrate_legacy_work_orders(*, uow_factory, profiles, principal, project_id, encode,
                               expected_plan_digest, idempotency_key, commit=None,
                               gitlab_identity_verified=False):
    if not isinstance(expected_plan_digest, str) or not re.fullmatch(r"[0-9a-f]{64}", expected_plan_digest):
        raise ValueError("Expected a reviewed legacy migration plan digest")
    key = require_non_empty(idempotency_key, field="idempotency_key", maximum=255)
    command = {"expected_plan_digest": expected_plan_digest}
    with uow_factory() as unit:
        project, principal = _authorize(unit, profiles, principal, project_id, gitlab_identity_verified=gitlab_identity_verified)
        prior = unit.event_store.find_by_idempotency_key(project.id, key)
        if prior:
            event = prior[-1]
            if (event.event_type != RECEIPT or event.actor.principal_id != principal.id
                    or encode(event.payload.get("command")) != command):
                raise ConcurrencyError("Migration idempotency key belongs to another command or actor")
            result = encode(event.payload["result"])
        else:
            plan, orders = _plan(unit, project.id, encode)
            if plan["digest"] != expected_plan_digest:
                raise ConcurrencyError("Legacy work orders changed; review a fresh migration plan")
            now = utc_now()
            actor = ActorSnapshot.from_principal(principal)
            for order, row in zip(orders, plan["legacy"], strict=True):
                stream = f"work-order:{order.id}"
                event = EventEnvelope.create(stream_id=stream, project_id=project.id, revision=1,
                    event_type=IMPORTED, recorded_at=now, effective_at=order.updated_at,
                    actor=actor, performer_id=order.assignee_id,
                    program=ProgramSnapshot("Kraken legacy work migration", "1"),
                    payload=dict(work_order=encode(order), snapshot_sha256=snapshot_digest(order, encode),
                                 origin="legacy_projection", prior_history="unknown",
                                 legacy_observations=row["observations"],
                                 observations_sha256=row["observations_sha256"]))
                unit.event_store.append(stream, expected_revision=0, events=(event,))
            result = dict(project_id=str(project.id), plan_digest=expected_plan_digest,
                          migrated=len(orders), order_ids=[str(order.id) for order in orders], prior_history="unknown")
            stream = f"work-order-migrations:{project.id}"
            event = EventEnvelope.create(stream_id=stream, project_id=project.id,
                revision=unit.event_store.current_revision(stream)+1, event_type=RECEIPT,
                recorded_at=now, actor=actor, program=ProgramSnapshot("Kraken legacy work migration", "1"),
                idempotency_key=key, payload=dict(command=command, result=result))
            unit.event_store.append(stream, expected_revision=event.revision-1, events=(event,))
        if commit is None:
            unit.commit()
        else:
            commit(unit)
        return result
