"""Reusable semantic port contracts for current and third-party backends."""

from __future__ import annotations

import hashlib
import unittest
from dataclasses import replace
from datetime import UTC, datetime, timedelta

from kraken_manager.domain.events import ActorSnapshot, EventEnvelope
from kraken_manager.application.errors import ConflictError, NotFoundError
from kraken_manager.domain.common import PerformerId, PrincipalId
from kraken_manager.domain.identity import Performer, Principal


PROJECT_ID = "10000000-0000-0000-0000-000000000001"
PRINCIPAL_ID = "20000000-0000-0000-0000-000000000001"
PERFORMER_ID = "30000000-0000-0000-0000-000000000001"
LINKED_PERFORMER_ID = "30000000-0000-0000-0000-000000000002"


class EventStoreContract:
    event_store = None

    def make_event(self, revision: int, *, key: str, at: datetime | None = None) -> EventEnvelope:
        principal = Principal.local(subject="contract", display_name="Contract", principal_id=PRINCIPAL_ID)
        return EventEnvelope.create(
            stream_id=f"project:{PROJECT_ID}",
            project_id=PROJECT_ID,
            revision=revision,
            event_type="ContractEvent",
            payload={"revision": revision},
            actor=ActorSnapshot.from_principal(principal),
            idempotency_key=key,
            recorded_at=at or datetime.now(UTC),
        )

    def assert_event_store_contract(self) -> None:
        store = self.event_store
        assert store is not None
        first_at = datetime.now(UTC)
        first = self.make_event(1, key="first", at=first_at)
        self.assertEqual(1, store.append(first.stream_id, expected_revision=0, events=(first,)))
        self.assertEqual((first,), store.load_stream(first.stream_id))
        self.assertEqual(1, store.current_revision(first.stream_id))
        self.assertEqual((first,), store.find_by_idempotency_key(first.project_id, "first"))
        self.assertEqual((), store.load_stream(first.stream_id, as_of=first_at - timedelta(microseconds=1)))
        conflicting = self.make_event(1, key="conflict")
        with self.assertRaises(Exception):
            store.append(conflicting.stream_id, expected_revision=0, events=(conflicting,))


class BlobStoreContract:
    blob_store = None

    def assert_blob_store_contract(self) -> None:
        store = self.blob_store
        assert store is not None
        payload = b"immutable-contract-payload"
        digest = hashlib.sha256(payload).hexdigest()
        first = store.put(iter((payload[:5], payload[5:])), expected_sha256=digest)
        second = store.put((payload,), expected_sha256=digest)
        self.assertEqual(digest, first.blob.sha256)
        self.assertFalse(first.already_existed)
        self.assertTrue(second.already_existed)
        self.assertEqual(payload, b"".join(store.iter_bytes(first.blob, chunk_size=3)))
        self.assertTrue(store.exists(first.blob))


class PerformerStoreContract:
    performer_store = None

    def assert_performer_store_contract(self) -> None:
        store = self.performer_store
        assert store is not None
        manual = Performer.create(
            performer_id=PERFORMER_ID,
            name="Manual Worker",
            color="#123ABC",
        )
        linked = Performer.create(
            performer_id=LINKED_PERFORMER_ID,
            name="GitLab Worker",
            color="#C65D21",
            principal_id=PrincipalId(PRINCIPAL_ID),
        )

        self.assertEqual(manual, store.create(manual))
        self.assertEqual(linked, store.create(linked))
        self.assertEqual(manual, store.get(PerformerId(PERFORMER_ID)))
        self.assertEqual(linked, store.get_by_principal(PrincipalId(PRINCIPAL_ID)))
        self.assertEqual((linked, manual), store.list())

        with self.assertRaises(ConflictError):
            store.create(manual)
        with self.assertRaises(ConflictError):
            store.create(
                Performer.create(
                    name="Duplicate link",
                    color="#000000",
                    principal_id=PrincipalId(PRINCIPAL_ID),
                )
            )

        updated = replace(linked, name="Renamed Worker", color="#ABCDEF")
        self.assertEqual(updated, store.update(updated))
        self.assertEqual(updated, store.get(PerformerId(LINKED_PERFORMER_ID)))
        with self.assertRaises(NotFoundError):
            store.update(
                Performer.create(
                    performer_id="30000000-0000-0000-0000-000000000099",
                    name="Missing",
                    color="#111111",
                )
            )

        archived = store.archive(PerformerId(PERFORMER_ID))
        self.assertFalse(archived.active)
        self.assertEqual((updated,), store.list())
        self.assertEqual((manual.archive(), updated), store.list(include_archived=True))
        self.assertEqual(archived, store.archive(PerformerId(PERFORMER_ID)))
        with self.assertRaises(NotFoundError):
            store.archive(PerformerId("30000000-0000-0000-0000-000000000098"))


__all__ = [
    "BlobStoreContract",
    "EventStoreContract",
    "LINKED_PERFORMER_ID",
    "PERFORMER_ID",
    "PerformerStoreContract",
    "PRINCIPAL_ID",
    "PROJECT_ID",
]
