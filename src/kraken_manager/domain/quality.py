"""Quality findings and review decisions over immutable project artifacts.

Correction artifacts and existing CIF review batches remain owned by their
existing workflows. An issue records their references and acceptance decisions.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from datetime import datetime
from enum import StrEnum
from math import isfinite
import operator

from .common import DomainValidationError, InvalidStateTransition, as_utc, require_non_empty, validate_uuid
from .planning import WorkScope
from .artifacts import validate_sha256


@dataclass(frozen=True, slots=True)
class QualityMetricRule:
    metric_key: str
    value_field: str
    comparison: str
    threshold: float
    target_binding: str
    analysis_signature: str

    def __post_init__(self):
        for name in ("metric_key", "target_binding"):
            object.__setattr__(self, name, require_non_empty(getattr(self, name), field=name))
        if self.value_field not in {"raw_value", "goodness"} or self.comparison not in {"lt", "le", "gt", "ge"}:
            raise DomainValidationError("Quality metric rule requires an explicit value field and comparison")
        if isinstance(self.threshold, bool) or not isinstance(self.threshold, (int, float)) or not isfinite(self.threshold):
            raise DomainValidationError("Quality metric threshold must be an explicit finite number")
        object.__setattr__(self, "analysis_signature", validate_sha256(self.analysis_signature))

    def failed(self, value):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value):
            raise DomainValidationError("A quality decision requires a stored finite metric value")
        return getattr(operator, self.comparison)(value, self.threshold)


class QualityState(StrEnum):
    OPEN = "open"
    ASSIGNED = "assigned"
    CORRECTED = "corrected"
    RECHECKED = "rechecked"
    REJECTED = "rejected"
    ACCEPTED = "accepted"
    EXCEPTION = "exception"


@dataclass(frozen=True, slots=True)
class QualityReview:
    action: str
    actor_id: str
    at: datetime
    note: str
    evidence_version_ids: tuple[str, ...] = ()
    passed: bool | None = None
    analysis_run_id: str | None = None
    observations: tuple[tuple[str, float], ...] = ()

    def __post_init__(self):
        object.__setattr__(self, "actor_id", validate_uuid(self.actor_id))
        object.__setattr__(self, "at", as_utc(self.at, field="review.at"))
        object.__setattr__(self, "note", require_non_empty(self.note, field="review.note"))
        versions = tuple(validate_uuid(value) for value in self.evidence_version_ids)
        if len(versions) != len(set(versions)):
            raise DomainValidationError("Review evidence versions must be unique")
        object.__setattr__(self, "evidence_version_ids", versions)
        if self.passed is not None and not isinstance(self.passed, bool):
            raise DomainValidationError("Recheck outcome must be boolean")
        if self.analysis_run_id is not None:
            object.__setattr__(self, "analysis_run_id", require_non_empty(self.analysis_run_id, field="analysis_run_id"))
        observed = tuple((validate_uuid(frame), float(value)) for frame, value in self.observations)
        if any(not isfinite(value) for _, value in observed):
            raise DomainValidationError("Observed analysis metrics must be finite")
        object.__setattr__(self, "observations", observed)


@dataclass(frozen=True, slots=True)
class QualityIssue:
    id: str
    project_id: str
    title: str
    description: str
    severity: str
    scope: WorkScope
    source_version_ids: tuple[str, ...]
    created_by: str
    created_at: datetime
    updated_at: datetime
    revision: int = 0
    state: QualityState = QualityState.OPEN
    assignee_id: str | None = None
    correction_version_ids: tuple[str, ...] = ()
    work_order_id: str | None = None
    review_batch_id: str | None = None
    reviews: tuple[QualityReview, ...] = ()
    analysis_run_id: str | None = None
    analysis_rule: QualityMetricRule | None = None

    def __post_init__(self):
        for name in ("id", "project_id", "created_by"):
            object.__setattr__(self, name, validate_uuid(getattr(self, name)))
        for name in ("assignee_id", "work_order_id", "review_batch_id"):
            if getattr(self, name) is not None:
                object.__setattr__(self, name, validate_uuid(getattr(self, name)))
        for name in ("source_version_ids", "correction_version_ids"):
            values = tuple(validate_uuid(value) for value in getattr(self, name))
            if len(values) != len(set(values)):
                raise DomainValidationError(f"{name} must be unique")
            object.__setattr__(self, name, values)
        if not self.source_version_ids:
            raise DomainValidationError("An issue must reference its source artifact versions")
        for name in ("title", "description"):
            object.__setattr__(self, name, require_non_empty(getattr(self, name), field=name))
        if self.severity not in {"blocker", "critical", "major", "minor", "improvement"}:
            raise DomainValidationError("Unknown quality severity")
        if isinstance(self.revision, bool) or not isinstance(self.revision, int) or self.revision < 0:
            raise DomainValidationError("Quality revision must be a nonnegative integer")
        object.__setattr__(self, "state", QualityState(self.state))
        for name in ("created_at", "updated_at"):
            object.__setattr__(self, name, as_utc(getattr(self, name), field=name))
        if self.updated_at < self.created_at:
            raise DomainValidationError("Quality issue time cannot move backwards")
        object.__setattr__(self, "reviews", tuple(self.reviews))
        if self.analysis_run_id is not None:
            object.__setattr__(self, "analysis_run_id", require_non_empty(self.analysis_run_id, field="analysis_run_id"))
        if (self.analysis_rule is None) != (self.analysis_run_id is None):
            raise DomainValidationError("Analysis findings require both origin run and metric rule")

    def change(self, operation: str, arguments: dict, *, actor_id: str, at: datetime) -> QualityIssue:
        if self.state in {QualityState.ACCEPTED, QualityState.EXCEPTION}:
            raise InvalidStateTransition("Closed quality issues cannot change")
        at = as_utc(at, field="quality.transition_at")
        if at < self.updated_at:
            raise DomainValidationError("Quality transition time cannot move backwards")
        changes = {}
        if operation == "assign":
            if self.state not in {QualityState.OPEN, QualityState.ASSIGNED, QualityState.REJECTED}:
                raise InvalidStateTransition("Only open or rejected findings can be assigned")
            target = QualityState.ASSIGNED
            changes = {"assignee_id": arguments["assignee_id"], "work_order_id": arguments.get("work_order_id")}
        elif operation == "correct":
            if self.state is not QualityState.ASSIGNED:
                raise InvalidStateTransition("A correction requires an assigned issue")
            versions = tuple(arguments["evidence_version_ids"])
            if not versions or set(versions) & set(self.source_version_ids):
                raise DomainValidationError("Correction requires new immutable artifact versions")
            target = QualityState.CORRECTED
            changes = {"correction_version_ids": versions}
        elif operation == "recheck":
            if self.state is not QualityState.CORRECTED or not arguments.get("evidence_version_ids"):
                raise InvalidStateTransition("Recheck requires corrected work and evidence")
            if not isinstance(arguments.get("passed"), bool):
                raise DomainValidationError("Recheck requires an explicit boolean outcome")
            target = QualityState.RECHECKED
        elif operation in {"accept", "reject"}:
            if self.state is not QualityState.RECHECKED:
                raise InvalidStateTransition("Acceptance decisions require a recheck")
            if operation == "accept" and not self.reviews[-1].passed:
                raise InvalidStateTransition("A failed recheck cannot be accepted")
            target = QualityState.ACCEPTED if operation == "accept" else QualityState.REJECTED
        elif operation == "exception":
            target = QualityState.EXCEPTION
        else:
            raise DomainValidationError("Unknown quality command")
        review = QualityReview(
            operation, actor_id, at, arguments["note"],
            tuple(arguments.get("evidence_version_ids", ())), arguments.get("passed"),
            arguments.get("_analysis_run_id"), tuple(arguments.get("_observations", ())),
        )
        return replace(self, state=target, revision=self.revision + 1, updated_at=at,
                       reviews=(*self.reviews, review), **changes)

    def to_dict(self) -> dict:
        result = asdict(self)
        result["state"] = self.state.value
        result["created_at"] = self.created_at.isoformat()
        result["updated_at"] = self.updated_at.isoformat()
        for review in result["reviews"]:
            review["at"] = review["at"].isoformat()
        return result

    @classmethod
    def from_dict(cls, payload) -> QualityIssue:
        values = dict(payload)
        values["scope"] = WorkScope(**values["scope"])
        for name in ("created_at", "updated_at"):
            values[name] = datetime.fromisoformat(values[name])
        values["reviews"] = tuple(QualityReview(**{**dict(review), "at": datetime.fromisoformat(review["at"])})
                                   for review in values["reviews"])
        if values.get("analysis_rule") is not None:
            values["analysis_rule"] = QualityMetricRule(**values["analysis_rule"])
        return cls(**values)
