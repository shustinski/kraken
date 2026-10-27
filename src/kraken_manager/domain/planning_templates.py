"""Versioned engineering-process templates that expand into existing WorkItems.

Templates define planned work only. They never execute plugins or introduce a
second operation graph or scheduler.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import StrEnum
import hashlib
import json
import re
from uuid import UUID, uuid5

from .common import DomainValidationError, require_non_empty, validate_uuid
from .planning import AcceptanceCriterion, WorkAssignment, WorkDependency, WorkEstimate, WorkItem, WorkScope

MAX_MATERIALIZED_ITEMS = 10_000
MAX_MATERIALIZED_DEPENDENCIES = 100_000
_KEY = re.compile(r"^[a-z][a-z0-9_-]{0,63}$")


def _key(value):
    if not isinstance(value, str) or not _KEY.fullmatch(value):
        raise DomainValidationError("template task keys must be lowercase identifiers of at most 64 characters")
    return value


class TemplateMultiplicity(StrEnum):
    PROJECT = "project"
    LAYER = "layer"
    FRAME = "frame"


@dataclass(frozen=True, slots=True)
class TemplateCriterion:
    description: str
    required: bool = True

    def __post_init__(self):
        object.__setattr__(
            self, "description", require_non_empty(self.description, field="criterion.description", maximum=4000)
        )
        if not isinstance(self.required, bool):
            raise DomainValidationError("criterion.required must be boolean")


@dataclass(frozen=True, slots=True)
class TemplateTask:
    key: str
    title: str
    multiplicity: TemplateMultiplicity = TemplateMultiplicity.LAYER
    parent_key: str | None = None
    effort_hours: float = 0
    priority: int = 0
    criteria: tuple[TemplateCriterion, ...] = ()

    def __post_init__(self):
        object.__setattr__(self, "key", _key(self.key))
        object.__setattr__(self, "title", require_non_empty(self.title, field="task.title"))
        object.__setattr__(self, "multiplicity", TemplateMultiplicity(self.multiplicity))
        if self.parent_key is not None:
            object.__setattr__(self, "parent_key", _key(self.parent_key))
        WorkEstimate(effort_hours=self.effort_hours)
        if isinstance(self.priority, bool) or not isinstance(self.priority, int) or self.priority < 0:
            raise DomainValidationError("template task priority must be a nonnegative integer")
        criteria = tuple(self.criteria)
        if any(not isinstance(c, TemplateCriterion) for c in criteria):
            raise DomainValidationError("invalid template acceptance criterion")
        object.__setattr__(self, "criteria", criteria)


@dataclass(frozen=True, slots=True)
class TemplateDependency:
    predecessor_key: str
    successor_key: str

    def __post_init__(self):
        object.__setattr__(self, "predecessor_key", _key(self.predecessor_key))
        object.__setattr__(self, "successor_key", _key(self.successor_key))
        if self.predecessor_key == self.successor_key:
            raise DomainValidationError("template task cannot depend on itself")


@dataclass(frozen=True, slots=True)
class TemplateMaterialization:
    application_id: str
    template_id: str
    template_version: int
    template_digest: str
    project_id: str
    items: tuple[WorkItem, ...]
    dependencies: tuple[WorkDependency, ...]

    def __post_init__(self):
        for name in ("application_id", "template_id", "project_id"):
            object.__setattr__(self, name, validate_uuid(getattr(self, name), field=name))
        if (
            isinstance(self.template_version, bool)
            or not isinstance(self.template_version, int)
            or self.template_version < 1
        ):
            raise DomainValidationError("template version must be positive")
        if not isinstance(self.template_digest, str) or not re.fullmatch("[a-f0-9]{64}", self.template_digest):
            raise DomainValidationError("template digest must be SHA256")
        object.__setattr__(self, "items", tuple(self.items))
        object.__setattr__(self, "dependencies", tuple(self.dependencies))
        if (
            not self.items
            or len(self.items) > MAX_MATERIALIZED_ITEMS
            or len(self.dependencies) > MAX_MATERIALIZED_DEPENDENCIES
        ):
            raise DomainValidationError("template materialization exceeds planning batch limits")


@dataclass(frozen=True, slots=True)
class WorkPlanTemplate:
    id: str
    project_id: str
    version: int
    title: str
    tasks: tuple[TemplateTask, ...]
    dependencies: tuple[TemplateDependency, ...] = ()
    description: str = ""

    def __post_init__(self):
        for name in ("id", "project_id"):
            object.__setattr__(self, name, validate_uuid(getattr(self, name), field=name))
        if isinstance(self.version, bool) or not isinstance(self.version, int) or self.version < 1:
            raise DomainValidationError("template version must be a positive integer")
        object.__setattr__(self, "title", require_non_empty(self.title, field="template.title"))
        if not isinstance(self.description, str) or len(self.description) > 10000:
            raise DomainValidationError("template description must be text of at most 10000 characters")
        object.__setattr__(self, "description", self.description.strip())
        tasks = tuple(self.tasks)
        dependencies = tuple(self.dependencies)
        object.__setattr__(self, "tasks", tasks)
        object.__setattr__(self, "dependencies", dependencies)
        if not tasks or len(tasks) > 1024 or any(not isinstance(t, TemplateTask) for t in tasks):
            raise DomainValidationError("template requires between 1 and 1024 typed tasks")
        by_key = {t.key: t for t in tasks}
        if len(by_key) != len(tasks):
            raise DomainValidationError("template task keys must be unique")
        if any(not isinstance(d, TemplateDependency) for d in dependencies) or len(set(dependencies)) != len(
            dependencies
        ):
            raise DomainValidationError("template dependencies must be typed and unique")
        edges = []
        breadth = {TemplateMultiplicity.PROJECT: 0, TemplateMultiplicity.LAYER: 1, TemplateMultiplicity.FRAME: 2}
        for task in tasks:
            if task.parent_key is not None:
                parent = by_key.get(task.parent_key)
                if parent is None or breadth[parent.multiplicity] > breadth[task.multiplicity]:
                    raise DomainValidationError("template parent must exist at the same or broader scope")
                # Child acceptance gates parent acceptance.
                edges.append((task.key, parent.key))
        for dependency in dependencies:
            if dependency.predecessor_key not in by_key or dependency.successor_key not in by_key:
                raise DomainValidationError("template dependency references an unknown task")
            edges.append((dependency.predecessor_key, dependency.successor_key))
        indegree = dict.fromkeys(by_key, 0)
        successors = {key: [] for key in by_key}
        for first, second in edges:
            successors[first].append(second)
            indegree[second] += 1
        ready = [key for key, degree in indegree.items() if degree == 0]
        visited = 0
        while ready:
            current = ready.pop()
            visited += 1
            for successor in successors[current]:
                indegree[successor] -= 1
                if indegree[successor] == 0:
                    ready.append(successor)
        if visited != len(tasks):
            raise DomainValidationError("template hierarchy/dependencies contain a cycle or acceptance deadlock")

    def to_dict(self):
        return {"schema_version": 1, "template": asdict(self)}

    @classmethod
    def from_dict(cls, payload):
        if payload.get("schema_version") != 1:
            raise DomainValidationError("unsupported work template schema version")
        data = dict(payload["template"])
        tasks = []
        for value in data["tasks"]:
            task = dict(value)
            task["criteria"] = tuple(TemplateCriterion(**c) for c in task.get("criteria", ()))
            tasks.append(TemplateTask(**task))
        data["tasks"] = tuple(tasks)
        data["dependencies"] = tuple(TemplateDependency(**d) for d in data.get("dependencies", ()))
        return cls(**data)

    @property
    def digest(self):
        return hashlib.sha256(
            json.dumps(self.to_dict(), sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        ).hexdigest()

    def materialize(self, scope: WorkScope, application_id: str, *, assignments=None):
        """Expand scope-aware fan-in/fan-out into one bounded, atomic batch."""
        application_id = validate_uuid(application_id)
        if not isinstance(scope, WorkScope):
            raise DomainValidationError("template scope must be a WorkScope")
        assignments = dict(assignments or {})
        if set(assignments) - {t.key for t in self.tasks} or any(
            not isinstance(a, WorkAssignment) for a in assignments.values()
        ):
            raise DomainValidationError("template assignments must reference known task keys and performers")
        if any(t.multiplicity is not TemplateMultiplicity.PROJECT for t in self.tasks) and not scope.layer_ids:
            raise DomainValidationError("template requires selected layers")
        if any(t.multiplicity is TemplateMultiplicity.FRAME for t in self.tasks) and not scope.frame_ids:
            raise DomainValidationError("frame template requires explicit frame selection")
        counts = {
            TemplateMultiplicity.PROJECT: 1,
            TemplateMultiplicity.LAYER: len(scope.layer_ids),
            TemplateMultiplicity.FRAME: len(scope.layer_ids) * len(scope.frame_ids),
        }
        if sum(counts[t.multiplicity] for t in self.tasks) > MAX_MATERIALIZED_ITEMS:
            raise DomainValidationError("template materialization exceeds planning batch limits")
        # Each record carries its spatial key for deterministic dependency joins.
        records = {}
        for task in self.tasks:
            if task.multiplicity is TemplateMultiplicity.PROJECT:
                targets = [(None, None, scope)]
            elif task.multiplicity is TemplateMultiplicity.LAYER:
                targets = [
                    (
                        layer,
                        None,
                        WorkScope(
                            layer_ids=(layer,),
                            frame_ids=scope.frame_ids,
                            region=scope.region,
                            coordinate_system=scope.coordinate_system,
                        ),
                    )
                    for layer in sorted(scope.layer_ids)
                ]
            else:
                targets = [
                    (
                        layer,
                        frame,
                        WorkScope(
                            layer_ids=(layer,),
                            frame_ids=(frame,),
                            region=scope.region,
                            coordinate_system=scope.coordinate_system,
                        ),
                    )
                    for layer in sorted(scope.layer_ids)
                    for frame in sorted(scope.frame_ids)
                ]
            records[task.key] = [
                (layer, frame, target, str(uuid5(UUID(application_id), f"{task.key}:{layer}:{frame}")))
                for layer, frame, target in targets
            ]

        by_scope = {key: {(r[0], r[1]): r for r in values} for key, values in records.items()}
        by_layer = {}
        task_modes = {task.key: task.multiplicity for task in self.tasks}
        for key, values in records.items():
            by_layer[key] = {}
            for value in values:
                by_layer[key].setdefault(value[0], []).append(value)

        def matching(first, target_key):
            mode = task_modes[target_key]
            if mode is TemplateMultiplicity.PROJECT or first[0] is None:
                return records[target_key]
            if mode is TemplateMultiplicity.LAYER:
                return (by_scope[target_key][(first[0], None)],)
            if first[1] is None:
                return by_layer[target_key][first[0]]
            return (by_scope[target_key][(first[0], first[1])],)

        items = []
        for task in self.tasks:
            for layer, frame, target, identifier in records[task.key]:
                parent_id = None
                if task.parent_key:
                    parents = matching((layer, frame), task.parent_key)
                    if len(parents) != 1:
                        raise DomainValidationError("template parent expansion is ambiguous")
                    parent_id = parents[0][3]
                criteria = tuple(
                    AcceptanceCriterion(str(uuid5(UUID(identifier), f"criterion:{index}")), c.description, c.required)
                    for index, c in enumerate(task.criteria)
                )
                items.append(
                    WorkItem(
                        identifier,
                        task.title,
                        scope=target,
                        parent_id=parent_id,
                        assignment=assignments.get(task.key),
                        priority=task.priority,
                        estimate=WorkEstimate(task.effort_hours),
                        criteria=criteria,
                    )
                )
        dependencies = []
        for edge in self.dependencies:
            for first in records[edge.predecessor_key]:
                for second in matching(first, edge.successor_key):
                    dependencies.append(WorkDependency(first[3], second[3]))
                    if len(dependencies) > MAX_MATERIALIZED_DEPENDENCIES:
                        raise DomainValidationError("template materialization exceeds dependency batch limit")
        return TemplateMaterialization(
            application_id, self.id, self.version, self.digest, self.project_id, tuple(items), tuple(dependencies)
        )
