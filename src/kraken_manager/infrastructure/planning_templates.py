"""Transport definition codec for safe engineering template authoring."""

from __future__ import annotations

from kraken_manager.domain.planning_templates import (
    TemplateCriterion,
    TemplateDependency,
    TemplateTask,
    WorkPlanTemplate,
)
from kraken_manager.infrastructure.planning_commands import _mapping


def decode_template_definition(definition, *, template_id, project_id, version):
    data = _mapping(definition, {"title", "description", "tasks", "dependencies"}, "template definition")
    tasks = []
    for value in data.get("tasks", ()):
        task = _mapping(
            value,
            {"key", "title", "multiplicity", "parent_key", "effort_hours", "priority", "criteria"},
            "template task",
        )
        task["criteria"] = tuple(
            TemplateCriterion(**_mapping(c, {"description", "required"}, "template criterion"))
            for c in task.get("criteria", ())
        )
        tasks.append(TemplateTask(**task))
    data["tasks"] = tuple(tasks)
    data["dependencies"] = tuple(
        TemplateDependency(**_mapping(d, {"predecessor_key", "successor_key"}, "template dependency"))
        for d in data.get("dependencies", ())
    )
    return WorkPlanTemplate(id=template_id, project_id=str(project_id), version=version, **data)


def list_templates(events, project_id):
    from kraken_manager.domain.planning_templates import WorkPlanTemplate

    latest = {}
    if hasattr(events, "list_project_events"):
        position = 0
        while True:
            batch = events.list_project_events(project_id, after_position=position, limit=1000)
            if not batch:
                break
            for position, event in batch:
                if event.stream_id.startswith("work-template:"):
                    latest[event.stream_id] = event.payload
    else:
        for event in events.iter_project():
            if event.stream_id.startswith("work-template:"):
                latest[event.stream_id] = event.data["payload"]
    result = (WorkPlanTemplate.from_dict(payload) for payload in latest.values())
    return tuple(sorted((t for t in result if t.project_id == str(project_id)), key=lambda t: (t.title, t.id)))
