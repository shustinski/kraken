"""Versioned wire codec for public project commands; never imports commands by user input."""

from collections.abc import Mapping
from dataclasses import fields
from functools import lru_cache
import json

from kraken_manager.application.project_commands import (
    user_project_command_definition,
    user_project_command_definitions,
)
from kraken_manager.domain.common import validate_uuid
from kraken_manager.infrastructure.filesystem._codec import decode_model, encode_model


def encode_project_command(command):
    definition = user_project_command_definition(type(command))
    arguments = {
        field.name: encode_model(getattr(command, field.name)) for field in fields(command) if field.name != "context"
    }
    return {"schema_version": 1, "type": definition.command_type.__name__, "arguments": arguments}


@lru_cache(maxsize=128)
def _adapter(command_type):
    # Pydantic belongs to the HTTP/server installation, not the application or
    # the desktop client's import graph.
    from pydantic import TypeAdapter

    return TypeAdapter(command_type)


def decode_user_project_command(project_id, payload, context):
    validate_uuid(str(project_id), field="project_id")
    if not isinstance(payload, Mapping) or set(payload) != {"schema_version", "type", "arguments"}:
        raise ValueError("A versioned project command envelope is required")
    if type(payload["schema_version"]) is not int or payload["schema_version"] != 1:
        raise ValueError("Unsupported project command schema version")
    definitions = {item.command_type.__name__: item for item in user_project_command_definitions()}
    definition = definitions.get(payload["type"]) if isinstance(payload["type"], str) else None
    if definition is None:
        raise ValueError("Unknown or non-public project command")
    arguments = payload["arguments"]
    if not isinstance(arguments, Mapping):
        raise ValueError("Project command arguments must be an object")
    allowed = {field.name for field in fields(definition.command_type)} - {"context"}
    if set(arguments) - allowed:
        raise ValueError("Unknown or protected project command arguments")
    if str(arguments.get("project_id", project_id)) != str(project_id):
        raise ValueError("Command project does not match the request project")
    values = {**arguments, "project_id": str(project_id)}
    for key in (
        "layer_id",
        "representation_id",
        "series_id",
        "version_id",
        "principal_id",
        "batch_id",
        "note_id",
        "frame_id",
        "package_id",
        "parent_version_id",
    ):
        if values.get(key) is not None:
            validate_uuid(str(values[key]), field=key)
    for key in ("layer_ids", "candidate_version_ids", "input_version_ids"):
        if key in values:
            if not isinstance(values[key], (tuple, list)):
                raise ValueError(f"{key} must be an array")
            for identifier in values[key]:
                validate_uuid(str(identifier), field=key)
    if "expected_revisions" in values:
        for item in values["expected_revisions"]:
            if not isinstance(item, (tuple, list)) or len(item) != 2:
                raise ValueError("Expected layer revisions must be identifier/revision pairs")
            validate_uuid(str(item[0]), field="expected_revisions")
    # Identity, verification flags, attribution, and idempotency are supplied by
    # the authenticated adapter; client command contexts are never decoded.
    values["context"] = encode_model(context)
    return _adapter(definition.command_type).validate_json(json.dumps(values), strict=True)


def encode_project_command_result(command, result):
    return {"schema_version": 1, "type": type(command).__name__, "result": encode_model(result)}


def decode_project_command_result(command, payload):
    definition = user_project_command_definition(type(command))
    if (
        not isinstance(payload, Mapping)
        or type(payload.get("schema_version")) is not int
        or payload.get("schema_version") != 1
        or payload.get("type") != type(command).__name__
        or "result" not in payload
    ):
        raise ValueError("Invalid project command response")
    return decode_model(definition.result_type, payload["result"])
