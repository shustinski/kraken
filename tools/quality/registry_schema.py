"""Schema validation for the quality registries.

Hand-rolled on purpose: no new dependencies, so the quality gates stay
runnable in the bare dev environment. ``validate_registry`` returns a list of
human-readable problems; an empty list means the registry file is valid.

Both registries share the shape ``{"<kind>s": [ ... ]}`` where ``<kind>`` is
``defect`` (``docs/quality/registry/defects.json``) or ``action``
(``docs/quality/registry/actions.json``). The kind is detected from the top
level key, so a single validator covers both files.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

DEFECT_STATUSES = ("OPEN", "RESOLVED", "EXCEPTION")
DEFECT_SEVERITIES = ("critical", "major", "minor")
ACTION_SURFACES = ("HUB", "MGR", "SRV", "AGT", "PLG", "GW")
ACTION_TIERS = ("smoke", "unit", "integration", "gui", "stateful", "e2e", "extended", "regression", "live")
ACTION_STATUSES = ("PASS", "FAIL", "PARTIAL", "BLOCKED", "NOT_TESTED", "NOT_APPLICABLE")


def _require_str(record: dict[str, Any], field: str, problems: list[str], prefix: str) -> None:
    value = record.get(field)
    if not isinstance(value, str) or not value:
        problems.append(f"{prefix}: '{field}' must be a non-empty string")


def _require_str_list(record: dict[str, Any], field: str, problems: list[str], prefix: str) -> None:
    value = record.get(field)
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        problems.append(f"{prefix}: '{field}' must be a list of strings")


def _require_enum(record: dict[str, Any], field: str, allowed: tuple[str, ...], problems: list[str], prefix: str) -> None:
    value = record.get(field)
    if value not in allowed:
        problems.append(f"{prefix}: '{field}' must be one of {list(allowed)} (got {value!r})")


def _validate_defect(record: dict[str, Any], problems: list[str], prefix: str) -> None:
    _require_str(record, "component", problems, prefix)
    _require_str(record, "title", problems, prefix)
    _require_enum(record, "status", DEFECT_STATUSES, problems, prefix)
    _require_enum(record, "severity", DEFECT_SEVERITIES, problems, prefix)
    _require_str_list(record, "regression_tests", problems, prefix)
    _require_str_list(record, "evidence", problems, prefix)
    if not isinstance(record.get("fixed_in"), str):
        problems.append(f"{prefix}: 'fixed_in' must be a string (empty when no commit is stated)")
    _require_str(record, "repro", problems, prefix)
    if record.get("status") == "RESOLVED":
        if "exception_reason" in record:
            problems.append(f"{prefix}: RESOLVED defect must not carry exception_reason")
    else:
        _require_str(record, "exception_reason", problems, prefix)


def _validate_action(record: dict[str, Any], problems: list[str], prefix: str) -> None:
    _require_enum(record, "surface", ACTION_SURFACES, problems, prefix)
    _require_str(record, "entry_point", problems, prefix)
    _require_str_list(record, "preconditions", problems, prefix)
    _require_str(record, "expected_behavior", problems, prefix)
    _require_str_list(record, "side_effects", problems, prefix)
    _require_str_list(record, "error_conditions", problems, prefix)
    _require_enum(record, "tier", ACTION_TIERS, problems, prefix)
    _require_enum(record, "status", ACTION_STATUSES, problems, prefix)
    _require_str_list(record, "tests", problems, prefix)
    _require_str_list(record, "evidence", problems, prefix)
    if not isinstance(record.get("mandatory"), bool):
        problems.append(f"{prefix}: 'mandatory' must be a boolean")
    if not isinstance(record.get("beta"), bool):
        problems.append(f"{prefix}: 'beta' must be a boolean")
    if record.get("beta") is False:
        _require_str(record, "beta_out_reason", problems, prefix)
    elif record.get("beta") is True and "beta_out_reason" in record:
        problems.append(f"{prefix}: beta action must not carry beta_out_reason")
    if record.get("status") == "BLOCKED" and not record.get("exception_reason"):
        problems.append(f"{prefix}: BLOCKED action requires exception_reason")
    if record.get("status") == "NOT_TESTED" and record.get("beta") is True and not record.get("exception_reason"):
        problems.append(f"{prefix}: beta NOT_TESTED action requires exception_reason")


def validate_registry(path: Path) -> list[str]:
    """Validate one registry file. Returns a list of problems (empty = valid)."""
    path = Path(path)
    if not path.is_file():
        return [f"{path}: file not found"]
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return [f"{path}: invalid JSON: {exc}"]
    if not isinstance(data, dict):
        return [f"{path}: top level must be an object"]
    if "defects" in data:
        kind, check = "defect", _validate_defect
        records = data["defects"]
    elif "actions" in data:
        kind, check = "action", _validate_action
        records = data["actions"]
    else:
        return [f"{path}: top level must contain a 'defects' or 'actions' list"]
    if not isinstance(records, list):
        return [f"{path}: '{kind}s' must be a list"]
    problems: list[str] = []
    seen_ids: set[str] = set()
    for index, record in enumerate(records):
        defect_id = record.get("id") if isinstance(record, dict) else None
        prefix = f"{path}: {kind} #{index}" + (f" ({defect_id})" if isinstance(defect_id, str) and defect_id else "")
        if not isinstance(record, dict):
            problems.append(f"{prefix}: record must be an object")
            continue
        if not isinstance(defect_id, str) or not defect_id:
            problems.append(f"{prefix}: 'id' must be a non-empty string")
        elif defect_id in seen_ids:
            problems.append(f"{prefix}: duplicate id")
        else:
            seen_ids.add(defect_id)
        check(record, problems, prefix)
    return problems
