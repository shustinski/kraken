"""Inventory public catalog operations and signature differences without opening a project.

Run with the repository Python environment. Output is JSON for review or CI use.
Availability alone does not establish equivalent behavior; see the catalog parity audit.
"""
from __future__ import annotations

import inspect
import json
from pathlib import Path

from kraken_hub.composition import EmbeddedProjectService
from kraken_hub.infrastructure.remote.client import RemoteServerProjectService
from kraken_manager.application.backend_contract import (
    BackendContractError, project_operation_names, validate_project_operations,
)


def operation_inventory():
    root = Path(__file__).resolve().parents[1]

    class CallableDefault:
        def __repr__(self):
            return "<callable>"

    def signature(method):
        original = inspect.signature(method)
        return str(original.replace(parameters=[
            parameter.replace(default=CallableDefault())
            if parameter.default is not inspect.Parameter.empty and callable(parameter.default)
            else parameter
            for parameter in original.parameters.values()
        ]))

    def methods(cls):
        result = {}
        for name, method in inspect.getmembers(cls, callable):
            if name.startswith("_"):
                continue
            source = Path(inspect.getsourcefile(method)).resolve()
            result[name] = {
                "signature": signature(method),
                "source": source.relative_to(root).as_posix(),
                "line": inspect.getsourcelines(method)[1],
            }
        return result

    local, remote = methods(EmbeddedProjectService), methods(RemoteServerProjectService)
    contract = project_operation_names()
    contract_errors = {}
    for label, cls in (("local", EmbeddedProjectService), ("remote", RemoteServerProjectService)):
        try:
            validate_project_operations(object.__new__(cls))
        except BackendContractError as exc:
            contract_errors[label] = str(exc)
    shared = sorted(local.keys() & remote.keys())
    mismatches = {}
    for name in shared:
        local_parameters = set(inspect.signature(getattr(EmbeddedProjectService, name)).parameters) - {"self"}
        remote_parameters = set(inspect.signature(getattr(RemoteServerProjectService, name)).parameters) - {"self"}
        if local_parameters != remote_parameters:
            mismatches[name] = {
                "local_only_parameters": sorted(local_parameters - remote_parameters),
                "remote_only_parameters": sorted(remote_parameters - local_parameters),
            }
    return {
        "local_count": len(local), "remote_count": len(remote), "shared_count": len(shared),
        "shared": shared, "local_only": sorted(local.keys() - remote.keys()),
        "remote_only": sorted(remote.keys() - local.keys()),
        "parameter_differences": mismatches, "local": local, "remote": remote,
        "required_operations": sorted(contract),
        "missing_local": sorted(contract - local.keys()),
        "missing_remote": sorted(contract - remote.keys()),
        "undeclared_shared": sorted(set(shared) - contract),
        "contract_errors": contract_errors,
    }


if __name__ == "__main__":
    inventory = operation_inventory()
    print(json.dumps(inventory, ensure_ascii=False, indent=2))
    if any(inventory[key] for key in ("missing_local", "missing_remote", "undeclared_shared", "contract_errors")):
        raise SystemExit(1)
