"""Validate adapter completeness before exposing an application to a frontend."""

import inspect

from kraken_manager.application.errors import ApplicationError
from kraken_manager.application.project_operations import ProjectOperations


class BackendContractError(ApplicationError):
    code = "backend_contract_invalid"


def project_operation_names() -> frozenset[str]:
    return frozenset(
        name
        for interface in ProjectOperations.__mro__
        for name, member in vars(interface).items()
        if not name.startswith("_") and callable(member)
    )


def validate_project_operations(adapter: ProjectOperations) -> None:
    errors = []
    for name in sorted(project_operation_names()):
        try:
            declared = inspect.getattr_static(adapter, name)
        except AttributeError:
            errors.append(f"{name}: missing operation")
            continue
        if declared is inspect.getattr_static(ProjectOperations, name):
            errors.append(f"{name}: inherited protocol declaration has no implementation")
            continue
        method = getattr(adapter, name)
        if not callable(method):
            errors.append(f"{name}: operation is not callable")
            continue
        contract = inspect.signature(getattr(ProjectOperations, name))
        implementation = inspect.signature(method)
        expected = {key: param for key, param in contract.parameters.items() if key != "self"}
        actual = implementation.parameters
        for key, parameter in expected.items():
            if key not in actual:
                errors.append(f"{name}: missing parameter {key}")
            elif parameter.default is not inspect.Parameter.empty and actual[key].default is inspect.Parameter.empty:
                errors.append(f"{name}: optional parameter {key} became required")
            elif (
                parameter.kind in {inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD}
                and actual[key].kind is inspect.Parameter.KEYWORD_ONLY
            ):
                errors.append(f"{name}: positional parameter {key} became keyword-only")
            elif (
                parameter.kind in {inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY}
                and actual[key].kind is inspect.Parameter.POSITIONAL_ONLY
            ):
                errors.append(f"{name}: keyword parameter {key} became positional-only")
        positional_kinds = {inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD}
        expected_positional = [key for key, parameter in expected.items() if parameter.kind in positional_kinds]
        actual_positional = [key for key, parameter in actual.items() if parameter.kind in positional_kinds]
        if actual_positional[:len(expected_positional)] != expected_positional:
            errors.append(f"{name}: positional parameter order does not match the contract")
        for key, parameter in actual.items():
            if (
                key not in expected
                and parameter.default is inspect.Parameter.empty
                and parameter.kind not in {inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD}
            ):
                errors.append(f"{name}: unexpected required parameter {key}")
    if errors:
        raise BackendContractError(f"{type(adapter).__name__}: " + "; ".join(errors))
