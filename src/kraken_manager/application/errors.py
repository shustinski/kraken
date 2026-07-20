"""Application-layer errors with stable machine-readable codes."""

from __future__ import annotations


class ApplicationError(RuntimeError):
    code = "application_error"


class NotFoundError(ApplicationError):
    code = "not_found"


class ConflictError(ApplicationError):
    code = "conflict"


class ConcurrencyError(ConflictError):
    code = "revision_conflict"


class AuthorizationError(ApplicationError):
    code = "forbidden"


class AuthenticationRequiredError(AuthorizationError):
    code = "authentication_required"


class StorageCapabilityError(ApplicationError):
    code = "storage_capability_missing"


__all__ = [
    "ApplicationError",
    "AuthenticationRequiredError",
    "AuthorizationError",
    "ConcurrencyError",
    "ConflictError",
    "NotFoundError",
    "StorageCapabilityError",
]
