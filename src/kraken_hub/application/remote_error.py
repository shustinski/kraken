"""Failure reported by Kraken Server, without an HTTP client attached."""

from __future__ import annotations

from collections.abc import Mapping


class RemoteServerError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        status: int | None = None,
        body: str = "",
        problem: Mapping[str, object] | None = None,
    ) -> None:
        super().__init__(message)
        self.status = status
        self.body = body
        self.problem = dict(problem or {})
        self.code = str(self.problem.get("code", ""))
