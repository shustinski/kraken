"""remote_client: request."""
from __future__ import annotations
import json
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Mapping
from typing import Any

from kraken_hub.remote_client_parts.extract_review_archive import RemoteServerError


class RequestMixin:
    def request(
        self,
        method: str,
        path: str,
        *,
        token: str,
        payload: Mapping[str, object] | None = None,
        headers: Mapping[str, str] | None = None,
        timeout: float | None = None,
    ) -> Any:
        url = f"{self.base_url}{path}"
        body = (
            None
            if payload is None
            else json.dumps(dict(payload), ensure_ascii=False).encode("utf-8")
        )
        request_headers = {
            "Authorization": f"Bearer {token}",
            "Accept": "application/json",
        }
        if body is not None:
            request_headers["Content-Type"] = "application/json"
        if headers:
            request_headers.update(headers)
        request = urllib.request.Request(
            url, data=body, headers=request_headers, method=method.upper()
        )
        try:
            with urllib.request.urlopen(
                request, timeout=self.timeout if timeout is None else timeout
            ) as response:
                raw = response.read()
                if not raw:
                    return None
                return json.loads(raw.decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            message = detail
            parsed: Mapping[str, object] | None = None
            try:
                candidate = json.loads(detail)
                parsed = candidate if isinstance(candidate, Mapping) else None
                if parsed is not None:
                    message = str(parsed.get("detail") or parsed.get("title") or detail)
            except (json.JSONDecodeError, TypeError, AttributeError):
                message = detail
            raise RemoteServerError(
                message, status=exc.code, body=detail, problem=parsed
            ) from exc
        except urllib.error.URLError as exc:
            raise RemoteServerError(
                f"Cannot reach Kraken Server at {self.base_url}: {exc.reason}"
            ) from exc
