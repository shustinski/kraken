"""remote_client: iter bytes."""
from __future__ import annotations
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Iterator

from kraken_hub.infrastructure.remote.parts.extract_review_archive import RemoteServerError


class IterBytesMixin:
    def __init__(self, base_url: str, *, timeout: float = 30.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def iter_bytes(self, path: str, *, token: str) -> Iterator[bytes]:
        """Read a server asset incrementally without creating a local file."""
        request = urllib.request.Request(
            f"{self.base_url}{path}",
            headers={"Authorization": f"Bearer {token}", "Accept": "application/octet-stream"},
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                while chunk := response.read(1024 * 1024):
                    yield chunk
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RemoteServerError(detail, status=exc.code, body=detail) from exc
        except urllib.error.URLError as exc:
            raise RemoteServerError(f"Cannot reach Kraken Server at {self.base_url}: {exc.reason}") from exc
