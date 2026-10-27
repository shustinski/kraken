"""remote_client: download."""
from __future__ import annotations
import json
import os
import shutil
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from kraken_hub.remote_client_parts.extract_review_archive import RemoteServerError


def _cleanup_partial_download(destination: Path, identity: tuple[int, int] | None) -> None:
    if identity is None:
        return
    try:
        current = destination.lstat()
    except FileNotFoundError:
        return
    if (current.st_dev, current.st_ino) == identity:
        destination.unlink(missing_ok=True)


class DownloadMixin:
    def upload(
        self,
        path: str,
        *,
        token: str,
        source: Path,
        headers: Mapping[str, str],
    ) -> Any:
        request_headers = {
            "Authorization": f"Bearer {token}",
            "Accept": "application/json",
            "Content-Type": "application/octet-stream",
            **dict(headers),
        }
        try:
            with source.open("rb") as stream:
                request = urllib.request.Request(
                    f"{self.base_url}{path}",
                    data=stream,
                    headers=request_headers,
                    method="POST",
                )
                request.add_header("Content-Length", str(source.stat().st_size))
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    raw = response.read()
            return None if not raw else json.loads(raw.decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            parsed: Mapping[str, object] | None = None
            message = detail
            try:
                candidate = json.loads(detail)
                parsed = candidate if isinstance(candidate, Mapping) else None
                if parsed is not None:
                    message = str(parsed.get("detail") or parsed.get("title") or detail)
            except (json.JSONDecodeError, TypeError, AttributeError):
                pass
            raise RemoteServerError(
                message, status=exc.code, body=detail, problem=parsed
            ) from exc
        except urllib.error.URLError as exc:
            raise RemoteServerError(
                f"Cannot reach Kraken Server at {self.base_url}: {exc.reason}"
            ) from exc

    def upload_url(self, url: str, *, token: str, source: Path) -> Any:
        request_headers = {
            "Authorization": f"Bearer {token}",
            "Accept": "application/json",
            "Content-Type": "application/octet-stream",
            "Content-Length": str(source.stat().st_size),
        }
        try:
            with source.open("rb") as stream:
                request = urllib.request.Request(url, data=stream, headers=request_headers, method="PUT")
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    raw = response.read()
            return None if not raw else json.loads(raw.decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RemoteServerError(detail, status=exc.code, body=detail) from exc
        except urllib.error.URLError as exc:
            raise RemoteServerError(f"Cannot reach Kraken Blob Gateway at {url}: {exc.reason}") from exc

    def download(
        self,
        path: str,
        *,
        token: str,
        destination: Path,
        method: str = "GET",
        headers: Mapping[str, str] | None = None,
    ) -> Path:
        request_headers = {
            "Authorization": f"Bearer {token}",
            "Accept": "application/octet-stream",
            **dict(headers or {}),
        }
        request = urllib.request.Request(
            f"{self.base_url}{path}",
            headers=request_headers,
            method=method,
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        created_identity = None
        try:
            with (
                urllib.request.urlopen(request, timeout=self.timeout) as response,
                destination.open("xb") as output,
            ):
                created = os.fstat(output.fileno())
                created_identity = (created.st_dev, created.st_ino)
                shutil.copyfileobj(response, output, length=1024 * 1024)
        except urllib.error.HTTPError as exc:
            _cleanup_partial_download(destination, created_identity)
            detail = exc.read().decode("utf-8", errors="replace")
            parsed: Mapping[str, object] | None = None
            message = detail
            try:
                candidate = json.loads(detail)
                parsed = candidate if isinstance(candidate, Mapping) else None
                if parsed is not None:
                    message = str(parsed.get("detail") or parsed.get("title") or detail)
            except (json.JSONDecodeError, TypeError, AttributeError):
                pass
            raise RemoteServerError(
                message, status=exc.code, body=detail, problem=parsed
            ) from exc
        except Exception:
            _cleanup_partial_download(destination, created_identity)
            raise
        return destination

    def download_url(self, url: str, *, token: str, destination: Path) -> Path:
        request = urllib.request.Request(
            url,
            headers={"Authorization": f"Bearer {token}", "Accept": "application/octet-stream"},
            method="GET",
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        created_identity = None
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response, destination.open("xb") as output:
                created = os.fstat(output.fileno())
                created_identity = (created.st_dev, created.st_ino)
                shutil.copyfileobj(response, output, length=1024 * 1024)
        except Exception:
            _cleanup_partial_download(destination, created_identity)
            raise
        return destination
