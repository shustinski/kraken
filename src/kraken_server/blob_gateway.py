"""Signed transfer tickets and lifecycle management for Kraken Blob Gateway."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import socket
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urlparse


def _base64url(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).rstrip(b"=").decode("ascii")


def _decode_base64url(value: str) -> bytes:
    return base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))


def _unix_time() -> int:
    return int(time.time())


def validate_public_gateway_url(value: str) -> str:
    url = value.strip().rstrip("/")
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.path not in {"", "/"}:
        raise ValueError("Blob Gateway public URL must contain only an http(s) scheme, host, and optional port")
    try:
        loopback = parsed.hostname == "localhost" or socket.gethostbyname(parsed.hostname).startswith("127.")
    except OSError:
        loopback = False
    if parsed.scheme != "https" and not loopback:
        raise ValueError("Remote Blob Gateway traffic must use HTTPS")
    return url


@dataclass(frozen=True, slots=True)
class BlobTicket:
    token: str
    expires_at: int


class BlobTicketSigner:
    """Issue compact HMAC tickets understood by the Rust data plane."""

    def __init__(self, secret: str | bytes, *, lifetime_seconds: int = 900) -> None:
        raw = secret.encode("utf-8") if isinstance(secret, str) else bytes(secret)
        if len(raw) < 32:
            raise ValueError("Blob Gateway secret must contain at least 32 bytes")
        if not 30 <= int(lifetime_seconds) <= 86_400:
            raise ValueError("Blob ticket lifetime must be between 30 and 86400 seconds")
        self._secret = raw
        self.lifetime_seconds = int(lifetime_seconds)

    def issue(
        self,
        operation: str,
        digest: str,
        size_bytes: int,
        *,
        context: Mapping[str, object] | None = None,
    ) -> BlobTicket:
        digest = _validated_digest(digest)
        if operation not in {"upload", "download"}:
            raise ValueError("Blob ticket operation must be upload or download")
        if isinstance(size_bytes, bool) or int(size_bytes) < 0:
            raise ValueError("Blob size must be non-negative")
        expires_at = _unix_time() + self.lifetime_seconds
        payload: dict[str, object] = {
            "v": 1,
            "op": operation,
            "digest": digest,
            "size": int(size_bytes),
            "exp": expires_at,
        }
        if context:
            payload["ctx"] = {str(key): str(value) for key, value in context.items()}
        encoded = _base64url(json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8"))
        signature = _base64url(hmac.digest(self._secret, encoded.encode("ascii"), hashlib.sha256))
        return BlobTicket(f"{encoded}.{signature}", expires_at)

    def verify(self, token: str, *, operation: str) -> dict[str, Any]:
        try:
            encoded, signature = token.split(".", 1)
            supplied = _decode_base64url(signature)
            expected = hmac.digest(self._secret, encoded.encode("ascii"), hashlib.sha256)
            if not hmac.compare_digest(supplied, expected):
                raise ValueError("signature mismatch")
            payload = json.loads(_decode_base64url(encoded))
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            raise ValueError("Invalid Blob Gateway ticket") from exc
        if not isinstance(payload, dict) or payload.get("v") != 1 or payload.get("op") != operation:
            raise ValueError("Blob Gateway ticket does not match the operation")
        if int(payload.get("exp", 0)) < _unix_time():
            raise ValueError("Blob Gateway ticket has expired")
        payload["digest"] = _validated_digest(str(payload.get("digest", "")))
        size = payload.get("size")
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise ValueError("Blob Gateway ticket has an invalid size")
        return payload


class BlobGatewayManager:
    """Own the optional local gateway process and issue public transfer contracts."""

    def __init__(
        self,
        *,
        public_url: str,
        bind: str,
        blob_root: Path,
        executable: Path,
        secret: str,
        ticket_lifetime_seconds: int = 900,
        tls_cert_file: Path | None = None,
        tls_key_file: Path | None = None,
    ) -> None:
        self.public_url = validate_public_gateway_url(public_url)
        self.bind = bind.strip()
        if not self.bind:
            raise ValueError("Blob Gateway bind address is required")
        self.blob_root = blob_root.resolve()
        self.executable = executable.resolve()
        self.signer = BlobTicketSigner(secret, lifetime_seconds=ticket_lifetime_seconds)
        if (tls_cert_file is None) != (tls_key_file is None):
            raise ValueError("Blob Gateway TLS certificate and key must be configured together")
        self.tls_cert_file = None if tls_cert_file is None else tls_cert_file.resolve()
        self.tls_key_file = None if tls_key_file is None else tls_key_file.resolve()
        if self.public_url.startswith("https://") != (self.tls_cert_file is not None):
            raise ValueError("Blob Gateway public URL scheme must match its TLS certificate configuration")
        self._secret = secret
        self._process: subprocess.Popen[bytes] | None = None

    def start(self) -> None:
        if self._process is not None:
            return
        if self._ready():
            return
        if not self.executable.is_file():
            raise RuntimeError(f"Kraken Blob Gateway executable was not found: {self.executable}")
        environment = os.environ.copy()
        environment.update(
            {
                "KRAKEN_BLOB_ROOT": str(self.blob_root),
                "KRAKEN_BLOB_GATEWAY_BIND": self.bind,
                "KRAKEN_BLOB_GATEWAY_SECRET": self._secret,
            }
        )
        if self.tls_cert_file is not None and self.tls_key_file is not None:
            environment["KRAKEN_BLOB_GATEWAY_TLS_CERT"] = str(self.tls_cert_file)
            environment["KRAKEN_BLOB_GATEWAY_TLS_KEY"] = str(self.tls_key_file)
        flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        self._process = subprocess.Popen(  # noqa: S603 - executable is administrator-owned configuration
            [str(self.executable)],
            env=environment,
            stdin=subprocess.DEVNULL,
            creationflags=flags,
        )
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline:
            if self._process.poll() is not None:
                code = self._process.returncode
                self._process = None
                raise RuntimeError(f"Kraken Blob Gateway exited during startup with code {code}")
            if self._ready():
                return
            time.sleep(0.1)
        self.stop()
        raise RuntimeError(f"Kraken Blob Gateway did not become ready at {self.public_url}")

    def _ready(self) -> bool:
        try:
            with urllib.request.urlopen(f"{self.public_url}/health", timeout=0.5) as response:  # noqa: S310
                health = json.loads(response.read())
            if health.get("service") != "kraken-blob-gateway":
                return False
            probe = self.signer.issue("download", "0" * 64, 0)
            request = urllib.request.Request(
                f"{self.public_url}/v1/blobs/{'0' * 64}",
                headers={"Authorization": f"Bearer {probe.token}"},
            )
            try:
                urllib.request.urlopen(request, timeout=0.5)  # noqa: S310
            except urllib.error.HTTPError as exc:
                return exc.code == 404
            return False
        except (OSError, ValueError, TypeError, json.JSONDecodeError, urllib.error.URLError):
            return False

    def stop(self) -> None:
        process = self._process
        self._process = None
        if process is None or process.poll() is not None:
            return
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)

    def ticket(
        self,
        operation: str,
        digest: str,
        size_bytes: int,
        *,
        context: Mapping[str, object] | None = None,
    ) -> dict[str, object]:
        issued = self.signer.issue(operation, digest, size_bytes, context=context)
        return {
            "mode": "gateway",
            "url": f"{self.public_url}/v1/blobs/{digest}",
            "token": issued.token,
            "expires_at": issued.expires_at,
            "sha256": digest,
            "size_bytes": int(size_bytes),
        }


def _validated_digest(value: str) -> str:
    digest = value.strip()
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError("A lowercase SHA-256 digest is required")
    return digest


__all__ = ["BlobGatewayManager", "BlobTicket", "BlobTicketSigner", "validate_public_gateway_url"]
