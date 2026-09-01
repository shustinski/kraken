"""Repeatable aggregate throughput check for the Blob Gateway data plane."""

from __future__ import annotations

import hashlib
import http.client
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

from .blob_gateway import BlobGatewayManager
from .configuration import ServerConfig


def _zero_digest(size: int) -> str:
    hasher = hashlib.sha256()
    block = bytes(1024 * 1024)
    remaining = size
    while remaining:
        count = min(remaining, len(block))
        hasher.update(block[:count])
        remaining -= count
    return hasher.hexdigest()


@dataclass(frozen=True, slots=True)
class BlobBenchmarkResult:
    clients: int
    bytes_per_client: int
    total_bytes: int
    elapsed_seconds: float

    @property
    def gibibits_per_second(self) -> float:
        return self.total_bytes * 8 / self.elapsed_seconds / 1024**3

    @property
    def gigabits_per_second(self) -> float:
        return self.total_bytes * 8 / self.elapsed_seconds / 1_000_000_000

    def to_dict(self) -> dict[str, int | float]:
        return {
            "clients": self.clients,
            "bytes_per_client": self.bytes_per_client,
            "total_bytes": self.total_bytes,
            "elapsed_seconds": round(self.elapsed_seconds, 3),
            "gigabits_per_second": round(self.gigabits_per_second, 3),
            "gibibits_per_second": round(self.gibibits_per_second, 3),
        }


def run_blob_benchmark(config_path: Path, *, clients: int, size_mib: int) -> BlobBenchmarkResult:
    if not 1 <= clients <= 256:
        raise ValueError("Client count must be between 1 and 256")
    if not 1 <= size_mib <= 1024 * 1024:
        raise ValueError("Size must be between 1 MiB and 1 TiB per client")
    config = ServerConfig.load(config_path)
    if (
        config.blob_gateway_public_url is None
        or config.blob_gateway_bind is None
        or config.blob_gateway_executable is None
        or config.blob_gateway_secret_value is None
    ):
        raise ValueError("Blob Gateway is not enabled in this Kraken Server configuration")
    manager = BlobGatewayManager(
        public_url=config.blob_gateway_public_url,
        bind=config.blob_gateway_bind,
        blob_root=config.blob_root,
        executable=config.blob_gateway_executable,
        secret=config.blob_gateway_secret_value,
        ticket_lifetime_seconds=config.blob_ticket_lifetime_seconds,
        tls_cert_file=config.blob_gateway_tls_cert_file,
        tls_key_file=config.blob_gateway_tls_key_file,
    )
    size = size_mib * 1024 * 1024
    digest = _zero_digest(size)
    url = f"{config.blob_gateway_public_url}/v1/blobs/{digest}"
    parsed_url = urlparse(url)

    try:
        manager.start()

        def upload(index: int) -> None:
            ticket = manager.signer.issue("upload", digest, size, context={"benchmark": index})
            connection_type = (
                http.client.HTTPSConnection if parsed_url.scheme == "https" else http.client.HTTPConnection
            )
            connection = connection_type(parsed_url.hostname, parsed_url.port, timeout=max(120, size_mib * 4))
            try:
                connection.putrequest("PUT", parsed_url.path)
                connection.putheader("Authorization", f"Bearer {ticket.token}")
                connection.putheader("Content-Type", "application/octet-stream")
                connection.putheader("Content-Length", str(size))
                connection.endheaders()
                block = bytes(1024 * 1024)
                remaining = size
                while remaining:
                    count = min(remaining, len(block))
                    connection.send(block if count == len(block) else block[:count])
                    remaining -= count
                response = connection.getresponse()
                raw = response.read()
                if response.status != 200:
                    raise RuntimeError(f"Blob Gateway benchmark upload failed: HTTP {response.status}: {raw!r}")
                payload = json.loads(raw)
            finally:
                connection.close()
            if payload.get("sha256") != digest or int(payload.get("size_bytes", -1)) != size:
                raise RuntimeError("Blob Gateway returned an invalid benchmark result")

        started = time.perf_counter()
        with ThreadPoolExecutor(max_workers=clients, thread_name_prefix="kraken-blob-benchmark") as pool:
            futures = [pool.submit(upload, index) for index in range(clients)]
            for future in as_completed(futures):
                future.result()
        elapsed = time.perf_counter() - started
        return BlobBenchmarkResult(clients, size, clients * size, elapsed)
    finally:
        manager.stop()


__all__ = ["BlobBenchmarkResult", "run_blob_benchmark"]
