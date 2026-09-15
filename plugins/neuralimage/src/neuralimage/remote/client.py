"""Streaming HTTP client; no Qt or neural-network dependencies."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlsplit
from urllib.request import Request, urlopen

from .contracts import API_VERSION, CHUNK_SIZE, digest, safe_path


class RemoteClient:
    def __init__(self, url: str, timeout: float = 30):
        parsed = urlsplit(url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc or parsed.query or parsed.fragment:
            raise ValueError("Enter an HTTP server URL, for example http://server:8765")
        self.url = url.rstrip("/") + "/api/v1"
        self.timeout = timeout

    def request(self, method: str, route: str, payload=None, *, raw: bytes | None = None):
        data = raw if raw is not None else json.dumps(payload).encode() if payload is not None else None
        request = Request(
            self.url + route,
            data=data,
            method=method,
            headers={"Content-Type": "application/octet-stream" if raw is not None else "application/json"},
        )
        with urlopen(request, timeout=self.timeout) as response:
            return json.load(response)

    def capabilities(self):
        result = self.request("GET", "/capabilities")
        if result["version"] != API_VERSION:
            raise ValueError("Client and server protocol versions differ")
        return result

    def upload(self, job: str, sources: dict[str, Path], progress=None, cancelled=None):
        total = sum(p.stat().st_size for p in sources.values())
        completed = 0
        for name, path in sources.items():
            route = f"/jobs/{job}/inputs/{quote(name, safe='/')}"
            failures = 0
            while True:
                if cancelled and cancelled.is_set():
                    raise InterruptedError("Transfer cancelled")
                try:
                    offset = self.request("GET", route)["offset"]
                    if offset > path.stat().st_size:
                        raise ValueError("Remote input is larger than local file")
                    with path.open("rb") as source:
                        source.seek(offset)
                        while chunk := source.read(CHUNK_SIZE):
                            if cancelled and cancelled.is_set():
                                raise InterruptedError("Client detached")
                            offset = self.request("PUT", route + f"?offset={offset}", raw=chunk)["offset"]
                            if progress:
                                progress(completed + offset, total)
                        if path.stat().st_size == 0:
                            self.request("PUT", route + "?offset=0", raw=b"")
                    break
                except HTTPError as error:
                    if error.code not in {409, 502, 503, 504}:
                        raise
                    failures += 1
                    if failures > 5:
                        raise
                    if cancelled:
                        cancelled.wait(min(failures, 5))
                    else:
                        time.sleep(min(failures, 5))
                except (URLError, TimeoutError):
                    failures += 1
                    if failures > 5:
                        raise
                    if cancelled:
                        cancelled.wait(min(failures, 5))
                    else:
                        time.sleep(min(failures, 5))
            completed += path.stat().st_size

    def download(self, job: str, destination: Path, cancelled=None) -> Path:
        output = destination / job
        output.mkdir(parents=True, exist_ok=True)
        for item in self.request("GET", f"/jobs/{job}/artifacts"):
            path = safe_path(output, item["path"])
            if path.is_file() and path.stat().st_size == item["size"] and digest(path) == item["sha256"]:
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            partial = path.with_name(path.name + ".partial")
            route = self.url + f"/jobs/{job}/artifacts/" + quote(item["path"], safe="/")
            with urlopen(route, timeout=self.timeout) as response, partial.open("wb") as target:
                while chunk := response.read(CHUNK_SIZE):
                    if cancelled and cancelled.is_set():
                        raise InterruptedError("Download detached")
                    target.write(chunk)
            if partial.stat().st_size != item["size"] or digest(partial) != item["sha256"]:
                raise ValueError(f"Downloaded file failed integrity check: {item['path']}")
            os.replace(partial, path)
        return output
