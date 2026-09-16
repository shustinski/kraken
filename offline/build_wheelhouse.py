"""Download and hash-check Windows x64 distributions named by a UV lockfile."""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import tomllib
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import unquote, urlparse
from urllib.request import Request, urlopen

# files.pythonhosted.org rejects the default Python-urllib User-Agent with HTTP 403.
_USER_AGENT = "kraken-offline-build-wheelhouse/1.0 (+https://github.com/)"


def is_windows_x64_wheel(url: str) -> bool:
    filename = Path(unquote(urlparse(url).path)).name.lower()
    return filename.endswith(".whl") and (filename.endswith("-none-any.whl") or "win_amd64" in filename)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def distribution_filename(url: str) -> str:
    filename = Path(unquote(urlparse(url).path)).name
    if not filename:
        raise ValueError(f"Distribution URL has no filename: {url}")
    return filename


def download(url: str, expected_hash: str, output: Path) -> str:
    """Return 'verified', 'downloaded', or raise after retries."""
    filename = distribution_filename(url)
    destination = output / filename
    if destination.exists() and sha256(destination) == expected_hash:
        return "verified"
    if destination.exists():
        destination.unlink()
    partial = destination.with_suffix(destination.suffix + ".partial")
    request = Request(url, headers={"User-Agent": _USER_AGENT})
    for attempt in range(4):
        try:
            with urlopen(request) as response, partial.open("wb") as stream:
                shutil.copyfileobj(response, stream)
            actual_hash = sha256(partial)
            if actual_hash != expected_hash:
                raise ValueError(f"SHA-256 mismatch for {url}: expected {expected_hash}, got {actual_hash}")
            partial.replace(destination)
            return "downloaded"
        except (OSError, URLError, HTTPError) as error:
            partial.unlink(missing_ok=True)
            if attempt == 3:
                raise RuntimeError(f"Failed to download {url}") from error
            time.sleep(2**attempt)
    raise RuntimeError(f"Failed to download {url}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    lock = tomllib.loads(args.lock.read_text(encoding="utf-8"))
    args.output.mkdir(parents=True, exist_ok=True)
    for leftover in args.output.glob("*.partial"):
        leftover.unlink(missing_ok=True)
    verified = 0
    downloaded = 0
    for package in lock.get("package", []):
        wheels = [wheel for wheel in package.get("wheels", []) if is_windows_x64_wheel(wheel["url"])]
        distributions = wheels or ([package["sdist"]] if package.get("sdist") else [])
        for distribution in distributions:
            expected = distribution["hash"].removeprefix("sha256:")
            result = download(distribution["url"], expected, args.output)
            if result == "verified":
                verified += 1
            else:
                downloaded += 1
                print(f"Downloaded {distribution_filename(distribution['url'])}", flush=True)
    selected = verified + downloaded
    if selected == 0:
        raise RuntimeError("No Windows x64 distributions were found in uv.lock")
    print(
        f"Wheelhouse ready in {args.output}: {verified} already present, {downloaded} downloaded "
        f"({selected} locked distributions total)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())