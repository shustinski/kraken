"""Download and hash-check Windows x64 distributions named by a UV lockfile."""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import tomllib
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import urlopen


def is_windows_x64_wheel(url: str) -> bool:
    filename = Path(urlparse(url).path).name.lower()
    return filename.endswith(".whl") and (filename.endswith("-none-any.whl") or "win_amd64" in filename)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, expected_hash: str, output: Path) -> None:
    filename = Path(urlparse(url).path).name
    if not filename:
        raise ValueError(f"Distribution URL has no filename: {url}")
    destination = output / filename
    if destination.exists() and sha256(destination) == expected_hash:
        return
    partial = destination.with_suffix(destination.suffix + ".partial")
    with urlopen(url) as response, partial.open("wb") as stream:
        shutil.copyfileobj(response, stream)
    actual_hash = sha256(partial)
    if actual_hash != expected_hash:
        partial.unlink(missing_ok=True)
        raise ValueError(f"SHA-256 mismatch for {url}: expected {expected_hash}, got {actual_hash}")
    partial.replace(destination)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    lock = tomllib.loads(args.lock.read_text(encoding="utf-8"))
    args.output.mkdir(parents=True, exist_ok=True)
    selected = 0
    for package in lock.get("package", []):
        wheels = [wheel for wheel in package.get("wheels", []) if is_windows_x64_wheel(wheel["url"])]
        distributions = wheels or ([package["sdist"]] if package.get("sdist") else [])
        for distribution in distributions:
            expected = distribution["hash"].removeprefix("sha256:")
            download(distribution["url"], expected, args.output)
            selected += 1
    if selected == 0:
        raise RuntimeError("No Windows x64 distributions were found in uv.lock")
    print(f"Downloaded or verified {selected} locked distributions in {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
