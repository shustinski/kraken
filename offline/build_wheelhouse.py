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


def ensure_local_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        try:
            if destination.samefile(source):
                return
        except OSError:
            pass
        destination.unlink()
    shutil.copy2(source, destination)


def download_into(url: str, expected_hash: str, destination: Path) -> None:
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
            return
        except (OSError, URLError, HTTPError) as error:
            partial.unlink(missing_ok=True)
            if attempt == 3:
                raise RuntimeError(f"Failed to download {url}") from error
            time.sleep(2**attempt)


def materialize(
    url: str,
    expected_hash: str,
    output: Path,
    cache: Path | None,
) -> str:
    """Return 'verified', 'from-cache', or 'downloaded'."""
    filename = distribution_filename(url)
    destination = output / filename
    if destination.exists() and sha256(destination) == expected_hash:
        return "verified"
    if destination.exists():
        destination.unlink()

    cache_file = (cache / filename) if cache is not None else None
    if cache_file is not None and cache_file.exists() and sha256(cache_file) == expected_hash:
        ensure_local_copy(cache_file, destination)
        return "from-cache"

    download_target = cache_file if cache_file is not None else destination
    if cache_file is not None:
        cache.mkdir(parents=True, exist_ok=True)
        if cache_file.exists():
            cache_file.unlink()
    download_into(url, expected_hash, download_target)
    if cache_file is not None:
        ensure_local_copy(cache_file, destination)
    return "downloaded"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--cache",
        type=Path,
        help="Persistent wheel cache on this PC. Downloads land here; only lock-needed files are copied to --output.",
    )
    args = parser.parse_args()
    lock = tomllib.loads(args.lock.read_text(encoding="utf-8"))
    args.output.mkdir(parents=True, exist_ok=True)
    if args.cache is not None:
        args.cache.mkdir(parents=True, exist_ok=True)
        for leftover in args.cache.glob("*.partial"):
            leftover.unlink(missing_ok=True)
    for leftover in args.output.glob("*.partial"):
        leftover.unlink(missing_ok=True)

    verified = 0
    from_cache = 0
    downloaded = 0
    needed_names: set[str] = set()
    for package in lock.get("package", []):
        wheels = [wheel for wheel in package.get("wheels", []) if is_windows_x64_wheel(wheel["url"])]
        distributions = wheels or ([package["sdist"]] if package.get("sdist") else [])
        for distribution in distributions:
            expected = distribution["hash"].removeprefix("sha256:")
            filename = distribution_filename(distribution["url"])
            needed_names.add(filename)
            result = materialize(distribution["url"], expected, args.output, args.cache)
            if result == "verified":
                verified += 1
            elif result == "from-cache":
                from_cache += 1
                print(f"From cache {filename}", flush=True)
            else:
                downloaded += 1
                print(f"Downloaded {filename}", flush=True)

    # Drop wheels in the kit output that are no longer in the lock (keep them in --cache).
    for existing in args.output.glob("*"):
        if existing.is_file() and existing.name not in needed_names and not existing.name.endswith(".partial"):
            existing.unlink()

    selected = verified + from_cache + downloaded
    if selected == 0:
        raise RuntimeError("No Windows x64 distributions were found in uv.lock")
    print(
        f"Wheelhouse ready in {args.output}: {verified} already in kit, {from_cache} from cache, "
        f"{downloaded} downloaded ({selected} locked distributions total)"
    )
    if args.cache is not None:
        print(f"Persistent wheel cache: {args.cache}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
