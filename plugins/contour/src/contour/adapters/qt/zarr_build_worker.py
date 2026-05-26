from __future__ import annotations

import json
import sys
from pathlib import Path

from ...application.frame_lod import ZarrFrameStore


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args:
        print("Missing Zarr build manifest path.", file=sys.stderr)
        return 2
    try:
        payload = json.loads(Path(args[0]).read_text(encoding="utf-8"))
        image_paths = [Path(path) for path in payload.get("image_paths", [])]
        zarr_path = Path(str(payload["zarr_path"]))
    except Exception as exc:
        print(f"Invalid Zarr build manifest: {exc}", file=sys.stderr)
        return 2
    built = ZarrFrameStore._build_zarr_pyramid(image_paths, zarr_path)
    if built is None:
        print("Failed to build Zarr pyramid.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
