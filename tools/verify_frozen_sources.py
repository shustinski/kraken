"""Check selected frozen modules against current source without executing them."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from types import CodeType

from PyInstaller.archive.readers import CArchiveReader


def normalized(code: CodeType) -> CodeType:
    return code.replace(
        co_filename="",
        co_consts=tuple(normalized(item) if isinstance(item, CodeType) else item for item in code.co_consts),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable", type=Path, required=True)
    parser.add_argument("--module", action="append", required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    executable = args.executable.resolve(strict=True)
    archive = CArchiveReader(str(executable)).open_embedded_archive("PYZ.pyz")
    result = {"status": "passed", "executable": str(executable),
              "executable_sha256": hashlib.sha256(executable.read_bytes()).hexdigest(), "modules": []}
    for module in args.module:
        if not module or not all(part.isidentifier() for part in module.split(".")):
            raise ValueError("Expected a Python module name")
        candidates = []
        for source_root in (root / "src", *(root / "plugins").glob("*/src")):
            candidate = source_root / Path(*module.split("."))
            candidate = candidate.with_suffix(".py") if candidate.with_suffix(".py").is_file() else candidate / "__init__.py"
            if candidate.is_file():
                candidates.append(candidate)
        if len(candidates) != 1:
            raise ValueError(f"Expected exactly one workspace source for {module}: {candidates}")
        source, = candidates
        data = source.read_bytes()
        expected = compile(data, str(source), "exec", dont_inherit=True, optimize=0)
        actual = archive.extract(module)
        if not isinstance(actual, CodeType) or normalized(actual) != normalized(expected):
            raise RuntimeError(f"Frozen module differs from current source: {module}")
        result["modules"].append({"name": module, "source_sha256": hashlib.sha256(data).hexdigest()})
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
