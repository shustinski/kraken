from pathlib import Path
from uuid import uuid4


def make_test_dir(prefix: str) -> Path:
    root = Path(".test_runtime").resolve()
    root.mkdir(exist_ok=True)
    target = root / f"{prefix}_{uuid4().hex}"
    target.mkdir(parents=True, exist_ok=True)
    return target
