from __future__ import annotations

import time
from pathlib import Path
from typing import Callable, TypeVar


T = TypeVar('T')

FILE_NOT_FOUND_MAX_RETRIES = 5
FILE_NOT_FOUND_RETRY_DELAY_SECONDS = 5.0


def retry_file_read(
    operation: Callable[[], T],
    *,
    path: str | Path | None = None,
    max_retries: int = FILE_NOT_FOUND_MAX_RETRIES,
    delay_seconds: float = FILE_NOT_FOUND_RETRY_DELAY_SECONDS,
) -> T:
    normalized_path = Path(path) if path is not None else None
    retries = max(0, int(max_retries))
    delay = max(0.0, float(delay_seconds))

    for attempt in range(retries + 1):
        try:
            return operation()
        except FileNotFoundError as error:
            if normalized_path is not None:
                error.filename = str(normalized_path)
                if len(error.args) >= 2:
                    error.args = (*error.args[:-1], str(normalized_path))
            if attempt >= retries:
                raise
            time.sleep(delay)

    raise RuntimeError('Unreachable retry_file_read state reached.')
