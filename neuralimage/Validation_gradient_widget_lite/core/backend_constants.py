"""Store backend-related constants used by the lite repository and matrix modules."""
from __future__ import annotations

import re
from pathlib import Path

NATURAL_SPLIT_PATTERN = re.compile(r"(\d+)")
INVALID_FILENAME_PATTERN = re.compile(r'[<>:"/\\|?*\x00-\x1F]+')
FRAME_NUMBER_PATTERN = re.compile(r"^\d+$")
CACHE_DIR = Path(__file__).resolve().parent / ".cache"
SCORE_CACHE_DB = CACHE_DIR / "score_cache.sqlite3"
IMAGE_CACHE_SIZE = 256
SQLITE_BATCH_SIZE = 900
MISMATCH_WAIT_TIMEOUT = 0.05
