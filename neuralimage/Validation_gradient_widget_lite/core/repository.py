"""Load frame pairs and compute mismatch-only comparison metrics for the lite widget."""
from __future__ import annotations

import hashlib
import json
import sqlite3
from collections import deque
from collections.abc import Callable, Iterable
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import replace
from functools import lru_cache
from pathlib import Path

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QImage

from .backend_constants import FRAME_NUMBER_PATTERN, IMAGE_CACHE_SIZE, INVALID_FILENAME_PATTERN, MISMATCH_WAIT_TIMEOUT, NATURAL_SPLIT_PATTERN, SCORE_CACHE_DB, SQLITE_BATCH_SIZE
from .domain import BuildOptions, BuildResult, ComparisonMode, FolderSpec, FrameIdentity, FrameRecord


class BuildCancelledError(RuntimeError):
    """Signal cooperative cancellation during matrix build or mismatch computation."""


class _ScoreCache:
    """Cache mismatch scores for previously processed frame pairs."""

    def __init__(self, db_path: Path) -> None:
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(str(self._db_path))
        self._connection.execute(
            'CREATE TABLE IF NOT EXISTS score_cache (cache_key TEXT PRIMARY KEY, score REAL NOT NULL)'
        )

    def get_many(self, keys: Iterable[str]) -> dict[str, float]:
        normalized = [str(key) for key in keys]
        if not normalized:
            return {}
        results: dict[str, float] = {}
        for start in range(0, len(normalized), SQLITE_BATCH_SIZE):
            batch = normalized[start:start + SQLITE_BATCH_SIZE]
            placeholders = ','.join('?' for _ in batch)
            rows = self._connection.execute(
                f'SELECT cache_key, score FROM score_cache WHERE cache_key IN ({placeholders})',
                batch,
            ).fetchall()
            results.update({str(cache_key): float(score) for cache_key, score in rows})
        return results

    def put_many(self, values: Iterable[tuple[str, float]]) -> None:
        payload = [(str(cache_key), float(score)) for cache_key, score in values]
        if not payload:
            return
        for start in range(0, len(payload), SQLITE_BATCH_SIZE):
            batch = payload[start:start + SQLITE_BATCH_SIZE]
            self._connection.executemany(
                'INSERT OR REPLACE INTO score_cache(cache_key, score) VALUES (?, ?)',
                batch,
            )
        self._connection.commit()

    def close(self) -> None:
        self._connection.close()


def natural_sort_key(value: str) -> tuple[object, ...]:
    """Split a string into digit and text chunks for natural sorting."""
    parts = NATURAL_SPLIT_PATTERN.split(str(value).lower())
    key: list[object] = []
    for part in parts:
        if not part:
            continue
        key.append(int(part) if part.isdigit() else part)
    return tuple(key)


def extract_frame_id(value: str) -> int:
    """Extract the frame number from the last underscore-separated filename segment."""
    stem = Path(str(value)).stem
    last_segment = stem.rsplit('_', 1)[-1]
    if not last_segment.isdigit():
        raise ValueError(f"Unable to extract frame id from '{value}'")
    return int(last_segment)


def sanitize_folder_name(value: str) -> str:
    """Convert arbitrary user text into a filesystem-safe folder name."""
    cleaned = INVALID_FILENAME_PATTERN.sub('_', str(value).strip())
    cleaned = cleaned.strip(' .')
    return cleaned or 'layer'


def build_frame_identity(key: str, *, has_base: bool, fallback_frame_id: int) -> FrameIdentity:
    """Build one frame identity from a filename and matrix-independent metadata."""
    display_name = Path(key).name
    try:
        frame_id = extract_frame_id(display_name)
    except ValueError:
        frame_id = int(fallback_frame_id)
    return FrameIdentity(
        frame_id=frame_id,
        base_id=frame_id if has_base else None,
        tile_x=None,
        tile_y=None,
        source_key=key,
    )


def iter_image_paths(folder: Path, *, recursive: bool, extensions: Iterable[str]) -> list[Path]:
    """Return image paths from a folder using the configured search mode."""
    normalized_extensions = {str(ext).lower() for ext in extensions}
    iterator = folder.rglob('*') if recursive else folder.glob('*')
    paths = [path for path in iterator if path.is_file() and path.suffix.lower() in normalized_extensions]
    return sorted(paths, key=lambda item: natural_sort_key(item.as_posix()))


def build_folder_index(
    folder: Path,
    *,
    recursive: bool,
    extensions: Iterable[str],
    cancel_check: Callable[[], bool] | None = None,
) -> dict[str, Path]:
    """Index frame files in one folder by their relative path."""
    index: dict[str, Path] = {}
    for image_path in iter_image_paths(folder, recursive=recursive, extensions=extensions):
        if cancel_check is not None and cancel_check():
            raise BuildCancelledError('Build cancelled')
        key = image_path.relative_to(folder).as_posix()
        index[key] = image_path
    return index


def _resolve_base_frame_path(
    key: str,
    base_index: dict[str, Path],
    base_name_index: dict[str, list[Path]],
    base_stem_index: dict[str, list[Path]],
) -> Path | None:
    """Resolve the best base-layer frame for one result-frame key."""
    exact = base_index.get(key)
    if exact is not None:
        return exact

    result_path = Path(key)
    result_name = result_path.name.lower()
    result_stem = result_path.stem.lower()
    result_suffix = result_path.suffix.lower()

    name_matches = base_name_index.get(result_name, [])
    if len(name_matches) == 1:
        return name_matches[0]

    stem_matches = base_stem_index.get(result_stem, [])
    if len(stem_matches) == 1:
        return stem_matches[0]

    same_suffix_matches = [path for path in stem_matches if path.suffix.lower() == result_suffix]
    if len(same_suffix_matches) == 1:
        return same_suffix_matches[0]
    return None


def _qimage_to_grayscale_array(image: QImage) -> np.ndarray:
    """Convert a Qt image into a compact grayscale numpy array."""
    grayscale = image.convertToFormat(QImage.Format.Format_Grayscale8)
    pointer = grayscale.bits()
    pointer.setsize(grayscale.height() * grayscale.bytesPerLine())
    buffer = np.frombuffer(pointer, dtype=np.uint8).reshape((grayscale.height(), grayscale.bytesPerLine()))
    return buffer[:, : grayscale.width()].copy()


def _grayscale_array_to_qimage(array: np.ndarray) -> QImage:
    """Convert a grayscale numpy array into a Qt image."""
    contiguous = np.ascontiguousarray(array.astype(np.uint8))
    height, width = contiguous.shape
    image = QImage(contiguous.data, width, height, contiguous.strides[0], QImage.Format.Format_Grayscale8)
    return image.copy()


def _image_signature(path: Path) -> tuple[str, int, int]:
    """Return one stable cache signature for a source image file."""
    resolved = Path(path).resolve()
    stat = resolved.stat()
    return str(resolved), int(stat.st_mtime_ns), int(stat.st_size)


@lru_cache(maxsize=IMAGE_CACHE_SIZE)
def _load_grayscale_image_cached(path_text: str, _mtime_ns: int, _size: int) -> np.ndarray:
    """Load one grayscale image into an in-memory LRU cache keyed by file signature."""
    image = QImage(path_text)
    if image.isNull():
        raise ValueError(f'Unable to decode image: {path_text}')
    return _qimage_to_grayscale_array(image)


@lru_cache(maxsize=IMAGE_CACHE_SIZE)
def _load_resized_grayscale_image_cached(path_text: str, mtime_ns: int, size: int, target_shape: tuple[int, int]) -> np.ndarray:
    """Load and resize one grayscale image inside the LRU cache."""
    source = _load_grayscale_image_cached(path_text, mtime_ns, size)
    return _resize_grayscale_array(source, target_shape)


def load_grayscale_image(path: Path) -> np.ndarray:
    """Load one image file as a grayscale numpy array."""
    return _load_grayscale_image_cached(*_image_signature(Path(path))).copy()


def _resize_grayscale_array(array: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    """Resize a grayscale array to the requested shape using Qt fast scaling."""
    target_height, target_width = int(target_shape[0]), int(target_shape[1])
    if array.shape == (target_height, target_width):
        return array.astype(np.uint8)
    image = _grayscale_array_to_qimage(array.astype(np.uint8))
    scaled = image.scaled(
        target_width,
        target_height,
        Qt.AspectRatioMode.IgnoreAspectRatio,
        Qt.TransformationMode.FastTransformation,
    )
    return _qimage_to_grayscale_array(scaled)


def resize_grayscale_image(array: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    """Resize a grayscale array to the requested shape using Qt fast scaling."""
    return _resize_grayscale_array(array, target_shape).copy()


def compute_comparison_score(first: np.ndarray, second: np.ndarray, mode: ComparisonMode) -> float:
    """Compute one scalar mismatch score for a pair of arrays."""
    if mode == ComparisonMode.OVERLAY_ONLY:
        return 0.0
    total = int(first.size)
    if total <= 0:
        return 0.0
    if mode == ComparisonMode.GRAYSCALE_DIFF:
        first_gray = np.asarray(first, dtype=np.int16)
        second_gray = np.asarray(second, dtype=np.int16)
        diff = np.abs(first_gray - second_gray)
        return float(diff.mean(dtype=np.float32) / 255.0)

    first_bool = first.astype(bool, copy=False)
    second_bool = second.astype(bool, copy=False)
    if mode in (ComparisonMode.XOR, ComparisonMode.DISAGREEMENT):
        mismatched = np.count_nonzero(np.logical_xor(first_bool, second_bool))
    elif mode == ComparisonMode.FIRST_MINUS_SECOND:
        mismatched = np.count_nonzero(np.logical_and(first_bool, np.logical_not(second_bool)))
    else:
        mismatched = np.count_nonzero(np.logical_and(np.logical_not(first_bool), second_bool))
    return float(mismatched / total)


def _process_frame_pair(key: str, first_path: Path, second_path: Path, mode: ComparisonMode) -> tuple[str, float]:
    """Load and compare one pair of frame files for threaded mismatch computation."""
    first_gray = load_grayscale_image(first_path)
    second_gray = resize_grayscale_image(load_grayscale_image(second_path), first_gray.shape)
    if mode == ComparisonMode.GRAYSCALE_DIFF:
        score = compute_comparison_score(first_gray, second_gray, mode)
    else:
        score = compute_comparison_score(first_gray >= 128, second_gray >= 128, mode)
    return key, score


def compute_comparison(first: np.ndarray, second: np.ndarray, mode: ComparisonMode) -> tuple[np.ndarray, float]:
    """Return a heatmap and scalar mismatch score for one pair of arrays."""
    if mode == ComparisonMode.GRAYSCALE_DIFF:
        first_gray = np.asarray(first, dtype=np.int16)
        second_gray = np.asarray(second, dtype=np.int16)
        heatmap = np.abs(first_gray - second_gray).astype(np.float32) / 255.0
        return heatmap, float(heatmap.mean(dtype=np.float32))

    first_bool = first.astype(bool)
    second_bool = second.astype(bool)
    if mode == ComparisonMode.OVERLAY_ONLY:
        heatmap = np.zeros_like(first_bool, dtype=np.float32)
    elif mode in (ComparisonMode.XOR, ComparisonMode.DISAGREEMENT):
        heatmap = np.logical_xor(first_bool, second_bool).astype(np.float32)
    elif mode == ComparisonMode.FIRST_MINUS_SECOND:
        heatmap = np.logical_and(first_bool, np.logical_not(second_bool)).astype(np.float32)
    else:
        heatmap = np.logical_and(np.logical_not(first_bool), second_bool).astype(np.float32)
    return heatmap, float(heatmap.mean(dtype=np.float32))


def collect_frame_records(
    first_folder: FolderSpec,
    second_folder: FolderSpec,
    options: BuildOptions,
    *,
    base_folder: FolderSpec | None = None,
    cancel_check: Callable[[], bool] | None = None,
) -> BuildResult:
    """Build the frame list for a matrix without calculating mismatches."""
    if cancel_check is not None and cancel_check():
        raise BuildCancelledError('Build cancelled')

    first_index = build_folder_index(first_folder.path, recursive=options.recursive, extensions=options.file_extensions, cancel_check=cancel_check)
    if cancel_check is not None and cancel_check():
        raise BuildCancelledError('Build cancelled')
    second_index = build_folder_index(second_folder.path, recursive=options.recursive, extensions=options.file_extensions, cancel_check=cancel_check)
    base_index = build_folder_index(base_folder.path, recursive=options.recursive, extensions=options.file_extensions, cancel_check=cancel_check) if base_folder is not None else {}
    base_name_index: dict[str, list[Path]] = {}
    base_stem_index: dict[str, list[Path]] = {}
    for base_path in base_index.values():
        base_name_index.setdefault(base_path.name.lower(), []).append(base_path)
        base_stem_index.setdefault(base_path.stem.lower(), []).append(base_path)

    common_keys = sorted(set(first_index.keys()) & set(second_index.keys()), key=natural_sort_key)
    if not common_keys:
        raise ValueError('Selected comparison folders do not contain matching image frames.')

    records = []
    for index, key in enumerate(common_keys):
        resolved_base_path = _resolve_base_frame_path(key, base_index, base_name_index, base_stem_index)
        records.append(
            FrameRecord(
                key=key,
                display_name=Path(key).name,
                identity=build_frame_identity(key, has_base=resolved_base_path is not None, fallback_frame_id=index),
                score=0.0,
                first_path=str(first_index[key]),
                second_path=str(second_index[key]),
                base_path=str(resolved_base_path) if resolved_base_path is not None else None,
                absolute_score=None,
                relative_score=None,
                score_ready=False,
            )
        )
    return BuildResult(
        records=tuple(records),
        first_folder=first_folder,
        second_folder=second_folder,
        base_folder=base_folder,
        options=options,
        min_score=0.0,
        max_score=0.0,
        eligible_key_count=len(common_keys),
        scores_computed=False,
        best_match_key=None,
        min_absolute_score=None,
        max_absolute_score=None,
    )


def _frame_pair_cache_key(first_path: Path, second_path: Path, mode: ComparisonMode) -> str:
    """Build a stable cache key for one pair of source frames."""
    first_stat = first_path.stat()
    second_stat = second_path.stat()
    payload = {
        'mode': mode.value,
        'first_path': str(first_path.resolve()),
        'first_mtime_ns': int(first_stat.st_mtime_ns),
        'first_size': int(first_stat.st_size),
        'second_path': str(second_path.resolve()),
        'second_mtime_ns': int(second_stat.st_mtime_ns),
        'second_size': int(second_stat.st_size),
    }
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode('utf-8')
    return hashlib.sha1(raw).hexdigest()


def compute_build_result_mismatches(
    build_result: BuildResult,
    *,
    comparison_mode: ComparisonMode | None = None,
    display_metric: str = 'relative',
    progress_callback: Callable[[int, int, str], None] | None = None,
    active_keys_callback: Callable[[tuple[str, ...]], None] | None = None,
    cancel_check: Callable[[], bool] | None = None,
) -> BuildResult:
    """Compute mismatch scores for an existing matrix build result."""
    if cancel_check is not None and cancel_check():
        raise BuildCancelledError('Build cancelled')

    records = list(build_result.records)
    if not records:
        return replace(build_result, records=tuple(), min_score=0.0, max_score=0.0, scores_computed=True, best_match_key=None, min_absolute_score=0.0, max_absolute_score=0.0)

    mode = comparison_mode or build_result.options.comparison_mode
    total = len(records)
    progress_step = max(1, int(build_result.options.progress_update_interval))
    max_workers = max(1, min(int(build_result.options.max_workers), total))
    scores_by_key: dict[str, float] = {}
    processed = 0
    last_reported = 0
    cache_keys_by_frame: dict[str, str] = {}
    cache_entries_to_store: list[tuple[str, float]] = []
    score_cache: _ScoreCache | None = _ScoreCache(SCORE_CACHE_DB) if build_result.options.cache_enabled else None

    try:
        if score_cache is not None:
            requested_cache_keys: list[str] = []
            for record in records:
                cache_key = _frame_pair_cache_key(Path(record.first_path), Path(record.second_path), mode)
                cache_keys_by_frame[record.key] = cache_key
                requested_cache_keys.append(cache_key)
            cached_scores = score_cache.get_many(requested_cache_keys)
        else:
            cached_scores = {}

        missing_records: list[FrameRecord] = []
        for record in records:
            cache_key = cache_keys_by_frame.get(record.key)
            if cache_key is not None and cache_key in cached_scores:
                scores_by_key[record.key] = float(cached_scores[cache_key])
                processed += 1
                if progress_callback is not None and (processed - last_reported >= progress_step or processed == total):
                    progress_callback(processed, total, record.key)
                    last_reported = processed
            else:
                missing_records.append(record)

        if active_keys_callback is not None:
            active_keys_callback(tuple())

        if missing_records:
            with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix='vgw-lite-score') as executor:
                pending: dict[Future[tuple[str, float]], FrameRecord] = {}
                queue = deque(missing_records)

                def submit_more() -> None:
                    while queue and len(pending) < max_workers:
                        record = queue.popleft()
                        future = executor.submit(_process_frame_pair, record.key, Path(record.first_path), Path(record.second_path), mode)
                        pending[future] = record
                    if active_keys_callback is not None:
                        active_keys_callback(tuple(record.key for record in pending.values()))

                submit_more()
                try:
                    while pending:
                        if cancel_check is not None and cancel_check():
                            for future in pending:
                                future.cancel()
                            raise BuildCancelledError('Build cancelled')
                        done, _not_done = wait(tuple(pending.keys()), timeout=MISMATCH_WAIT_TIMEOUT, return_when=FIRST_COMPLETED)
                        if not done:
                            continue
                        for future in done:
                            record = pending.pop(future)
                            result_key, score = future.result()
                            scores_by_key[result_key] = float(score)
                            cache_key = cache_keys_by_frame.get(result_key)
                            if score_cache is not None and cache_key is not None:
                                cache_entries_to_store.append((cache_key, float(score)))
                            processed += 1
                            if progress_callback is not None and (processed - last_reported >= progress_step or processed == total):
                                progress_callback(processed, total, record.key)
                                last_reported = processed
                        submit_more()
                finally:
                    for future in pending:
                        future.cancel()
                    if active_keys_callback is not None:
                        active_keys_callback(tuple())

        if score_cache is not None:
            score_cache.put_many(cache_entries_to_store)
    finally:
        if score_cache is not None:
            score_cache.close()

    absolute_scores = [float(scores_by_key[record.key]) for record in records]
    min_absolute = min(absolute_scores)
    max_absolute = max(absolute_scores)
    span = max(1e-12, max_absolute - min_absolute)
    best_record = min(records, key=lambda item: float(scores_by_key[item.key]))
    use_absolute_display = str(display_metric or 'relative').lower() == 'absolute'

    updated_records = []
    for record in records:
        absolute_score = float(scores_by_key[record.key])
        relative_score = 0.0 if max_absolute <= min_absolute else float((absolute_score - min_absolute) / span)
        display_score = absolute_score if use_absolute_display else relative_score
        updated_records.append(replace(record, score=display_score, absolute_score=absolute_score, relative_score=relative_score, score_ready=True))

    display_scores = [record.score for record in updated_records]
    return replace(
        build_result,
        records=tuple(updated_records),
        options=replace(build_result.options, comparison_mode=mode),
        min_score=min(display_scores) if display_scores else 0.0,
        max_score=max(display_scores) if display_scores else 0.0,
        scores_computed=True,
        best_match_key=best_record.key,
        min_absolute_score=min_absolute,
        max_absolute_score=max_absolute,
    )


def build_frame_records(
    first_folder: FolderSpec,
    second_folder: FolderSpec,
    options: BuildOptions,
    *,
    base_folder: FolderSpec | None = None,
    progress_callback: Callable[[int, int, str], None] | None = None,
    cancel_check: Callable[[], bool] | None = None,
) -> BuildResult:
    """Build a matrix and immediately compute mismatch values for it."""
    initial = collect_frame_records(first_folder, second_folder, options, base_folder=base_folder, cancel_check=cancel_check)
    return compute_build_result_mismatches(initial, comparison_mode=options.comparison_mode, display_metric='relative', progress_callback=progress_callback, cancel_check=cancel_check)


def load_frame_layers(record: FrameRecord) -> dict[str, object]:
    """Load grayscale and binary layers for one frame record."""
    first_signature = _image_signature(Path(record.first_path))
    first_gray = _load_grayscale_image_cached(*first_signature).copy()
    second_signature = _image_signature(Path(record.second_path))
    second_gray = _load_resized_grayscale_image_cached(*second_signature, tuple(int(value) for value in first_gray.shape)).copy()
    base_gray = None
    if record.base_path:
        base_signature = _image_signature(Path(record.base_path))
        base_gray = _load_resized_grayscale_image_cached(*base_signature, tuple(int(value) for value in first_gray.shape)).copy()
    return {
        'first_gray': first_gray,
        'second_gray': second_gray,
        'first_binary': first_gray >= 128,
        'second_binary': second_gray >= 128,
        'base_gray': base_gray,
        'shape': tuple(int(value) for value in first_gray.shape),
    }
