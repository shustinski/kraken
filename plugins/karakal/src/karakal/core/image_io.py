"""Image discovery, decoding, resizing, and runtime image caches."""

from __future__ import annotations

from .mask_primitives import (
    _boundary_mask,
    _label_components,
)

from .repository_shared import (
    BuildCancelledError,
    ByteLruCache,
    EPS,
    FrameIdentity,
    IMAGE_CACHE_SIZE,
    NATURAL_SPLIT_PATTERN,
    Path,
    PerformanceConfig,
    QImage,
    Qt,
    _OriginalFrameFeatures,
    _PredictionPoint,
    _PredictionRegionSummary,
    _PredictionView,
    _LOGGER,
    _active_performance_config,
    current_profiler,
    cv2,
    lru_cache,
    np,
)

from .image_formats import SUPPORTED_IMAGE_EXTENSIONS, SUPPORTED_IMAGE_EXTENSION_SET


def _resize_like(array: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    if tuple(int(v) for v in array.shape) == tuple(int(v) for v in target_shape):
        return np.asarray(array)
    return np.asarray(resize_grayscale_image(np.asarray(array, dtype=np.uint8), target_shape))


def extract_frame_id(value: str) -> int:
    stem = Path(str(value)).stem
    token = stem.rsplit("_", 1)[-1]
    return int(token) if token.isdigit() else 0


def build_frame_identity(key: str, fallback_frame_id: int) -> FrameIdentity:
    path = Path(key)
    sequence_id = path.parent.as_posix() if path.parent.as_posix() not in {"", "."} else None
    try:
        frame_id = extract_frame_id(path.name)
    except Exception:
        frame_id = int(fallback_frame_id)
    return FrameIdentity(
        frame_id=frame_id,
        base_id=frame_id,
        tile_x=None,
        tile_y=None,
        source_key=key,
        sequence_id=sequence_id,
    )


_FolderPathLookup = tuple[dict[str, Path], dict[str, Path], dict[str, Path]]


def _unique_path_lookup(index: dict[str, Path], key_fn) -> dict[str, Path]:
    lookup: dict[str, Path] = {}
    duplicates: set[str] = set()
    for path in index.values():
        key = str(key_fn(path)).lower()
        if key in duplicates:
            continue
        if key in lookup:
            lookup.pop(key, None)
            duplicates.add(key)
            continue
        lookup[key] = path
    return lookup


def _build_folder_path_lookup(index: dict[str, Path]) -> _FolderPathLookup:
    return (
        index,
        _unique_path_lookup(index, lambda path: Path(path).name),
        _unique_path_lookup(index, lambda path: Path(path).stem),
    )


def _resolve_aux_path_from_lookup(key: str, lookup: _FolderPathLookup) -> Path | None:
    index, name_lookup, stem_lookup = lookup
    exact = index.get(key)
    if exact is not None:
        return exact
    name = Path(key).name.lower()
    stem = Path(key).stem.lower()
    name_match = name_lookup.get(name)
    if name_match is not None:
        return name_match
    stem_match = stem_lookup.get(stem)
    if stem_match is not None:
        return stem_match
    return None


def _resolve_aux_path(key: str, index: dict[str, Path]) -> Path | None:
    return _resolve_aux_path_from_lookup(key, _build_folder_path_lookup(index))


@lru_cache(maxsize=65536)
def natural_sort_key(value: str) -> tuple[object, ...]:
    """Split a string into digit and text chunks for natural sorting."""

    parts = NATURAL_SPLIT_PATTERN.split(str(value).lower())
    key: list[object] = []
    for part in parts:
        if not part:
            continue
        key.append(int(part) if part.isdigit() else part)
    return tuple(key)


def iter_image_paths(folder: Path, *, recursive: bool, extensions: tuple[str, ...]) -> list[Path]:
    """Return image paths from one folder using natural sorting."""

    normalized_extensions = {str(ext).lower() for ext in extensions}
    iterator = folder.rglob("*") if recursive else folder.glob("*")
    paths = [path for path in iterator if path.is_file() and path.suffix.lower() in normalized_extensions]
    return sorted(paths, key=lambda item: natural_sort_key(item.as_posix()))


def build_folder_index(
    folder: Path,
    *,
    recursive: bool,
    extensions: tuple[str, ...],
    cancel_check=None,
    progress_callback=None,
    progress_interval: int = 2048,
) -> dict[str, Path]:
    """Index frame files in one folder by relative path."""

    folder = Path(folder)
    normalized_extensions = {str(ext).lower() for ext in extensions}
    iterator = folder.rglob("*") if recursive else folder.glob("*")
    paths: list[Path] = []
    accepted = 0
    interval = max(1, int(progress_interval))
    for path in iterator:
        if cancel_check is not None and cancel_check():
            raise BuildCancelledError("Build cancelled")
        if not path.is_file() or path.suffix.lower() not in normalized_extensions:
            continue
        paths.append(path)
        accepted += 1
        if progress_callback is not None and accepted % interval == 0:
            progress_callback(accepted, 0, path.name)
    paths.sort(key=lambda item: natural_sort_key(item.as_posix()))
    index: dict[str, Path] = {}
    for image_path in paths:
        if cancel_check is not None and cancel_check():
            raise BuildCancelledError("Build cancelled")
        index[image_path.relative_to(folder).as_posix()] = image_path
    if progress_callback is not None:
        progress_callback(len(index), len(index), Path(folder).name)
    return index


def _qimage_to_grayscale_array(image: QImage) -> np.ndarray:
    """Convert Qt image to contiguous grayscale ndarray."""

    grayscale = image.convertToFormat(QImage.Format.Format_Grayscale8)
    pointer = grayscale.bits()
    pointer.setsize(grayscale.height() * grayscale.bytesPerLine())
    buffer = np.frombuffer(pointer, dtype=np.uint8).reshape((grayscale.height(), grayscale.bytesPerLine()))
    return buffer[:, : grayscale.width()].copy()


def _grayscale_array_to_qimage(array: np.ndarray) -> QImage:
    """Convert contiguous grayscale ndarray to Qt image."""

    contiguous = np.ascontiguousarray(array.astype(np.uint8))
    height, width = contiguous.shape
    image = QImage(contiguous.data, width, height, contiguous.strides[0], QImage.Format.Format_Grayscale8)
    return image.copy()


def _rgb_array_to_qimage(array: np.ndarray) -> QImage:
    """Convert contiguous RGB ndarray to Qt image."""

    contiguous = np.ascontiguousarray(np.asarray(array, dtype=np.uint8))
    if contiguous.ndim != 3 or contiguous.shape[2] != 3:
        return QImage()
    height, width, _channels = contiguous.shape
    image = QImage(contiguous.data, width, height, int(contiguous.strides[0]), QImage.Format.Format_RGB888)
    return image.copy()


def _load_grayscale_image_raw(path: Path) -> np.ndarray:
    image = QImage(str(path))
    if image.isNull():
        raise ValueError(f"Unable to decode image: {path}")
    return _qimage_to_grayscale_array(image)


def load_grayscale_image(path: Path) -> np.ndarray:
    """Load one image as grayscale ndarray."""

    return _load_grayscale_image_raw(Path(path))


def _load_export_grayscale_image(path: Path | str, target_shape: tuple[int, int] | None = None) -> np.ndarray:
    """Load grayscale image for bulk export without populating analytics caches."""

    source_path = Path(path)
    image = None
    if cv2 is not None:
        try:
            encoded = np.fromfile(str(source_path), dtype=np.uint8)
            if encoded.size > 0:
                image = cv2.imdecode(encoded, cv2.IMREAD_GRAYSCALE)
        except (OSError, ValueError, cv2.error) as error:
            _LOGGER.debug("OpenCV could not decode export source %s; using Qt fallback: %s", source_path, error)
            image = None
    if image is None:
        image = load_grayscale_image(source_path)
    result = np.asarray(image, dtype=np.uint8)
    if target_shape is not None and tuple(int(v) for v in result.shape) != tuple(int(v) for v in target_shape):
        target_height, target_width = int(target_shape[0]), int(target_shape[1])
        if cv2 is not None:
            result = cv2.resize(result, (target_width, target_height), interpolation=cv2.INTER_NEAREST)
        else:
            result = resize_grayscale_image(result, (target_height, target_width))
    return np.asarray(result, dtype=np.uint8)


def _save_export_rgb_jpg(path: Path | str, rgb: np.ndarray, quality: int) -> bool:
    """Save RGB image as JPG using OpenCV when available."""

    target_path = Path(path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    rgb_uint8 = np.ascontiguousarray(np.asarray(rgb, dtype=np.uint8))
    if cv2 is not None:
        try:
            bgr = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2BGR)
            ok, encoded = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
            if ok:
                encoded.tofile(str(target_path))
                return True
        except (OSError, ValueError, cv2.error) as error:
            _LOGGER.warning("OpenCV could not encode %s; using Qt fallback: %s", target_path, error)
    return bool(_rgb_array_to_qimage(rgb_uint8).save(str(target_path), "JPG", int(quality)))


def resize_grayscale_image(array: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    """Resize grayscale image using Qt fast transformation."""

    target_height, target_width = int(target_shape[0]), int(target_shape[1])
    source = np.asarray(array, dtype=np.uint8)
    if source.shape == (target_height, target_width):
        return source.copy()
    image = _grayscale_array_to_qimage(source)
    scaled = image.scaled(
        target_width,
        target_height,
        Qt.AspectRatioMode.IgnoreAspectRatio,
        Qt.TransformationMode.FastTransformation,
    )
    return _qimage_to_grayscale_array(scaled)


def _mean_filter_3x3(image: np.ndarray) -> np.ndarray:
    """Compute 3x3 mean filter without SciPy dependency."""

    image_f = np.asarray(image, dtype=np.float32)
    padded = np.pad(image_f, 1, mode="edge")
    return (
        padded[:-2, :-2]
        + padded[:-2, 1:-1]
        + padded[:-2, 2:]
        + padded[1:-1, :-2]
        + padded[1:-1, 1:-1]
        + padded[1:-1, 2:]
        + padded[2:, :-2]
        + padded[2:, 1:-1]
        + padded[2:, 2:]
    ) / 9.0


def extract_original_frame_features(image: np.ndarray | None) -> _OriginalFrameFeatures | None:
    """Extract scalar quality features from original grayscale frame."""

    if image is None:
        return None
    grayscale = np.asarray(image, dtype=np.uint8)
    image_f = grayscale.astype(np.float32)
    normalized = image_f / 255.0
    mean_brightness = float(normalized.mean())
    contrast = float(normalized.std())

    histogram = np.bincount(grayscale.ravel(), minlength=256).astype(np.float64)
    histogram /= max(1.0, histogram.sum())
    non_zero = histogram > 0.0
    entropy = float(-(histogram[non_zero] * np.log2(histogram[non_zero])).sum())

    padded = np.pad(image_f, 1, mode="edge")
    center = padded[1:-1, 1:-1]
    laplacian = -4.0 * center + padded[1:-1, :-2] + padded[1:-1, 2:] + padded[:-2, 1:-1] + padded[2:, 1:-1]
    blur_score = float(laplacian.var() / (255.0 * 255.0))

    smoothed = _mean_filter_3x3(grayscale)
    noise_score = float(np.mean(np.abs(image_f - smoothed)) / 255.0)

    gradient_x = padded[1:-1, 2:] - padded[1:-1, :-2]
    gradient_y = padded[2:, 1:-1] - padded[:-2, 1:-1]
    gradient_magnitude = np.hypot(gradient_x, gradient_y)
    edge_density = float(np.mean(gradient_magnitude >= 20.0))

    neighbors = [
        padded[:-2, :-2],
        padded[:-2, 1:-1],
        padded[:-2, 2:],
        padded[1:-1, :-2],
        padded[1:-1, 2:],
        padded[2:, :-2],
        padded[2:, 1:-1],
        padded[2:, 2:],
    ]
    strict_peaks = np.logical_and.reduce([center > neighbor for neighbor in neighbors])
    local_peak_density = float(np.mean(strict_peaks))

    dynamic_range = float((float(grayscale.max()) - float(grayscale.min())) / 255.0)
    saturation_ratio = float(np.mean((grayscale <= 5) | (grayscale >= 250)))

    return _OriginalFrameFeatures(
        mean_brightness=mean_brightness,
        contrast=contrast,
        entropy=entropy,
        blur_score=blur_score,
        noise_score=noise_score,
        edge_density=edge_density,
        local_peak_density=local_peak_density,
        dynamic_range=dynamic_range,
        saturation_ratio=saturation_ratio,
    )


def _extract_patch(image: np.ndarray, center_x: float, center_y: float, radius: int = 2) -> np.ndarray:
    height, width = image.shape
    x = int(round(center_x))
    y = int(round(center_y))
    x0 = max(0, x - radius)
    x1 = min(width, x + radius + 1)
    y0 = max(0, y - radius)
    y1 = min(height, y + radius + 1)
    return image[y0:y1, x0:x1]


def build_prediction_view_from_gray(model_name: str, pred_gray: np.ndarray, *, threshold: int = 128) -> _PredictionView:
    """Build lightweight prediction view from one grayscale output."""

    grayscale = np.asarray(pred_gray, dtype=np.uint8)
    pred_bin = grayscale >= int(threshold)
    labels, count = _label_components(pred_bin)
    points: list[_PredictionPoint] = []
    areas: list[int] = []
    for label in range(1, int(count) + 1):
        ys, xs = np.where(labels == label)
        if xs.size == 0:
            continue
        area = int(xs.size)
        areas.append(area)
        centroid_x = float(np.mean(xs, dtype=np.float64))
        centroid_y = float(np.mean(ys, dtype=np.float64))
        px = int(np.clip(round(centroid_x), 0, grayscale.shape[1] - 1))
        py = int(np.clip(round(centroid_y), 0, grayscale.shape[0] - 1))
        patch = _extract_patch(grayscale, centroid_x, centroid_y)
        peak = float(grayscale[py, px] / 255.0)
        patch_mean = float(patch.mean(dtype=np.float32) / 255.0) if patch.size else peak
        patch_std = float(patch.std(dtype=np.float32) / 255.0) if patch.size else 0.0
        local_contrast = float(max(0.0, peak - patch_mean))
        local_snr = float(local_contrast / max(1e-6, patch_std))
        radius = float(np.sqrt(area / np.pi))
        boundary = _boundary_mask(labels == label)
        perimeter = float(max(1, np.count_nonzero(boundary)))
        compactness = float(np.clip((4.0 * np.pi * area) / max(EPS, perimeter * perimeter), 0.0, 1.0))
        points.append(
            _PredictionPoint(
                x=centroid_x,
                y=centroid_y,
                score=peak,
                peak_intensity=peak * 255.0,
                local_contrast=local_contrast,
                blob_score=compactness,
                local_snr=local_snr,
                radius=radius,
                spot_area=float(area),
            )
        )
    region_summary = _PredictionRegionSummary(
        area_fraction=float(np.count_nonzero(pred_bin) / max(1, pred_bin.size)),
        mean_area=float(np.mean(np.asarray(areas, dtype=np.float64))) if areas else 0.0,
    )
    return _PredictionView(
        model_name=str(model_name),
        pred_gray=grayscale,
        pred_bin=np.asarray(pred_bin, dtype=bool),
        points=tuple(points),
        region_summary=region_summary,
    )


def _candidate_paths_for_known_key(folder: Path, key: str, extensions: tuple[str, ...]) -> list[Path]:
    relative = Path(key)
    candidates: list[Path] = []
    direct = folder / relative
    candidates.append(direct)
    stem_name = relative.stem
    parent = folder / relative.parent
    seen: set[str] = {str(direct).lower()}
    for extension in extensions:
        normalized = str(extension or "").strip()
        if not normalized:
            continue
        if not normalized.startswith("."):
            normalized = f".{normalized}"
        candidate = parent / f"{stem_name}{normalized}"
        marker = str(candidate).lower()
        if marker in seen:
            continue
        seen.add(marker)
        candidates.append(candidate)
    return candidates


def _resolve_model_path_from_known_key(folder: Path, key: str, extensions: tuple[str, ...]) -> Path | None:
    for candidate in _candidate_paths_for_known_key(folder, key, extensions):
        if candidate.is_file():
            return candidate
    return None


def _resolve_model_path_for_key(
    folder: Path,
    key: str,
    extensions: tuple[str, ...],
    fallback_index_cache: dict[str, _FolderPathLookup],
    *,
    recursive: bool,
    cancel_check=None,
    progress_callback=None,
) -> Path | None:
    cache_key = str(folder.resolve())
    lookup = fallback_index_cache.get(cache_key)
    if lookup is not None:
        return _resolve_aux_path_from_lookup(key, lookup)
    resolved = _resolve_model_path_from_known_key(folder, key, extensions)
    if resolved is not None:
        return resolved
    if cancel_check is not None and cancel_check():
        raise BuildCancelledError("Build cancelled")
    index = build_folder_index(
        folder,
        recursive=recursive,
        extensions=extensions,
        cancel_check=cancel_check,
        progress_callback=progress_callback,
    )
    lookup = _build_folder_path_lookup(index)
    fallback_index_cache[cache_key] = lookup
    return _resolve_aux_path_from_lookup(key, lookup)


def _fit_shape_to_max_side(shape: tuple[int, int], max_side: int | None) -> tuple[int, int]:
    height, width = (int(shape[0]), int(shape[1]))
    limit = int(max_side or 0)
    if limit <= 0 or max(height, width) <= limit:
        return height, width
    scale = float(limit) / float(max(height, width))
    return max(1, int(round(height * scale))), max(1, int(round(width * scale)))


def _image_signature(path: Path) -> tuple[str, int, int]:
    resolved = Path(path).resolve()
    stat = resolved.stat()
    return str(resolved), int(stat.st_mtime_ns), int(stat.st_size)


_DEFAULT_IMAGE_CACHE_BYTES = int(PerformanceConfig().ram_cache_limit_mb) * 1024 * 1024 // 2

_GRAYSCALE_IMAGE_CACHE: ByteLruCache[tuple[str, int, int], np.ndarray] = ByteLruCache(
    _DEFAULT_IMAGE_CACHE_BYTES,
    max_items=IMAGE_CACHE_SIZE,
)

_RESIZED_IMAGE_CACHE: ByteLruCache[tuple[str, int, int, tuple[int, int]], np.ndarray] = ByteLruCache(
    _DEFAULT_IMAGE_CACHE_BYTES,
    max_items=IMAGE_CACHE_SIZE,
)


def _load_grayscale_image_cached(path_text: str, _mtime_ns: int, _size: int) -> np.ndarray:
    key = (path_text, int(_mtime_ns), int(_size))
    cached = _GRAYSCALE_IMAGE_CACHE.get(key)
    profiler = current_profiler()
    if cached is not None:
        if profiler is not None:
            profiler.increment("cache.ram.hits")
        return cached
    if profiler is not None:
        profiler.increment("cache.ram.misses")
    loaded = load_grayscale_image(Path(path_text))
    _GRAYSCALE_IMAGE_CACHE.put(key, loaded)
    return loaded


def _load_resized_grayscale_image_cached(
    path_text: str, _mtime_ns: int, _size: int, target_shape: tuple[int, int]
) -> np.ndarray:
    key = (path_text, int(_mtime_ns), int(_size), tuple(int(value) for value in target_shape))
    cached = _RESIZED_IMAGE_CACHE.get(key)
    profiler = current_profiler()
    if cached is not None:
        if profiler is not None:
            profiler.increment("cache.ram.hits")
        return cached
    if profiler is not None:
        profiler.increment("cache.ram.misses")
    source = load_grayscale_image(Path(path_text))
    resized = resize_grayscale_image(source, target_shape)
    _RESIZED_IMAGE_CACHE.put(key, resized)
    return resized


def _clear_runtime_image_caches() -> None:
    """Release in-memory grayscale caches used during batch analytics."""

    _GRAYSCALE_IMAGE_CACHE.clear()
    _RESIZED_IMAGE_CACHE.clear()


def _load_optional_gray(
    path_text: str | None, target_shape: tuple[int, int] | None = None, max_side: int | None = None
) -> np.ndarray | None:
    if not path_text:
        return None
    per_cache_limit = max(1, int(_active_performance_config().ram_cache_limit_mb) * 1024 * 1024 // 2)
    _GRAYSCALE_IMAGE_CACHE.set_max_bytes(per_cache_limit)
    _RESIZED_IMAGE_CACHE.set_max_bytes(per_cache_limit)
    source_path = Path(path_text)
    source = _load_grayscale_image_cached(*_image_signature(source_path))
    limited_shape = _fit_shape_to_max_side(tuple(int(v) for v in source.shape), max_side)
    final_shape = tuple(int(v) for v in target_shape) if target_shape is not None else limited_shape
    if tuple(int(v) for v in source.shape) == final_shape:
        return np.asarray(source, dtype=np.uint8)
    return np.asarray(_load_resized_grayscale_image_cached(*_image_signature(source_path), final_shape), dtype=np.uint8)


def _path_signature(path_text: str | None) -> tuple[str, int, int] | None:
    if not path_text:
        return None
    return _image_signature(Path(path_text))


__all__ = ["SUPPORTED_IMAGE_EXTENSIONS", "SUPPORTED_IMAGE_EXTENSION_SET"]
