"""Exact binary-mask connected components, with explicit adjacency and voids."""
from __future__ import annotations

from hashlib import sha256
from io import BytesIO
import json
from uuid import UUID, uuid5

from kraken_manager.domain.common import DomainValidationError
from kraken_manager.domain.electrical import ConductorComponent, ConductorExtraction, runs_digest


def _runs(indices, width):
    values = sorted(indices)
    output = []
    for value in values:
        y, x = divmod(int(value), width)
        if output and output[-1][0] == y and output[-1][2] == x:
            output[-1] = (y, output[-1][1], x + 1)
        else:
            output.append((y, x, x + 1))
    return tuple(output)


def extract_conductors(data: bytes, *, project_id, mask, calibration, connectivity, foreground_value,
                       nodata_value=None, max_pixels=16_777_216):
    """Decode an immutable mask; never threshold or interpret voids as conductors."""
    if sha256(data).hexdigest() != mask.sha256:
        raise DomainValidationError("binary mask bytes do not match artifact SHA-256")
    if isinstance(connectivity, bool) or connectivity not in {4, 8}:
        raise DomainValidationError("explicit 4/8 pixel adjacency is required")
    if isinstance(foreground_value, bool) or foreground_value not in {0, 255}:
        raise DomainValidationError("explicit foreground 0/255 is required")
    if nodata_value is not None and (isinstance(nodata_value, bool) or not isinstance(nodata_value, int)
                                    or nodata_value not in range(1, 255)):
        raise DomainValidationError("nodata marker must be distinct from binary values")
    try:
        from PIL import Image
    except ImportError as exc:
        raise DomainValidationError("Conductor mask decoding requires the optional Pillow runtime") from exc
    try:
        with Image.open(BytesIO(data)) as image:
            width, height = image.size
            if width * height > max_pixels or image.mode not in {"1", "L", "LA"}:
                raise DomainValidationError("mask exceeds pixel limit or is not a grayscale binary image")
            if image.mode == "LA":
                pixels = image.getchannel("L").tobytes()
                alpha = image.getchannel("A").tobytes()
            else:
                pixels, alpha = image.convert("L").tobytes(), None
    except (OSError, ValueError) as exc:
        raise DomainValidationError(f"Invalid binary-mask image: {exc}") from exc
    total = width * height
    foreground, unknown = bytearray(total), bytearray(total)
    for i, value in enumerate(pixels):
        if (alpha is not None and alpha[i] != 255) or (nodata_value is not None and value == nodata_value):
            unknown[i] = 1
        elif value not in {0, 255}:
            raise DomainValidationError("mask contains nonbinary samples without an explicit nodata marker")
        elif value == foreground_value:
            foreground[i] = 1
    try:
        import numpy as np
        import scipy
        from scipy import ndimage as ndi
    except ImportError as exc:
        raise DomainValidationError("Conductor extraction requires NumPy and SciPy; no approximate fallback is used") from exc
    binary = np.frombuffer(foreground, dtype=np.uint8).reshape(height, width).astype(bool)
    voids = np.frombuffer(unknown, dtype=np.uint8).reshape(height, width).astype(bool)
    structure = ndi.generate_binary_structure(2, 1 if connectivity == 4 else 2)
    labels, count = ndi.label(binary, structure=structure)
    void_neighbors = ndi.binary_dilation(voids, structure=structure)
    diagonal_counts = np.zeros(int(count) + 1, dtype=np.int64)
    if connectivity == 8:
        down_right = binary[:-1, :-1] & binary[1:, 1:] & ~binary[:-1, 1:] & ~binary[1:, :-1]
        down_left = binary[:-1, 1:] & binary[1:, :-1] & ~binary[:-1, :-1] & ~binary[1:, 1:]
        diagonal_counts += np.bincount(labels[:-1, :-1][down_right], minlength=int(count) + 1)
        diagonal_counts += np.bincount(labels[:-1, 1:][down_left], minlength=int(count) + 1)
    parameters = [calibration.revision, calibration.pixel_size_m, calibration.origin_m, calibration.y_axis,
                  connectivity, foreground_value, nodata_value, "explicit-binary-components-v1", scipy.__version__]
    extraction_id = str(uuid5(UUID(mask.version_id), json.dumps(parameters, sort_keys=True)))
    components = []
    for label, bounds in enumerate(ndi.find_objects(labels, max_label=int(count)), 1):
        if bounds is None:
            continue
        local_mask = labels[bounds] == label
        ys, xs = np.nonzero(local_mask)
        indices = ((ys + bounds[0].start) * width + xs + bounds[1].start).tolist()
        runs = _runs(indices, width)
        digest = runs_digest(runs)
        boundary = (bounds[0].start == 0 or bounds[0].stop == height
                    or bounds[1].start == 0 or bounds[1].stop == width)
        components.append(ConductorComponent(
            str(uuid5(UUID(extraction_id), digest)), runs, len(indices), digest,
            bool(np.any(void_neighbors[bounds][local_mask])), boundary, int(diagonal_counts[label]),
        ))
    return ConductorExtraction(
        id=extraction_id, project_id=str(project_id), layer_id=calibration.layer_id, mask=mask,
        calibration=calibration, width=width, height=height, connectivity=connectivity,
        foreground_value=foreground_value, components=tuple(components),
        nodata_runs=_runs(np.flatnonzero(voids), width), nodata_value=nodata_value,
        backend_version=scipy.__version__,
    )
