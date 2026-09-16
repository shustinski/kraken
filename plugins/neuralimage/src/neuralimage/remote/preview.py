"""Bounded preview protocol using lossless NumPy arrays without pickle."""

from __future__ import annotations
import base64
import io
import math
from pathlib import Path
import numpy as np
from .codec import decode

MAX_PREVIEW_BYTES = 64 * 1024 * 1024


def encode_array(array):
    stream = io.BytesIO()
    np.save(stream, array, allow_pickle=False)
    return base64.b64encode(stream.getvalue()).decode("ascii")


def decode_array(value):
    raw = base64.b64decode(value, validate=True)
    if len(raw) > MAX_PREVIEW_BYTES:
        raise ValueError("Preview image is too large")
    stream = io.BytesIO(raw)
    version = np.lib.format.read_magic(stream)
    if version == (1, 0):
        shape, _, dtype = np.lib.format.read_array_header_1_0(stream)
    elif version == (2, 0):
        shape, _, dtype = np.lib.format.read_array_header_2_0(stream)
    else:
        raise ValueError("Unsupported array format")
    if dtype.hasobject or len(shape) not in {2, 3} or math.prod(shape) * dtype.itemsize > MAX_PREVIEW_BYTES:
        raise ValueError("Invalid preview dimensions or dtype")
    stream.seek(0)
    array = np.load(stream, allow_pickle=False)
    if array.dtype.kind not in "bufi" or array.ndim not in {2, 3} or array.nbytes > MAX_PREVIEW_BYTES:
        raise ValueError("Invalid preview array")
    return array


def compute_preview(payload):
    from .preview_engine import PreviewEngine, _seeded_random

    training = decode(payload["training"])
    # Only the supplied in-memory samples can be read by this engine.
    arrays = {int(k): tuple(decode_array(v) for v in pair) for k, pair in payload["arrays"].items()}
    pairs = [(Path(name), Path(name)) for name in payload["names"]]
    engine = PreviewEngine(
        training,
        payload["options"],
        pairs,
        arrays,
        payload["index"],
        payload["variant"],
        payload["salt"],
        payload["cutter_item"],
    )
    with _seeded_random(engine._seed_for(payload["index"], "preview")):
        return [encode_array(array) for array in engine._build_preview_arrays(payload["index"])]
