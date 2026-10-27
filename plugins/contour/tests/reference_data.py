"""Explicit location for original, unmodified microscopy regression samples."""
import os
from pathlib import Path


def reference_cif(filename):
    root = Path(os.environ.get("CONTOUR_REFERENCE_DATASET", r"D:\OZI\Нейронка"))
    return root / "cif_metal" / filename
