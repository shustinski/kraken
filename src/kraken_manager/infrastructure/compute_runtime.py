"""Initialize bundled numerical DLLs before starting application worker threads."""

from __future__ import annotations

import threading


def initialize_packaged_compute_runtime():
    # On the audited Windows/Python3.14 frozen build, the first NumPy extension
    # import from an AnyIO worker hangs in create_module. Main-thread import is
    # verified by the same packaged HTTP extraction acceptance, not only a unit
    # test. Keep optional source deployments lazy; frozen distributions ship it.
    if threading.current_thread() is not threading.main_thread():
        raise RuntimeError("Packaged numerical runtime must initialize on the main thread")
    import numpy
    import scipy
    from scipy import ndimage

    # Resolve the native algorithm as well as package metadata before workers.
    labels, count = ndimage.label(
        numpy.array([[1, 0], [0, 1]], dtype=numpy.uint8), structure=ndimage.generate_binary_structure(2, 1)
    )
    if count != 2 or labels.shape != (2, 2):
        raise RuntimeError("Bundled numerical runtime initialization failed")
    return {"numpy": numpy.__version__, "scipy": scipy.__version__}
