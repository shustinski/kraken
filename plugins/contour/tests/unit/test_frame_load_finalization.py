from types import SimpleNamespace

import pytest

from contour.adapters.qt.frame_load import FrameLoadRunnable

pytestmark = pytest.mark.fast


def test_deleted_completion_receiver_does_not_suppress_interruption() -> None:
    def load_source(_path: str) -> object:
        raise KeyboardInterrupt

    def emit_finished(*_args: object) -> None:
        raise RuntimeError("wrapped C++ object has been deleted")

    runnable = FrameLoadRunnable(
        1,
        "frame.png",
        load_source_image=load_source,
        load_cif_overlay=lambda _path: [],
        load_vectors=False,
        vectors_only=False,
    )
    runnable.signals = SimpleNamespace(finished=SimpleNamespace(emit=emit_finished))
    with pytest.raises(KeyboardInterrupt):
        runnable.run()
