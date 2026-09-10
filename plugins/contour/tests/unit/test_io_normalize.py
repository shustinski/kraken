import pytest

from contour.vision import io_normalize

pytestmark = pytest.mark.fast


def test_fallback_loader_reports_unreadable_image(monkeypatch: pytest.MonkeyPatch) -> None:
    import cv2

    monkeypatch.setattr(io_normalize, "load_image_color", None)
    monkeypatch.setattr(cv2, "imread", lambda *_args: None)
    with pytest.raises(ValueError, match="Could not load image"):
        io_normalize.load_bgr("missing.png")
