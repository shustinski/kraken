import pytest

pytest.importorskip("PIL")

from PIL import Image

from neuralimage.lib.data_interfaces import SampleGenerationSettings
from neuralimage.model.image_workers import CutImageThread
from tests.helpers import make_test_dir


class _Bus:
    def publish(self, _topic: str, _payload):
        return


def test_cut_image_thread_accepts_png_inputs(monkeypatch):
    root = make_test_dir("image_workers_png")
    source = root / "source"
    target = root / "target"
    source.mkdir(parents=True, exist_ok=True)

    Image.new("L", (8, 8), 0).save(source / "sample_a.png")
    Image.new("L", (8, 8), 0).save(source / "sample_b.jpg")
    (source / "skip.txt").write_text("x", encoding="utf-8")

    processed_files: list[str] = []

    def _fake_frame_cut(frame_path, *_args, **_kwargs):
        processed_files.append(frame_path.name)

    monkeypatch.setattr("model.image_workers.backend.frame_cut", _fake_frame_cut)

    settings = SampleGenerationSettings(
        step=1,
        segment_size=(4, 4),
        vertical_rotation=False,
        horizontal_rotation=False,
        channels=1,
    )
    worker = CutImageThread(source, target, settings, _Bus())
    worker.run()

    assert sorted(processed_files) == ["sample_a.png", "sample_b.jpg"]
