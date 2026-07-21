import pytest

pytest.importorskip("PIL")

from PIL import Image

from neuralimage.lib.data_interfaces import SampleGenerationSettings
from neuralimage.model.image_workers import ConvertCifThread, CutImageThread
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

    monkeypatch.setattr("neuralimage.model.image_workers.backend.frame_cut", _fake_frame_cut)

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


def test_convert_cif_thread_writes_true_one_bit_png():
    root = make_test_dir('image_workers_binary_cif_png')
    source = root / 'source'
    target = root / 'binary_cif'
    source.mkdir(parents=True, exist_ok=True)
    target.mkdir(parents=True, exist_ok=True)
    (source / 'frame.cif').write_text('0 0 S 8 8\nB 4 4 4 4;;\n', encoding='utf-8')
    Image.new('L', (8, 8), 0).save(target / 'frame.jpg')

    worker = ConvertCifThread(source, target, _Bus())
    worker.run()

    output_path = target / 'frame.png'
    assert output_path.exists()
    assert not (target / 'frame.jpg').exists()
    with Image.open(output_path) as binary_mask:
        assert binary_mask.mode == '1'
        assert binary_mask.getextrema() == (0, 255)
    assert output_path.read_bytes()[24] == 1
