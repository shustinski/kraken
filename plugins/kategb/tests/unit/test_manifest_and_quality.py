from __future__ import annotations

from pathlib import Path

from kategb.application.dto import GenerateManifestRequest
from kategb.application.use_cases import GenerateVerificationManifest, ReadVerificationManifest
from kategb.domain.models import IncorrectVector, LayerInfo
from kategb.domain.quality import calculate_author_results


def test_manifest_is_encrypted_and_readable(tmp_path: Path) -> None:
    manifest_path = GenerateVerificationManifest().execute(
        GenerateManifestRequest(
            vector_folder="D:/vectors",
            layer_name="M1",
            check_name="qa",
            frame_range_text="1-3",
            selection_mode="all",
            frames=(1, 2, 3),
            encryption_key="secret",
            output_folder=tmp_path,
        )
    )

    assert "D:/vectors" not in manifest_path.read_text(encoding="utf-8")
    manifest = ReadVerificationManifest().execute(manifest_path, "secret")
    assert manifest.layer_name == "M1"
    assert manifest.frames == (1, 2, 3)


def test_calculate_author_quality_results() -> None:
    layer = LayerInfo(
        name="M1",
        author_frames={"Alice": (1, 2), "Bob": (3, 4)},
        frames_in_layer=4,
        frames_in_row=2,
    )
    results = calculate_author_results(
        layer,
        original_frames=(1, 2, 3, 4),
        checked_vector_numbers=(101, 102, 103, 104),
        incorrect_vectors={
            101: IncorrectVector(101, True),
            102: IncorrectVector(102, False),
            103: IncorrectVector(103, False),
            104: IncorrectVector(104, True),
        },
    )

    assert results[0].author == "Alice"
    assert results[0].incorrect_frames == (2,)
    assert results[0].incorrect_percent == 50.0
    assert results[1].incorrect_frames == (3,)
