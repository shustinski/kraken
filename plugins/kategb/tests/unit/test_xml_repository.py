from __future__ import annotations

from pathlib import Path

from kategb.infrastructure.xml_repository import IncorrectXmlReader


def test_incorrect_xml_reader_extracts_frame_number_from_legacy_name(tmp_path: Path) -> None:
    path = tmp_path / "result.xml"
    path.write_text(
        '<Root><Vector Name="folder/frame_0012_result.cif" Correct="0" />'
        '<Vector Name="folder/frame_0013_result.cif" Correct="1" /></Root>',
        encoding="utf-8",
    )

    result = IncorrectXmlReader().read(path)

    assert result[12].is_correct is False
    assert result[13].is_correct is True
