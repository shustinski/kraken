from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from pathlib import Path

from kategb.domain.models import IncorrectVector

_FRAME_NUMBER_RE = re.compile(r"[/_.]+")


class IncorrectXmlReader:
    def read(self, path: Path) -> dict[int, IncorrectVector]:
        root = ET.parse(path).getroot()
        result: dict[int, IncorrectVector] = {}
        for node in root:
            name = node.attrib.get("Name", "")
            numeric_parts = [part for part in _FRAME_NUMBER_RE.split(name) if part.isdigit()]
            if not numeric_parts:
                continue
            number = int(numeric_parts[-1])
            correct_value = node.attrib.get("Correct", "1")
            result[number] = IncorrectVector(
                number=number,
                is_correct=correct_value not in {"0", "false", "False"},
                attributes=dict(node.attrib),
            )
        return result
