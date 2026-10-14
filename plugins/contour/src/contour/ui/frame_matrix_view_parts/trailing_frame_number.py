"""frame_matrix_view: trailing frame number."""
from __future__ import annotations
import re

def _trailing_frame_number(text: str) -> str:
    matches = re.findall(r"\d+", text)
    if not matches:
        return text
    return matches[-1]
