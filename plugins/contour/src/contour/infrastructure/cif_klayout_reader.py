"""CIF syntax reader ported from KLayout ``dbCIFReader.cc``.

Like KLayout, this layer reads every ``P`` command as one polygon hull. Conversion
from a standard self-touching cutline hull to Contour's editable outer/hole model is
performed by the serializer after parsing.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path

from .cif_primitives import CifBox, CifComment, CifPolygon, CifPrimitive

_BLANK_STOP = frozenset("-();")
_SEP_STOP = frozenset("-();")
_NAME_CHARS = frozenset("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_")


class CifParseError(ValueError):
    """Raised when a CIF file cannot be parsed."""


@dataclass(slots=True)
class CifKLayoutLoadResult:
    primitives: tuple[CifPrimitive, ...]
    warnings: tuple[str, ...]


def klayout_cif_reader_enabled() -> bool:
    return str(os.environ.get("CONTOUR_CIF_USE_KLAYOUT", "1")).strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
    }


def load_cif_primitives_klayout(path: str | Path) -> CifKLayoutLoadResult:
    text = _read_cif_text(path)
    stream = _CifTextStream(text)
    reader = _KLayoutCifReader(stream)
    return CifKLayoutLoadResult(
        primitives=tuple(reader.read_top_cell()),
        warnings=tuple(reader.warnings),
    )


def _read_cif_text(path: str | Path) -> str:
    cif_path = Path(path)
    payload = cif_path.read_bytes()
    for encoding in ("utf-8", "utf-8-sig", "cp1251", "cp866"):
        try:
            return payload.decode(encoding)
        except UnicodeDecodeError:
            continue
    return payload.decode("cp1251", errors="replace")


class _CifTextStream:
    """Index-based CIF text cursor; avoids per-character method dispatch in hot paths."""

    __slots__ = ("_text", "_pos", "_length", "line_number")

    def __init__(self, text: str) -> None:
        self._text = text
        self._pos = 0
        self._length = len(text)
        self.line_number = 1

    def at_end(self) -> bool:
        return self._pos >= self._length

    def peek_char(self) -> str:
        pos = self._pos
        if pos >= self._length:
            return ""
        return self._text[pos]

    def get_char(self) -> str:
        pos = self._pos
        if pos >= self._length:
            raise CifParseError("Unexpected end of file")
        char = self._text[pos]
        self._pos = pos + 1
        if char == "\n":
            self.line_number += 1
        return char

    def skip_blanks(self) -> None:
        """Match KLayout ``CIFReader::skip_blanks`` — stop before commands and tokens."""

        text = self._text
        pos = self._pos
        length = self._length
        line_number = self.line_number
        while pos < length:
            char = text[pos]
            if char.isupper() or char.isdigit() or char in _BLANK_STOP:
                break
            if char == "\n":
                line_number += 1
            pos += 1
        self._pos = pos
        self.line_number = line_number

    def skip_sep(self) -> None:
        """Match KLayout ``CIFReader::skip_sep``."""

        text = self._text
        pos = self._pos
        length = self._length
        while pos < length:
            char = text[pos]
            if char.isdigit() or char in _SEP_STOP:
                break
            pos += 1
        self._pos = pos

    def read_integer_digits(self) -> int:
        text = self._text
        pos = self._pos
        length = self._length
        if pos >= length or not text[pos].isdigit():
            raise CifParseError("Digit expected")
        value = 0
        while pos < length:
            char = text[pos]
            if not char.isdigit():
                break
            value = (value * 10) + ord(char) - 48
            pos += 1
        self._pos = pos
        return value

    def read_integer(self) -> int:
        self.skip_sep()
        return self.read_integer_digits()

    def read_sinteger(self) -> int:
        self.skip_sep()
        text = self._text
        pos = self._pos
        negative = False
        if pos < self._length and text[pos] == "-":
            negative = True
            pos += 1
        self._pos = pos
        value = self.read_integer_digits()
        return -value if negative else value

    def read_name(self) -> str:
        self.skip_blanks()
        text = self._text
        pos = self._pos
        start = pos
        length = self._length
        while pos < length and text[pos] in _NAME_CHARS:
            pos += 1
        self._pos = pos
        return text[start:pos]

    def read_string(self) -> str:
        self.skip_sep()
        text = self._text
        pos = self._pos
        length = self._length
        if pos >= length:
            return ""
        quote = text[pos]
        if quote in {'"', "'"}:
            pos += 1
            chars: list[str] = []
            while pos < length:
                char = text[pos]
                if char == quote:
                    self._pos = pos + 1
                    return "".join(chars)
                if char == "\\" and pos + 1 < length:
                    pos += 1
                    chars.append(text[pos])
                    pos += 1
                    continue
                chars.append(char)
                pos += 1
            self._pos = pos
            return "".join(chars)
        start = pos
        while pos < length:
            char = text[pos]
            if char.isspace() or char == ";":
                break
            pos += 1
        self._pos = pos
        return text[start:pos]

    def test_semi(self) -> bool:
        self.skip_blanks()
        return self._pos < self._length and self._text[self._pos] == ";"

    def expect_semi(self) -> None:
        if not self.test_semi():
            raise CifParseError("Expected ';' command terminator")
        self.get_char()

    def skip_to_semicolon(self) -> None:
        text = self._text
        pos = self._pos
        length = self._length
        line_number = self.line_number
        while pos < length:
            char = text[pos]
            if char == "\n":
                line_number += 1
            pos += 1
            if char == ";":
                break
        self._pos = pos
        self.line_number = line_number

    def read_comment_from_open_paren(self) -> str:
        """Read a ``(...)`` comment body; opening ``(`` was already consumed."""

        text = self._text
        start = self._pos - 1
        pos = self._pos
        length = self._length
        nesting = 1
        line_number = self.line_number
        while pos < length and nesting > 0:
            char = text[pos]
            if char == "\n":
                line_number += 1
            if char == "(":
                nesting += 1
            elif char == ")":
                nesting -= 1
            pos += 1
        while pos < length and text[pos] != ";":
            char = text[pos]
            if char == "\n":
                line_number += 1
            pos += 1
        self._pos = pos
        self.line_number = line_number
        return text[start:pos]


class _KLayoutCifReader:
    def __init__(self, stream: _CifTextStream) -> None:
        self._stream = stream
        self.warnings: list[str] = []
        self._primitives: list[CifPrimitive] = []

    def read_top_cell(self) -> list[CifPrimitive]:
        self._read_cell(level=0, scale_factor=1.0)
        self._stream.skip_blanks()
        if not self._stream.at_end():
            self._warn("E command is followed by more text")
        return self._primitives

    def _read_cell(self, *, level: int, scale_factor: float) -> None:
        del scale_factor  # retained for KLayout parity
        layer_selected = False

        while True:
            self._stream.skip_blanks()
            if self._stream.at_end():
                raise CifParseError("Unexpected end of file")

            command_char = self._stream.get_char()
            if command_char == ";":
                continue
            if command_char == "(":
                comment = self._stream.read_comment_from_open_paren()
                self._primitives.append(CifComment(content=comment))
                continue
            if command_char == "E":
                if level > 0:
                    raise CifParseError("'E' command must be outside a cell specification")
                return
            if command_char == "D":
                self._stream.skip_blanks()
                sub = self._stream.get_char()
                if sub == "S":
                    self._stream.read_integer()
                    if not self._stream.test_semi():
                        self._stream.read_integer()
                        divider = self._stream.read_integer()
                        if divider == 0:
                            raise CifParseError("'DS' command: divider cannot be zero")
                    self._stream.expect_semi()
                    self._read_cell(level=level + 1, scale_factor=1.0)
                elif sub == "F":
                    if level == 0:
                        raise CifParseError("'DF' command must be inside a cell specification")
                    self._stream.skip_to_semicolon()
                    return
                elif sub == "D":
                    self._stream.read_integer()
                    self._warn("'DD' command ignored")
                    self._stream.skip_to_semicolon()
                else:
                    raise CifParseError("Invalid 'D' sub-command")
                continue
            if command_char == "L":
                name = self._stream.read_name()
                if not name:
                    raise CifParseError("Missing layer name in 'L' command")
                layer_selected = True
                self._stream.expect_semi()
                continue
            if command_char == "B":
                if not layer_selected:
                    self._warn("'B' command ignored since no layer was selected")
                    self._stream.skip_to_semicolon()
                    continue
                width = self._stream.read_integer()
                height = self._stream.read_integer()
                center_x = self._stream.read_sinteger()
                center_y = self._stream.read_sinteger()
                rotation_x = 1
                rotation_y = 0
                if not self._stream.test_semi():
                    rotation_x = self._stream.read_sinteger()
                    rotation_y = self._stream.read_sinteger()
                self._primitives.append(
                    CifBox(
                        width=width,
                        height=height,
                        center_x=center_x,
                        center_y=center_y,
                        rotation_x=rotation_x,
                        rotation_y=rotation_y,
                    )
                )
                self._stream.expect_semi()
                continue
            if command_char == "P":
                if not layer_selected:
                    self._warn("'P' command ignored since no layer was selected")
                    self._stream.skip_to_semicolon()
                    continue
                points: list[tuple[int, int]] = []
                while not self._stream.test_semi():
                    x_coord = self._stream.read_sinteger()
                    y_coord = self._stream.read_sinteger()
                    points.append((x_coord, y_coord))
                self._primitives.append(CifPolygon(points=tuple(points)))
                self._stream.expect_semi()
                continue
            if command_char == "C":
                self._stream.read_integer()
                while not self._stream.test_semi():
                    self._stream.skip_blanks()
                    transform = self._stream.get_char()
                    if transform == "T":
                        self._stream.read_sinteger()
                        self._stream.read_sinteger()
                    elif transform == "M":
                        self._stream.skip_blanks()
                        axis = self._stream.get_char()
                        if axis not in {"X", "Y"}:
                            raise CifParseError("Invalid 'M' transformation specification")
                    elif transform == "R":
                        self._stream.read_sinteger()
                        self._stream.read_sinteger()
                    else:
                        raise CifParseError("Invalid transformation specification")
                self._stream.expect_semi()
                continue
            if command_char in {"R", "W"}:
                self._stream.skip_to_semicolon()
                continue
            if command_char.isdigit():
                next_char = self._stream.peek_char()
                if command_char == "9" and next_char == "3":
                    self._stream.get_char()
                    self._stream.read_sinteger()
                    self._stream.read_sinteger()
                    self._stream.read_sinteger()
                    self._stream.read_sinteger()
                elif command_char == "9" and next_char == "4":
                    self._stream.get_char()
                    self._stream.read_sinteger()
                    self._stream.read_sinteger()
                elif command_char == "9" and next_char == "5":
                    self._stream.get_char()
                    self._stream.read_integer()
                elif command_char == "9" and next_char == "1":
                    self._stream.get_char()
                    self._stream.read_string()
                self._stream.skip_to_semicolon()
                continue

            self._warn("Unknown command ignored")
            self._stream.skip_to_semicolon()

    def _warn(self, message: str) -> None:
        self.warnings.append(message)


def rotated_box_points(
    width: int,
    height: int,
    center_x: int,
    center_y: int,
    rotation_x: int,
    rotation_y: int,
) -> list[tuple[int, int]]:
    """Match KLayout's rotated ``B`` command when ``rx, ry`` are provided."""

    if rotation_x >= 0 and rotation_y == 0:
        half_w = width / 2.0
        half_h = height / 2.0
        return [
            (int(round(center_x - half_w)), int(round(center_y - half_h))),
            (int(round(center_x + half_w)), int(round(center_y - half_h))),
            (int(round(center_x + half_w)), int(round(center_y + half_h))),
            (int(round(center_x - half_w)), int(round(center_y + half_h))),
        ]

    norm = math.hypot(float(rotation_x), float(rotation_y))
    if norm <= 0.0:
        return rotated_box_points(width, height, center_x, center_y, 1, 0)
    xw = width * 0.5 * rotation_x / norm
    yw = width * 0.5 * rotation_y / norm
    xh = -height * 0.5 * rotation_y / norm
    yh = height * 0.5 * rotation_x / norm
    return [
        (int(round(center_x - xw - xh)), int(round(center_y - yw - yh))),
        (int(round(center_x - xw + xh)), int(round(center_y - yw + yh))),
        (int(round(center_x + xw + xh)), int(round(center_y + yw + yh))),
        (int(round(center_x + xw - xh)), int(round(center_y + yw - yh))),
    ]
