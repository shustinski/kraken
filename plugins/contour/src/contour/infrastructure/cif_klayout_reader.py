"""CIF reader ported from KLayout ``dbCIFReader.cc``.

KLayout stores each ``P`` command as a single polygon hull (``assign_hull``) without
splitting keyhole bridges into hole contours. Contour follows the same rule on load.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path

from .cif_primitives import CifBox, CifComment, CifPolygon, CifPrimitive


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
    def __init__(self, text: str) -> None:
        self._text = text
        self._pos = 0
        self.line_number = 1

    def at_end(self) -> bool:
        return self._pos >= len(self._text)

    def peek_char(self) -> str:
        if self.at_end():
            return ""
        return self._text[self._pos]

    def get_char(self) -> str:
        if self.at_end():
            raise CifParseError("Unexpected end of file")
        char = self._text[self._pos]
        self._pos += 1
        if char == "\n":
            self.line_number += 1
        return char


class _KLayoutCifReader:
    def __init__(self, stream: _CifTextStream) -> None:
        self._stream = stream
        self.warnings: list[str] = []
        self._primitives: list[CifPrimitive] = []

    def read_top_cell(self) -> list[CifPrimitive]:
        self._read_cell(level=0, scale_factor=1.0)
        self._skip_blanks()
        if not self._stream.at_end():
            self._warn("E command is followed by more text")
        return self._primitives

    def _read_cell(self, *, level: int, scale_factor: float) -> None:
        nx = ny = dx = dy = 0
        layer_selected = False
        path_mode = -1

        while True:
            self._skip_blanks()
            if self._stream.at_end():
                raise CifParseError("Unexpected end of file")

            command_char = self._get_char()
            if command_char == ";":
                continue
            if command_char == "(":
                comment = self._read_comment_body()
                self._primitives.append(CifComment(content=comment))
                continue
            if command_char == "E":
                if level > 0:
                    raise CifParseError("'E' command must be outside a cell specification")
                return
            if command_char == "D":
                self._skip_blanks()
                sub = self._get_char()
                if sub == "S":
                    self._read_integer()
                    if not self._test_semi():
                        self._read_integer()
                        divider = self._read_integer()
                        if divider == 0:
                            raise CifParseError("'DS' command: divider cannot be zero")
                    self._expect_semi()
                    self._read_cell(level=level + 1, scale_factor=scale_factor)
                elif sub == "F":
                    if level == 0:
                        raise CifParseError("'DF' command must be inside a cell specification")
                    self._skip_to_end()
                    return
                elif sub == "D":
                    self._read_integer()
                    self._warn("'DD' command ignored")
                    self._skip_to_end()
                else:
                    raise CifParseError("Invalid 'D' sub-command")
                continue
            if command_char == "L":
                name = self._read_name()
                if not name:
                    raise CifParseError("Missing layer name in 'L' command")
                layer_selected = True
                self._expect_semi()
                continue
            if command_char == "B":
                if not layer_selected:
                    self._warn("'B' command ignored since no layer was selected")
                    self._skip_to_end()
                    continue
                width = self._read_integer()
                height = self._read_integer()
                center_x = self._read_sinteger()
                center_y = self._read_sinteger()
                rotation_x = 1
                rotation_y = 0
                if not self._test_semi():
                    rotation_x = self._read_sinteger()
                    rotation_y = self._read_sinteger()
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
                self._expect_semi()
                continue
            if command_char == "P":
                if not layer_selected:
                    self._warn("'P' command ignored since no layer was selected")
                    self._skip_to_end()
                    continue
                points: list[tuple[int, int]] = []
                while not self._test_semi():
                    x_coord = self._read_sinteger()
                    y_coord = self._read_sinteger()
                    points.append((x_coord, y_coord))
                self._primitives.append(CifPolygon(points=tuple(points)))
                self._expect_semi()
                continue
            if command_char == "C":
                self._read_integer()
                while not self._test_semi():
                    self._skip_blanks()
                    transform = self._get_char()
                    if transform == "T":
                        self._read_sinteger()
                        self._read_sinteger()
                    elif transform == "M":
                        self._skip_blanks()
                        axis = self._get_char()
                        if axis not in {"X", "Y"}:
                            raise CifParseError("Invalid 'M' transformation specification")
                    elif transform == "R":
                        self._read_sinteger()
                        self._read_sinteger()
                    else:
                        raise CifParseError("Invalid transformation specification")
                self._expect_semi()
                continue
            if command_char in {"R", "W"}:
                self._skip_to_end()
                continue
            if command_char.isdigit():
                next_char = self._stream.peek_char()
                if command_char == "9" and next_char == "3":
                    self._get_char()
                    self._read_sinteger()
                    self._read_sinteger()
                    self._read_sinteger()
                    self._read_sinteger()
                elif command_char == "9" and next_char == "4":
                    self._get_char()
                    self._read_sinteger()
                    self._read_sinteger()
                elif command_char == "9" and next_char == "5":
                    self._get_char()
                    self._read_integer()
                elif command_char == "9" and next_char == "1":
                    self._get_char()
                    self._read_string()
                self._skip_to_end()
                continue

            self._warn("Unknown command ignored")
            self._skip_to_end()

    def _read_comment_body(self) -> str:
        content = "("
        nesting = 1
        while nesting > 0 and not self._stream.at_end():
            char = self._get_char()
            content += char
            if char == "(":
                nesting += 1
            elif char == ")":
                nesting -= 1
        while not self._stream.at_end() and self._stream.peek_char() != ";":
            self._get_char()
        return content

    def _skip_blanks(self) -> None:
        # Match KLayout ``CIFReader::skip_blanks`` — stop before commands and tokens.
        while not self._stream.at_end():
            char = self._stream.peek_char()
            if char.isupper() or char.isdigit() or char in "-();":
                return
            self._get_char()

    def _skip_sep(self) -> None:
        # Match KLayout ``CIFReader::skip_sep``.
        while not self._stream.at_end():
            char = self._stream.peek_char()
            if char.isdigit() or char in "-();":
                return
            self._get_char()

    def _read_integer_digits(self) -> int:
        if self._stream.at_end() or not self._stream.peek_char().isdigit():
            raise CifParseError("Digit expected")
        value = 0
        while not self._stream.at_end() and self._stream.peek_char().isdigit():
            value = value * 10 + int(self._get_char())
        return value

    def _read_integer(self) -> int:
        self._skip_sep()
        return self._read_integer_digits()

    def _read_sinteger(self) -> int:
        self._skip_sep()
        negative = False
        if self._stream.peek_char() == "-":
            self._get_char()
            negative = True
        value = self._read_integer_digits()
        return -value if negative else value

    def _read_name(self) -> str:
        self._skip_blanks()
        name_chars: list[str] = []
        while not self._stream.at_end():
            char = self._stream.peek_char()
            if not (char.isalpha() or char == "_" or char.isdigit()):
                break
            name_chars.append(self._get_char())
        return "".join(name_chars)

    def _read_string(self) -> str:
        self._skip_sep()
        if self._stream.at_end():
            return ""
        quote = self._stream.peek_char()
        if quote in {'"', "'"}:
            self._get_char()
            chars: list[str] = []
            while not self._stream.at_end() and self._stream.peek_char() != quote:
                char = self._get_char()
                if char == "\\" and not self._stream.at_end():
                    char = self._get_char()
                chars.append(char)
            if not self._stream.at_end():
                self._get_char()
            return "".join(chars)
        chars = []
        while not self._stream.at_end():
            char = self._stream.peek_char()
            if char.isspace() or char == ";":
                break
            chars.append(self._get_char())
        return "".join(chars)

    def _test_semi(self) -> bool:
        self._skip_blanks()
        if not self._stream.at_end() and self._stream.peek_char() == ";":
            return True
        return False

    def _expect_semi(self) -> None:
        if not self._test_semi():
            raise CifParseError("Expected ';' command terminator")
        self._get_char()

    def _skip_to_end(self) -> None:
        while not self._stream.at_end() and self._get_char() != ";":
            pass

    def _get_char(self) -> str:
        return self._stream.get_char()

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
