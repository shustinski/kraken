"""Shared CIF primitive records for Contour loaders."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class CifComment:
    content: str


@dataclass(frozen=True, slots=True)
class CifPolygon:
    points: tuple[tuple[int, int], ...]


@dataclass(frozen=True, slots=True)
class CifBox:
    width: int
    height: int
    center_x: int
    center_y: int
    rotation_x: int
    rotation_y: int


CifPrimitive = CifComment | CifPolygon | CifBox
