from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Connection:
    port: str
    instance: str | None = None


@dataclass(frozen=True)
class Point:
    x: int
    y: int


@dataclass(frozen=True)
class Wire:
    points: list[Point] = field(default_factory=list)


@dataclass(frozen=True)
class Net:
    name: str
    connections: list[Connection] = field(default_factory=list)
    wires: list[Wire] = field(default_factory=list)


@dataclass(frozen=True)
class Instance:
    name: str
    view: str | None = None
    cell: str | None = None
    library: str | None = None
    designator: str | None = None
    x: int | None = None
    y: int | None = None
    orientation: str | None = None


@dataclass(frozen=True)
class TopLevelNetlist:
    design: str
    library: str
    cell: str
    view: str
    instances: list[Instance]
    nets: list[Net]
