"""scene_renderer: affine2 d."""
from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True)
class Affine2D:
    a: int = 1
    b: int = 0
    c: int = 0
    d: int = 1
    tx: int = 0
    ty: int = 0

    def apply(self, x: float, y: float) -> tuple[float, float]:
        return self.a * x + self.b * y + self.tx, self.c * x + self.d * y + self.ty

    def compose(self, child: "Affine2D") -> "Affine2D":
        return Affine2D(
            a=self.a * child.a + self.b * child.c,
            b=self.a * child.b + self.b * child.d,
            c=self.c * child.a + self.d * child.c,
            d=self.c * child.b + self.d * child.d,
            tx=self.a * child.tx + self.b * child.ty + self.tx,
            ty=self.c * child.tx + self.d * child.ty + self.ty,
        )

@dataclass
class RenderStats:
    primitive_count: int = 0
    wire_segment_count: int = 0
    text_count: int = 0

def re_search_first_quoted(text: str) -> str | None:
    import re

    match = re.search(r'"([^"]+)"', text)
    return match.group(1) if match else None
