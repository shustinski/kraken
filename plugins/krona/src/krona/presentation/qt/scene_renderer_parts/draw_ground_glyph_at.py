"""scene_renderer: draw ground glyph at."""
from __future__ import annotations
from PyQt6.QtGui import QColor, QPen


class DrawGroundGlyphAtMixin:
    def _draw_ground_glyph_at(
        self,
        world_x: float,
        world_y: float,
        color: tuple[int, int, int],
        label_text: str | None = None,
    ) -> None:
        draw_color = self._visible_port_color("ground", color)
        pen = QPen(QColor(*draw_color, 230))
        pen.setWidthF(1.0)
        pen.setCosmetic(True)
        lines = [
            ((-10, 0), (10, 0)),
            ((-7, -3), (7, -3)),
            ((-4, -6), (4, -6)),
            ((-1, -9), (1, -9)),
        ]
        for start, end in lines:
            x1, y1 = self._to_scene(world_x + start[0], world_y + start[1])
            x2, y2 = self._to_scene(world_x + end[0], world_y + end[1])
            item = self._scene.addLine(x1, y1, x2, y2, pen)
            item.setZValue(920)
            self._stats.primitive_count += 1

        if label_text:
            gap_world = 4.0 / max(self._scale, 1e-6)
            self._draw_text(
                text=label_text,
                world_x=world_x + gap_world,
                world_y=world_y,
                justify="CENTERLEFT",
                color=draw_color,
                text_height=24,
                text_scale=0.1,
            )
