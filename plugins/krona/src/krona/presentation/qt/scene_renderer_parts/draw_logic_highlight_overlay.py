"""scene_renderer: draw logic highlight overlay."""
from __future__ import annotations
from PyQt6.QtGui import QBrush, QColor, QPen
from krona.application.ports import SceneData


class DrawLogicHighlightOverlayMixin:
    def _draw_logic_highlight_overlay(self, scene_data: SceneData) -> None:
        if not self._highlight_nets and not self._highlight_instances:
            return

        net_pen = QPen(QColor(79, 171, 255, 235))
        net_pen.setWidthF(2.2)
        net_pen.setCosmetic(True)

        for net in scene_data.netlist.nets:
            if net.name not in self._highlight_nets:
                continue
            for wire in net.wires:
                if len(wire.points) == 1:
                    x, y = self._to_scene(wire.points[0].x, wire.points[0].y)
                    dot = self._scene.addEllipse(x - 2.6, y - 2.6, 5.2, 5.2, net_pen, QBrush(net_pen.color()))
                    dot.setZValue(3000)
                    continue
                if len(wire.points) < 2:
                    continue
                for start, end in zip(wire.points, wire.points[1:]):
                    x1, y1 = self._to_scene(start.x, start.y)
                    x2, y2 = self._to_scene(end.x, end.y)
                    item = self._scene.addLine(x1, y1, x2, y2, net_pen)
                    item.setZValue(3000)
