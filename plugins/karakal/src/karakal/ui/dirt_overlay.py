"""The «Dirt» layer of a frame window: each dirty region's box and outline over the frame, its
hint on hover, and the dirtiness map under them."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from PyQt6.QtCore import QPointF, QRectF, Qt
from PyQt6.QtGui import QBrush, QColor, QFont, QImage, QPainterPath, QPen, QPixmap, QPolygonF, QTransform
from PyQt6.QtWidgets import (
    QCheckBox,
    QGraphicsItem,
    QGraphicsScene,
    QGraphicsSimpleTextItem,
    QHBoxLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from ..core.dirt_regions import (
    FULL_DIRTINESS,
    LEVEL_HEAVY,
    LEVEL_LIGHT,
    LEVEL_MEDIUM,
    DirtConfig,
    FrameDirtResult,
    dirt_layers,
    read_gray,
)

Translate = Callable[..., str]

LEVEL_QCOLORS = {
    LEVEL_LIGHT: QColor(255, 200, 80),
    LEVEL_MEDIUM: QColor(255, 140, 0),
    LEVEL_HEAVY: QColor(230, 40, 40),
}
BOX_COLOR = QColor(0, 220, 255)
# Above the cell marks of the window (result layer 3.0, marks just above it).
DIRT_Z = 5.0


def region_tooltip(region, translate: Translate) -> str:
    x, y, w, h = region.bbox
    return translate(
        "dirt.region_tip",
        id=region.id,
        level=translate(f"dirt.level.{region.level}"),
        dirtiness=int(round(region.dirtiness * 100)),
        x=x,
        y=y,
        w=w,
        h=h,
        edge=translate("dirt.edge") if region.edge else "",
        parts=region.parts,
        blur=region.blur_px,
        unsure=region.unsure_px,
        nick=region.nick_px,
        crumb=region.crumb_px,
    )


class DirtOverlay:
    """Owns the scene items of the layer and its switches; the frame window shows ``widget``."""

    def __init__(self, scene: QGraphicsScene, translate: Translate, parent: QWidget | None = None) -> None:
        self._scene = scene
        self._t = translate
        self._result: FrameDirtResult | None = None
        self._known = False
        self._config = DirtConfig()
        self._active = True
        self._items: list[QGraphicsItem] = []
        self._heat_item = scene.addPixmap(QPixmap())
        self._heat_item.setZValue(DIRT_Z - 0.1)
        self._heat_item.setVisible(False)
        self._heat_for: object = None
        self.widget = QWidget(parent)
        layout = QVBoxLayout(self.widget)
        layout.setContentsMargins(0, 4, 0, 0)
        layout.setSpacing(2)
        self.show_check = QCheckBox(self.widget)
        self.show_check.setChecked(True)
        layout.addWidget(self.show_check)
        row = QHBoxLayout()
        row.setContentsMargins(16, 0, 0, 0)
        self.box_check = QCheckBox(self.widget)
        self.box_check.setChecked(True)
        self.contour_check = QCheckBox(self.widget)
        self.contour_check.setChecked(True)
        self.heat_check = QCheckBox(self.widget)
        self.heat_check.setChecked(False)
        for check in (self.box_check, self.contour_check, self.heat_check):
            row.addWidget(check)
        row.addStretch(1)
        layout.addLayout(row)
        self.status = QLabel(self.widget)
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        for check in (self.show_check, self.box_check, self.contour_check, self.heat_check):
            check.toggled.connect(lambda _checked: self.rebuild())
        self.retranslate(translate)

    def retranslate(self, translate: Translate) -> None:
        self._t = translate
        self.show_check.setText(self._t("dirt.layer"))
        self.box_check.setText(self._t("dirt.show_box"))
        self.contour_check.setText(self._t("dirt.show_contour"))
        self.heat_check.setText(self._t("dirt.show_heat"))
        self._update_status()

    @property
    def result(self) -> FrameDirtResult | None:
        return self._result

    def items(self) -> tuple[QGraphicsItem, ...]:
        return tuple(self._items)

    def set_active(self, active: bool) -> None:
        """Only the cell check view of the window shows dirt."""

        if bool(active) != self._active:
            self._active = bool(active)
            self.rebuild()

    def set_result(self, result: FrameDirtResult | None, *, known: bool = True, config: DirtConfig | None = None) -> None:
        self._result = result
        self._known = bool(known and result is not None)
        if config is not None:
            self._config = config
        self._heat_for = None
        self.rebuild()

    def _update_status(self) -> None:
        result = self._result
        if result is None or not self._known:
            self.status.setText(self._t("dirt.frame_unknown"))
        elif not result.regions:
            self.status.setText(self._t("dirt.frame_none"))
        else:
            self.status.setText(
                self._t("dirt.frame_summary", count=len(result.regions), max=int(round(result.max_dirtiness * 100)))
            )

    def _scale(self) -> tuple[float, float]:
        result = self._result
        rect = self._scene.sceneRect()
        if result is None or result.width <= 0 or result.height <= 0 or rect.width() <= 0:
            return 1.0, 1.0
        return rect.width() / float(result.width), rect.height() / float(result.height)

    def clear(self) -> None:
        for item in self._items:
            if item.scene() is self._scene:
                self._scene.removeItem(item)
        self._items = []

    def rebuild(self) -> None:
        self.clear()
        self._update_status()
        result = self._result
        visible = self._active and bool(self.show_check.isChecked()) and result is not None and self._known
        for check in (self.box_check, self.contour_check, self.heat_check):
            check.setEnabled(bool(self.show_check.isChecked()))
        self._heat_item.setVisible(visible and self.heat_check.isChecked())
        if not visible:
            return
        sx, sy = self._scale()
        if self.heat_check.isChecked():
            self._update_heat(sx, sy)
        for region in result.regions:
            color = QColor(LEVEL_QCOLORS.get(region.level, LEVEL_QCOLORS[LEVEL_LIGHT]))
            tip = region_tooltip(region, self._t)
            if self.box_check.isChecked():
                x, y, w, h = region.bbox
                pen = QPen(BOX_COLOR, 1.5, Qt.PenStyle.DashLine)
                pen.setCosmetic(True)
                box = self._scene.addRect(QRectF(x * sx, y * sy, w * sx, h * sy), pen, QBrush(Qt.BrushStyle.NoBrush))
                box.setToolTip(tip)
                box.setZValue(DIRT_Z)
                self._items.append(box)
            if self.contour_check.isChecked() and len(region.contour) >= 3:
                polygon = QPolygonF([QPointF(px * sx, py * sy) for px, py in region.contour])
                path = QPainterPath()
                path.addPolygon(polygon)
                path.closeSubpath()
                fill = QColor(color)
                fill.setAlpha(40)
                pen = QPen(color, 2.5)
                pen.setCosmetic(True)
                outline = self._scene.addPath(path, pen, QBrush(fill))
                outline.setToolTip(tip)
                outline.setZValue(DIRT_Z + 0.1)
                self._items.append(outline)
            label = QGraphicsSimpleTextItem(
                f"#{region.id} {self._t(f'dirt.level.{region.level}')} {int(round(region.dirtiness * 100))}%"
            )
            font = QFont()
            font.setBold(True)
            font.setPointSize(10)
            label.setFont(font)
            label.setBrush(QBrush(QColor(255, 255, 255)))
            label.setPen(QPen(QColor(0, 0, 0), 0.6))
            label.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations, True)
            label.setPos(region.bbox[0] * sx, region.bbox[1] * sy)
            label.setToolTip(tip)
            label.setZValue(DIRT_Z + 0.2)
            self._scene.addItem(label)
            self._items.append(label)

    def _update_heat(self, sx: float, sy: float) -> None:
        result = self._result
        if result is None:
            return
        if self._heat_for is not result:
            self._heat_for = result
            self._heat_item.setPixmap(QPixmap())
            mask = read_gray(result.mask_path)
            if mask is None:
                return
            confidence = read_gray(result.confidence_path) if result.used_confidence else None
            if confidence is not None and confidence.shape != mask.shape:
                confidence = None
            density = dirt_layers(mask, confidence, self._config).density
            heat = np.clip(density / (FULL_DIRTINESS * 0.67), 0.0, 1.0)
            rgba = np.zeros((*heat.shape, 4), np.uint8)
            rgba[..., 0] = 255
            rgba[..., 1] = np.clip((1.0 - heat) * 200.0, 0, 255).astype(np.uint8)
            rgba[..., 3] = np.clip(heat * 180.0, 0, 255).astype(np.uint8)
            image = QImage(rgba.data, heat.shape[1], heat.shape[0], int(rgba.strides[0]), QImage.Format.Format_RGBA8888).copy()
            self._heat_item.setPixmap(QPixmap.fromImage(image))
        self._heat_item.setTransform(QTransform.fromScale(sx, sy))
