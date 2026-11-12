"""How the findings of the validation checks are marked on a frame (core/findings.py).

The kind of a finding is its colour; whose fault it probably is (the diagnosis) is the way it is drawn,
so the two read apart at a glance and for a colour-blind eye:

- a model's error on a good photo: a solid outline;
- an error the model was sure of: a thick outline with corner brackets;
- two models disagree, whose error is unknown: a dashed outline;
- a poor photo (the data, not the model): faint hatching under a dotted outline;
- doubt where the model is right: a dash-dot outline;
- a finding about the whole frame (edges shifted, the photo not aligned, blur): a band along the frame.

The marks are scene items over the frame (they stay sharp when zoomed); ``finding_heat`` draws the
findings as one heat picture, worst places brightest.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
from PyQt6.QtCore import QPointF, QRectF, Qt
from PyQt6.QtGui import QBrush, QColor, QFont, QFontMetricsF, QIcon, QPainter, QPen, QPixmap
from PyQt6.QtWidgets import QGraphicsItem, QGraphicsRectItem, QGraphicsSimpleTextItem

from ..core.findings import (
    DIAG_BAD_DATA,
    DIAG_CONFIDENT_ERROR,
    DIAG_MODEL_ERROR,
    DIAG_NOT_ALIGNED,
    DIAG_OK,
    DIAG_UNCERTAIN,
    DIAG_UNKNOWN,
    Finding,
)

try:
    import cv2
except Exception:  # pragma: no cover - OpenCV is a hard dependency of the app
    cv2 = None

# Colour of every kind of finding; kinds of one family share a hue.
KIND_COLORS: dict[str, str] = {
    # Connections: what is joined to what.
    "open": "#ef4444",
    "bridge": "#f97316",
    "merged_points": "#f97316",
    "missing_object": "#a855f7",
    "missing_point": "#a855f7",
    "extra_object": "#ec4899",
    "extra_point": "#ec4899",
    "shifted_point": "#e879f9",
    # Shape: wider, narrower, shifted.
    "neck": "#facc15",
    "bulge": "#84cc16",
    "boundary_shift": "#22d3ee",
    # The photo.
    "bad_photo": "#94a3b8",
    "not_aligned": "#60a5fa",
    # Doubt.
    "uncertain_region": "#38bdf8",
}
FALLBACK_COLOR = "#cbd5e1"
# Order of the legend: connections, shape, the photo, doubt, everything else.
KIND_ORDER = tuple(KIND_COLORS)

STYLE_SOLID = "solid"
STYLE_SURE = "sure"
STYLE_DASHED = "dashed"
STYLE_HATCHED = "hatched"
STYLE_DASH_DOT = "dash_dot"
STYLE_FAINT = "faint"
DIAGNOSIS_STYLES: dict[str, str] = {
    DIAG_MODEL_ERROR: STYLE_SOLID,
    DIAG_CONFIDENT_ERROR: STYLE_SURE,
    DIAG_UNKNOWN: STYLE_DASHED,
    DIAG_BAD_DATA: STYLE_HATCHED,
    DIAG_NOT_ALIGNED: STYLE_HATCHED,
    DIAG_UNCERTAIN: STYLE_DASH_DOT,
    DIAG_OK: STYLE_FAINT,
}


def color_of(kind: str) -> QColor:
    return QColor(KIND_COLORS.get(str(kind), FALLBACK_COLOR))


def style_of(diagnosis: str) -> str:
    return DIAGNOSIS_STYLES.get(str(diagnosis), STYLE_DASHED)


def kind_sort_key(kind: str) -> tuple[int, str]:
    order = {name: index for index, name in enumerate(KIND_ORDER)}
    return order.get(str(kind), len(order)), str(kind)


def kind_icon(kind: str, size: int = 14) -> QIcon:
    pixmap = QPixmap(size, size)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    color = color_of(kind)
    pen = QPen(color, 2.0)
    painter.setPen(pen)
    fill = QColor(color)
    fill.setAlpha(70)
    painter.setBrush(QBrush(fill))
    painter.drawRect(QRectF(2, 2, size - 4, size - 4))
    painter.end()
    return QIcon(pixmap)


def scene_rect(finding: Finding, scene_width: float, scene_height: float) -> QRectF | None:
    """The finding's box in scene units (the frame's box is ``None``: the whole frame)."""

    if finding.box is None:
        return None
    x, y, w, h = finding.box
    return QRectF(
        float(x) * scene_width,
        float(y) * scene_height,
        max(1.0, float(w) * scene_width),
        max(1.0, float(h) * scene_height),
    )


def _pen(color: QColor, width: float, style: Qt.PenStyle) -> QPen:
    pen = QPen(color, width, style)
    pen.setCosmetic(True)
    return pen


def add_finding_items(
    scene,
    finding: Finding,
    scene_width: float,
    scene_height: float,
    *,
    dim: bool = False,
    hot: bool = False,
    label: str | None = None,
    frame_band: int = 0,
) -> list:
    """Scene items marking one finding. ``frame_band``: the how-many-th finding of the whole frame
    (their bands lie inside each other along the frame's edge)."""

    color = color_of(finding.kind)
    alpha = 70 if dim else 235
    color.setAlpha(alpha)
    style = style_of(finding.diagnosis)
    width = 3.0 if hot else 2.0
    items: list = []
    rect = scene_rect(finding, scene_width, scene_height)
    whole_frame = rect is None
    if whole_frame:
        inset = 3.0 + 6.0 * float(frame_band)
        rect = QRectF(inset, inset, max(1.0, scene_width - 2 * inset), max(1.0, scene_height - 2 * inset))
    if hot and not whole_frame:
        halo = scene.addRect(rect.adjusted(-3, -3, 3, 3), _pen(QColor(255, 230, 72, 220), 5.0, Qt.PenStyle.SolidLine))
        items.append(halo)
    if style == STYLE_HATCHED:
        fill = QColor(color)
        fill.setAlpha(28 if dim else 70)
        brush = QBrush(fill, Qt.BrushStyle.BDiagPattern) if not whole_frame else QBrush(Qt.BrushStyle.NoBrush)
        items.append(scene.addRect(rect, _pen(color, width, Qt.PenStyle.DotLine), brush))
    elif style == STYLE_SURE:
        items.append(scene.addRect(rect, _pen(color, width + 1.5, Qt.PenStyle.SolidLine)))
        items.extend(_brackets(scene, rect, color, width + 1.0))
    elif style == STYLE_DASHED:
        items.append(scene.addRect(rect, _pen(color, width, Qt.PenStyle.DashLine)))
    elif style == STYLE_DASH_DOT:
        items.append(scene.addRect(rect, _pen(color, width, Qt.PenStyle.DashDotLine)))
    elif style == STYLE_FAINT:
        items.append(scene.addRect(rect, _pen(color, 1.0, Qt.PenStyle.DotLine)))
    else:
        items.append(scene.addRect(rect, _pen(color, width, Qt.PenStyle.SolidLine)))
    if label:
        text = QGraphicsSimpleTextItem(label)
        font = QFont()
        font.setPointSizeF(8.0)
        font.setBold(bool(hot))
        text.setFont(font)
        text.setBrush(QBrush(color))
        # The label keeps its size on screen whatever the zoom.
        text.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations, True)
        anchor = rect.topLeft() if not whole_frame else QPointF(rect.left() + 4.0, rect.top() + 2.0)
        text.setPos(anchor + QPointF(0.0, -1.0))
        scene.addItem(text)
        items.append(text)
    return items


def add_label_stack(scene, anchor: QPointF, lines: list[tuple[str, QColor, bool]]) -> list:
    """Labels of the findings of one place, one line each in its own colour, stacked downwards from
    ``anchor``. The stack keeps its size on screen whatever the zoom, so its lines never overlap."""

    if not lines:
        return []
    width, height = label_stack_size(lines)
    # A dark plate under the lines keeps them readable over bright metal; it ignores the zoom, so its
    # children's offsets are in screen pixels.
    plate = QGraphicsRectItem(0.0, 0.0, width, height)
    plate.setPen(QPen(Qt.PenStyle.NoPen))
    plate.setBrush(QBrush(QColor(15, 23, 42, 185)))
    plate.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations, True)
    plate.setPos(anchor + QPointF(1.0, 1.0))
    scene.addItem(plate)
    items: list = [plate]
    for index, (text, color, bold) in enumerate(lines):
        item = QGraphicsSimpleTextItem(text, plate)
        item.setFont(_label_font(bold))
        item.setBrush(QBrush(color))
        item.setPos(LABEL_PAD, LABEL_PAD + LABEL_LINE * index)
        items.append(item)
    return items


LABEL_PAD = 2.0
LABEL_LINE = 13.0


def _label_font(bold: bool = False) -> QFont:
    font = QFont()
    font.setPointSizeF(8.0)
    font.setBold(bool(bold))
    return font


def label_stack_size(lines: list[tuple[str, QColor, bool]]) -> tuple[float, float]:
    """The size on screen, in pixels, of a stack of labels (add_label_stack)."""

    width = 0.0
    for text, _color, bold in lines:
        width = max(width, QFontMetricsF(_label_font(bold)).horizontalAdvance(text))
    return width + 2 * LABEL_PAD, LABEL_LINE * len(lines) + 2 * LABEL_PAD


def place_labels(stacks, scale: float) -> list[int]:
    """Which stacks of labels to show at a zoom of ``scale`` screen pixels per scene unit so that none
    covers another: ``stacks`` are (anchor in the scene, (width, height) on screen) in the order of
    importance; a stack that would cover one already placed is left out (the hover still tells)."""

    placed: list[QRectF] = []
    chosen: list[int] = []
    for index, (anchor, (width, height)) in enumerate(stacks):
        rect = QRectF(float(anchor.x()) * scale, float(anchor.y()) * scale, float(width), float(height))
        if any(rect.intersects(other) for other in placed):
            continue
        placed.append(rect)
        chosen.append(index)
    return chosen


# Whose fault it is, most telling first: of two findings of one place the more telling one leads.
DIAGNOSIS_RANK: dict[str, int] = {
    DIAG_CONFIDENT_ERROR: 0,
    DIAG_MODEL_ERROR: 1,
    DIAG_NOT_ALIGNED: 2,
    DIAG_BAD_DATA: 3,
    DIAG_UNKNOWN: 4,
    DIAG_UNCERTAIN: 5,
    DIAG_OK: 6,
}
_DATA_DIAGNOSES = frozenset({DIAG_BAD_DATA, DIAG_NOT_ALIGNED})


def boxes_overlap(first, second, share: float = 0.5) -> bool:
    """Two boxes are one place when their common part is at least ``share`` of the smaller one."""

    if first is None or second is None:
        return first is None and second is None
    ax, ay, aw, ah = (float(value) for value in first)
    bx, by, bw, bh = (float(value) for value in second)
    across = min(ax + aw, bx + bw) - max(ax, bx)
    down = min(ay + ah, by + bh) - max(ay, by)
    if across <= 0.0 or down <= 0.0:
        return False
    smaller = min(aw * ah, bw * bh)
    return smaller > 0.0 and across * down >= share * smaller


def _same_finding(first: Finding, second: Finding) -> bool:
    """One thing seen twice: the same kind at the same place, of the same model; a fault of the photo
    is one fault whichever model's mask it was found against."""

    if first.kind != second.kind:
        return False
    one_model = first.model_id == second.model_id
    of_the_photo = first.diagnosis in _DATA_DIAGNOSES and second.diagnosis in _DATA_DIAGNOSES
    return (one_model or of_the_photo) and boxes_overlap(first.box, second.box)


def merge_findings(findings) -> list[tuple[Finding, tuple[Finding, ...]]]:
    """The findings with repeats folded: (the leading finding, the ones it stands for).

    Two checks often find the same thing (the comparison of the models and the check against the
    photo both see an open), and a fault of the photo is found once per model. The most telling of a
    group leads (an error the model was sure of before a plain error before a disagreement), the worst
    severity of the group is its severity, and a group of several models' masks belongs to none.
    """

    ordered = sorted(findings, key=lambda item: (DIAGNOSIS_RANK.get(item.diagnosis, len(DIAGNOSIS_RANK)), -float(item.severity)))
    groups: list[list[Finding]] = []
    for finding in ordered:
        for group in groups:
            if _same_finding(group[0], finding):
                group.append(finding)
                break
        else:
            groups.append([finding])
    merged: list[tuple[Finding, tuple[Finding, ...]]] = []
    for group in groups:
        lead = group[0]
        if len(group) > 1:
            severity = max(float(item.severity) for item in group)
            models = {item.model_id for item in group}
            changes: dict[str, object] = {}
            if severity > float(lead.severity):
                changes["severity"] = severity
            if len(models) > 1:
                changes["model_id"] = ""
            if changes:
                lead = replace(lead, **changes)
        merged.append((lead, tuple(group[1:])))
    return merged


def label_places(findings) -> list[list[Finding]]:
    """The findings in places for their labels: findings whose boxes overlap share one stack of labels
    (a mirrored pair of models, a neck found next to a bulge), so the labels do not overlap; the first
    of a place is where its labels hang. The findings of the whole frame are one place."""

    places: list[list[Finding]] = []
    whole_frame: list[Finding] = []
    for finding in findings:
        if finding.box is None:
            if not whole_frame:
                places.append(whole_frame)
            whole_frame.append(finding)
            continue
        for place in places:
            if place is not whole_frame and boxes_overlap(place[0].box, finding.box):
                place.append(finding)
                break
        else:
            places.append([finding])
    return places


def _brackets(scene, rect: QRectF, color: QColor, width: float) -> list:
    """Corner brackets just outside a box: the mark of an error the model was sure of."""

    pen = _pen(color, width, Qt.PenStyle.SolidLine)
    reach = max(4.0, min(rect.width(), rect.height()) * 0.35)
    gap = 3.0
    left, top, right, bottom = rect.left() - gap, rect.top() - gap, rect.right() + gap, rect.bottom() + gap
    items = []
    for x, y, dx, dy in ((left, top, 1, 1), (right, top, -1, 1), (left, bottom, 1, -1), (right, bottom, -1, -1)):
        items.append(scene.addLine(x, y, x + dx * reach, y, pen))
        items.append(scene.addLine(x, y, x, y + dy * reach, pen))
    return items


def finding_heat(findings, shape: tuple[int, int], *, blur: float = 0.0) -> np.ndarray:
    """One picture of all the findings: every pixel the severity of the worst finding over it (0..1).

    Findings of the whole frame are left out (they would cover everything); a light blur, when asked,
    joins close findings into one hot place.
    """

    height, width = int(shape[0]), int(shape[1])
    heat = np.zeros((max(1, height), max(1, width)), dtype=np.float32)
    for finding in findings:
        if finding.box is None or finding.diagnosis == DIAG_OK:
            continue
        x, y, w, h = finding.box
        x0, y0 = int(np.floor(float(x) * width)), int(np.floor(float(y) * height))
        x1 = int(np.ceil((float(x) + float(w)) * width))
        y1 = int(np.ceil((float(y) + float(h)) * height))
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(width, max(x0 + 1, x1)), min(height, max(y0 + 1, y1))
        region = heat[y0:y1, x0:x1]
        np.maximum(region, float(finding.severity), out=region)
    if blur > 0.0 and cv2 is not None and np.any(heat):
        heat = np.maximum(heat, cv2.GaussianBlur(heat, (0, 0), float(blur)))
    return heat


__all__ = [
    "DIAGNOSIS_RANK",
    "DIAGNOSIS_STYLES",
    "KIND_COLORS",
    "KIND_ORDER",
    "add_finding_items",
    "add_label_stack",
    "boxes_overlap",
    "color_of",
    "finding_heat",
    "kind_icon",
    "kind_sort_key",
    "label_places",
    "label_stack_size",
    "place_labels",
    "merge_findings",
    "scene_rect",
    "style_of",
]
