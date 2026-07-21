from __future__ import annotations

from math import pi, sin

from PyQt6.QtCore import QPointF, QRectF, QSize, Qt, QTimer
from PyQt6.QtGui import QColor, QPainter, QPainterPath, QPen, QPolygonF
from PyQt6.QtWidgets import QWidget

from .models import PetMood, PetType, RewardEventType
from .registry import DEFAULT_SKIN_BY_PET
from .visuals import mood_for_reward_event, skin_visual_for


class PetAvatarWidget(QWidget):
    """Animated vector avatar for the selected gamification pet."""

    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        avatar_size: int = 132,
        animated: bool = True,
    ) -> None:
        super().__init__(parent)
        self._avatar_size = max(48, int(avatar_size))
        self._pet_type = PetType.KRAKEN
        self._skin_id = DEFAULT_SKIN_BY_PET[PetType.KRAKEN]
        self._mood = PetMood.IDLE
        self._locked = False
        self._phase = 0.0
        self._pulse_ticks = 0
        self.setMinimumSize(self._avatar_size, self._avatar_size)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self._timer = QTimer(self)
        self._timer.setInterval(80)
        self._timer.timeout.connect(self._advance_animation)
        if animated:
            self._timer.start()

    def sizeHint(self) -> QSize:
        return QSize(self._avatar_size, self._avatar_size)

    def set_pet(
        self,
        pet_type: PetType,
        skin_id: str | None = None,
        mood: PetMood = PetMood.IDLE,
        *,
        locked: bool = False,
    ) -> None:
        previous = (self._pet_type, self._skin_id, self._mood, self._locked)
        self._pet_type = pet_type
        self._skin_id = skin_id or DEFAULT_SKIN_BY_PET[pet_type]
        self._locked = bool(locked)
        self.set_mood(mood)
        if previous[:2] != (self._pet_type, self._skin_id):
            self._pulse_ticks = max(self._pulse_ticks, 10)
        self.update()

    def set_mood(self, mood: PetMood) -> None:
        if self._mood != mood:
            self._pulse_ticks = max(self._pulse_ticks, 16 if mood in {PetMood.CELEBRATING, PetMood.LEVEL_UP} else 8)
        self._mood = mood
        self.update()

    def react_to_event(self, event_type: RewardEventType | None) -> None:
        self.set_mood(mood_for_reward_event(event_type))

    def _advance_animation(self) -> None:
        step = 0.12
        if self._mood == PetMood.HAPPY:
            step = 0.20
        elif self._mood in {PetMood.CELEBRATING, PetMood.LEVEL_UP}:
            step = 0.28
        elif self._mood == PetMood.TIRED:
            step = 0.06
        self._phase = (self._phase + step) % (2 * pi)
        if self._pulse_ticks > 0:
            self._pulse_ticks -= 1
        self.update()

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        side = max(1.0, float(min(self.width(), self.height())))
        painter.translate((self.width() - side) / 2.0, (self.height() - side) / 2.0)
        painter.scale(side / 120.0, side / 120.0)
        style = skin_visual_for(self._pet_type, self._skin_id)
        body = self._qcolor(style.body_color)
        accent = self._qcolor(style.accent_color)
        secondary = self._qcolor(style.secondary_color)
        accessory = self._qcolor(style.accessory_color)
        line = self._qcolor("#6B7280" if self._locked else "#1F2937")
        if self._locked:
            body = self._muted(body)
            accent = self._muted(accent)
            secondary = self._muted(secondary)
            accessory = self._muted(accessory)
            painter.setOpacity(0.58)
        bob = self._bob_offset()
        scale = self._pulse_scale()
        painter.translate(60.0, 60.0 + bob)
        painter.scale(scale, scale)
        painter.translate(-60.0, -60.0)
        self._draw_shadow(painter)
        self._draw_pet(painter, body, accent, secondary, accessory, line)
        if self._locked:
            self._draw_lock_overlay(painter, line)
        else:
            self._draw_mood_effect(painter, accessory, line)

    def _bob_offset(self) -> float:
        amplitude = {
            PetMood.IDLE: 1.8,
            PetMood.FOCUSED: 1.0,
            PetMood.HAPPY: 3.8,
            PetMood.CELEBRATING: 5.5,
            PetMood.LEVEL_UP: 5.0,
            PetMood.TIRED: 0.8,
            PetMood.HUNGRY: 1.4,
        }.get(self._mood, 1.8)
        return sin(self._phase) * amplitude

    def _pulse_scale(self) -> float:
        if self._pulse_ticks <= 0:
            return 1.0
        return 1.0 + 0.035 * sin(self._phase * 2.0)

    def _draw_pet(
        self,
        painter: QPainter,
        body: QColor,
        accent: QColor,
        secondary: QColor,
        accessory: QColor,
        line: QColor,
    ) -> None:
        if self._pet_type == PetType.KRAKEN:
            self._draw_kraken(painter, body, accent, secondary, accessory, line)
        elif self._pet_type == PetType.CAT:
            self._draw_cat(painter, body, accent, secondary, accessory, line)
        elif self._pet_type == PetType.DOG:
            self._draw_dog(painter, body, accent, secondary, accessory, line)
        elif self._pet_type == PetType.CAPYBARA:
            self._draw_capybara(painter, body, accent, secondary, accessory, line)
        elif self._pet_type == PetType.CARNIVOROUS_PLANT:
            self._draw_plant(painter, body, accent, secondary, accessory, line)
        elif self._pet_type == PetType.HORSE:
            self._draw_horse(painter, body, accent, secondary, accessory, line)
        elif self._pet_type == PetType.FROG:
            self._draw_frog(painter, body, accent, secondary, accessory, line)
        else:
            self._draw_hamster(painter, body, accent, secondary, accessory, line)

    def _draw_shadow(self, painter: QPainter) -> None:
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(15, 23, 42, 34))
        painter.drawEllipse(QRectF(24, 94, 72, 12))

    def _draw_kraken(
        self,
        painter: QPainter,
        body: QColor,
        accent: QColor,
        secondary: QColor,
        accessory: QColor,
        line: QColor,
    ) -> None:
        self._set_fill(painter, secondary, line)
        for x_coord in (30, 44, 58, 72):
            painter.drawRoundedRect(QRectF(x_coord, 66, 12, 30 + 4 * sin(self._phase + x_coord)), 6, 6)
        self._set_fill(painter, body, line)
        painter.drawEllipse(QRectF(28, 20, 64, 58))
        self._set_fill(painter, accent, line)
        painter.drawEllipse(QRectF(42, 38, 10, 13))
        painter.drawEllipse(QRectF(68, 38, 10, 13))
        self._draw_eye(painter, 47, 44, line)
        self._draw_eye(painter, 73, 44, line)
        self._draw_smile(painter, QRectF(50, 50, 20, 12), line)
        self._draw_badge(painter, accessory, line)

    def _draw_cat(
        self,
        painter: QPainter,
        body: QColor,
        accent: QColor,
        _secondary: QColor,
        accessory: QColor,
        line: QColor,
    ) -> None:
        self._set_fill(painter, body, line)
        painter.drawPolygon(QPolygonF([QPointF(34, 42), QPointF(43, 18), QPointF(54, 43)]))
        painter.drawPolygon(QPolygonF([QPointF(66, 43), QPointF(78, 18), QPointF(86, 42)]))
        painter.drawEllipse(QRectF(28, 34, 64, 56))
        self._set_fill(painter, accent, line)
        painter.drawEllipse(QRectF(48, 58, 24, 18))
        self._draw_eye(painter, 47, 56, line)
        self._draw_eye(painter, 73, 56, line)
        self._draw_nose(painter, 60, 66, line)
        for y_coord in (62, 68):
            painter.drawLine(QPointF(22, y_coord), QPointF(43, y_coord - 2))
            painter.drawLine(QPointF(77, y_coord - 2), QPointF(98, y_coord))
        self._draw_badge(painter, accessory, line)

    def _draw_dog(
        self,
        painter: QPainter,
        body: QColor,
        accent: QColor,
        secondary: QColor,
        accessory: QColor,
        line: QColor,
    ) -> None:
        self._set_fill(painter, secondary, line)
        painter.drawEllipse(QRectF(21, 35, 25, 42))
        painter.drawEllipse(QRectF(74, 35, 25, 42))
        self._set_fill(painter, body, line)
        painter.drawEllipse(QRectF(30, 28, 60, 62))
        self._set_fill(painter, accent, line)
        painter.drawRoundedRect(QRectF(47, 58, 26, 20), 10, 10)
        self._draw_eye(painter, 48, 52, line)
        self._draw_eye(painter, 72, 52, line)
        self._draw_nose(painter, 60, 66, line)
        self._draw_smile(painter, QRectF(51, 68, 18, 9), line)
        self._draw_badge(painter, accessory, line)

    def _draw_capybara(
        self,
        painter: QPainter,
        body: QColor,
        accent: QColor,
        secondary: QColor,
        accessory: QColor,
        line: QColor,
    ) -> None:
        self._set_fill(painter, body, line)
        painter.drawEllipse(QRectF(16, 48, 76, 36))
        painter.drawEllipse(QRectF(55, 30, 42, 42))
        self._set_fill(painter, secondary, line)
        painter.drawEllipse(QRectF(82, 50, 16, 13))
        self._set_fill(painter, accent, line)
        painter.drawEllipse(QRectF(48, 72, 12, 8))
        self._draw_eye(painter, 72, 46, line)
        self._draw_eye(painter, 88, 47, line)
        self._draw_nose(painter, 89, 56, line)
        self._draw_badge(painter, accessory, line)

    def _draw_plant(
        self,
        painter: QPainter,
        body: QColor,
        accent: QColor,
        secondary: QColor,
        accessory: QColor,
        line: QColor,
    ) -> None:
        self._set_fill(painter, accessory, line)
        painter.drawRoundedRect(QRectF(42, 78, 36, 20), 5, 5)
        self._set_fill(painter, secondary, line)
        painter.drawLine(QPointF(60, 76), QPointF(60, 58))
        painter.drawEllipse(QRectF(36, 60, 22, 12))
        painter.drawEllipse(QRectF(62, 60, 22, 12))
        self._set_fill(painter, body, line)
        painter.drawEllipse(QRectF(35, 24, 50, 42))
        self._set_fill(painter, accent, line)
        painter.drawPie(QRectF(42, 34, 36, 22), 0, -180 * 16)
        self._draw_eye(painter, 50, 40, line)
        self._draw_eye(painter, 70, 40, line)
        self._draw_teeth(painter, line)

    def _draw_horse(
        self,
        painter: QPainter,
        body: QColor,
        accent: QColor,
        secondary: QColor,
        accessory: QColor,
        line: QColor,
    ) -> None:
        self._set_fill(painter, body, line)
        painter.drawRoundedRect(QRectF(43, 44, 28, 44), 12, 12)
        painter.drawEllipse(QRectF(49, 24, 42, 38))
        painter.drawPolygon(QPolygonF([QPointF(48, 35), QPointF(44, 18), QPointF(58, 29)]))
        self._set_fill(painter, secondary, line)
        painter.drawPolygon(QPolygonF([QPointF(54, 28), QPointF(70, 24), QPointF(55, 65), QPointF(45, 58)]))
        self._set_fill(painter, accent, line)
        painter.drawRoundedRect(QRectF(73, 43, 22, 16), 8, 8)
        self._draw_eye(painter, 68, 39, line)
        self._draw_nose(painter, 86, 51, line)
        self._draw_badge(painter, accessory, line)

    def _draw_frog(
        self,
        painter: QPainter,
        body: QColor,
        accent: QColor,
        secondary: QColor,
        accessory: QColor,
        line: QColor,
    ) -> None:
        self._set_fill(painter, body, line)
        painter.drawEllipse(QRectF(26, 36, 68, 50))
        painter.drawEllipse(QRectF(34, 24, 20, 20))
        painter.drawEllipse(QRectF(66, 24, 20, 20))
        self._set_fill(painter, accent, line)
        painter.drawEllipse(QRectF(49, 60, 22, 12))
        self._draw_eye(painter, 44, 34, line)
        self._draw_eye(painter, 76, 34, line)
        self._draw_smile(painter, QRectF(46, 58, 28, 13), line)
        self._set_fill(painter, secondary, line)
        painter.drawEllipse(QRectF(28, 68, 16, 10))
        painter.drawEllipse(QRectF(76, 68, 16, 10))
        self._draw_badge(painter, accessory, line)

    def _draw_hamster(
        self,
        painter: QPainter,
        body: QColor,
        accent: QColor,
        secondary: QColor,
        accessory: QColor,
        line: QColor,
    ) -> None:
        self._set_fill(painter, body, line)
        painter.drawEllipse(QRectF(24, 38, 24, 24))
        painter.drawEllipse(QRectF(72, 38, 24, 24))
        painter.drawEllipse(QRectF(28, 32, 64, 58))
        self._set_fill(painter, accent, line)
        painter.drawEllipse(QRectF(38, 62, 16, 14))
        painter.drawEllipse(QRectF(66, 62, 16, 14))
        self._set_fill(painter, secondary, line)
        painter.drawEllipse(QRectF(50, 58, 20, 18))
        self._draw_eye(painter, 48, 54, line)
        self._draw_eye(painter, 72, 54, line)
        self._draw_nose(painter, 60, 66, line)
        self._draw_badge(painter, accessory, line)

    def _draw_eye(self, painter: QPainter, x_coord: float, y_coord: float, line: QColor) -> None:
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(line)
        height = 2.0 if self._mood == PetMood.HAPPY else 6.0
        painter.drawEllipse(QRectF(x_coord - 2.5, y_coord - height / 2.0, 5.0, height))

    def _draw_nose(self, painter: QPainter, x_coord: float, y_coord: float, line: QColor) -> None:
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(line)
        painter.drawEllipse(QRectF(x_coord - 3, y_coord - 2, 6, 4))

    def _draw_smile(self, painter: QPainter, rect: QRectF, line: QColor) -> None:
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setPen(QPen(line, 2.0, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        if self._mood == PetMood.TIRED:
            painter.drawArc(rect, 15 * 16, 150 * 16)
        else:
            painter.drawArc(rect, 205 * 16, 130 * 16)

    def _draw_teeth(self, painter: QPainter, line: QColor) -> None:
        painter.setPen(QPen(line, 1.5))
        painter.setBrush(QColor("#FFFFFF"))
        painter.drawPolygon(QPolygonF([QPointF(50, 43), QPointF(54, 43), QPointF(52, 48)]))
        painter.drawPolygon(QPolygonF([QPointF(66, 43), QPointF(70, 43), QPointF(68, 48)]))

    def _draw_badge(self, painter: QPainter, color: QColor, line: QColor) -> None:
        self._set_fill(painter, color, line)
        painter.drawRoundedRect(QRectF(74, 76, 18, 12), 4, 4)

    def _draw_lock_overlay(self, painter: QPainter, line: QColor) -> None:
        painter.setOpacity(0.80)
        painter.setBrush(QColor("#F8FAFC"))
        painter.setPen(QPen(line, 2.0))
        painter.drawRoundedRect(QRectF(44, 42, 32, 28), 6, 6)
        painter.drawArc(QRectF(50, 30, 20, 24), 0, 180 * 16)
        painter.drawEllipse(QRectF(57, 53, 6, 6))

    def _draw_mood_effect(self, painter: QPainter, color: QColor, line: QColor) -> None:
        if self._mood not in {PetMood.HAPPY, PetMood.CELEBRATING, PetMood.LEVEL_UP}:
            return
        painter.setOpacity(0.9)
        painter.setPen(QPen(line, 1.2, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        painter.setBrush(color)
        for index, (x_coord, y_coord) in enumerate(((22, 26), (94, 28), (86, 16))):
            radius = 2.8 + 1.2 * sin(self._phase + index)
            path = QPainterPath(QPointF(x_coord, y_coord - radius))
            path.lineTo(QPointF(x_coord + radius, y_coord))
            path.lineTo(QPointF(x_coord, y_coord + radius))
            path.lineTo(QPointF(x_coord - radius, y_coord))
            path.closeSubpath()
            painter.drawPath(path)

    @staticmethod
    def _qcolor(value: str) -> QColor:
        color = QColor(value)
        return color if color.isValid() else QColor("#94A3B8")

    @staticmethod
    def _muted(color: QColor) -> QColor:
        gray = int((color.red() + color.green() + color.blue()) / 3)
        return QColor(gray, gray, gray, color.alpha())

    @staticmethod
    def _set_fill(painter: QPainter, brush: QColor, line: QColor) -> None:
        painter.setPen(QPen(line, 2.0))
        painter.setBrush(brush)


__all__ = ["PetAvatarWidget"]
