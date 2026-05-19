from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QImage, QPainter

from contour.gamification import DEFAULT_SKIN_BY_PET, PET_DEFINITIONS, SKIN_DEFINITIONS, PetMood
from contour.gamification.avatar import PetAvatarWidget


def _render_avatar(widget: PetAvatarWidget) -> QImage:
    image = QImage(120, 120, QImage.Format.Format_ARGB32)
    image.fill(Qt.GlobalColor.transparent)
    painter = QPainter(image)
    widget.render(painter)
    painter.end()
    return image


def _has_visible_pixels(image: QImage) -> bool:
    for x_coord in range(0, image.width(), 6):
        for y_coord in range(0, image.height(), 6):
            if image.pixelColor(x_coord, y_coord).alpha() > 0:
                return True
    return False


def test_avatar_renders_every_pet_and_skin() -> None:
    widget = PetAvatarWidget(avatar_size=120, animated=False)
    widget.resize(120, 120)

    for pet_type in PET_DEFINITIONS:
        widget.set_pet(pet_type, DEFAULT_SKIN_BY_PET[pet_type], PetMood.HAPPY)
        assert _has_visible_pixels(_render_avatar(widget))

    for skin_id, definition in SKIN_DEFINITIONS.items():
        widget.set_pet(definition.pet_type, skin_id, PetMood.CELEBRATING)
        assert _has_visible_pixels(_render_avatar(widget))
