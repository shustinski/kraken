from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QImage, QPainter, QPixmap
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication

from contour.gamification import PetType, Rarity
from contour.gamification.ui.animated_pet_widget import AnimatedPetWidget
from contour.gamification.ui.animation.pet_animation_controller import (
    KrakenAnimationRegistry,
    PetAnimationController,
    PetAnimationState,
)
from contour.gamification.ui.animation.sprite_sheet_player import SpriteSheetAnimation, SpriteSheetPlayer


def _sheet_pixmap(colors: list[QColor], *, frame_width: int = 8, frame_height: int = 6) -> QPixmap:
    image = QImage(frame_width * len(colors), frame_height, QImage.Format.Format_ARGB32)
    image.fill(Qt.GlobalColor.transparent)
    painter = QPainter(image)
    for index, color in enumerate(colors):
        painter.fillRect(index * frame_width, 0, frame_width, frame_height, color)
    painter.end()
    return QPixmap.fromImage(image)


def _animation(colors: list[QColor], *, fps: int = 8, loop: bool = True) -> SpriteSheetAnimation:
    pixmap = _sheet_pixmap(colors)
    return SpriteSheetAnimation(
        pixmap=pixmap,
        frame_count=len(colors),
        frame_width=8,
        frame_height=6,
        fps=fps,
        loop=loop,
    )


def _process_events() -> None:
    app = QApplication.instance()
    if app is not None:
        app.processEvents()


def test_kraken_sprite_sheet_loads_successfully() -> None:
    animation = KrakenAnimationRegistry().get_animation(Rarity.COMMON, PetAnimationState.IDLE)

    assert animation is not None
    assert not animation.pixmap.isNull()
    assert animation.frame_count == 6
    assert animation.fps in {7, 8, 10, 12}


def test_frame_extraction_uses_cached_sheet_frames() -> None:
    player = SpriteSheetPlayer()
    player.set_animation(_animation([QColor("#ff0000"), QColor("#00ff00"), QColor("#0000ff")]))

    assert player.current_frame is not None
    assert player.current_frame.toImage().pixelColor(0, 0) == QColor("#ff0000")
    player.set_frame(2)
    assert player.current_frame is not None
    assert player.current_frame.toImage().pixelColor(0, 0) == QColor("#0000ff")


def test_animation_loops_correctly() -> None:
    player = SpriteSheetPlayer()
    player.set_animation(_animation([QColor("#111111"), QColor("#222222"), QColor("#333333")]))

    player.set_frame(2)
    player.next_frame()

    assert player.current_frame_index == 0


def test_animation_pause_resume_and_hidden_widget_timer() -> None:
    player = SpriteSheetPlayer()
    player.set_animation(_animation([QColor("#111111"), QColor("#222222")]))
    player.show()
    _process_events()

    player.play()
    assert player.timer_active
    player.pause()
    assert player.is_paused
    assert not player.timer_active
    player.resume()
    assert player.timer_active

    player.hide()
    _process_events()
    assert not player.timer_active
    player.show()
    _process_events()
    assert player.timer_active
    player.close()


def test_switching_animation_resets_frame_safely() -> None:
    player = SpriteSheetPlayer()
    player.set_animation(_animation([QColor("#111111"), QColor("#222222"), QColor("#333333")]))
    player.set_frame(2)

    player.set_animation(_animation([QColor("#aaaaaa"), QColor("#bbbbbb")]))

    assert player.current_frame_index == 0
    assert player.current_frame is not None
    assert player.current_frame.toImage().pixelColor(0, 0) == QColor("#aaaaaa")


def test_fallback_works_if_asset_missing(tmp_path: Path) -> None:
    registry = KrakenAnimationRegistry(
        resource_path=":/gamification/missing/kraken_sheet.png",
        fallback_file=tmp_path / "missing.png",
    )
    widget = AnimatedPetWidget(registry=registry)

    widget.set_pet(PetType.KRAKEN, "kraken_default", rarity=Rarity.COMMON)

    assert registry.get_animation(Rarity.COMMON, PetAnimationState.IDLE) is None
    assert widget.is_showing_fallback


def test_controller_returns_to_idle_after_temporary_animation() -> None:
    player = SpriteSheetPlayer()
    controller = PetAnimationController(player)
    controller.set_pet(PetType.KRAKEN, Rarity.COMMON)

    controller.set_state(PetAnimationState.HAPPY, temporary_ms=20)
    assert controller.state == PetAnimationState.HAPPY
    QTest.qWait(60)

    assert controller.state == PetAnimationState.IDLE


def test_invalid_animation_state_does_not_crash() -> None:
    player = SpriteSheetPlayer()
    controller = PetAnimationController(player)

    controller.set_state("not-a-real-state")

    assert controller.state == PetAnimationState.IDLE
