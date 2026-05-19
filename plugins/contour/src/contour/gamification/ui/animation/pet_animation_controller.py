from __future__ import annotations

from collections.abc import Mapping
from enum import StrEnum
from pathlib import Path

from PyQt6.QtCore import QObject, QTimer
from PyQt6.QtGui import QPixmap

from ...assets.resources import ensure_gamification_qt_resources_registered
from ...models import PetType, Rarity
from .sprite_sheet_player import SpriteSheetAnimation, SpriteSheetPlayer

KRAKEN_COMMON_SHEET_RESOURCE = ":/gamification/pets/kraken/common/kraken_sheet.png"
KRAKEN_COMMON_SHEET_PACKAGE_FILE = "assets/pets/kraken/common/kraken_sheet.png"


class PetAnimationState(StrEnum):
    IDLE = "idle"
    HAPPY = "happy"
    FOCUSED = "focused"
    LEVEL_UP = "level_up"
    CELEBRATING = "celebrating"


KRAKEN_ANIMATIONS: dict[tuple[str, str], SpriteSheetAnimation] = {}
_PIXMAP_CACHE: dict[str, QPixmap] = {}
_RESOURCES_CHECKED = False


class KrakenAnimationRegistry:
    """Loads Kraken sprite sheet animations from Qt resources with package fallback.

    To add a new pet:
    1. Add its sprite sheet under ``gamification/assets/pets/<pet>/<rarity>/``.
    2. Add the file to ``gamification/assets/gamification.qrc``.
    3. Create a registry like this one that maps ``(rarity, state)`` to
       ``SpriteSheetAnimation``.

    To tune an animation, adjust ``frame_count``, ``fps``, and optional
    ``start_frame`` / ``end_frame`` ranges. The sheet is assumed horizontal with
    equal-size frames.
    """

    def __init__(
        self,
        *,
        resource_path: str = KRAKEN_COMMON_SHEET_RESOURCE,
        fallback_file: Path | None = None,
        frame_count: int = 6,
    ) -> None:
        self._resource_path = resource_path
        self._fallback_file = fallback_file or self._default_fallback_file()
        self._frame_count = max(1, int(frame_count))
        self._animations: dict[tuple[str, str], SpriteSheetAnimation] | None = None

    def get_animation(self, rarity: Rarity | str | None, state: PetAnimationState | str) -> SpriteSheetAnimation | None:
        rarity_key = self._normalize_rarity(rarity)
        state_key = self._normalize_state(state)
        animations = self.animations()
        return animations.get((rarity_key, state_key.value)) or animations.get((Rarity.COMMON.value, state_key.value))

    def has_animation(self, rarity: Rarity | str | None = Rarity.COMMON) -> bool:
        return self.get_animation(rarity, PetAnimationState.IDLE) is not None

    def animations(self) -> Mapping[tuple[str, str], SpriteSheetAnimation]:
        global KRAKEN_ANIMATIONS
        if self._is_default_registry() and KRAKEN_ANIMATIONS:
            return KRAKEN_ANIMATIONS
        if self._animations is not None:
            return self._animations
        pixmap = self._load_pixmap()
        if pixmap.isNull():
            self._animations = {}
            return {}
        frame_width = pixmap.width() // self._frame_count
        frame_height = pixmap.height()
        common = Rarity.COMMON.value
        animations = {
            (common, PetAnimationState.IDLE.value): SpriteSheetAnimation(
                pixmap=pixmap,
                frame_count=self._frame_count,
                frame_width=frame_width,
                frame_height=frame_height,
                fps=7,
                loop=True,
                start_frame=0,
                end_frame=2,
            ),
            (common, PetAnimationState.FOCUSED.value): SpriteSheetAnimation(
                pixmap=pixmap,
                frame_count=self._frame_count,
                frame_width=frame_width,
                frame_height=frame_height,
                fps=8,
                loop=True,
                start_frame=1,
                end_frame=3,
            ),
            (common, PetAnimationState.HAPPY.value): SpriteSheetAnimation(
                pixmap=pixmap,
                frame_count=self._frame_count,
                frame_width=frame_width,
                frame_height=frame_height,
                fps=12,
                loop=True,
                start_frame=0,
                end_frame=5,
            ),
            (common, PetAnimationState.CELEBRATING.value): SpriteSheetAnimation(
                pixmap=pixmap,
                frame_count=self._frame_count,
                frame_width=frame_width,
                frame_height=frame_height,
                fps=12,
                loop=True,
                start_frame=2,
                end_frame=5,
            ),
            (common, PetAnimationState.LEVEL_UP.value): SpriteSheetAnimation(
                pixmap=pixmap,
                frame_count=self._frame_count,
                frame_width=frame_width,
                frame_height=frame_height,
                fps=10,
                loop=True,
                start_frame=3,
                end_frame=5,
            ),
        }
        if self._is_default_registry():
            KRAKEN_ANIMATIONS = animations
        self._animations = animations
        return animations

    def _load_pixmap(self) -> QPixmap:
        self._ensure_qt_resources()
        cache_key = f"{self._resource_path}|{self._fallback_file}"
        cached = _PIXMAP_CACHE.get(cache_key)
        if cached is not None:
            return cached
        pixmap = QPixmap(self._resource_path)
        if pixmap.isNull() and self._fallback_file.exists():
            pixmap = QPixmap(str(self._fallback_file))
        _PIXMAP_CACHE[cache_key] = pixmap
        return pixmap

    def _is_default_registry(self) -> bool:
        return (
            self._resource_path == KRAKEN_COMMON_SHEET_RESOURCE
            and self._fallback_file == self._default_fallback_file()
            and self._frame_count == 6
        )

    @staticmethod
    def _ensure_qt_resources() -> None:
        global _RESOURCES_CHECKED
        if _RESOURCES_CHECKED:
            return
        ensure_gamification_qt_resources_registered()
        _RESOURCES_CHECKED = True

    @staticmethod
    def _normalize_rarity(rarity: Rarity | str | None) -> str:
        if rarity is None:
            return Rarity.COMMON.value
        try:
            return Rarity(str(rarity)).value
        except ValueError:
            return Rarity.COMMON.value

    @staticmethod
    def _normalize_state(state: PetAnimationState | str) -> PetAnimationState:
        try:
            return PetAnimationState(str(state))
        except ValueError:
            return PetAnimationState.IDLE

    @staticmethod
    def _default_fallback_file() -> Path:
        return Path(__file__).resolve().parents[2] / KRAKEN_COMMON_SHEET_PACKAGE_FILE


class PetAnimationController(QObject):
    """Connects pet/rarity/state selection to a reusable ``SpriteSheetPlayer``."""

    def __init__(
        self,
        player: SpriteSheetPlayer,
        *,
        registry: KrakenAnimationRegistry | None = None,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._player = player
        self._registry = registry or KrakenAnimationRegistry()
        self._pet_type = PetType.KRAKEN
        self._rarity: Rarity | None = Rarity.COMMON
        self._state = PetAnimationState.IDLE
        self._return_timer = QTimer(self)
        self._return_timer.setSingleShot(True)
        self._return_timer.timeout.connect(self.return_to_idle)

    @property
    def state(self) -> PetAnimationState:
        return self._state

    def has_animation(self, pet_type: PetType, rarity: Rarity | None = Rarity.COMMON) -> bool:
        return pet_type == PetType.KRAKEN and self._registry.has_animation(rarity)

    def set_pet(self, pet_type: PetType, rarity: Rarity | None = Rarity.COMMON) -> None:
        self._pet_type = pet_type
        self._rarity = rarity or Rarity.COMMON
        if not self.has_animation(pet_type, self._rarity):
            self._player.set_animation(None)
            return
        self.set_state(self._state)

    def set_state(self, state: PetAnimationState | str, *, temporary_ms: int | None = None) -> None:
        normalized = KrakenAnimationRegistry._normalize_state(state)
        animation = self._registry.get_animation(self._rarity, normalized)
        if animation is None:
            normalized = PetAnimationState.IDLE
            animation = self._registry.get_animation(self._rarity, normalized)
        self._state = normalized
        self._return_timer.stop()
        if animation is None:
            self._player.set_animation(None)
            return
        self._player.set_animation(animation)
        self._player.play()
        if temporary_ms is not None and normalized != PetAnimationState.IDLE:
            self._return_timer.start(max(1, int(temporary_ms)))

    def return_to_idle(self) -> None:
        self.set_state(PetAnimationState.IDLE)


__all__ = [
    "KRAKEN_ANIMATIONS",
    "KRAKEN_COMMON_SHEET_RESOURCE",
    "KrakenAnimationRegistry",
    "PetAnimationController",
    "PetAnimationState",
]
