from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import QGraphicsDropShadowEffect, QLabel, QStackedWidget, QVBoxLayout, QWidget

from ..avatar import PetAvatarWidget
from ..models import PetMood, PetType, Rarity, RewardEventType
from ..registry import DEFAULT_SKIN_BY_PET
from .animation.pet_animation_controller import KrakenAnimationRegistry, PetAnimationController, PetAnimationState
from .animation.sprite_sheet_player import SpriteSheetPlayer


class AnimatedPetWidget(QWidget):
    """Panel widget that hosts animated pets without coupling to game rewards.

    Kraken uses ``SpriteSheetPlayer``. Other pets currently use the lightweight
    static vector placeholder. To add another animated pet, create a registry
    like ``KrakenAnimationRegistry`` and route that pet type in ``set_pet``.
    """

    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        avatar_size: int = 132,
        registry: KrakenAnimationRegistry | None = None,
    ) -> None:
        super().__init__(parent)
        self._pet_type = PetType.KRAKEN
        self._rarity: Rarity | None = Rarity.COMMON
        self._avatar_size = max(48, int(avatar_size))
        self._player = SpriteSheetPlayer(scaled_size=self._avatar_size)
        self._controller = PetAnimationController(self._player, registry=registry, parent=self)
        self._static_avatar = PetAvatarWidget(avatar_size=self._avatar_size, animated=False)
        self._fallback_label = QLabel("Kraken animation missing")
        self._fallback_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._fallback_label.setWordWrap(True)
        self._rarity_badge = QLabel("")
        self._rarity_badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._stack = QStackedWidget()
        self._stack.addWidget(self._player)
        self._stack.addWidget(self._static_avatar)
        self._stack.addWidget(self._fallback_label)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 8)
        layout.setSpacing(6)
        layout.addWidget(self._stack, 1)
        layout.addWidget(self._rarity_badge)
        self.setMinimumSize(self._avatar_size + 24, self._avatar_size + 42)
        self._apply_rarity_style()

    @property
    def controller(self) -> PetAnimationController:
        return self._controller

    @property
    def player(self) -> SpriteSheetPlayer:
        return self._player

    @property
    def is_showing_fallback(self) -> bool:
        return self._stack.currentWidget() is self._fallback_label

    def set_pet(
        self,
        pet_type: PetType,
        skin_id: str | None = None,
        mood: PetMood | None = None,
        *,
        rarity: Rarity | None = None,
        locked: bool = False,
    ) -> None:
        self._pet_type = pet_type
        self._rarity = rarity
        skin_key = skin_id or DEFAULT_SKIN_BY_PET[pet_type]
        self._rarity_badge.setText("locked" if locked or rarity is None else rarity.value)
        self._apply_rarity_style()
        if pet_type == PetType.KRAKEN:
            if self._controller.has_animation(pet_type, rarity):
                self._stack.setCurrentWidget(self._player)
                self._controller.set_pet(pet_type, rarity)
                if mood is not None:
                    self.set_animation_state(self._state_for_mood(mood))
                return
            self._stack.setCurrentWidget(self._fallback_label)
            return
        self._static_avatar.set_pet(pet_type, skin_key, mood or PetMood.IDLE, locked=locked)
        self._stack.setCurrentWidget(self._static_avatar)

    def react_to_event(self, event_type: RewardEventType | None) -> None:
        if self._pet_type != PetType.KRAKEN:
            self._static_avatar.react_to_event(event_type)
            return
        state, duration_ms = self._state_for_event(event_type)
        self._controller.set_state(state, temporary_ms=duration_ms)

    def set_animation_state(self, state: PetAnimationState | str, *, temporary_ms: int | None = None) -> None:
        if self._pet_type == PetType.KRAKEN:
            self._controller.set_state(state, temporary_ms=temporary_ms)

    @staticmethod
    def _state_for_event(event_type: RewardEventType | None) -> tuple[PetAnimationState, int | None]:
        if event_type in {RewardEventType.CORRECTION_REWARDED, RewardEventType.CURRENCY_EARNED}:
            return PetAnimationState.HAPPY, 2500
        if event_type in {RewardEventType.PET_FRAGMENT_DROPPED, RewardEventType.SKIN_FRAGMENT_DROPPED}:
            return PetAnimationState.CELEBRATING, 3000
        if event_type in {RewardEventType.PET_UPGRADED, RewardEventType.SKIN_UPGRADED}:
            return PetAnimationState.LEVEL_UP, 3000
        if event_type in {RewardEventType.PET_UNLOCKED, RewardEventType.SKIN_UNLOCKED}:
            return PetAnimationState.CELEBRATING, 3000
        if event_type in {RewardEventType.PET_SELECTED, RewardEventType.SKIN_SELECTED, RewardEventType.IMAGE_VIEWED}:
            return PetAnimationState.FOCUSED, 1800
        return PetAnimationState.IDLE, None

    @staticmethod
    def _state_for_mood(mood: PetMood) -> PetAnimationState:
        if mood == PetMood.HAPPY:
            return PetAnimationState.HAPPY
        if mood == PetMood.FOCUSED:
            return PetAnimationState.FOCUSED
        if mood == PetMood.LEVEL_UP:
            return PetAnimationState.LEVEL_UP
        if mood == PetMood.CELEBRATING:
            return PetAnimationState.CELEBRATING
        return PetAnimationState.IDLE

    def _apply_rarity_style(self) -> None:
        color = self._glow_color(self._rarity)
        border = color.name() if color.alpha() else "#D1D5DB"
        self.setStyleSheet(
            "AnimatedPetWidget {"
            "background: #F8FAFC;"
            f"border: 1px solid {border};"
            "border-radius: 8px;"
            "}"
            "QLabel { color: #334155; }"
        )
        effect = QGraphicsDropShadowEffect(self)
        effect.setBlurRadius(0 if color.alpha() == 0 else 14)
        effect.setOffset(0, 2)
        effect.setColor(color if color.alpha() else QColor(15, 23, 42, 28))
        self.setGraphicsEffect(effect)
        self._rarity_badge.setStyleSheet("font-size: 11px; font-weight: 600; color: #475569;")

    @staticmethod
    def _glow_color(rarity: Rarity | None) -> QColor:
        if rarity == Rarity.RARE:
            return QColor(59, 130, 246, 110)
        if rarity == Rarity.EPIC:
            return QColor(147, 51, 234, 105)
        if rarity == Rarity.LEGENDARY:
            return QColor(234, 179, 8, 120)
        return QColor(0, 0, 0, 0)


__all__ = ["AnimatedPetWidget"]
