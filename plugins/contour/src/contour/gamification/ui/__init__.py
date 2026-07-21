from __future__ import annotations

from collections.abc import Callable, Mapping
from random import Random
from typing import Any

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from ..avatar import PetAvatarWidget
from ..config import DEFAULT_GAMIFICATION_BALANCE, GamificationBalance
from ..models import GamificationProfile, PetMood, PetType, Rarity, RewardEventType, ServiceResult, next_rarity
from ..registry import (
    COLLECTIBLE_PET_TYPES,
    DEFAULT_SKIN_BY_PET,
    PAID_SKIN_IDS,
    PET_DEFINITIONS,
    SKIN_DEFINITIONS,
)
from ..services import (
    GamificationProfileService,
    PetFragmentExchangeService,
    PetReactionService,
    PetUnlockService,
    SelectionService,
    ShopService,
    SkinFragmentExchangeService,
    SkinUnlockService,
)
from .animated_pet_widget import AnimatedPetWidget


def _rarity_text(rarity: Rarity | None) -> str:
    return "locked" if rarity is None else rarity.value


def _fragment_line(values: Callable[[Rarity], int]) -> str:
    return " / ".join(f"{rarity.value}: {values(rarity)}" for rarity in Rarity)


def _clear_layout(layout: QVBoxLayout | QGridLayout | QFormLayout) -> None:
    while layout.count():
        item = layout.takeAt(0)
        widget = item.widget()
        child_layout = item.layout()
        if widget is not None:
            widget.setParent(None)
            widget.deleteLater()
        elif child_layout is not None:
            _clear_layout(child_layout)  # type: ignore[arg-type]


class GamificationPanel(QWidget):
    messageRequested = pyqtSignal(str)

    def __init__(
        self,
        profile_service: GamificationProfileService,
        *,
        balance: GamificationBalance = DEFAULT_GAMIFICATION_BALANCE,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._profile_service = profile_service
        self._balance = balance
        self._reaction_service = PetReactionService()
        self._last_event_type: RewardEventType | None = None
        self._last_reward = "Пока нет наград"
        self._build_ui()
        self.refresh_profile()

    def set_last_results(self, results: list[ServiceResult]) -> None:
        messages = [result.message for result in results if result.success and result.message]
        successful_results = [result for result in results if result.success]
        if successful_results:
            self._last_event_type = successful_results[-1].event_type
        if messages:
            self._last_reward = " | ".join(messages)
            for message in messages:
                self.messageRequested.emit(message)
        self.refresh_profile()
        if self._last_event_type is not None:
            self.avatar.react_to_event(self._last_event_type)

    def react_to_event(self, event_type: RewardEventType, message: str | None = None) -> None:
        self._last_event_type = event_type
        if message:
            self._last_reward = message
        self.refresh_profile()
        self.avatar.react_to_event(event_type)

    def refresh_profile(self) -> None:
        profile = self._profile_service.load_profile()
        pet = PET_DEFINITIONS.get(profile.selected_pet, PET_DEFINITIONS[PetType.KRAKEN])
        skin_id = profile.selected_skin_by_pet.get(profile.selected_pet, DEFAULT_SKIN_BY_PET[profile.selected_pet])
        skin = SKIN_DEFINITIONS.get(skin_id, SKIN_DEFINITIONS[DEFAULT_SKIN_BY_PET[profile.selected_pet]])
        pet_progress = profile.pet_progress[profile.selected_pet]
        self.avatar.set_pet(profile.selected_pet, skin_id, rarity=pet_progress.rarity)
        self.pet_label.setText(f"{pet.title} / {skin.title}")
        self.rarity_label.setText(f"Редкость: {_rarity_text(pet_progress.rarity)}")
        self.level_label.setText(f"Уровень: {profile.level}   XP: {profile.xp}")
        self.balance_label.setText(f"Фрагменты кристалла: {profile.wallet_balance}")
        self.reaction_label.setText(self._reaction_service.reaction_for_event(profile, self._last_event_type))
        self.last_reward_label.setText(f"Последняя награда: {self._last_reward}")

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        self.title_label = QLabel("Питомец")
        self.title_label.setStyleSheet("font-weight: 700;")
        self.avatar = AnimatedPetWidget(avatar_size=136)
        self.pet_label = QLabel("")
        self.rarity_label = QLabel("")
        self.level_label = QLabel("")
        self.balance_label = QLabel("")
        self.reaction_label = QLabel("")
        self.reaction_label.setWordWrap(True)
        self.last_reward_label = QLabel("")
        self.last_reward_label.setWordWrap(True)
        self.open_button = QPushButton("Коллекция")
        self.open_button.clicked.connect(self._open_dialog)
        for widget in (
            self.title_label,
            self.avatar,
            self.pet_label,
            self.rarity_label,
            self.level_label,
            self.balance_label,
            self.reaction_label,
            self.last_reward_label,
        ):
            layout.addWidget(widget)
        layout.addWidget(self.open_button)
        layout.addStretch(1)

    def _open_dialog(self) -> None:
        dialog = GamificationDialog(self._profile_service, balance=self._balance, parent=self)
        dialog.messageRequested.connect(self.messageRequested.emit)
        dialog.finished.connect(lambda _result: self.refresh_profile())
        dialog.exec()


class GamificationDialog(QDialog):
    messageRequested = pyqtSignal(str)

    def __init__(
        self,
        profile_service: GamificationProfileService,
        *,
        balance: GamificationBalance = DEFAULT_GAMIFICATION_BALANCE,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Геймификация")
        self.resize(820, 680)
        self._profile_service = profile_service
        self._balance = balance
        self._rng = Random()
        self._pet_unlocks = PetUnlockService(balance)
        self._skin_unlocks = SkinUnlockService(balance)
        self._pet_exchange = PetFragmentExchangeService(balance)
        self._skin_exchange = SkinFragmentExchangeService(balance)
        self._shop = ShopService(balance)
        self._selection = SelectionService()
        self._reaction_service = PetReactionService()
        self._last_event_type: RewardEventType | None = None
        self._status_label = QLabel("")
        self._status_label.setWordWrap(True)
        self._build_ui()
        self.refresh()

    def refresh(self) -> None:
        profile = self._profile_service.load_profile()
        self._rebuild_overview(profile)
        self._rebuild_pet_tab(profile)
        self._rebuild_skin_tab(profile)
        self._rebuild_shop_tab(profile)
        self._rebuild_exchange_tab(profile)

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        self.tabs = QTabWidget()
        self.overview_widget = QWidget()
        self.overview_layout = QVBoxLayout(self.overview_widget)
        self.pets_scroll, self.pets_content, self.pets_layout = self._scroll_tab()
        self.skins_scroll, self.skins_content, self.skins_layout = self._scroll_tab()
        self.shop_widget = QWidget()
        self.shop_layout = QVBoxLayout(self.shop_widget)
        self.exchange_scroll, self.exchange_content, self.exchange_layout = self._scroll_tab()
        self.tabs.addTab(self.overview_widget, "Главная")
        self.tabs.addTab(self.pets_scroll, "Питомцы")
        self.tabs.addTab(self.skins_scroll, "Скины")
        self.tabs.addTab(self.shop_widget, "Магазин")
        self.tabs.addTab(self.exchange_scroll, "Обмен")
        layout.addWidget(self.tabs, 1)
        layout.addWidget(self._status_label)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    @staticmethod
    def _scroll_tab() -> tuple[QScrollArea, QWidget, QVBoxLayout]:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        scroll.setWidget(content)
        return scroll, content, layout

    def _save_report_refresh(self, profile: GamificationProfile, result: ServiceResult) -> None:
        if result.success:
            self._profile_service.save_profile(profile)
        self._last_event_type = result.event_type
        self._report(result)
        self.refresh()

    def _report(self, result: ServiceResult) -> None:
        if result.message:
            self._status_label.setText(result.message)
            self.messageRequested.emit(result.message)

    def _rebuild_overview(self, profile: GamificationProfile) -> None:
        _clear_layout(self.overview_layout)
        pet = PET_DEFINITIONS[profile.selected_pet]
        skin_id = profile.selected_skin_by_pet.get(profile.selected_pet, DEFAULT_SKIN_BY_PET[profile.selected_pet])
        skin = SKIN_DEFINITIONS[skin_id]
        avatar = AnimatedPetWidget(avatar_size=170)
        avatar.set_pet(profile.selected_pet, skin_id, rarity=profile.pet_progress[profile.selected_pet].rarity)
        if self._last_event_type is not None:
            avatar.react_to_event(self._last_event_type)
        self.overview_layout.addWidget(avatar, 0, Qt.AlignmentFlag.AlignHCenter)
        reaction = self._reaction_service.reaction_for_event(profile, self._last_event_type)
        for text in (
            f"Текущий питомец: {pet.title}",
            f"Текущий skin: {skin.title}",
            f"Редкость питомца: {_rarity_text(profile.pet_progress[profile.selected_pet].rarity)}",
            f"Уровень: {profile.level}",
            f"XP: {profile.xp}",
            f"Фрагменты кристалла: {profile.wallet_balance}",
            reaction,
        ):
            label = QLabel(text)
            label.setWordWrap(True)
            self.overview_layout.addWidget(label)
        self.overview_layout.addStretch(1)

    def _rebuild_pet_tab(self, profile: GamificationProfile) -> None:
        _clear_layout(self.pets_layout)
        for pet_type in PET_DEFINITIONS:
            self.pets_layout.addWidget(self._pet_card(profile, pet_type))
        self.pets_layout.addStretch(1)

    def _pet_card(self, profile: GamificationProfile, pet_type: PetType) -> QWidget:
        definition = PET_DEFINITIONS[pet_type]
        progress = profile.pet_progress[pet_type]
        group = QGroupBox(definition.title)
        layout = QVBoxLayout(group)
        avatar = PetAvatarWidget(avatar_size=96)
        avatar.set_pet(pet_type, DEFAULT_SKIN_BY_PET[pet_type], PetMood.IDLE, locked=not progress.unlocked)
        layout.addWidget(avatar, 0, Qt.AlignmentFlag.AlignLeft)
        selected = "выбран" if profile.selected_pet == pet_type else ""
        layout.addWidget(QLabel(f"{definition.description} {selected}".strip()))
        layout.addWidget(QLabel(f"Статус: {'unlocked' if progress.unlocked else 'locked'} / {_rarity_text(progress.rarity)}"))
        layout.addWidget(QLabel(f"Фрагменты: {_fragment_line(lambda rarity: profile.pet_fragments.get(pet_type, rarity))}"))
        if not progress.unlocked:
            layout.addWidget(
                QLabel(
                    f"Открытие: {profile.pet_fragments.get(pet_type, Rarity.COMMON)}/"
                    f"{self._balance.pet_common_fragments_required} common"
                )
            )
        elif progress.rarity is not None and next_rarity(progress.rarity) is not None:
            target = next_rarity(progress.rarity)
            assert target is not None
            layout.addWidget(
                QLabel(
                    f"Повышение: direct {profile.pet_fragments.get(pet_type, target)}/"
                    f"{self._balance.direct_upgrade_fragments_required} {target.value}; "
                    f"fallback {profile.pet_fragments.get(pet_type, progress.rarity)}/"
                    f"{self._balance.fallback_upgrade_fragments_required} {progress.rarity.value}"
                )
            )
        row = QHBoxLayout()
        unlock_button = QPushButton("Открыть")
        unlock_button.setEnabled(self._pet_unlocks.can_unlock_pet(profile, pet_type))
        unlock_button.clicked.connect(lambda _checked=False, pet=pet_type: self._unlock_pet(pet))
        row.addWidget(unlock_button)
        method_combo = self._upgrade_method_combo(profile, pet_type, self._pet_unlocks.can_upgrade_pet)
        row.addWidget(method_combo)
        upgrade_button = QPushButton("Повысить")
        upgrade_button.setEnabled(self._pet_unlocks.can_upgrade_pet(profile, pet_type))
        upgrade_button.clicked.connect(
            lambda _checked=False, pet=pet_type, combo=method_combo: self._upgrade_pet(pet, combo.currentData())
        )
        row.addWidget(upgrade_button)
        select_button = QPushButton("Выбрать")
        select_button.setEnabled(progress.unlocked)
        select_button.clicked.connect(lambda _checked=False, pet=pet_type: self._select_pet(pet))
        row.addWidget(select_button)
        layout.addLayout(row)
        return group

    def _rebuild_skin_tab(self, profile: GamificationProfile) -> None:
        _clear_layout(self.skins_layout)
        for skin_id in SKIN_DEFINITIONS:
            self.skins_layout.addWidget(self._skin_card(profile, skin_id))
        self.skins_layout.addStretch(1)

    def _skin_card(self, profile: GamificationProfile, skin_id: str) -> QWidget:
        definition = SKIN_DEFINITIONS[skin_id]
        progress = profile.skin_progress[skin_id]
        pet_progress = profile.pet_progress[definition.pet_type]
        group = QGroupBox(definition.title)
        layout = QVBoxLayout(group)
        avatar = PetAvatarWidget(avatar_size=96)
        avatar.set_pet(
            definition.pet_type,
            skin_id,
            PetMood.IDLE,
            locked=not progress.unlocked or not pet_progress.unlocked,
        )
        layout.addWidget(avatar, 0, Qt.AlignmentFlag.AlignLeft)
        selected_skin = profile.selected_skin_by_pet.get(definition.pet_type)
        selected = "выбран" if selected_skin == skin_id and profile.selected_pet == definition.pet_type else ""
        layout.addWidget(QLabel(f"Питомец: {PET_DEFINITIONS[definition.pet_type].title}. {definition.description} {selected}".strip()))
        layout.addWidget(QLabel(f"Статус питомца: {'unlocked' if pet_progress.unlocked else 'locked'}"))
        layout.addWidget(QLabel(f"Статус skin: {'unlocked' if progress.unlocked else 'locked'} / {_rarity_text(progress.rarity)}"))
        layout.addWidget(QLabel(f"Фрагменты: {_fragment_line(lambda rarity: profile.skin_fragments.get(skin_id, rarity))}"))
        row = QHBoxLayout()
        unlock_button = QPushButton("Открыть")
        unlock_button.setEnabled(self._skin_unlocks.can_unlock_skin(profile, skin_id))
        unlock_button.clicked.connect(lambda _checked=False, skin=skin_id: self._unlock_skin(skin))
        row.addWidget(unlock_button)
        method_combo = self._upgrade_method_combo(profile, skin_id, self._skin_unlocks.can_upgrade_skin)
        row.addWidget(method_combo)
        upgrade_button = QPushButton("Повысить")
        upgrade_button.setEnabled(self._skin_unlocks.can_upgrade_skin(profile, skin_id))
        upgrade_button.clicked.connect(
            lambda _checked=False, skin=skin_id, combo=method_combo: self._upgrade_skin(skin, combo.currentData())
        )
        row.addWidget(upgrade_button)
        select_button = QPushButton("Выбрать")
        select_button.setEnabled(progress.unlocked and pet_progress.unlocked)
        select_button.clicked.connect(lambda _checked=False, skin=skin_id: self._select_skin(skin))
        row.addWidget(select_button)
        layout.addLayout(row)
        return group

    def _upgrade_method_combo(
        self,
        profile: GamificationProfile,
        item_id: PetType | str,
        can_upgrade: Callable[[GamificationProfile, Any, str | None], bool],
    ) -> QComboBox:
        combo = QComboBox()
        direct = can_upgrade(profile, item_id, "direct")
        fallback = can_upgrade(profile, item_id, "fallback")
        if direct:
            combo.addItem("direct", "direct")
        if fallback:
            combo.addItem("fallback", "fallback")
        if not direct and not fallback:
            combo.addItem("-", None)
        combo.setEnabled(direct and fallback)
        return combo

    def _rebuild_shop_tab(self, profile: GamificationProfile) -> None:
        _clear_layout(self.shop_layout)
        items = self._shop.get_available_shop_items(profile)
        self.shop_layout.addWidget(QLabel(f"Баланс: {profile.wallet_balance} фрагментов кристалла"))
        pet_button = QPushButton(f"Купить pet capsule ({items['pet_fragment_capsule_price']})")
        pet_button.clicked.connect(self._buy_pet_capsule)
        self.shop_layout.addWidget(pet_button)
        skin_button = QPushButton(f"Купить skin capsule ({items['skin_fragment_capsule_price']})")
        skin_button.setEnabled(bool(items["skin_fragment_capsule_available"]))
        skin_button.clicked.connect(self._buy_skin_capsule)
        self.shop_layout.addWidget(skin_button)
        self.shop_layout.addStretch(1)

    def _rebuild_exchange_tab(self, profile: GamificationProfile) -> None:
        _clear_layout(self.exchange_layout)
        self.exchange_layout.addWidget(self._pet_exchange_group(profile))
        self.exchange_layout.addWidget(self._skin_exchange_group(profile))
        self.exchange_layout.addStretch(1)

    def _pet_exchange_group(self, profile: GamificationProfile) -> QWidget:
        group = QGroupBox("Обмен фрагментов питомцев")
        layout = QVBoxLayout(group)
        rarity_combo = self._rarity_combo()
        layout.addWidget(rarity_combo)
        spins: dict[PetType, QSpinBox] = {}
        grid = QGridLayout()
        for row, pet_type in enumerate(PET_DEFINITIONS):
            label = QLabel(f"{PET_DEFINITIONS[pet_type].title}: {_fragment_line(lambda rarity, pet=pet_type: profile.pet_fragments.get(pet, rarity))}")
            spin = QSpinBox()
            spin.setRange(0, 1_000_000)
            spins[pet_type] = spin
            grid.addWidget(label, row, 0)
            grid.addWidget(spin, row, 1)
        layout.addLayout(grid)
        target_combo = QComboBox()
        for pet_type in COLLECTIBLE_PET_TYPES:
            target_combo.addItem(PET_DEFINITIONS[pet_type].title, pet_type)
        layout.addWidget(QLabel("Target pet"))
        layout.addWidget(target_combo)
        button = QPushButton("Обменять")
        button.clicked.connect(
            lambda _checked=False: self._exchange_pet_fragments(
                spins,
                rarity_combo.currentData(),
                target_combo.currentData(),
            )
        )
        layout.addWidget(button)
        return group

    def _skin_exchange_group(self, profile: GamificationProfile) -> QWidget:
        group = QGroupBox("Обмен skin fragments")
        layout = QVBoxLayout(group)
        rarity_combo = self._rarity_combo()
        layout.addWidget(rarity_combo)
        spins: dict[str, QSpinBox] = {}
        grid = QGridLayout()
        for row, skin_id in enumerate(PAID_SKIN_IDS):
            label = QLabel(f"{SKIN_DEFINITIONS[skin_id].title}: {_fragment_line(lambda rarity, skin=skin_id: profile.skin_fragments.get(skin, rarity))}")
            spin = QSpinBox()
            spin.setRange(0, 1_000_000)
            spins[skin_id] = spin
            grid.addWidget(label, row, 0)
            grid.addWidget(spin, row, 1)
        layout.addLayout(grid)
        target_combo = QComboBox()
        for skin_id in PAID_SKIN_IDS:
            definition = SKIN_DEFINITIONS[skin_id]
            if profile.pet_progress[definition.pet_type].unlocked:
                target_combo.addItem(definition.title, skin_id)
        layout.addWidget(QLabel("Target skin"))
        layout.addWidget(target_combo)
        button = QPushButton("Обменять")
        button.setEnabled(target_combo.count() > 0)
        button.clicked.connect(
            lambda _checked=False: self._exchange_skin_fragments(
                spins,
                rarity_combo.currentData(),
                target_combo.currentData(),
            )
        )
        layout.addWidget(button)
        return group

    @staticmethod
    def _rarity_combo() -> QComboBox:
        combo = QComboBox()
        for rarity in Rarity:
            combo.addItem(rarity.value, rarity)
        return combo

    def _unlock_pet(self, pet_type: PetType) -> None:
        profile = self._profile_service.load_profile()
        self._save_report_refresh(profile, self._pet_unlocks.unlock_pet(profile, pet_type))

    def _upgrade_pet(self, pet_type: PetType, method: str | None) -> None:
        profile = self._profile_service.load_profile()
        self._save_report_refresh(profile, self._pet_unlocks.upgrade_pet(profile, pet_type, method))

    def _select_pet(self, pet_type: PetType) -> None:
        profile = self._profile_service.load_profile()
        self._save_report_refresh(profile, self._selection.select_pet(profile, pet_type))

    def _unlock_skin(self, skin_id: str) -> None:
        profile = self._profile_service.load_profile()
        self._save_report_refresh(profile, self._skin_unlocks.unlock_skin(profile, skin_id))

    def _upgrade_skin(self, skin_id: str, method: str | None) -> None:
        profile = self._profile_service.load_profile()
        self._save_report_refresh(profile, self._skin_unlocks.upgrade_skin(profile, skin_id, method))

    def _select_skin(self, skin_id: str) -> None:
        profile = self._profile_service.load_profile()
        self._save_report_refresh(profile, self._selection.select_skin(profile, skin_id))

    def _buy_pet_capsule(self) -> None:
        profile = self._profile_service.load_profile()
        self._save_report_refresh(profile, self._shop.buy_random_common_pet_fragment_capsule(profile, self._rng))

    def _buy_skin_capsule(self) -> None:
        profile = self._profile_service.load_profile()
        self._save_report_refresh(profile, self._shop.buy_random_common_skin_fragment_capsule(profile, self._rng))

    def _exchange_pet_fragments(
        self,
        spins: Mapping[PetType, QSpinBox],
        rarity: Rarity,
        target_pet: PetType,
    ) -> None:
        profile = self._profile_service.load_profile()
        sources = {pet: spin.value() for pet, spin in spins.items() if spin.value()}
        self._save_report_refresh(profile, self._pet_exchange.exchange_pet_fragments(profile, sources, rarity, target_pet))

    def _exchange_skin_fragments(
        self,
        spins: Mapping[str, QSpinBox],
        rarity: Rarity,
        target_skin_id: str | None,
    ) -> None:
        profile = self._profile_service.load_profile()
        if target_skin_id is None:
            self._report(ServiceResult.fail(RewardEventType.NOT_ENOUGH_FRAGMENTS, "Нет доступного target skin"))
            return
        sources = {skin_id: spin.value() for skin_id, spin in spins.items() if spin.value()}
        self._save_report_refresh(
            profile,
            self._skin_exchange.exchange_skin_fragments(profile, sources, rarity, target_skin_id),
        )


__all__ = ["AnimatedPetWidget", "GamificationDialog", "GamificationPanel"]
