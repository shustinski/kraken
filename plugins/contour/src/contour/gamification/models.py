from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class PetType(StrEnum):
    KRAKEN = "kraken"
    CAT = "cat"
    DOG = "dog"
    CAPYBARA = "capybara"
    CARNIVOROUS_PLANT = "carnivorous_plant"
    HORSE = "horse"
    FROG = "frog"
    HAMSTER = "hamster"


class Rarity(StrEnum):
    COMMON = "common"
    RARE = "rare"
    EPIC = "epic"
    LEGENDARY = "legendary"


class PetMood(StrEnum):
    IDLE = "idle"
    HAPPY = "happy"
    FOCUSED = "focused"
    TIRED = "tired"
    HUNGRY = "hungry"
    LEVEL_UP = "level_up"
    CELEBRATING = "celebrating"


class CorrectionType(StrEnum):
    MINOR = "minor"
    MEDIUM = "medium"
    MAJOR = "major"
    EXPERT = "expert"
    UNKNOWN = "unknown"


class RewardEventType(StrEnum):
    IMAGE_VIEWED = "image_viewed"
    IMAGE_ACCEPTED_WITHOUT_CHANGES = "image_accepted_without_changes"
    CORRECTION_REWARDED = "correction_rewarded"
    CURRENCY_EARNED = "currency_earned"
    PET_FRAGMENT_DROPPED = "pet_fragment_dropped"
    PET_FRAGMENT_BOUGHT = "pet_fragment_bought"
    PET_FRAGMENT_EXCHANGED = "pet_fragment_exchanged"
    PET_UNLOCKED = "pet_unlocked"
    PET_UPGRADED = "pet_upgraded"
    PET_SELECTED = "pet_selected"
    SKIN_FRAGMENT_DROPPED = "skin_fragment_dropped"
    SKIN_FRAGMENT_BOUGHT = "skin_fragment_bought"
    SKIN_FRAGMENT_EXCHANGED = "skin_fragment_exchanged"
    SKIN_UNLOCKED = "skin_unlocked"
    SKIN_UPGRADED = "skin_upgraded"
    SKIN_SELECTED = "skin_selected"
    NOT_ENOUGH_CURRENCY = "not_enough_currency"
    NOT_ENOUGH_FRAGMENTS = "not_enough_fragments"


RARITY_ORDER: tuple[Rarity, ...] = (
    Rarity.COMMON,
    Rarity.RARE,
    Rarity.EPIC,
    Rarity.LEGENDARY,
)


def next_rarity(rarity: Rarity) -> Rarity | None:
    try:
        index = RARITY_ORDER.index(rarity)
    except ValueError:
        return None
    next_index = index + 1
    if next_index >= len(RARITY_ORDER):
        return None
    return RARITY_ORDER[next_index]


@dataclass(slots=True)
class PetProgress:
    pet_type: PetType
    unlocked: bool = False
    rarity: Rarity | None = None


@dataclass(slots=True)
class SkinProgress:
    skin_id: str
    pet_type: PetType
    unlocked: bool = False
    rarity: Rarity | None = None
    is_default: bool = False


@dataclass(slots=True)
class PetFragmentInventory:
    fragments_by_pet: dict[PetType, dict[Rarity, int]] = field(default_factory=dict)

    def get(self, pet_type: PetType, rarity: Rarity) -> int:
        return max(0, int(self.fragments_by_pet.get(pet_type, {}).get(rarity, 0)))

    def add(self, pet_type: PetType, rarity: Rarity, amount: int = 1) -> None:
        self.fragments_by_pet.setdefault(pet_type, {})
        current = self.get(pet_type, rarity)
        self.fragments_by_pet[pet_type][rarity] = max(0, current + int(amount))

    def spend(self, pet_type: PetType, rarity: Rarity, amount: int) -> bool:
        amount = int(amount)
        if amount < 0 or self.get(pet_type, rarity) < amount:
            return False
        self.fragments_by_pet.setdefault(pet_type, {})
        self.fragments_by_pet[pet_type][rarity] = self.get(pet_type, rarity) - amount
        return True


@dataclass(slots=True)
class SkinFragmentInventory:
    fragments_by_skin_id: dict[str, dict[Rarity, int]] = field(default_factory=dict)

    def get(self, skin_id: str, rarity: Rarity) -> int:
        return max(0, int(self.fragments_by_skin_id.get(str(skin_id), {}).get(rarity, 0)))

    def add(self, skin_id: str, rarity: Rarity, amount: int = 1) -> None:
        key = str(skin_id)
        self.fragments_by_skin_id.setdefault(key, {})
        current = self.get(key, rarity)
        self.fragments_by_skin_id[key][rarity] = max(0, current + int(amount))

    def spend(self, skin_id: str, rarity: Rarity, amount: int) -> bool:
        amount = int(amount)
        key = str(skin_id)
        if amount < 0 or self.get(key, rarity) < amount:
            return False
        self.fragments_by_skin_id.setdefault(key, {})
        self.fragments_by_skin_id[key][rarity] = self.get(key, rarity) - amount
        return True


@dataclass(slots=True)
class CurrencyTransaction:
    id: str
    amount: int
    reason: str
    created_at: str
    correction_id: str | None = None
    image_id: str | None = None


@dataclass(slots=True)
class PetFragmentTransaction:
    id: str
    pet_type: PetType
    rarity: Rarity
    amount: int
    source: str
    created_at: str
    correction_id: str | None = None


@dataclass(slots=True)
class SkinFragmentTransaction:
    id: str
    skin_id: str
    rarity: Rarity
    amount: int
    source: str
    created_at: str
    correction_id: str | None = None


@dataclass(slots=True)
class PurchaseTransaction:
    id: str
    item_type: str
    price: int
    created_at: str


@dataclass(slots=True)
class CorrectionEvent:
    correction_id: str
    image_id: str
    has_real_mask_changes: bool
    accepted_without_changes: bool = False
    correction_type: CorrectionType | None = None
    view_only: bool = False
    changed_pixels: int | None = None
    changed_area_ratio: float | None = None
    edit_count: int | None = None
    removed_objects: int | None = None
    added_objects: int | None = None
    fixed_breaks: int | None = None
    separated_objects: int | None = None
    duration_seconds: float | None = None

    @property
    def is_eligible(self) -> bool:
        return (
            bool(self.correction_id)
            and bool(self.image_id)
            and bool(self.has_real_mask_changes)
            and not self.accepted_without_changes
            and not self.view_only
        )


@dataclass(slots=True)
class GamificationProfile:
    xp: int
    level: int
    wallet_balance: int
    lifetime_currency_earned: int
    lifetime_currency_spent: int
    selected_pet: PetType
    selected_skin_by_pet: dict[PetType, str]
    pet_progress: dict[PetType, PetProgress]
    skin_progress: dict[str, SkinProgress]
    pet_fragments: PetFragmentInventory
    skin_fragments: SkinFragmentInventory
    rewarded_correction_ids: set[str] = field(default_factory=set)
    pet_drop_attempted_correction_ids: set[str] = field(default_factory=set)
    skin_drop_attempted_correction_ids: set[str] = field(default_factory=set)
    pet_pity_counter: int = 0
    skin_pity_counter: int = 0
    currency_transactions: list[CurrencyTransaction] = field(default_factory=list)
    pet_fragment_transactions: list[PetFragmentTransaction] = field(default_factory=list)
    skin_fragment_transactions: list[SkinFragmentTransaction] = field(default_factory=list)
    purchase_transactions: list[PurchaseTransaction] = field(default_factory=list)


@dataclass(slots=True)
class ServiceResult:
    success: bool
    event_type: RewardEventType
    message: str = ""
    payload: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def ok(
        cls,
        event_type: RewardEventType,
        message: str = "",
        **payload: Any,
    ) -> ServiceResult:
        return cls(True, event_type, message, dict(payload))

    @classmethod
    def fail(
        cls,
        event_type: RewardEventType,
        message: str = "",
        **payload: Any,
    ) -> ServiceResult:
        return cls(False, event_type, message, dict(payload))
