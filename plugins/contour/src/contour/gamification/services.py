from __future__ import annotations

import uuid
from collections.abc import Mapping, MutableMapping
from dataclasses import asdict
from datetime import UTC, datetime
from random import Random
from typing import Any, Protocol

from .config import DEFAULT_GAMIFICATION_BALANCE, GamificationBalance
from .models import (
    RARITY_ORDER,
    CorrectionEvent,
    CorrectionType,
    CurrencyTransaction,
    GamificationProfile,
    PetFragmentInventory,
    PetFragmentTransaction,
    PetMood,
    PetProgress,
    PetType,
    PurchaseTransaction,
    Rarity,
    RewardEventType,
    ServiceResult,
    SkinFragmentInventory,
    SkinFragmentTransaction,
    SkinProgress,
    next_rarity,
)
from .registry import (
    COLLECTIBLE_PET_TYPES,
    DEFAULT_SKIN_BY_PET,
    PAID_SKIN_IDS_BY_PET,
    PET_DEFINITIONS,
    SKIN_DEFINITIONS,
)
from .visuals import mood_for_reward_event


class GamificationPayloadStore(Protocol):
    def load_payload(self) -> dict[str, Any] | None: ...

    def save_payload(self, payload: dict[str, Any]) -> None: ...


class InMemoryGamificationPayloadStore:
    def __init__(self, payload: dict[str, Any] | None = None) -> None:
        self.payload = payload

    def load_payload(self) -> dict[str, Any] | None:
        return self.payload

    def save_payload(self, payload: dict[str, Any]) -> None:
        self.payload = payload


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _new_transaction_id() -> str:
    return uuid.uuid4().hex


def _enum_value(value: Any) -> str:
    return getattr(value, "value", str(value))


def _parse_pet_type(value: Any, default: PetType | None = None) -> PetType | None:
    if isinstance(value, PetType):
        return value
    try:
        return PetType(str(value))
    except ValueError:
        return default


def _parse_rarity(value: Any, default: Rarity | None = None) -> Rarity | None:
    if isinstance(value, Rarity):
        return value
    try:
        return Rarity(str(value))
    except ValueError:
        return default


def _parse_correction_type(value: Any, default: CorrectionType = CorrectionType.UNKNOWN) -> CorrectionType:
    if isinstance(value, CorrectionType):
        return value
    try:
        return CorrectionType(str(value))
    except ValueError:
        return default


def _clamped_int(value: Any, default: int = 0) -> int:
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return max(0, default)


def _profile_level_for_xp(xp: int, balance: GamificationBalance) -> int:
    xp_per_level = max(1, int(balance.xp_per_level))
    return max(1, int(xp) // xp_per_level + 1)


class GamificationProfileService:
    def __init__(
        self,
        store: GamificationPayloadStore | None = None,
        balance: GamificationBalance = DEFAULT_GAMIFICATION_BALANCE,
    ) -> None:
        self._store = store or InMemoryGamificationPayloadStore()
        self._balance = balance

    def create_default_profile(self) -> GamificationProfile:
        pet_progress = {
            pet_type: PetProgress(
                pet_type=pet_type,
                unlocked=pet_type == PetType.KRAKEN,
                rarity=Rarity.COMMON if pet_type == PetType.KRAKEN else None,
            )
            for pet_type in PET_DEFINITIONS
        }
        skin_progress: dict[str, SkinProgress] = {}
        for skin_id, skin in SKIN_DEFINITIONS.items():
            unlocked = skin_id == DEFAULT_SKIN_BY_PET[PetType.KRAKEN]
            skin_progress[skin_id] = SkinProgress(
                skin_id=skin_id,
                pet_type=skin.pet_type,
                unlocked=unlocked,
                rarity=Rarity.COMMON if unlocked else None,
                is_default=skin.is_default,
            )
        return GamificationProfile(
            xp=0,
            level=1,
            wallet_balance=0,
            lifetime_currency_earned=0,
            lifetime_currency_spent=0,
            selected_pet=PetType.KRAKEN,
            selected_skin_by_pet={pet_type: DEFAULT_SKIN_BY_PET[pet_type] for pet_type in PET_DEFINITIONS},
            pet_progress=pet_progress,
            skin_progress=skin_progress,
            pet_fragments=self._default_pet_inventory(),
            skin_fragments=self._default_skin_inventory(),
            pet_pity_counter=0,
            skin_pity_counter=0,
        )

    def load_profile(self) -> GamificationProfile:
        payload = self._store.load_payload()
        if not payload:
            profile = self.create_default_profile()
            self.save_profile(profile)
            return profile
        profile = self._profile_from_payload(payload)
        return self.repair_profile(profile)

    def save_profile(self, profile: GamificationProfile) -> None:
        self._store.save_payload(self.profile_to_payload(self.repair_profile(profile)))

    def repair_profile(self, profile: GamificationProfile) -> GamificationProfile:
        profile.xp = _clamped_int(profile.xp)
        profile.level = max(1, _clamped_int(profile.level, 1))
        profile.wallet_balance = _clamped_int(profile.wallet_balance)
        profile.lifetime_currency_earned = _clamped_int(profile.lifetime_currency_earned)
        profile.lifetime_currency_spent = _clamped_int(profile.lifetime_currency_spent)
        profile.pet_pity_counter = _clamped_int(profile.pet_pity_counter)
        profile.skin_pity_counter = _clamped_int(profile.skin_pity_counter)

        for pet_type in PET_DEFINITIONS:
            progress = profile.pet_progress.get(pet_type)
            if progress is None:
                progress = PetProgress(pet_type=pet_type)
                profile.pet_progress[pet_type] = progress
            progress.pet_type = pet_type
            if pet_type == PetType.KRAKEN:
                progress.unlocked = True
                progress.rarity = progress.rarity or Rarity.COMMON
            if not progress.unlocked:
                progress.rarity = None
            elif progress.rarity not in RARITY_ORDER:
                progress.rarity = Rarity.COMMON

        for skin_id, definition in SKIN_DEFINITIONS.items():
            progress = profile.skin_progress.get(skin_id)
            if progress is None:
                progress = SkinProgress(
                    skin_id=skin_id,
                    pet_type=definition.pet_type,
                    is_default=definition.is_default,
                )
                profile.skin_progress[skin_id] = progress
            progress.skin_id = skin_id
            progress.pet_type = definition.pet_type
            progress.is_default = definition.is_default
            if not profile.pet_progress[definition.pet_type].unlocked:
                progress.unlocked = False
                progress.rarity = None
            if definition.is_default and profile.pet_progress[definition.pet_type].unlocked:
                progress.unlocked = True
                progress.rarity = progress.rarity or Rarity.COMMON
            if not progress.unlocked:
                progress.rarity = None
            elif progress.rarity not in RARITY_ORDER:
                progress.rarity = Rarity.COMMON

        self._repair_pet_inventory(profile.pet_fragments.fragments_by_pet)
        self._repair_skin_inventory(profile.skin_fragments.fragments_by_skin_id)

        if profile.selected_pet not in PET_DEFINITIONS or not profile.pet_progress[profile.selected_pet].unlocked:
            profile.selected_pet = PetType.KRAKEN
        for pet_type in PET_DEFINITIONS:
            selected_skin = profile.selected_skin_by_pet.get(pet_type)
            default_skin = DEFAULT_SKIN_BY_PET[pet_type]
            if selected_skin not in SKIN_DEFINITIONS:
                profile.selected_skin_by_pet[pet_type] = default_skin
                continue
            selected_progress = profile.skin_progress.get(selected_skin)
            if (
                selected_progress is None
                or selected_progress.pet_type != pet_type
                or not selected_progress.unlocked
            ):
                profile.selected_skin_by_pet[pet_type] = default_skin
        selected_skin = profile.selected_skin_by_pet.get(profile.selected_pet, DEFAULT_SKIN_BY_PET[profile.selected_pet])
        if not profile.skin_progress[selected_skin].unlocked:
            profile.selected_skin_by_pet[profile.selected_pet] = DEFAULT_SKIN_BY_PET[profile.selected_pet]
        profile.level = max(profile.level, _profile_level_for_xp(profile.xp, self._balance))
        return profile

    def profile_to_payload(self, profile: GamificationProfile) -> dict[str, Any]:
        return {
            "version": 1,
            "xp": profile.xp,
            "level": profile.level,
            "wallet_balance": profile.wallet_balance,
            "lifetime_currency_earned": profile.lifetime_currency_earned,
            "lifetime_currency_spent": profile.lifetime_currency_spent,
            "selected_pet": profile.selected_pet.value,
            "selected_skin_by_pet": {pet.value: skin_id for pet, skin_id in profile.selected_skin_by_pet.items()},
            "pet_progress": {
                pet.value: {
                    "pet_type": progress.pet_type.value,
                    "unlocked": progress.unlocked,
                    "rarity": None if progress.rarity is None else progress.rarity.value,
                }
                for pet, progress in profile.pet_progress.items()
            },
            "skin_progress": {
                skin_id: {
                    "skin_id": progress.skin_id,
                    "pet_type": progress.pet_type.value,
                    "unlocked": progress.unlocked,
                    "rarity": None if progress.rarity is None else progress.rarity.value,
                    "is_default": progress.is_default,
                }
                for skin_id, progress in profile.skin_progress.items()
            },
            "pet_fragments": {
                pet.value: {rarity.value: count for rarity, count in fragments.items()}
                for pet, fragments in profile.pet_fragments.fragments_by_pet.items()
            },
            "skin_fragments": {
                skin_id: {rarity.value: count for rarity, count in fragments.items()}
                for skin_id, fragments in profile.skin_fragments.fragments_by_skin_id.items()
            },
            "rewarded_correction_ids": sorted(profile.rewarded_correction_ids),
            "pet_drop_attempted_correction_ids": sorted(profile.pet_drop_attempted_correction_ids),
            "skin_drop_attempted_correction_ids": sorted(profile.skin_drop_attempted_correction_ids),
            "pet_pity_counter": profile.pet_pity_counter,
            "skin_pity_counter": profile.skin_pity_counter,
            "currency_transactions": [asdict(transaction) for transaction in profile.currency_transactions],
            "pet_fragment_transactions": [
                {
                    **asdict(transaction),
                    "pet_type": transaction.pet_type.value,
                    "rarity": transaction.rarity.value,
                }
                for transaction in profile.pet_fragment_transactions
            ],
            "skin_fragment_transactions": [
                {
                    **asdict(transaction),
                    "rarity": transaction.rarity.value,
                }
                for transaction in profile.skin_fragment_transactions
            ],
            "purchase_transactions": [asdict(transaction) for transaction in profile.purchase_transactions],
        }

    def _profile_from_payload(self, payload: Mapping[str, Any]) -> GamificationProfile:
        profile = self.create_default_profile()
        profile.xp = _clamped_int(payload.get("xp"))
        profile.level = max(1, _clamped_int(payload.get("level"), 1))
        profile.wallet_balance = _clamped_int(payload.get("wallet_balance"))
        profile.lifetime_currency_earned = _clamped_int(payload.get("lifetime_currency_earned"))
        profile.lifetime_currency_spent = _clamped_int(payload.get("lifetime_currency_spent"))
        profile.selected_pet = _parse_pet_type(payload.get("selected_pet"), PetType.KRAKEN) or PetType.KRAKEN

        selected_skin_payload = payload.get("selected_skin_by_pet")
        if isinstance(selected_skin_payload, Mapping):
            for raw_pet, raw_skin in selected_skin_payload.items():
                pet = _parse_pet_type(raw_pet)
                if pet is not None:
                    profile.selected_skin_by_pet[pet] = str(raw_skin)

        pet_progress_payload = payload.get("pet_progress")
        if isinstance(pet_progress_payload, Mapping):
            for raw_pet, raw_progress in pet_progress_payload.items():
                pet = _parse_pet_type(raw_pet)
                if pet is None or not isinstance(raw_progress, Mapping):
                    continue
                profile.pet_progress[pet] = PetProgress(
                    pet_type=pet,
                    unlocked=bool(raw_progress.get("unlocked", False)),
                    rarity=_parse_rarity(raw_progress.get("rarity")),
                )

        skin_progress_payload = payload.get("skin_progress")
        if isinstance(skin_progress_payload, Mapping):
            for skin_id, raw_progress in skin_progress_payload.items():
                skin_key = str(skin_id)
                definition = SKIN_DEFINITIONS.get(skin_key)
                if definition is None or not isinstance(raw_progress, Mapping):
                    continue
                profile.skin_progress[skin_key] = SkinProgress(
                    skin_id=skin_key,
                    pet_type=definition.pet_type,
                    unlocked=bool(raw_progress.get("unlocked", False)),
                    rarity=_parse_rarity(raw_progress.get("rarity")),
                    is_default=definition.is_default,
                )

        profile.pet_fragments = self._pet_inventory_from_payload(payload.get("pet_fragments"))
        profile.skin_fragments = self._skin_inventory_from_payload(payload.get("skin_fragments"))
        profile.rewarded_correction_ids = {str(item) for item in payload.get("rewarded_correction_ids", [])}
        profile.pet_drop_attempted_correction_ids = {
            str(item) for item in payload.get("pet_drop_attempted_correction_ids", [])
        }
        profile.skin_drop_attempted_correction_ids = {
            str(item) for item in payload.get("skin_drop_attempted_correction_ids", [])
        }
        profile.pet_pity_counter = _clamped_int(payload.get("pet_pity_counter"))
        profile.skin_pity_counter = _clamped_int(payload.get("skin_pity_counter"))
        profile.currency_transactions = self._currency_transactions_from_payload(
            payload.get("currency_transactions")
        )
        profile.pet_fragment_transactions = self._pet_fragment_transactions_from_payload(
            payload.get("pet_fragment_transactions")
        )
        profile.skin_fragment_transactions = self._skin_fragment_transactions_from_payload(
            payload.get("skin_fragment_transactions")
        )
        profile.purchase_transactions = self._purchase_transactions_from_payload(
            payload.get("purchase_transactions")
        )
        return profile

    def _default_pet_inventory(self) -> PetFragmentInventory:
        return PetFragmentInventory(
            {pet_type: {rarity: 0 for rarity in RARITY_ORDER} for pet_type in PET_DEFINITIONS}
        )

    def _default_skin_inventory(self) -> SkinFragmentInventory:
        return SkinFragmentInventory(
            {skin_id: {rarity: 0 for rarity in RARITY_ORDER} for skin_id in SKIN_DEFINITIONS}
        )

    def _pet_inventory_from_payload(self, raw: Any) -> PetFragmentInventory:
        inventory = self._default_pet_inventory()
        if isinstance(raw, Mapping):
            for raw_pet, raw_fragments in raw.items():
                pet = _parse_pet_type(raw_pet)
                if pet is None or not isinstance(raw_fragments, Mapping):
                    continue
                for raw_rarity, raw_count in raw_fragments.items():
                    rarity = _parse_rarity(raw_rarity)
                    if rarity is not None:
                        inventory.fragments_by_pet[pet][rarity] = _clamped_int(raw_count)
        return inventory

    def _skin_inventory_from_payload(self, raw: Any) -> SkinFragmentInventory:
        inventory = self._default_skin_inventory()
        if isinstance(raw, Mapping):
            for raw_skin, raw_fragments in raw.items():
                skin_id = str(raw_skin)
                if skin_id not in SKIN_DEFINITIONS or not isinstance(raw_fragments, Mapping):
                    continue
                for raw_rarity, raw_count in raw_fragments.items():
                    rarity = _parse_rarity(raw_rarity)
                    if rarity is not None:
                        inventory.fragments_by_skin_id[skin_id][rarity] = _clamped_int(raw_count)
        return inventory

    @staticmethod
    def _currency_transactions_from_payload(raw: Any) -> list[CurrencyTransaction]:
        transactions: list[CurrencyTransaction] = []
        if not isinstance(raw, list):
            return transactions
        for item in raw:
            if not isinstance(item, Mapping):
                continue
            transactions.append(
                CurrencyTransaction(
                    id=str(item.get("id") or _new_transaction_id()),
                    correction_id=None if item.get("correction_id") is None else str(item.get("correction_id")),
                    image_id=None if item.get("image_id") is None else str(item.get("image_id")),
                    amount=int(item.get("amount") or 0),
                    reason=str(item.get("reason") or ""),
                    created_at=str(item.get("created_at") or ""),
                )
            )
        return transactions

    @staticmethod
    def _pet_fragment_transactions_from_payload(raw: Any) -> list[PetFragmentTransaction]:
        transactions: list[PetFragmentTransaction] = []
        if not isinstance(raw, list):
            return transactions
        for item in raw:
            if not isinstance(item, Mapping):
                continue
            pet_type = _parse_pet_type(item.get("pet_type"))
            rarity = _parse_rarity(item.get("rarity"))
            if pet_type is None or rarity is None:
                continue
            transactions.append(
                PetFragmentTransaction(
                    id=str(item.get("id") or _new_transaction_id()),
                    correction_id=None if item.get("correction_id") is None else str(item.get("correction_id")),
                    pet_type=pet_type,
                    rarity=rarity,
                    amount=int(item.get("amount") or 0),
                    source=str(item.get("source") or ""),
                    created_at=str(item.get("created_at") or ""),
                )
            )
        return transactions

    @staticmethod
    def _skin_fragment_transactions_from_payload(raw: Any) -> list[SkinFragmentTransaction]:
        transactions: list[SkinFragmentTransaction] = []
        if not isinstance(raw, list):
            return transactions
        for item in raw:
            if not isinstance(item, Mapping):
                continue
            skin_id = str(item.get("skin_id") or "")
            rarity = _parse_rarity(item.get("rarity"))
            if skin_id not in SKIN_DEFINITIONS or rarity is None:
                continue
            transactions.append(
                SkinFragmentTransaction(
                    id=str(item.get("id") or _new_transaction_id()),
                    correction_id=None if item.get("correction_id") is None else str(item.get("correction_id")),
                    skin_id=skin_id,
                    rarity=rarity,
                    amount=int(item.get("amount") or 0),
                    source=str(item.get("source") or ""),
                    created_at=str(item.get("created_at") or ""),
                )
            )
        return transactions

    @staticmethod
    def _purchase_transactions_from_payload(raw: Any) -> list[PurchaseTransaction]:
        transactions: list[PurchaseTransaction] = []
        if not isinstance(raw, list):
            return transactions
        for item in raw:
            if not isinstance(item, Mapping):
                continue
            transactions.append(
                PurchaseTransaction(
                    id=str(item.get("id") or _new_transaction_id()),
                    item_type=str(item.get("item_type") or ""),
                    price=int(item.get("price") or 0),
                    created_at=str(item.get("created_at") or ""),
                )
            )
        return transactions

    @staticmethod
    def _repair_pet_inventory(raw: MutableMapping[PetType, dict[Rarity, int]]) -> None:
        for pet_type in PET_DEFINITIONS:
            raw.setdefault(pet_type, {})
            for rarity in RARITY_ORDER:
                raw[pet_type][rarity] = _clamped_int(raw[pet_type].get(rarity))

    @staticmethod
    def _repair_skin_inventory(raw: MutableMapping[str, dict[Rarity, int]]) -> None:
        for skin_id in SKIN_DEFINITIONS:
            raw.setdefault(skin_id, {})
            for rarity in RARITY_ORDER:
                raw[skin_id][rarity] = _clamped_int(raw[skin_id].get(rarity))


class CorrectionEventClassifier:
    def classify(self, correction_event: CorrectionEvent) -> CorrectionType:
        if correction_event.correction_type is not None:
            return _parse_correction_type(correction_event.correction_type)
        if correction_event.has_real_mask_changes:
            return CorrectionType.UNKNOWN
        return CorrectionType.UNKNOWN


class CurrencyRewardService:
    def __init__(
        self,
        balance: GamificationBalance = DEFAULT_GAMIFICATION_BALANCE,
        classifier: CorrectionEventClassifier | None = None,
    ) -> None:
        self._balance = balance
        self._classifier = classifier or CorrectionEventClassifier()

    def calculate_currency_reward(self, correction_event: CorrectionEvent) -> int:
        if not correction_event.is_eligible:
            return 0
        correction_type = self._classifier.classify(correction_event)
        return max(0, int(self._balance.currency_reward.get(correction_type, 0)))

    def ensure_no_duplicate_reward(self, profile: GamificationProfile, correction_id: str) -> bool:
        return str(correction_id) not in profile.rewarded_correction_ids

    def add_currency_for_correction(
        self,
        profile: GamificationProfile,
        correction_event: CorrectionEvent,
    ) -> ServiceResult:
        if not correction_event.is_eligible:
            return ServiceResult.fail(RewardEventType.IMAGE_ACCEPTED_WITHOUT_CHANGES)
        if not self.ensure_no_duplicate_reward(profile, correction_event.correction_id):
            return ServiceResult.fail(RewardEventType.CORRECTION_REWARDED, "Награда за это исправление уже начислена")
        amount = self.calculate_currency_reward(correction_event)
        if amount <= 0:
            profile.rewarded_correction_ids.add(correction_event.correction_id)
            return ServiceResult.fail(RewardEventType.CURRENCY_EARNED)
        profile.wallet_balance += amount
        profile.lifetime_currency_earned += amount
        profile.xp += amount * max(1, int(self._balance.xp_per_currency))
        profile.level = _profile_level_for_xp(profile.xp, self._balance)
        profile.rewarded_correction_ids.add(correction_event.correction_id)
        profile.currency_transactions.append(
            CurrencyTransaction(
                id=_new_transaction_id(),
                correction_id=correction_event.correction_id,
                image_id=correction_event.image_id,
                amount=amount,
                reason="correction",
                created_at=_now_iso(),
            )
        )
        return ServiceResult.ok(
            RewardEventType.CURRENCY_EARNED,
            f"+{amount} фрагмента кристалла",
            amount=amount,
        )


class _RarityRoller:
    def __init__(self, balance: GamificationBalance) -> None:
        self._balance = balance

    def roll(self, rng: Random) -> Rarity:
        value = float(rng.random())
        total = 0.0
        for rarity in RARITY_ORDER:
            total += float(self._balance.rarity_distribution.get(rarity, 0.0))
            if value <= total:
                return rarity
        return Rarity.COMMON


class PetFragmentDropService:
    def __init__(
        self,
        balance: GamificationBalance = DEFAULT_GAMIFICATION_BALANCE,
        classifier: CorrectionEventClassifier | None = None,
    ) -> None:
        self._balance = balance
        self._classifier = classifier or CorrectionEventClassifier()
        self._rarity_roller = _RarityRoller(balance)

    def should_attempt_drop(self, profile: GamificationProfile, correction_event: CorrectionEvent) -> bool:
        return (
            correction_event.is_eligible
            and correction_event.correction_id not in profile.pet_drop_attempted_correction_ids
        )

    def calculate_drop_chance(self, correction_type: CorrectionType) -> float:
        return max(0.0, float(self._balance.pet_drop_chance.get(correction_type, 0.0)))

    def roll_pet_fragment(self, profile: GamificationProfile, correction_event: CorrectionEvent, rng: Random) -> ServiceResult:
        del profile, correction_event
        pet_type = rng.choice(list(COLLECTIBLE_PET_TYPES))
        rarity = self._rarity_roller.roll(rng)
        return ServiceResult.ok(
            RewardEventType.PET_FRAGMENT_DROPPED,
            f"Выпал фрагмент: {PET_DEFINITIONS[pet_type].title} / {rarity.value}",
            pet_type=pet_type,
            rarity=rarity,
        )

    def apply_pet_pity_if_needed(
        self,
        profile: GamificationProfile,
        correction_event: CorrectionEvent,
        rng: Random,
    ) -> ServiceResult | None:
        if profile.pet_pity_counter < self._balance.pet_fragment_pity_threshold:
            return None
        return self._grant_pet_fragment(profile, correction_event, rng, source="pity")

    def apply_pet_fragment_drop(
        self,
        profile: GamificationProfile,
        correction_event: CorrectionEvent,
        rng: Random,
    ) -> ServiceResult:
        if not self.should_attempt_drop(profile, correction_event):
            return ServiceResult.fail(RewardEventType.PET_FRAGMENT_DROPPED)
        profile.pet_drop_attempted_correction_ids.add(correction_event.correction_id)
        pity = self.apply_pet_pity_if_needed(profile, correction_event, rng)
        if pity is not None:
            return pity
        correction_type = self._classifier.classify(correction_event)
        if rng.random() < self.calculate_drop_chance(correction_type):
            return self._grant_pet_fragment(profile, correction_event, rng, source="drop")
        profile.pet_pity_counter += 1
        return ServiceResult.fail(RewardEventType.PET_FRAGMENT_DROPPED)

    def _grant_pet_fragment(
        self,
        profile: GamificationProfile,
        correction_event: CorrectionEvent | None,
        rng: Random,
        *,
        source: str,
    ) -> ServiceResult:
        pet_type = rng.choice(list(COLLECTIBLE_PET_TYPES))
        rarity = self._rarity_roller.roll(rng)
        profile.pet_fragments.add(pet_type, rarity, 1)
        profile.pet_pity_counter = 0
        profile.pet_fragment_transactions.append(
            PetFragmentTransaction(
                id=_new_transaction_id(),
                correction_id=None if correction_event is None else correction_event.correction_id,
                pet_type=pet_type,
                rarity=rarity,
                amount=1,
                source=source,
                created_at=_now_iso(),
            )
        )
        prefix = "Pity: выпал фрагмент" if source == "pity" else "Выпал фрагмент"
        return ServiceResult.ok(
            RewardEventType.PET_FRAGMENT_DROPPED,
            f"{prefix}: {PET_DEFINITIONS[pet_type].title} / {rarity.value}",
            pet_type=pet_type,
            rarity=rarity,
            source=source,
        )


class SkinFragmentDropService:
    def __init__(
        self,
        balance: GamificationBalance = DEFAULT_GAMIFICATION_BALANCE,
        classifier: CorrectionEventClassifier | None = None,
    ) -> None:
        self._balance = balance
        self._classifier = classifier or CorrectionEventClassifier()
        self._rarity_roller = _RarityRoller(balance)

    def should_attempt_drop(self, profile: GamificationProfile, correction_event: CorrectionEvent) -> bool:
        return (
            correction_event.is_eligible
            and correction_event.correction_id not in profile.skin_drop_attempted_correction_ids
            and bool(self._available_skin_ids(profile))
        )

    def calculate_drop_chance(self, correction_type: CorrectionType) -> float:
        return max(0.0, float(self._balance.skin_drop_chance.get(correction_type, 0.0)))

    def roll_skin_fragment(self, profile: GamificationProfile, correction_event: CorrectionEvent, rng: Random) -> ServiceResult:
        del correction_event
        skin_id = rng.choice(self._available_skin_ids(profile))
        rarity = self._rarity_roller.roll(rng)
        return ServiceResult.ok(
            RewardEventType.SKIN_FRAGMENT_DROPPED,
            f"Выпал фрагмент скина: {SKIN_DEFINITIONS[skin_id].title} / {rarity.value}",
            skin_id=skin_id,
            rarity=rarity,
        )

    def apply_skin_pity_if_needed(
        self,
        profile: GamificationProfile,
        correction_event: CorrectionEvent,
        rng: Random,
    ) -> ServiceResult | None:
        if profile.skin_pity_counter < self._balance.skin_fragment_pity_threshold:
            return None
        return self._grant_skin_fragment(profile, correction_event, rng, source="pity")

    def apply_skin_fragment_drop(
        self,
        profile: GamificationProfile,
        correction_event: CorrectionEvent,
        rng: Random,
    ) -> ServiceResult:
        if not self.should_attempt_drop(profile, correction_event):
            return ServiceResult.fail(RewardEventType.SKIN_FRAGMENT_DROPPED)
        profile.skin_drop_attempted_correction_ids.add(correction_event.correction_id)
        pity = self.apply_skin_pity_if_needed(profile, correction_event, rng)
        if pity is not None:
            return pity
        correction_type = self._classifier.classify(correction_event)
        if rng.random() < self.calculate_drop_chance(correction_type):
            return self._grant_skin_fragment(profile, correction_event, rng, source="drop")
        profile.skin_pity_counter += 1
        return ServiceResult.fail(RewardEventType.SKIN_FRAGMENT_DROPPED)

    @staticmethod
    def _available_skin_ids(profile: GamificationProfile) -> list[str]:
        return [
            skin_id
            for pet_type, progress in profile.pet_progress.items()
            if progress.unlocked
            for skin_id in PAID_SKIN_IDS_BY_PET.get(pet_type, ())
        ]

    def _grant_skin_fragment(
        self,
        profile: GamificationProfile,
        correction_event: CorrectionEvent | None,
        rng: Random,
        *,
        source: str,
    ) -> ServiceResult:
        available = self._available_skin_ids(profile)
        if not available:
            return ServiceResult.fail(RewardEventType.SKIN_FRAGMENT_DROPPED, "Нет доступных скинов для фрагментов")
        skin_id = rng.choice(available)
        rarity = self._rarity_roller.roll(rng)
        profile.skin_fragments.add(skin_id, rarity, 1)
        profile.skin_pity_counter = 0
        profile.skin_fragment_transactions.append(
            SkinFragmentTransaction(
                id=_new_transaction_id(),
                correction_id=None if correction_event is None else correction_event.correction_id,
                skin_id=skin_id,
                rarity=rarity,
                amount=1,
                source=source,
                created_at=_now_iso(),
            )
        )
        prefix = "Pity: выпал фрагмент скина" if source == "pity" else "Выпал фрагмент скина"
        return ServiceResult.ok(
            RewardEventType.SKIN_FRAGMENT_DROPPED,
            f"{prefix}: {SKIN_DEFINITIONS[skin_id].title} / {rarity.value}",
            skin_id=skin_id,
            rarity=rarity,
            source=source,
        )


class PetUnlockService:
    def __init__(self, balance: GamificationBalance = DEFAULT_GAMIFICATION_BALANCE) -> None:
        self._balance = balance

    def can_unlock_pet(self, profile: GamificationProfile, pet_type: PetType) -> bool:
        progress = profile.pet_progress.get(pet_type)
        return bool(
            progress is not None
            and not progress.unlocked
            and profile.pet_fragments.get(pet_type, Rarity.COMMON) >= self._balance.pet_common_fragments_required
        )

    def unlock_pet(self, profile: GamificationProfile, pet_type: PetType) -> ServiceResult:
        if pet_type not in PET_DEFINITIONS:
            return ServiceResult.fail(RewardEventType.NOT_ENOUGH_FRAGMENTS, "Неизвестный питомец")
        if pet_type == PetType.KRAKEN:
            return ServiceResult.fail(RewardEventType.PET_UNLOCKED, "Кракен уже открыт")
        progress = profile.pet_progress[pet_type]
        if progress.unlocked:
            return ServiceResult.fail(RewardEventType.PET_UNLOCKED, "Питомец уже открыт")
        if not self.can_unlock_pet(profile, pet_type):
            return ServiceResult.fail(RewardEventType.NOT_ENOUGH_FRAGMENTS, "Недостаточно common-фрагментов питомца")
        profile.pet_fragments.spend(pet_type, Rarity.COMMON, self._balance.pet_common_fragments_required)
        progress.unlocked = True
        progress.rarity = Rarity.COMMON
        default_skin_id = DEFAULT_SKIN_BY_PET[pet_type]
        default_skin = profile.skin_progress[default_skin_id]
        default_skin.unlocked = True
        default_skin.rarity = Rarity.COMMON
        profile.selected_skin_by_pet[pet_type] = default_skin_id
        return ServiceResult.ok(
            RewardEventType.PET_UNLOCKED,
            f"Питомец открыт: {PET_DEFINITIONS[pet_type].title}",
            pet_type=pet_type,
        )

    def can_upgrade_pet(self, profile: GamificationProfile, pet_type: PetType, method: str | None = None) -> bool:
        return self._pet_upgrade_method(profile, pet_type, method) is not None

    def upgrade_pet(self, profile: GamificationProfile, pet_type: PetType, method: str | None = None) -> ServiceResult:
        selected_method = self._pet_upgrade_method(profile, pet_type, method)
        if selected_method is None:
            return ServiceResult.fail(RewardEventType.NOT_ENOUGH_FRAGMENTS, "Недостаточно фрагментов для повышения")
        progress = profile.pet_progress[pet_type]
        assert progress.rarity is not None
        target_rarity = next_rarity(progress.rarity)
        if target_rarity is None:
            return ServiceResult.fail(RewardEventType.PET_UPGRADED, "Питомец уже legendary")
        spend_rarity = target_rarity if selected_method == "direct" else progress.rarity
        spend_amount = (
            self._balance.direct_upgrade_fragments_required
            if selected_method == "direct"
            else self._balance.fallback_upgrade_fragments_required
        )
        profile.pet_fragments.spend(pet_type, spend_rarity, spend_amount)
        progress.rarity = target_rarity
        return ServiceResult.ok(
            RewardEventType.PET_UPGRADED,
            f"Питомец повышен: {PET_DEFINITIONS[pet_type].title} -> {target_rarity.value}",
            pet_type=pet_type,
            rarity=target_rarity,
            method=selected_method,
        )

    def _pet_upgrade_method(self, profile: GamificationProfile, pet_type: PetType, method: str | None) -> str | None:
        progress = profile.pet_progress.get(pet_type)
        if progress is None or not progress.unlocked or progress.rarity is None:
            return None
        target_rarity = next_rarity(progress.rarity)
        if target_rarity is None:
            return None
        direct_available = (
            profile.pet_fragments.get(pet_type, target_rarity) >= self._balance.direct_upgrade_fragments_required
        )
        fallback_available = (
            profile.pet_fragments.get(pet_type, progress.rarity) >= self._balance.fallback_upgrade_fragments_required
        )
        if method == "direct":
            return "direct" if direct_available else None
        if method == "fallback":
            return "fallback" if fallback_available else None
        if direct_available:
            return "direct"
        if fallback_available:
            return "fallback"
        return None


class SkinUnlockService:
    def __init__(self, balance: GamificationBalance = DEFAULT_GAMIFICATION_BALANCE) -> None:
        self._balance = balance

    def can_unlock_skin(self, profile: GamificationProfile, skin_id: str) -> bool:
        definition = SKIN_DEFINITIONS.get(str(skin_id))
        if definition is None or definition.is_default:
            return False
        return bool(
            profile.pet_progress[definition.pet_type].unlocked
            and not profile.skin_progress[skin_id].unlocked
            and profile.skin_fragments.get(skin_id, Rarity.COMMON) >= self._balance.skin_common_fragments_required
        )

    def unlock_skin(self, profile: GamificationProfile, skin_id: str) -> ServiceResult:
        skin_id = str(skin_id)
        definition = SKIN_DEFINITIONS.get(skin_id)
        if definition is None:
            return ServiceResult.fail(RewardEventType.NOT_ENOUGH_FRAGMENTS, "Неизвестный скин")
        if definition.is_default:
            return ServiceResult.fail(RewardEventType.SKIN_UNLOCKED, "Базовый скин открывается вместе с питомцем")
        if not profile.pet_progress[definition.pet_type].unlocked:
            return ServiceResult.fail(RewardEventType.SKIN_UNLOCKED, "Сначала откройте питомца")
        progress = profile.skin_progress[skin_id]
        if progress.unlocked:
            return ServiceResult.fail(RewardEventType.SKIN_UNLOCKED, "Скин уже открыт")
        if not self.can_unlock_skin(profile, skin_id):
            return ServiceResult.fail(RewardEventType.NOT_ENOUGH_FRAGMENTS, "Недостаточно common-фрагментов скина")
        profile.skin_fragments.spend(skin_id, Rarity.COMMON, self._balance.skin_common_fragments_required)
        progress.unlocked = True
        progress.rarity = Rarity.COMMON
        return ServiceResult.ok(
            RewardEventType.SKIN_UNLOCKED,
            f"Скин открыт: {definition.title}",
            skin_id=skin_id,
        )

    def can_upgrade_skin(self, profile: GamificationProfile, skin_id: str, method: str | None = None) -> bool:
        return self._skin_upgrade_method(profile, skin_id, method) is not None

    def upgrade_skin(self, profile: GamificationProfile, skin_id: str, method: str | None = None) -> ServiceResult:
        skin_id = str(skin_id)
        selected_method = self._skin_upgrade_method(profile, skin_id, method)
        if selected_method is None:
            return ServiceResult.fail(RewardEventType.NOT_ENOUGH_FRAGMENTS, "Недостаточно фрагментов для скина")
        progress = profile.skin_progress[skin_id]
        assert progress.rarity is not None
        target_rarity = next_rarity(progress.rarity)
        if target_rarity is None:
            return ServiceResult.fail(RewardEventType.SKIN_UPGRADED, "Скин уже legendary")
        spend_rarity = target_rarity if selected_method == "direct" else progress.rarity
        spend_amount = (
            self._balance.direct_upgrade_fragments_required
            if selected_method == "direct"
            else self._balance.fallback_upgrade_fragments_required
        )
        profile.skin_fragments.spend(skin_id, spend_rarity, spend_amount)
        progress.rarity = target_rarity
        return ServiceResult.ok(
            RewardEventType.SKIN_UPGRADED,
            f"Скин повышен: {SKIN_DEFINITIONS[skin_id].title} -> {target_rarity.value}",
            skin_id=skin_id,
            rarity=target_rarity,
            method=selected_method,
        )

    def _skin_upgrade_method(self, profile: GamificationProfile, skin_id: str, method: str | None) -> str | None:
        skin_id = str(skin_id)
        definition = SKIN_DEFINITIONS.get(skin_id)
        if definition is None or definition.is_default or not profile.pet_progress[definition.pet_type].unlocked:
            return None
        progress = profile.skin_progress.get(skin_id)
        if progress is None or not progress.unlocked or progress.rarity is None:
            return None
        target_rarity = next_rarity(progress.rarity)
        if target_rarity is None:
            return None
        direct_available = (
            profile.skin_fragments.get(skin_id, target_rarity) >= self._balance.direct_upgrade_fragments_required
        )
        fallback_available = (
            profile.skin_fragments.get(skin_id, progress.rarity)
            >= self._balance.fallback_upgrade_fragments_required
        )
        if method == "direct":
            return "direct" if direct_available else None
        if method == "fallback":
            return "fallback" if fallback_available else None
        if direct_available:
            return "direct"
        if fallback_available:
            return "fallback"
        return None


class PetFragmentExchangeService:
    def __init__(self, balance: GamificationBalance = DEFAULT_GAMIFICATION_BALANCE) -> None:
        self._balance = balance

    def exchange_pet_fragments(
        self,
        profile: GamificationProfile,
        source_fragments: Mapping[PetType, int],
        rarity: Rarity,
        target_pet: PetType,
    ) -> ServiceResult:
        if target_pet not in PET_DEFINITIONS or target_pet == PetType.KRAKEN:
            return ServiceResult.fail(RewardEventType.NOT_ENOUGH_FRAGMENTS, "Нельзя выбрать этого питомца как цель")
        normalized_sources: dict[PetType, int] = {}
        for pet, amount in source_fragments.items():
            parsed_pet = _parse_pet_type(pet)
            if parsed_pet is None:
                return ServiceResult.fail(RewardEventType.NOT_ENOUGH_FRAGMENTS, "Неизвестный source-питомец")
            amount_int = int(amount)
            if amount_int < 0:
                return ServiceResult.fail(RewardEventType.NOT_ENOUGH_FRAGMENTS, "Количество source-фрагментов не может быть отрицательным")
            if amount_int:
                normalized_sources[parsed_pet] = amount_int
        total = sum(normalized_sources.values())
        if total != self._balance.exchange_source_fragments_required:
            return ServiceResult.fail(RewardEventType.NOT_ENOUGH_FRAGMENTS, "Нужно выбрать ровно 10 фрагментов для обмена")
        for pet, amount in normalized_sources.items():
            if profile.pet_fragments.get(pet, rarity) < amount:
                return ServiceResult.fail(RewardEventType.NOT_ENOUGH_FRAGMENTS, "Нельзя списать больше фрагментов, чем есть")
        for pet, amount in normalized_sources.items():
            profile.pet_fragments.spend(pet, rarity, amount)
        profile.pet_fragments.add(target_pet, rarity, self._balance.exchange_target_fragments_received)
        profile.pet_fragment_transactions.append(
            PetFragmentTransaction(
                id=_new_transaction_id(),
                correction_id=None,
                pet_type=target_pet,
                rarity=rarity,
                amount=self._balance.exchange_target_fragments_received,
                source="exchange",
                created_at=_now_iso(),
            )
        )
        return ServiceResult.ok(
            RewardEventType.PET_FRAGMENT_EXCHANGED,
            f"Обмен: +1 {rarity.value}-фрагмент {PET_DEFINITIONS[target_pet].title}",
            pet_type=target_pet,
            rarity=rarity,
        )


class SkinFragmentExchangeService:
    def __init__(self, balance: GamificationBalance = DEFAULT_GAMIFICATION_BALANCE) -> None:
        self._balance = balance

    def exchange_skin_fragments(
        self,
        profile: GamificationProfile,
        source_fragments: Mapping[str, int],
        rarity: Rarity,
        target_skin_id: str,
    ) -> ServiceResult:
        target_skin_id = str(target_skin_id)
        target_definition = SKIN_DEFINITIONS.get(target_skin_id)
        if target_definition is None:
            return ServiceResult.fail(RewardEventType.NOT_ENOUGH_FRAGMENTS, "Неизвестный target skin")
        if target_definition.is_default:
            return ServiceResult.fail(RewardEventType.NOT_ENOUGH_FRAGMENTS, "Нельзя выбрать default skin как target")
        if not profile.pet_progress[target_definition.pet_type].unlocked:
            return ServiceResult.fail(RewardEventType.NOT_ENOUGH_FRAGMENTS, "Target skin принадлежит закрытому питомцу")
        normalized_sources: dict[str, int] = {}
        for skin_id, amount in source_fragments.items():
            skin_key = str(skin_id)
            if skin_key not in SKIN_DEFINITIONS:
                return ServiceResult.fail(RewardEventType.NOT_ENOUGH_FRAGMENTS, "Неизвестный source skin")
            amount_int = int(amount)
            if amount_int < 0:
                return ServiceResult.fail(RewardEventType.NOT_ENOUGH_FRAGMENTS, "Количество source-фрагментов не может быть отрицательным")
            if amount_int:
                normalized_sources[skin_key] = amount_int
        total = sum(normalized_sources.values())
        if total != self._balance.exchange_source_fragments_required:
            return ServiceResult.fail(RewardEventType.NOT_ENOUGH_FRAGMENTS, "Нужно выбрать ровно 10 фрагментов для обмена")
        for skin_id, amount in normalized_sources.items():
            if profile.skin_fragments.get(skin_id, rarity) < amount:
                return ServiceResult.fail(RewardEventType.NOT_ENOUGH_FRAGMENTS, "Нельзя списать больше фрагментов, чем есть")
        for skin_id, amount in normalized_sources.items():
            profile.skin_fragments.spend(skin_id, rarity, amount)
        profile.skin_fragments.add(target_skin_id, rarity, self._balance.exchange_target_fragments_received)
        profile.skin_fragment_transactions.append(
            SkinFragmentTransaction(
                id=_new_transaction_id(),
                correction_id=None,
                skin_id=target_skin_id,
                rarity=rarity,
                amount=self._balance.exchange_target_fragments_received,
                source="exchange",
                created_at=_now_iso(),
            )
        )
        return ServiceResult.ok(
            RewardEventType.SKIN_FRAGMENT_EXCHANGED,
            f"Обмен: +1 {rarity.value}-фрагмент скина {target_definition.title}",
            skin_id=target_skin_id,
            rarity=rarity,
        )


class ShopService:
    def __init__(self, balance: GamificationBalance = DEFAULT_GAMIFICATION_BALANCE) -> None:
        self._balance = balance

    def get_available_shop_items(self, profile: GamificationProfile) -> dict[str, bool | int]:
        return {
            "pet_fragment_capsule_price": self._balance.random_common_pet_fragment_capsule_price,
            "pet_fragment_capsule_available": bool(COLLECTIBLE_PET_TYPES),
            "skin_fragment_capsule_price": self._balance.random_common_skin_fragment_capsule_price,
            "skin_fragment_capsule_available": bool(self._available_paid_skins_for_open_pets(profile)),
        }

    def buy_random_common_pet_fragment_capsule(self, profile: GamificationProfile, rng: Random) -> ServiceResult:
        price = self._balance.random_common_pet_fragment_capsule_price
        if profile.wallet_balance < price:
            return ServiceResult.fail(RewardEventType.NOT_ENOUGH_CURRENCY, "Недостаточно фрагментов кристалла")
        profile.wallet_balance -= price
        profile.lifetime_currency_spent += price
        pet_type = rng.choice(list(COLLECTIBLE_PET_TYPES))
        profile.pet_fragments.add(pet_type, Rarity.COMMON, 1)
        profile.purchase_transactions.append(
            PurchaseTransaction(
                id=_new_transaction_id(),
                item_type="pet_fragment_capsule",
                price=price,
                created_at=_now_iso(),
            )
        )
        profile.pet_fragment_transactions.append(
            PetFragmentTransaction(
                id=_new_transaction_id(),
                correction_id=None,
                pet_type=pet_type,
                rarity=Rarity.COMMON,
                amount=1,
                source="capsule",
                created_at=_now_iso(),
            )
        )
        return ServiceResult.ok(
            RewardEventType.PET_FRAGMENT_BOUGHT,
            f"Куплена капсула: выпал common-фрагмент {PET_DEFINITIONS[pet_type].title}",
            pet_type=pet_type,
            rarity=Rarity.COMMON,
        )

    def buy_random_common_skin_fragment_capsule(self, profile: GamificationProfile, rng: Random) -> ServiceResult:
        price = self._balance.random_common_skin_fragment_capsule_price
        available = self._available_paid_skins_for_open_pets(profile)
        if not available:
            return ServiceResult.fail(RewardEventType.SKIN_FRAGMENT_BOUGHT, "Нет доступных платных скинов открытых питомцев")
        if profile.wallet_balance < price:
            return ServiceResult.fail(RewardEventType.NOT_ENOUGH_CURRENCY, "Недостаточно фрагментов кристалла")
        profile.wallet_balance -= price
        profile.lifetime_currency_spent += price
        skin_id = rng.choice(available)
        profile.skin_fragments.add(skin_id, Rarity.COMMON, 1)
        profile.purchase_transactions.append(
            PurchaseTransaction(
                id=_new_transaction_id(),
                item_type="skin_fragment_capsule",
                price=price,
                created_at=_now_iso(),
            )
        )
        profile.skin_fragment_transactions.append(
            SkinFragmentTransaction(
                id=_new_transaction_id(),
                correction_id=None,
                skin_id=skin_id,
                rarity=Rarity.COMMON,
                amount=1,
                source="capsule",
                created_at=_now_iso(),
            )
        )
        return ServiceResult.ok(
            RewardEventType.SKIN_FRAGMENT_BOUGHT,
            f"Куплена капсула: выпал common-фрагмент скина {SKIN_DEFINITIONS[skin_id].title}",
            skin_id=skin_id,
            rarity=Rarity.COMMON,
        )

    @staticmethod
    def _available_paid_skins_for_open_pets(profile: GamificationProfile) -> list[str]:
        return [
            skin_id
            for pet_type, progress in profile.pet_progress.items()
            if progress.unlocked
            for skin_id in PAID_SKIN_IDS_BY_PET.get(pet_type, ())
        ]


class SelectionService:
    def select_pet(self, profile: GamificationProfile, pet_type: PetType) -> ServiceResult:
        progress = profile.pet_progress.get(pet_type)
        if progress is None or not progress.unlocked:
            return ServiceResult.fail(RewardEventType.PET_SELECTED, "Питомец закрыт")
        profile.selected_pet = pet_type
        default_skin = DEFAULT_SKIN_BY_PET[pet_type]
        selected_skin = profile.selected_skin_by_pet.get(pet_type, default_skin)
        if selected_skin not in profile.skin_progress or not profile.skin_progress[selected_skin].unlocked:
            profile.selected_skin_by_pet[pet_type] = default_skin
        return ServiceResult.ok(
            RewardEventType.PET_SELECTED,
            f"Выбран питомец: {PET_DEFINITIONS[pet_type].title}",
            pet_type=pet_type,
        )

    def select_skin(self, profile: GamificationProfile, skin_id: str) -> ServiceResult:
        skin_id = str(skin_id)
        definition = SKIN_DEFINITIONS.get(skin_id)
        progress = profile.skin_progress.get(skin_id)
        if definition is None or progress is None:
            return ServiceResult.fail(RewardEventType.SKIN_SELECTED, "Неизвестный скин")
        if not profile.pet_progress[definition.pet_type].unlocked or not progress.unlocked:
            return ServiceResult.fail(RewardEventType.SKIN_SELECTED, "Скин закрыт")
        profile.selected_skin_by_pet[definition.pet_type] = skin_id
        if profile.selected_pet != definition.pet_type:
            profile.selected_pet = definition.pet_type
        return ServiceResult.ok(
            RewardEventType.SKIN_SELECTED,
            f"Выбран скин: {definition.title}",
            skin_id=skin_id,
        )


class PetReactionService:
    def mood_for_event(self, event_type: RewardEventType | None = None) -> PetMood:
        return mood_for_reward_event(event_type)

    def reaction_for_event(
        self,
        profile: GamificationProfile,
        event_type: RewardEventType | None = None,
    ) -> str:
        pet_title = PET_DEFINITIONS.get(profile.selected_pet, PET_DEFINITIONS[PetType.KRAKEN]).title
        if event_type in {
            RewardEventType.PET_FRAGMENT_DROPPED,
            RewardEventType.SKIN_FRAGMENT_DROPPED,
            RewardEventType.PET_UNLOCKED,
            RewardEventType.SKIN_UNLOCKED,
        }:
            return f"{pet_title} празднует находку."
        if event_type in {RewardEventType.PET_UPGRADED, RewardEventType.SKIN_UPGRADED}:
            return f"{pet_title} сияет новым уровнем."
        if event_type == RewardEventType.CURRENCY_EARNED:
            return f"{pet_title} доволен точной правкой."
        return f"{pet_title} отдыхает и ждёт следующей полезной правки."


class GamificationService:
    def __init__(
        self,
        profile_service: GamificationProfileService,
        *,
        balance: GamificationBalance = DEFAULT_GAMIFICATION_BALANCE,
        rng: Random | None = None,
    ) -> None:
        self.profile_service = profile_service
        self.balance = balance
        self.rng = rng or Random()
        self.classifier = CorrectionEventClassifier()
        self.currency_rewards = CurrencyRewardService(balance, self.classifier)
        self.pet_drops = PetFragmentDropService(balance, self.classifier)
        self.skin_drops = SkinFragmentDropService(balance, self.classifier)

    def handle_correction_event(self, correction_event: CorrectionEvent) -> list[ServiceResult]:
        profile = self.profile_service.load_profile()
        results: list[ServiceResult] = []
        currency_result = self.currency_rewards.add_currency_for_correction(profile, correction_event)
        if currency_result.success:
            results.append(currency_result)
        pet_drop_result = self.pet_drops.apply_pet_fragment_drop(profile, correction_event, self.rng)
        if pet_drop_result.success:
            results.append(pet_drop_result)
        skin_drop_result = self.skin_drops.apply_skin_fragment_drop(profile, correction_event, self.rng)
        if skin_drop_result.success:
            results.append(skin_drop_result)
        self.profile_service.save_profile(profile)
        return results


__all__ = [
    "CorrectionEventClassifier",
    "CurrencyRewardService",
    "GamificationPayloadStore",
    "GamificationProfileService",
    "GamificationService",
    "InMemoryGamificationPayloadStore",
    "PetFragmentDropService",
    "PetFragmentExchangeService",
    "PetReactionService",
    "PetUnlockService",
    "SelectionService",
    "ShopService",
    "SkinFragmentDropService",
    "SkinFragmentExchangeService",
    "SkinUnlockService",
]
