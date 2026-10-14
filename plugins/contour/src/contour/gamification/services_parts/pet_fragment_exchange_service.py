"""services: pet fragment exchange service."""
from __future__ import annotations
from collections.abc import Mapping
from contour.gamification.config import DEFAULT_GAMIFICATION_BALANCE, GamificationBalance
from contour.gamification.models import (
    GamificationProfile,
    PetFragmentTransaction,
    PetType,
    Rarity,
    RewardEventType,
    ServiceResult,
)
from contour.gamification.registry import (
    PET_DEFINITIONS,
)

from contour.gamification.services_parts.in_memory_gamification_payload import (
    _new_transaction_id,
    _now_iso,
    _parse_pet_type,
)

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
