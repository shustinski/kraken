"""services: skin fragment exchange service."""
from __future__ import annotations
from collections.abc import Mapping
from contour.gamification.config import DEFAULT_GAMIFICATION_BALANCE, GamificationBalance
from contour.gamification.models import (
    GamificationProfile,
    Rarity,
    RewardEventType,
    ServiceResult,
    SkinFragmentTransaction,
)
from contour.gamification.registry import (
    SKIN_DEFINITIONS,
)

from contour.gamification.services_parts.in_memory_gamification_payload import (
    _new_transaction_id,
    _now_iso,
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
