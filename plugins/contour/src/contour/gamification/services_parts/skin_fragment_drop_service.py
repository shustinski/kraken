"""services: skin fragment drop service."""
from __future__ import annotations
from random import Random
from contour.gamification.config import DEFAULT_GAMIFICATION_BALANCE, GamificationBalance
from contour.gamification.models import (
    CorrectionEvent,
    CorrectionType,
    GamificationProfile,
    RewardEventType,
    ServiceResult,
    SkinFragmentTransaction,
)
from contour.gamification.registry import (
    PAID_SKIN_IDS_BY_PET,
    SKIN_DEFINITIONS,
)

from contour.gamification.services_parts.in_memory_gamification_payload import (
    CorrectionEventClassifier,
    _new_transaction_id,
    _now_iso,
)

from contour.gamification.services_parts.rarity_roller import _RarityRoller

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
