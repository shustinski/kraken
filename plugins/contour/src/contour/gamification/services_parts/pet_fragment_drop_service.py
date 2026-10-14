"""services: pet fragment drop service."""
from __future__ import annotations
from random import Random
from contour.gamification.config import DEFAULT_GAMIFICATION_BALANCE, GamificationBalance
from contour.gamification.models import (
    CorrectionEvent,
    CorrectionType,
    GamificationProfile,
    PetFragmentTransaction,
    RewardEventType,
    ServiceResult,
)
from contour.gamification.registry import (
    COLLECTIBLE_PET_TYPES,
    PET_DEFINITIONS,
)

from contour.gamification.services_parts.in_memory_gamification_payload import (
    CorrectionEventClassifier,
    _new_transaction_id,
    _now_iso,
)

from contour.gamification.services_parts.rarity_roller import _RarityRoller

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
