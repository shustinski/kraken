"""services: selection service."""
from __future__ import annotations
from random import Random
from contour.gamification.config import DEFAULT_GAMIFICATION_BALANCE, GamificationBalance
from contour.gamification.models import (
    CorrectionEvent,
    GamificationProfile,
    PetMood,
    PetType,
    RewardEventType,
    ServiceResult,
)
from contour.gamification.registry import (
    DEFAULT_SKIN_BY_PET,
    PET_DEFINITIONS,
    SKIN_DEFINITIONS,
)
from contour.gamification.visuals import mood_for_reward_event

from contour.gamification.services_parts.currency_reward_service import CurrencyRewardService

from contour.gamification.services_parts.in_memory_gamification_payload import CorrectionEventClassifier

from contour.gamification.services_parts.pet_fragment_drop_service import PetFragmentDropService

from contour.gamification.services_parts.skin_fragment_drop_service import SkinFragmentDropService

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
