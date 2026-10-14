"""services: currency reward service."""
from __future__ import annotations
from contour.gamification.config import DEFAULT_GAMIFICATION_BALANCE, GamificationBalance
from contour.gamification.models import (
    CorrectionEvent,
    CurrencyTransaction,
    GamificationProfile,
    RewardEventType,
    ServiceResult,
)

from contour.gamification.services_parts.in_memory_gamification_payload import (
    CorrectionEventClassifier,
    _new_transaction_id,
    _now_iso,
    _profile_level_for_xp,
)

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
