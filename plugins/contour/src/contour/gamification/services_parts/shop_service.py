"""services: shop service."""
from __future__ import annotations
from random import Random
from contour.gamification.config import DEFAULT_GAMIFICATION_BALANCE, GamificationBalance
from contour.gamification.models import (
    GamificationProfile,
    PetFragmentTransaction,
    PurchaseTransaction,
    Rarity,
    RewardEventType,
    ServiceResult,
    SkinFragmentTransaction,
)
from contour.gamification.registry import (
    COLLECTIBLE_PET_TYPES,
    PAID_SKIN_IDS_BY_PET,
    PET_DEFINITIONS,
    SKIN_DEFINITIONS,
)

from contour.gamification.services_parts.in_memory_gamification_payload import (
    _new_transaction_id,
    _now_iso,
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
