"""services: profile payload."""
from __future__ import annotations
from collections.abc import Mapping
from dataclasses import asdict
from typing import Any
from contour.gamification.models import (
    GamificationProfile,
    PetProgress,
    PetType,
    SkinProgress,
)
from contour.gamification.registry import (
    SKIN_DEFINITIONS,
)

from contour.gamification.services_parts.in_memory_gamification_payload import (
    _clamped_int,
    _parse_pet_type,
    _parse_rarity,
)


class ProfilePayloadMixin:
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
