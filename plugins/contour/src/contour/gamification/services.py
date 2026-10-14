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

from .services_parts.create_default_profile import CreateDefaultProfileMixin
from .services_parts.repair_profile import RepairProfileMixin
from .services_parts.profile_payload import ProfilePayloadMixin
from .services_parts.pet_fragment_transactions_payload import PetFragmentTransactionsPayloadMixin
from .services_parts.in_memory_gamification_payload import (
    GamificationPayloadStore,
    InMemoryGamificationPayloadStore,
    _now_iso,
    _new_transaction_id,
    _enum_value,
    _parse_pet_type,
    _parse_rarity,
    _parse_correction_type,
    _clamped_int,
    _profile_level_for_xp,
    CorrectionEventClassifier,
)
from .services_parts.currency_reward_service import CurrencyRewardService
from .services_parts.rarity_roller import _RarityRoller
from .services_parts.pet_fragment_drop_service import PetFragmentDropService
from .services_parts.skin_fragment_drop_service import SkinFragmentDropService
from .services_parts.pet_unlock_service import PetUnlockService
from .services_parts.skin_unlock_service import SkinUnlockService
from .services_parts.pet_fragment_exchange_service import PetFragmentExchangeService
from .services_parts.skin_fragment_exchange_service import SkinFragmentExchangeService
from .services_parts.shop_service import ShopService
from .services_parts.selection_service import (
    SelectionService,
    PetReactionService,
    GamificationService,
)

class GamificationProfileService(
    CreateDefaultProfileMixin,
    RepairProfileMixin,
    ProfilePayloadMixin,
    PetFragmentTransactionsPayloadMixin,
):
    pass

# Imported so existing callers and tests can patch or import these names
# from this module after the implementation moved into focused units.
_FACADE_EXPORTS = (
    GamificationPayloadStore,
    InMemoryGamificationPayloadStore,
    _now_iso,
    _new_transaction_id,
    _enum_value,
    _parse_pet_type,
    _parse_rarity,
    _parse_correction_type,
    _clamped_int,
    _profile_level_for_xp,
    CorrectionEventClassifier,
    CurrencyRewardService,
    _RarityRoller,
    PetFragmentDropService,
    SkinFragmentDropService,
    PetUnlockService,
    SkinUnlockService,
    PetFragmentExchangeService,
    SkinFragmentExchangeService,
    ShopService,
    SelectionService,
    PetReactionService,
    GamificationService,
)

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
