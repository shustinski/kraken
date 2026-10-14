"""services: pet fragment transactions payload."""
from __future__ import annotations
from collections.abc import Mapping, MutableMapping
from typing import Any
from contour.gamification.models import (
    RARITY_ORDER,
    CurrencyTransaction,
    PetFragmentInventory,
    PetFragmentTransaction,
    PetType,
    PurchaseTransaction,
    Rarity,
    SkinFragmentInventory,
    SkinFragmentTransaction,
)
from contour.gamification.registry import (
    PET_DEFINITIONS,
    SKIN_DEFINITIONS,
)

from contour.gamification.services_parts.in_memory_gamification_payload import (
    _clamped_int,
    _new_transaction_id,
    _parse_pet_type,
    _parse_rarity,
)


class PetFragmentTransactionsPayloadMixin:
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
