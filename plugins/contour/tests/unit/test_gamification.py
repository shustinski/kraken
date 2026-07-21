from __future__ import annotations

import unittest
from random import Random

from contour.gamification import (
    COLLECTIBLE_PET_TYPES,
    DEFAULT_SKIN_BY_PET,
    PAID_SKIN_IDS,
    PET_DEFINITIONS,
    PET_VISUALS,
    SKIN_DEFINITIONS,
    SKIN_VISUALS,
    CorrectionEvent,
    CorrectionType,
    CurrencyRewardService,
    GamificationBalance,
    GamificationProfile,
    GamificationProfileService,
    PetFragmentDropService,
    PetFragmentExchangeService,
    PetMood,
    PetType,
    PetUnlockService,
    Rarity,
    RewardEventType,
    ShopService,
    SkinFragmentDropService,
    SkinFragmentExchangeService,
    SkinUnlockService,
    mood_for_reward_event,
    skin_visual_for,
    validate_visual_registry,
)


def _profile() -> GamificationProfile:
    return GamificationProfileService().create_default_profile()


def _event(
    correction_id: str = "c1",
    *,
    real_changes: bool = True,
    accepted_without_changes: bool = False,
    view_only: bool = False,
    correction_type: CorrectionType | None = CorrectionType.MINOR,
) -> CorrectionEvent:
    return CorrectionEvent(
        correction_id=correction_id,
        image_id="image-a",
        has_real_mask_changes=real_changes,
        accepted_without_changes=accepted_without_changes,
        view_only=view_only,
        correction_type=correction_type,
    )


def _drop_balance(*, pet_chance: float = 1.0, skin_chance: float = 1.0) -> GamificationBalance:
    balance = GamificationBalance()
    balance.pet_drop_chance = {correction_type: pet_chance for correction_type in CorrectionType}
    balance.skin_drop_chance = {correction_type: skin_chance for correction_type in CorrectionType}
    balance.rarity_distribution = {
        Rarity.COMMON: 1.0,
        Rarity.RARE: 0.0,
        Rarity.EPIC: 0.0,
        Rarity.LEGENDARY: 0.0,
    }
    return balance


class DefaultProfileTests(unittest.TestCase):
    def test_default_profile_opens_kraken_only_and_initializes_inventories(self) -> None:
        profile = _profile()

        self.assertTrue(profile.pet_progress[PetType.KRAKEN].unlocked)
        self.assertEqual(profile.pet_progress[PetType.KRAKEN].rarity, Rarity.COMMON)
        self.assertEqual(profile.selected_pet, PetType.KRAKEN)
        self.assertTrue(profile.skin_progress["kraken_default"].unlocked)
        self.assertEqual(profile.selected_skin_by_pet[PetType.KRAKEN], "kraken_default")
        self.assertEqual(profile.wallet_balance, 0)
        for pet_type in PET_DEFINITIONS:
            for rarity in Rarity:
                self.assertEqual(profile.pet_fragments.get(pet_type, rarity), 0)
            if pet_type != PetType.KRAKEN:
                self.assertFalse(profile.pet_progress[pet_type].unlocked)
        for skin_id in SKIN_DEFINITIONS:
            for rarity in Rarity:
                self.assertEqual(profile.skin_fragments.get(skin_id, rarity), 0)


class VisualReactionTests(unittest.TestCase):
    def test_every_pet_and_skin_has_visual_style(self) -> None:
        validate_visual_registry()

        self.assertEqual(set(PET_VISUALS), set(PET_DEFINITIONS))
        self.assertEqual(set(SKIN_VISUALS), set(SKIN_DEFINITIONS))
        for skin_id, definition in SKIN_DEFINITIONS.items():
            self.assertEqual(SKIN_VISUALS[skin_id].pet_type, definition.pet_type)
            self.assertEqual(skin_visual_for(definition.pet_type, skin_id).skin_id, skin_id)

    def test_reward_events_map_to_pet_moods(self) -> None:
        self.assertEqual(mood_for_reward_event(RewardEventType.CURRENCY_EARNED), PetMood.HAPPY)
        self.assertEqual(mood_for_reward_event(RewardEventType.PET_FRAGMENT_DROPPED), PetMood.CELEBRATING)
        self.assertEqual(mood_for_reward_event(RewardEventType.PET_UPGRADED), PetMood.LEVEL_UP)
        self.assertEqual(mood_for_reward_event(RewardEventType.IMAGE_VIEWED), PetMood.FOCUSED)
        self.assertEqual(mood_for_reward_event(RewardEventType.IMAGE_ACCEPTED_WITHOUT_CHANGES), PetMood.IDLE)
        self.assertEqual(mood_for_reward_event(RewardEventType.NOT_ENOUGH_CURRENCY), PetMood.TIRED)


class PetUnlockUpgradeTests(unittest.TestCase):
    def test_pet_unlock_requires_and_spends_common_fragments(self) -> None:
        profile = _profile()
        service = PetUnlockService()
        profile.pet_fragments.add(PetType.CAT, Rarity.RARE, 10)

        self.assertFalse(service.unlock_pet(profile, PetType.CAT).success)
        profile.pet_fragments.add(PetType.CAT, Rarity.COMMON, 10)
        result = service.unlock_pet(profile, PetType.CAT)

        self.assertTrue(result.success)
        self.assertTrue(profile.pet_progress[PetType.CAT].unlocked)
        self.assertEqual(profile.pet_progress[PetType.CAT].rarity, Rarity.COMMON)
        self.assertEqual(profile.pet_fragments.get(PetType.CAT, Rarity.COMMON), 0)
        self.assertTrue(profile.skin_progress["cat_default"].unlocked)
        self.assertFalse(service.unlock_pet(profile, PetType.CAT).success)

    def test_pet_upgrade_paths_and_limits(self) -> None:
        service = PetUnlockService()
        cases = [
            (Rarity.COMMON, Rarity.RARE, Rarity.RARE, 10, "direct"),
            (Rarity.COMMON, Rarity.RARE, Rarity.COMMON, 100, "fallback"),
            (Rarity.RARE, Rarity.EPIC, Rarity.EPIC, 10, "direct"),
            (Rarity.RARE, Rarity.EPIC, Rarity.RARE, 100, "fallback"),
            (Rarity.EPIC, Rarity.LEGENDARY, Rarity.LEGENDARY, 10, "direct"),
            (Rarity.EPIC, Rarity.LEGENDARY, Rarity.EPIC, 100, "fallback"),
        ]
        for current, target, fragment_rarity, amount, method in cases:
            with self.subTest(current=current, method=method):
                profile = _profile()
                profile.pet_progress[PetType.CAT].unlocked = True
                profile.pet_progress[PetType.CAT].rarity = current
                profile.pet_fragments.add(PetType.CAT, fragment_rarity, amount)

                result = service.upgrade_pet(profile, PetType.CAT, method)

                self.assertTrue(result.success)
                self.assertEqual(profile.pet_progress[PetType.CAT].rarity, target)
                self.assertEqual(profile.pet_fragments.get(PetType.CAT, fragment_rarity), 0)

    def test_pet_upgrade_rejects_skip_locked_and_legendary(self) -> None:
        service = PetUnlockService()
        profile = _profile()
        profile.pet_fragments.add(PetType.CAT, Rarity.EPIC, 10)
        self.assertFalse(service.can_upgrade_pet(profile, PetType.CAT))

        profile.pet_progress[PetType.CAT].unlocked = True
        profile.pet_progress[PetType.CAT].rarity = Rarity.COMMON
        self.assertFalse(service.can_upgrade_pet(profile, PetType.CAT, "direct"))

        profile.pet_progress[PetType.CAT].rarity = Rarity.LEGENDARY
        profile.pet_fragments.add(PetType.CAT, Rarity.LEGENDARY, 10)
        self.assertFalse(service.can_upgrade_pet(profile, PetType.CAT))


class PetExchangeTests(unittest.TestCase):
    def test_exchange_ten_same_rarity_pet_fragments_to_target(self) -> None:
        profile = _profile()
        service = PetFragmentExchangeService()
        profile.pet_progress[PetType.CAT].unlocked = True
        profile.pet_fragments.add(PetType.CAT, Rarity.COMMON, 3)
        profile.pet_fragments.add(PetType.DOG, Rarity.COMMON, 2)
        profile.pet_fragments.add(PetType.FROG, Rarity.COMMON, 5)

        result = service.exchange_pet_fragments(
            profile,
            {PetType.CAT: 3, PetType.DOG: 2, PetType.FROG: 5},
            Rarity.COMMON,
            PetType.HAMSTER,
        )

        self.assertTrue(result.success)
        self.assertEqual(profile.pet_fragments.get(PetType.HAMSTER, Rarity.COMMON), 1)
        self.assertEqual(profile.pet_fragments.get(PetType.CAT, Rarity.COMMON), 0)

    def test_exchange_rejects_wrong_count_target_kraken_and_overspend(self) -> None:
        profile = _profile()
        service = PetFragmentExchangeService()
        profile.pet_fragments.add(PetType.CAT, Rarity.COMMON, 10)

        self.assertFalse(service.exchange_pet_fragments(profile, {PetType.CAT: 9}, Rarity.COMMON, PetType.DOG).success)
        self.assertFalse(service.exchange_pet_fragments(profile, {PetType.CAT: 11}, Rarity.COMMON, PetType.DOG).success)
        self.assertFalse(service.exchange_pet_fragments(profile, {PetType.CAT: 10}, Rarity.COMMON, PetType.KRAKEN).success)
        self.assertFalse(service.exchange_pet_fragments(profile, {PetType.CAT: 10}, Rarity.RARE, PetType.DOG).success)


class CurrencyTests(unittest.TestCase):
    def test_currency_only_rewards_real_unique_corrections(self) -> None:
        profile = _profile()
        service = CurrencyRewardService()

        self.assertFalse(service.add_currency_for_correction(profile, _event(real_changes=False, view_only=True)).success)
        self.assertFalse(
            service.add_currency_for_correction(
                profile,
                _event("accept", real_changes=False, accepted_without_changes=True),
            ).success
        )
        result = service.add_currency_for_correction(profile, _event("reward", correction_type=CorrectionType.MAJOR))
        duplicate = service.add_currency_for_correction(profile, _event("reward", correction_type=CorrectionType.MAJOR))

        self.assertTrue(result.success)
        self.assertEqual(profile.wallet_balance, 4)
        self.assertFalse(duplicate.success)
        self.assertEqual(profile.wallet_balance, 4)


class DropTests(unittest.TestCase):
    def test_pet_drop_attempt_rules_success_and_duplicate_guard(self) -> None:
        profile = _profile()
        service = PetFragmentDropService(_drop_balance(pet_chance=1.0))
        rng = Random(1)

        self.assertFalse(service.apply_pet_fragment_drop(profile, _event("view", view_only=True), rng).success)
        self.assertFalse(
            service.apply_pet_fragment_drop(
                profile,
                _event("accept", real_changes=False, accepted_without_changes=True),
                rng,
            ).success
        )
        result = service.apply_pet_fragment_drop(profile, _event("drop"), rng)
        duplicate = service.apply_pet_fragment_drop(profile, _event("drop"), rng)

        self.assertTrue(result.success)
        self.assertEqual(sum(profile.pet_fragments.get(pet, Rarity.COMMON) for pet in COLLECTIBLE_PET_TYPES), 1)
        self.assertFalse(duplicate.success)
        self.assertEqual(sum(profile.pet_fragments.get(pet, Rarity.COMMON) for pet in COLLECTIBLE_PET_TYPES), 1)

    def test_pet_pity_grants_fragment_and_resets(self) -> None:
        profile = _profile()
        balance = _drop_balance(pet_chance=0.0)
        balance.pet_fragment_pity_threshold = 1
        profile.pet_pity_counter = 1
        result = PetFragmentDropService(balance).apply_pet_fragment_drop(profile, _event("pity"), Random(2))

        self.assertTrue(result.success)
        self.assertEqual(profile.pet_pity_counter, 0)
        self.assertEqual(sum(profile.pet_fragments.get(pet, Rarity.COMMON) for pet in COLLECTIBLE_PET_TYPES), 1)


class ShopTests(unittest.TestCase):
    def test_pet_capsule_spends_currency_and_adds_random_common_fragment(self) -> None:
        profile = _profile()
        profile.wallet_balance = 300
        result = ShopService().buy_random_common_pet_fragment_capsule(profile, Random(3))

        self.assertTrue(result.success)
        self.assertEqual(profile.wallet_balance, 0)
        self.assertEqual(profile.lifetime_currency_spent, 300)
        self.assertEqual(sum(profile.pet_fragments.get(pet, Rarity.COMMON) for pet in COLLECTIBLE_PET_TYPES), 1)

    def test_shop_rejects_insufficient_currency_and_has_no_direct_purchase_api(self) -> None:
        profile = _profile()
        shop = ShopService()

        self.assertFalse(shop.buy_random_common_pet_fragment_capsule(profile, Random(4)).success)
        self.assertFalse(hasattr(shop, "buy_pet_fragment"))


class SkinTests(unittest.TestCase):
    def test_skin_unlock_requires_open_pet_and_common_fragments(self) -> None:
        profile = _profile()
        service = SkinUnlockService()
        skin_id = "cat_cyber"
        profile.skin_fragments.add(skin_id, Rarity.COMMON, 10)

        self.assertFalse(service.unlock_skin(profile, skin_id).success)
        profile.pet_progress[PetType.CAT].unlocked = True
        profile.pet_progress[PetType.CAT].rarity = Rarity.COMMON
        result = service.unlock_skin(profile, skin_id)

        self.assertTrue(result.success)
        self.assertEqual(profile.skin_fragments.get(skin_id, Rarity.COMMON), 0)
        self.assertTrue(profile.skin_progress[skin_id].unlocked)

    def test_skin_upgrade_and_default_skin_opened_with_pet(self) -> None:
        profile = _profile()
        pet_service = PetUnlockService()
        skin_service = SkinUnlockService()
        profile.pet_fragments.add(PetType.CAT, Rarity.COMMON, 10)
        pet_service.unlock_pet(profile, PetType.CAT)
        self.assertTrue(profile.skin_progress[DEFAULT_SKIN_BY_PET[PetType.CAT]].unlocked)

        skin_id = "cat_lab"
        profile.skin_progress[skin_id].unlocked = True
        profile.skin_progress[skin_id].rarity = Rarity.COMMON
        profile.skin_fragments.add(skin_id, Rarity.RARE, 10)
        result = skin_service.upgrade_skin(profile, skin_id, "direct")

        self.assertTrue(result.success)
        self.assertEqual(profile.skin_progress[skin_id].rarity, Rarity.RARE)

    def test_skin_exchange_rules(self) -> None:
        profile = _profile()
        service = SkinFragmentExchangeService()
        profile.pet_progress[PetType.CAT].unlocked = True
        profile.pet_progress[PetType.CAT].rarity = Rarity.COMMON
        profile.skin_fragments.add("cat_lab", Rarity.COMMON, 10)

        result = service.exchange_skin_fragments(profile, {"cat_lab": 10}, Rarity.COMMON, "cat_cyber")

        self.assertTrue(result.success)
        self.assertEqual(profile.skin_fragments.get("cat_cyber", Rarity.COMMON), 1)
        self.assertFalse(
            service.exchange_skin_fragments(profile, {"cat_cyber": 1}, Rarity.COMMON, "cat_default").success
        )
        self.assertFalse(
            service.exchange_skin_fragments(profile, {"cat_cyber": 1}, Rarity.COMMON, "dog_space").success
        )
        self.assertFalse(
            service.exchange_skin_fragments(profile, {"cat_cyber": 1}, Rarity.RARE, "cat_lab").success
        )

    def test_skin_drop_requires_open_pet_with_paid_skins(self) -> None:
        profile = _profile()
        service = SkinFragmentDropService(_drop_balance(skin_chance=1.0))

        result = service.apply_skin_fragment_drop(profile, _event("skin-drop"), Random(5))

        self.assertTrue(result.success)
        self.assertEqual(sum(profile.skin_fragments.get(skin_id, Rarity.COMMON) for skin_id in PAID_SKIN_IDS if skin_id.startswith("kraken")), 1)


class RepairTests(unittest.TestCase):
    def test_repair_falls_back_selection_and_clamps_negative_values(self) -> None:
        service = GamificationProfileService()
        profile = _profile()
        profile.selected_pet = PetType.CAT
        profile.selected_skin_by_pet[PetType.KRAKEN] = "missing"
        profile.wallet_balance = -10
        profile.pet_fragments.fragments_by_pet.pop(PetType.DOG)
        profile.skin_fragments.fragments_by_skin_id["cat_lab"][Rarity.COMMON] = -5

        repaired = service.repair_profile(profile)

        self.assertEqual(repaired.selected_pet, PetType.KRAKEN)
        self.assertEqual(repaired.selected_skin_by_pet[PetType.KRAKEN], "kraken_default")
        self.assertEqual(repaired.wallet_balance, 0)
        self.assertEqual(repaired.pet_fragments.get(PetType.DOG, Rarity.COMMON), 0)
        self.assertEqual(repaired.skin_fragments.get("cat_lab", Rarity.COMMON), 0)


if __name__ == "__main__":
    unittest.main()
