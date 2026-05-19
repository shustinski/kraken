from __future__ import annotations

from dataclasses import dataclass

from .models import PetMood, PetType, RewardEventType
from .registry import DEFAULT_SKIN_BY_PET, PET_DEFINITIONS, SKIN_DEFINITIONS


@dataclass(frozen=True, slots=True)
class PetVisualDefinition:
    pet_type: PetType
    body_color: str
    accent_color: str
    secondary_color: str
    line_color: str = "#1f2937"


@dataclass(frozen=True, slots=True)
class SkinVisualDefinition:
    skin_id: str
    pet_type: PetType
    body_color: str
    accent_color: str
    secondary_color: str
    accessory_color: str


PET_VISUALS: dict[PetType, PetVisualDefinition] = {
    PetType.KRAKEN: PetVisualDefinition(PetType.KRAKEN, "#5B8DEF", "#8EE7FF", "#314C8F"),
    PetType.CAT: PetVisualDefinition(PetType.CAT, "#F2A65A", "#FFE6A7", "#8F563B"),
    PetType.DOG: PetVisualDefinition(PetType.DOG, "#C48A54", "#F7D9A4", "#6F4D32"),
    PetType.CAPYBARA: PetVisualDefinition(PetType.CAPYBARA, "#B08968", "#E6CCB2", "#6B4F3F"),
    PetType.CARNIVOROUS_PLANT: PetVisualDefinition(
        PetType.CARNIVOROUS_PLANT,
        "#46A758",
        "#F87171",
        "#2F6B3A",
    ),
    PetType.HORSE: PetVisualDefinition(PetType.HORSE, "#A66A3F", "#E9C46A", "#5F3A24"),
    PetType.FROG: PetVisualDefinition(PetType.FROG, "#5DBB63", "#D9F99D", "#2F7D32"),
    PetType.HAMSTER: PetVisualDefinition(PetType.HAMSTER, "#D9A066", "#FFF1C2", "#8C5A32"),
}


SKIN_VISUALS: dict[str, SkinVisualDefinition] = {
    "kraken_default": SkinVisualDefinition("kraken_default", PetType.KRAKEN, "#5B8DEF", "#8EE7FF", "#314C8F", "#E0F2FE"),
    "kraken_deep": SkinVisualDefinition("kraken_deep", PetType.KRAKEN, "#244A73", "#2DD4BF", "#112B46", "#A7F3D0"),
    "kraken_cyber": SkinVisualDefinition("kraken_cyber", PetType.KRAKEN, "#4C1D95", "#22D3EE", "#111827", "#F0ABFC"),
    "cat_default": SkinVisualDefinition("cat_default", PetType.CAT, "#F2A65A", "#FFE6A7", "#8F563B", "#FFF7ED"),
    "cat_lab": SkinVisualDefinition("cat_lab", PetType.CAT, "#F6B26B", "#FFFFFF", "#6B7280", "#38BDF8"),
    "cat_cyber": SkinVisualDefinition("cat_cyber", PetType.CAT, "#2D3748", "#F472B6", "#111827", "#67E8F9"),
    "dog_default": SkinVisualDefinition("dog_default", PetType.DOG, "#C48A54", "#F7D9A4", "#6F4D32", "#FFF7ED"),
    "dog_engineer": SkinVisualDefinition("dog_engineer", PetType.DOG, "#A16207", "#FACC15", "#44403C", "#60A5FA"),
    "dog_space": SkinVisualDefinition("dog_space", PetType.DOG, "#64748B", "#F8FAFC", "#1E293B", "#93C5FD"),
    "capybara_default": SkinVisualDefinition(
        "capybara_default",
        PetType.CAPYBARA,
        "#B08968",
        "#E6CCB2",
        "#6B4F3F",
        "#FFF7ED",
    ),
    "capybara_labcoat": SkinVisualDefinition(
        "capybara_labcoat",
        PetType.CAPYBARA,
        "#A47551",
        "#FFFFFF",
        "#4B5563",
        "#34D399",
    ),
    "capybara_zen": SkinVisualDefinition(
        "capybara_zen",
        PetType.CAPYBARA,
        "#8B7355",
        "#FDE68A",
        "#365314",
        "#A3E635",
    ),
    "carnivorous_plant_default": SkinVisualDefinition(
        "carnivorous_plant_default",
        PetType.CARNIVOROUS_PLANT,
        "#46A758",
        "#F87171",
        "#2F6B3A",
        "#FDE68A",
    ),
    "carnivorous_plant_lab": SkinVisualDefinition(
        "carnivorous_plant_lab",
        PetType.CARNIVOROUS_PLANT,
        "#3F8F55",
        "#FFFFFF",
        "#2563EB",
        "#86EFAC",
    ),
    "carnivorous_plant_neon": SkinVisualDefinition(
        "carnivorous_plant_neon",
        PetType.CARNIVOROUS_PLANT,
        "#064E3B",
        "#FB7185",
        "#111827",
        "#5EEAD4",
    ),
    "horse_default": SkinVisualDefinition("horse_default", PetType.HORSE, "#A66A3F", "#E9C46A", "#5F3A24", "#FEF3C7"),
    "horse_worker": SkinVisualDefinition("horse_worker", PetType.HORSE, "#8B5E34", "#FBBF24", "#374151", "#60A5FA"),
    "horse_cyber": SkinVisualDefinition("horse_cyber", PetType.HORSE, "#312E81", "#38BDF8", "#111827", "#C084FC"),
    "frog_default": SkinVisualDefinition("frog_default", PetType.FROG, "#5DBB63", "#D9F99D", "#2F7D32", "#ECFCCB"),
    "frog_swamp": SkinVisualDefinition("frog_swamp", PetType.FROG, "#4D7C0F", "#BEF264", "#365314", "#A16207"),
    "frog_quantum": SkinVisualDefinition("frog_quantum", PetType.FROG, "#0F766E", "#67E8F9", "#111827", "#C4B5FD"),
    "hamster_default": SkinVisualDefinition(
        "hamster_default",
        PetType.HAMSTER,
        "#D9A066",
        "#FFF1C2",
        "#8C5A32",
        "#FFF7ED",
    ),
    "hamster_engineer": SkinVisualDefinition(
        "hamster_engineer",
        PetType.HAMSTER,
        "#C08457",
        "#FACC15",
        "#44403C",
        "#60A5FA",
    ),
    "hamster_archivist": SkinVisualDefinition(
        "hamster_archivist",
        PetType.HAMSTER,
        "#B77946",
        "#FDE68A",
        "#78350F",
        "#C4B5FD",
    ),
}


def mood_for_reward_event(event_type: RewardEventType | None) -> PetMood:
    if event_type in {
        RewardEventType.PET_UNLOCKED,
        RewardEventType.SKIN_UNLOCKED,
        RewardEventType.PET_FRAGMENT_DROPPED,
        RewardEventType.SKIN_FRAGMENT_DROPPED,
    }:
        return PetMood.CELEBRATING
    if event_type in {RewardEventType.PET_UPGRADED, RewardEventType.SKIN_UPGRADED}:
        return PetMood.LEVEL_UP
    if event_type in {
        RewardEventType.CURRENCY_EARNED,
        RewardEventType.CORRECTION_REWARDED,
        RewardEventType.PET_FRAGMENT_BOUGHT,
        RewardEventType.SKIN_FRAGMENT_BOUGHT,
        RewardEventType.PET_FRAGMENT_EXCHANGED,
        RewardEventType.SKIN_FRAGMENT_EXCHANGED,
    }:
        return PetMood.HAPPY
    if event_type in {RewardEventType.PET_SELECTED, RewardEventType.SKIN_SELECTED, RewardEventType.IMAGE_VIEWED}:
        return PetMood.FOCUSED
    if event_type == RewardEventType.IMAGE_ACCEPTED_WITHOUT_CHANGES:
        return PetMood.IDLE
    if event_type in {RewardEventType.NOT_ENOUGH_CURRENCY, RewardEventType.NOT_ENOUGH_FRAGMENTS}:
        return PetMood.TIRED
    return PetMood.IDLE


def skin_visual_for(pet_type: PetType, skin_id: str | None) -> SkinVisualDefinition:
    style = SKIN_VISUALS.get(str(skin_id)) if skin_id else None
    if style is not None and style.pet_type == pet_type:
        return style
    default_skin_id = DEFAULT_SKIN_BY_PET.get(pet_type, DEFAULT_SKIN_BY_PET[PetType.KRAKEN])
    return SKIN_VISUALS[default_skin_id]


def validate_visual_registry() -> None:
    missing_pets = set(PET_DEFINITIONS) - set(PET_VISUALS)
    if missing_pets:
        raise ValueError(f"Missing gamification pet visuals: {sorted(missing_pets)}")
    missing_skins = set(SKIN_DEFINITIONS) - set(SKIN_VISUALS)
    if missing_skins:
        raise ValueError(f"Missing gamification skin visuals: {sorted(missing_skins)}")


__all__ = [
    "PET_VISUALS",
    "SKIN_VISUALS",
    "PetVisualDefinition",
    "SkinVisualDefinition",
    "mood_for_reward_event",
    "skin_visual_for",
    "validate_visual_registry",
]
