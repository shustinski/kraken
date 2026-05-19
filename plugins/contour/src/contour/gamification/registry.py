from __future__ import annotations

from dataclasses import dataclass

from .models import PetType


@dataclass(frozen=True, slots=True)
class PetDefinition:
    pet_type: PetType
    title: str
    description: str
    collectible: bool = True


@dataclass(frozen=True, slots=True)
class SkinDefinition:
    skin_id: str
    pet_type: PetType
    title: str
    description: str
    is_default: bool = False
    collectible: bool = True


PET_DEFINITIONS: dict[PetType, PetDefinition] = {
    PetType.KRAKEN: PetDefinition(PetType.KRAKEN, "Кракен", "Стартовый помощник редактора.", collectible=False),
    PetType.CAT: PetDefinition(PetType.CAT, "Кошка", "Сосредоточенно следит за аккуратностью контуров."),
    PetType.DOG: PetDefinition(PetType.DOG, "Собака", "Радуется каждому исправленному артефакту."),
    PetType.CAPYBARA: PetDefinition(PetType.CAPYBARA, "Капибара", "Спокойно держит темп длинной разметки."),
    PetType.CARNIVOROUS_PLANT: PetDefinition(
        PetType.CARNIVOROUS_PLANT,
        "Хищное растение",
        "Любит чистые маски без лишних шумов.",
    ),
    PetType.HORSE: PetDefinition(PetType.HORSE, "Лошадь", "Помогает не сбиваться на больших наборах кадров."),
    PetType.FROG: PetDefinition(PetType.FROG, "Лягушка", "Отмечает точные мелкие правки."),
    PetType.HAMSTER: PetDefinition(PetType.HAMSTER, "Хомяк", "Собирает полезные фрагменты в инвентарь."),
}

COLLECTIBLE_PET_TYPES: tuple[PetType, ...] = tuple(
    pet_type for pet_type, definition in PET_DEFINITIONS.items() if definition.collectible
)

SKIN_DEFINITIONS: dict[str, SkinDefinition] = {
    "kraken_default": SkinDefinition("kraken_default", PetType.KRAKEN, "Обычный кракен", "Базовый облик.", True, False),
    "kraken_deep": SkinDefinition("kraken_deep", PetType.KRAKEN, "Глубинный кракен", "Темный морской стиль."),
    "kraken_cyber": SkinDefinition("kraken_cyber", PetType.KRAKEN, "Кибер-кракен", "Неоновый технический стиль."),
    "cat_default": SkinDefinition("cat_default", PetType.CAT, "Обычная кошка", "Базовый облик.", True, False),
    "cat_lab": SkinDefinition("cat_lab", PetType.CAT, "Лабораторная кошка", "Аккуратный лабораторный стиль."),
    "cat_cyber": SkinDefinition("cat_cyber", PetType.CAT, "Кибер-кошка", "Светящийся технический стиль."),
    "dog_default": SkinDefinition("dog_default", PetType.DOG, "Обычная собака", "Базовый облик.", True, False),
    "dog_engineer": SkinDefinition("dog_engineer", PetType.DOG, "Собака-инженер", "Рабочий инженерный стиль."),
    "dog_space": SkinDefinition("dog_space", PetType.DOG, "Космическая собака", "Скафандр для сложных задач."),
    "capybara_default": SkinDefinition("capybara_default", PetType.CAPYBARA, "Обычная капибара", "Базовый облик.", True, False),
    "capybara_labcoat": SkinDefinition("capybara_labcoat", PetType.CAPYBARA, "Капибара в халате", "Спокойный лабораторный стиль."),
    "capybara_zen": SkinDefinition("capybara_zen", PetType.CAPYBARA, "Дзен-капибара", "Минималистичный спокойный стиль."),
    "carnivorous_plant_default": SkinDefinition(
        "carnivorous_plant_default",
        PetType.CARNIVOROUS_PLANT,
        "Обычное растение",
        "Базовый облик.",
        True,
        False,
    ),
    "carnivorous_plant_lab": SkinDefinition(
        "carnivorous_plant_lab",
        PetType.CARNIVOROUS_PLANT,
        "Лабораторное растение",
        "Стиль исследовательской станции.",
    ),
    "carnivorous_plant_neon": SkinDefinition(
        "carnivorous_plant_neon",
        PetType.CARNIVOROUS_PLANT,
        "Неоновое растение",
        "Яркий контрастный стиль.",
    ),
    "horse_default": SkinDefinition("horse_default", PetType.HORSE, "Обычная лошадь", "Базовый облик.", True, False),
    "horse_worker": SkinDefinition("horse_worker", PetType.HORSE, "Рабочая лошадь", "Практичный рабочий стиль."),
    "horse_cyber": SkinDefinition("horse_cyber", PetType.HORSE, "Кибер-лошадь", "Технический неоновый стиль."),
    "frog_default": SkinDefinition("frog_default", PetType.FROG, "Обычная лягушка", "Базовый облик.", True, False),
    "frog_swamp": SkinDefinition("frog_swamp", PetType.FROG, "Болотная лягушка", "Естественный зеленый стиль."),
    "frog_quantum": SkinDefinition("frog_quantum", PetType.FROG, "Квантовая лягушка", "Футуристический стиль."),
    "hamster_default": SkinDefinition("hamster_default", PetType.HAMSTER, "Обычный хомяк", "Базовый облик.", True, False),
    "hamster_engineer": SkinDefinition("hamster_engineer", PetType.HAMSTER, "Хомяк-инженер", "Практичный технический стиль."),
    "hamster_archivist": SkinDefinition("hamster_archivist", PetType.HAMSTER, "Хомяк-архивист", "Стиль хранителя коллекции."),
}

DEFAULT_SKIN_BY_PET: dict[PetType, str] = {
    definition.pet_type: definition.skin_id for definition in SKIN_DEFINITIONS.values() if definition.is_default
}

PAID_SKIN_IDS: tuple[str, ...] = tuple(
    skin_id for skin_id, definition in SKIN_DEFINITIONS.items() if not definition.is_default and definition.collectible
)

SKIN_IDS_BY_PET: dict[PetType, tuple[str, ...]] = {
    pet_type: tuple(skin_id for skin_id, skin in SKIN_DEFINITIONS.items() if skin.pet_type == pet_type)
    for pet_type in PET_DEFINITIONS
}

PAID_SKIN_IDS_BY_PET: dict[PetType, tuple[str, ...]] = {
    pet_type: tuple(
        skin_id
        for skin_id, skin in SKIN_DEFINITIONS.items()
        if skin.pet_type == pet_type and not skin.is_default and skin.collectible
    )
    for pet_type in PET_DEFINITIONS
}
