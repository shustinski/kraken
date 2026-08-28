"""Localized text for conductor-recognition strategy metadata."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..vision.metal_recovery.strategy_registry import ParameterSpec, StrategySpec


_STRATEGY_NAMES_RU = {
    "auto": "Авто (контроль топологии)",
    "legacy_otsu": "Порог Otsu",
    "local_adaptive": "Адаптивный порог",
    "gradient_watershed": "Градиентный водораздел",
    "random_walker": "Случайное блуждание",
    "graph_cut": "Графовый разрез",
    "reconstruction": "Морфологическая реконструкция",
    "closed_boundary": "Замкнутые границы",
    "structural_watershed": "Структурный водораздел",
    "owt_ucm": "OWT-UCM",
    "graph_multi_separator": "Графовый мульти-разделитель",
    "gasp": "GASP",
    "mutex_watershed": "Взаимоисключающий водораздел",
    "multicut": "Мультиразрез",
    "lifted_multicut": "Расширенный мультиразрез",
    "ic_sem_expert": "Экспертные правила IC-SEM",
}

_STRATEGY_DESCRIPTIONS_RU = {
    "auto": "Автоматический выбор производственного pipeline с контролем топологии.",
    "legacy_otsu": "Глобальный порог Otsu с последующей морфологической обработкой.",
    "local_adaptive": "Локальный адаптивный порог для неравномерно освещённых кадров.",
    "gradient_watershed": "Водораздел по градиенту с затравками металла и подложки.",
    "random_walker": "Сегментация случайным блужданием по общим затравкам.",
    "graph_cut": "Графовый разрез по общим затравкам металла и подложки.",
    "reconstruction": "Морфологическая реконструкция проводников из надёжных ядер.",
    "closed_boundary": "Выделение областей, окружённых устойчивыми границами.",
    "structural_watershed": "Структурный водораздел с сохранением экземпляров проводников.",
    "owt_ucm": "Иерархическая сегментация по ориентированной силе контуров.",
    "graph_multi_separator": "Разбиение графа с явными пикселями-разделителями и дальними взаимодействиями.",
    "gasp": "Агломеративная кластеризация графа притяжения и отталкивания.",
    "mutex_watershed": "Быстрое знаковое разбиение графа с взаимоисключающими связями.",
    "multicut": "Глобальное разбиение графа с минимизацией стоимости притяжения и отталкивания.",
    "lifted_multicut": "Мультиразрез с дополнительными дальними связями для контроля топологии.",
    "ic_sem_expert": "Зарезервированная точка расширения для правил проектирования IC-SEM. Непроверенный алгоритм не реализован.",
}

_PARAMETER_LABELS_RU = {
    "Affinity normalization": "Нормализация сходства",
    "Affinity temperature": "Температура сходства",
    "Ambiguous region policy": "Обработка неоднозначных областей",
    "Atomic region scale": "Масштаб атомарных областей",
    "Atomic segmentation method": "Метод атомарной сегментации",
    "Attraction cost scale": "Масштаб стоимости притяжения",
    "Attractive neighborhood offsets": "Окрестность связей притяжения",
    "Attractive weight scale": "Масштаб веса притяжения",
    "Boundary aggregation": "Агрегация границ",
    "Boundary repulsion weight": "Вес отталкивания границы",
    "Boundary separator weight": "Вес границы-разделителя",
    "Boundary-side evidence weight": "Вес признака по сторонам границы",
    "Connectivity": "Связность",
    "Contour continuity weight": "Вес непрерывности контура",
    "Contour smoothing sigma": "Сглаживание контура, σ",
    "Contour source": "Источник контура",
    "Convergence tolerance": "Допуск сходимости",
    "Core metal evidence weight": "Вес признака ядра металла",
    "Core/interior attraction weight": "Вес притяжения внутренней области",
    "Cost transform": "Преобразование стоимости",
    "Cross-boundary lifted repulsion": "Дальнее отталкивание через границу",
    "Edge ordering / normalization": "Порядок и нормализация рёбер",
    "Fallback minimum core fraction": "Мин. доля ядер для fallback",
    "Fallback missing-core fraction": "Доля пропущенных ядер для fallback",
    "GASP linkage criterion": "Критерий объединения GASP",
    "Gradient field sensitivity": "Чувствительность градиентного поля",
    "Gradient repulsion weight": "Вес градиентного отталкивания",
    "Graph domain": "Область графа",
    "Hierarchy level": "Уровень иерархии",
    "Initialization": "Инициализация",
    "Intensity affinity weight": "Вес сходства яркости",
    "Intensity attraction weight": "Вес притяжения по яркости",
    "Intensity evidence weight": "Вес яркостного признака",
    "Lifted attraction weight": "Вес дальнего притяжения",
    "Lifted confidence threshold": "Порог уверенности дальних связей",
    "Lifted distance step": "Шаг дальних связей",
    "Lifted edges enabled": "Использовать дальние связи",
    "Lifted repulsion weight": "Вес дальнего отталкивания",
    "Local connectivity": "Локальная связность",
    "Local contrast attraction weight": "Вес притяжения по локальному контрасту",
    "Local contrast evidence weight": "Вес локального контраста",
    "Long-range attraction weight": "Вес дальнего притяжения",
    "Long-range interactions enabled": "Использовать дальние взаимодействия",
    "Long-range mutex distance": "Дальность взаимоисключающих связей",
    "Long-range radius": "Радиус дальних взаимодействий",
    "Long-range repulsion weight": "Вес дальнего отталкивания",
    "Maximum graph distance": "Макс. расстояние в графе",
    "Maximum iterations": "Макс. число итераций",
    "Maximum lifted distance": "Макс. дальность связей",
    "Maximum lifted edges": "Макс. число дальних рёбер",
    "Maximum operations": "Макс. число операций",
    "Maximum repulsive conflict": "Макс. конфликт отталкивания",
    "Merge stopping threshold": "Порог остановки объединения",
    "Metal merge separator ceiling": "Макс. уверенность разделителя при слиянии металла",
    "Minimum atomic region area": "Мин. площадь атомарной области",
    "Minimum attractive confidence": "Мин. уверенность притяжения",
    "Minimum background confidence": "Мин. уверенность фона",
    "Minimum contour strength": "Мин. сила контура",
    "Minimum initial basin area": "Мин. площадь исходного бассейна",
    "Minimum lifted distance": "Мин. дальность связей",
    "Minimum merge affinity": "Мин. сходство для объединения",
    "Minimum metal confidence": "Мин. уверенность металла",
    "Minimum mutex confidence": "Мин. уверенность взаимоисключения",
    "Minimum output region area": "Мин. площадь выходной области",
    "Minimum region area": "Мин. площадь области",
    "Minimum repulsive confidence": "Мин. уверенность отталкивания",
    "Minimum separator confidence": "Мин. уверенность разделителя",
    "Minimum separator length": "Мин. длина разделителя",
    "Mutex neighborhood offsets": "Окрестность взаимоисключающих связей",
    "Mutex weight scale": "Масштаб веса взаимоисключения",
    "Native solver tile overlap": "Перекрытие плиток решателя",
    "Native solver tile size": "Размер плитки решателя",
    "Native solver workers": "Потоки нативного решателя",
    "Orientation attraction weight": "Вес притяжения по ориентации",
    "Orientation bins": "Число направлений",
    "Orientation consistency weight": "Вес согласованности ориентации",
    "Orientation smoothing sigma": "Сглаживание ориентации, σ",
    "Orientation-aligned lifted edges": "Дальние связи вдоль ориентации",
    "Orientation-coherent boundary weight": "Вес согласованной границы",
    "Paired-rim evidence threshold": "Порог признака парных кромок",
    "Paired-rim fallback enabled": "Fallback по парным кромкам",
    "Paired-rim ribbon recovery": "Восстановление лент по парным кромкам",
    "Probability / affinity bias": "Смещение вероятности сходства",
    "Project material separators": "Возвращать разделители в металл",
    "Projection core evidence": "Признак ядра для проекции",
    "Projection core margin": "Запас ядра для проекции",
    "Region unary weight": "Унарный вес области",
    "Repulsion cost scale": "Масштаб стоимости отталкивания",
    "Rim repulsion weight": "Вес отталкивания кромки",
    "Same-trace lifted attraction": "Дальнее притяжение одной трассы",
    "Separator continuity weight": "Вес непрерывности разделителя",
    "Separator projection radius": "Радиус проекции разделителя",
    "Separator unary weight": "Унарный вес разделителя",
    "Solver": "Решатель",
    "Solver / heuristic": "Решатель / эвристика",
    "Substrate evidence weight": "Вес признака подложки",
    "Time limit": "Ограничение времени",
    "Use signed edges": "Использовать знаковые рёбра",
    "Watershed minima suppression": "Подавление минимумов водораздела",
}

_CHOICE_LABELS_RU = {
    "Atomic regions": "Атомарные области",
    "Average linkage": "Средняя связь",
    "BSR dynamic mean": "Динамическое среднее BSR",
    "Background": "Фон",
    "Combined": "Комбинированный",
    "Descending confidence": "По убыванию уверенности",
    "Greedy Additive": "Жадный аддитивный",
    "Local": "Локальная",
    "Local graph offsets": "Локальные смещения графа",
    "Local plus diagonal": "Локальные и диагональные",
    "Local plus long-range": "Локальные и дальние",
    "Log odds": "Логарифм шансов",
    "Metal": "Металл",
    "Mutex/absolute-max linkage": "Взаимоисключение / абсолютный максимум",
    "Oriented gradient": "Ориентированный градиент",
    "Oriented watershed": "Ориентированный водораздел",
    "Pixels": "Пиксели",
    "Positive-edge components": "Компоненты положительных рёбер",
    "Preserve seed evidence": "Сохранить признак затравок",
    "Regular grid": "Регулярная сетка",
    "Signed linear": "Знаковое линейное",
    "Signed margin": "Знаковый запас",
    "Singleton regions": "Отдельные области",
    "Structural gradient": "Структурный градиент",
    "Sum linkage": "Суммарная связь",
    "Upstream greedy separator growing": "Жадное наращивание разделителя",
    "Upstream greedy separator shrinking": "Жадное сокращение разделителя",
    "Weighted mean": "Взвешенное среднее",
    "Weighted sum": "Взвешенная сумма",
}


def strategy_name(spec: StrategySpec, language: str) -> str:
    if language == "ru":
        return _STRATEGY_NAMES_RU.get(spec.strategy_id, spec.display_name)
    return spec.display_name


def strategy_description(spec: StrategySpec, language: str) -> str:
    if language == "ru":
        return _STRATEGY_DESCRIPTIONS_RU.get(spec.strategy_id, spec.description)
    return spec.description


def parameter_label(parameter: ParameterSpec, language: str) -> str:
    if language == "ru":
        return _PARAMETER_LABELS_RU.get(parameter.label, parameter.label)
    return parameter.label


def parameter_tooltip(parameter: ParameterSpec, language: str) -> str:
    if language == "ru":
        return f"Параметр выбранного алгоритма: {parameter_label(parameter, language).lower()}."
    return parameter.tooltip


def choice_label(label: str, language: str) -> str:
    if language == "ru":
        return _CHOICE_LABELS_RU.get(label, label)
    return label


__all__ = [
    "choice_label",
    "parameter_label",
    "parameter_tooltip",
    "strategy_description",
    "strategy_name",
]
