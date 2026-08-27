from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Mapping

from neuralimage.configuration.sem_segmentation import (
    SemSegmentationConfig,
    build_sem_segmentation_config,
)
from neuralimage.targets.config import BASIC_TARGET_NAMES, GEOMETRY_TARGET_NAMES


FieldKind = Literal['bool', 'int', 'float', 'choice', 'text']
PathPart = str | int


@dataclass(frozen=True)
class SemUiField:
    key: str
    path: tuple[PathPart, ...]
    section: str
    kind: FieldKind
    label_en: str
    default: Any
    minimum: float | int | None = None
    maximum: float | int | None = None
    step: float | int | None = None
    decimals: int = 3
    choices: tuple[tuple[str, str], ...] = ()

    @property
    def form_name(self) -> str:
        return f'sem__{self.key}'


SEM_UI_SECTIONS: tuple[tuple[str, str], ...] = (
    ('preprocessing', 'Preprocessing'),
    ('augmentation', 'SEM augmentation'),
    ('basic_targets', 'Basic supervision'),
    ('geometry_targets', 'Geometry supervision'),
    ('losses', 'Loss weighting'),
    ('hard_mining', 'Hard example mining'),
    ('context', 'Context branch'),
    ('uncertainty', 'Confidence and uncertainty'),
    ('active_learning', 'Active Learning export'),
    ('validation', 'Validation'),
    ('experiment', 'Experiment'),
)


def _field(
    key: str,
    path: tuple[PathPart, ...],
    section: str,
    kind: FieldKind,
    label: str,
    default: Any,
    minimum: float | int | None = None,
    maximum: float | int | None = None,
    step: float | int | None = None,
    decimals: int = 3,
    choices: tuple[tuple[str, str], ...] = (),
) -> SemUiField:
    return SemUiField(key, path, section, kind, label, default, minimum, maximum, step, decimals, choices)


SEM_UI_FIELDS: tuple[SemUiField, ...] = (
    # Shared preprocessing.
    _field(
        'pre_mode',
        ('preprocessing', 'mode'),
        'preprocessing',
        'choice',
        'Normalization mode',
        'none',
        choices=(
            ('none', 'None'),
            ('per_image_percentile', 'Per-image percentile (P1–P99)'),
            ('dataset_zscore', 'Dataset z-score (train statistics)'),
        ),
    ),
    # Acquisition augmentation.
    _field('aug_enabled', ('augmentation', 'enabled'), 'augmentation', 'bool', 'Enable SEM augmentation', False),
    _field('aug_plan', ('augmentation', 'plan'), 'augmentation', 'choice', 'Augmentation plan', 'legacy_v1', choices=(('legacy_v1', 'Legacy v1'), ('sem_v2', 'SEM v2'))),
    *tuple(
        item
        for prefix, enabled_key, probability_key, label, default_probability in (
            ('charging', 'charging_artifacts', 'charging_probability', 'Charging artifacts', 0.15),
            ('drift', 'scan_drift', 'scan_drift_probability', 'Scan drift', 0.10),
            ('focus', 'local_focus_variation', 'focus_variation_probability', 'Local focus variation', 0.12),
            ('noise', 'detector_noise', 'detector_noise_probability', 'Detector noise', 0.20),
            ('gradient', 'brightness_gradients', 'brightness_gradient_probability', 'Brightness gradients', 0.15),
            ('defects', 'realistic_defects', 'realistic_defect_probability', 'Realistic scan defects', 0.10),
        )
        for item in (
            _field(f'aug_{prefix}', ('augmentation', enabled_key), 'augmentation', 'bool', label, True),
            _field(f'aug_{prefix}_probability', ('augmentation', probability_key), 'augmentation', 'float', f'{label} probability', default_probability, 0.0, 1.0, 0.01, 2),
        )
    ),
    _field('aug_charging_strength', ('augmentation', 'charging_strength'), 'augmentation', 'float', 'Charging strength', 0.25, 0.0, 5.0, 0.05, 2),
    _field('aug_drift_pixels', ('augmentation', 'drift_max_pixels'), 'augmentation', 'float', 'Maximum drift, px', 3.0, 0.0, 128.0, 0.25, 2),
    _field('aug_focus_sigma', ('augmentation', 'focus_sigma_max'), 'augmentation', 'float', 'Maximum focus sigma', 2.5, 0.0, 32.0, 0.1, 2),
    _field('aug_peak_electrons', ('augmentation', 'detector_peak_electrons'), 'augmentation', 'float', 'Detector peak electrons', 80.0, 0.01, 100000.0, 1.0, 2),
    _field('aug_read_noise', ('augmentation', 'read_noise_sigma'), 'augmentation', 'float', 'Read noise sigma', 0.015, 0.0, 10.0, 0.005, 4),
    _field('aug_gain_strength', ('augmentation', 'gain_field_strength'), 'augmentation', 'float', 'Gain-field strength', 0.2, 0.0, 5.0, 0.05, 2),
    # Basic targets and fixed-scale geometry.
    *tuple(_field(f'target_{name}', ('targets', 'basic', name), 'basic_targets', 'bool', name.replace('_', ' ').title(), False) for name in BASIC_TARGET_NAMES),
    _field('target_boundary_kernel', ('targets', 'basic', 'boundary_kernel_size'), 'basic_targets', 'int', 'Boundary kernel', 3, 1, 99, 2),
    _field('target_skeleton_iterations', ('targets', 'basic', 'skeleton_iterations'), 'basic_targets', 'int', 'Bounded thinning iterations (0 = convergence)', 0, 0, 10000, 1),
    _field('target_sdf_clip', ('targets', 'basic', 'sdf_clip'), 'basic_targets', 'float', 'SDF clipping distance', 32.0, 0.01, 10000.0, 1.0, 2),
    _field('target_distance_clip', ('targets', 'basic', 'distance_clip'), 'basic_targets', 'float', 'Distance clipping distance', 32.0, 0.01, 10000.0, 1.0, 2),
    _field('target_thickness_max', ('targets', 'basic', 'thickness_max'), 'basic_targets', 'float', 'Maximum local thickness', 64.0, 0.01, 10000.0, 1.0, 2),
    _field('target_border_ignore', ('targets', 'basic', 'border_ignore'), 'basic_targets', 'int', 'Ignored crop border', 2, 0, 1024, 1),
    _field('target_cldice_iterations', ('targets', 'basic', 'cldice_iterations'), 'basic_targets', 'int', 'clDice thinning iterations', 10, 1, 1000, 1),
    _field('target_distance_boundary_weight', ('targets', 'distance_boundary_weight'), 'basic_targets', 'float', 'SDF boundary-loss weight', 0.0, 0.0, 100.0, 0.05, 3),
    _field('target_cache', ('targets', 'cache_enabled'), 'basic_targets', 'bool', 'Cache deterministic targets', False),
    _field('target_cache_size', ('targets', 'cache_size'), 'basic_targets', 'int', 'Target cache entries', 256, 1, 1000000, 1),
    *tuple(_field(f'target_{name}', ('targets', 'geometry', name), 'geometry_targets', 'bool', name.replace('_', ' ').title(), False) for name in GEOMETRY_TARGET_NAMES),
    _field('geometry_corner_sigma', ('targets', 'geometry', 'corner_sigma'), 'geometry_targets', 'float', 'Corner heatmap sigma', 1.5, 0.01, 100.0, 0.1, 2),
    _field('geometry_junction_degree', ('targets', 'geometry', 'junction_min_degree'), 'geometry_targets', 'int', 'Minimum junction degree', 3, 3, 8, 1),
    _field('geometry_orientation_bins', ('targets', 'geometry', 'orientation_bins'), 'geometry_targets', 'int', 'Orientation bins (0 = vector)', 0, 0, 360, 1),
    _field('geometry_orientation_radius', ('targets', 'geometry', 'orientation_radius'), 'geometry_targets', 'int', 'Orientation radius', 5, 1, 255, 1),
    _field('geometry_border_ignore', ('targets', 'geometry', 'border_ignore'), 'geometry_targets', 'int', 'Ignored geometry border', 3, 0, 1024, 1),
    *tuple(
        _field(f'weight_{name}', ('targets', 'auxiliary_head_weights', name), 'geometry_targets' if name in GEOMETRY_TARGET_NAMES else 'basic_targets', 'float', f'{name.replace("_", " ").title()} loss weight', 0.1, 0.0, 100.0, 0.05, 3)
        for name in BASIC_TARGET_NAMES + GEOMETRY_TARGET_NAMES
    ),
    # Losses, mining and model context.
    _field('loss_strategy', ('losses', 'weighting_strategy'), 'losses', 'choice', 'Task weighting', 'static', choices=(('static', 'Static'), ('homoscedastic_uncertainty', 'Learned uncertainty'))),
    _field('loss_mask_floor', ('losses', 'mask_weight_floor'), 'losses', 'float', 'Minimum mask-loss weight', 0.25, 0.001, 1.0, 0.05, 3),
    _field('hard_mode', ('hard_mining', 'mode'), 'hard_mining', 'choice', 'Hard-mining mode', 'off', choices=(('off', 'Off'), ('online', 'Online'), ('offline', 'Offline'), ('online_and_offline', 'Online + offline'))),
    _field('hard_geometry_weight', ('hard_mining', 'geometry_weight'), 'hard_mining', 'float', 'Geometry score weight', 0.5, 0.0, 100.0, 0.05, 3),
    _field('hard_loss_weight', ('hard_mining', 'loss_weight'), 'hard_mining', 'float', 'Historical loss weight', 0.5, 0.0, 100.0, 0.05, 3),
    _field('hard_exploration', ('hard_mining', 'exploration_floor'), 'hard_mining', 'float', 'Exploration floor', 0.1, 0.0, 1.0, 0.01, 3),
    _field('hard_ema', ('hard_mining', 'ema_alpha'), 'hard_mining', 'float', 'Loss EMA alpha', 0.1, 0.001, 1.0, 0.01, 3),
    _field('hard_clip', ('hard_mining', 'score_clip'), 'hard_mining', 'float', 'Score clipping', 5.0, 0.001, 100000.0, 0.5, 3),
    _field('hard_refresh', ('hard_mining', 'refresh_epochs'), 'hard_mining', 'int', 'Refresh every epochs', 1, 1, 100000, 1),
    _field('hard_manifest', ('hard_mining', 'offline_manifest'), 'hard_mining', 'text', 'Offline manifest path', ''),
    _field('context_enabled', ('context', 'enabled'), 'context', 'bool', 'Enable context branch', False),
    _field('context_fusion', ('context', 'fusion_type'), 'context', 'choice', 'Fusion type', 'concat', choices=(('concat', 'Concatenate'), ('add', 'Add'))),
    _field('context_attention', ('context', 'cross_attention'), 'context', 'bool', 'Cross-attention fusion', True),
    _field('context_dim', ('context', 'attention_dim'), 'context', 'int', 'Attention dimension', 128, 1, 16384, 1),
    _field('context_heads', ('context', 'attention_heads'), 'context', 'int', 'Attention heads', 4, 1, 1024, 1),
    _field('context_tokens', ('context', 'max_global_tokens'), 'context', 'int', 'Maximum global tokens', 1024, 1, 1000000, 1),
    # Confidence, AL and validation.
    _field('uncertainty_enabled', ('uncertainty', 'enabled'), 'uncertainty', 'bool', 'Enable uncertainty estimation', False),
    _field('uncertainty_method', ('uncertainty', 'method'), 'uncertainty', 'choice', 'Uncertainty method', 'confidence_head', choices=(('confidence_head', 'Confidence head'), ('mc_dropout', 'MC Dropout'), ('tta_variance', 'TTA variance'), ('combined', 'Combined'), ('auto', 'Automatic'))),
    _field('uncertainty_samples', ('uncertainty', 'mc_dropout_samples'), 'uncertainty', 'int', 'MC Dropout samples', 8, 2, 1000, 1),
    _field('uncertainty_rate', ('uncertainty', 'mc_dropout_rate'), 'uncertainty', 'float', 'MC Dropout rate', 0.1, 0.0, 0.999, 0.01, 3),
    _field('uncertainty_tta_flips', ('uncertainty', 'tta_flips'), 'uncertainty', 'bool', 'TTA flips', True),
    _field('uncertainty_tta_rotations', ('uncertainty', 'tta_rotations'), 'uncertainty', 'bool', 'TTA rotations', False),
    _field('uncertainty_export', ('uncertainty', 'export_confidence_map'), 'uncertainty', 'bool', 'Export confidence map', True),
    _field('uncertainty_loss_weight', ('uncertainty', 'confidence_loss_weight'), 'uncertainty', 'float', 'Confidence loss weight', 0.1, 0.0, 100.0, 0.05, 3),
    _field('al_enabled', ('active_learning', 'enabled'), 'active_learning', 'bool', 'Export NeedsAnnotation samples', False),
    _field('al_export_dir', ('active_learning', 'export_dir'), 'active_learning', 'text', 'NeedsAnnotation directory', ''),
    _field('al_low_confidence', ('active_learning', 'low_confidence_threshold'), 'active_learning', 'float', 'Low-confidence threshold', 0.35, 0.0, 1.0, 0.01, 3),
    _field('al_entropy', ('active_learning', 'high_entropy_threshold'), 'active_learning', 'float', 'High-entropy threshold', 0.65, 0.0, 1.0, 0.01, 3),
    _field('al_instability', ('active_learning', 'instability_threshold'), 'active_learning', 'float', 'Instability threshold', 0.15, 0.0, 1.0, 0.01, 3),
    _field('al_disagreement', ('active_learning', 'disagreement_threshold'), 'active_learning', 'float', 'Disagreement threshold', 0.2, 0.0, 1.0, 0.01, 3),
    _field('al_max_exports', ('active_learning', 'max_exports_per_run'), 'active_learning', 'int', 'Maximum exports per run', 256, 1, 1000000, 1),
    _field('al_max_rois', ('active_learning', 'max_rois_per_frame'), 'active_learning', 'int', 'Maximum ROIs per frame', 16, 1, 1000000, 1),
    _field('al_min_area', ('active_learning', 'min_roi_area'), 'active_learning', 'int', 'Minimum ROI area', 64, 1, 1000000000, 1),
    _field('al_padding', ('active_learning', 'roi_padding'), 'active_learning', 'int', 'ROI padding', 16, 0, 100000, 1),
    _field('al_merge', ('active_learning', 'merge_distance'), 'active_learning', 'int', 'ROI merge distance', 8, 0, 100000, 1),
    _field('validation_enabled', ('validation', 'enabled'), 'validation', 'bool', 'Enable advanced validation', True),
    _field('validation_full_frame', ('validation', 'full_frame'), 'validation', 'bool', 'Validate stitched full frames', True),
    _field('validation_tolerance', ('validation', 'boundary_tolerance'), 'validation', 'int', 'Boundary tolerance, px', 2, 0, 1024, 1),
    _field('validation_hd95', ('validation', 'include_hd95'), 'validation', 'bool', 'Calculate HD95', True),
    _field('validation_bins', ('validation', 'confidence_bins'), 'validation', 'int', 'Confidence histogram bins', 10, 2, 10000, 1),
    _field('experiment_topology_first', ('experiment', 'topology_first'), 'experiment', 'bool', 'Topology-first model selection', True),
    _field('experiment_seed_1', ('experiment', 'seeds', 0), 'experiment', 'int', 'Seed 1', 17, 0, 2147483647, 1),
    _field('experiment_seed_2', ('experiment', 'seeds', 1), 'experiment', 'int', 'Seed 2', 29, 0, 2147483647, 1),
    _field('experiment_seed_3', ('experiment', 'seeds', 2), 'experiment', 'int', 'Seed 3', 43, 0, 2147483647, 1),
    _field('experiment_manifest', ('experiment', 'dataset_manifest'), 'experiment', 'text', 'Dataset manifest path', ''),
)


SEM_UI_FIELDS_BY_KEY = {field.key: field for field in SEM_UI_FIELDS}

SEM_UI_SECTION_LABELS_RU = {
    'preprocessing': 'Препроцессинг',
    'augmentation': 'SEM-аугментация',
    'basic_targets': 'Основные цели',
    'geometry_targets': 'Геометрия',
    'losses': 'Взвешивание потерь',
    'hard_mining': 'Сложные примеры',
    'context': 'Контекстная ветвь',
    'uncertainty': 'Уверенность',
    'active_learning': 'Active Learning',
    'validation': 'Валидация',
    'experiment': 'Эксперимент',
}

SEM_UI_SECTION_HELP_EN = {
    'preprocessing': 'Shared deterministic SEM preprocessing. The same operations and order are used for training and recognition.',
    'augmentation': 'Models SEM acquisition defects in training images without changing their masks.',
    'basic_targets': 'Auxiliary targets generated automatically from each binary mask; no additional annotation is required.',
    'geometry_targets': 'Geometry and topology targets reconstructed automatically from the raster binary mask.',
    'losses': 'Controls how the main mask loss and auxiliary training tasks are balanced.',
    'hard_mining': 'Prioritizes geometrically important or previously difficult training patches while retaining random exploration.',
    'context': 'Controls the model branch that combines a local patch with a larger image context.',
    'uncertainty': 'Estimates prediction uncertainty and optionally exports a confidence map; it does not replace the binary mask.',
    'active_learning': 'Exports the most uncertain full-frame regions to NeedsAnnotation for later expert review.',
    'validation': 'Adds topology- and boundary-aware metrics to the existing validation dataset.',
    'experiment': 'Makes topology-first comparison runs reproducible. These settings do not change inference output.',
}

SEM_UI_SECTION_HELP_RU = {
    'preprocessing': 'Нормализация SEM, одинаковая для обучения и распознавания. Для dataset z-score статистика вычисляется только по train-части.',
    'augmentation': 'Имитирует дефекты получения SEM-изображения только при обучении и не изменяет маску.',
    'basic_targets': 'Вспомогательные цели автоматически строятся из бинарной маски; дополнительная разметка не нужна.',
    'geometry_targets': 'Геометрия и топология автоматически восстанавливаются из растровой бинарной маски.',
    'losses': 'Настраивает баланс основной ошибки маски и вспомогательных обучающих задач.',
    'hard_mining': 'Чаще выбирает геометрически важные или ранее сложные патчи, сохраняя долю случайных примеров.',
    'context': 'Настраивает ветвь модели, объединяющую локальный патч с более крупным контекстом изображения.',
    'uncertainty': 'Оценивает неопределённость прогноза и при необходимости сохраняет карту уверенности; бинарную маску не заменяет.',
    'active_learning': 'Экспортирует наиболее неопределённые области целого кадра в NeedsAnnotation для последующей проверки экспертом.',
    'validation': 'Добавляет к существующей валидации метрики границ и сохранения топологии.',
    'experiment': 'Обеспечивает воспроизводимое topology-first сравнение запусков и не изменяет результат распознавания.',
}

SEM_UI_LABELS_RU = {
    'pre_mode': 'Режим нормализации',
    'aug_enabled': 'Включить SEM-аугментацию',
    'aug_plan': 'Набор аугментаций',
    'aug_charging': 'Эффекты заряда',
    'aug_charging_probability': 'Вероятность эффекта заряда',
    'aug_drift': 'Дрейф сканирования',
    'aug_drift_probability': 'Вероятность дрейфа',
    'aug_focus': 'Локальная расфокусировка',
    'aug_focus_probability': 'Вероятность расфокусировки',
    'aug_noise': 'Шум детектора',
    'aug_noise_probability': 'Вероятность шума детектора',
    'aug_gradient': 'Градиенты яркости',
    'aug_gradient_probability': 'Вероятность градиента яркости',
    'aug_defects': 'Реалистичные дефекты сканирования',
    'aug_defects_probability': 'Вероятность дефекта сканирования',
    'aug_charging_strength': 'Сила эффекта заряда',
    'aug_drift_pixels': 'Максимальный дрейф, пикс.',
    'aug_focus_sigma': 'Максимальная сигма расфокусировки',
    'aug_peak_electrons': 'Пиковое число электронов детектора',
    'aug_read_noise': 'Сигма шума считывания',
    'aug_gain_strength': 'Сила поля усиления',
    'target_boundary': 'Граница',
    'target_skeleton': 'Скелет',
    'target_sdf': 'Знаковое поле расстояний (SDF)',
    'target_distance_transform': 'Преобразование расстояний',
    'target_thickness': 'Локальная толщина',
    'target_vertex': 'Вершины полигонов',
    'target_corner': 'Тепловая карта углов',
    'target_endpoint': 'Концевые точки',
    'target_junction': 'T/X/Y-пересечения',
    'target_orientation': 'Поле ориентации',
    'target_tangent': 'Касательная центральной линии',
    'target_curvature': 'Кривизна',
    'target_topology': 'Топологические критические линии',
    'target_boundary_kernel': 'Ядро карты границ',
    'target_skeleton_iterations': 'Итерации thinning (0 = до сходимости)',
    'target_sdf_clip': 'Ограничение расстояния SDF',
    'target_distance_clip': 'Ограничение distance transform',
    'target_thickness_max': 'Максимальная локальная толщина',
    'target_border_ignore': 'Игнорируемая граница кропа',
    'target_cldice_iterations': 'Итерации thinning для clDice',
    'target_distance_boundary_weight': 'Вес SDF boundary loss',
    'target_cache': 'Кэшировать детерминированные цели',
    'target_cache_size': 'Размер кэша целей',
    'geometry_corner_sigma': 'Сигма тепловой карты углов',
    'geometry_junction_degree': 'Минимальная степень пересечения',
    'geometry_orientation_bins': 'Бины ориентации (0 = вектор)',
    'geometry_orientation_radius': 'Радиус оценки ориентации',
    'geometry_border_ignore': 'Игнорируемая граница геометрии',
    **{
        f'weight_{name}': f'Вес loss: {translated_label}'
        for name, translated_label in {
            'boundary': 'граница', 'skeleton': 'скелет', 'sdf': 'SDF',
            'distance_transform': 'расстояние', 'thickness': 'толщина',
            'vertex': 'вершины', 'corner': 'углы', 'endpoint': 'концы',
            'junction': 'пересечения', 'orientation': 'ориентация',
            'tangent': 'касательная', 'curvature': 'кривизна', 'topology': 'топология',
        }.items()
    },
    'loss_strategy': 'Стратегия взвешивания задач',
    'loss_mask_floor': 'Минимальный вес mask loss',
    'hard_mode': 'Режим hard mining',
    'hard_geometry_weight': 'Вес геометрической сложности',
    'hard_loss_weight': 'Вес исторического loss',
    'hard_exploration': 'Минимальная доля исследования',
    'hard_ema': 'EMA-коэффициент loss',
    'hard_clip': 'Ограничение оценки сложности',
    'hard_refresh': 'Обновлять каждые N эпох',
    'hard_manifest': 'Путь к offline-манифесту',
    'context_enabled': 'Включить контекстную ветвь',
    'context_fusion': 'Способ объединения',
    'context_attention': 'Cross-attention',
    'context_dim': 'Размерность attention',
    'context_heads': 'Количество attention heads',
    'context_tokens': 'Максимум глобальных токенов',
    'uncertainty_enabled': 'Оценивать неопределённость',
    'uncertainty_method': 'Метод неопределённости',
    'uncertainty_samples': 'Число проходов MC Dropout',
    'uncertainty_rate': 'Вероятность MC Dropout',
    'uncertainty_tta_flips': 'Отражения для TTA',
    'uncertainty_tta_rotations': 'Повороты для TTA',
    'uncertainty_export': 'Экспортировать карту уверенности',
    'uncertainty_loss_weight': 'Вес confidence loss',
    'al_enabled': 'Экспортировать NeedsAnnotation',
    'al_export_dir': 'Каталог NeedsAnnotation',
    'al_low_confidence': 'Порог низкой уверенности',
    'al_entropy': 'Порог высокой энтропии',
    'al_instability': 'Порог нестабильности',
    'al_disagreement': 'Порог расхождения',
    'al_max_exports': 'Максимум экспортов за запуск',
    'al_max_rois': 'Максимум ROI на кадр',
    'al_min_area': 'Минимальная площадь ROI',
    'al_padding': 'Отступ ROI',
    'al_merge': 'Расстояние объединения ROI',
    'validation_enabled': 'Включить расширенную валидацию',
    'validation_full_frame': 'Валидировать склеенные полные кадры',
    'validation_tolerance': 'Допуск границы, пикс.',
    'validation_hd95': 'Вычислять HD95',
    'validation_bins': 'Интервалы гистограммы уверенности',
    'experiment_topology_first': 'Topology-first выбор модели',
    'experiment_seed_1': 'Seed 1',
    'experiment_seed_2': 'Seed 2',
    'experiment_seed_3': 'Seed 3',
    'experiment_manifest': 'Путь к манифесту датасета',
}

SEM_UI_CHOICE_LABELS_RU = {
    'none': 'Без нормализации',
    'per_image_percentile': 'Процентили каждого кадра (P1–P99)',
    'dataset_zscore': 'Z-score датасета (статистика train)',
    'rows': 'Строки', 'columns': 'Столбцы', 'legacy_v1': 'Legacy v1', 'sem_v2': 'SEM v2',
    'static': 'Фиксированные веса', 'homoscedastic_uncertainty': 'Обучаемая неопределённость',
    'off': 'Выключено', 'online': 'Онлайн', 'offline': 'Офлайн',
    'online_and_offline': 'Онлайн + офлайн', 'concat': 'Конкатенация', 'add': 'Сложение',
    'confidence_head': 'Голова уверенности', 'mc_dropout': 'MC Dropout',
    'tta_variance': 'Дисперсия TTA', 'combined': 'Комбинированный', 'auto': 'Автоматически',
}


def sem_ui_section_label(section: str, english_label: str, language: str) -> str:
    return SEM_UI_SECTION_LABELS_RU.get(section, english_label) if str(language).startswith('ru') else english_label


def sem_ui_field_label(field: SemUiField, language: str) -> str:
    return SEM_UI_LABELS_RU.get(field.key, field.label_en) if str(language).startswith('ru') else field.label_en


def sem_ui_choice_label(value: str, english_label: str, language: str) -> str:
    return SEM_UI_CHOICE_LABELS_RU.get(value, english_label) if str(language).startswith('ru') else english_label


def sem_ui_section_help(section: str, language: str) -> str:
    descriptions = SEM_UI_SECTION_HELP_RU if str(language).startswith('ru') else SEM_UI_SECTION_HELP_EN
    return descriptions.get(section, '')


def sem_ui_field_help(field: SemUiField, language: str) -> str:
    section_help = sem_ui_section_help(field.section, language)
    label = sem_ui_field_label(field, language)
    if field.kind in {'int', 'float'} and field.minimum is not None and field.maximum is not None:
        bounds = (
            f' Допустимый диапазон: {field.minimum}–{field.maximum}.'
            if str(language).startswith('ru')
            else f' Allowed range: {field.minimum}–{field.maximum}.'
        )
    else:
        bounds = ''
    prefix = f'{label}. ' if label else ''
    return f'{prefix}{section_help}{bounds}'.strip()


def _read_path(payload: Any, path: tuple[PathPart, ...], default: Any) -> Any:
    value = payload
    for part in path:
        if isinstance(part, int):
            if not isinstance(value, (list, tuple)) or part >= len(value):
                return default
            value = value[part]
        elif isinstance(value, Mapping):
            value = value.get(part, default)
        else:
            return default
    return default if value is None else value


def _write_path(payload: dict[str, Any], path: tuple[PathPart, ...], value: Any) -> None:
    current: Any = payload
    for index, part in enumerate(path[:-1]):
        following = path[index + 1]
        if isinstance(part, int):
            while len(current) <= part:
                current.append({} if isinstance(following, str) else None)
            current = current[part]
        else:
            if part not in current or not isinstance(current[part], (dict, list)):
                current[part] = [] if isinstance(following, int) else {}
            current = current[part]
    final = path[-1]
    if isinstance(final, int):
        while len(current) <= final:
            current.append(None)
        current[final] = value
    else:
        current[final] = value


def sem_config_to_form_values(payload: Mapping[str, Any] | None) -> dict[str, Any]:
    canonical = build_sem_segmentation_config(payload).to_dict()
    return {
        field.form_name: _read_path(canonical, field.path, field.default)
        for field in SEM_UI_FIELDS
    } | {'sem_preset': str(canonical.get('preset', 'legacy_v1'))}


def sem_config_from_form_values(values: Mapping[str, Any], *, preset: str | None = None) -> dict[str, Any]:
    raw = SemSegmentationConfig().to_dict()
    resolved_preset = str(preset or values.get('sem_preset') or 'custom').strip() or 'custom'
    raw['preset'] = resolved_preset
    for field in SEM_UI_FIELDS:
        value = values.get(field.form_name, field.default)
        if value is None or (field.kind == 'choice' and value == ''):
            value = field.default
        _write_path(raw, field.path, value)

    enabled_targets = [
        name
        for name in BASIC_TARGET_NAMES + GEOMETRY_TARGET_NAMES
        if bool(values.get(f'sem__target_{name}', False))
    ]
    if 'sdf' not in enabled_targets:
        raw['targets']['distance_boundary_weight'] = 0.0
    raw['heads']['enabled'] = enabled_targets
    resolved_weights = {
        name: float(values.get(f'sem__weight_{name}') or 0.0)
        for name in enabled_targets
    }
    raw['targets']['auxiliary_head_weights'] = {
        name: weight
        for name, weight in resolved_weights.items()
        if weight > 0.0
    }
    config = build_sem_segmentation_config(raw)
    if config.to_dict() == SemSegmentationConfig().to_dict():
        return {}
    return config.to_dict()


def fields_for_section(section: str) -> tuple[SemUiField, ...]:
    return tuple(field for field in SEM_UI_FIELDS if field.section == section)
