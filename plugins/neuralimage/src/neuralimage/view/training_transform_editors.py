from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QGroupBox,
    QHeaderView,
    QPushButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)


# Compatibility name retained for integrations that enumerate the former
# stacked editor sections. The visible editor now uses the first three blocks.
TRAINING_AUGMENTATION_SECTION_ORDER = (
    'sem_acquisition',
    'spatial',
    'photometric',
    'topology_variations',
    'batch',
    'synthetic',
)

AUGMENTATION_TOOLTIPS_RU = {
    'rotate_90': 'Поворачивает изображение и маску на 90°. Вероятность проверяется для каждого варианта.',
    'rotate_180': 'Поворачивает изображение и маску на 180° без изменения разметки.',
    'flip_x': 'Зеркально отражает изображение и маску по горизонтальной оси.',
    'flip_y': 'Зеркально отражает изображение и маску по вертикальной оси.',
    'scale': 'Случайно меняет масштаб патча и возвращает его к обучающему размеру.',
    'brightness': 'Случайно изменяет яркость изображения; бинарная маска не меняется.',
    'contrast': 'Случайно изменяет контраст изображения относительно среднего значения.',
    'gamma': 'Применяет нелинейное gamma-преобразование яркости.',
    'noise': 'Добавляет гауссов шум с указанным стандартным отклонением.',
    'blur': 'Применяет гауссово размытие с указанным максимальным радиусом.',
    'cutout': 'Закрывает случайные области изображения; справа задаются число и размер областей.',
    'random_artifacts': 'Добавляет выбранные типы локальных артефактов изображения.',
    'mixup': 'Смешивает пары обучающих примеров; значение задаёт параметр beta-распределения.',
    'topology_width': 'Меняет ширину проводников одновременно в изображении и маске.',
    'topology_scale': 'Масштабирует бинарную геометрию с повторной бинаризацией.',
    'topology_blur': 'Варьирует границу маски размытием и повторной бинаризацией.',
    'topology_boundary': 'Создаёт локальные изменения вдоль границ проводников.',
    'topology_local': 'Применяет локальные морфологические изменения геометрии.',
    'topology_gap': 'Создаёт контролируемые вариации разрывов проводников.',
    'charging': 'Имитирует локальное накопление заряда и насыщение SEM-сигнала.',
    'drift': 'Имитирует построчный дрейф сканирования в пикселях.',
    'focus': 'Имитирует пространственно меняющуюся расфокусировку.',
    'detector_noise': 'Имитирует пуассоновский шум детектора и шум считывания.',
    'gain': 'Имитирует плавное двумерное поле усиления яркости.',
    'scan_defects': 'Добавляет реалистичные локальные дефекты и сбои строк сканирования.',
}

AUGMENTATION_TOOLTIPS_EN = {
    key: value
    for key, value in {
        'rotate_90': 'Rotate image and mask by 90°. Probability is evaluated per variant.',
        'rotate_180': 'Rotate image and mask by 180°.',
        'flip_x': 'Mirror image and mask across the horizontal axis.',
        'flip_y': 'Mirror image and mask across the vertical axis.',
        'scale': 'Randomly scale a patch and resize it back to the training size.',
        'brightness': 'Randomly change image brightness without changing the binary mask.',
        'contrast': 'Randomly change contrast around the image mean.',
        'gamma': 'Apply a nonlinear gamma intensity transform.',
        'noise': 'Add Gaussian noise with the configured standard deviation.',
        'blur': 'Apply Gaussian blur up to the configured radius.',
        'cutout': 'Mask random image regions; value controls count and size.',
        'random_artifacts': 'Add selected local image artifact types.',
        'mixup': 'Mix training pairs using the configured beta-distribution alpha.',
        'topology_width': 'Change conductor width in both image and mask.',
        'topology_scale': 'Scale binary geometry and threshold it again.',
        'topology_blur': 'Vary mask boundaries by blur and re-thresholding.',
        'topology_boundary': 'Create local variations along conductor boundaries.',
        'topology_local': 'Apply local morphological geometry changes.',
        'topology_gap': 'Create controlled conductor-gap variations.',
        'charging': 'Simulate local charging and SEM signal saturation.',
        'drift': 'Simulate row-wise scan drift in pixels.',
        'focus': 'Simulate spatially varying defocus.',
        'detector_noise': 'Simulate Poisson detector and read noise.',
        'gain': 'Simulate a smooth two-dimensional gain field.',
        'scan_defects': 'Add realistic local defects and scan-line failures.',
    }.items()
}


class SemNormalizationEditor(QGroupBox):
    """Shared visual owner of SEM normalization settings."""

    def __init__(
        self,
        content: QWidget,
        *,
        title: str = '',
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(title, parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0 if not title else 6, 0 if not title else 6, 0 if not title else 6, 0)
        layout.addWidget(content)


@dataclass(frozen=True)
class AugmentationRow:
    key: str
    block: str
    name_ru: str
    name_en: str
    enabled: QCheckBox | QGroupBox
    probability: QWidget | None = None
    value: QWidget | None = None
    tooltip_ru: str = ''
    tooltip_en: str = ''


@dataclass(frozen=True)
class AugmentationBlock:
    key: str
    name_ru: str
    name_en: str
    enabled: QCheckBox | QGroupBox | None = None
    value: QWidget | None = None


class TrainingAugmentationEditor(QGroupBox):
    """Four-column hierarchical editor shared by training and preview.

    Existing controls remain the source of truth. The tree only owns their
    presentation and mirrors check states, so there is no duplicate config.
    """

    changed = pyqtSignal()

    def __init__(
        self,
        blocks: tuple[AugmentationBlock, ...],
        rows: tuple[AugmentationRow, ...],
        *,
        reset_defaults: Callable[[], None],
        title: str = '',
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(title, parent)
        self.blocks = blocks
        self.rows = rows
        self._reset_defaults = reset_defaults
        self._sync_guard = False
        self._change_guard = False
        self.block_items: dict[str, QTreeWidgetItem] = {}
        self.row_items: dict[str, QTreeWidgetItem] = {}
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0 if not title else 6, 0 if not title else 6, 0 if not title else 6, 0)
        layout.setSpacing(6)
        self.tree = QTreeWidget(self)
        self.tree.setColumnCount(4)
        self.tree.setHeaderLabels(('Вкл.', 'Аугментация', 'Вероятность', 'Значение'))
        self.tree.setRootIsDecorated(True)
        self.tree.setAlternatingRowColors(True)
        self.tree.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self.tree.header().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.tree.header().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.tree.header().setSectionResizeMode(2, QHeaderView.ResizeMode.Interactive)
        self.tree.header().setSectionResizeMode(3, QHeaderView.ResizeMode.Interactive)
        self.tree.setColumnWidth(2, 110)
        self.tree.setColumnWidth(3, 180)
        layout.addWidget(self.tree)
        self.defaults_button = QPushButton('По умолчанию', self)
        self.defaults_button.setToolTip('Восстановить исходные значения только для аугментаций.')
        layout.addWidget(self.defaults_button, 0, Qt.AlignmentFlag.AlignRight)
        self._build_tree()
        self.tree.itemChanged.connect(self._on_item_changed)
        self.defaults_button.clicked.connect(self._restore_defaults)
        self._connect_controls()
        self.sync_from_controls()

    def _build_tree(self) -> None:
        for block in self.blocks:
            item = QTreeWidgetItem(self.tree)
            item.setData(0, Qt.ItemDataRole.UserRole, ('block', block.key))
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setText(1, block.name_ru)
            if block.value is not None:
                self.tree.setItemWidget(item, 3, block.value)
            self.block_items[block.key] = item
        for row in self.rows:
            item = QTreeWidgetItem(self.block_items[row.block])
            item.setData(0, Qt.ItemDataRole.UserRole, ('row', row.key))
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setText(1, row.name_ru)
            tooltip = row.tooltip_ru or AUGMENTATION_TOOLTIPS_RU.get(row.key, row.name_ru)
            for column in range(4):
                item.setToolTip(column, tooltip)
            if row.probability is not None:
                self.tree.setItemWidget(item, 2, row.probability)
            if row.value is not None:
                self.tree.setItemWidget(item, 3, row.value)
            self.row_items[row.key] = item
        self.tree.expandAll()

    def _connect_controls(self) -> None:
        for block in self.blocks:
            if block.enabled is not None:
                block.enabled.toggled.connect(self.sync_from_controls)
        for row in self.rows:
            row.enabled.toggled.connect(self.sync_from_controls)
            for control in (row.probability, row.value):
                if control is None:
                    continue
                for widget in (control, *control.findChildren(QWidget)):
                    for signal_name in ('valueChanged', 'toggled', 'currentIndexChanged'):
                        signal = getattr(widget, signal_name, None)
                        if signal is not None:
                            signal.connect(self._emit_changed)
                            break

    def _emit_changed(self, *_args: object) -> None:
        if not self._change_guard:
            self.changed.emit()

    def _on_item_changed(self, item: QTreeWidgetItem, column: int) -> None:
        if self._sync_guard or column != 0:
            return
        payload = item.data(0, Qt.ItemDataRole.UserRole)
        if not payload:
            return
        kind, key = payload
        checked = item.checkState(0) == Qt.CheckState.Checked
        if kind == 'row':
            next(row for row in self.rows if row.key == key).enabled.setChecked(checked)
        else:
            block = next(block for block in self.blocks if block.key == key)
            if block.enabled is not None:
                block.enabled.setChecked(checked)
            else:
                for row in self.rows:
                    if row.block == key:
                        row.enabled.setChecked(checked)
        self.sync_from_controls()
        self.changed.emit()

    def sync_from_controls(self, *_args: object) -> None:
        self._sync_guard = True
        try:
            for row in self.rows:
                checked = row.enabled.isChecked()
                self.row_items[row.key].setCheckState(
                    0, Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked
                )
            for block in self.blocks:
                block_rows = [row for row in self.rows if row.block == block.key]
                checked = block.enabled.isChecked() if block.enabled is not None else any(
                    row.enabled.isChecked() for row in block_rows
                )
                self.block_items[block.key].setCheckState(
                    0, Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked
                )
                for row in block_rows:
                    enabled = checked and row.enabled.isChecked()
                    self.row_items[row.key].setDisabled(not checked)
                    for control in (row.probability, row.value):
                        if control is not None:
                            control.setEnabled(enabled)
        finally:
            self._sync_guard = False

    def _restore_defaults(self) -> None:
        self._change_guard = True
        try:
            self._reset_defaults()
        finally:
            self._change_guard = False
        self.sync_from_controls()
        self.changed.emit()

    def set_language(self, language: str) -> None:
        russian = str(language).lower().startswith('ru')
        self.tree.setHeaderLabels(
            ('Вкл.', 'Аугментация', 'Вероятность', 'Значение')
            if russian
            else ('On', 'Augmentation', 'Probability', 'Value')
        )
        self.defaults_button.setText('По умолчанию' if russian else 'Defaults')
        for block in self.blocks:
            self.block_items[block.key].setText(1, block.name_ru if russian else block.name_en)
        for row in self.rows:
            item = self.row_items[row.key]
            item.setText(1, row.name_ru if russian else row.name_en)
            tooltip = (
                row.tooltip_ru or AUGMENTATION_TOOLTIPS_RU.get(row.key, item.text(1))
                if russian
                else row.tooltip_en or AUGMENTATION_TOOLTIPS_EN.get(row.key, item.text(1))
            )
            for column in range(4):
                item.setToolTip(column, tooltip)
