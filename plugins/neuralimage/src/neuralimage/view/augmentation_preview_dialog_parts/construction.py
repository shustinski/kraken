"""augmentation_preview_dialog: construction."""
from __future__ import annotations
from collections import OrderedDict
import numpy as np
from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtWidgets import (
    QWidget,
)
from neuralimage.augmentations import (
    PCBDefectAugmentor as PCBDefectAugmentor,
)
from neuralimage.lib.data_interfaces import (
    TrainingParameters,
)
from neuralimage.lib.rare_patch_masks import (
    collect_matching_sample_label_pairs,
)
from neuralimage.lib.ui_texts import get_ui_section
from neuralimage.view.settings_panel import SettingsPanel


class ConstructionMixin:
    def __init__(
        self,
        training_parameters: TrainingParameters,
        settings_panel: SettingsPanel,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(None)
        self.setWindowFlags(
            Qt.WindowType.Window
            | Qt.WindowType.WindowMinimizeButtonHint
            | Qt.WindowType.WindowMaximizeButtonHint
            | Qt.WindowType.WindowCloseButtonHint
        )
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        self._parent_window = parent
        self._panel = settings_panel
        self._training_parameters = training_parameters
        self._texts = get_ui_section('augmentation_preview_dialog')
        self._settings_texts = get_ui_section('settings_panel')
        self._is_russian_ui = any('\u0400' <= char <= '\u04FF' for char in str(self._texts.get('window_title', '')))
        settings_form = dict(self._settings_texts.get('settings_form', {}))
        self._settings_form_labels = dict(settings_form.get('labels', {}))
        self._settings_form_tooltips = dict(settings_form.get('tooltips', {}))
        self._sample_pairs, self._load_error = collect_matching_sample_label_pairs(
            training_parameters.image_path,
            training_parameters.label_path,
            strict=False,
            allow_cif_labels=True,
            recursive=bool(getattr(training_parameters, 'recursive_file_search', False)),
        )
        self._current_sample_index = 0
        self._current_frame_index = 0
        self._frame_plan: list[tuple[int, int, int]] = []
        self._cutter_length_cache: dict[int, int] = {}
        self._variant_serial = 0
        self._cutter_item_index = 0
        self._resample_salt = 0
        self._show_augmented = True
        self._sample_list_updating = False
        self._sidebar_restore: list[tuple[QWidget, QWidget | None, object, int | None]] = []
        self._original_image_array: np.ndarray | None = None
        self._augmented_image_array: np.ndarray | None = None
        self._original_label_array: np.ndarray | None = None
        self._augmented_label_array: np.ndarray | None = None
        self._prepared_arrays_cache: OrderedDict[int, tuple[np.ndarray, np.ndarray]] = OrderedDict()

        self.sem_normalization_editor = self._panel.sem_segmentation_section_editors['preprocessing']
        self.sem_augmentation_editor = self._panel.training_augmentation_editor
        self._build_ui()
        self.sample_list_widget.installEventFilter(self)
        self._connect_signals()
        self._sync_group_boxes()
        self._show_loading_state()
        # Paint the window before reading and augmenting a potentially large
        # SEM frame. The work still starts immediately on the next event turn.
        QTimer.singleShot(0, self._refresh_preview)
