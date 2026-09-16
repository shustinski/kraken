from pathlib import Path
import re

path = Path(__file__).resolve().parents[1] / 'src/neuralimage/view/augmentation_preview_dialog.py'
text = path.read_text(encoding='utf-8')

if 'from neuralimage.view.settings_panel import SettingsPanel' not in text:
    text = text.replace(
        'from neuralimage.view.training_transform_editors import (\n',
        'from neuralimage.view.settings_panel import SettingsPanel\nfrom neuralimage.view.training_transform_editors import (\n',
    )

old_init = """    def __init__(self, training_parameters: TrainingParameters, parent: QWidget | None = None) -> None:
        super().__init__(None)
        self.setWindowFlags(
            Qt.WindowType.Window
            | Qt.WindowType.WindowMinimizeButtonHint
            | Qt.WindowType.WindowMaximizeButtonHint
            | Qt.WindowType.WindowCloseButtonHint
        )
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        self._parent_window = parent
        self._training_parameters = training_parameters
        self._texts = get_ui_section('augmentation_preview_dialog')
        self._settings_texts = get_ui_section('settings_panel')
        self._is_russian_ui = any('\\u0400' <= char <= '\\u04FF' for char in str(self._texts.get('window_title', '')))
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
        self._variant_serial = 0
        self._show_augmented = True
        self._sample_list_updating = False
        self._toggle_boxes: dict[str, QCheckBox] = {}
        self._value_widgets: dict[str, list[QWidget]] = {}
        self._value_rows: dict[str, list[QWidget]] = {}
        self._original_image_array: np.ndarray | None = None
        self._augmented_image_array: np.ndarray | None = None
        self._original_label_array: np.ndarray | None = None
        self._augmented_label_array: np.ndarray | None = None
        self._prepared_arrays_cache: OrderedDict[int, tuple[np.ndarray, np.ndarray]] = OrderedDict()

        self._create_value_controls()
        self._build_ui()
        self._initialize_toggle_states()
        self._connect_signals()
        self._sync_group_boxes()
        self._show_loading_state()"""

new_init = """    def __init__(
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
        self._is_russian_ui = any('\\u0400' <= char <= '\\u04FF' for char in str(self._texts.get('window_title', '')))
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
        self._variant_serial = 0
        self._show_augmented = True
        self._sample_list_updating = False
        self._sidebar_restore: list[tuple[QWidget, QWidget | None, object, int | None]] = []
        self._original_image_array: np.ndarray | None = None
        self._augmented_image_array: np.ndarray | None = None
        self._original_label_array: np.ndarray | None = None
        self._augmented_label_array: np.ndarray | None = None
        self._prepared_arrays_cache: OrderedDict[int, tuple[np.ndarray, np.ndarray]] = OrderedDict()

        self.sem_normalization_editor = self._panel.sem_segmentation_section_editors['preprocessing']
        self.sem_augmentation_editor = self._panel.sem_segmentation_section_editors['augmentation']
        self._build_ui()
        self._connect_signals()
        self._sync_group_boxes()
        self._show_loading_state()"""

if old_init not in text:
    raise SystemExit('init block not found')
text = text.replace(old_init, new_init)

start = text.index('    def _create_value_controls(self) -> None:')
end = text.index('    def _connect_signals(self) -> None:')
text = text[:start] + text[end:]

old_shared = """    def _build_shared_training_transform_groups(self) -> tuple[QGroupBox, QGroupBox]:
        language = 'ru' if self._is_russian_ui else 'en'
        source_config = build_sem_segmentation_config(
            {
                'preprocessing': asdict(self._training_parameters.preprocessing),
                'augmentation': asdict(self._training_parameters.sem_augmentation),
            }
        )
        form_values = sem_config_to_form_values(source_config.to_dict())

        self.sem_normalization_editor = CompactSemSectionEditor(
            'preprocessing',
            language='ru' if self._is_russian_ui else 'en',
        )
        self.sem_normalization_editor.set_form_values(form_values)
        normalization_group = SemNormalizationEditor(
            self.sem_normalization_editor,
            title='Нормализация SEM' if self._is_russian_ui else 'SEM normalization',
        )

        self.sem_augmentation_editor = SemAugmentationSectionEditor(
            language='ru' if self._is_russian_ui else 'en',
        )
        self.sem_augmentation_editor.set_form_values(form_values)
        batch_group = self._build_batch_group()
        augmentation_group = TrainingAugmentationEditor(
            {
                'sem_acquisition': self.sem_augmentation_editor,
                'spatial': self._build_spatial_group(),
                'photometric': self._build_photometric_group(),
                'topology_variations': self._build_mask_variation_group(),
                'batch': batch_group,
                'synthetic': self._build_synthetic_group(),
            },
            title='Аугментации обучения' if self._is_russian_ui else 'Training augmentations',
        )

        self.sem_normalization_editor.changed.connect(self._on_sem_pipeline_changed)
        self.sem_augmentation_editor.changed.connect(self._on_sem_pipeline_changed)
        self._sync_sem_pipeline_editor_visibility()
        return normalization_group, augmentation_group"""

new_shared = """    def _build_shared_training_transform_groups(self) -> tuple[QGroupBox, QGroupBox]:
        return self._panel.sem_normalization_editor, self._panel.training_augmentation_editor

    def _attach_sidebar_widget(self, widget: QGroupBox, layout: QVBoxLayout) -> None:
        old_parent = widget.parentWidget()
        old_layout = old_parent.layout() if old_parent is not None else None
        index: int | None = None
        if old_layout is not None:
            index = old_layout.indexOf(widget)
            if index >= 0:
                old_layout.removeWidget(widget)
        layout.addWidget(widget)
        self._sidebar_restore.append((widget, old_parent, old_layout, index))

    def closeEvent(self, event) -> None:
        for widget, old_parent, old_layout, index in reversed(self._sidebar_restore):
            current_parent = widget.parentWidget()
            if current_parent is not None:
                current_layout = current_parent.layout()
                if current_layout is not None:
                    current_layout.removeWidget(widget)
            if old_layout is not None and index is not None and index >= 0:
                widget.setParent(old_parent)
                if isinstance(old_layout, QVBoxLayout):
                    old_layout.insertWidget(index, widget)
                else:
                    old_layout.addWidget(widget)
            elif old_parent is not None:
                widget.setParent(old_parent)
        self._sidebar_restore.clear()
        super().closeEvent(event)"""

if old_shared not in text:
    raise SystemExit('shared groups block not found')
text = text.replace(old_shared, new_shared)

text = text.replace(
    """        normalization_group, augmentation_group = self._build_shared_training_transform_groups()
        right_layout.addWidget(normalization_group)
        right_layout.addWidget(augmentation_group)""",
    """        normalization_group, augmentation_group = self._build_shared_training_transform_groups()
        self._attach_sidebar_widget(normalization_group, right_layout)
        self._attach_sidebar_widget(augmentation_group, right_layout)""",
)

text = text.replace(
    """    def _current_sem_config(self):
        values = self.sem_normalization_editor.form_values()
        values.update(self.sem_augmentation_editor.form_values())
        return build_sem_segmentation_config(sem_config_from_form_values(values))""",
    """    def _current_sem_config(self):
        return build_sem_segmentation_config(self._panel.get_sem_segmentation_config())""",
)

text = text.replace(
    """    def _sync_sem_pipeline_editor_visibility(self) -> None:
        enabled = bool(self.sem_augmentation_editor.isChecked())
        self.sem_augmentation_editor._sync_effect_rows()
        for key, control in self.sem_augmentation_editor.controls.items():
            if key == 'aug_enabled':
                continue
            if not enabled:
                control.setEnabled(False)
            self.sem_augmentation_editor.set_field_visible(key, enabled)""",
    """    def _sync_sem_pipeline_editor_visibility(self) -> None:
        self._panel._sync_sem_segmentation_controls()""",
)

path.write_text(text, encoding='utf-8')
print('phase 1 done')
