from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from django import forms
from django.core.exceptions import ValidationError

from application.dto import MainWindowState, SettingsState
from lib.data_interfaces import (
    OptimizerName,
    MixedPrecisionMode,
    SampleCutMode,
    SchedulerName,
    ValidationSource,
    WorkMode,
    normalize_scheduler_name,
    normalize_validation_source,
    normalize_work_mode,
)
from lib.loss_config import (
    LOSS_SELECTION_NAMES,
    dominant_loss_function,
    resolve_loss_term_weights,
)
from lib.ui_texts import get_ui_section


_BASE_TEXT_INPUT_ATTRS = {
    'class': 'control-input',
    'spellcheck': 'false',
}

_BASE_NUM_INPUT_ATTRS = {
    'class': 'control-input',
}

_BASE_SELECT_ATTRS = {
    'class': 'control-input',
}

_START_FORM_ATTRS = {
    'form': 'start-form',
}


def _copy_dict(value):
    return dict(value) if isinstance(value, dict) else {}


def _read_text(mapping: dict, key: str, default: str = '') -> str:
    value = mapping.get(key, default)
    return str(value if value is not None else default)


class MainWindowForm(forms.Form):
    WORK_MODE_CHOICES = (
        (WorkMode.train_and_recognition.value, 'Train and recognition'),
        (WorkMode.further_training.value, 'Further training and recognition'),
        (WorkMode.recognition_only.value, 'Recognition only'),
        (WorkMode.train_only.value, 'Training only'),
    )

    work_mode = forms.ChoiceField(label='Work mode', choices=WORK_MODE_CHOICES)
    source_folder = forms.CharField(label='Source folder', required=False)
    result_folder = forms.CharField(label='Result folder', required=False)
    sample_folder = forms.CharField(label='Sample folder', required=False)
    label_folder = forms.CharField(label='Label folder', required=False)
    model_path = forms.CharField(label='Model path (.pth)', required=False)
    epochs = forms.IntegerField(label='Epochs', min_value=1, max_value=10000)

    def __init__(self, *args, **kwargs):
        language = kwargs.pop('language', None)
        ui_texts = kwargs.pop('ui_texts', None)
        self._texts = ui_texts if isinstance(ui_texts, dict) else get_ui_section('webui', language)
        self._main_form_texts = _copy_dict(self._texts.get('main_form', {}))
        args, kwargs = self._normalize_legacy_work_mode_payload(args, kwargs)
        super().__init__(*args, **kwargs)
        self.fields['work_mode'].widget.attrs.update(_BASE_SELECT_ATTRS)

        placeholders = _copy_dict(self._main_form_texts.get('placeholders', {}))
        self.fields['source_folder'].widget.attrs.update(
            _BASE_TEXT_INPUT_ATTRS | {'placeholder': _read_text(placeholders, 'source_folder', r'D:\data\source')}
        )
        self.fields['result_folder'].widget.attrs.update(
            _BASE_TEXT_INPUT_ATTRS | {'placeholder': _read_text(placeholders, 'result_folder', r'D:\data\result')}
        )
        self.fields['sample_folder'].widget.attrs.update(
            _BASE_TEXT_INPUT_ATTRS | {'placeholder': _read_text(placeholders, 'sample_folder', r'D:\data\sample')}
        )
        self.fields['label_folder'].widget.attrs.update(
            _BASE_TEXT_INPUT_ATTRS | {'placeholder': _read_text(placeholders, 'label_folder', r'D:\data\label')}
        )
        self.fields['model_path'].widget.attrs.update(
            _BASE_TEXT_INPUT_ATTRS | {'placeholder': _read_text(placeholders, 'model_path', r'D:\models\model.pth')}
        )
        self.fields['epochs'].widget.attrs.update(_BASE_NUM_INPUT_ATTRS)
        self._apply_localized_texts()

    def _apply_localized_texts(self) -> None:
        labels = _copy_dict(self._main_form_texts.get('labels', {}))
        tooltips = _copy_dict(self._main_form_texts.get('tooltips', {}))
        work_modes = _copy_dict(self._main_form_texts.get('work_modes', {}))

        self.fields['work_mode'].label = _read_text(labels, 'work_mode', self.fields['work_mode'].label)
        self.fields['source_folder'].label = _read_text(labels, 'source_folder', self.fields['source_folder'].label)
        self.fields['result_folder'].label = _read_text(labels, 'result_folder', self.fields['result_folder'].label)
        self.fields['sample_folder'].label = _read_text(labels, 'sample_folder', self.fields['sample_folder'].label)
        self.fields['label_folder'].label = _read_text(labels, 'label_folder', self.fields['label_folder'].label)
        self.fields['model_path'].label = _read_text(labels, 'model_path', self.fields['model_path'].label)
        self.fields['epochs'].label = _read_text(labels, 'epochs', self.fields['epochs'].label)

        self.fields['work_mode'].choices = (
            (
                WorkMode.train_and_recognition.value,
                _read_text(work_modes, WorkMode.train_and_recognition.value, 'Train and recognition'),
            ),
            (
                WorkMode.further_training.value,
                _read_text(work_modes, WorkMode.further_training.value, 'Further training and recognition'),
            ),
            (
                WorkMode.recognition_only.value,
                _read_text(work_modes, WorkMode.recognition_only.value, 'Recognition only'),
            ),
            (
                WorkMode.train_only.value,
                _read_text(work_modes, WorkMode.train_only.value, 'Training only'),
            ),
        )

        for field_name, field in self.fields.items():
            help_text = _read_text(tooltips, field_name, '')
            field.help_text = help_text
            if help_text:
                field.widget.attrs['title'] = help_text
            field.widget.attrs['aria-label'] = str(field.label)

    @staticmethod
    def _normalize_legacy_work_mode_payload(args, kwargs):
        data = args[0] if args else kwargs.get('data')
        if data is None:
            return args, kwargs

        prefix = kwargs.get('prefix')
        work_mode_key = f'{prefix}-work_mode' if prefix else 'work_mode'
        posted_mode = data.get(work_mode_key)
        normalized_mode = normalize_work_mode(posted_mode)
        if not posted_mode or normalized_mode == posted_mode:
            return args, kwargs

        normalized_data = data.copy()
        normalized_data[work_mode_key] = normalized_mode
        if args:
            args = (normalized_data, *args[1:])
        else:
            kwargs['data'] = normalized_data
        return args, kwargs

    def clean(self):
        cleaned = super().clean()
        work_mode = normalize_work_mode(cleaned.get('work_mode', ''))
        cleaned['work_mode'] = work_mode
        source_folder = (cleaned.get('source_folder') or '').strip()
        result_folder = (cleaned.get('result_folder') or '').strip()
        sample_folder = (cleaned.get('sample_folder') or '').strip()
        label_folder = (cleaned.get('label_folder') or '').strip()
        model_path = (cleaned.get('model_path') or '').strip()
        errors = _copy_dict(self._main_form_texts.get('errors', {}))

        requires_source_result = work_mode in (
            WorkMode.train_and_recognition.value,
            WorkMode.recognition_only.value,
            WorkMode.further_training.value,
        )
        if requires_source_result:
            if not source_folder or not Path(source_folder).is_dir():
                raise ValidationError(
                    _read_text(
                        errors,
                        'source_folder_missing',
                        'Поле "Исходные файлы" должно указывать на существующую папку.',
                    )
                )
            if not result_folder or not Path(result_folder).is_dir():
                raise ValidationError(
                    _read_text(
                        errors,
                        'result_folder_missing',
                        'Поле "Папка результата" должно указывать на существующую папку.',
                    )
                )

        if work_mode in (WorkMode.train_and_recognition.value, WorkMode.further_training.value):
            if not sample_folder or not Path(sample_folder).is_dir():
                raise ValidationError(
                    _read_text(
                        errors,
                        'sample_folder_missing',
                        'Для обучения нужна существующая папка с обучающими изображениями.',
                    )
                )
            if not label_folder or not Path(label_folder).is_dir():
                raise ValidationError(
                    _read_text(
                        errors,
                        'label_folder_missing',
                        'Для обучения нужна существующая папка с масками или метками.',
                    )
                )

        if work_mode in (WorkMode.recognition_only.value, WorkMode.further_training.value):
            if not model_path or not Path(model_path).is_file():
                raise ValidationError(
                    _read_text(
                        errors,
                        'model_path_missing',
                        'Для выбранного режима нужен корректный путь к файлу модели `.pth`.',
                    )
                )

        return cleaned

    def to_state(self) -> MainWindowState:
        cleaned = self.cleaned_data
        return MainWindowState(
            work_mode=cleaned['work_mode'],
            source_folder=cleaned.get('source_folder', ''),
            result_folder=cleaned.get('result_folder', ''),
            model_path=cleaned.get('model_path', ''),
            label_folder=cleaned.get('label_folder', ''),
            sample_folder=cleaned.get('sample_folder', ''),
            epochs=cleaned['epochs'],
        )


class SettingsForm(forms.Form):
    step = forms.IntegerField(label='Step', min_value=4, max_value=1024)
    vertical_rotation = forms.BooleanField(label='Rotate 180°', required=False)
    horizontal_rotation = forms.BooleanField(label='Rotate 90°', required=False)
    additional_augmentation = forms.BooleanField(label='Additional augmentation', required=False)
    random_crop = forms.BooleanField(label='Random crop in online mode', required=False)
    crops_per_image = forms.IntegerField(label='Crops per image', min_value=1, max_value=5000, required=False)
    scale_augmentation = forms.BooleanField(label='Scale augmentation in online mode', required=False)
    scale_augmentation_strength = forms.FloatField(
        label='Scale augmentation strength', min_value=0.0, max_value=1.0, required=False
    )
    augmentation_brightness_strength = forms.FloatField(
        label='Augmentation brightness strength', min_value=0.0, max_value=1.0, required=False
    )
    augmentation_contrast_strength = forms.FloatField(
        label='Augmentation contrast strength', min_value=0.0, max_value=1.0, required=False
    )
    augmentation_noise_probability = forms.FloatField(
        label='Augmentation noise probability', min_value=0.0, max_value=1.0, required=False
    )
    augmentation_noise_sigma = forms.FloatField(
        label='Augmentation noise sigma', min_value=0.0, max_value=1.0, required=False
    )
    sample_x = forms.IntegerField(label='Sample X', min_value=8, max_value=4096)
    sample_y = forms.IntegerField(label='Sample Y', min_value=8, max_value=4096)
    model = forms.CharField(label='Model architecture')
    color_mode = forms.ChoiceField(label='Color mode', choices=[('RGB', 'RGB'), ('ЧБ', 'ЧБ')])
    use_validation = forms.BooleanField(label='Use validation', required=False)
    validation_source = forms.ChoiceField(
        label='Validation source',
        choices=[
            (ValidationSource.split.value, 'Split'),
            (ValidationSource.external.value, 'External folders'),
        ],
        required=False,
    )
    validation_percent = forms.IntegerField(label='Validation percent', min_value=0, max_value=90, required=False)
    validation_image_folder = forms.CharField(label='Validation image folder', required=False)
    validation_label_folder = forms.CharField(label='Validation label folder', required=False)
    save_validation_binary_images = forms.BooleanField(label='Save validation binary images', required=False)
    shuffle = forms.BooleanField(label='Shuffle dataset', required=False)
    sample_cut_mode = forms.ChoiceField(
        label='Sample cut mode',
        choices=[(SampleCutMode.disk.value, 'To disk'), (SampleCutMode.online.value, 'Online')],
    )
    batch_size = forms.IntegerField(label='Batch size', min_value=1, max_value=512)
    dataloader_num_workers = forms.IntegerField(label='DataLoader workers', min_value=-1, max_value=64, required=False)
    overlap = forms.IntegerField(label='Overlap', min_value=0, max_value=256)
    log_update_frequency = forms.IntegerField(label='Log update frequency', min_value=0, max_value=5000)
    crop_enabled = forms.BooleanField(label='Enable edge crop', required=False)
    resize_enabled = forms.BooleanField(label='Enable resize', required=False)
    edge_cut_size = forms.IntegerField(label='Edge cut size', min_value=0, max_value=2000, required=False)
    target_x = forms.IntegerField(label='Target X', min_value=0, max_value=10000, required=False)
    target_y = forms.IntegerField(label='Target Y', min_value=0, max_value=10000, required=False)
    optimizer_name = forms.ChoiceField(label='Optimizer', choices=[(o.value, o.value) for o in OptimizerName])
    mixed_precision = forms.ChoiceField(
        label='Mixed precision',
        choices=[(m.value, m.value) for m in MixedPrecisionMode],
    )
    loss_function = forms.ChoiceField(
        label='Loss function',
        choices=[
            ('bce', 'bce'),
            ('dice', 'dice'),
            ('iou', 'iou'),
            ('focal_bce', 'focal_bce'),
            ('boundary', 'boundary'),
            ('focal_tversky', 'focal_tversky'),
            ('ce', 'ce'),
        ],
    )
    dice_loss_weight = forms.FloatField(label='Dice loss weight', min_value=0.0, max_value=1.0, required=False)
    iou_loss_weight = forms.FloatField(label='IoU loss weight', min_value=0.0, max_value=1.0, required=False)
    learning_rate = forms.FloatField(label='Learning rate', min_value=1e-8, max_value=10)
    weight_decay = forms.FloatField(label='Weight decay', min_value=0.0, max_value=10)
    warmup_enabled = forms.BooleanField(label='Enable warmup', required=False)
    warmup_epochs = forms.IntegerField(label='Warmup epochs', min_value=1, max_value=2000)
    warmup_start_factor = forms.FloatField(label='Warmup start factor', min_value=0.0, max_value=1.0)
    scheduler_name = forms.ChoiceField(
        label='Scheduler',
        choices=[(scheduler.value, scheduler.value) for scheduler in SchedulerName],
        required=False,
    )
    scheduler_plateau_factor = forms.FloatField(label='Plateau factor', min_value=0.01, max_value=1.0, required=False)
    scheduler_plateau_patience = forms.IntegerField(label='Plateau patience', min_value=0, max_value=2000, required=False)
    scheduler_plateau_threshold = forms.FloatField(
        label='Plateau threshold',
        min_value=0.0,
        max_value=10.0,
        required=False,
    )
    scheduler_plateau_min_lr = forms.FloatField(label='Plateau min LR', min_value=0.0, max_value=10.0, required=False)
    scheduler_plateau_cooldown = forms.IntegerField(label='Plateau cooldown', min_value=0, max_value=2000, required=False)
    scheduler_cosine_t_max = forms.IntegerField(label='Cosine T_max', min_value=1, max_value=10000, required=False)
    scheduler_cosine_eta_min = forms.FloatField(label='Cosine eta_min', min_value=0.0, max_value=10.0, required=False)
    scheduler_one_cycle_max_lr = forms.FloatField(label='OneCycle max LR', min_value=0.0, max_value=10.0, required=False)
    scheduler_one_cycle_pct_start = forms.FloatField(
        label='OneCycle pct_start',
        min_value=0.0,
        max_value=1.0,
        required=False,
    )
    scheduler_one_cycle_anneal_strategy = forms.ChoiceField(
        label='OneCycle anneal strategy',
        choices=[('cos', 'cos'), ('linear', 'linear')],
        required=False,
    )
    scheduler_one_cycle_div_factor = forms.FloatField(
        label='OneCycle div factor',
        min_value=1.0,
        max_value=1000000.0,
        required=False,
    )
    scheduler_one_cycle_final_div_factor = forms.FloatField(
        label='OneCycle final div factor',
        min_value=1.0,
        max_value=1000000.0,
        required=False,
    )
    scheduler_one_cycle_three_phase = forms.BooleanField(label='OneCycle three phase', required=False)
    scheduler_step_lr_step_size = forms.IntegerField(label='StepLR step size', min_value=1, max_value=10000, required=False)
    scheduler_step_lr_gamma = forms.FloatField(label='StepLR gamma', min_value=0.01, max_value=1.0, required=False)
    hard_mining_enabled = forms.BooleanField(label='Enable hard mining', required=False)
    hard_mining_strength = forms.FloatField(label='Hard mining strength', min_value=0.0, max_value=10.0, required=False)
    hard_mining_ema_alpha = forms.FloatField(
        label='Hard mining EMA alpha', min_value=0.0, max_value=1.0, required=False
    )
    hard_pixel_mining_enabled = forms.BooleanField(label='Enable hard pixel mining', required=False)
    hard_pixel_mining_ratio = forms.FloatField(
        label='Hard pixel keep ratio', min_value=0.01, max_value=1.0, required=False
    )
    cutout_enabled = forms.BooleanField(label='Enable cutout', required=False)
    cutout_probability = forms.FloatField(label='Cutout probability', min_value=0.0, max_value=1.0, required=False)
    cutout_holes = forms.IntegerField(label='Cutout holes', min_value=1, max_value=32, required=False)
    cutout_size_ratio = forms.FloatField(label='Cutout size ratio', min_value=0.0, max_value=1.0, required=False)
    random_artifacts_enabled = forms.BooleanField(label='Enable random artifacts', required=False)
    random_artifacts_probability = forms.FloatField(
        label='Random artifacts probability',
        min_value=0.0,
        max_value=1.0,
        required=False,
    )
    random_artifacts_count = forms.IntegerField(label='Random artifacts count', min_value=1, max_value=16, required=False)
    random_artifacts_size_ratio = forms.FloatField(
        label='Random artifacts size ratio',
        min_value=0.0,
        max_value=1.0,
        required=False,
    )
    mixup_enabled = forms.BooleanField(label='Enable mixup', required=False)
    mixup_probability = forms.FloatField(label='Mixup probability', min_value=0.0, max_value=1.0, required=False)
    mixup_alpha = forms.FloatField(label='Mixup alpha', min_value=0.0, max_value=10.0, required=False)
    skip_uniform_labels = forms.BooleanField(label='Skip all-0/all-1 labels in training', required=False)
    early_stopping_enabled = forms.BooleanField(label='Enable early stopping', required=False)
    early_stopping_patience = forms.IntegerField(label='Early stopping patience', min_value=0, max_value=2000)
    early_stopping_min_delta = forms.FloatField(label='Early stopping min delta', min_value=0.0, max_value=10.0)
    early_stopping_restore_best_weights = forms.BooleanField(label='Restore best weights', required=False)
    torch_compile_enabled = forms.BooleanField(label='Enable torch.compile', required=False)
    show_batch_preview = forms.BooleanField(label='Show batch preview', required=False)
    use_multi_gpu = forms.BooleanField(label='Use multi-GPU', required=False)

    def __init__(self, *args, **kwargs):
        language = kwargs.pop('language', None)
        ui_texts = kwargs.pop('ui_texts', None)
        self._texts = ui_texts if isinstance(ui_texts, dict) else get_ui_section('webui', language)
        self._settings_form_texts = _copy_dict(self._texts.get('settings_form', {}))
        super().__init__(*args, **kwargs)
        for field in self.fields.values():
            field.widget.attrs.update(_START_FORM_ATTRS)

        for field_name in (
            'step',
            'crops_per_image',
            'scale_augmentation_strength',
            'sample_x',
            'sample_y',
            'validation_percent',
            'batch_size',
            'dataloader_num_workers',
            'overlap',
            'log_update_frequency',
            'edge_cut_size',
            'target_x',
            'target_y',
            'learning_rate',
            'augmentation_brightness_strength',
            'augmentation_contrast_strength',
            'augmentation_noise_probability',
            'augmentation_noise_sigma',
            'dice_loss_weight',
            'iou_loss_weight',
            'weight_decay',
            'warmup_epochs',
            'warmup_start_factor',
            'scheduler_plateau_factor',
            'scheduler_plateau_patience',
            'scheduler_plateau_threshold',
            'scheduler_plateau_min_lr',
            'scheduler_plateau_cooldown',
            'scheduler_cosine_t_max',
            'scheduler_cosine_eta_min',
            'scheduler_one_cycle_max_lr',
            'scheduler_one_cycle_pct_start',
            'scheduler_one_cycle_div_factor',
            'scheduler_one_cycle_final_div_factor',
            'scheduler_step_lr_step_size',
            'scheduler_step_lr_gamma',
            'hard_mining_strength',
            'hard_mining_ema_alpha',
            'hard_pixel_mining_ratio',
            'cutout_probability',
            'cutout_holes',
            'cutout_size_ratio',
            'random_artifacts_probability',
            'random_artifacts_count',
            'random_artifacts_size_ratio',
            'mixup_probability',
            'mixup_alpha',
            'early_stopping_patience',
            'early_stopping_min_delta',
        ):
            self.fields[field_name].widget.attrs.update(_BASE_NUM_INPUT_ATTRS)

        for field_name in (
            'color_mode',
            'validation_source',
            'sample_cut_mode',
            'optimizer_name',
            'mixed_precision',
            'loss_function',
            'scheduler_name',
            'scheduler_one_cycle_anneal_strategy',
        ):
            self.fields[field_name].widget.attrs.update(_BASE_SELECT_ATTRS)

        placeholders = _copy_dict(self._settings_form_texts.get('placeholders', {}))
        self.fields['model'].widget.attrs.update(
            _BASE_TEXT_INPUT_ATTRS | {'placeholder': _read_text(placeholders, 'model', 'M 720k')}
        )
        self.fields['validation_image_folder'].widget.attrs.update(
            _BASE_TEXT_INPUT_ATTRS
            | {'placeholder': _read_text(placeholders, 'validation_image_folder', r'D:\data\validation\images')}
        )
        self.fields['validation_label_folder'].widget.attrs.update(
            _BASE_TEXT_INPUT_ATTRS
            | {'placeholder': _read_text(placeholders, 'validation_label_folder', r'D:\data\validation\labels')}
        )
        self._apply_localized_texts()

    def _apply_localized_texts(self) -> None:
        labels = _copy_dict(self._settings_form_texts.get('labels', {}))
        tooltips = _copy_dict(self._settings_form_texts.get('tooltips', {}))
        choices = _copy_dict(self._settings_form_texts.get('choices', {}))

        for field_name, field in self.fields.items():
            field.label = _read_text(labels, field_name, str(field.label))
            help_text = _read_text(tooltips, field_name, '')
            field.help_text = help_text
            if help_text:
                field.widget.attrs['title'] = help_text
            field.widget.attrs['aria-label'] = str(field.label)

        work_choices = _copy_dict(choices.get('color_mode', {}))
        self.fields['color_mode'].choices = [
            ('RGB', _read_text(work_choices, 'RGB', 'RGB')),
            ('ЧБ', _read_text(work_choices, 'ЧБ', 'Grayscale')),
        ]

        sample_cut_choices = _copy_dict(choices.get('sample_cut_mode', {}))
        self.fields['sample_cut_mode'].choices = [
            (SampleCutMode.disk.value, _read_text(sample_cut_choices, SampleCutMode.disk.value, 'To disk')),
            (SampleCutMode.online.value, _read_text(sample_cut_choices, SampleCutMode.online.value, 'Online')),
        ]

        validation_source_choices = _copy_dict(choices.get('validation_source', {}))
        self.fields['validation_source'].choices = [
            (
                ValidationSource.split.value,
                _read_text(validation_source_choices, ValidationSource.split.value, 'Split'),
            ),
            (
                ValidationSource.external.value,
                _read_text(validation_source_choices, ValidationSource.external.value, 'External folders'),
            ),
        ]

        optimizer_choices = _copy_dict(choices.get('optimizer_name', {}))
        self.fields['optimizer_name'].choices = [
            (optimizer.value, _read_text(optimizer_choices, optimizer.value, optimizer.value))
            for optimizer in OptimizerName
        ]

        scheduler_choices = _copy_dict(choices.get('scheduler_name', {}))
        self.fields['scheduler_name'].choices = [
            (scheduler.value, _read_text(scheduler_choices, scheduler.value, scheduler.value))
            for scheduler in SchedulerName
        ]

        mixed_precision_choices = _copy_dict(choices.get('mixed_precision', {}))
        self.fields['mixed_precision'].choices = [
            (mode.value, _read_text(mixed_precision_choices, mode.value, mode.value.upper()))
            for mode in MixedPrecisionMode
        ]

        anneal_strategy_choices = _copy_dict(choices.get('scheduler_one_cycle_anneal_strategy', {}))
        self.fields['scheduler_one_cycle_anneal_strategy'].choices = [
            ('cos', _read_text(anneal_strategy_choices, 'cos', 'cos')),
            ('linear', _read_text(anneal_strategy_choices, 'linear', 'linear')),
        ]

        loss_function_choices = _copy_dict(choices.get('loss_function', {}))
        default_loss_labels = {
            'bce': 'BCE',
            'dice': 'Dice',
            'iou': 'IoU',
            'focal_bce': 'Focal BCE',
            'boundary': 'Boundary',
            'focal_tversky': 'Focal Tversky',
            'ce': 'CE',
        }
        self.fields['loss_function'].choices = [
            (
                loss_name,
                _read_text(loss_function_choices, loss_name, default_loss_labels[loss_name]),
            )
            for loss_name in LOSS_SELECTION_NAMES
        ]

    def clean(self):
        cleaned = super().clean()
        use_validation = bool(cleaned.get('use_validation', False))
        validation_source = normalize_validation_source(cleaned.get('validation_source'))
        validation_image_folder = str(cleaned.get('validation_image_folder') or '').strip()
        validation_label_folder = str(cleaned.get('validation_label_folder') or '').strip()

        cleaned['validation_source'] = validation_source
        cleaned['validation_image_folder'] = validation_image_folder
        cleaned['validation_label_folder'] = validation_label_folder

        if not use_validation or validation_source != ValidationSource.external.value:
            return cleaned

        if not validation_image_folder or not Path(validation_image_folder).is_dir():
            self.add_error('validation_image_folder', 'Validation image folder must point to an existing directory.')
        if not validation_label_folder or not Path(validation_label_folder).is_dir():
            self.add_error('validation_label_folder', 'Validation label folder must point to an existing directory.')

        return cleaned

    def to_state(self) -> SettingsState:
        cleaned = self.cleaned_data
        defaults = SettingsState()

        def _with_default(name: str):
            value = cleaned.get(name)
            if value is not None:
                return value
            fallback = {
                'target_x': defaults.target_size[0],
                'target_y': defaults.target_size[1],
            }
            if name in fallback:
                return fallback[name]
            return getattr(defaults, name)

        return SettingsState(
            step=cleaned['step'],
            vertical_rotation=cleaned.get('vertical_rotation', False),
            horizontal_rotation=cleaned.get('horizontal_rotation', False),
            additional_augmentation=cleaned.get('additional_augmentation', False),
            random_crop=cleaned.get('random_crop', False),
            crops_per_image=_with_default('crops_per_image'),
            scale_augmentation=cleaned.get('scale_augmentation', False),
            scale_augmentation_strength=_with_default('scale_augmentation_strength'),
            augmentation_brightness_strength=_with_default('augmentation_brightness_strength'),
            augmentation_contrast_strength=_with_default('augmentation_contrast_strength'),
            augmentation_noise_probability=_with_default('augmentation_noise_probability'),
            augmentation_noise_sigma=_with_default('augmentation_noise_sigma'),
            sample_size=(cleaned['sample_x'], cleaned['sample_y']),
            model=cleaned['model'],
            color_mode=cleaned['color_mode'],
            shuffle=cleaned.get('shuffle', False),
            use_validation=cleaned.get('use_validation', False),
            validation_source=normalize_validation_source(cleaned.get('validation_source')),
            validation_percent=_with_default('validation_percent'),
            validation_image_folder=str(cleaned.get('validation_image_folder', '') or ''),
            validation_label_folder=str(cleaned.get('validation_label_folder', '') or ''),
            save_validation_binary_images=cleaned.get('save_validation_binary_images', False),
            sample_cut_mode=cleaned['sample_cut_mode'],
            batch_size=cleaned['batch_size'],
            dataloader_num_workers=_with_default('dataloader_num_workers'),
            overlap=cleaned['overlap'],
            log_update_frequency=cleaned['log_update_frequency'],
            crop_enabled=cleaned.get('crop_enabled', False),
            resize_enabled=cleaned.get('resize_enabled', False),
            edge_cut_size=_with_default('edge_cut_size'),
            target_size=(
                _with_default('target_x'),
                _with_default('target_y'),
            ),
            optimizer_name=cleaned['optimizer_name'],
            mixed_precision=cleaned['mixed_precision'],
            loss_function=cleaned['loss_function'],
            loss_term_weights=resolve_loss_term_weights(
                {},
                fallback_loss_function=cleaned['loss_function'],
            ),
            dice_loss_weight=_with_default('dice_loss_weight'),
            iou_loss_weight=_with_default('iou_loss_weight'),
            learning_rate=cleaned['learning_rate'],
            weight_decay=cleaned['weight_decay'],
            warmup_enabled=cleaned.get('warmup_enabled', False),
            warmup_epochs=cleaned['warmup_epochs'],
            warmup_start_factor=cleaned['warmup_start_factor'],
            scheduler_name=normalize_scheduler_name(cleaned.get('scheduler_name')),
            scheduler_plateau_factor=_with_default('scheduler_plateau_factor'),
            scheduler_plateau_patience=_with_default('scheduler_plateau_patience'),
            scheduler_plateau_threshold=_with_default('scheduler_plateau_threshold'),
            scheduler_plateau_min_lr=_with_default('scheduler_plateau_min_lr'),
            scheduler_plateau_cooldown=_with_default('scheduler_plateau_cooldown'),
            scheduler_cosine_t_max=_with_default('scheduler_cosine_t_max'),
            scheduler_cosine_eta_min=_with_default('scheduler_cosine_eta_min'),
            scheduler_one_cycle_max_lr=_with_default('scheduler_one_cycle_max_lr'),
            scheduler_one_cycle_pct_start=_with_default('scheduler_one_cycle_pct_start'),
            scheduler_one_cycle_anneal_strategy=(
                str(cleaned.get('scheduler_one_cycle_anneal_strategy') or '')
                or defaults.scheduler_one_cycle_anneal_strategy
            ),
            scheduler_one_cycle_div_factor=_with_default('scheduler_one_cycle_div_factor'),
            scheduler_one_cycle_final_div_factor=_with_default('scheduler_one_cycle_final_div_factor'),
            scheduler_one_cycle_three_phase=cleaned.get('scheduler_one_cycle_three_phase', False),
            scheduler_step_lr_step_size=_with_default('scheduler_step_lr_step_size'),
            scheduler_step_lr_gamma=_with_default('scheduler_step_lr_gamma'),
            hard_mining_enabled=cleaned.get('hard_mining_enabled', False),
            hard_mining_strength=_with_default('hard_mining_strength'),
            hard_mining_ema_alpha=_with_default('hard_mining_ema_alpha'),
            hard_pixel_mining_enabled=cleaned.get('hard_pixel_mining_enabled', False),
            hard_pixel_mining_ratio=_with_default('hard_pixel_mining_ratio'),
            cutout_enabled=cleaned.get('cutout_enabled', False),
            cutout_probability=_with_default('cutout_probability'),
            cutout_holes=_with_default('cutout_holes'),
            cutout_size_ratio=_with_default('cutout_size_ratio'),
            random_artifacts_enabled=cleaned.get('random_artifacts_enabled', False),
            random_artifacts_probability=_with_default('random_artifacts_probability'),
            random_artifacts_count=_with_default('random_artifacts_count'),
            random_artifacts_size_ratio=_with_default('random_artifacts_size_ratio'),
            mixup_enabled=cleaned.get('mixup_enabled', False),
            mixup_probability=_with_default('mixup_probability'),
            mixup_alpha=_with_default('mixup_alpha'),
            skip_uniform_labels=cleaned.get('skip_uniform_labels', False),
            early_stopping_enabled=cleaned.get('early_stopping_enabled', False),
            early_stopping_patience=cleaned['early_stopping_patience'],
            early_stopping_min_delta=cleaned['early_stopping_min_delta'],
            early_stopping_restore_best_weights=cleaned.get('early_stopping_restore_best_weights', False),
            torch_compile_enabled=cleaned.get('torch_compile_enabled', False),
            show_batch_preview=cleaned.get('show_batch_preview', False),
            use_multi_gpu=cleaned.get('use_multi_gpu', False),
        )


def defaults_from_main_state(state: MainWindowState) -> dict:
    return asdict(state)


def defaults_from_settings_state(state: SettingsState) -> dict:
    resolved_loss_weights = resolve_loss_term_weights(
        getattr(state, 'loss_term_weights', None),
        fallback_loss_function=state.loss_function,
        dice_weight=float(getattr(state, 'dice_loss_weight', 0.5)),
        iou_weight=float(getattr(state, 'iou_loss_weight', 0.5)),
    )
    return {
        'step': state.step,
        'vertical_rotation': state.vertical_rotation,
        'horizontal_rotation': state.horizontal_rotation,
        'additional_augmentation': state.additional_augmentation,
        'random_crop': getattr(state, 'random_crop', False),
        'crops_per_image': getattr(state, 'crops_per_image', 64),
        'scale_augmentation': getattr(state, 'scale_augmentation', False),
        'scale_augmentation_strength': getattr(state, 'scale_augmentation_strength', 0.2),
        'augmentation_brightness_strength': state.augmentation_brightness_strength,
        'augmentation_contrast_strength': state.augmentation_contrast_strength,
        'augmentation_noise_probability': state.augmentation_noise_probability,
        'augmentation_noise_sigma': state.augmentation_noise_sigma,
        'sample_x': state.sample_size[0],
        'sample_y': state.sample_size[1],
        'model': state.model,
        'color_mode': state.color_mode,
        'shuffle': state.shuffle,
        'use_validation': state.use_validation,
        'validation_source': getattr(state, 'validation_source', ValidationSource.split.value),
        'validation_percent': state.validation_percent,
        'validation_image_folder': getattr(state, 'validation_image_folder', ''),
        'validation_label_folder': getattr(state, 'validation_label_folder', ''),
        'save_validation_binary_images': getattr(state, 'save_validation_binary_images', False),
        'sample_cut_mode': state.sample_cut_mode,
        'batch_size': state.batch_size,
        'dataloader_num_workers': getattr(state, 'dataloader_num_workers', -1),
        'overlap': state.overlap,
        'log_update_frequency': state.log_update_frequency,
        'crop_enabled': state.crop_enabled,
        'resize_enabled': state.resize_enabled,
        'edge_cut_size': state.edge_cut_size,
        'target_x': state.target_size[0],
        'target_y': state.target_size[1],
        'optimizer_name': state.optimizer_name,
        'mixed_precision': state.mixed_precision,
        'loss_function': dominant_loss_function(
            resolved_loss_weights,
            fallback=state.loss_function,
        ),
        'dice_loss_weight': state.dice_loss_weight,
        'iou_loss_weight': state.iou_loss_weight,
        'learning_rate': state.learning_rate,
        'weight_decay': state.weight_decay,
        'warmup_enabled': state.warmup_enabled,
        'warmup_epochs': state.warmup_epochs,
        'warmup_start_factor': state.warmup_start_factor,
        'scheduler_name': normalize_scheduler_name(getattr(state, 'scheduler_name', 'off')),
        'scheduler_plateau_factor': getattr(state, 'scheduler_plateau_factor', 0.5),
        'scheduler_plateau_patience': getattr(state, 'scheduler_plateau_patience', 3),
        'scheduler_plateau_threshold': getattr(state, 'scheduler_plateau_threshold', 1e-4),
        'scheduler_plateau_min_lr': getattr(state, 'scheduler_plateau_min_lr', 1e-6),
        'scheduler_plateau_cooldown': getattr(state, 'scheduler_plateau_cooldown', 0),
        'scheduler_cosine_t_max': getattr(state, 'scheduler_cosine_t_max', 10),
        'scheduler_cosine_eta_min': getattr(state, 'scheduler_cosine_eta_min', 1e-6),
        'scheduler_one_cycle_max_lr': getattr(state, 'scheduler_one_cycle_max_lr', 1e-3),
        'scheduler_one_cycle_pct_start': getattr(state, 'scheduler_one_cycle_pct_start', 0.3),
        'scheduler_one_cycle_anneal_strategy': getattr(state, 'scheduler_one_cycle_anneal_strategy', 'cos'),
        'scheduler_one_cycle_div_factor': getattr(state, 'scheduler_one_cycle_div_factor', 25.0),
        'scheduler_one_cycle_final_div_factor': getattr(state, 'scheduler_one_cycle_final_div_factor', 10000.0),
        'scheduler_one_cycle_three_phase': getattr(state, 'scheduler_one_cycle_three_phase', False),
        'scheduler_step_lr_step_size': getattr(state, 'scheduler_step_lr_step_size', 10),
        'scheduler_step_lr_gamma': getattr(state, 'scheduler_step_lr_gamma', 0.1),
        'hard_mining_enabled': state.hard_mining_enabled,
        'hard_mining_strength': state.hard_mining_strength,
        'hard_mining_ema_alpha': state.hard_mining_ema_alpha,
        'hard_pixel_mining_enabled': getattr(state, 'hard_pixel_mining_enabled', False),
        'hard_pixel_mining_ratio': getattr(state, 'hard_pixel_mining_ratio', 0.25),
        'cutout_enabled': getattr(state, 'cutout_enabled', False),
        'cutout_probability': getattr(state, 'cutout_probability', 1.0),
        'cutout_holes': getattr(state, 'cutout_holes', 1),
        'cutout_size_ratio': getattr(state, 'cutout_size_ratio', 0.25),
        'random_artifacts_enabled': getattr(state, 'random_artifacts_enabled', False),
        'random_artifacts_probability': getattr(state, 'random_artifacts_probability', 1.0),
        'random_artifacts_count': getattr(state, 'random_artifacts_count', 1),
        'random_artifacts_size_ratio': getattr(state, 'random_artifacts_size_ratio', 0.25),
        'mixup_enabled': getattr(state, 'mixup_enabled', False),
        'mixup_probability': getattr(state, 'mixup_probability', 1.0),
        'mixup_alpha': getattr(state, 'mixup_alpha', 0.2),
        'skip_uniform_labels': state.skip_uniform_labels,
        'early_stopping_enabled': state.early_stopping_enabled,
        'early_stopping_patience': state.early_stopping_patience,
        'early_stopping_min_delta': state.early_stopping_min_delta,
        'early_stopping_restore_best_weights': state.early_stopping_restore_best_weights,
        'torch_compile_enabled': state.torch_compile_enabled,
        'show_batch_preview': state.show_batch_preview,
        'use_multi_gpu': state.use_multi_gpu,
    }

