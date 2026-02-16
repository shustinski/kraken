from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from django import forms
from django.core.exceptions import ValidationError

from lib.data_interfaces import OptimizerName, MixedPrecisionMode, SampleCutMode, WorkMode
from view.window_dataclasses import MainWindowState, SettingsState


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


class MainWindowForm(forms.Form):
    WORK_MODE_CHOICES = (
        (WorkMode.train_and_recognition.value, 'Train and recognition'),
        (WorkMode.futher_training.value, 'Further training and recognition'),
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
        super().__init__(*args, **kwargs)
        self.fields['work_mode'].widget.attrs.update(_BASE_SELECT_ATTRS)

        self.fields['source_folder'].widget.attrs.update(_BASE_TEXT_INPUT_ATTRS | {'placeholder': r'D:\data\source'})
        self.fields['result_folder'].widget.attrs.update(_BASE_TEXT_INPUT_ATTRS | {'placeholder': r'D:\data\result'})
        self.fields['sample_folder'].widget.attrs.update(_BASE_TEXT_INPUT_ATTRS | {'placeholder': r'D:\data\sample'})
        self.fields['label_folder'].widget.attrs.update(_BASE_TEXT_INPUT_ATTRS | {'placeholder': r'D:\data\label'})
        self.fields['model_path'].widget.attrs.update(_BASE_TEXT_INPUT_ATTRS | {'placeholder': r'D:\models\model.pth'})
        self.fields['epochs'].widget.attrs.update(_BASE_NUM_INPUT_ATTRS)

    def clean(self):
        cleaned = super().clean()
        work_mode = cleaned.get('work_mode', '')
        source_folder = (cleaned.get('source_folder') or '').strip()
        result_folder = (cleaned.get('result_folder') or '').strip()
        sample_folder = (cleaned.get('sample_folder') or '').strip()
        label_folder = (cleaned.get('label_folder') or '').strip()
        model_path = (cleaned.get('model_path') or '').strip()

        if not source_folder or not Path(source_folder).is_dir():
            raise ValidationError('Field "Source folder" must point to an existing directory.')
        if not result_folder or not Path(result_folder).is_dir():
            raise ValidationError('Field "Result folder" must point to an existing directory.')

        if work_mode in (WorkMode.train_and_recognition.value, WorkMode.futher_training.value):
            if not sample_folder or not Path(sample_folder).is_dir():
                raise ValidationError('Training mode requires an existing sample folder.')
            if not label_folder or not Path(label_folder).is_dir():
                raise ValidationError('Training mode requires an existing label folder.')

        if work_mode in (WorkMode.recognition_only.value, WorkMode.futher_training.value):
            if not model_path or not Path(model_path).is_file():
                raise ValidationError('Selected mode requires a valid model file path (.pth).')

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
    sample_x = forms.IntegerField(label='Sample X', min_value=8, max_value=4096)
    sample_y = forms.IntegerField(label='Sample Y', min_value=8, max_value=4096)
    model = forms.CharField(label='Model architecture')
    color_mode = forms.ChoiceField(label='Color mode', choices=[('RGB', 'RGB'), ('ЧБ', 'ЧБ')])
    use_validation = forms.BooleanField(label='Use validation', required=False)
    validation_percent = forms.IntegerField(label='Validation percent', min_value=0, max_value=90)
    shuffle = forms.BooleanField(label='Shuffle dataset', required=False)
    sample_cut_mode = forms.ChoiceField(
        label='Sample cut mode',
        choices=[(SampleCutMode.disk.value, 'To disk'), (SampleCutMode.online.value, 'Online')],
    )
    batch_size = forms.IntegerField(label='Batch size', min_value=1, max_value=512)
    overlap = forms.IntegerField(label='Overlap', min_value=0, max_value=256)
    additional_processing = forms.BooleanField(label='Additional processing', required=False)
    edge_cut_size = forms.IntegerField(label='Edge cut size', min_value=0, max_value=2000)
    target_x = forms.IntegerField(label='Target X', min_value=0, max_value=10000)
    target_y = forms.IntegerField(label='Target Y', min_value=0, max_value=10000)
    optimizer_name = forms.ChoiceField(label='Optimizer', choices=[(o.value, o.value) for o in OptimizerName])
    mixed_precision = forms.ChoiceField(
        label='Mixed precision',
        choices=[(m.value, m.value) for m in MixedPrecisionMode],
    )
    learning_rate = forms.FloatField(label='Learning rate', min_value=1e-8, max_value=10)
    weight_decay = forms.FloatField(label='Weight decay', min_value=0.0, max_value=10)
    warmup_enabled = forms.BooleanField(label='Enable warmup', required=False)
    warmup_epochs = forms.IntegerField(label='Warmup epochs', min_value=1, max_value=2000)
    warmup_start_factor = forms.FloatField(label='Warmup start factor', min_value=0.0, max_value=1.0)
    early_stopping_enabled = forms.BooleanField(label='Enable early stopping', required=False)
    early_stopping_patience = forms.IntegerField(label='Early stopping patience', min_value=0, max_value=2000)
    early_stopping_min_delta = forms.FloatField(label='Early stopping min delta', min_value=0.0, max_value=10.0)
    early_stopping_restore_best_weights = forms.BooleanField(label='Restore best weights', required=False)
    show_batch_preview = forms.BooleanField(label='Show batch preview', required=False)
    use_multi_gpu = forms.BooleanField(label='Use multi-GPU', required=False)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for field in self.fields.values():
            field.widget.attrs.update(_START_FORM_ATTRS)

        for field_name in (
            'step',
            'sample_x',
            'sample_y',
            'validation_percent',
            'batch_size',
            'overlap',
            'edge_cut_size',
            'target_x',
            'target_y',
            'learning_rate',
            'weight_decay',
            'warmup_epochs',
            'warmup_start_factor',
            'early_stopping_patience',
            'early_stopping_min_delta',
        ):
            self.fields[field_name].widget.attrs.update(_BASE_NUM_INPUT_ATTRS)

        for field_name in ('color_mode', 'sample_cut_mode', 'optimizer_name', 'mixed_precision'):
            self.fields[field_name].widget.attrs.update(_BASE_SELECT_ATTRS)

        self.fields['model'].widget.attrs.update(_BASE_TEXT_INPUT_ATTRS | {'placeholder': 'M 720k'})

    def to_state(self) -> SettingsState:
        cleaned = self.cleaned_data
        return SettingsState(
            step=cleaned['step'],
            vertical_rotation=cleaned.get('vertical_rotation', False),
            horizontal_rotation=cleaned.get('horizontal_rotation', False),
            sample_size=(cleaned['sample_x'], cleaned['sample_y']),
            model=cleaned['model'],
            color_mode=cleaned['color_mode'],
            shuffle=cleaned.get('shuffle', False),
            use_validation=cleaned.get('use_validation', False),
            validation_percent=cleaned['validation_percent'],
            sample_cut_mode=cleaned['sample_cut_mode'],
            batch_size=cleaned['batch_size'],
            overlap=cleaned['overlap'],
            additional_processing=cleaned.get('additional_processing', False),
            edge_cut_size=cleaned['edge_cut_size'],
            target_size=(cleaned['target_x'], cleaned['target_y']),
            optimizer_name=cleaned['optimizer_name'],
            mixed_precision=cleaned['mixed_precision'],
            learning_rate=cleaned['learning_rate'],
            weight_decay=cleaned['weight_decay'],
            warmup_enabled=cleaned.get('warmup_enabled', False),
            warmup_epochs=cleaned['warmup_epochs'],
            warmup_start_factor=cleaned['warmup_start_factor'],
            early_stopping_enabled=cleaned.get('early_stopping_enabled', False),
            early_stopping_patience=cleaned['early_stopping_patience'],
            early_stopping_min_delta=cleaned['early_stopping_min_delta'],
            early_stopping_restore_best_weights=cleaned.get('early_stopping_restore_best_weights', False),
            show_batch_preview=cleaned.get('show_batch_preview', False),
            use_multi_gpu=cleaned.get('use_multi_gpu', False),
        )


def defaults_from_main_state(state: MainWindowState) -> dict:
    return asdict(state)


def defaults_from_settings_state(state: SettingsState) -> dict:
    return {
        'step': state.step,
        'vertical_rotation': state.vertical_rotation,
        'horizontal_rotation': state.horizontal_rotation,
        'sample_x': state.sample_size[0],
        'sample_y': state.sample_size[1],
        'model': state.model,
        'color_mode': state.color_mode,
        'shuffle': state.shuffle,
        'use_validation': state.use_validation,
        'validation_percent': state.validation_percent,
        'sample_cut_mode': state.sample_cut_mode,
        'batch_size': state.batch_size,
        'overlap': state.overlap,
        'additional_processing': state.additional_processing,
        'edge_cut_size': state.edge_cut_size,
        'target_x': state.target_size[0],
        'target_y': state.target_size[1],
        'optimizer_name': state.optimizer_name,
        'mixed_precision': state.mixed_precision,
        'learning_rate': state.learning_rate,
        'weight_decay': state.weight_decay,
        'warmup_enabled': state.warmup_enabled,
        'warmup_epochs': state.warmup_epochs,
        'warmup_start_factor': state.warmup_start_factor,
        'early_stopping_enabled': state.early_stopping_enabled,
        'early_stopping_patience': state.early_stopping_patience,
        'early_stopping_min_delta': state.early_stopping_min_delta,
        'early_stopping_restore_best_weights': state.early_stopping_restore_best_weights,
        'show_batch_preview': state.show_batch_preview,
        'use_multi_gpu': state.use_multi_gpu,
    }

