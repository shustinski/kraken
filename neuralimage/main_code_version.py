import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from lib.data_interfaces import (
    EarlyStoppingParameters,
    HardMiningParameters,
    MixedPrecisionMode,
    OptimizerName,
    OptimizerParameters,
    RecognitionParameters,
    SampleCutMode,
    SampleGenerationSettings,
    SamplePrepareSettings,
    TrainingParameters,
    WarmupParameters,
    WorkMode,
    parse_work_mode,
)
from lib.message_bus import MessageBus
from model.general_neural_handler import GeneralNeuralHandler


def _log(message: Any) -> None:
    print(str(message))


def _question(theme: str, message: str) -> bool:
    while True:
        print(theme)
        print(message)
        answer = input('Y/N: ').strip().lower()
        if answer in ('y', 'yes', 'д', 'да'):
            return True
        if answer in ('n', 'no', 'н', 'нет'):
            return False
        print('Cannot parse the answer. Please enter Y or N.')


def _to_tuple2(value: Any, field_name: str) -> tuple[int, int]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f'{field_name} must be [x, y].')
    return int(value[0]), int(value[1])


def _to_work_mode(value: str) -> WorkMode:
    mode = parse_work_mode(value)
    if mode is not None:
        return mode
    raise ValueError(
        f"Unknown work_mode: {value!r}. "
        "Use one of: train_only, train_and_recognition, recognition_only, further_training."
    )


def _build_training_parameters(raw: dict[str, Any]) -> TrainingParameters:
    generation_raw = raw.get('generation', {})
    prepare_raw = raw.get('prepare', {})
    optimizer_raw = raw.get('optimizer', {})
    early_stopping_raw = raw.get('early_stopping', {})
    warmup_raw = raw.get('warmup', {})
    hard_mining_raw = raw.get('hard_mining', {})

    generation = SampleGenerationSettings(
        step=int(generation_raw.get('step', 100)),
        segment_size=_to_tuple2(generation_raw.get('segment_size', [256, 256]), 'generation.segment_size'),
        vertical_rotation=bool(generation_raw.get('vertical_rotation', True)),
        horizontal_rotation=bool(generation_raw.get('horizontal_rotation', True)),
        channels=int(generation_raw.get('channels', 3)),
        additional_augmentation=bool(generation_raw.get('additional_augmentation', False)),
        augmentation_brightness_strength=float(generation_raw.get('augmentation_brightness_strength', 0.1)),
        augmentation_contrast_strength=float(generation_raw.get('augmentation_contrast_strength', 0.1)),
        augmentation_noise_probability=float(generation_raw.get('augmentation_noise_probability', 0.5)),
        augmentation_noise_sigma=float(generation_raw.get('augmentation_noise_sigma', 0.01)),
    )

    edge_cut_raw = prepare_raw.get('edge_cut')
    target_size_raw = prepare_raw.get('target_size')
    prepare = SamplePrepareSettings(
        enable_crop=bool(prepare_raw.get('enable_crop', False)),
        enable_resize=bool(prepare_raw.get('enable_resize', False)),
        edge_cut=_to_tuple2(edge_cut_raw, 'prepare.edge_cut') if edge_cut_raw is not None else None,
        target_size=_to_tuple2(target_size_raw, 'prepare.target_size') if target_size_raw is not None else None,
    )

    optimizer = OptimizerParameters(
        name=OptimizerName(optimizer_raw.get('name', OptimizerName.adam.value)),
        learning_rate=float(optimizer_raw.get('learning_rate', 1e-3)),
        weight_decay=float(optimizer_raw.get('weight_decay', 0.0)),
    )

    early_stopping = EarlyStoppingParameters(
        enabled=bool(early_stopping_raw.get('enabled', False)),
        patience=int(early_stopping_raw.get('patience', 10)),
        min_delta=float(early_stopping_raw.get('min_delta', 0.0)),
        restore_best_weights=bool(early_stopping_raw.get('restore_best_weights', True)),
    )

    warmup = WarmupParameters(
        enabled=bool(warmup_raw.get('enabled', False)),
        epochs=int(warmup_raw.get('epochs', 3)),
        start_factor=float(warmup_raw.get('start_factor', 0.1)),
    )

    hard_mining = HardMiningParameters(
        enabled=bool(hard_mining_raw.get('enabled', False)),
        strength=float(hard_mining_raw.get('strength', 2.0)),
        ema_alpha=float(hard_mining_raw.get('ema_alpha', 0.2)),
    )

    return TrainingParameters(
        image_path=Path(raw.get('image_path', '')),
        label_path=Path(raw.get('label_path', '')),
        shuffle=bool(raw.get('shuffle', True)),
        validation=bool(raw.get('validation', False)),
        validation_percent=int(raw.get('validation_percent', 20)),
        batch_size=int(raw.get('batch_size', 16)),
        cut_mode=SampleCutMode(raw.get('cut_mode', SampleCutMode.online.value)),
        colors=int(raw.get('colors', generation.channels)),
        epochs=int(raw.get('epochs', 20)),
        generation=generation,
        prepare=prepare,
        optimizer=optimizer,
        mixed_precision=MixedPrecisionMode(raw.get('mixed_precision', MixedPrecisionMode.bf16.value)),
        loss_function=str(raw.get('loss_function', 'bce')),
        dice_loss_weight=float(raw.get('dice_loss_weight', 0.5)),
        iou_loss_weight=float(raw.get('iou_loss_weight', 0.5)),
        early_stopping=early_stopping,
        warmup=warmup,
        hard_mining=hard_mining,
        skip_uniform_labels=bool(raw.get('skip_uniform_labels', False)),
        use_multi_gpu=bool(raw.get('use_multi_gpu', True)),
        show_batch_preview=bool(raw.get('show_batch_preview', True)),
        log_update_frequency=int(raw.get('log_update_frequency', 0)),
    )


def _build_recognition_parameters(raw: dict[str, Any]) -> RecognitionParameters:
    source_files = [Path(p) for p in raw.get('source_files', [])]
    model_value = raw.get('model', '')
    model = Path(model_value) if isinstance(model_value, str) and model_value.lower().endswith('.pth') else model_value

    return RecognitionParameters(
        source_files=source_files,
        result_folder=Path(raw.get('result_folder', '')),
        model=model,
        part_size=_to_tuple2(raw.get('part_size', [256, 256]), 'recogniton_parameters.part_size'),
        batch_size=int(raw.get('batch_size', 16)),
        overlap=int(raw.get('overlap', 8)),
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='CLI runner using direct business settings: work_mode, recogniton_parameters, tranining_parameters.'
    )
    parser.add_argument('--config', type=Path, help='JSON config path.')
    parser.add_argument('--print-config-template', action='store_true', help='Print JSON template and exit.')
    parser.add_argument('--work-mode', default='train_and_recognition', help='Override work mode from config.')
    return parser


def _config_template() -> dict[str, Any]:
    return {
        'work_mode': 'train_and_recognition',
        'recogniton_parameters': {
            'source_files': ['D:/data/inference/source_1.jpg'],
            'result_folder': 'D:/data/inference/results',
            'model': 'M 720k',
            'part_size': [256, 256],
            'batch_size': 8,
            'overlap': 16,
        },
        'tranining_parameters': {
            'image_path': 'D:/data/train/images',
            'label_path': 'D:/data/train/labels',
            'shuffle': True,
            'validation': True,
            'validation_percent': 20,
            'batch_size': 8,
            'cut_mode': 'online',
            'colors': 3,
            'epochs': 30,
            'generation': {
                'step': 128,
                'segment_size': [256, 256],
                'vertical_rotation': True,
                'horizontal_rotation': True,
                'channels': 3,
                'additional_augmentation': False,
                'augmentation_brightness_strength': 0.1,
                'augmentation_contrast_strength': 0.1,
                'augmentation_noise_probability': 0.5,
                'augmentation_noise_sigma': 0.01,
            },
            'prepare': {
                'enable_crop': False,
                'enable_resize': False,
                'edge_cut': None,
                'target_size': None,
            },
            'optimizer': {
                'name': 'adamw',
                'learning_rate': 0.001,
                'weight_decay': 0.0001,
            },
            'mixed_precision': 'bf16',
            'loss_function': 'bce_dice',
            'dice_loss_weight': 0.7,
            'iou_loss_weight': 0.3,
            'early_stopping': {
                'enabled': True,
                'patience': 10,
                'min_delta': 0.001,
                'restore_best_weights': True,
            },
            'warmup': {
                'enabled': True,
                'epochs': 3,
                'start_factor': 0.1,
            },
            'hard_mining': {
                'enabled': False,
                'strength': 2.0,
                'ema_alpha': 0.2,
            },
            'skip_uniform_labels': False,
            'use_multi_gpu': True,
            'show_batch_preview': True,
            'log_update_frequency': 50,
        },
    }


def _load_settings(path: Path, work_mode_override: str | None = None) -> tuple[WorkMode, TrainingParameters, RecognitionParameters]:
    payload = json.loads(path.read_text(encoding='utf-8'))

    work_mode_raw = work_mode_override if work_mode_override else payload.get('work_mode', 'train_and_recognition')
    work_mode = _to_work_mode(work_mode_raw)
    training_parameters = _build_training_parameters(payload.get('tranining_parameters', {}))
    recognition_parameters = _build_recognition_parameters(payload.get('recogniton_parameters', {}))
    return work_mode, training_parameters, recognition_parameters


def main(argv: Sequence[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)

    if args.print_config_template:
        print(json.dumps(_config_template(), ensure_ascii=False, indent=2))
        return

    if args.config is None:
        raise ValueError('Config is required. Use --config <path> or --print-config-template.')

    mode_override = args.work_mode if args.work_mode else None
    work_mode, training_parameters, recognition_parameters = _load_settings(args.config, mode_override)

    bus = MessageBus()
    bus.subscribe('logging', _log)
    bus.subscribe('training', _log)
    bus.subscribe('error', _log)

    handler = GeneralNeuralHandler(
        work_mode=work_mode,
        recogniton_parameters=recognition_parameters,
        tranining_parameters=training_parameters,
        question_module=_question,
        message_bus=bus,
    )
    handler.start()


if __name__ == '__main__':
    main()
