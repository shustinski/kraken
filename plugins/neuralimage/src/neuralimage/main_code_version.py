import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from neuralimage.lib.data_interfaces import (
    build_pcb_defect_parameters,
    CutoutParameters,
    EarlyStoppingParameters,
    HardMiningParameters,
    MixedPrecisionMode,
    MixupParameters,
    OptimizerName,
    OptimizerParameters,
    RandomArtifactsParameters,
    RecognitionParameters,
    SampleCutMode,
    SampleGenerationSettings,
    SamplePrepareSettings,
    SchedulerName,
    SchedulerParameters,
    TrainingParameters,
    WarmupParameters,
    WorkMode,
    normalize_scheduler_name,
    parse_work_mode,
    normalize_multi_gpu_mode,
)
from neuralimage.lib.message_bus import MessageBus
from neuralimage.model.general_neural_handler import GeneralNeuralHandler


def _log(message: Any) -> None:
    print(str(message))


def _question(
    text: str,
    header: str,
    default_answer: bool = False,
    timeout_seconds: int | None = None,
) -> bool:
    while True:
        print(header)
        print(text)
        if timeout_seconds:
            default_label = 'Y' if default_answer else 'N'
            print(f'Default answer is {default_label}; CLI auto-timeout is not supported.')
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
        "Use one of: train_only, continue_training, train_and_recognition, recognition_only, further_training."
    )


def _build_training_parameters(raw: dict[str, Any], *, model_name: str | None = None) -> TrainingParameters:
    generation_raw = raw.get('generation', {})
    prepare_raw = raw.get('prepare', {})
    optimizer_raw = raw.get('optimizer', {})
    early_stopping_raw = raw.get('early_stopping', {})
    warmup_raw = raw.get('warmup', {})
    scheduler_raw = raw.get('scheduler', {})
    hard_mining_raw = raw.get('hard_mining', {})
    cutout_raw = raw.get('cutout', {})
    random_artifacts_raw = raw.get('random_artifacts', {})
    mixup_raw = raw.get('mixup', {})
    pcb_defects_raw = raw.get('pcb_defects', {})
    resolved_model_name = str(model_name or raw.get('model_name', '')).strip()
    default_context_branch = resolved_model_name in {'FrameUnet', 'quasi_dual_scale_unet', 'UNetWithContextBranch'}
    local_crop_size = _to_tuple2(
        raw.get('local_crop_size', generation_raw.get('segment_size', [256, 256])),
        'tranining_parameters.local_crop_size',
    )

    generation = SampleGenerationSettings(
        step=int(generation_raw.get('step', 100)),
        segment_size=local_crop_size,
        vertical_rotation=bool(generation_raw.get('vertical_rotation', True)),
        horizontal_rotation=bool(generation_raw.get('horizontal_rotation', True)),
        channels=int(generation_raw.get('channels', 3)),
        additional_augmentation=bool(generation_raw.get('additional_augmentation', False)),
        augmentation_brightness_strength=float(generation_raw.get('augmentation_brightness_strength', 0.1)),
        augmentation_contrast_strength=float(generation_raw.get('augmentation_contrast_strength', 0.1)),
        augmentation_noise_probability=float(generation_raw.get('augmentation_noise_probability', 0.5)),
        augmentation_noise_sigma=float(generation_raw.get('augmentation_noise_sigma', 0.01)),
        random_crop=bool(generation_raw.get('random_crop', False)),
        crops_per_image=int(generation_raw.get('crops_per_image', 64)),
        scale_augmentation=bool(generation_raw.get('scale_augmentation', False)),
        scale_augmentation_strength=float(generation_raw.get('scale_augmentation_strength', 0.2)),
    )

    edge_cut_raw = prepare_raw.get('edge_cut')
    target_size_raw = prepare_raw.get('target_size')
    compression_factor = max(1, int(prepare_raw.get('compression_factor', 1) or 1))
    prepare = SamplePrepareSettings(
        enable_crop=bool(prepare_raw.get('enable_crop', False)),
        enable_resize=bool(prepare_raw.get('enable_resize', False)),
        edge_cut=_to_tuple2(edge_cut_raw, 'prepare.edge_cut') if edge_cut_raw is not None else None,
        target_size=(
            _to_tuple2(target_size_raw, 'prepare.target_size')
            if target_size_raw is not None and compression_factor <= 1
            else None
        ),
        compression_factor=compression_factor,
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

    scheduler = SchedulerParameters(
        name=SchedulerName(normalize_scheduler_name(scheduler_raw.get('name', SchedulerName.off.value))),
        plateau_factor=float(scheduler_raw.get('plateau_factor', 0.5)),
        plateau_patience=int(scheduler_raw.get('plateau_patience', 3)),
        plateau_threshold=float(scheduler_raw.get('plateau_threshold', 1e-4)),
        plateau_min_lr=float(scheduler_raw.get('plateau_min_lr', 1e-6)),
        plateau_cooldown=int(scheduler_raw.get('plateau_cooldown', 0)),
        cosine_t_max=int(scheduler_raw.get('cosine_t_max', 10)),
        cosine_eta_min=float(scheduler_raw.get('cosine_eta_min', 1e-6)),
        one_cycle_max_lr=float(scheduler_raw.get('one_cycle_max_lr', 1e-3)),
        one_cycle_pct_start=float(scheduler_raw.get('one_cycle_pct_start', 0.3)),
        one_cycle_anneal_strategy=str(scheduler_raw.get('one_cycle_anneal_strategy', 'cos')).strip().lower(),
        one_cycle_div_factor=float(scheduler_raw.get('one_cycle_div_factor', 25.0)),
        one_cycle_final_div_factor=float(scheduler_raw.get('one_cycle_final_div_factor', 10000.0)),
        one_cycle_three_phase=bool(scheduler_raw.get('one_cycle_three_phase', False)),
        step_lr_step_size=int(scheduler_raw.get('step_lr_step_size', 10)),
        step_lr_gamma=float(scheduler_raw.get('step_lr_gamma', 0.1)),
    )

    hard_mining = HardMiningParameters(
        enabled=bool(hard_mining_raw.get('enabled', False)),
        strength=float(hard_mining_raw.get('strength', 2.0)),
        ema_alpha=float(hard_mining_raw.get('ema_alpha', 0.2)),
        pixel_enabled=bool(hard_mining_raw.get('pixel_enabled', False)),
        pixel_keep_ratio=float(hard_mining_raw.get('pixel_keep_ratio', 0.25)),
    )
    legacy_use_multi_gpu = bool(raw.get('use_multi_gpu', True))
    multi_gpu_mode = normalize_multi_gpu_mode(
        raw.get('multi_gpu_mode', ''),
        use_multi_gpu_fallback=legacy_use_multi_gpu,
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
        scheduler=scheduler,
        hard_mining=hard_mining,
        cutout=CutoutParameters(
            enabled=bool(cutout_raw.get('enabled', False)),
            probability=float(cutout_raw.get('probability', 1.0)),
            holes=max(1, int(cutout_raw.get('holes', 1))),
            size_ratio=float(cutout_raw.get('size_ratio', 0.25)),
        ),
        random_artifacts=RandomArtifactsParameters(
            enabled=bool(random_artifacts_raw.get('enabled', False)),
            probability=float(random_artifacts_raw.get('probability', 1.0)),
            count=max(1, int(random_artifacts_raw.get('count', 1))),
            size_ratio=float(random_artifacts_raw.get('size_ratio', 0.25)),
        ),
        mixup=MixupParameters(
            enabled=bool(mixup_raw.get('enabled', False)),
            probability=float(mixup_raw.get('probability', 1.0)),
            alpha=float(mixup_raw.get('alpha', 0.2)),
        ),
        skip_uniform_labels=bool(raw.get('skip_uniform_labels', False)),
        use_multi_gpu=bool(multi_gpu_mode != 'off'),
        multi_gpu_mode=multi_gpu_mode,
        show_batch_preview=bool(raw.get('show_batch_preview', True)),
        log_update_frequency=int(raw.get('log_update_frequency', 0)),
        local_crop_size=local_crop_size,
        context_crop_size=(
            _to_tuple2(raw.get('context_crop_size'), 'tranining_parameters.context_crop_size')
            if raw.get('context_crop_size') is not None
            else None
        ),
        context_input_size=(
            _to_tuple2(raw.get('context_input_size'), 'tranining_parameters.context_input_size')
            if raw.get('context_input_size') is not None
            else None
        ),
        context_branch_channels=tuple(int(value) for value in raw.get('context_branch_channels', [16, 32, 64, 128])),
        fusion_type=str(raw.get('fusion_type', 'concat')),
        use_context_branch=bool(raw.get('use_context_branch', default_context_branch)),
        use_cross_attention=bool(raw.get('use_cross_attention', True)),
        attention_dim=int(raw.get('attention_dim', 128)),
        attention_heads=int(raw.get('attention_heads', 4)),
        attention_max_global_tokens=int(raw.get('attention_max_global_tokens', 1024)),
        deep_supervision=bool(raw.get('deep_supervision', True)),
        dataloader_num_workers=int(raw.get('dataloader_num_workers', -1)),
        recursive_file_search=bool(raw.get('recursive_file_search', False)),
        pcb_defects=build_pcb_defect_parameters(pcb_defects_raw),
    )


def _build_recognition_parameters(raw: dict[str, Any]) -> RecognitionParameters:
    source_files = [Path(p) for p in raw.get('source_files', [])]
    model_value = raw.get('model', raw.get('model_name', ''))
    model = Path(model_value) if isinstance(model_value, str) and model_value.lower().endswith('.pth') else model_value

    return RecognitionParameters(
        source_files=source_files,
        result_folder=Path(raw.get('result_folder', '')),
        model=model,
        part_size=_to_tuple2(
            raw.get('part_size', raw.get('local_crop_size', [256, 256])),
            'recogniton_parameters.part_size',
        ),
        batch_size=int(raw.get('batch_size', 16)),
        overlap=int(raw.get('overlap', 8)),
        jpeg_quality=int(raw.get('jpeg_quality', 95)),
        use_context_branch=(
            bool(raw.get('use_context_branch'))
            if 'use_context_branch' in raw
            else None
        ),
        use_cross_attention=(
            bool(raw.get('use_cross_attention'))
            if 'use_cross_attention' in raw
            else None
        ),
        context_crop_size=(
            _to_tuple2(raw.get('context_crop_size'), 'recogniton_parameters.context_crop_size')
            if raw.get('context_crop_size') is not None
            else None
        ),
        context_input_size=(
            _to_tuple2(raw.get('context_input_size'), 'recogniton_parameters.context_input_size')
            if raw.get('context_input_size') is not None
            else None
        ),
        recursive_file_search=bool(raw.get('recursive_file_search', False)),
        compression_factor=max(1, int(raw.get('compression_factor', 1) or 1)),
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
            'model_name': 'quasi_dual_scale_unet',
            'local_crop_size': [256, 256],
            'use_context_branch': True,
            'use_cross_attention': True,
            'context_crop_size': [512, 512],
            'context_input_size': [256, 256],
            'batch_size': 8,
            'overlap': 16,
            'jpeg_quality': 95,
        },
        'tranining_parameters': {
            'image_path': 'D:/data/train/images',
            'label_path': 'D:/data/train/labels',
            'shuffle': True,
            'validation': True,
            'validation_percent': 20,
            'batch_size': 8,
            'dataloader_num_workers': -1,
            'cut_mode': 'online',
            'colors': 3,
            'epochs': 30,
            'local_crop_size': [256, 256],
            'context_crop_size': [512, 512],
            'context_input_size': [256, 256],
            'context_branch_channels': [16, 32, 64, 128],
            'fusion_type': 'concat',
            'use_context_branch': True,
            'use_cross_attention': True,
            'attention_dim': 128,
            'attention_heads': 4,
            'attention_max_global_tokens': 1024,
            'deep_supervision': True,
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
                'random_crop': False,
                'crops_per_image': 64,
                'scale_augmentation': False,
                'scale_augmentation_strength': 0.2,
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
            'loss_function': 'focal_dice',
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
            'scheduler': {
                'name': 'off',
                'plateau_factor': 0.5,
                'plateau_patience': 3,
                'plateau_threshold': 0.0001,
                'plateau_min_lr': 0.000001,
                'plateau_cooldown': 0,
                'cosine_t_max': 10,
                'cosine_eta_min': 0.000001,
                'one_cycle_max_lr': 0.001,
                'one_cycle_pct_start': 0.3,
                'one_cycle_anneal_strategy': 'cos',
                'one_cycle_div_factor': 25.0,
                'one_cycle_final_div_factor': 10000.0,
                'one_cycle_three_phase': False,
                'step_lr_step_size': 10,
                'step_lr_gamma': 0.1,
            },
            'hard_mining': {
                'enabled': False,
                'strength': 2.0,
                'ema_alpha': 0.2,
                'pixel_enabled': False,
                'pixel_keep_ratio': 0.25,
            },
            'cutout': {
                'enabled': False,
                'probability': 1.0,
                'holes': 1,
                'size_ratio': 0.25,
            },
            'random_artifacts': {
                'enabled': False,
                'probability': 1.0,
                'count': 1,
                'size_ratio': 0.25,
            },
            'mixup': {
                'enabled': False,
                'probability': 1.0,
                'alpha': 0.2,
            },
            'pcb_defects': {
                'enabled': False,
                'defect_probability': 0.5,
                'min_defects': 1,
                'max_defects': 2,
                'max_attempts_per_defect': 8,
                'use_input_mask': True,
                'use_defect_mask_as_label': True,
                'defect_probabilities': {
                    'break': 1.0,
                    'short': 1.0,
                    'missing_copper': 1.0,
                    'excess_copper': 1.0,
                    'pinhole': 1.0,
                    'spurious_copper': 1.0,
                    'via': 1.0,
                    'misalignment': 1.0,
                },
            },
            'skip_uniform_labels': False,
            'use_multi_gpu': True,
            'multi_gpu_mode': 'distributeddataparallel',
            'show_batch_preview': True,
            'log_update_frequency': 50,
        },
    }


def _load_settings(path: Path, work_mode_override: str | None = None) -> tuple[WorkMode, TrainingParameters, RecognitionParameters]:
    payload = json.loads(path.read_text(encoding='utf-8'))

    work_mode_raw = work_mode_override if work_mode_override else payload.get('work_mode', 'train_and_recognition')
    work_mode = _to_work_mode(work_mode_raw)
    recognition_payload = payload.get('recogniton_parameters', {})
    model_name = str(recognition_payload.get('model', recognition_payload.get('model_name', ''))).strip() or None
    training_parameters = _build_training_parameters(payload.get('tranining_parameters', {}), model_name=model_name)
    recognition_parameters = _build_recognition_parameters(recognition_payload)
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
