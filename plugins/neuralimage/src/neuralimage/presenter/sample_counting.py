from __future__ import annotations

import os
from pathlib import Path

from PyQt6 import QtCore
from PyQt6.QtCore import QObject

from neuralimage.lib.data_interfaces import CutSettings, SampleCutMode, build_synthetic_defect_generator_parameters


class SampleCountSignals(QObject):
    calculated = QtCore.pyqtSignal(int, str, object, int)
    failed = QtCore.pyqtSignal(int, str)


def calculate_expected_samples(presenter) -> None:
    presenter._update_settings_window_state()
    calculator_settings = presenter.get_cut_settings_from_window_state()
    presenter.sample_calculator.set_settings(calculator_settings)
    presenter._set_sample_number(calculator_settings)


def get_cut_settings_from_window_state(presenter) -> CutSettings:
    state = presenter.settings_state
    train_patch_size = tuple(getattr(state, 'train_patch_size', None) or state.sample_size)
    online_mode = getattr(state, 'sample_cut_mode', SampleCutMode.online.value) == SampleCutMode.online.value
    return CutSettings(
        step=state.step,
        x_size=train_patch_size[0],
        y_size=train_patch_size[1],
        vertical_rotation=state.vertical_rotation,
        horizontal_rotation=state.horizontal_rotation,
        flip_x=bool(getattr(state, 'flip_x', False)),
        flip_y=bool(getattr(state, 'flip_y', False)),
        color_mode=state.color_mode,
        model=state.model,
        additional_augmentation=state.additional_augmentation,
        augmentation_gamma_strength=float(getattr(state, 'augmentation_gamma_strength', 0.15)),
        augmentation_blur_probability=float(getattr(state, 'augmentation_blur_probability', 0.25)),
        augmentation_blur_radius=float(getattr(state, 'augmentation_blur_radius', 1.0)),
        random_crop=bool(getattr(state, 'random_crop', False) and online_mode),
        crops_per_image=int(getattr(state, 'crops_per_image', 64)),
        scale_augmentation=bool(getattr(state, 'scale_augmentation', False) and online_mode),
        scale_augmentation_strength=float(getattr(state, 'scale_augmentation_strength', 0.2)),
        recursive_file_search=bool(getattr(state, 'recursive_file_search', False)),
    )


def set_sample_number(presenter, calculator_settings: CutSettings) -> None:
    sample_folder = str(getattr(presenter.main_window_state, 'sample_folder', '') or '').strip()
    if not sample_folder or not Path(sample_folder).is_dir():
        presenter._invalidate_sample_count_requests()
        presenter.settings_panel.set_samples_count(0)
        if hasattr(presenter.view, 'set_samples_count'):
            presenter.view.set_samples_count(0)
        return

    presenter._sample_count_request_serial += 1
    request_id = presenter._sample_count_request_serial
    presenter._latest_sample_count_request_id = request_id
    presenter._debounced_sample_count_request = (
        request_id,
        sample_folder,
        calculator_settings,
        getattr(presenter.settings_state, 'synthetic_defect_generator', None),
    )
    presenter.settings_panel.set_samples_count_loading()
    if hasattr(presenter.view, 'set_samples_count_loading'):
        presenter.view.set_samples_count_loading()
    presenter._sample_count_debounce_timer.start()


def invalidate_sample_count_requests(presenter) -> None:
    presenter._sample_count_request_serial += 1
    presenter._latest_sample_count_request_id = presenter._sample_count_request_serial
    presenter._debounced_sample_count_request = None
    presenter._pending_sample_count_request = None
    presenter._sample_count_debounce_timer.stop()


def dispatch_sample_count_request(presenter) -> None:
    request = presenter._debounced_sample_count_request
    presenter._debounced_sample_count_request = None
    if request is None:
        return
    if presenter._sample_count_worker_thread is not None and presenter._sample_count_worker_thread.is_alive():
        presenter._pending_sample_count_request = request
        return
    presenter._start_sample_count_request(request)


def start_sample_count_request(
    presenter,
    request: tuple[int, str, CutSettings, object],
    *,
    sample_worker_cls,
    threading_module,
) -> None:
    request_id, sample_folder, calculator_settings, synthetic_config = request
    cached_path = presenter._sample_count_cache_path
    cached_sizes = (
        list(presenter._sample_count_cache_sizes)
        if presenter._sample_count_cache_sizes is not None
        else None
    )
    normalized_path = _sample_count_cache_key(
        sample_folder,
        recursive=bool(getattr(calculator_settings, 'recursive_file_search', False)),
    )
    if normalized_path:
        if normalized_path != cached_path or cached_sizes is None:
            presenter._publish_log_message(
                f'Индексация файлов выборки запущена в отдельном потоке: {sample_folder}'
            )
        else:
            presenter._publish_log_message(
                f'Пересчет количества кадров запущен в отдельном потоке: {sample_folder}'
            )

    worker_thread = threading_module.Thread(
        target=run_sample_count_request,
        args=(
            presenter,
            request_id,
            sample_folder,
            calculator_settings,
            synthetic_config,
            cached_path,
            cached_sizes,
            sample_worker_cls,
        ),
        daemon=True,
        name=f'sample-count-{request_id}',
    )
    presenter._sample_count_worker_thread = worker_thread
    worker_thread.start()


def run_sample_count_request(
    presenter,
    request_id: int,
    sample_folder: str,
    calculator_settings: CutSettings,
    synthetic_config: object,
    cached_path: str | None,
    cached_sizes: list[tuple[int, int]] | None,
    sample_worker_cls,
) -> None:
    try:
        sample_path = Path(sample_folder)
        normalized_path = _sample_count_cache_key(
            sample_path,
            recursive=bool(getattr(calculator_settings, 'recursive_file_search', False)),
        )
        if not sample_path.is_dir():
            presenter._sample_count_signals.calculated.emit(request_id, normalized_path, [], 0)
            return

        if normalized_path == cached_path and cached_sizes is not None:
            image_sizes = list(cached_sizes)
        else:
            presenter._publish_log_message(
                f'Выполняется индексация списка файлов выборки: {sample_folder}'
            )
            recursive = bool(getattr(calculator_settings, 'recursive_file_search', False))
            try:
                image_paths = sample_worker_cls.collect_image_paths(sample_path, recursive=recursive)
            except TypeError:
                image_paths = sample_worker_cls.collect_image_paths(sample_path)
            image_sizes = sample_worker_cls.collect_image_sizes(image_paths)

        total_samples = sample_worker_cls.calculate_total_samples(image_sizes, calculator_settings)
        synthetic_generator = build_synthetic_defect_generator_parameters(synthetic_config)
        if synthetic_generator.enabled and float(synthetic_generator.epoch_size_factor) > 0.0 and image_sizes:
            synthetic_frame_count = max(
                1,
                int(round(len(image_sizes) * float(synthetic_generator.epoch_size_factor))),
            )
            synthetic_size_xy = (
                max(int(calculator_settings.x_size), int(synthetic_generator.image_size_xy[0])),
                max(int(calculator_settings.y_size), int(synthetic_generator.image_size_xy[1])),
            )
            total_samples += (
                synthetic_frame_count
                * sample_worker_cls.calculate_image_parts_for_settings(
                    (int(synthetic_size_xy[1]), int(synthetic_size_xy[0])),
                    calculator_settings,
                )
            )
        presenter._sample_count_signals.calculated.emit(
            request_id,
            normalized_path,
            image_sizes,
            total_samples,
        )
    except Exception as exc:
        presenter._sample_count_signals.failed.emit(request_id, str(exc))


def normalize_sample_count_path(path: Path | str) -> str:
    return os.path.normcase(os.path.abspath(str(path)))


def _sample_count_cache_key(path: Path | str, *, recursive: bool) -> str:
    return f'{normalize_sample_count_path(path)}|recursive={int(bool(recursive))}'


def on_sample_count_calculated(
    presenter,
    request_id: int,
    normalized_path: str,
    image_sizes: object,
    total_samples: int,
) -> None:
    presenter._sample_count_worker_thread = None
    if normalized_path:
        presenter._sample_count_cache_path = normalized_path
        presenter._sample_count_cache_sizes = list(image_sizes) if isinstance(image_sizes, list) else []
    if request_id == presenter._latest_sample_count_request_id:
        presenter.settings_panel.set_samples_count(total_samples)
        if hasattr(presenter.view, 'set_samples_count'):
            presenter.view.set_samples_count(total_samples)
        presenter._publish_log_message(f'Количество кадров в выборке пересчитано: {total_samples}')
    presenter._start_pending_sample_count_request_if_needed()


def on_sample_count_failed(presenter, request_id: int, error_message: str) -> None:
    presenter._sample_count_worker_thread = None
    if request_id == presenter._latest_sample_count_request_id:
        presenter.settings_panel.set_samples_count(0)
        if hasattr(presenter.view, 'set_samples_count'):
            presenter.view.set_samples_count(0)
        presenter.message_bus.publish('logging', f'Не удалось рассчитать количество кадров: {error_message}')
    presenter._start_pending_sample_count_request_if_needed()


def start_pending_sample_count_request_if_needed(presenter) -> None:
    if presenter._debounced_sample_count_request is not None:
        return
    request = presenter._pending_sample_count_request
    if request is None:
        return
    presenter._pending_sample_count_request = None
    presenter._start_sample_count_request(request)


def set_max_shift(presenter) -> None:
    panel = presenter.settings_panel
    x_size = panel.sample_x_size.value()
    y_size = panel.sample_y_size.value()
    min_size = min(x_size, y_size)
    panel.shift_spinbox.setMaximum(min_size)
    if panel.shift_spinbox.value() > min_size:
        panel.shift_spinbox.setValue(min_size)
