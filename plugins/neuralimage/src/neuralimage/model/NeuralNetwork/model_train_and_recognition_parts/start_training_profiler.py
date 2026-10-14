"""model_train_and_recognition: start training profiler."""
from __future__ import annotations
import datetime
from typing import Any
import torch

from neuralimage.model.NeuralNetwork.model_train_and_recognition_parts.epoch_stats import _ActiveTrainProfiler


class StartTrainingProfilerMixin:
    def _start_training_profiler(
        self,
        *,
        device: torch.device,
        train_size: int,
        is_main_process: bool,
        distributed: bool,
    ) -> _ActiveTrainProfiler | None:
        cfg = self._training_profiler_config
        if not cfg.enabled or not is_main_process:
            return None
        if distributed:
            self._bus.put(['logging', 'Профилирование отключено в режиме DDP (distributed).'])
            return None
        if train_size <= 0:
            return None
        if not hasattr(torch, 'profiler'):
            self._bus.put(['logging', 'torch.profiler недоступен, профилирование пропущено.'])
            return None

        steps = max(1, min(cfg.max_batches, int(train_size)))
        warmup_steps = 1 if steps > 1 else 0
        active_steps = max(1, steps - warmup_steps)
        wait_steps = 0
        profile_dir = self._save_path.parent / cfg.output_dir_name
        profile_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        trace_path = profile_dir / f'train_profile_{timestamp}.json'

        activities: list[Any] = [torch.profiler.ProfilerActivity.CPU]
        if device.type == 'cuda':
            activities.append(torch.profiler.ProfilerActivity.CUDA)

        def _on_trace_ready(prof: Any) -> None:
            try:
                prof.export_chrome_trace(str(trace_path))
            except Exception as export_error:
                self._bus.put(['logging', f'Не удалось сохранить trace профайлера: {export_error}'])

        profiler = torch.profiler.profile(
            activities=activities,
            schedule=torch.profiler.schedule(wait=wait_steps, warmup=warmup_steps, active=active_steps, repeat=1),
            on_trace_ready=_on_trace_ready,
            record_shapes=cfg.record_shapes,
            profile_memory=cfg.profile_memory,
            with_stack=cfg.with_stack,
        )
        profiler.__enter__()
        sort_key = 'self_cuda_time_total' if device.type == 'cuda' else 'self_cpu_time_total'
        self._bus.put([
            'logging',
            (
                f'Профилирование обучения включено: шагов={steps}, warmup={warmup_steps}, active={active_steps}, '
                f'trace={trace_path.name}'
            ),
        ])
        if self._random_patch_profiling_active():
            self._bus.put([
                'logging',
                (
                    'Random-patch diagnostic profiling enabled in code. '
                    f'Outputs: {profile_dir / "random_patch_batches.csv"}, '
                    f'{profile_dir / "random_patch_shapes.csv"}, {trace_path}.'
                ),
            ])
        return _ActiveTrainProfiler(
            profiler=profiler,
            steps_left=steps,
            trace_path=trace_path,
            summary_sort_key=sort_key,
            row_limit=cfg.row_limit,
        )
