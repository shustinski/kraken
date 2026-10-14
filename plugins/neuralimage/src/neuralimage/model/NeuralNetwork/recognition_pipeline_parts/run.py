"""recognition_pipeline: run."""
from __future__ import annotations
import multiprocessing as mp
import torch
from multiprocessing.synchronize import Event as MpEvent

from neuralimage.model.NeuralNetwork.recognition_pipeline_parts.recognition_workload import (
    PipelineQueues,
    ProgressReporter,
    WorkerGroups,
)

from neuralimage.model.NeuralNetwork.recognition_pipeline_parts.run_cut_worker import (
    _run_cut_worker,
    _run_predict_worker,
    _run_sew_worker,
)


class RunMixin:
    def __init__(
        self,
        *,
        config: RuntimeWorkerConfig,
        stop_event: MpEvent,
        publish: Publisher,
    ) -> None:
        self._config = config
        self._stop_event = stop_event
        self._publish = publish
        self._reporter = ProgressReporter(publish=publish, total_frames=config.workload.frame_count)

    def run(self) -> None:
        queues = self._create_queues()
        self._errors_queue_ref = getattr(queues, 'errors', None)
        groups = WorkerGroups(cut=[], predict=[], sew=[])
        predict_stopped = False
        sew_stopped = False
        completed = 0
        failure_message: str | None = None
        try:
            self._prime_cut_queue(queues)
            self._publish_runtime_plan()
            groups = self._start_workers(queues)
            predict_stopped, sew_stopped, completed, failure_message = self._monitor_processing(
                groups=groups,
                queues=queues,
            )
        finally:
            self._send_missing_stop_tokens(
                queues=queues,
                predict_stopped=predict_stopped,
                sew_stopped=sew_stopped,
            )
            self._shutdown_workers(groups)
            queues.close()
        if failure_message is not None:
            raise RuntimeError(failure_message)
        if (not self._stop_event.is_set()) and completed < self._config.workload.frame_count:
            raise RuntimeError(
                'Recognition pipeline stopped before all frames were processed '
                f'({completed}/{self._config.workload.frame_count}).'
            )

    def _create_queues(self) -> PipelineQueues:
        worker_total = max(
            4,
            int(self._config.worker_counts.cut + self._config.worker_counts.predict + self._config.worker_counts.sew),
        )
        predict_queue_size = max(8, worker_total * 2, len(self._config.workload.devices) * 6)
        return PipelineQueues(
            cut=mp.Queue(),
            predict=mp.Queue(maxsize=predict_queue_size),
            sew=mp.Queue(maxsize=predict_queue_size),
            sewed=mp.Queue(),
            errors=mp.Queue(),
        )

    def _prime_cut_queue(self, queues: PipelineQueues) -> None:
        for image in self._config.workload.source_files:
            queues.cut.put(image)
        for _ in range(self._config.worker_counts.cut):
            queues.cut.put(self._config.stop_token)

    def _publish_runtime_plan(self) -> None:
        self._reporter.publish_started()
        self._reporter.publish_worker_plan(self._config.worker_counts)

    def _start_workers(self, queues: PipelineQueues) -> WorkerGroups:
        cut_workers = self._start_cut_workers(queues)
        predict_workers = self._start_predict_workers(queues)
        sew_workers = self._start_sew_workers(queues)
        return WorkerGroups(cut=cut_workers, predict=predict_workers, sew=sew_workers)

    def _start_cut_workers(self, queues: PipelineQueues) -> list[mp.Process]:
        self._publish('logging', 'Starting cut workers.')
        processes: list[mp.Process] = []
        for _ in range(self._config.worker_counts.cut):
            process = mp.Process(
                target=_run_cut_worker,
                args=(
                    queues.errors,
                    queues.cut,
                    queues.predict,
                    self._config.workload.segment_shape,
                    self._config.workload.overlap,
                    self._stop_event,
                    self._config.workload.use_context_branch,
                    self._config.workload.use_cross_attention,
                    self._config.workload.context_crop_size,
                    self._config.workload.context_input_size,
                    self._config.workload.compression_factor,
                    self._config.workload.source_root,
                    self._config.workload.preprocessing,
                    self._config.stop_token,
                ),
            )
            process.start()
            processes.append(process)
        return processes

    def _start_predict_workers(self, queues: PipelineQueues) -> list[mp.Process]:
        self._publish('logging', 'Starting predict workers.')
        processes: list[mp.Process] = []
        for worker_index in range(self._config.worker_counts.predict):
            device = self._resolve_worker_device(worker_index)
            process = mp.Process(
                target=_run_predict_worker,
                args=(
                    queues.errors,
                    queues.predict,
                    queues.sew,
                    self._config.workload.model_source,
                    device,
                    self._config.workload.batch_size,
                    self._stop_event,
                    self._config.workload.recognition_tta_enabled,
                    self._config.workload.confidence_tta_enabled,
                    self._config.workload.uncertainty,
                    self._config.stop_token,
                ),
            )
            process.start()
            processes.append(process)
        return processes

    def _start_sew_workers(self, queues: PipelineQueues) -> list[mp.Process]:
        self._publish('logging', 'Starting sew workers.')
        processes: list[mp.Process] = []
        for _ in range(self._config.worker_counts.sew):
            process = mp.Process(
                target=_run_sew_worker,
                args=(
                    queues.errors,
                    self._config.workload.result_folder,
                    queues.sew,
                    queues.sewed,
                    self._config.workload.jpeg_quality,
                    self._stop_event,
                    self._config.workload.threshold,
                    (
                        self._config.workload.postprocess_kernel_size
                        if self._config.workload.threshold is not None and self._config.workload.postprocess_enabled
                        else 0
                    ),
                    self._config.workload.confidence_save_mode,
                    self._config.workload.active_learning,
                    self._config.workload.active_learning_metadata,
                    self._config.stop_token,
                ),
            )
            process.start()
            processes.append(process)
        return processes

    def _resolve_worker_device(self, worker_index: int) -> torch.device:
        devices = self._config.workload.devices
        if not devices:
            return torch.device('cpu')
        device_index = min(worker_index, len(devices) - 1)
        return devices[device_index]
