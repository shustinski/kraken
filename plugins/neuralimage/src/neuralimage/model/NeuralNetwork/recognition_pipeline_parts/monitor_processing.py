"""recognition_pipeline: monitor processing."""
from __future__ import annotations
import multiprocessing as mp
from typing import Iterable

from neuralimage.model.NeuralNetwork.recognition_pipeline_parts.module_state import _QUEUE_EMPTY

from neuralimage.model.NeuralNetwork.recognition_pipeline_parts.load_preview_array import _publish_recognition_preview

from neuralimage.model.NeuralNetwork.recognition_pipeline_parts.recognition_workload import _try_get_queue_item

from neuralimage.model.NeuralNetwork.recognition_pipeline_parts.run_cut_worker import _try_get_worker_error_message


class MonitorProcessingMixin:
    def _monitor_processing(
        self,
        *,
        groups: WorkerGroups,
        queues: PipelineQueues,
    ) -> tuple[bool, bool, int, str | None]:
        completed = 0
        predict_stopped = False
        sew_stopped = False
        failure_message: str | None = None
        while completed < self._config.workload.frame_count:
            if self._stop_event.is_set():
                break

            if not predict_stopped and self._all_stopped(groups.cut):
                self._signal_predict_stop(queues.predict)
                predict_stopped = True

            if predict_stopped and (not sew_stopped) and self._all_stopped(groups.predict):
                self._signal_sew_stop(queues.sew)
                sew_stopped = True

            completed = self._consume_completed_frame(queues=queues, completed=completed)
            failed_process_message = self._get_failed_process_message(groups)
            if failed_process_message is not None:
                failure_message = failed_process_message
                self._stop_event.set()
                break

        return predict_stopped, sew_stopped, completed, failure_message

    @staticmethod
    def _all_stopped(processes: Iterable[mp.Process]) -> bool:
        return all(not process.is_alive() for process in processes)

    def _signal_predict_stop(self, predict_queue: mp.Queue) -> None:
        for _ in range(self._config.worker_counts.predict):
            predict_queue.put(self._config.stop_token)

    def _signal_sew_stop(self, sew_queue: mp.Queue) -> None:
        for _ in range(self._config.worker_counts.sew):
            sew_queue.put(self._config.stop_token)

    def _consume_completed_frame(self, *, queues: PipelineQueues, completed: int) -> int:
        completed_item = _try_get_queue_item(queues.sewed, timeout=0.2)
        if completed_item is _QUEUE_EMPTY:
            return completed
        if isinstance(completed_item, dict):
            _publish_recognition_preview(
                publish=self._publish,
                source_path=completed_item.get('source_path'),
                output_path=completed_item.get('output_path'),
                frame_name=str(completed_item.get('name', '')).strip(),
            )
        updated = completed + 1
        self._reporter.publish_frame(updated)
        if isinstance(completed_item, dict):
            self._publish('metrics', {
                'type': 'recognition_completed',
                'source_path': str(completed_item.get('source_path') or ''),
                'output_path': str(completed_item.get('output_path') or ''),
            })
        return updated

    def _get_failed_process_message(self, groups: WorkerGroups) -> str | None:
        detailed_error = _try_get_worker_error_message(getattr(self, '_errors_queue_ref', None))
        if detailed_error is not None:
            self._publish('logging', detailed_error)
            return detailed_error
        for process in groups.all():
            if process.exitcode in (None, 0):
                continue
            message = f'Child process failed. pid={process.pid}, code={process.exitcode}.'
            self._publish('logging', message)
            return message
        return None

    def _send_missing_stop_tokens(
        self,
        *,
        queues: PipelineQueues,
        predict_stopped: bool,
        sew_stopped: bool,
    ) -> None:
        if not predict_stopped:
            self._signal_predict_stop(queues.predict)
        if not sew_stopped:
            self._signal_sew_stop(queues.sew)

    @staticmethod
    def _shutdown_workers(groups: WorkerGroups) -> None:
        for process in groups.all():
            process.join(timeout=5)
        for process in groups.all():
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
            if process.is_alive():
                process.kill()
                process.join(timeout=5)
