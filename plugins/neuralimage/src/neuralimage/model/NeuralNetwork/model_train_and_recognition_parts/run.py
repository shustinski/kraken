"""model_train_and_recognition: run."""
from __future__ import annotations
import time
from typing import Any

from neuralimage.model.NeuralNetwork.model_train_and_recognition_parts.combined_optimizer import (
    MuonOptimizerUnavailableError,
    TrainingInstabilityError,
    _join_process_with_escalation,
)

from neuralimage.model.NeuralNetwork.model_train_and_recognition_parts.recognizer_process import _is_debugger_attached

from neuralimage.model.NeuralNetwork.model_train_and_recognition_parts.validation_no_cut_frame import _release_torch_memory


class RunMixin:
    def run(self):
        from neuralimage.model.NeuralNetwork.model_train_and_recognition import _is_debugger_attached

        training_process: Any = None
        try:
            if not self._validate_training_inputs():
                return

            if _is_debugger_attached():
                self._bus.publish(
                    'logging',
                    'Debugger detected: training multiprocessing disabled for this run.',
                )
                training_process = self._create_training_process()
                started_at = time.perf_counter()
                try:
                    training_process.run()
                except MuonOptimizerUnavailableError as error:
                    self.error_message = str(error)
                    self._bus.publish('error', self.error_message)
                    self.succeeded = False
                    return
                except TrainingInstabilityError as error:
                    self.error_message = str(error)
                    self._bus.publish('error', self.error_message)
                    self.succeeded = False
                    return
                except Exception as error:
                    self.error_message = f'Training error: {error}'
                    self._bus.publish('error', self.error_message)
                    self.succeeded = False
                    return
                self._drain_training_queue(append_elapsed_suffix=False, started_at=started_at)
                self.succeeded = True
                if self.callback is not None:
                    self.callback()
                return

            training_process = self._create_training_process()
            training_process.start()

            started_at = time.perf_counter()
            while training_process.is_alive():
                if self._stop_event.is_set():
                    _join_process_with_escalation(training_process)
                    break
                self._drain_training_queue(
                    append_elapsed_suffix=True,
                    started_at=started_at,
                )
                time.sleep(1)

            _join_process_with_escalation(training_process)
            self._drain_training_queue(append_elapsed_suffix=False, started_at=started_at)
            self._finalize_training_result(training_process)
        finally:
            if training_process is not None and training_process.is_alive():
                _join_process_with_escalation(training_process)
            try:
                self.message_queue.close()
            except Exception:
                pass
            # Drop heavyweight references from parent thread after process completion.
            self._model = None
            self._train_dataloader = None
            self._val_dataloader = None
            self._control_dataloader = None
            _release_torch_memory()
