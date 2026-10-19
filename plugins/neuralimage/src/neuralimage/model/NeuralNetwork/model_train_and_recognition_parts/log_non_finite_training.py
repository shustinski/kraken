"""model_train_and_recognition: log non finite training."""
from __future__ import annotations
from typing import TYPE_CHECKING
from typing import Any


class LogNonFiniteTrainingMixin:
    def _log_non_finite_training_state(self, *, stage: str, batch: _PreparedTrainBatch, outputs: Any = None) -> None:
        input_issue = self._find_first_non_finite_input_tensor(batch.inputs)
        if input_issue is not None:
            input_name, input_tensor = input_issue
            self._bus.put([
                'logging',
                f'Training debug [{stage}]: first non-finite input tensor "{input_name}": '
                f'{self._format_tensor_debug_stats(input_tensor)}',
            ])
        label_issue = self._find_first_non_finite_input_tensor(batch.label)
        if label_issue is not None:
            label_name, label_tensor = label_issue
            self._bus.put([
                'logging',
                f'Training debug [{stage}]: label tensor "{label_name}" is non-finite: '
                f'{self._format_tensor_debug_stats(label_tensor)}',
            ])
        if outputs is not None:
            primary_output = self._extract_primary_output_tensor(outputs)
            if not self._is_finite_tensor(primary_output):
                self._bus.put([
                    'logging',
                    f'Training debug [{stage}]: primary model output is non-finite: '
                    f'{self._format_tensor_debug_stats(primary_output)}',
                ])
            confidence_output = self._extract_confidence_output_tensor(outputs)
            if confidence_output is not None and not self._is_finite_tensor(confidence_output):
                self._bus.put([
                    'logging',
                    f'Training debug [{stage}]: confidence output is non-finite: '
                    f'{self._format_tensor_debug_stats(confidence_output)}',
                ])
        parameter_issue = self._find_first_non_finite_parameter()
        if parameter_issue is not None:
            param_name, param_tensor = parameter_issue
            self._bus.put([
                'logging',
                f'Training debug [{stage}]: first non-finite parameter "{param_name}": '
                f'{self._format_tensor_debug_stats(param_tensor)}',
            ])
        buffer_issue = self._find_first_non_finite_buffer()
        if buffer_issue is not None:
            buffer_name, buffer_tensor = buffer_issue
            self._bus.put([
                'logging',
                f'Training debug [{stage}]: first non-finite buffer "{buffer_name}": '
                f'{self._format_tensor_debug_stats(buffer_tensor)}',
            ])
        gradient_issue = self._find_first_non_finite_gradient()
        if gradient_issue is not None:
            grad_name, grad_tensor = gradient_issue
            self._bus.put([
                'logging',
                f'Training debug [{stage}]: first non-finite gradient "{grad_name}": '
                f'{self._format_tensor_debug_stats(grad_tensor)}',
            ])

if TYPE_CHECKING:
    from neuralimage.model.NeuralNetwork.model_train_and_recognition_parts.epoch_stats import _PreparedTrainBatch
