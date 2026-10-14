"""model_train_and_recognition: resolve context branch settings."""
from __future__ import annotations
import torch.nn as nn
from neuralimage.model.NeuralNetwork.context_utils import normalize_size_pair


class ResolveContextBranchSettingsMixin:
    def _resolve_context_branch_settings(self, model: nn.Module) -> None:
        model_kwargs = getattr(model, '_neuralimage_model_kwargs', {})
        if not isinstance(model_kwargs, dict):
            model_kwargs = {}

        use_context_override = getattr(self._parameters, 'use_context_branch', None)
        requested_context_branch = (
            bool(model_kwargs.get('use_context_branch', False))
            if use_context_override is None
            else bool(use_context_override)
        )
        if not requested_context_branch:
            self._use_context_branch = False
            self._use_cross_attention = False
            self._context_crop_size = None
            self._context_input_size = None
            return

        use_cross_override = getattr(self._parameters, 'use_cross_attention', None)
        self._use_cross_attention = (
            bool(model_kwargs.get('use_cross_attention', True))
            if use_cross_override is None
            else bool(use_cross_override)
        )

        fallback_local = tuple(getattr(self._parameters, 'part_size', (256, 256)))
        context_crop_raw = getattr(self._parameters, 'context_crop_size', None)
        if context_crop_raw is None:
            context_crop_raw = model_kwargs.get('context_crop_size')
        context_input_raw = getattr(self._parameters, 'context_input_size', None)
        if context_input_raw is None:
            context_input_raw = model_kwargs.get('context_input_size')

        if context_crop_raw is None or context_input_raw is None:
            self._use_context_branch = False
            self._use_cross_attention = False
            self._context_crop_size = None
            self._context_input_size = None
            self._bus.publish(
                'logging',
                'Recognition context branch disabled: model/context sizes are missing.',
            )
            return

        self._use_context_branch = True
        self._context_crop_size = normalize_size_pair(
            context_crop_raw,
            fallback=(int(fallback_local[0]) * 2, int(fallback_local[1]) * 2),
        )
        self._context_input_size = normalize_size_pair(
            context_input_raw,
            fallback=(int(fallback_local[0]), int(fallback_local[1])),
        )
        self._bus.publish(
            'logging',
            (
                'Recognition context branch enabled: '
                f'crop={self._context_crop_size}, input={self._context_input_size}, '
                f'cross_attention={bool(self._use_cross_attention)}.'
            ),
        )
