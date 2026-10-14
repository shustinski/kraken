"""general_neural_handler: resolve model creation kwargs."""
from __future__ import annotations
from typing import Any
from neuralimage.lib.data_interfaces import (
    SampleCutMode,
)
from neuralimage.model.NeuralNetwork import model_supports_init_kwarg
from neuralimage.model.NeuralNetwork.context_utils import normalize_size_pair


class ResolveModelCreationKwargsMixin:
    def _resolve_model_creation_kwargs(self, model_name: str) -> dict[str, Any]:
        resolved_name = str(model_name)
        model_kwargs: dict[str, Any] = {}
        if resolved_name == 'Transformer':
            patch_size = self.tranining_parameters.generation.segment_size
            img_size = max(1, int(patch_size[0]))
            if int(patch_size[0]) != int(patch_size[1]):
                self.message_bus.publish(
                    'logging',
                    (
                        'Transformer expects square inputs. '
                        f'Using img_size={img_size} derived from patch size {tuple(patch_size)}.'
                    ),
                )
            model_kwargs['img_size'] = img_size
            return model_kwargs

        if resolved_name in {'quasi_dual_scale_unet', 'FrameUnet', 'UNetWithContextBranch'}:
            requested_context_branch = getattr(self.tranining_parameters, 'use_context_branch', None)
            if requested_context_branch is None:
                requested_context_branch = True
            if bool(requested_context_branch) and self.tranining_parameters.cut_mode != SampleCutMode.online:
                raise ValueError(
                    'Context branch requires online patch generation (cut_mode=online) because '
                    'the dataset must build context crops from the full prepared frame.'
                )

            local_crop_size = normalize_size_pair(
                getattr(self.tranining_parameters, 'local_crop_size', None),
                fallback=tuple(self.tranining_parameters.generation.segment_size),
            )
            context_crop_size = normalize_size_pair(
                getattr(self.tranining_parameters, 'context_crop_size', None),
                fallback=(local_crop_size[0] * 2, local_crop_size[1] * 2),
            )
            context_input_size = normalize_size_pair(
                getattr(self.tranining_parameters, 'context_input_size', None),
                fallback=local_crop_size,
            )
            context_branch_channels = tuple(
                int(channel)
                for channel in getattr(self.tranining_parameters, 'context_branch_channels', (16, 32, 64, 128))
            )
            model_kwargs.update(
                {
                    'local_crop_size': local_crop_size,
                    'context_crop_size': context_crop_size,
                    'context_input_size': context_input_size,
                    'context_branch_channels': context_branch_channels,
                    'fusion_type': str(getattr(self.tranining_parameters, 'fusion_type', 'concat')),
                    'use_context_branch': bool(requested_context_branch),
                    'use_cross_attention': bool(getattr(self.tranining_parameters, 'use_cross_attention', True)),
                    'attention_dim': int(getattr(self.tranining_parameters, 'attention_dim', 128)),
                    'attention_heads': int(getattr(self.tranining_parameters, 'attention_heads', 4)),
                    'attention_max_global_tokens': int(
                        getattr(self.tranining_parameters, 'attention_max_global_tokens', 1024)
                    ),
                }
            )

        if model_supports_init_kwarg(resolved_name, 'deep_supervision'):
            model_kwargs['deep_supervision'] = bool(getattr(self.tranining_parameters, 'deep_supervision', True))

        supervision_targets = getattr(self.tranining_parameters, 'supervision_targets', None)
        if supervision_targets is not None and getattr(supervision_targets, 'any_enabled', lambda: False)():
            enabled_heads = tuple(supervision_targets.enabled_targets())
            if not model_supports_init_kwarg(resolved_name, 'supervision_heads'):
                raise ValueError(
                    f'Model {resolved_name!r} does not support auxiliary supervision heads: '
                    f'{", ".join(enabled_heads)}.'
                )
            model_kwargs['supervision_heads'] = enabled_heads

        return model_kwargs
