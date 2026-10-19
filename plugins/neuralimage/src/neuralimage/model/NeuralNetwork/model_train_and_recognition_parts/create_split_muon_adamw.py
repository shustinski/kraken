"""model_train_and_recognition: create split muon adamw."""
from __future__ import annotations
from typing import Any, Iterator
import torch
import torch.nn as nn
from torch import optim
from neuralimage.lib.data_interfaces import (
    OptimizerParameters,
    OptimizerName,
    MixedPrecisionMode,
)
from neuralimage.lib.optimizer_availability import MUON_UNAVAILABLE_MESSAGE, resolve_muon_optimizer_class

from neuralimage.model.NeuralNetwork.model_train_and_recognition_parts.combined_optimizer import (
    MuonOptimizerUnavailableError,
    _CombinedOptimizer,
)


class CreateSplitMuonAdamwMixin:
    def _create_optimizer(self):
        params = self._optimizer_params
        if params.name == OptimizerName.adam:
            adam_kwargs: dict[str, Any] = {'lr': params.learning_rate, 'weight_decay': params.weight_decay}
            if self._can_use_fused_optim():
                adam_kwargs['fused'] = True
            try:
                return optim.Adam(self._all_trainable_parameters(), **adam_kwargs)
            except TypeError:
                adam_kwargs.pop('fused', None)
                return optim.Adam(self._all_trainable_parameters(), **adam_kwargs)
        if params.name == OptimizerName.adamw:
            return self._create_adamw_optimizer(params)
        if params.name == OptimizerName.adamw_muon:
            return self._create_adamw_muon_optimizer(params)
        raise ValueError(f'Unsupported optimizer: {params.name}')

    def _all_trainable_parameters(self) -> Iterator[torch.nn.Parameter]:
        yield from self._model.parameters()
        if self._loss_weighter is not None:
            yield from self._loss_weighter.parameters()

    def _can_use_fused_optim(self) -> bool:
        if not torch.cuda.is_available():
            return False
        try:
            return any(p.is_cuda for p in self._model.parameters())
        except Exception:
            return False

    def _resolve_mixed_precision(self, device: torch.device) -> tuple[MixedPrecisionMode, torch.dtype | None, bool]:
        if device.type != 'cuda':
            return MixedPrecisionMode.off, None, False

        mode = self._mixed_precision
        if mode == MixedPrecisionMode.off:
            return mode, None, False
        if mode == MixedPrecisionMode.fp16:
            return mode, torch.float16, True
        if mode == MixedPrecisionMode.bf16:
            is_bf16_supported = False
            try:
                is_bf16_supported = bool(torch.cuda.is_bf16_supported())
            except Exception:
                is_bf16_supported = False
            if is_bf16_supported:
                return mode, torch.bfloat16, False
            return MixedPrecisionMode.fp16, torch.float16, True
        return MixedPrecisionMode.off, None, False

    def _build_adamw_param_groups(self, params: OptimizerParameters) -> list[dict[str, Any]]:
        return self._build_adamw_param_groups_for(params, self._all_trainable_parameters())

    def _build_adamw_param_groups_for(
        self,
        params: OptimizerParameters,
        model_params: Iterator[torch.nn.Parameter] | list[torch.nn.Parameter],
    ) -> list[dict[str, Any]]:
        norm_param_ids: set[int] = set()
        batch_norm_base = getattr(nn.modules.batchnorm, '_BatchNorm', ())
        norm_types = (nn.GroupNorm, batch_norm_base) if batch_norm_base else (nn.GroupNorm,)

        for module in self._base_model.modules():
            if isinstance(module, norm_types):
                for p in module.parameters(recurse=False):
                    if p.requires_grad:
                        norm_param_ids.add(id(p))
        if self._loss_weighter is not None:
            norm_param_ids.update(id(parameter) for parameter in self._loss_weighter.parameters())

        decay_params: list[torch.nn.Parameter] = []
        no_decay_params: list[torch.nn.Parameter] = []
        for param in model_params:
            if not param.requires_grad:
                continue
            if id(param) in norm_param_ids:
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        param_groups: list[dict[str, Any]] = []
        if decay_params:
            param_groups.append({'params': decay_params, 'weight_decay': params.weight_decay})
        if no_decay_params:
            param_groups.append({'params': no_decay_params, 'weight_decay': 0.0})
        return param_groups

    def _create_adamw_optimizer(self, params: OptimizerParameters):
        use_fused = self._can_use_fused_optim()
        param_groups = self._build_adamw_param_groups(params)
        if param_groups:
            self._bus.put([
                'logging',
                f'AdamW: исключены norm-параметры из weight decay (групп: {len(param_groups)}).',
            ])
            adamw_kwargs: dict[str, Any] = {'lr': params.learning_rate}
            if use_fused:
                adamw_kwargs['fused'] = True
            try:
                return optim.AdamW(param_groups, **adamw_kwargs)
            except TypeError:
                adamw_kwargs.pop('fused', None)
                return optim.AdamW(param_groups, **adamw_kwargs)
        adamw_kwargs = {'lr': params.learning_rate, 'weight_decay': params.weight_decay}
        if use_fused:
            adamw_kwargs['fused'] = True
        try:
            return optim.AdamW(self._all_trainable_parameters(), **adamw_kwargs)
        except TypeError:
            adamw_kwargs.pop('fused', None)
            return optim.AdamW(self._all_trainable_parameters(), **adamw_kwargs)

    def _create_adamw_muon_optimizer(self, params: OptimizerParameters):
        from neuralimage.model.NeuralNetwork.model_train_and_recognition import resolve_muon_optimizer_class

        muon_cls = resolve_muon_optimizer_class()
        if muon_cls is None:
            raise MuonOptimizerUnavailableError(MUON_UNAVAILABLE_MESSAGE)

        try:
            return self._create_split_muon_adamw_optimizer(params, muon_cls)
        except Exception as error:
            self._bus.put([
                'logging',
                f'Muon optimizer initialization failed ({error}). Using AdamW.',
            ])
            return self._create_adamw_optimizer(params)

    def _create_split_muon_adamw_optimizer(
        self,
        params: OptimizerParameters,
        muon_cls: type[optim.Optimizer],
    ) -> optim.Optimizer:
        muon_params: list[torch.nn.Parameter] = []
        adamw_params: list[torch.nn.Parameter] = []
        for param in self._all_trainable_parameters():
            if not param.requires_grad:
                continue
            if param.ndim == 2:
                muon_params.append(param)
            else:
                adamw_params.append(param)

        optimizers: list[optim.Optimizer] = []
        if muon_params:
            optimizers.append(
                muon_cls(
                    muon_params,
                    lr=params.learning_rate,
                    weight_decay=params.weight_decay,
                )
            )
        if adamw_params:
            adamw_groups = self._build_adamw_param_groups_for(params, adamw_params)
            adamw_kwargs: dict[str, Any] = {'lr': params.learning_rate}
            if self._can_use_fused_optim():
                adamw_kwargs['fused'] = True
            try:
                optimizers.append(optim.AdamW(adamw_groups, **adamw_kwargs))
            except TypeError:
                adamw_kwargs.pop('fused', None)
                optimizers.append(optim.AdamW(adamw_groups, **adamw_kwargs))

        if not optimizers:
            raise ValueError('No trainable parameters found for AdamW + Muon optimizer.')
        if len(optimizers) == 1:
            return optimizers[0]
        self._bus.put([
            'logging',
            (
                'AdamW + Muon: 2D parameters assigned to Muon, '
                f'other trainable parameters assigned to AdamW (Muon: {len(muon_params)}, AdamW: {len(adamw_params)}).'
            ),
        ])
        return _CombinedOptimizer(optimizers)
