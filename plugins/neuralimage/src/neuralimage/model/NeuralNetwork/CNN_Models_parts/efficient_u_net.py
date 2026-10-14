"""CNN_Models: efficient u net."""
from __future__ import annotations
from neuralimage.model.NeuralNetwork.registrator import *
from neuralimage.model.NeuralNetwork.blocks import *

from neuralimage.model.NeuralNetwork.CNN_Models_parts.efficient_u_net_base import _EfficientUNetBase

@register_model('EfficientUNet', model_type=ModelType.stable)
class EfficientUNet(_EfficientUNetBase):
    def __init__(
        self,
        in_ch=1,
        deep_supervision: bool = False,
        dropout_stem: float = 0.0,
        dropout_down: float = 0.08,
        dropout_bottleneck: float = 0.15,
        dropout_up: float = 0.03,
    ):
        super().__init__(
            in_ch=in_ch,
            base_ch=32,
            deep_supervision=deep_supervision,
            dropout_stem=dropout_stem,
            dropout_down=dropout_down,
            dropout_bottleneck=dropout_bottleneck,
            dropout_up=dropout_up,
        )

@register_model('EfficientUNetMax', model_type=ModelType.experimental)
class EfficientUNetMax(_EfficientUNetBase):
    def __init__(
        self,
        in_ch=1,
        deep_supervision: bool = False,
        dropout_stem: float = 0.0,
        dropout_down: float = 0.08,
        dropout_bottleneck: float = 0.15,
        dropout_up: float = 0.03,
    ):
        super().__init__(
            in_ch=in_ch,
            base_ch=64,
            deep_supervision=deep_supervision,
            dropout_stem=dropout_stem,
            dropout_down=dropout_down,
            dropout_bottleneck=dropout_bottleneck,
            dropout_up=dropout_up,
        )
