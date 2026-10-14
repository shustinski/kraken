import math
from collections.abc import Sequence
import torch
from torch import nn
import torch.nn.functional as F
from torch.cpu.amp import GradScaler,autocast
from neuralimage.lib import System
from .registrator import *
from .blocks import *
from neuralimage.heads.multi_target import MultiTargetHeadBundle

from .CNN_Models_parts.u_net_plus_plus import (
    _conv_bn_lrelu,
    _FCNNEncoderStage,
    _FCNNDecoderStage,
    _UNetPlusPlusNode,
)
from .CNN_Models_parts.efficient_u_net_base import _EfficientUNetBase
from .CNN_Models_parts.small_fcnn import SmallFCNN
from .CNN_Models_parts.medium_fcnn_transpose import (
    MediumFCNN,
    MediumFCNNTranspose,
)
from .CNN_Models_parts.medium_fcnnselu import MediumFCNNSELU
from .CNN_Models_parts.medium_fcnn_less_pool import MediumFCNNLessPool
from .CNN_Models_parts.medium_optimised import MediumOptimised
from .CNN_Models_parts.big_cnn import BigCnn
from .CNN_Models_parts.unet import Unet
from .CNN_Models_parts.wellnet import Wellnet
from .CNN_Models_parts.wellnet2 import Wellnet2
from .CNN_Models_parts.wellnet2_mini import Wellnet2Mini
from .CNN_Models_parts.efficient_u_net import (
    EfficientUNet,
    EfficientUNetMax,
)
from .CNN_Models_parts.unet_plus_plus import UnetPlusPlus
from .CNN_Models_parts.wellnet_ultra import WellnetUltra
from .CNN_Models_parts.multi_layer_fcnn import MultiLayerFCNN
from .CNN_Models_parts.dense_residual_block import (
    DepthwiseSeparableConv,
    DenseResidualBlock,
    UpsampleBlock,
)
from .CNN_Models_parts.uber_model import UberModel
from .CNN_Models_parts.vision_transformer import (
    ConvBNAct,
    DropPath,
    SwiGLUMLP,
    HybridOverlapPatchEmbedding,
    Learnable2DPositionalEncoding,
    TransformerEncoderLayer,
    VisionTransformer,
)
from .CNN_Models_parts.image_binarization_transformer import ImageBinarizationTransformer

# Imported so existing callers and tests can patch or import these names
# from this module after the implementation moved into focused units.
_FACADE_EXPORTS = (
    _conv_bn_lrelu,
    _FCNNEncoderStage,
    _FCNNDecoderStage,
    _UNetPlusPlusNode,
    _EfficientUNetBase,
    SmallFCNN,
    MediumFCNN,
    MediumFCNNTranspose,
    MediumFCNNSELU,
    MediumFCNNLessPool,
    MediumOptimised,
    BigCnn,
    Unet,
    Wellnet,
    Wellnet2,
    Wellnet2Mini,
    EfficientUNet,
    EfficientUNetMax,
    UnetPlusPlus,
    WellnetUltra,
    MultiLayerFCNN,
    DepthwiseSeparableConv,
    DenseResidualBlock,
    UpsampleBlock,
    UberModel,
    ConvBNAct,
    DropPath,
    SwiGLUMLP,
    HybridOverlapPatchEmbedding,
    Learnable2DPositionalEncoding,
    TransformerEncoderLayer,
    VisionTransformer,
    ImageBinarizationTransformer,
)
