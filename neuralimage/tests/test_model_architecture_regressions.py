import pytest

torch = pytest.importorskip('torch')
nn = pytest.importorskip('torch.nn')

from model.NeuralNetwork import create_model, get_registered_model_registry, get_registered_models
from model.NeuralNetwork.registrator import ModelType
from model.NeuralNetwork.CNN_Models import (
    ImageBinarizationTransformer,
    MultiLayerFCNN,
    UberModel,
    Unet,
    Wellnet,
    Wellnet2,
    Wellnet2Mini,
)
from model.NeuralNetwork import transformer_segmentation as transformer_segmentation_module
from model.NeuralNetwork.transformer_segmentation import NativeHierarchicalBackbone


def test_quasi_dual_scale_unet_alias_is_registered():
    registered = get_registered_models()

    assert 'quasi_dual_scale_unet' in registered
    assert registered['quasi_dual_scale_unet'] is registered['FrameUnet']


def test_registered_models_expose_model_type_metadata():
    registry = get_registered_model_registry()

    assert registry['EfficientUNet']['model_type'] is ModelType.stable
    assert registry['UNET++']['model_type'] is ModelType.experimental
    assert registry['S 660k']['model_type'] is ModelType.deprecated
    assert registry['Wellnet2']['model_type'] is ModelType.deprecated
    assert registry['EfficientUNetMax']['model_type'] is ModelType.experimental
    assert registry['FrameUnet']['model_type'] is ModelType.experimental
    assert registry['Swin UPerNet B']['model_type'] is ModelType.experimental
    assert registry['Transformer']['model_type'] is ModelType.experimental


def test_transformer_restores_original_non_square_output_size():
    model = ImageBinarizationTransformer(in_channels=1, img_size=64, patch_size=16, embed_dim=120, depth=2, num_heads=6)

    outputs = model(torch.randn(2, 1, 48, 80))

    assert tuple(outputs['mask'].shape) == (2, 1, 48, 80)
    assert tuple(outputs['confidence'].shape) == (2, 1, 48, 80)


def test_legacy_unet_family_uses_raw_logit_output_heads():
    assert isinstance(Unet(in_channels=1).out_conv, nn.Conv2d)
    assert isinstance(Wellnet(in_channels=1).finish_upsample[-1], nn.Conv2d)
    assert isinstance(Wellnet2(in_channels=1).finish_upsample[-1], nn.Conv2d)
    assert isinstance(Wellnet2Mini(in_channels=1).finish_upsample[-1], nn.Conv2d)


def test_multilayer_fcnn_returns_single_channel_mask():
    model = MultiLayerFCNN(input_channels=1, layers=2, start_filter=16, step_filter=8)

    outputs = model(torch.randn(1, 1, 64, 64))

    assert tuple(outputs['mask'].shape) == (1, 1, 64, 64)
    assert tuple(outputs['confidence'].shape) == (1, 1, 64, 64)


def test_uber_model_tta_uses_original_input_flips():
    model = UberModel(input_channels=1)
    model.encoder = nn.Identity()

    class _AddOne(nn.Module):
        def forward(self, x):
            return x + 1

    model.decoder = _AddOne()

    inputs = torch.zeros(1, 1, 32, 48)
    outputs = model(inputs, tta=True)

    assert tuple(outputs.shape) == (1, 1, 32, 48)
    assert torch.allclose(outputs, torch.ones_like(outputs))


def test_quasi_dual_scale_alias_can_instantiate():
    model = create_model('quasi_dual_scale_unet', 1, use_context_branch=False)

    outputs = model(torch.randn(1, 1, 64, 64))

    assert tuple(outputs['mask'].shape) == (1, 1, 64, 64)
    assert tuple(outputs['confidence'].shape) == (1, 1, 64, 64)


@pytest.mark.parametrize(
    ('model_name', 'kwargs'),
    [
        ('S 660k', {}),
        ('M 720k', {}),
        ('Unet 21.6M', {}),
        ('Wellnet2 mini', {}),
        ('EfficientUNet', {}),
        ('UNET++', {}),
        ('quasi_dual_scale_unet', {'use_context_branch': False}),
    ],
)
def test_convolutional_models_support_deep_supervision_in_train_only(model_name, kwargs):
    model = create_model(model_name, 1, deep_supervision=True, **kwargs)
    model.train()

    outputs = model(torch.randn(1, 1, 64, 64))

    assert isinstance(outputs, dict)
    assert isinstance(outputs['mask'], tuple)
    assert len(outputs['mask']) >= 2
    assert tuple(outputs['mask'][0].shape) == (1, 1, 64, 64)
    assert all(tuple(output.shape) == (1, 1, 64, 64) for output in outputs['mask'])
    assert tuple(outputs['confidence'].shape) == (1, 1, 64, 64)


def test_unet_plus_plus_returns_single_mask_in_eval_mode_even_with_deep_supervision():
    model = create_model('UNET++', 1, deep_supervision=True)
    model.eval()

    outputs = model(torch.randn(1, 1, 96, 96))

    assert tuple(outputs['mask'].shape) == (1, 1, 96, 96)
    assert tuple(outputs['confidence'].shape) == (1, 1, 96, 96)


def test_unet_plus_plus_nodes_use_group_norm():
    model = create_model('UNET++', 1)

    assert isinstance(model.x0_0.block[0].norm, nn.GroupNorm)
    assert isinstance(model.x0_0.block[1].norm, nn.GroupNorm)


def test_swin_upernet_uses_real_timm_swin_backbone_and_preserves_output_size():
    model = create_model('Swin UPerNet B', 1)

    assert not isinstance(model.backbone, NativeHierarchicalBackbone)
    assert type(model.backbone).__module__.startswith('timm.')

    outputs = model(torch.randn(1, 1, 96, 128))

    assert tuple(outputs['mask'].shape) == (1, 1, 96, 128)
    assert tuple(outputs['confidence'].shape) == (1, 1, 96, 128)


def test_mask2former_swin_uses_real_timm_swin_backbone_and_preserves_output_size():
    model = create_model('Mask2Former Swin B', 1)

    assert not isinstance(model.backbone, NativeHierarchicalBackbone)
    assert type(model.backbone).__module__.startswith('timm.')

    outputs = model(torch.randn(1, 1, 96, 128))

    assert tuple(outputs['mask'].shape) == (1, 1, 96, 128)
    assert tuple(outputs['confidence'].shape) == (1, 1, 96, 128)


def test_offline_swin_pretrained_uses_internal_weight_file(monkeypatch, tmp_path):
    captured: dict[str, object] = {}
    local_weight_file = tmp_path / 'model.safetensors'
    local_weight_file.write_bytes(b'test')

    class _FeatureInfo:
        def channels(self):
            return [128, 256, 512, 1024]

    class _Backbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.feature_info = _FeatureInfo()

        def forward(self, x):
            return [
                torch.randn(x.shape[0], 128, max(1, x.shape[-2] // 4), max(1, x.shape[-1] // 4)),
                torch.randn(x.shape[0], 256, max(1, x.shape[-2] // 8), max(1, x.shape[-1] // 8)),
                torch.randn(x.shape[0], 512, max(1, x.shape[-2] // 16), max(1, x.shape[-1] // 16)),
                torch.randn(x.shape[0], 1024, max(1, x.shape[-2] // 32), max(1, x.shape[-1] // 32)),
            ]

    def _fake_create_model(model_name, **kwargs):
        captured['model_name'] = model_name
        captured['kwargs'] = kwargs
        return _Backbone()

    monkeypatch.setattr(transformer_segmentation_module, 'timm', type('TimmStub', (), {'create_model': staticmethod(_fake_create_model)}))
    monkeypatch.setattr(transformer_segmentation_module, '_allow_pretrained_weights', lambda: True)
    monkeypatch.setattr(transformer_segmentation_module, '_resolve_local_swin_weight_file', lambda _name: local_weight_file)

    model = create_model('Swin UPerNet B', 1)

    assert captured['model_name'] == 'swin_base_patch4_window7_224'
    assert captured['kwargs']['pretrained'] is True
    assert captured['kwargs']['pretrained_cfg_overlay'] == {'file': str(local_weight_file)}
    outputs = model(torch.randn(1, 1, 64, 64))
    assert tuple(outputs['mask'].shape) == (1, 1, 64, 64)
    assert tuple(outputs['confidence'].shape) == (1, 1, 64, 64)


def test_offline_swin_pretrained_requires_internal_weight_file(monkeypatch, tmp_path):
    missing_file = tmp_path / 'missing' / 'model.safetensors'

    monkeypatch.setattr(transformer_segmentation_module, '_allow_pretrained_weights', lambda: True)
    monkeypatch.setattr(transformer_segmentation_module, '_resolve_local_swin_weight_file', lambda _name: missing_file)

    with pytest.raises(FileNotFoundError):
        create_model('Swin UPerNet B', 1)
