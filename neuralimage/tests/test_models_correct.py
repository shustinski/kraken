import torch

from model.NeuralNetwork.blocks import extract_mask_outputs
from model.NeuralNetwork.registrator import create_model, get_registered_models


CHANNELS = 3


def test_registered_models_return_normalized_mask_outputs():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    dummy = torch.randn(1, CHANNELS, 32, 32, device=device)

    for model_name in get_registered_models():
        model = create_model(model_name, CHANNELS).to(device).eval()
        model_input = dummy
        if bool(getattr(model, 'use_context_branch', False)):
            model_input = {
                'local_image': dummy,
                'context_image': dummy,
            }
        with torch.inference_mode():
            outputs = model(model_input)
        mask_outputs = extract_mask_outputs(outputs)
        if isinstance(mask_outputs, (list, tuple)):
            assert mask_outputs
            primary = mask_outputs[0]
        else:
            primary = mask_outputs
        assert torch.is_tensor(primary), f'{model_name} returned unsupported primary output type'
        assert primary.ndim == 4, f'{model_name} returned invalid output rank {primary.ndim}'
        assert int(primary.shape[0]) == 1, f'{model_name} changed batch dimension'
