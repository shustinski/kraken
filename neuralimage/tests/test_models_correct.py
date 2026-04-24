import pytest

import torch
from torch import nn
import torch.nn.functional as F
from torch.cpu.amp import GradScaler,autocast

from lib import System
from model.NeuralNetwork.registrator import *
from model.NeuralNetwork.blocks import *

CHANNELS = 3
def test_model_correctness():
    model_names = get_registered_models()
    gpus = System.check_gpu_availability()
    devices_list = [torch.device(f"cuda:{gpu}") for gpu in range(gpus)]
    scaler = GradScaler()

    # Set environment variable to avoid fragmentation
    import os

    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

    for model in model_names:
        # try:
        model_test = create_model(model, CHANNELS)
        model_test = model_test.to(devices_list[0])

        trainable = sum(p.numel() for p in model_test.parameters() if p.requires_grad)
        print(f"Trainable parameters for {model}: {trainable:,}")

        # Reduce batch size to prevent memory issues
        dummy = torch.randn(1, CHANNELS, 32, 32)  # batch=1, single-channel image
        dummy = dummy.to(devices_list[0])

        with autocast():
            out = model_test(dummy)

        out = out.to('cpu')
        torch.cuda.empty_cache()
        print('output shape:', out.shape)

            # except torch.cuda.OutOfMemoryError as e:
        #     print(f"OutOfMemoryError with model {model}: {e}")
        #     torch.cuda.empty_cache()  # Clear cache and continue with next model
        #     continue
        # except Exception as e:
        #     print(f"Error with model {model}: {e}")
        #     torch.cuda.empty_cache()
        #     continue  # expected shape: (4, 1, 256, 256)
    # model_test = create_model('Transformer', 1, 256, 16)
    # model_test = model_test.to(devices_list[0])
    # trainable = sum(p.numel() for p in model_test.parameters() if p.requires_grad)
    # rainable = model_test.parameters()
    # print(f"Trainable parameters for {'Transformer'}: {trainable:,}")
    # dummy = torch.randn(4, 1, 512, 512)  # batch=4, single-channel image
    # dummy = dummy.to(devices_list[0])
    # out = model_test(dummy)
    # out = out.to('cpu')
    # print('output shape:', out.shape)  # expected shape: (4, 1, 256, 256)

    # model = BigCnnV2(inputs=1, base_ch=64, latent_dim=256,
    #                  use_se=True, use_res=True,
    #                  norm='gn', act='gelu')

