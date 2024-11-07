import os, sys
from depth_anything.dpt import DepthAnything
import torch

def test_model():

    model_configs = {
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]}
    }

    encoder = 'vitl' # or 'vitb', 'vits'
    depth_anything = DepthAnything(model_configs[encoder])
    
    state_dict = torch.load(f'./checkpoints/depth_anything_{encoder}14.pth')
    print(state_dict.keys())
    
    depth_anything.load_state_dict(state_dict)

    print()

if __name__ == "__main__":
    test_model()