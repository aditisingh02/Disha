import torch
from pathlib import Path

ckpt = torch.load(Path('ml_models/weights/stgcn_singapore_v1.pth'), map_location='cpu', weights_only=False)

# Print all keys and values
for k, v in ckpt.items():
    if k != 'model_state_dict':
        print(f"{k}: {v}")
