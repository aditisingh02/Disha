import torch
import torch.nn as nn

class DeepTrafficModel(nn.Module):
    """
    Deep Residual Network for traffic prediction (Singapore Model).
    Architecture: Input -> [Linear->LayerNorm->GELU->Dropout] x 8 -> Output
    """
    
    def __init__(self, num_features, hidden=512, num_layers=8):
        super().__init__()
        
        # Input layer
        self.input_layer = nn.Sequential(
            nn.Linear(num_features, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(0.2)
        )
        
        # Residual blocks
        self.res_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden, hidden),
                nn.LayerNorm(hidden),
                nn.GELU(),
                nn.Dropout(0.15),
                nn.Linear(hidden, hidden),
                nn.LayerNorm(hidden)
            ) for _ in range(num_layers)
        ])
        
        # Output head
        self.output = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden // 2, 1)
        )
        
        self.gelu = nn.GELU()
        
    def forward(self, x):
        h = self.input_layer(x)
        
        for block in self.res_blocks:
            h = self.gelu(h + block(h))  # Residual connection
        
        return self.output(h).squeeze(-1)
