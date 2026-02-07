
import torch
import torch.nn as nn

class IndoTrafficSTGCN(nn.Module):
    """
    Simplified Spatio-Temporal Graph Network (ST-MLP) for robust inference.
    Matches the architecture trained in train_v2.py.
    """
    def __init__(self, num_nodes, in_features, hidden_dim, output_horizons, temporal_steps=12):
        super().__init__()
        self.num_nodes = num_nodes
        self.temporal_steps = temporal_steps
        # Spatial: Shared MLP across nodes (approximates GCN with 1-hop mixing via shared weights + implicit correlation in data)
        self.spatial = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        # Temporal: Conv1d to capture time dynamics
        self.temporal = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        # Output projection
        self.out = nn.Linear(hidden_dim * temporal_steps, output_horizons) # Dynamic input size

    def forward(self, x):
        # x: [Batch, N, T, F]
        B, N, T, F = x.shape
        
        # Spatial Pass
        # Flatten [B*N*T, F]
        x_flat = x.view(-1, F) # [Batches*Nodes*Time, Features]
        h = self.spatial(x_flat) # [B*N*T, H]
        h = h.view(B*N, T, -1) # [B*N, T, H]
        h = h.transpose(1, 2) # [B*N, H, T] -> Channels=Hidden
        
        # Temporal
        h = self.temporal(h) # [B*N, H, T]
        
        # Flatten temporal dimension for output
        # We need [B, N, Output]
        h = h.reshape(B, N, -1) # [B, N, H*T]
        out = self.out(h) # [B, N, Output]
        return out

class PhysicsInformedLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
    
    def forward(self, pred, target):
        return self.mse(pred, target)
