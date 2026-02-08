
import torch
import torch.nn as nn


class DeepTrafficModel(nn.Module):
    """
    Deep model with residual connections for traffic prediction.
    Matches the architecture from train_singapore.py.
    """
    
    def __init__(self, num_features, hidden=512, num_layers=8):
        super().__init__()
        self.num_features = num_features
        
        # Input layer
        self.input_layer = nn.Sequential(
            nn.Linear(num_features, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(0.2)
        )
        
        # Residual blocks for better gradient flow
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
        
        # Output head - predicts single speed band value
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


class DeepTrafficModelV2(nn.Module):
    """
    Deep traffic model V2 with attention mechanism.
    Matches the architecture from train_singapore.py V2 checkpoint.
    Keys: input_embed, res_blocks.X.net.Y, attention, attn_norm, classifier
    """
    
    def __init__(self, num_features, hidden=512, num_layers=8):
        super().__init__()
        self.num_features = num_features
        
        # Input embedding (matches input_embed.0/1 in checkpoint)
        self.input_embed = nn.Sequential(
            nn.Linear(num_features, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(0.2)
        )
        
        # Residual blocks with nested 'net' module
        self.res_blocks = nn.ModuleList([
            self._make_res_block(hidden) for _ in range(num_layers)
        ])
        
        # Self-attention layer
        self.attention = nn.MultiheadAttention(hidden, num_heads=8, batch_first=True)
        self.attn_norm = nn.LayerNorm(hidden)
        
        # Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.LayerNorm(hidden // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden // 2, hidden // 4),
            nn.LayerNorm(hidden // 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden // 4, 1)
        )
        
        self.gelu = nn.GELU()
        
    def _make_res_block(self, hidden):
        """Create a residual block with nested 'net' structure."""
        return nn.ModuleDict({
            'net': nn.Sequential(
                nn.Linear(hidden, hidden),
                nn.LayerNorm(hidden),
                nn.GELU(),
                nn.Dropout(0.15),
                nn.Linear(hidden, hidden),
                nn.LayerNorm(hidden)
            )
        })
        
    def forward(self, x):
        # Input embedding
        h = self.input_embed(x)
        
        # Residual blocks
        for block in self.res_blocks:
            h = self.gelu(h + block['net'](h))
        
        # Attention (for 2D input, add sequence dimension)
        if h.dim() == 2:
            h = h.unsqueeze(1)  # [B, 1, H]
        attn_out, _ = self.attention(h, h, h)
        h = self.attn_norm(h + attn_out)
        h = h.squeeze(1)  # [B, H]
        
        # Classifier
        return self.classifier(h).squeeze(-1)


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

class ResidualGCN(nn.Module):
    """
    Residual-only GCN for q90 (and optionally q95) tail risk.
    Does NOT predict median. Outputs q90_residual to add to baseline q50.
    Input features include spatial enrichment: neighbor_baseline_gradient, neighbor_residual_mean,
    corridor_bottleneck_score, adjacency_degree (in_features = 9 + 4 = 13 by default).
    """

    def __init__(self, num_nodes, in_features=13, hidden_dim=64, output_horizons=48, temporal_steps=24):
        super().__init__()
        self.num_nodes = num_nodes
        self.temporal_steps = temporal_steps
        self.spatial = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.temporal = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.out = nn.Linear(hidden_dim * temporal_steps, output_horizons)

    def forward(self, x):
        # x: [B, N, T, F]
        B, N, T, F = x.shape
        x_flat = x.view(-1, F)
        h = self.spatial(x_flat)
        h = h.view(B * N, T, -1).transpose(1, 2)
        h = self.temporal(h)
        h = h.reshape(B, N, -1)
        return self.out(h)


class PhysicsInformedLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, pred, target):
        return self.mse(pred, target)

class SimpleGCNLSTM(nn.Module):
    """
    Spatial: shared linear (simplified GCN); Temporal: LSTM. 
    Matches the architecture used in phase2_forecasting.py.
    In: [B, N, T, F]; Out: [B, N, n_horizons]
    """

    def __init__(self, num_nodes: int, in_features: int, hidden: int = 64, n_horizons: int = 3):
        super().__init__()
        self.n_horizons = n_horizons
        self.spatial = nn.Linear(in_features, hidden)
        self.lstm = nn.LSTM(hidden, hidden, batch_first=True)
        self.out = nn.Linear(hidden, n_horizons)

    def forward(self, x):
        # x: [B, N, T, F]
        B, N, T, F = x.shape
        x = x.permute(0, 2, 1, 3).reshape(B * T, N, F)
        h = torch.relu(self.spatial(x))
        h = h.reshape(B, T, N, -1).permute(0, 2, 1, 3)
        h = h.reshape(B * N, T, -1)
        out, _ = self.lstm(h)
        out = out[:, -1]
        out = self.out(out)
        out = out.reshape(B, N, self.n_horizons)
        return out
