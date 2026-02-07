
import logging
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from torch.utils.data import Dataset, DataLoader

# Force simple model
class SimpleSTGCN(nn.Module):
    def __init__(self, num_nodes, in_features, hidden_dim, output_horizons):
        super().__init__()
        self.num_nodes = num_nodes
        # Spatial: Shared MLP across nodes
        self.spatial = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        # Temporal: Conv1d
        self.temporal = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.out = nn.Linear(hidden_dim * 12, output_horizons) # 12 timesteps

    def forward(self, x):
        # x: [Batch, Nodes, Time, Feat]
        B, N, T, F = x.shape
        
        # Spatial Pass
        # Flatten [B*N*T, F]
        x_flat = x.view(-1, F)
        h = self.spatial(x_flat) # [B*N*T, H]
        h = h.view(B*N, T, -1) # [B*N, T, H]
        h = h.transpose(1, 2) # [B*N, H, T]
        
        # Temporal
        h = self.temporal(h) # [B*N, H, T]
        
        # Flatten output
        h = h.reshape(B, N, -1) # [B, N, H*T]
        out = self.out(h) # [B, N, 3]
        return out

class FallbackTrainer:
    def __init__(self, city):
        self.city = city
        self.data_dir = Path("data/datasets")
        self.weights_dir = Path("ml_models/weights")
        self.weights_dir.mkdir(parents=True, exist_ok=True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def train(self):
        print(f"Training fallback model for {self.city}...")
        df = pd.read_csv(self.data_dir / f"{self.city}_train.csv")
        
        # Naive fillna
        if df.isnull().values.any():
            print("  Found NaNs, filling with 0")
            df = df.fillna(0)
            
        nodes = sorted(df['segment_id'].unique())
        num_nodes = len(nodes)
        print(f"  Nodes: {num_nodes}")
        
        # Pivot
        df['node_idx'] = df['segment_id'].astype('category').cat.codes
        df['time_idx'] = pd.factorize(df['timestamp'])[0]
        
        vals = df.pivot(index='time_idx', columns='node_idx', values='speed').values
        num_times = vals.shape[0]
        
        # Feature: Speed only for fallback
        data = vals[:, :, np.newaxis] # [T, N, 1]
        
        # Dataset
        class SimpleDataset(Dataset):
            def __init__(self, d):
                self.d = d
            def __len__(self):
                return self.d.shape[0] - 15
            def __getitem__(self, idx):
                X = self.d[idx : idx+12] # [12, N, 1]
                X = np.transpose(X, (1, 0, 2)) # [N, 12, 1]
                Y = self.d[idx+12, :, 0:1] # [N, 1] -> Target T+1 (simplified)
                # Expand Y to 3 horizons by repeating
                Y = np.concatenate([Y, Y, Y], axis=1) # [N, 3]
                return torch.FloatTensor(X), torch.FloatTensor(Y)
                
        dataset = SimpleDataset(data)
        loader = DataLoader(dataset, batch_size=16, shuffle=True)
        
        model = SimpleSTGCN(num_nodes, 1, 32, 3).to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        criterion = nn.MSELoss()
        
        for epoch in range(5):
            total_loss = 0
            for X, Y in loader:
                X, Y = X.to(self.device), Y.to(self.device)
                optimizer.zero_grad()
                pred = model(X)
                loss = criterion(pred, Y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print(f"  Epoch {epoch}: {total_loss/len(loader):.4f}")
            
        torch.save(model.state_dict(), self.weights_dir / f"stgcn_{self.city}_v1.pth")
        print("  Model saved.")

if __name__ == "__main__":
    for city in ["bengaluru", "mumbai", "delhi"]:
        try:
             FallbackTrainer(city).train()
        except Exception as e:
            print(f"Error {city}: {e}")
