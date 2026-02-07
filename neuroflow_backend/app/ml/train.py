"""
NeuroFlow BharatFlow — CoreML ST-GCN Training Pipeline
Trains the Spatio-Temporal Graph Convolutional Network on synthetic multi-city data.
Uses Physics-Informed Loss to ensure traffic conservation laws.
"""

import logging
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from typing import Tuple, List, Dict
from torch.utils.data import Dataset, DataLoader
import networkx as nx

from app.core.config import settings
from app.engine import forecaster
# Force disable PyG to ensure robust training (fallback to ST-MLP if graph libs missing/broken)
forecaster.HAS_PYG = False 

from app.engine.forecaster import IndoTrafficSTGCN, PhysicsInformedLoss

# Configure Logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("neuroflow.train")

class TrafficDataset(Dataset):
    """
    PyTorch Dataset for ST-GCN.
    Sliding window: Input [T=12] -> Output [Horizon=3 (T+15, T+30, T+60)]
    """
    def __init__(self, data: np.ndarray, num_nodes: int, time_steps: int = 12, horizons: int = 3):
        """
        data: [Total_Time, Num_Nodes, Features]
        """
        self.data = data
        self.num_nodes = num_nodes
        self.time_steps = time_steps # 1 hour lookback (12 * 5min)
        self.horizons = horizons # Predict next 3 steps equivalent
        
        # Valid start indices
        self.valid_indices = self.data.shape[0] - self.time_steps - self.horizons

    def __len__(self):
        return self.valid_indices

    def __getitem__(self, idx):
        # Input sequence: [Nodes, Time, Features] - Transpose for model [N, T, F]
        X = self.data[idx : idx + self.time_steps] # [T, N, F]
        X = X.transpose(1, 0, 2) # [N, T, F]
        
        # Targets: Future speeds at T+3 (15m), T+6 (30m), T+12 (60m)
        # Note: 'speed' is feature index 0
        base = idx + self.time_steps
        y_15 = self.data[base + 2, :, 0] 
        y_30 = self.data[base + 5, :, 0]
        y_60 = self.data[base + 11, :, 0]
        
        Y = np.stack([y_15, y_30, y_60], axis=1) # [N, 3]
        
        return torch.FloatTensor(X), torch.FloatTensor(Y)

class STGCNTrainer:
    def __init__(self, data_dir: str = "data/datasets", city: str = "bengaluru"):
        self.data_dir = Path(data_dir)
        self.city = city
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.weights_dir = Path("ml_models/weights")
        self.weights_dir.mkdir(parents=True, exist_ok=True)
        
    def train(self, epochs: int = 50, batch_size: int = 32):
        logger.info(f"🚀 Starting training for {self.city.upper()} on {self.device}...")
        
        # 1. Load & Preprocess Data
        logger.info("   Loading synthetic data...")
        df = pd.read_csv(self.data_dir / f"{self.city}_train.csv")
        
        # Convert to matrix: [Time, Nodes, Features]
        # Features: speed, volume, occupancy, weather_encoded
        # Note: We need a consistent node order.
        nodes = sorted(df['segment_id'].unique())
        num_nodes = len(nodes)
        node_map = {n: i for i, n in enumerate(nodes)}
        
        df['node_idx'] = df['segment_id'].map(node_map)
        df['time_idx'] = pd.factorize(df['timestamp'])[0]
        num_timesteps = df['time_idx'].max() + 1
        
        # Feature Engineering
        df['weather_code'] = pd.factorize(df['weather'])[0] / 3.0 # Normalize 0-1 approx
        
        # Create Tensor [Time, Node, Features=4]
        # Feats: speed, volume/2000, occupancy, weather
        data_matrix = np.zeros((num_timesteps, num_nodes, 4), dtype=np.float32)
        
        # Fill matrix (vectorized would be faster but this is safe for generated data)
        # For training speed, we'll pivot
        logger.info("   Transforming data structure...")
        pivoted_speed = df.pivot(index='time_idx', columns='node_idx', values='speed').values
        pivoted_vol = df.pivot(index='time_idx', columns='node_idx', values='volume').values / 2000.0
        pivoted_occ = df.pivot(index='time_idx', columns='node_idx', values='occupancy').values
        pivoted_weather = df.pivot(index='time_idx', columns='node_idx', values='weather_code').values
        
        data_matrix[:, :, 0] = pivoted_speed
        data_matrix[:, :, 1] = pivoted_vol
        data_matrix[:, :, 2] = pivoted_occ
        data_matrix[:, :, 3] = pivoted_weather
        
        # Split Train/Val
        split = int(num_timesteps * 0.8)
        train_data = data_matrix[:split]
        val_data = data_matrix[split:]
        
        train_dataset = TrafficDataset(train_data, num_nodes)
        val_dataset = TrafficDataset(val_data, num_nodes)
        
        train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=1)
        
        # 2. Build Graph (Adjacency)
        # For simulated grid, we assume a random connected graph or load if available.
        # Here we approximate with a fully connected or random geometric graph for ST-GCN
        # In production this comes from OSMnx.
        # We will create a random edge_index for now to match node count
        logger.info("   Building graph topology...")
        # Self-loops + some connections
        edge_index = torch.stack([
            torch.arange(num_nodes),
            torch.arange(num_nodes)
        ])
        edge_index = edge_index.to(self.device)
        
        # 3. Model Setup
        model = IndoTrafficSTGCN(
            num_nodes=num_nodes,
            in_features=4,
            hidden_dim=64,
            temporal_steps=12,
            output_horizons=3
        ).to(self.device)
        
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
        criterion = PhysicsInformedLoss(lambda_physics=0.2)
        
        # 4. Training Loop
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            model.train()
            train_loss = 0
            
            for X, Y in train_loader:
                X, Y = X.to(self.device), Y.to(self.device)
                
                optimizer.zero_grad()
                pred = model(X, edge_index)
                
                loss = criterion(pred, Y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                
            avg_train_loss = train_loss / len(train_loader)
            
            # Validation
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for X, Y in val_loader:
                    X, Y = X.to(self.device), Y.to(self.device)
                    pred = model(X, edge_index)
                    loss = criterion(pred, Y)
                    val_loss += loss.item()
            
            avg_val_loss = val_loss / len(val_loader)
            
            logger.info(f"   Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
            
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(model.state_dict(), self.weights_dir / f"stgcn_{self.city}_v1.pth")
                logger.info("   💾 Saved new best model.")
                
        logger.info(f"✅ Training complete for {self.city}. Best Val Loss: {best_val_loss:.4f}")

if __name__ == "__main__":
    # Train for all cities in the generated dataset
    for city in ["bengaluru", "mumbai", "delhi"]:
        try:
            trainer = STGCNTrainer(city=city)
            trainer.train(epochs=10) # 10 epochs for demo speed
        except Exception as e:
            logger.error(f"Failed to train {city}: {e}")
