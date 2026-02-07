
"""
NeuroFlow BharatFlow — Advanced CoreML Training (V2)
Trains the upgraded ST-GCN on rich synthetic data for 12-hour forecasting.
"""

import logging
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import joblib
import sys
import os

# Add neuroflow_backend to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
backend_dir = os.path.abspath(os.path.join(current_dir, "../../.."))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir) # Insert at 0 to prioritize local

import app
print(f"Imported app from: {app.__file__}")

from app.engine.models import IndoTrafficSTGCN, PhysicsInformedLoss
from app.utils.scaler import ManualScaler

# Configure Logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("neuroflow.train_v2")

class AdvancedTrafficDataset(Dataset):
    """
    Dataset for Long-Horizon Forecasting.
    Input: [T_in=24] (6 hours)
    Output: [T_out=48] (12 hours)
    """
    def __init__(self, data: np.ndarray, num_nodes: int, time_steps: int = 24, horizons: int = 48):
        self.data = data
        self.num_nodes = num_nodes
        self.time_steps = time_steps
        self.horizons = horizons
        
        # Valid start indices
        self.valid_indices = self.data.shape[0] - self.time_steps - self.horizons

    def __len__(self):
        return self.valid_indices

    def __getitem__(self, idx):
        # Input sequence: [T, N, F] -> Transpose to [N, T, F]
        X = self.data[idx : idx + self.time_steps] 
        X = X.transpose(1, 0, 2) # [N, T, F]
        
        # Targets: Next 48 steps of Speed (Feature 0)
        # [T_out, N] -> Transpose to [N, T_out]
        base = idx + self.time_steps
        Y_seq = self.data[base : base + self.horizons, :, 0] # Speed only
        Y = Y_seq.transpose(1, 0) # [N, T_out]
        
        return torch.FloatTensor(X), torch.FloatTensor(Y)

class AdvancedTrainer:
    def __init__(self, data_dir: str = "neuroflow_backend/data/datasets_v2", city: str = "bengaluru"):
        self.data_dir = Path(data_dir)
        self.city = city
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.weights_dir = Path("neuroflow_backend/ml_models/weights")
        self.weights_dir.mkdir(parents=True, exist_ok=True)
        self.scaler_path = self.weights_dir / f"scaler_{city}_v2.pkl"
        
    def train(self, epochs: int = 30, batch_size: int = 16):
        logger.info(f"🚀 Starting V2 Training for {self.city.upper()}...")
        
        # 1. Load Data
        csv_path = self.data_dir / f"{self.city}_train_v2.csv"
        if not csv_path.exists():
            logger.error(f"❌ Data file not found: {csv_path}")
            return
            
        logger.info(f"   Loading {csv_path}...")
        df = pd.read_csv(csv_path)
        
        # 2. Select Features & Preprocess
        # Features to use:
        feature_cols = [
            'speed', 'volume', 'occupancy', 
            'rain_intensity', 'weather_severity_index',
            'event_attendance', 'holiday_intensity_score',
            'is_peak_hour', 'is_weekday'
        ]
        
        # Ensure booleans are numeric
        df['is_peak_hour'] = df['is_peak_hour'].astype(float)
        df['is_weekday'] = df['is_weekday'].astype(float)
        
        # Pivot to [Time, Node, Features]
        nodes = sorted(df['segment_id'].unique())
        num_nodes = len(nodes)
        node_map = {n: i for i, n in enumerate(nodes)}
        df['node_idx'] = df['segment_id'].map(node_map)
        df['time_idx'] = pd.factorize(df['timestamp'])[0]
        num_timesteps = df['time_idx'].max() + 1
        
        logger.info(f"   Nodes: {num_nodes}, TimeSteps: {num_timesteps}")
        
        # Create Data Matrix
        num_features = len(feature_cols)
        data_matrix = np.zeros((num_timesteps, num_nodes, num_features), dtype=np.float32)
        
        logger.info("   Pivoting data structure...")
        for i, col in enumerate(feature_cols):
            # Fill NaNs with 0 strictly
            val = df.pivot(index='time_idx', columns='node_idx', values=col).fillna(0).values
            data_matrix[:, :, i] = val
            
        # Normalize
        logger.info("   Normalizing features...")
        # Flatten to fit scaler
        flat_data = data_matrix.reshape(-1, num_features)
        scaler = ManualScaler()
        flat_data = scaler.fit_transform(flat_data)
        data_matrix = flat_data.reshape(num_timesteps, num_nodes, num_features)
        
        # Save scaler
        joblib.dump(scaler, self.scaler_path)
        
        # 3. Split
        split = int(num_timesteps * 0.85)
        train_data = data_matrix[:split]
        val_data = data_matrix[split:]
        
        train_dataset = AdvancedTrafficDataset(train_data, num_nodes, time_steps=24, horizons=48)
        val_dataset = AdvancedTrafficDataset(val_data, num_nodes, time_steps=24, horizons=48)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        # 4. Model
        model = IndoTrafficSTGCN(
            num_nodes=num_nodes,
            in_features=num_features,
            hidden_dim=64,
            output_horizons=48, # 12 hours
            temporal_steps=24   # 6 hours lookback
        ).to(self.device)
        
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.MSELoss() # Standard MSE for v2, physics loss can be added if needed
        
        # 5. Loop
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            model.train()
            train_loss = 0
            for X, Y in train_loader:
                X, Y = X.to(self.device), Y.to(self.device)
                optimizer.zero_grad()
                pred = model(X) # [B, N, 48]
                loss = criterion(pred, Y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            # Validation
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for X, Y in val_loader:
                    X, Y = X.to(self.device), Y.to(self.device)
                    pred = model(X)
                    loss = criterion(pred, Y)
                    val_loss += loss.item()
            
            avg_val = val_loss / len(val_loader)
            logger.info(f"Epoch {epoch+1}: Val Loss {avg_val:.4f}")
            
            if avg_val < best_val_loss:
                best_val_loss = avg_val
                torch.save(model.state_dict(), self.weights_dir / f"stgcn_{self.city}_v2.pth")
                
        logger.info(f"✅ Training Complete. Saved to {self.weights_dir}")

if __name__ == "__main__":
    import traceback
    try:
        for city in ["mumbai", "bengaluru", "delhi"]:
            try:
                trainer = AdvancedTrainer(city=city)
                trainer.train(epochs=5) # 5 epochs for testing
            except Exception as e:
                logger.error(f"Error training {city}: {e}")
                # Log traceback
                with open("train_error.log", "a") as f:
                    f.write(f"Error for {city}:\n")
                    traceback.print_exc(file=f)
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        with open("train_error.log", "a") as f:
            f.write("Fatal error:\n")
            traceback.print_exc(file=f)
