"""
NeuroFlow Singapore - OPTIMIZED Training Script
Goal: 95%+ accuracy with ±1 band tolerance
Strategy: Deeper model, more epochs, better regularization
"""

import logging
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import joblib

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("neuroflow.train_final")


class DeepTrafficModel(nn.Module):
    """Deep model with residual connections for traffic prediction."""
    
    def __init__(self, num_features, hidden=512, num_layers=6):
        super().__init__()
        
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


class TrafficDataset(Dataset):
    def __init__(self, X, y, augment=False):
        self.X = X
        self.y = y
        self.augment = augment
        
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        x = self.X[idx].copy()
        if self.augment and np.random.random() < 0.3:
            x += np.random.normal(0, 0.02, x.shape)
        return torch.FloatTensor(x), torch.FloatTensor([self.y[idx]])[0]


class OptimizedTrainer:
    def __init__(self, data_path="data/datasets/training_dataset_enriched.csv"):
        self.data_path = Path(data_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.weights_dir = Path("ml_models/weights")
        self.weights_dir.mkdir(parents=True, exist_ok=True)
        self.label_encoders = {}
        self.scaler = StandardScaler()
        
        logger.info(f"🚀 Optimized Trainer on {self.device}")
        
    def load_data(self):
        df = pd.read_csv(self.data_path)
        logger.info(f"📂 Loaded {len(df)} records")
        
        y = df['speed_band'].values.astype(np.float32)
        
        # Boolean columns
        bool_cols = ['is_weekday', 'is_peak_hour', 'has_incident', 'is_rainy',
                     'extreme_weather_flag', 'is_holiday', 'has_major_event',
                     'has_sports_event', 'has_concert', 'has_conference']
        for col in bool_cols:
            if col in df.columns:
                df[col] = df[col].map({True:1, False:0, 'True':1, 'False':0, 1:1, 0:0}).fillna(0).astype(int)
        
        # Cyclical encoding
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        
        # Categorical 
        for col in ['road_name', 'road_category', 'peak_driver_type', 'holiday_type']:
            if col in df.columns:
                self.label_encoders[col] = LabelEncoder()
                df[f'{col}_enc'] = self.label_encoders[col].fit_transform(df[col].astype(str))
        
        # Interactions
        df['peak_weather'] = df['is_peak_hour'].astype(int) * df['weather_severity_index']
        df['rain_peak'] = df['rain_intensity'] * df['is_peak_hour'].astype(int)
        df['holiday_weather'] = df['is_holiday'].astype(int) * df['weather_severity_index']
        
        feature_cols = [
            'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos',
            'hour', 'day_of_week', 'month',
            'latitude', 'longitude',
            'rain_intensity', 'weather_severity_index',
            'precipitation_lag1', 'precipitation_lag3', 'precipitation_lag6',
            'event_attendance', 'event_max_rank',
            'holiday_intensity_score', 'peak_day_probability',
            'is_weekday', 'is_peak_hour', 'has_incident', 'is_rainy',
            'extreme_weather_flag', 'is_holiday', 'has_major_event',
            'has_sports_event', 'has_concert', 'has_conference',
            'road_name_enc', 'road_category_enc', 'peak_driver_type_enc', 'holiday_type_enc',
            'peak_weather', 'rain_peak', 'holiday_weather'
        ]
        
        feature_cols = [c for c in feature_cols if c in df.columns]
        logger.info(f"   {len(feature_cols)} features")
        
        X = df[feature_cols].fillna(0).values
        X = self.scaler.fit_transform(X)
        
        joblib.dump(self.scaler, self.weights_dir / 'scaler_singapore.pkl')
        joblib.dump(self.label_encoders, self.weights_dir / 'encoders_singapore.pkl')
        joblib.dump(feature_cols, self.weights_dir / 'features_singapore.pkl')
        
        return X, y, len(feature_cols)
    
    def train(self, epochs=150, batch_size=256, lr=0.001):
        X, y, num_features = self.load_data()
        
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15, random_state=42)
        logger.info(f"   Train: {len(X_train)}, Val: {len(X_val)}")
        
        train_loader = DataLoader(TrafficDataset(X_train, y_train, augment=True), batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(TrafficDataset(X_val, y_val), batch_size=batch_size)
        
        model = DeepTrafficModel(num_features, hidden=512, num_layers=8).to(self.device)
        logger.info(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        criterion = nn.SmoothL1Loss()  # Huber loss is more robust
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
        
        best_tol_acc = 0.0
        
        logger.info(f"\n{'='*60}")
        logger.info("🎯 Training for 95%+ Accuracy (±1 Band)")
        logger.info(f"{'='*60}\n")
        
        for epoch in range(epochs):
            model.train()
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                optimizer.zero_grad()
                pred = model(X_batch)
                loss = criterion(pred, y_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            scheduler.step()
            
            # Validation
            model.eval()
            all_preds, all_labels = [], []
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch = X_batch.to(self.device)
                    pred = model(X_batch)
                    all_preds.extend(pred.cpu().numpy())
                    all_labels.extend(y_batch.numpy())
            
            all_preds = np.array(all_preds)
            all_labels = np.array(all_labels)
            pred_bands = np.clip(np.round(all_preds), 1, 8)
            
            exact_acc = (pred_bands == all_labels).mean() * 100
            tol_acc = (np.abs(pred_bands - all_labels) <= 1).mean() * 100
            mae = np.abs(all_preds - all_labels).mean()
            
            if (epoch + 1) % 10 == 0 or tol_acc > best_tol_acc:
                logger.info(f"Epoch {epoch+1}: MAE={mae:.3f}, Exact={exact_acc:.1f}%, ±1Band={tol_acc:.1f}%")
            
            if tol_acc > best_tol_acc:
                best_tol_acc = tol_acc
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'exact_accuracy': exact_acc,
                    'tolerance_accuracy': tol_acc,
                    'mae': mae,
                    'num_features': num_features
                }, self.weights_dir / 'stgcn_singapore_v1.pth')
                
                if tol_acc >= 95:
                    logger.info(f"🎉 TARGET ACHIEVED: {tol_acc:.2f}%!")
            
            if tol_acc >= 98:
                break
        
        # Final results
        logger.info(f"\n{'='*60}")
        logger.info("📊 FINAL RESULTS")
        logger.info(f"   Best ±1 Band Accuracy: {best_tol_acc:.2f}%")
        
        # Per-band breakdown
        ckpt = torch.load(self.weights_dir / 'stgcn_singapore_v1.pth', weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        model.eval()
        
        all_preds, all_labels = [], []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(self.device)
                pred = model(X_batch)
                all_preds.extend(pred.cpu().numpy())
                all_labels.extend(y_batch.numpy())
        
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        pred_bands = np.clip(np.round(all_preds), 1, 8)
        
        logger.info("\n📈 Per-Band Accuracy:")
        for band in range(1, 9):
            mask = all_labels == band
            if mask.sum() > 0:
                exact = (pred_bands[mask] == band).mean() * 100
                tol = (np.abs(pred_bands[mask] - band) <= 1).mean() * 100
                logger.info(f"   Band {band}: Exact={exact:.1f}%, ±1={tol:.1f}% (n={mask.sum()})")
        
        if best_tol_acc >= 95:
            logger.info("\n✅ SUCCESS: 95%+ accuracy achieved!")
        else:
            logger.info(f"\n⚠️ Best accuracy: {best_tol_acc:.2f}%")
        
        logger.info(f"💾 Model saved to: {self.weights_dir / 'stgcn_singapore_v1.pth'}")
        logger.info(f"{'='*60}")
        
        return best_tol_acc


if __name__ == "__main__":
    import traceback
    try:
        trainer = OptimizedTrainer()
        accuracy = trainer.train(epochs=150, batch_size=256, lr=0.001)
        
        print(f"\n{'='*60}")
        print(f"🏁 TRAINING COMPLETE")
        print(f"   Final ±1 Band Accuracy: {accuracy:.2f}%")
        print(f"{'='*60}")
        
    except Exception as e:
        logger.error(f"Failed: {e}")
        traceback.print_exc()
