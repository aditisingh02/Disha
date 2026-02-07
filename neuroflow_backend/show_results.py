import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
from pathlib import Path
import sys
sys.path.insert(0, '.')

# Load checkpoint
weights_dir = Path('ml_models/weights')
ckpt = torch.load(weights_dir / 'stgcn_singapore_v1.pth', map_location='cpu', weights_only=False)

print("=" * 60)
print("🎯 SINGAPORE TRAFFIC MODEL - FINAL RESULTS")
print("=" * 60)
print(f"Training Epochs: {ckpt.get('epoch', 'N/A') + 1}")
print(f"Number of Features: {ckpt.get('num_features', 'N/A')}")
print()
print("📊 ACCURACY METRICS:")
print(f"   Mean Absolute Error: {ckpt.get('mae', 'N/A'):.3f} bands")
print(f"   Exact Band Accuracy: {ckpt.get('exact_accuracy', ckpt.get('accuracy', 'N/A')):.2f}%")
print(f"   ±1 Band Accuracy: {ckpt.get('tolerance_accuracy', 'N/A'):.2f}%")
print()

tolerance_acc = ckpt.get('tolerance_accuracy', 0)
if tolerance_acc >= 95:
    print("✅ TARGET ACHIEVED: 95%+ accuracy (±1 band tolerance)!")
else:
    print(f"⚠️ Current accuracy: {tolerance_acc:.2f}% (target: 95%)")
    
print("=" * 60)
