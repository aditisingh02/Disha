
import numpy as np

class ManualScaler:
    def __init__(self):
        self.mean = None
        self.std = None
        
    def fit_transform(self, data):
        self.mean = np.mean(data, axis=0)
        self.std = np.std(data, axis=0) + 1e-8 # Avoid div by zero
        return (data - self.mean) / self.std

    def transform(self, data):
        if self.mean is None:
            return data
        return (data - self.mean) / self.std
