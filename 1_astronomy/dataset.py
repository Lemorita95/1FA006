# Load libraries
import torch
from torch.utils.data import Dataset
import numpy as np

class CustomDataset(Dataset):
    '''
    object to represent a custom dataset \n\n
    
    initialize as CustomDataset(X, y) where:\n
    X: features\n
    y: labels\n\n

    apply normalization:\n
    X: logarithmic\n
    y: standarization
    '''
    def __init__(self, X_path, y_path):
        self.X = np.load(X_path).astype(np.float32)
        self.y = np.load(y_path).astype(np.float32)
        self.y = self.y[:, -4:-1] # Use second-to-last three labels

        # normalize x
        self.X = np.log(np.maximum(0.2, self.X))

        # Compute mean and std for normalization, save for denormalization
        self.y_mean = self.y.mean(axis=0)
        self.y_std = self.y.std(axis=0)

        # Normalize labels
        self.y = (self.y - self.y_mean) / self.y_std

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # return self.X[idx], self.y[idx]
        return torch.tensor(self.X[idx]), torch.tensor(self.y[idx])
    
    def to_device(self, device):
        '''
        create tensors from numpy array and move to device
        '''
        self.X = torch.tensor(self.X, dtype=torch.float32).to(device)
        self.y = torch.tensor(self.y, dtype=torch.float32).to(device)