# Load libraries
import torch
from torch.utils.data import Dataset
import numpy as np

from helpers import normalize

class CustomDataset(Dataset):
    """
    A custom PyTorch Dataset for loading and preprocessing data.

    Args:
        X_path (str): Path to the .npy file containing the input features.
        y_path (str): Path to the .npy file containing the target labels.

    Attributes:
        X (numpy.ndarray): Preprocessed input features.
        y (numpy.ndarray): Preprocessed target labels (second-to-last three labels).
        y_mean (numpy.ndarray): Mean of the target labels used for normalization.
        y_std (numpy.ndarray): Standard deviation of the target labels used for normalization.

    Methods:
        __len__(): Returns the number of samples in the dataset.
        __getitem__(idx): Returns the input features and target labels for a given index.

    Notes:
        - Input features are log-transformed with a minimum value of 0.2 to avoid log(0).
        - Target labels are normalized using the `normalize` function from helpers.py
    """

    def __init__(self, X_path, y_path):
        self.X = np.load(X_path)
        self.y = np.load(y_path)
        self.name_labels = ["mass", "age", "l_bol", "dist", "t_eff", "log_g", "fe_h", "SNR"]
        
        # Use second-to-last three labels
        self.y = self.y[:, -4:-1] 

        # store number of labels and names
        self.num_labels = self.y.shape[1]
        self.name_labels = self.name_labels[-4:-1]

        # normalize features
        self.X = np.log(np.maximum(0.2, self.X))

        # normalize labels
        self.y, self.y_mean, self.y_std = normalize(self.y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx], dtype=torch.float32), torch.tensor(self.y[idx], dtype=torch.float32)
