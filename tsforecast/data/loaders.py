import torch
import numpy as np
from torch.utils.data import Dataset

class SlidingWindowDataset(Dataset):
    """
    Creates a dataset of sliding windows over a time series.
    
    For standard forecasting models that predict the future based on the past.
    """
    def __init__(self, data: np.ndarray, past_len: int, future_len: int):
        """
        Args:
            data (np.ndarray): The time series data, shape (n_samples, n_features).
            past_len (int): The number of time steps in the input window.
            future_len (int): The number of time steps in the output (forecast) window.
        """
        self.data = torch.from_numpy(data).float()
        self.past_len = past_len
        self.future_len = future_len
        self.window_len = past_len + future_len

    def __len__(self):
        return len(self.data) - self.window_len + 1

    def __getitem__(self, idx):
        window = self.data[idx : idx + self.window_len]
        past = window[:self.past_len]
        future = window[self.past_len:]
        return past, future

class FewShotDataset(Dataset):
    """
    Creates a dataset for few-shot learning models.
    
    Each item consists of a query window and its corresponding support window,
    based on a pre-computed array of nearest neighbors.
    """
    def __init__(self, query_data: np.ndarray, support_data: np.ndarray, 
                 query_neighbors: np.ndarray, past_len: int, future_len: int):
        """
        Args:
            query_data (np.ndarray): The query time series data.
            support_data (np.ndarray): The support time series data.
            query_neighbors (np.ndarray): Array mapping each query index to a support index.
            past_len (int): The number of time steps in the past window.
            future_len (int): The number of time steps in the future window.
        """
        self.query_data = torch.from_numpy(query_data).float()
        self.support_data = torch.from_numpy(support_data).float()
        self.query_neighbors = query_neighbors
        self.past_len = past_len
        self.future_len = future_len
        
        # The total length of a single sample window (e.g., 96+96=192)
        self.window_len = past_len + future_len
        
        if self.query_data.shape[1] != self.window_len or self.support_data.shape[1] != self.window_len:
            raise ValueError(
                f"The sequence length of query ({self.query_data.shape[1]}) and "
                f"support ({self.support_data.shape[1]}) data must match the "
                f"total window length ({self.window_len})."
            )

    def __len__(self):
        return len(self.query_data)

    def __getitem__(self, idx):
        # Get the query window
        query_window = self.query_data[idx]
        query_past = query_window[:self.past_len]
        query_future = query_window[self.past_len:]
        
        # Get the corresponding support window
        support_idx = self.query_neighbors[idx]
        support_window = self.support_data[support_idx]
        support_past = support_window[:self.past_len]
        support_future = support_window[self.past_len:]
        
        return query_past, query_future, support_past, support_future 