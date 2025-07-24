import numpy as np
from sklearn.preprocessing import MinMaxScaler

class DataPreprocessor:
    """
    Handles feature-wise scaling for multivariate time series data.
    
    This preprocessor fits a separate MinMaxScaler to each feature (column)
    of the training data and can then transform or inverse-transform data
    with the same feature structure.
    """
    def __init__(self, feature_range=(0, 1)):
        """
        Initializes the preprocessor.
        
        Args:
            feature_range (tuple): The desired range for the scaled features,
                                   passed directly to MinMaxScaler.
        """
        self.feature_range = feature_range
        self.scalers_ = []
        self.is_fitted = False
        self.n_features_in_ = 0

    def fit(self, data: np.ndarray):
        """
        Fits the scalers to the training data.
        
        A separate scaler is fitted for each feature column.
        
        Args:
            data (np.ndarray): The training data, expected in a shape like
                               (n_samples, sequence_length, n_features).
        
        Returns:
            self: The fitted preprocessor instance.
        """
        if data.ndim != 3:
            raise ValueError(f"Expected 3D array, but got {data.ndim} dimensions.")
        
        self.n_features_in_ = data.shape[-1]
        self.scalers_ = [MinMaxScaler(self.feature_range) for _ in range(self.n_features_in_)]
        
        # Fit each scaler to its corresponding feature
        for i in range(self.n_features_in_):
            feature_data = data[..., i].reshape(-1, 1)
            self.scalers_[i].fit(feature_data)
            
        self.is_fitted = True
        return self

    def _check_and_reshape(self, data: np.ndarray):
        if not self.is_fitted:
            raise RuntimeError("Preprocessor must be fitted before transforming data.")
        if data.ndim != 3:
            raise ValueError(f"Expected 3D array for transform, but got {data.ndim} dimensions.")
        if data.shape[-1] != self.n_features_in_:
            raise ValueError(f"Data has {data.shape[-1]} features, but preprocessor was fitted on {self.n_features_in_}.")
        return data

    def transform(self, data: np.ndarray) -> np.ndarray:
        """
        Applies feature-wise scaling to the data.
        """
        data = self._check_and_reshape(data)
        scaled_data = np.zeros_like(data, dtype=np.float32)
        
        for i in range(self.n_features_in_):
            feature_data = data[..., i].reshape(-1, 1)
            scaled_feature = self.scalers_[i].transform(feature_data)
            scaled_data[..., i] = scaled_feature.reshape(data[..., i].shape)
            
        return scaled_data

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """
        Applies the inverse feature-wise scaling to the data.
        """
        data = self._check_and_reshape(data)
        original_data = np.zeros_like(data, dtype=np.float32)

        for i in range(self.n_features_in_):
            feature_data = data[..., i].reshape(-1, 1)
            original_feature = self.scalers_[i].inverse_transform(feature_data)
            original_data[..., i] = original_feature.reshape(data[..., i].shape)

        return original_data 