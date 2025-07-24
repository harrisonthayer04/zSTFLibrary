import torch
import torch.nn as nn
from .base import BaseForecastModel
from .common import PositionalEncoding

class TransformerForecaster(BaseForecastModel):
    """
    A standard Transformer-based forecasting model.
    
    This model uses a Transformer Encoder to process a sequence of past
    values and predict a sequence of future values. It does not use the
    few-shot learning paradigm.
    """
    def __init__(self, input_dim: int, output_dim: int, past_len: int, future_len: int,
                 d_model: int, nhead: int, num_encoder_layers: int, ff_dim: int, 
                 dropout: float = 0.1):
        super().__init__()
        self._hparams = {k: v for k, v in locals().items() if k not in ['self', '__class__']}

        self.past_len = past_len
        self.future_len = future_len
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=ff_dim, 
            dropout=dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        
        # Use a linear layer to project encoded past to future predictions
        self.future_proj = nn.Linear(past_len * d_model, future_len * d_model)
        self.output_proj = nn.Linear(d_model, output_dim)

    def forward(self, x_past):
        """
        Args:
            x_past (torch.Tensor): The historical time series data.
                                   Shape: (batch, past_len, input_dim).
        
        Returns:
            torch.Tensor: The forecast for the future.
                          Shape: (batch, future_len, output_dim).
        """
        batch_size = x_past.size(0)
        
        # Project input to d_model, add positional encoding
        x = self.input_proj(x_past)
        x = self.pos_encoder(x)
        
        # Pass through transformer encoder
        encoded = self.transformer_encoder(x)  # (batch, past_len, d_model)
        
        # Flatten and project to future length
        encoded_flat = encoded.reshape(batch_size, -1)  # (batch, past_len * d_model)
        future_flat = self.future_proj(encoded_flat)  # (batch, future_len * d_model)
        future_encoded = future_flat.reshape(batch_size, self.future_len, -1)  # (batch, future_len, d_model)
        
        # Project to output dimension
        prediction = self.output_proj(future_encoded)  # (batch, future_len, output_dim)
        
        return prediction

    @property
    def hparams(self):
        return self._hparams 