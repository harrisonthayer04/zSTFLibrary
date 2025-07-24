import torch
import torch.nn as nn
from .base import BaseForecastModel
from .common import SelfAttention

class LSTMAttention(BaseForecastModel):
    """
    A few-shot forecasting model using an LSTM with self-attention.
    
    This model is conditioned on the difference between a 'query' time series
    and a 'support' time series from a library of examples.
    """
    def __init__(self, input_dim: int, output_dim: int, past_len: int,
                 hidden_dim: int = 64, num_layers: int = 1):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.past_len = past_len
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Encoder for the difference vector between query and support pasts
        self.diff_encoder = nn.Linear(past_len * input_dim, hidden_dim)
        
        # Core LSTM for processing the support future sequence
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        
        # Attention mechanism to weigh the LSTM outputs
        self.attention = SelfAttention(hidden_dim)
        
        # Final output layer
        self.fc_out = nn.Linear(hidden_dim, output_dim)

    def forward(self, query_past, support_past, support_future):
        """
        Args:
            query_past (torch.Tensor): Past window of the query series.
                                       Shape: (batch, past_len, input_dim).
            support_past (torch.Tensor): Past window of the support series.
                                        Shape: (batch, past_len, input_dim).
            support_future (torch.Tensor): Future window of the support series.
                                           Shape: (batch, future_len, input_dim).
        
        Returns:
            torch.Tensor: The forecast for the query series' future.
                          Shape: (batch, future_len, output_dim).
        """
        batch_size = query_past.size(0)
        
        # Flatten and compute difference vector
        diff_vec = query_past.reshape(batch_size, -1) - support_past.reshape(batch_size, -1)
        
        # Encode difference to initialize LSTM hidden state
        h0 = self.diff_encoder(diff_vec)
        c0 = torch.zeros_like(h0)
        
        # Reshape hidden states for LSTM (num_layers, batch, hidden_dim)
        h0 = h0.unsqueeze(0).repeat(self.num_layers, 1, 1)
        c0 = c0.unsqueeze(0).repeat(self.num_layers, 1, 1)
        
        # Process support future, conditioned on the difference
        lstm_out, _ = self.lstm(support_future, (h0, c0))
        
        # Apply attention
        attn_out = self.attention(lstm_out)
        
        # Final prediction
        prediction = self.fc_out(attn_out)
        
        return prediction

    @property
    def hparams(self):
        return {
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "past_len": self.past_len,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers
        } 