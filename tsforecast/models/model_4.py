import torch
import torch.nn as nn
from .base import BaseForecastModel
from .common import SelfAttention

def _downsample(x, factor):
    """Average pools a tensor of shape (B, T, D) by a given factor."""
    return nn.functional.avg_pool1d(x.transpose(1, 2), kernel_size=factor).transpose(1, 2)

class MultiResolutionTransformer(nn.Module):
    """
    A Transformer-based embedder that processes a time series at multiple
    resolutions (scales) and combines the results.
    """
    def __init__(self, input_dim: int, d_model: int, nhead: int, ff_dim: int, num_layers: int, output_dim: int):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        
        # Create four separate transformer encoders for different resolutions
        self.t1 = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model, nhead, ff_dim, batch_first=True), num_layers)
        self.t2 = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model, nhead, ff_dim, batch_first=True), num_layers)
        self.t3 = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model, nhead, ff_dim, batch_first=True), num_layers)
        self.t4 = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model, nhead, ff_dim, batch_first=True), num_layers)
        
        self.final_proj = nn.Linear(d_model * 4, output_dim)

    def forward(self, x):
        x_proj = self.input_proj(x)
        
        # Process at original and downsampled resolutions
        out1 = self.t1(x_proj)
        out2 = self.t2(_downsample(x_proj, 2))
        out3 = self.t3(_downsample(x_proj, 4))
        out4 = self.t4(_downsample(x_proj, 8))
        
        # Average pool over time and concatenate
        out1 = out1.mean(dim=1)
        out2 = out2.mean(dim=1)
        out3 = out3.mean(dim=1)
        out4 = out4.mean(dim=1)
        
        concatenated = torch.cat([out1, out2, out3, out4], dim=-1)
        return self.final_proj(concatenated)

class SiameseMultiResLSTM(BaseForecastModel):
    """
    A full Siamese model using the Multi-Resolution Transformer for embedding
    and an LSTM for forecasting.
    """
    def __init__(self, input_dim: int, output_dim: int, future_len: int,
                 d_model: int, nhead: int, num_encoder_layers: int, ff_dim: int,
                 lstm_hidden_dim: int, num_lstm_layers: int):
        super().__init__()
        self._hparams = {k: v for k, v in locals().items() if k not in ['self', '__class__']}
        
        # The embedding dimension from the Siamese part must match what the
        # forecaster expects. We'll use a `diff_dim` parameter.
        diff_dim = 32 

        self.embedder = MultiResolutionTransformer(
            input_dim=input_dim, d_model=d_model, nhead=nhead, ff_dim=ff_dim,
            num_layers=num_encoder_layers, output_dim=diff_dim
        )

        self.forecaster_lstm = nn.LSTM(
            input_size=input_dim + diff_dim,
            hidden_size=lstm_hidden_dim,
            num_layers=num_lstm_layers,
            batch_first=True
        )

        self.attention = SelfAttention(lstm_hidden_dim)
        self.fc_out = nn.Linear(lstm_hidden_dim, output_dim)

    def forward(self, query_past, support_past, support_future):
        # 1. Get difference embedding
        query_emb = self.embedder(query_past)
        support_emb = self.embedder(support_past)
        diff_emb = query_emb - support_emb
        
        # 2. Prepare for LSTM
        # Expand diff_emb to match the sequence length of the future window
        future_len = support_future.size(1)
        expanded_diff = diff_emb.unsqueeze(1).expand(-1, future_len, -1)
        
        # 3. Combine with support_future
        lstm_input = torch.cat([support_future, expanded_diff], dim=-1)
        
        # 4. Forecast
        lstm_out, _ = self.forecaster_lstm(lstm_input)
        attn_out = self.attention(lstm_out)
        prediction = self.fc_out(attn_out)
        
        return prediction

    @property
    def hparams(self):
        return self._hparams 