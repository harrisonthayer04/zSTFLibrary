import torch
import torch.nn as nn
from .base import BaseForecastModel
from .common import SelfAttention

class TransformerEmbedder(nn.Module):
    """Encodes a time series into a fixed-size embedding using a Transformer."""
    def __init__(self, input_dim: int, d_model: int, nhead: int, num_layers: int, ff_dim: int):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=ff_dim, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.final_proj = nn.Linear(d_model, d_model) # Project to a consistent embedding size

    def forward(self, x):
        x = self.input_proj(x)
        x = self.transformer_encoder(x)
        # Average pooling over the sequence length to get a fixed-size representation
        x = x.mean(dim=1)
        x = self.final_proj(x)
        return x

class SiameseLSTM(BaseForecastModel):
    """
    A Siamese network that uses a Transformer to embed query and support series,
    and an LSTM to make a forecast based on the difference.
    """
    def __init__(self, input_dim: int, output_dim: int, future_len: int,
                 d_model: int, nhead: int, num_encoder_layers: int, ff_dim: int,
                 lstm_hidden_dim: int, num_lstm_layers: int):
        super().__init__()
        # Storing all hparams
        self._hparams = {k: v for k, v in locals().items() if k not in ['self', '__class__']}

        self.embedder = TransformerEmbedder(
            input_dim=input_dim, d_model=d_model, nhead=nhead,
            num_layers=num_encoder_layers, ff_dim=ff_dim
        )
        
        # This part of the original siamese model is simplified. Instead of a complex
        # comparison network, we use the difference of embeddings directly.
        
        self.lstm_forecaster = nn.LSTM(
            input_size=input_dim,
            hidden_size=lstm_hidden_dim,
            num_layers=num_lstm_layers,
            batch_first=True
        )
        
        # The initial hidden state of the LSTM will be conditioned by the embedding difference.
        self.embedding_to_hidden = nn.Linear(d_model, num_lstm_layers * lstm_hidden_dim)
        self.embedding_to_cell = nn.Linear(d_model, num_lstm_layers * lstm_hidden_dim)

        self.attention = SelfAttention(lstm_hidden_dim)
        self.fc_out = nn.Linear(lstm_hidden_dim, output_dim)

    def forward(self, query_past, support_past, support_future):
        # 1. Get embeddings
        query_emb = self.embedder(query_past)
        support_emb = self.embedder(support_past)
        
        # 2. Calculate difference embedding
        diff_emb = query_emb - support_emb
        
        # 3. Condition LSTM hidden state
        h0 = self.embedding_to_hidden(diff_emb)
        c0 = self.embedding_to_cell(diff_emb)
        
        # Reshape to (num_layers, batch, hidden_dim)
        h0 = h0.reshape(self._hparams['num_lstm_layers'], query_past.size(0), self._hparams['lstm_hidden_dim'])
        c0 = c0.reshape(self._hparams['num_lstm_layers'], query_past.size(0), self._hparams['lstm_hidden_dim'])

        # 4. Forecast with LSTM
        lstm_out, _ = self.lstm_forecaster(support_future, (h0, c0))
        attn_out = self.attention(lstm_out)
        prediction = self.fc_out(attn_out)
        
        return prediction

    @property
    def hparams(self):
        return self._hparams 