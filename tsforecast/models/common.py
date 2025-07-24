import torch
import torch.nn as nn
import math

class SelfAttention(nn.Module):
    """
    A minimal single-head self-attention layer.
    """
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.Wq = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.Wk = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.Wv = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.scale = math.sqrt(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch, seq_len, hidden_dim).
        
        Returns:
            torch.Tensor: Output tensor of the same shape.
        """
        Q = self.Wq(x)
        K = self.Wk(x)
        V = self.Wv(x)

        attn_scores = torch.bmm(Q, K.transpose(1, 2)) / self.scale
        attn_weights = torch.softmax(attn_scores, dim=-1)
        
        out = torch.bmm(attn_weights, V)
        return out

class PositionalEncoding(nn.Module):
    """
    Standard positional encoding for transformer models.
    """
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor of shape (seq_len, batch, d_model).
        
        Returns:
            torch.Tensor: Output tensor with positional information added.
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x) 