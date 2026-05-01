"""
model.py
--------
PyTorch LSTM that mirrors the architecture used in the original Keras
replication of Xiong et al. (2016).

Architecture
~~~~~~~~~~~~
  Input → LSTM(hidden_size=32) → Dense(dense_size=16, sigmoid) → Linear(1)

This matches the public Keras replication at
  https://github.com/philipperemy/stock-volatility-google-trends
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn


class KerasReplicationLSTM(nn.Module):
    """Single-layer LSTM followed by a sigmoid dense layer and a linear head.

    Parameters
    ----------
    input_size:
        Number of input features (25 for LSTM0, 6 for LSTMr).
    hidden_size:
        LSTM hidden dimension (default: 32, matching the Keras replication).
    dense_size:
        Size of the intermediate dense layer (default: 16).
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 32,
        dense_size: int = 16,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
        )
        self.dense = nn.Linear(hidden_size, dense_size)
        self.activation = nn.Sigmoid()
        self.head = nn.Linear(dense_size, 1)

        self._init_weights()

    def _init_weights(self) -> None:
        """Small positive constant initialisation (paper §Methods)."""
        for name, param in self.named_parameters():
            if "weight" in name:
                nn.init.constant_(param, 0.01)
            elif "bias" in name:
                nn.init.constant_(param, 0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (batch, seq_len, input_size)

        Returns
        -------
        (batch, 1)
        """
        out, _ = self.lstm(x)          # (batch, seq, hidden)
        last = out[:, -1, :]           # take last time-step
        h = self.activation(self.dense(last))
        return self.head(h)


# ---------------------------------------------------------------------------
# Transformer baseline (extension section of the notebook)
# ---------------------------------------------------------------------------
class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding (Vaswani et al., 2017)."""

    def __init__(self, d_model: int, max_len: int = 500) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1), :]


class TransformerVolatility(nn.Module):
    """Transformer encoder used as an extension baseline.

    Parameters
    ----------
    input_size : int
    d_model : int
        Embedding dimension (default 64).
    nhead : int
        Number of attention heads (default 4).
    num_layers : int
        Number of encoder layers (default 2).
    dropout : float
    """

    def __init__(
        self,
        input_size: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(input_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            batch_first=True,
            dropout=dropout,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        x = self.transformer(x)
        return self.fc(x[:, -1, :])
