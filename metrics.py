"""
metrics.py
----------
MAPE and RMSE helpers — both NumPy and PyTorch variants.
"""

from __future__ import annotations

import numpy as np
import torch


# ---------------------------------------------------------------------------
# NumPy helpers (used at evaluation time)
# ---------------------------------------------------------------------------

def mape_np(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float:
    """Mean Absolute Percentage Error (%).

    Matches the loss function used in the paper and the Keras replication.
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    return float(np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + eps))) * 100.0)


def rmse_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error."""
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


# ---------------------------------------------------------------------------
# PyTorch helpers (used during training)
# ---------------------------------------------------------------------------

def torch_mape(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Differentiable MAPE loss for PyTorch training."""
    return torch.mean(torch.abs((y_true - y_pred) / (torch.abs(y_true) + eps)))
