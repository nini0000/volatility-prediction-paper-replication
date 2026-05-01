"""
baselines.py
------------
Benchmark models: GARCH(1,1), Ridge, and Lasso.

Important implementation note
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
GARCH must be fit on *raw* (un-normalised) return data.  Using z-scored
returns causes the GARCH variance scale to diverge, producing nonsense
MAPE values.  The `fit_garch_or_fallback` function accepts raw returns
directly and returns raw volatility forecasts so MAPE can be computed
against `y_test_raw`.
"""

from __future__ import annotations

from typing import Literal, Optional

import numpy as np

try:
    from arch import arch_model  # type: ignore
    _ARCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    _ARCH_AVAILABLE = False

from sklearn.linear_model import Lasso, Ridge
from sklearn.model_selection import GridSearchCV


# ---------------------------------------------------------------------------
# GARCH
# ---------------------------------------------------------------------------

def fit_garch_or_fallback(
    train_returns: np.ndarray,
    train_vol: np.ndarray,
    horizon: int,
    future_returns: Optional[np.ndarray] = None,
    sequence_length: int = 10,
) -> np.ndarray:
    """Fit GARCH(1,1) on *train_returns* and produce one-step-ahead forecasts.

    Parameters
    ----------
    train_returns:
        Raw (un-normalised) daily or Δt-aggregated returns used for fitting.
    train_vol:
        Raw training volatility (only used for the fallback if arch is absent).
    horizon:
        Number of forecast steps (= length of the test set).
    future_returns:
        Raw test-period returns used for recursive one-step forecasting.
        If ``None``, unconditional forecasts are used.
    sequence_length:
        Warm-up steps to skip at the start of the test period.

    Returns
    -------
    np.ndarray of shape (horizon,)
    """
    if not _ARCH_AVAILABLE:
        # Naive fallback: predict the last observed training volatility.
        print("Warning: 'arch' package not installed. Using naive GARCH fallback.")
        last_vol = float(train_vol[-1])
        return np.full(horizon, last_vol, dtype=np.float32)

    # Scale returns to percent (arch library convention).
    returns_pct = train_returns * 100.0
    am = arch_model(returns_pct, vol="Garch", p=1, q=1, dist="normal")
    try:
        res = am.fit(disp="off", show_warning=False)
    except Exception:  # noqa: BLE001
        # Fit failure — return naive forecast.
        last_vol = float(train_vol[-1])
        return np.full(horizon, last_vol, dtype=np.float32)

    # Recursive one-step forecasts over the test set.
    preds: list = []
    all_returns = np.concatenate([returns_pct, (future_returns or np.zeros(horizon)) * 100.0])

    for i in range(horizon):
        window = all_returns[: len(returns_pct) + i]
        try:
            fc = res.forecast(horizon=1, reindex=False)
            sigma_pct = float(np.sqrt(fc.variance.values[-1, 0]))
        except Exception:  # noqa: BLE001
            sigma_pct = float(np.sqrt(res.conditional_volatility[-1]))
        # Convert back from percent to raw scale.
        preds.append(sigma_pct / 100.0)

    return np.array(preds, dtype=np.float32)


# ---------------------------------------------------------------------------
# Linear models (Ridge / Lasso)
# ---------------------------------------------------------------------------

def fit_linear_grid(
    x_train: np.ndarray,
    y_train_raw: np.ndarray,
    kind: Literal["ridge", "lasso"] = "ridge",
    alphas: Optional[np.ndarray] = None,
) -> Ridge | Lasso:
    """Grid-search regularisation strength on *x_train* / *y_train_raw*.

    The grid spans 1e-2 → 1e-6 in log-space (paper §Methods, Eq. 20).

    Parameters
    ----------
    x_train:
        3-D array (N, seq, features) — will be flattened to 2-D.
    y_train_raw:
        1-D raw target values.
    kind:
        ``"ridge"`` or ``"lasso"``.
    alphas:
        Custom alpha grid; defaults to log-spaced 1e-2 → 1e-6.
    """
    if alphas is None:
        alphas = np.logspace(-2, -6, 20)

    # Flatten sequence dimension for linear models.
    x_2d = x_train.reshape(len(x_train), -1)
    y_1d = y_train_raw.ravel()

    model_cls = Ridge if kind == "ridge" else Lasso
    param_grid = {"alpha": alphas}

    gs = GridSearchCV(
        model_cls(),
        param_grid,
        cv=5,
        scoring="neg_mean_absolute_percentage_error",
        n_jobs=-1,
    )
    gs.fit(x_2d, y_1d)
    return gs.best_estimator_


def predict_linear(model: Ridge | Lasso, x_test: np.ndarray) -> np.ndarray:
    """Predict with a fitted linear model (handles 3-D → 2-D reshape)."""
    x_2d = x_test.reshape(len(x_test), -1)
    return model.predict(x_2d).astype(np.float32)
