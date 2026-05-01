"""
data.py
-------
Data loading, aggregation, normalization, and mutual-information
utilities for the stock volatility replication.

Paper reference:
  Xiong, Nichols, Shen (2016) — "Deep Learning Stock Volatility
  with Google Domestic Trends"  arXiv:1512.04916

Key design decisions
~~~~~~~~~~~~~~~~~~~~
* `delta_t=3, k=inf` is the optimal scheme identified in the paper
  (Fig. 2 / Eq. 11).  `k=inf` means z-score is computed over the
  entire *training* set, which is what `prepare_data` does when
  `norm_window` is left as `None`.
* The public CSV from philipperemy/stock-volatility-google-trends
  starts in 2006 rather than 2004, so we use a 70/30 *fraction*
  split rather than the original absolute dates.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_regression

# ---------------------------------------------------------------------------
# Paper Table 1 — 25 Google Domestic Trends
# ---------------------------------------------------------------------------
DOMESTIC_TRENDS: List[str] = [
    "advert", "airtvl", "autoby", "autofi", "bizind", "bnkrpt",
    "comput", "crcard", "durble", "educat", "invest", "finpln",
    "furntr", "insur", "jobs", "luxury", "mobile", "mrtge",
    "rlest", "rental", "shop", "smallbiz", "travel",
]

# The public repo uses "mtge.csv" for the mortgage trend.
_ALIAS = {"mtge": "mrtge"}


# ---------------------------------------------------------------------------
# Volatility estimator — Garman-Klass (paper Eq. 2)
# ---------------------------------------------------------------------------
def garman_klass_volatility(
    high: np.ndarray,
    low: np.ndarray,
    open_: np.ndarray,
    close: np.ndarray,
) -> np.ndarray:
    """Per-day Garman-Klass volatility (paper Eq. 1–2)."""
    u = np.log(high / open_)
    d = np.log(low / open_)
    c = np.log(close / open_)
    return np.sqrt(0.511 * (u - d) ** 2 - 0.019 * (c * (u + d) - 2 * u * d) - 0.383 * c ** 2)


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------
def _read_trend_csv(path: Path, name: str) -> pd.Series:
    """Read a single Google Domestic Trend CSV and return a named daily series."""
    df = pd.read_csv(path, index_col=0, parse_dates=True, encoding="utf-8-sig")
    df.index = pd.to_datetime(df.index, errors="coerce")
    df = df[~df.index.isna()].sort_index()
    # Drop any non-numeric header rows
    col = df.columns[0]
    s = pd.to_numeric(df[col], errors="coerce").dropna()
    s.name = name
    return s


def load_repo_data(
    trends_dir: Path,
    include_extra_trends: bool = False,
) -> Tuple[pd.DataFrame, List[str]]:
    """Load S&P 500 prices + Google Domestic Trends from *trends_dir*.

    Parameters
    ----------
    trends_dir:
        Directory that contains ``SP500.csv`` and one CSV per trend.
    include_extra_trends:
        If ``True``, include any extra trend CSVs found beyond the 23
        listed in the paper.

    Returns
    -------
    raw_daily : pd.DataFrame
        Daily merged data with columns ``return``, ``volatility``, and
        one column per trend.
    feature_columns : list[str]
        Column names that should be used as model inputs.
    """
    trends_dir = Path(trends_dir)

    # ---- S&P 500 ----
    sp500 = pd.read_csv(
        trends_dir / "SP500.csv",
        index_col=0,
        parse_dates=True,
        encoding="utf-8-sig",
    )
    sp500.index = pd.to_datetime(sp500.index, errors="coerce")
    sp500 = sp500[~sp500.index.isna()].sort_index()
    sp500.columns = [c.strip().lower() for c in sp500.columns]

    # Daily log return
    adj_close = pd.to_numeric(sp500["adj close"], errors="coerce")
    daily_return = np.log(adj_close / adj_close.shift(1))

    # Daily GK volatility
    daily_vol = garman_klass_volatility(
        pd.to_numeric(sp500["high"], errors="coerce").values,
        pd.to_numeric(sp500["low"], errors="coerce").values,
        pd.to_numeric(sp500["open"], errors="coerce").values,
        pd.to_numeric(sp500["close"], errors="coerce").values,
    )
    daily_vol = pd.Series(daily_vol, index=sp500.index, name="volatility")

    merged = pd.DataFrame({"return": daily_return, "volatility": daily_vol})

    # ---- Trends ----
    found_trends: List[str] = []
    for csv_path in sorted(trends_dir.glob("*.csv")):
        stem = csv_path.stem.lower()
        if stem == "sp500":
            continue
        canonical = _ALIAS.get(stem, stem)
        if not include_extra_trends and canonical not in DOMESTIC_TRENDS:
            continue
        try:
            s = _read_trend_csv(csv_path, canonical)
            merged = merged.join(s, how="left")
            found_trends.append(canonical)
        except Exception as exc:  # noqa: BLE001
            print(f"  Warning: could not load {csv_path.name}: {exc}")

    merged = merged.dropna(how="any").sort_index()
    feature_columns = ["return", "volatility"] + found_trends
    return merged, feature_columns


# ---------------------------------------------------------------------------
# Aggregation helpers (paper Eq. 4–6)
# ---------------------------------------------------------------------------
def _aggregate(df: pd.DataFrame, delta_t: int, feature_columns: List[str]) -> pd.DataFrame:
    """Aggregate daily data to *delta_t*-day intervals (paper §Data Sources)."""
    rows = []
    n = len(df)
    for start in range(0, n - delta_t + 1, delta_t):
        block = df.iloc[start : start + delta_t]
        row: dict = {}
        row["return"] = block["return"].sum()  # Eq. 4
        # Eq. 6: per-day quadratic variation
        row["volatility"] = np.sqrt((block["volatility"] ** 2).sum())
        for col in feature_columns:
            if col not in ("return", "volatility"):
                row[col] = block[col].mean()  # Eq. 5
        row["date"] = block.index[-1]
        rows.append(row)
    agg = pd.DataFrame(rows).set_index("date")
    return agg


# ---------------------------------------------------------------------------
# PreparedData dataclass
# ---------------------------------------------------------------------------
@dataclass
class PreparedData:
    train: pd.DataFrame
    test: pd.DataFrame
    feature_columns: List[str]
    target_column: str
    target_mean: float
    target_std: float


# ---------------------------------------------------------------------------
# prepare_data — main entry point
# ---------------------------------------------------------------------------
def prepare_data(
    raw_daily: pd.DataFrame,
    feature_columns: List[str],
    delta_t: int = 3,
    sequence_length: int = 10,
    train_fraction: float = 0.70,
    norm_window: Optional[int] = None,
) -> PreparedData:
    """Aggregate, split, and z-score the data.

    Parameters
    ----------
    raw_daily:
        Output of ``load_repo_data``.
    feature_columns:
        Columns to keep as model inputs.
    delta_t:
        Observation interval in days (paper: 3).
    sequence_length:
        Sequence length used by the LSTM; only used to know how many
        samples to discard from the front of the test set (warm-up).
    train_fraction:
        Fraction of *aggregated* rows used for training.
    norm_window:
        Rolling window size *k*.  ``None`` / ``0`` uses the full
        training set (k=∞ in the paper).

    Returns
    -------
    PreparedData
        Normalised train / test DataFrames plus statistics needed to
        invert the target normalisation at evaluation time.
    """
    # ---- Aggregate ----
    agg = _aggregate(raw_daily, delta_t, feature_columns)

    # ---- Create forward target: next-period volatility ----
    agg = agg.copy()
    agg["target_raw"] = agg["volatility"].shift(-1)
    agg["return_raw"] = agg["return"]
    agg = agg.dropna()

    # ---- Train / test split ----
    split_idx = int(len(agg) * train_fraction)
    train_raw = agg.iloc[:split_idx].copy()
    test_raw = agg.iloc[split_idx:].copy()

    # ---- Z-score normalisation (k=∞ → statistics from training set) ----
    cols_to_norm = feature_columns + ["target_raw"]
    stats: dict = {}

    def _zscore(series: pd.Series, mean: float, std: float) -> pd.Series:
        return (series - mean) / (std if std > 0 else 1.0)

    for col in cols_to_norm:
        if norm_window and norm_window > 0:
            mean = train_raw[col].rolling(norm_window).mean()
            std = train_raw[col].rolling(norm_window).std().replace(0, np.nan)
            train_raw[col] = (train_raw[col] - mean) / std
        else:
            m = float(train_raw[col].mean())
            s = float(train_raw[col].std())
            stats[col] = (m, s)
            train_raw[col] = _zscore(train_raw[col], m, s)
            test_raw[col] = _zscore(test_raw[col], m, s)

    train_raw = train_raw.dropna()
    test_raw = test_raw.dropna()

    target_mean, target_std = stats.get("target_raw", (0.0, 1.0))

    return PreparedData(
        train=train_raw,
        test=test_raw,
        feature_columns=feature_columns,
        target_column="target_raw",
        target_mean=target_mean,
        target_std=target_std,
    )


# ---------------------------------------------------------------------------
# Sequence builder
# ---------------------------------------------------------------------------
def make_sequences(
    df: pd.DataFrame,
    feature_columns: List[str],
    target_column: str,
    sequence_length: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, pd.DatetimeIndex]:
    """Build (X, y_z, y_raw, dates) arrays for the LSTM.

    Returns
    -------
    x : float32 array of shape (N, sequence_length, n_features)
    y_z : float32 array of shape (N, 1)   — z-scored target
    y_raw : float32 array of shape (N, 1) — raw target (for MAPE)
    dates : DatetimeIndex of length N
    """
    X_arr = df[feature_columns].values.astype(np.float32)
    y_arr = df[target_column].values.astype(np.float32)
    # raw target stored in "target_raw" column (same as target_column here)
    y_raw_arr = y_arr  # both are z-scored; caller inverts with target_mean/std

    xs, ys, ys_raw, idxs = [], [], [], []
    for i in range(sequence_length, len(df)):
        xs.append(X_arr[i - sequence_length : i])
        ys.append(y_arr[i])
        ys_raw.append(y_raw_arr[i])
        idxs.append(df.index[i])

    return (
        np.array(xs, dtype=np.float32),
        np.array(ys, dtype=np.float32).reshape(-1, 1),
        np.array(ys_raw, dtype=np.float32).reshape(-1, 1),
        pd.DatetimeIndex(idxs),
    )


# ---------------------------------------------------------------------------
# Mutual information helper
# ---------------------------------------------------------------------------
def mutual_information_by_feature(
    prepared: PreparedData,
    n_neighbors: int = 5,
) -> pd.Series:
    """Compute MI between each input feature and the next-period volatility.

    Mirrors Fig. 3 of the paper (Eq. 9).
    """
    X = prepared.train[prepared.feature_columns]
    y = prepared.train[prepared.target_column]

    mi = mutual_info_regression(X, y, n_neighbors=n_neighbors, random_state=0)
    return pd.Series(mi, index=prepared.feature_columns).sort_values(ascending=False)
