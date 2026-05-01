# Deep Learning Stock Volatility with Google Domestic Trends — PyTorch Replication

A PyTorch replication of the paper:

> **Deep Learning Stock Volatility with Google Domestic Trends**  
> Ruoxuan Xiong, Eric P. Nichols, Yuan Shen (2016)  
> arXiv: [1512.04916](https://arxiv.org/abs/1512.04916)

The original paper trains an LSTM on S&P 500 price data combined with 25 Google Domestic Trends to forecast 3-day rolling volatility. This repo reproduces the core experiment in PyTorch and extends it with Elastic Net, Random Forest, XGBoost, and Transformer baselines.

---

## Results

| Model | RMSE | Test MAPE | Paper MAPE |
|-------|------|-----------|------------|
| LSTM0 (full features) | — | ~24–27% | 24.2% |
| LSTMr (top-6 features) | — | ~27–30% | 27.2% |
| GARCH | — | ~35% | 34.9% |
| Ridge | — | — | — |
| Lasso | — | — | — |

> Note: exact numbers vary slightly from the paper because the public CSV starts in 2006 rather than 2004, and a 70/30 train-test split is preserved throughout.

---

## Project Structure

```
.
├── README.md
├── requirements.txt
├── notebook.ipynb                        # Main replication notebook
├── src/
│   └── replicate_volatility/
│       ├── __init__.py
│       ├── data.py                       # Data loading, aggregation, normalization
│       ├── model.py                      # PyTorch LSTM (KerasReplicationLSTM)
│       ├── metrics.py                    # MAPE, RMSE helpers
│       └── baselines.py                  # GARCH, Ridge, Lasso
├── data/
│   └── README.md                         # Instructions for obtaining the CSVs
└── outputs/
    └── notebook/                         # Saved plots and metrics JSON
```

---

## Data

The notebook expects one CSV per Google Domestic Trend plus `SP500.csv`. The easiest way to obtain them is from the companion repository:

```
https://github.com/nini0000/volatility-prediction-paper-replication
```

Clone it into `external/stock-volatility-google-trends`, or copy the `trends/` folder into `data/trends/`. The notebook auto-detects both layouts.

See [`data/README.md`](data/README.md) for full instructions.

---

## Installation

```bash
# 1. Clone this repo
git clone https://github.com/nini0000/volatility-prediction-paper-replication
cd volatility-prediction-paper-replication

# 2. Create a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

Python 3.9+ is recommended.

---

## Running

### Jupyter notebook (recommended)

```bash
jupyter notebook notebook.ipynb
```

Update `PROJECT_ROOT` in the first cell to point to your local clone.

### As a Python package

```python
from pathlib import Path
from replicate_volatility.data import load_repo_data, prepare_data, make_sequences
from replicate_volatility.model import KerasReplicationLSTM

TRENDS_DIR = Path("data/trends")

raw_daily, feature_columns = load_repo_data(TRENDS_DIR)
prepared = prepare_data(raw_daily, feature_columns, delta_t=3, sequence_length=10)
```

---

## Extensions

Beyond the paper's replication, the notebook also compares:

- **Elastic Net** feature selection + LSTM
- **Random Forest** importance-based feature selection + LSTM (varying top-k)
- **XGBoost** direct regression baseline
- **Transformer** encoder (with positional encoding) as a drop-in replacement for the LSTM

---

## Citation

If you use this code, please also cite the original paper:

```bibtex
@article{xiong2015deep,
  title   = {Deep Learning Stock Volatility with Google Domestic Trends},
  author  = {Xiong, Ruoxuan and Nichols, Eric P. and Shen, Yuan},
  journal = {arXiv preprint arXiv:1512.04916},
  year    = {2015}
}
```

---

## License

MIT
