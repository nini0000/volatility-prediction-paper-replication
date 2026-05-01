# Data

This project uses S&P 500 daily OHLC prices and 25 Google Domestic Trend CSVs.

## Obtaining the CSVs

The easiest source is the companion GitHub repository:

```
https://github.com/philipperemy/stock-volatility-google-trends
```

### Option A — clone into `external/`

```bash
mkdir -p external
git clone --depth 1 \
    https://github.com/philipperemy/stock-volatility-google-trends.git \
    external/stock-volatility-google-trends
```

The notebook will automatically pick up `external/stock-volatility-google-trends/trends/`.

### Option B — copy CSVs into `data/trends/`

Copy all `*.csv` files from the cloned repo's `trends/` folder into this directory:

```
data/
└── trends/
    ├── SP500.csv
    ├── advert.csv
    ├── airtvl.csv
    └── ...
```

## Expected files

| File | Description |
|------|-------------|
| `SP500.csv` | Daily OHLC prices for the S&P 500 ETF |
| `advert.csv` | Google Domestic Trend — Advertising & Marketing |
| `airtvl.csv` | Air Travel |
| `autoby.csv` | Auto Buyers |
| `autofi.csv` | Auto Financing |
| `bizind.csv` | Business & Industrial |
| `bnkrpt.csv` | Bankruptcy |
| `comput.csv` | Computers & Electronics |
| `crcard.csv` | Credit Cards |
| `durble.csv` | Durable Goods |
| `educat.csv` | Education |
| `invest.csv` | Finance & Investing |
| `finpln.csv` | Financial Planning |
| `furntr.csv` | Furniture |
| `insur.csv` | Insurance |
| `jobs.csv` | Jobs |
| `luxury.csv` | Luxury Goods |
| `mobile.csv` | Mobile & Wireless |
| `mtge.csv` | Mortgage (mapped to `mrtge` in the paper) |
| `rlest.csv` | Real Estate |
| `rental.csv` | Rental |
| `shop.csv` | Shopping |
| `smallbiz.csv` | Small Business |
| `travel.csv` | Travel |

> The public CSV collection starts in 2006 rather than the 2004 start date used in the original paper. The notebook preserves the paper's 70/30 train/test split by fraction.
