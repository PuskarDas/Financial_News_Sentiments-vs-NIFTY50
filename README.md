# Financial News Sentiment vs. NIFTY50 Analysis

A data science project investigating whether the tone of financial news headlines can explain or predict movement in India's NIFTY50 stock index — built as a homework project for the KIIT VISTA Data Science Bootcamp.

Reference project: [eklavyakiit/Financial-News-Sentiment-vs-NIFTY50-Analysis](https://github.com/eklavyakiit/Financial-News-Sentiment-vs-NIFTY50-Analysis)

## Overview

The project is split into two parts:

- **Part A — Rule-Based Approach:** Cleans and merges two real datasets (3,000+ financial news headlines and NIFTY50 OHLC data, Feb–Aug 2025), scores each day's headlines using a hand-built financial sentiment vocabulary (223 positive / 247 negative terms), and tests a simple same-day prediction rule against a majority-class baseline.
- **Part B — Machine Learning Pipeline:** Fixes the same-day framing (Part A's target and features come from the same trading day, so a good score there just reflects co-movement, not prediction) by shifting the target to predict the *next* day's direction. Adds engineered features (rolling sentiment, lagged price action, volume/volatility), uses a chronological train/test split, and trains and tunes Logistic Regression, Random Forest, and XGBoost models. Includes a backtest against buy-and-hold and a feature importance analysis.

## Key Results

| Metric | Value |
|---|---|
| Part A same-day accuracy | 56.82% |
| Part A majority baseline | 50.76% |
| Part B next-day majority baseline (test) | 56.0% |
| Part B Logistic Regression accuracy | 56.0% |
| Part B Tuned Random Forest accuracy | 44.0% |
| Top feature (Random Forest importance) | `high_low_spread` (price volatility) |

**Honest takeaway:** sentiment is a weak but non-zero signal in the same-day setting, largely explained by news and price moves sharing a common cause. Once the target is shifted to next-day prediction, price-history and volatility features outperform sentiment features, and the tuned model does not beat a simple majority-class guess — a legitimate finding for a small (~125-row) dataset, not a failure of the approach.

## Tech Stack

- Python, Pandas
- Matplotlib, Seaborn
- scikit-learn (Logistic Regression, Random Forest, GridSearchCV, TimeSeriesSplit)
- XGBoost
- JupyterLab / Google Colab

## Repository Structure

```
├── Financial_News_Sentiment_vs_NIFTY50.ipynb   # Main notebook (Part A + Part B, run top to bottom)
├── data/
│   ├── news.csv                                 # Raw financial news headlines
│   └── nifty50.csv                               # Raw NIFTY50 OHLC data
├── plots/                                        # Saved charts (generated when notebook runs)
└── README.md
```

## How to Run

### Option 1 — Google Colab (recommended)
1. Upload `Financial_News_Sentiment_vs_NIFTY50.ipynb` to [Google Colab](https://colab.research.google.com)
2. Upload `data/news.csv` and `data/nifty50.csv` into the Colab session (folder icon → upload)
3. Run all cells top to bottom

### Option 2 — Local Jupyter
```bash
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>
pip install pandas matplotlib seaborn scikit-learn xgboost jupyter
jupyter notebook Financial_News_Sentiment_vs_NIFTY50.ipynb
```

## Methodology Notes

- **Sentiment scoring** is an unbounded raw count (positive matches minus negative matches per day), not normalized — so higher-volume news days naturally skew toward larger-magnitude scores.
- **Train/test split** in Part B is strictly chronological (no shuffling), since shuffling a time series leaks future information into training.
- **Backtest** in Part B is illustrative only — it ignores transaction costs, slippage, and taxes, and uses a small test window. It is not a trading signal.

## Limitations

- Small dataset (132 matched trading days, 125 after feature engineering) means Part B's test set is only ~25 days — results are sensitive to the exact train/test split.
- Same-day sentiment/price correlation in Part A does not imply predictive power, only co-movement.
- Rule-based sentiment vocabulary is hand-built and English-only; it does not use any pretrained NLP model.

## Author

Puskar — B.Tech CSE, KIIT University

## Acknowledgements

Built by following the guidebook and reference notebook published by the Project Wing, KIIT VISTA (Machine Intelligence & Computer Vision Society).
