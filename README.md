# 📈 Statistical Arbitrage on Indian Banking Equities

> Cointegration, Volatility Modeling, and Spread Prediction using Classical Econometrics + Machine Learning

**Course:** Analysis and Forecasting of Time Series  
**Group:** 6 — MSc Data Science, Div B  
**Team:** Devika Jonjale (B045) · Saachi Shinde (B048) · Manjiri Apshinge (B061)

---

## 🧠 Overview

This project implements a **pair trading strategy** on two major Indian private-sector banks listed on the NSE:

- **HDFC Bank** (`HDFCBANK.NS`)
- **Kotak Mahindra Bank** (`KOTAKBANK.NS`)

Pair trading is a market-neutral strategy that bets on the mean-reversion of the price spread between two cointegrated stocks. This project extends the classical framework by integrating GARCH volatility modeling, a Kalman filter for dynamic hedge ratios, and machine learning models (XGBoost and LSTM) for spread direction prediction.

---

## 📦 Tech Stack

| Category | Libraries |
|---|---|
| Data | `yfinance`, `pandas`, `numpy` |
| Statistical Testing | `statsmodels` (ADF, KPSS, Engle-Granger) |
| Volatility Modeling | `arch` (GARCH) |
| Dynamic Hedge Ratio | `pykalman` |
| Forecasting | `pmdarima` (auto ARIMA), `statsmodels` |
| Machine Learning | `xgboost`, `scikit-learn` |
| Deep Learning | `tensorflow` / `keras` (LSTM) |
| Visualization | `matplotlib`, `seaborn` |

---

## 🗂️ Project Pipeline

```
Raw Price Data
     │
     ▼
Data Collection (yfinance, 2016–present)
     │
     ▼
Preprocessing (log transformation)
     │
     ▼
Cointegration Testing (Engle-Granger, ADF, KPSS)
     │
     ▼
Spread Construction (OLS hedge ratio)
     │
     ├──► Half-life Estimation (Ornstein-Uhlenbeck)
     ├──► Volatility Modeling (Rolling GARCH(1,1))
     └──► Dynamic Hedge Ratio (Kalman Filter)
               │
               ▼
         Z-Score & Trading Signals (±2 thresholds)
               │
               ├──► ARIMA Spread Forecasting (rolling 1-step-ahead)
               ├──► XGBoost Direction Classifier
               └──► LSTM Spread Predictor
                         │
                         ▼
              Backtesting (Walk-forward, TimeSeriesSplit)
                         │
                         ▼
              Risk Management (Sharpe, Win Rate, Max Drawdown)
```

---

## 🔬 Methodology

### 1. Data Collection & Preprocessing
Daily adjusted closing prices downloaded from Yahoo Finance from **January 2016 to present** using `yfinance`. Log transformation applied to stabilize variance and prepare for cointegration analysis.

### 2. Cointegration Testing
The **Engle-Granger two-step test** confirmed strong cointegration between the pair (p-value = 0.00077, well below 0.05), validating the pair for mean-reversion trading.

### 3. Hedge Ratio & Spread Construction
- **Static hedge ratio (β = 1.1047)** estimated via OLS regression of log(HDFC) on log(Kotak)
- Spread: `spread = log(HDFC) - β × log(Kotak)`
- **Half-life ≈ 42 trading days** estimated from an AR(1) regression (Ornstein-Uhlenbeck formula)

### 4. Stationarity Tests
| Test | Result | Interpretation |
|---|---|---|
| ADF | p < 0.001 | Spread is mean-reverting ✅ |
| KPSS | p = 0.01 | Trend stationarity (slowly drifting mean) |

### 5. GARCH(1,1) Volatility Modeling
Rolling GARCH(1,1) applied to spread returns using a **100-day window**, producing leak-free one-step-ahead volatility forecasts. Average volatility: **1.465**, with clear clustering around the COVID-19 shock in 2020.

### 6. Kalman Filter — Dynamic Hedge Ratio
A state-space model tracks the time-varying hedge ratio, ranging between **0.9 and 1.15** over the sample period. Adapts to structural shifts (e.g., the HDFC–HDFC Ltd merger period) where the static OLS estimate would be misleading.

### 7. Z-Score & Trading Signals
Rolling z-score with a **41-day lookback window** (= half-life):

| Z-score | Signal |
|---|---|
| z > +2 | Short the spread |
| z < −2 | Long the spread |
| z ≈ 0 | No action / close position |

### 8. ARIMA Spread Forecasting
- Best model selected by AIC: **ARIMA(0,1,1)**
- Rolling one-step-ahead evaluation on 20% test set (508 predictions)
- **RMSE: 0.015950**
- Residuals normally distributed around zero

### 9. XGBoost Classification
Predicts next-day spread **direction** (up/down) using 8 engineered features:
`spread`, `z_score`, `volatility`, `lag1`, `lag2`, `rolling_mean_5`, `rolling_std_5`, `momentum`

- **Test Accuracy: 51.89%**
- **Mean CV Accuracy: 52.1%** (TimeSeriesSplit, 5 folds)

### 10. LSTM Spread Prediction
Two-layer LSTM with 20-day lookback window, dropout regularization, and early stopping.

- **Test Accuracy: 51.82%**

### 11. Model Comparison

| Model | Task | Metric | Value |
|---|---|---|---|
| ARIMA (rolling 1-step) | Spread level forecast | RMSE | 0.015950 |
| XGBoost | Direction classification | Accuracy | 51.89% |
| LSTM | Direction classification | Accuracy | 51.82% |

---

## 📊 Backtesting Results

A **walk-forward backtest** with 5 TimeSeriesSplit folds evaluated out-of-sample performance:

| Strategy | Gross Return | Sharpe (ann.) | Win Rate | Max Drawdown |
|---|---|---|---|---|
| ML (XGBoost) | **2.687** | **4.7638** | **64.70%** | -10.68% |
| Z-score Baseline | 0.093 | — | — | — |

The ML strategy substantially outperformed the simple z-score rule, with consistent positive returns across all 5 folds.

---

## 📚 References

**Indian Market:**
- [HDFCBANK & ICICIBANK Pair Trading](https://github.com/harshu722/pairs_trading_strategy_on_hdfcbank_and_icicibank-personal-project)
- [Pairs Trading — Indian IT Stocks](https://github.com/anirudhjayaraman/Pairs-Trading-Indian-IT-Stocks)
- [Cointegration StatArb NSE](https://github.com/rugvedshete/Cointegration-StatArb-PairsTrading-NSE)
- [Nifty50 Statistical Arbitrage](https://github.com/beastytitan18/nifty50-statistical-arbitrage)

**General:**
- [Pairs Trading & Cointegration Strategy](https://github.com/pandeyyyy/Pairs_Tradind_and_CoIntegration_Strategy)

---
