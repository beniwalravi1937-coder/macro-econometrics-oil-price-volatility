# 📊 Macro + Econometrics Project

## Impact of Geopolitical Events on Oil Price Volatility

A comprehensive macroeconometrics research project analyzing how geopolitical events shape oil price dynamics, volatility regimes, and market uncertainty — spanning from **1987 to 2026**.

---

## 🧭 Project Overview

This project investigates the relationship between geopolitical risk and oil market behavior using a multi-stage econometric and machine learning pipeline. The central research question is:

> **How do geopolitical events (Gulf War, Oil Crisis 2007–2008, COVID-19, Russia-Ukraine War, 2026 US-Iran War) affect oil price levels, returns, and volatility?**

---

## 📁 Project Structure

```
Macro + Econometrics Project/
│
├── README.md                                          ← This file
│
├── MacroEconometricsProject.ipynb                    ← Core analysis notebook (Colab)
├── MacroEconometrics_Analysis.pdf                    ← Summary analysis document
├── Econometric Analysis.pdf                          ← Full econometric report
├── Econometric Analysis.pptx                        ← Presentation slides
│
├── oil-prices-vs-geopolitical-events-eda.ipynb      ← EDA: Oil markets & geopolitics (2010–2026)
├── global-petrol-prices-impact-of-2026-us-iran-war.ipynb  ← ML analysis: 2026 US-Iran war impact
├── phillips-curve-analysis-with-fred-data.ipynb     ← Modern Phillips Curve analysis (FRED data)
│
├── oil_geopolitics_dataset_2010_2026.xlsx           ← Primary geopolitics + oil dataset
│
├── Best Regression.png                               ← Best regression model output
├── New regression.png                                ← Updated regression output
├── islm graph.png                                    ← IS-LM model diagram
├── Klayan chart.png                                  ← Klayan analysis chart
├── macro_project_pipeline.svg                        ← Full analytical pipeline diagram
├── dataset-cover.png                                 ← Dataset visualization cover
│
└── Macro Project Code/                              ← Versioned code & data directory
    ├── MacroEconometricsProject_FINAL.ipynb         ← Final production notebook (v4.0)
    ├── MacroEconometricsProject_v2.ipynb            ← Version 2
    ├── MacroEconometricsProject_v3.ipynb            ← Version 3
    ├── MacroEconometricsProject_v4.ipynb            ← Version 4
    ├── Econometrics_macro_dataset.csv               ← Raw dataset (CSV)
    ├── Mcroproject_updated.xlsx                     ← Cleaned & updated dataset (Excel)
    ├── Mcroproject_updated.csv                      ← Cleaned dataset (CSV)
    ├── Econometric Analysis.pptx                    ← Presentation (versioned copy)
    └── Images of project/
        ├── Dollar and Finance index Trend.png
        ├── Oil Prices Trends.png
        ├── Stuctural break.png
        ├── Volatility.png
        ├── Volatlity GARCH.png
        └── XG Booast Prediction Graph.png
```

---

## 📈 Dataset Description

### Primary Dataset (`Mcroproject_updated.xlsx` / `.csv`)

| Column | Description |
|--------|-------------|
| `Date` | Daily trading date (1987-05-20 onwards) |
| `Oil Price` | Daily crude oil closing price (USD/barrel) |
| `Log Return` | Daily log return: ln(Pt / Pt-1) |
| `GPR INDEX` | Geopolitical Risk Index (Caldara & Iacoviello) — text-based measure from news |
| `Gulf_War` | Binary dummy: 1 = Gulf War period (1990-08-02 to 1991-01-17) |
| `Oil_Crisis` | Binary dummy: 1 = Global Oil Crisis (2007-12-01 to 2008-06-30) |
| `Covid_Period` | Binary dummy: 1 = COVID-19 period (2020-01-01 to 2021-12-31) |
| `Russia_Ukraine_War` | Binary dummy: 1 = Russia-Ukraine War (2022-02-24 onwards) |

### Extended Dataset (`oil_geopolitics_dataset_2010_2026.xlsx`)

| Column | Description |
|--------|-------------|
| `date` | Trading date |
| `brent_price`, `wti_price` | Daily Brent and WTI closing prices (USD/barrel) |
| `dxy_index` | US Dollar Index |
| `vix` | VIX fear index (CBOE Volatility Index) |
| `gpr_index` | Geopolitical Risk Index |
| `brent_return`, `wti_return` | Daily log returns |
| `brent_lag_1/3/7`, `wti_lag_1/3/7` | Lagged prices (1, 3, 7 days) |
| `brent_volatility_7d/30d` | Rolling volatility |
| `brent_wti_spread` | Price spread between Brent and WTI |
| `event_flag` | Binary: 1 = geopolitical event occurred |
| `event_type` | Category (war, sanctions, attack, etc.) |
| `event_description` | Text description of the event |
| `event_severity` | Severity score of the event |

### Descriptive Statistics (Core Dataset, ~9,828 observations)

| Variable | Mean | Std Dev | Skewness | Kurtosis |
|----------|------|---------|----------|----------|
| Oil Price | $50.92 | $32.45 | 0.56 | -0.87 |
| Log Return | 0.000167 | 0.023 | -0.77 | 15.28 |
| GPR Index | 111.09 | 63.56 | 4.15 | 33.19 |

**Event Coverage:**
- Gulf War: **119** trading days
- Oil Crisis: **144** trading days
- COVID-19: **505** trading days
- Russia-Ukraine War: **1,019** trading days

---

## 🔬 Analytical Pipeline (FINAL Notebook — v4.0)

```
Setup → Data → Features → EDA → Stationarity → Granger → OLS
→ Heteroskedasticity → ARCH → GARCH(1,1)-X → GJR/EGARCH
→ Diagnostics → Bai-Perron → VAR+IRF
→ Markov Switching → Event Study → XGBoost → LSTM → DM Test → Summary
```

### Econometric Models

**Mean Equation (OLS & GARCH):**

  r_t = β0 + φ·r_{t-1} + γ·ΔGPR_{t-1} + θ1·ΔDXY_{t-1} + θ2·ΔVIX_{t-1} + δ·D_t + ε_t

**Variance Equation (GARCH-X):**

  h_t = ω + α·ε²_{t-1} + β·h_{t-1} + λ1·GPR_{t-1} + λ2·D_t

> **Note (v4.0 fix):** OLS and GARCH mean equation use **first-differenced** regressors (ΔGPR, ΔDXY, ΔVIX) to resolve stationarity mismatch. GPR in levels is retained only in the **variance equation** (GARCH-X).

### Analytical Stages

| Stage | Description |
|-------|-------------|
| **1. Setup** | Install packages (arch, statsmodels, ruptures, xgboost, shap, tensorflow) |
| **2. Data Loading** | Load & clean Excel dataset, parse dates, create event dummies |
| **3. Feature Engineering** | Log returns, lag features, rolling volatility, dummy variables |
| **4. EDA** | Descriptive stats, time-series plots, correlation heatmaps |
| **5. Stationarity Tests** | ADF, KPSS, Phillips-Perron unit root tests |
| **6. Granger Causality** | Test if GPR Granger-causes oil returns |
| **7. OLS Regression** | Baseline linear regression with event dummies |
| **8. Heteroskedasticity** | Breusch-Pagan, White test for variance non-stationarity |
| **9. ARCH Test** | ARCH-LM test to confirm conditional heteroskedasticity |
| **10. GARCH(1,1)-X** | Volatility modeling with external regressors (GPR, event dummies) |
| **11. GJR/EGARCH** | Asymmetric GARCH models for leverage effects |
| **12. Diagnostics** | Residual analysis, autocorrelation checks |
| **13. Bai-Perron** | Structural break detection in the time series |
| **14. VAR + IRF** | Vector Auto-Regression and Impulse Response Functions |
| **15. Markov Switching** | Regime-switching models for bull/bear oil markets |
| **16. Event Study** | Abnormal return analysis around geopolitical events |
| **17. XGBoost** | Gradient boosting for return/volatility prediction |
| **18. LSTM** | Long Short-Term Memory neural network for time-series forecasting |
| **19. DM Test** | Diebold-Mariano forecast comparison test |
| **20. Summary** | Key findings and policy implications |

---

## 📓 Notebooks

### 1. `MacroEconometricsProject.ipynb` (Core Notebook)
- **Platform:** Google Colab
- **Purpose:** Step-by-step implementation of the full econometric pipeline
- **Coverage:** Data loading, dummy variable creation, descriptive statistics, summary analysis
- **Key tabs:** Import → Load & Process → Summary Statistics → Visualization → Regression → GARCH → Diagnostics

### 2. `Macro Project Code/MacroEconometricsProject_FINAL.ipynb` (Production Notebook — v4.0)
- **Platform:** Google Colab
- **Purpose:** Final, production-ready version with complete pipeline
- **Coverage:** Full 20-stage workflow from raw data to ML forecasts
- **Key improvements:** First-differenced regressors, stationarity fix, GJR/EGARCH, LSTM

### 3. `oil-prices-vs-geopolitical-events-eda.ipynb` (EDA Notebook)
- **Platform:** Kaggle
- **Executed:** 2026-03-12
- **Purpose:** Exploratory Data Analysis of oil market data enriched with geopolitical annotations
- **Coverage:** 2010–2026 daily data, Brent/WTI prices, GPR, DXY, VIX, event flags

### 4. `global-petrol-prices-impact-of-2026-us-iran-war.ipynb` (2026 US-Iran War Analysis)
- **Platform:** Kaggle
- **Executed:** 2026-03-11
- **Purpose:** EDA + ML analysis of the 2026 US-Iran war shock on global petroleum markets
- **Key findings:**
  - Brent surged **+40.5%** in 21 trading days after Strait of Hormuz closure
  - Pakistan suffered the largest retail price hike (+20.66%)
  - South Asian countries most vulnerable (>85% oil-import dependency)
  - Random Forest achieved best regression performance
- **Coverage:** 14–17 countries, 5 linked datasets, Brent time series, country-level economic impact

### 5. `phillips-curve-analysis-with-fred-data.ipynb` (Phillips Curve Analysis)
- **Platform:** Kaggle
- **Executed:** 2025-07-01
- **Purpose:** Empirical analysis of the Modern (New Keynesian) Phillips Curve using US FRED data
- **Coverage:** Inflation, unemployment, output gap, inflation expectations, ARCH/GARCH for residuals
- **Key models:** NKPC estimation, ADF/KPSS unit root tests, Phillips-Perron, GARCH

---

## 🛠️ Python Libraries Used

| Library | Purpose |
|---------|---------|
| `pandas` | Data manipulation and loading |
| `numpy` | Numerical computation |
| `matplotlib` | Data visualization |
| `seaborn` | Statistical visualization |
| `statsmodels` | OLS regression, VAR, unit root tests, Granger causality |
| `arch` | GARCH/EGARCH/GJR-GARCH volatility modeling, ARCH-LM test |
| `xgboost` | Gradient boosting regression |
| `shap` | Model explainability (SHAP values) |
| `tensorflow` | LSTM neural network implementation |
| `ruptures` | Structural break detection (Bai-Perron) |
| `scikit-learn` | Machine learning utilities, train/test split |

---

## 🌍 Geopolitical Events Analyzed

| Event | Period | Trading Days |
|-------|--------|-------------|
| Gulf War | Aug 2, 1990 – Jan 17, 1991 | 119 |
| Global Oil Crisis | Dec 2007 – Jun 2008 | 144 |
| Oil Crash | 2014–2016 | — |
| COVID-19 Pandemic | Jan 2020 – Dec 2021 | 505 |
| Russia-Ukraine War | Feb 24, 2022 – ongoing | 1,019+ |
| 2026 US-Iran War | Feb–Mar 2026 | 16+ |

---

## 📊 Key Visualizations

- **Oil Prices Trends** — Long-run Brent crude price time series with event annotations
- **Dollar and Finance Index Trend** — DXY index vs. oil price co-movement
- **Volatility Charts** — Rolling volatility and GARCH-estimated conditional variance
- **GARCH Volatility** — Conditional volatility from GARCH(1,1)-X model
- **Structural Break** — Bai-Perron detected break points in oil price series
- **XGBoost Prediction** — ML model predictions vs. actual oil price returns
- **IS-LM Graph** — Macroeconomic IS-LM model diagram
- **Regression Charts** — OLS fitted values and residual plots

---

## 🔑 Key Findings

1. **Geopolitical Risk Granger-causes oil price volatility** — GPR index has statistically significant predictive power for oil return volatility
2. **Asymmetric effects** — Negative oil price shocks (crashes) generate greater volatility than positive shocks (leverage effect confirmed via GJR/EGARCH)
3. **Structural breaks** — Significant regime shifts detected around the 2008 Financial Crisis, 2014 Oil Crash, and 2020 COVID-19 shock
4. **Event studies** — Gulf War and Russia-Ukraine War produced the most persistent abnormal returns
5. **ML forecasting** — XGBoost outperforms ARIMA in out-of-sample oil return prediction; LSTM captures non-linear volatility dynamics
6. **2026 US-Iran War** — Brent surged +40.5% in 3 weeks; South Asian nations most vulnerable due to high oil import dependency
7. **Phillips Curve** — Modern NKPC with forward-looking expectations fits US data better than traditional specification

---

## 🚀 Getting Started

### Requirements

```bash
pip install arch matplotlib seaborn statsmodels ruptures xgboost shap tensorflow pandas numpy openpyxl
```

### Running the Notebooks

#### Option A: Google Colab (Recommended for Main Analysis)
1. Upload `Macro Project Code/Mcroproject_updated.xlsx` to your Colab session
2. Open `Macro Project Code/MacroEconometricsProject_FINAL.ipynb` in Google Colab
3. Run all cells sequentially

#### Option B: Local Jupyter
```bash
jupyter notebook MacroEconometricsProject.ipynb
```

#### Option C: Kaggle Notebooks
- `oil-prices-vs-geopolitical-events-eda.ipynb` and others were run on Kaggle with pre-loaded datasets

---

## 📖 References

- **GPR Index:** Caldara, D. & Iacoviello, M. (2022). Measuring Geopolitical Risk. *American Economic Review*, 112(4), 1194–1225.
- **FRED Data:** Federal Reserve Bank of St. Louis Economic Data
- **Kaggle Dataset:** [global-petrol-prices-impact-of-2026-us-iran-war](https://www.kaggle.com/datasets/zkskhurram/global-petrol-prices-impact-of-2026-us-iran-war)
- **GARCH Methodology:** Bollerslev, T. (1986). Generalized Autoregressive Conditional Heteroskedasticity. *Journal of Econometrics*, 31(3), 307–327.
- **Bai-Perron:** Bai, J. & Perron, P. (2003). Computation and Analysis of Multiple Structural Change Models. *Journal of Applied Econometrics*, 18(1), 1–22.

---

*Last Updated: September 2026*
