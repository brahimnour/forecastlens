# ForecastLens — Sales Demand Forecasting Model

A time-series forecasting project predicting U.S. vehicle sales demand,
comparing two modeling approaches (SARIMA and Prophet) with rigorous
backtesting, and serving the result through an interactive Streamlit
dashboard.

## Dataset
**Total Vehicle Sales (TOTALSA)** — U.S. Bureau of Economic Analysis, via
[FRED (Federal Reserve Bank of St. Louis)](https://fred.stlouisfed.org/series/TOTALSA).
607 monthly observations, January 1976 to July 2026. Millions of units,
Seasonally Adjusted Annual Rate.

The series captures major economic shocks (2008 financial crisis, 2020
COVID collapse and rebound), making it a realistic and challenging
forecasting target — not a synthetic or trivially seasonal dataset.

## Approach

```
Historical data (607 months)
        │
        ▼
 Exploratory analysis + STL decomposition
        │
        ▼
 ┌─────────────┐        ┌─────────────┐
 │   SARIMA    │        │   Prophet   │
 │(statsmodels)│        │   (Meta)    │
 └─────────────┘        └─────────────┘
        │                       │
        └───────────┬───────────┘
                     ▼
      Backtest on last 12 months (MAE / MAPE)
                     ▼
        Best model → 6-month forward forecast
                     ▼
             Streamlit dashboard
```

## Data quality note
This series is **already seasonally adjusted** by the source (FRED/BEA).
STL decomposition confirms a weak residual seasonal signal (amplitude
~1.3M units) relative to noise (std ~0.89M) — meaning the forecast
mostly captures **trend and business-cycle dynamics**, not calendar
seasonality. This is documented explicitly rather than presented as a
stronger seasonal effect than what the data supports.

## Results (backtest, last 12 months held out)

| Model   | MAE (million units) | MAPE  |
|---------|----------------------|-------|
| SARIMA  | 0.515                | 3.25% |
| **Prophet** | **0.426**        | **2.67%** |

Prophet was selected as the production model based on lower backtest error.

## Setup & Run

```bash
pip install -r requirements.txt

# Run the full pipeline (backtest + forecast) from the command line
python run_forecast.py

# Launch the interactive dashboard
streamlit run app.py
```

## Project Structure
```
forecastlens/
├── data/total_vehicle_sales.csv   # raw FRED data
├── src/
│   ├── data_loader.py             # data loading
│   └── models.py                  # SARIMA + Prophet, backtesting, forecasting
├── outputs/                        # generated forecast/historical CSVs
├── run_forecast.py                 # CLI pipeline entry point
├── app.py                          # Streamlit dashboard
└── requirements.txt
```

## Tech Stack
- **Python**: pandas, statsmodels (SARIMA), Prophet (Meta)
- **Visualization**: Plotly, Streamlit
- **Methodology**: STL decomposition, train/holdout backtesting, MAE/MAPE evaluation
## Live Demo
Interactive dashboard: https://forecastlensapp.streamlit.app
