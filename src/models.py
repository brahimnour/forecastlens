"""
ForecastLens — Forecasting models and backtesting.

Two models are implemented and compared:
- SARIMA (statsmodels): classical statistical approach, explicit seasonal terms.
- Prophet (Meta): additive model, robust to trend changepoints and outliers
  (e.g. the 2020 COVID shock in this series).

Backtesting holds out the last 12 months, trains on everything before that,
and compares predictions to the real values using MAE and MAPE.
"""
import logging
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet

logging.getLogger("prophet").setLevel(logging.WARNING)
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)


def _mae_mape(pred: np.ndarray, actual: np.ndarray) -> tuple[float, float]:
    mae = np.mean(np.abs(pred - actual))
    mape = np.mean(np.abs((pred - actual) / actual)) * 100
    return mae, mape


def backtest_sarima(df: pd.DataFrame, holdout: int = 12) -> dict:
    """Backtest a SARIMA(2,1,2)(1,1,1,12) model on the last `holdout` months."""
    train, test = df.iloc[:-holdout], df.iloc[-holdout:]
    series = train.set_index("ds")["y"]

    model = SARIMAX(
        series, order=(2, 1, 2), seasonal_order=(1, 1, 1, 12),
        enforce_stationarity=False, enforce_invertibility=False,
    )
    fit = model.fit(disp=False)
    forecast = fit.get_forecast(steps=holdout)
    pred = forecast.predicted_mean.values

    mae, mape = _mae_mape(pred, test["y"].values)
    return {"model": "SARIMA", "mae": mae, "mape": mape, "predictions": pred, "actual": test["y"].values}


def backtest_prophet(df: pd.DataFrame, holdout: int = 12) -> dict:
    """Backtest a Prophet model on the last `holdout` months."""
    train, test = df.iloc[:-holdout].copy(), df.iloc[-holdout:].copy()

    model = Prophet(yearly_seasonality=True, weekly_seasonality=False,
                     daily_seasonality=False, changepoint_prior_scale=0.05)
    model.fit(train)

    future = model.make_future_dataframe(periods=holdout, freq="MS")
    forecast = model.predict(future)
    pred = forecast.tail(holdout)["yhat"].values

    mae, mape = _mae_mape(pred, test["y"].values)
    return {"model": "Prophet", "mae": mae, "mape": mape, "predictions": pred, "actual": test["y"].values}


def forecast_future(df: pd.DataFrame, periods: int = 6) -> pd.DataFrame:
    """Train Prophet on ALL available data and forecast `periods` months ahead."""
    model = Prophet(yearly_seasonality=True, weekly_seasonality=False,
                     daily_seasonality=False, changepoint_prior_scale=0.05)
    model.fit(df)

    future = model.make_future_dataframe(periods=periods, freq="MS")
    forecast = model.predict(future)
    return forecast
