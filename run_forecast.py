"""
ForecastLens — Main pipeline.

Runs both models' backtests, compares them, then generates the final
forward-looking forecast using the best-performing model (Prophet).

Usage:
    python run_forecast.py
"""
import os
from src.data_loader import load_data
from src.models import backtest_sarima, backtest_prophet, forecast_future

os.makedirs("outputs", exist_ok=True)


def main():
    df = load_data()
    print(f"Loaded {len(df)} months of data ({df['ds'].min().date()} to {df['ds'].max().date()})\n")

    print("=" * 55)
    print("BACKTESTING (last 12 months held out)")
    print("=" * 55)

    sarima_result = backtest_sarima(df)
    print(f"SARIMA  -> MAE: {sarima_result['mae']:.3f}M units | MAPE: {sarima_result['mape']:.2f}%")

    prophet_result = backtest_prophet(df)
    print(f"Prophet -> MAE: {prophet_result['mae']:.3f}M units | MAPE: {prophet_result['mape']:.2f}%")

    best = "Prophet" if prophet_result["mae"] < sarima_result["mae"] else "SARIMA"
    print(f"\nBest model: {best}")

    print("\n" + "=" * 55)
    print("6-MONTH FORWARD FORECAST (Prophet, trained on all data)")
    print("=" * 55)

    forecast = forecast_future(df, periods=6)
    future_only = forecast.tail(6)[["ds", "yhat", "yhat_lower", "yhat_upper"]]
    for _, row in future_only.iterrows():
        print(f"  {row['ds'].strftime('%Y-%m')}: {row['yhat']:.2f}M  "
              f"(90% CI: {row['yhat_lower']:.2f} - {row['yhat_upper']:.2f})")

    forecast.to_csv("outputs/forecast_full.csv", index=False)
    df.to_csv("outputs/historical_clean.csv", index=False)
    print("\nSaved: outputs/forecast_full.csv, outputs/historical_clean.csv")
    print("Run 'streamlit run app.py' to view the interactive dashboard.")


if __name__ == "__main__":
    main()
