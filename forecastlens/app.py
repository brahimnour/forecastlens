"""
ForecastLens — Interactive forecasting dashboard.
Run with: streamlit run app.py
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from src.data_loader import load_data
from src.models import backtest_sarima, backtest_prophet, forecast_future

st.set_page_config(page_title="ForecastLens",layout="wide")

st.title("ForecastLens — US Vehicle Sales Demand Forecasting")
st.caption(
    "Source: FRED (Federal Reserve Bank of St. Louis), series TOTALSA — "
    "Total Vehicle Sales, Seasonally Adjusted Annual Rate, monthly, 1976–present."
)

# ---------- Load data (cached) ----------
@st.cache_data
def get_data():
    return load_data()

df = get_data()

# ---------- Sidebar controls ----------
st.sidebar.header("Settings")
horizon = st.sidebar.slider("Forecast horizon (months)", min_value=1, max_value=12, value=6)
show_history_years = st.sidebar.slider("Years of history to display", min_value=2, max_value=50, value=10)

# ---------- KPI row ----------
@st.cache_data
def get_backtest():
    sarima = backtest_sarima(df)
    prophet = backtest_prophet(df)
    return sarima, prophet

sarima_result, prophet_result = get_backtest()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Latest value", f"{df['y'].iloc[-1]:.2f}M units", help=f"As of {df['ds'].iloc[-1].strftime('%B %Y')}")
col2.metric("Prophet MAE (backtest)", f"{prophet_result['mae']:.3f}M units")
col3.metric("Prophet MAPE (backtest)", f"{prophet_result['mape']:.2f}%")
col4.metric("SARIMA MAPE (comparison)", f"{sarima_result['mape']:.2f}%")

st.divider()

# ---------- Forecast ----------
@st.cache_data
def get_forecast(periods):
    return forecast_future(df, periods=periods)

forecast = get_forecast(horizon)

cutoff = df["ds"].max() - pd.DateOffset(years=show_history_years)
hist_view = df[df["ds"] >= cutoff]
future_view = forecast[forecast["ds"] > df["ds"].max()]

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=hist_view["ds"], y=hist_view["y"], mode="lines", name="Historical",
    line=dict(color="#2563eb", width=2),
))
fig.add_trace(go.Scatter(
    x=future_view["ds"], y=future_view["yhat"], mode="lines+markers", name="Forecast",
    line=dict(color="#f97316", width=2, dash="dash"),
))
fig.add_trace(go.Scatter(
    x=pd.concat([future_view["ds"], future_view["ds"][::-1]]),
    y=pd.concat([future_view["yhat_upper"], future_view["yhat_lower"][::-1]]),
    fill="toself", fillcolor="rgba(249,115,22,0.15)", line=dict(color="rgba(0,0,0,0)"),
    name="90% Confidence Interval", showlegend=True,
))
fig.update_layout(
    title=f"Total Vehicle Sales — History & {horizon}-Month Forecast",
    xaxis_title="Date", yaxis_title="Million Units (SAAR)",
    hovermode="x unified", height=500,
)
st.plotly_chart(fig, use_container_width=True)

# ---------- Forecast table ----------
st.subheader("Forecast Detail")
table = future_view[["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()
table.columns = ["Month", "Forecast (M units)", "Lower Bound (90%)", "Upper Bound (90%)"]
table["Month"] = table["Month"].dt.strftime("%B %Y")
st.dataframe(table.round(2), use_container_width=True, hide_index=True)

# ---------- Backtest detail ----------
with st.expander("Model comparison detail (backtest on last 12 months)"):
    bt_df = pd.DataFrame({
        "Month": df["ds"].tail(12).dt.strftime("%Y-%m"),
        "Actual": sarima_result["actual"],
        "SARIMA Prediction": sarima_result["predictions"],
        "Prophet Prediction": prophet_result["predictions"],
    })
    st.dataframe(bt_df.round(2), use_container_width=True, hide_index=True)
    st.caption(
        "Prophet was selected as the production model for the forward forecast "
        "above because it achieved lower error (MAE/MAPE) on this backtest. "
        "Note: this series is already seasonally adjusted by the source (FRED), "
        "so the residual seasonal signal is weak — forecasts mostly capture "
        "trend and business-cycle dynamics rather than calendar seasonality."
    )

st.divider()
st.caption("Data: U.S. Bureau of Economic Analysis via FRED, Federal Reserve Bank of St. Louis.")
