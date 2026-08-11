"""
ForecastLens — Data loading module.
"""
import pandas as pd


def load_data(csv_path: str = "data/total_vehicle_sales.csv") -> pd.DataFrame:
    """
    Load the FRED Total Vehicle Sales monthly series.
    Returns a DataFrame with columns ['ds', 'y'] (Prophet's expected format).
    """
    df = pd.read_csv(csv_path, parse_dates=["DATE"])
    df = df.set_index("DATE").asfreq("MS").reset_index()
    df.columns = ["ds", "y"]
    return df
