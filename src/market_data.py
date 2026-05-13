"""Historical market data loading helpers."""

from __future__ import annotations

import pandas as pd


def load_price_history(path: str) -> pd.DataFrame:
    """Load historical prices, validate them and return a date-indexed DataFrame."""

    df = pd.read_csv(path)
    if "date" not in df.columns:
        raise ValueError("Price history file must contain a 'date' column.")

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    if df["date"].isna().any():
        bad_rows = df.index[df["date"].isna()].tolist()
        raise ValueError(f"Invalid date value(s) in price history rows: {bad_rows}")

    df = df.sort_values("date").set_index("date")

    if df.empty:
        raise ValueError("Price history file contains no data.")

    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        if df[col].isna().any():
            bad_rows = df.index[df[col].isna()].tolist()
            raise ValueError(f"Column '{col}' contains non-numeric price values at dates: {bad_rows}")
        if (df[col] <= 0).any():
            bad_rows = df.index[df[col] <= 0].tolist()
            raise ValueError(f"Column '{col}' contains non-positive prices at dates: {bad_rows}")

    return df

