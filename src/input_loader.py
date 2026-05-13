"""Portfolio input loading and validation."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = [
    "instrument_id",
    "type",
    "underlying",
    "position",
    "strike",
    "maturity",
    "option_type",
    "implied_vol",
    "risk_free_rate",
]


def _require_columns(df: pd.DataFrame) -> None:
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Portfolio file is missing required columns: {', '.join(missing)}")


def load_portfolio(path: str) -> pd.DataFrame:
    """Load and validate a standardized portfolio CSV."""

    csv_path = Path(path)
    if not csv_path.exists():
        raise ValueError(f"Portfolio file not found: {path}")

    df = pd.read_csv(csv_path)
    _require_columns(df)

    df = df.copy()
    df["type"] = df["type"].astype(str).str.lower().str.strip()

    valid_types = {"stock", "option"}
    invalid_types = sorted(set(df["type"]) - valid_types)
    if invalid_types:
        raise ValueError(f"Invalid instrument types: {', '.join(invalid_types)}")

    numeric_fields = ["position", "strike", "implied_vol", "risk_free_rate"]
    for field in numeric_fields:
        df[field] = pd.to_numeric(df[field], errors="coerce")

    if df["position"].isna().any():
        raise ValueError("Column 'position' must contain numeric values for all rows.")

    option_mask = df["type"] == "option"
    stock_mask = df["type"] == "stock"

    option_required = ["strike", "maturity", "option_type", "implied_vol", "risk_free_rate"]
    for field in option_required:
        missing = option_mask & df[field].isna()
        if missing.any():
            rows = df.index[missing].tolist()
            raise ValueError(f"Missing required option field '{field}' for row indices: {rows}")

    df.loc[stock_mask, ["strike", "maturity", "option_type", "implied_vol", "risk_free_rate"]] = pd.NA

    df.loc[option_mask, "option_type"] = df.loc[option_mask, "option_type"].astype(str).str.lower().str.strip()
    invalid_option_types = sorted(set(df.loc[option_mask, "option_type"]) - {"call", "put"})
    if invalid_option_types:
        raise ValueError(f"Invalid option_type value(s): {', '.join(invalid_option_types)}")

    if option_mask.any():
        df.loc[option_mask, "maturity"] = pd.to_datetime(df.loc[option_mask, "maturity"], errors="coerce")
        if df.loc[option_mask, "maturity"].isna().any():
            rows = df.index[option_mask & df["maturity"].isna()].tolist()
            raise ValueError(f"Invalid maturity dates for option row indices: {rows}")

    return df

