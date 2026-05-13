"""Simple reporting helpers for future expansion."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def format_currency(value: float) -> str:
    """Format a numeric value as a USD currency string."""
    return f"${value:,.2f}"


def create_risk_report(results: dict, output_path: str):
    """Compile a dictionary of risk results into a DataFrame and save it as a CSV report."""
    rows = []
    for method, result in results.items():
        row = {
            "method": method,
            "VaR": result.get("VaR"),
            "ES": result.get("ES"),
            "confidence_level": result.get("confidence_level"),
            "current_value": result.get("current_value"),
            "number_of_scenarios": result.get("number_of_scenarios"),
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)
    return df


def save_backtest_results(backtest_df: pd.DataFrame, output_path: str) -> pd.DataFrame:
    """Save the detailed rolling backtest DataFrame to a CSV file."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    backtest_df.to_csv(output, index=False)
    return backtest_df


def save_backtest_summary(summary_list, output_path: str) -> pd.DataFrame:
    """Convert a list of backtest summary dictionaries into a DataFrame and save to CSV."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(summary_list)
    df.to_csv(output, index=False)
    return df