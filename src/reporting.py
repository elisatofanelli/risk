"""Simple reporting helpers for future expansion."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def format_currency(value: float) -> str:
    return f"${value:,.2f}"


def create_risk_report(results: dict, output_path: str):
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
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    backtest_df.to_csv(output, index=False)
    return backtest_df


def save_backtest_summary(summary_list, output_path: str) -> pd.DataFrame:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(summary_list)
    df.to_csv(output, index=False)
    return df
