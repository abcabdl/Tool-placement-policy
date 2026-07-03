from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def aggregate_overall_results(input_file: str | Path, output_file: str | Path) -> pd.DataFrame:
    """Aggregate an experiment-result spreadsheet/CSV into overall metrics.

    The function expects a tidy table with columns such as model, benchmark,
    domain, accuracy, input_tokens, output_tokens, latency. Extra columns are
    preserved only for grouping if present.
    """
    path = Path(input_file)
    if path.suffix.lower() in {".xlsx", ".xls"}:
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)

    rename_map = {
        "acc": "accuracy",
        "avg_input_tokens": "input_tokens",
        "avg_output_tokens": "output_tokens",
        "avg_latency": "latency",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
    group_cols = [col for col in ["benchmark", "model"] if col in df.columns]
    metric_cols = [col for col in ["accuracy", "input_tokens", "output_tokens", "latency"] if col in df.columns]
    if not group_cols or not metric_cols:
        raise ValueError("Input must contain benchmark/model columns and at least one metric column.")

    result = df.groupby(group_cols, as_index=False)[metric_cols].mean(numeric_only=True)
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix.lower() == ".xlsx":
        result.to_excel(output_path, index=False)
    else:
        result.to_csv(output_path, index=False)
    return result

