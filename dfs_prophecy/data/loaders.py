from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import pandas as pd


DEFAULT_SCHEMA = [
    "DFS_ID",
    "Name",
    "Position",
    "Team",
    "Opponent",
    "Salary",
]


def load_feature_table(path: Path, required_columns: Optional[Iterable[str]] = None) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [col for col in (required_columns or []) if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns {missing} in {path}")
    return df


def load_multiple_tables(paths: Iterable[Path]) -> pd.DataFrame:
    frames = [load_feature_table(path) for path in paths]
    if not frames:
        raise ValueError("No input sources provided")
    return pd.concat(frames, ignore_index=True)


def ensure_schema(df: pd.DataFrame, schema: Iterable[str] = DEFAULT_SCHEMA) -> pd.DataFrame:
    missing = [col for col in schema if col not in df.columns]
    if missing:
        raise ValueError(f"Input dataframe missing columns {missing}")
    return df
