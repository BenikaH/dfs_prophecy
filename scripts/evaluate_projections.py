#!/usr/bin/env python3
"""Evaluate deterministic NFL projections against provided actual scores.

This script implements the same heuristic scoring logic that powers
`ProjectionGenerator._generate_nfl` without requiring third-party
numerical libraries, which allows it to run in constrained environments.
"""
from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, List, Tuple

INJURY_MULTIPLIERS: Dict[str, float] = {
    "Out": 0.0,
    "Doubtful": 0.4,
    "Questionable": 0.85,
    "Probable": 1.0,
}


def _load_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return [dict(row) for row in reader]


def _as_float(row: Dict[str, str], key: str, default: float = 0.0) -> float:
    value = row.get(key)
    if value is None or value == "":
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _usage_index(row: Dict[str, str]) -> float:
    return 0.65 * _as_float(row, "Targets") + 0.35 * _as_float(row, "Rush_Attempts")


def _position_flags(position: str) -> Tuple[bool, bool, bool, bool, bool]:
    position = position.upper()
    return (
        position.startswith("QB"),
        position.startswith("RB"),
        position.startswith("WR"),
        position.startswith("TE"),
        "DST" in position,
    )


def _team_usage(rows: Iterable[Dict[str, str]]) -> Dict[str, float]:
    totals: Dict[str, float] = defaultdict(float)
    for row in rows:
        totals[row["Team"].upper()] += _usage_index(row)
    return totals


def _apply_projection(row: Dict[str, str], team_usage: Dict[str, float]) -> float:
    qb, rb, wr, te, dst = _position_flags(row["Position"])
    usage_index = _usage_index(row)
    share = usage_index / (team_usage[row["Team"].upper()] + 1e-6)
    pace_factor = max(0.6, min(1.4, 0.85 + 0.15 * (_as_float(row, "Opp_Pace", 62.0) / 65.0)))
    snap_factor = min(1.05, max(0.35, _as_float(row, "Proj_Snap_Pct", 0.65)))
    team_total = max(10.0, _as_float(row, "Team_Total", 22.0))
    injury_adj = INJURY_MULTIPLIERS.get(row.get("InOutStatus", "Probable"), 1.0)

    projection = 0.0

    if qb:
        pass_attempts = _as_float(row, "Targets")
        pass_yards = 0.55 * _as_float(row, "AirYards") + 8 * pass_attempts
        rush_yards = _as_float(row, "Rush_Attempts") * _as_float(row, "Yards_per_Rush", 4.2)
        pass_tds = max(_as_float(row, "TD_Rate", 0.02), 0.02) * pass_attempts
        projection = (
            0.04 * pass_yards
            + 4.0 * pass_tds
            + 0.1 * rush_yards
            + 0.6 * _as_float(row, "RedZoneTouches")
        )
    elif rb:
        rush_yards = _as_float(row, "Rush_Attempts") * _as_float(row, "Yards_per_Rush", 4.2)
        rec_points = _as_float(row, "Targets") * 0.55
        rec_yards = _as_float(row, "Targets") * _as_float(row, "Yards_per_Target", 7.8) * 0.1
        projection = 0.1 * rush_yards + rec_points + rec_yards + 6 * share * (team_total / 7.0)
    elif wr or te:
        catch_rate = max(0.45, min(0.85, 0.55 + 0.02 * (_as_float(row, "Yards_per_Target", 7.8) - 7.5)))
        receptions = _as_float(row, "Targets") * catch_rate
        rec_yards = _as_float(row, "Targets") * _as_float(row, "Yards_per_Target", 7.8)
        air_bonus = 0.04 * _as_float(row, "AirYards") if wr else 0.0
        projection = 0.5 * receptions + 0.1 * rec_yards + 6 * share * (team_total / 7.0) + air_bonus
    elif dst:
        dst_pressure = max(_as_float(row, "Opp_Pace", 62.0), 55.0)
        projection = 3.0 + 0.05 * (dst_pressure - 60.0)

    base = projection * pace_factor * snap_factor
    base *= (0.85 + (team_total - 10.0) * (1.2 - 0.85) / 30.0)
    base *= injury_adj

    regression_target = _as_float(row, "Salary", 5500.0) / 1000.0
    projected = (1 - 0.15) * base + 0.15 * regression_target

    if qb:
        projected *= 1.05

    return max(projected, 0.0)


def _compute_metrics(preds: List[float], actuals: List[float]) -> Dict[str, float]:
    errors = [p - a for p, a in zip(preds, actuals)]
    abs_errors = [abs(e) for e in errors]
    sq_errors = [e * e for e in errors]
    mae = mean(abs_errors)
    rmse = math.sqrt(mean(sq_errors))
    bias = mean(errors)
    return {"mae": mae, "rmse": rmse, "bias": bias}


def evaluate(features_path: Path, actuals_path: Path) -> None:
    features = _load_csv(features_path)
    actual_map = {row["DFS_ID"]: float(row["Actual_Points"]) for row in _load_csv(actuals_path)}

    usage = _team_usage(features)
    preds: List[float] = []
    actuals: List[float] = []
    position_buckets: Dict[str, List[Tuple[float, float]]] = defaultdict(list)

    for row in features:
        dfs_id = row["DFS_ID"]
        if dfs_id not in actual_map:
            continue
        projection = _apply_projection(row, usage)
        actual = actual_map[dfs_id]
        preds.append(projection)
        actuals.append(actual)
        position_buckets[row["Position"].upper()].append((projection, actual))

    overall = _compute_metrics(preds, actuals)
    print("Overall metrics:")
    for key, value in overall.items():
        print(f"  {key.upper()}: {value:.3f}")

    print("\nPer-position metrics:")
    for position, bucket in sorted(position_buckets.items()):
        p, a = zip(*bucket)
        metrics = _compute_metrics(list(p), list(a))
        formatted = ", ".join(f"{k.upper()}={v:.3f}" for k, v in metrics.items())
        print(f"  {position}: {formatted}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate heuristic NFL projections")
    parser.add_argument("--features", type=Path, default=Path("data/sample_inputs/nfl_features.csv"))
    parser.add_argument("--actuals", type=Path, default=Path("data/sample_inputs/nfl_actuals.csv"))
    args = parser.parse_args()
    evaluate(args.features, args.actuals)
