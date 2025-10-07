from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

from dfs_prophecy.config import ProjectionConfig
from dfs_prophecy.data.loaders import ensure_schema, load_multiple_tables


@dataclass
class ProjectionResult:
    data: pd.DataFrame
    diagnostics: pd.DataFrame


def _normal_ppf(p: float) -> float:
    if not 0.0 < p < 1.0:
        raise ValueError("Quantile must be between 0 and 1")
    # Acklam's inverse normal approximation
    a = [
        -3.969683028665376e01,
        2.209460984245205e02,
        -2.759285104469687e02,
        1.383577518672690e02,
        -3.066479806614716e01,
        2.506628277459239e00,
    ]
    b = [
        -5.447609879822406e01,
        1.615858368580409e02,
        -1.556989798598866e02,
        6.680131188771972e01,
        -1.328068155288572e01,
    ]
    c = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e00,
        -2.549732539343734e00,
        4.374664141464968e00,
        2.938163982698783e00,
    ]
    d = [
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e00,
        3.754408661907416e00,
    ]
    plow = 0.02425
    phigh = 1 - plow
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (
            (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
            / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
        )
    if phigh < p:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(
            (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
            / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
        )
    q = p - 0.5
    r = q * q
    return (
        (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q
    ) / (
        ((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1
    )


class ProjectionGenerator:
    def __init__(self, config: ProjectionConfig, rng: np.random.Generator) -> None:
        self.config = config
        self.rng = rng

    def generate(self, feature_sources: Iterable[Path]) -> ProjectionResult:
        features = load_multiple_tables(feature_sources)
        features = ensure_schema(features)
        features = self._standardize(features)
        sport = self.config.sport.lower()
        if sport == "nfl":
            df = self._generate_nfl(features)
        elif sport == "nba":
            df = self._generate_nba(features)
        elif sport == "pga":
            df = self._generate_pga(features)
        else:
            raise ValueError(f"Unsupported sport {self.config.sport}")
        df = self._blend_third_party(df)
        df = self._apply_uncertainty(df)
        diagnostics = self._compute_diagnostics(df)
        return ProjectionResult(df, diagnostics)

    def _standardize(self, features: pd.DataFrame) -> pd.DataFrame:
        df = features.copy()
        df["Position"] = df["Position"].str.upper()
        df["Team"] = df["Team"].str.upper()
        df["Opponent"] = df["Opponent"].str.upper()
        df = df.drop_duplicates("DFS_ID", keep="last")
        numeric_defaults: Dict[str, float] = {
            "Targets": 0.0,
            "Rush_Attempts": 0.0,
            "AirYards": 0.0,
            "RedZoneTouches": 0.0,
            "Team_Total": 22.0,
            "Opp_Pace": 62.0,
            "Proj_Snap_Pct": 0.65,
            "Yards_per_Target": 7.8,
            "Yards_per_Rush": 4.2,
            "TD_Rate": 0.04,
        }
        for column, default in numeric_defaults.items():
            if column not in df:
                df[column] = default
            else:
                df[column] = df[column].fillna(default)
        if "InOutStatus" not in df:
            df["InOutStatus"] = "Probable"
        if "Notes" not in df:
            df["Notes"] = ""
        return df

    def _generate_nfl(self, features: pd.DataFrame) -> pd.DataFrame:
        df = features.copy()
        df["Usage_Index"] = 0.65 * df["Targets"] + 0.35 * df["Rush_Attempts"]
        team_usage = df.groupby("Team")["Usage_Index"].transform(lambda x: x.sum() + 1e-6)
        usage_share = df["Usage_Index"] / team_usage
        pace_factor = (0.85 + 0.15 * (df["Opp_Pace"] / 65.0)).clip(0.6, 1.4)
        snap_factor = df["Proj_Snap_Pct"].clip(0.35, 1.05)
        team_total = df["Team_Total"].clip(lower=10.0)
        injury_adj = df["InOutStatus"].map(self.config.injury_multipliers).fillna(1.0)

        # Baseline expectation by archetype
        qb_mask = df["Position"].str.startswith("QB")
        rb_mask = df["Position"].str.startswith("RB")
        wr_mask = df["Position"].str.startswith("WR")
        te_mask = df["Position"].str.startswith("TE")
        dst_mask = df["Position"].str.contains("DST")

        projection = np.zeros(len(df))

        # Quarterbacks: passing yards, passing TDs, rushing component
        pass_attempts = df["Targets"].where(qb_mask, 0).fillna(0)
        pass_yards = (df["AirYards"] * 0.55 + 8 * pass_attempts).where(qb_mask, 0)
        rush_yards_qb = (df["Rush_Attempts"] * df["Yards_per_Rush"]).where(qb_mask, 0)
        pass_tds = np.maximum(df["TD_Rate"], 0.02) * pass_attempts
        qb_points = (
            0.04 * pass_yards
            + 4.0 * pass_tds
            + 0.1 * rush_yards_qb
            + 0.6 * df["RedZoneTouches"].where(qb_mask, 0)
        )
        projection[qb_mask] = qb_points[qb_mask]

        # Running backs: rushing, receiving, TD share
        rush_yards_rb = (df["Rush_Attempts"] * df["Yards_per_Rush"]).where(rb_mask, 0)
        rec_points_rb = (df["Targets"] * 0.55).where(rb_mask, 0)
        rec_yards_rb = (df["Targets"] * df["Yards_per_Target"] * 0.1).where(rb_mask, 0)
        rb_td_share = usage_share.where(rb_mask, 0)
        rb_points = 0.1 * rush_yards_rb + rec_points_rb + rec_yards_rb + 6 * rb_td_share * (
            team_total / 7.0
        )
        projection[rb_mask] = rb_points[rb_mask]

        # Wide receivers / tight ends: receptions, yardage, air yards bonus
        catch_rate = np.clip(0.55 + 0.02 * (df["Yards_per_Target"] - 7.5), 0.45, 0.85)
        receptions = (df["Targets"] * catch_rate).where(wr_mask | te_mask, 0)
        rec_yards = (df["Targets"] * df["Yards_per_Target"]).where(wr_mask | te_mask, 0)
        air_bonus = (df["AirYards"] * 0.04).where(wr_mask, 0)
        td_share = usage_share.where(wr_mask | te_mask, 0)
        pass_points = 0.5 * receptions + 0.1 * rec_yards + 6 * td_share * (team_total / 7.0)
        projection[wr_mask | te_mask] = pass_points[wr_mask | te_mask] + air_bonus[wr_mask | te_mask]

        # DST: pace-adjusted turnover expectation plus touchdown upside
        dst_pressure = np.maximum(df["Opp_Pace"], 55)
        dst_projection = 3.0 + 0.05 * (dst_pressure - 60)
        projection[dst_mask] = dst_projection[dst_mask]

        base = projection * pace_factor * snap_factor
        base *= np.interp(team_total, [10, 40], [0.85, 1.2])
        base *= injury_adj.to_numpy()

        regression_target = df["Salary"].fillna(df["Salary"].median()).to_numpy() / 1000.0
        projection = (
            (1 - self.config.historical_regression) * base
            + self.config.historical_regression * regression_target
        )

        df["Projection"] = projection
        df.loc[qb_mask, "Projection"] *= self.config.qb_floor_boost

        df["Ceiling"] = df["Projection"]
        df["Floor"] = df["Projection"]
        df["Stdev"] = np.maximum(df["Projection"] * 0.35, self.config.stdev_floor)
        df.loc[wr_mask, "Stdev"] *= self.config.wr_spike_multiplier
        df.loc[te_mask, "Stdev"] *= 1.1
        df.loc[qb_mask, "Stdev"] *= 0.75
        df.loc[dst_mask, "Stdev"] = np.maximum(3.0, df.loc[dst_mask, "Stdev"])

        return df[
            [
                "DFS_ID",
                "Name",
                "Position",
                "Team",
                "Opponent",
                "Salary",
                "Projection",
                "Ceiling",
                "Floor",
                "Stdev",
                "InOutStatus",
                "Notes",
            ]
        ]

    def _generate_nba(self, features: pd.DataFrame) -> pd.DataFrame:
        df = features.copy()
        df["Minutes"] = df.get("Minutes", 32)
        df["Usage"] = df.get("Usage", 0.24)
        df["Pace"] = df.get("Pace", 100)
        df["AssistRate"] = df.get("AssistRate", 0.2)
        df["ReboundRate"] = df.get("ReboundRate", 0.12)

        per_min = 1.25 * df["Usage"] + 0.6 * df["AssistRate"] + 0.4 * df["ReboundRate"]
        pace_adj = df["Pace"] / 100
        df["Projection"] = df["Minutes"] * per_min * pace_adj

        stdev = np.sqrt(df["Projection"]) * 1.1
        df["Stdev"] = np.maximum(stdev, self.config.stdev_floor)
        df["Ceiling"] = df["Projection"] + df["Stdev"] * 1.75
        df["Floor"] = np.maximum(df["Projection"] - df["Stdev"] * 1.2, 0)
        df["InOutStatus"] = df.get("InOutStatus", "Questionable")
        df["Notes"] = df.get("Notes", "")
        return df[
            [
                "DFS_ID",
                "Name",
                "Position",
                "Team",
                "Opponent",
                "Salary",
                "Projection",
                "Ceiling",
                "Floor",
                "Stdev",
                "InOutStatus",
                "Notes",
            ]
        ]

    def _generate_pga(self, features: pd.DataFrame) -> pd.DataFrame:
        df = features.copy()
        df["Course_Fit"] = df.get("Course_Fit", 0.0)
        df["Strokes_Gained"] = df.get("Strokes_Gained", 0.1)
        df["Wind_Risk"] = df.get("Wind_Risk", 0.0)
        df["Recent_Form"] = df.get("Recent_Form", 0.05)

        df["Projection"] = 55 + 7 * df["Strokes_Gained"] + 2 * df["Course_Fit"] + 5 * df["Recent_Form"]
        df["Projection"] -= 3 * df["Wind_Risk"].clip(lower=0)
        df["Stdev"] = np.maximum(4.0 + 3 * df["Wind_Risk"], self.config.stdev_floor)
        df["Ceiling"] = df["Projection"] + 1.2 * df["Stdev"]
        df["Floor"] = df["Projection"] - 1.5 * df["Stdev"]
        df["InOutStatus"] = df.get("InOutStatus", "Playing")
        df["Notes"] = df.get("Notes", "")
        return df[
            [
                "DFS_ID",
                "Name",
                "Position",
                "Team",
                "Opponent",
                "Salary",
                "Projection",
                "Ceiling",
                "Floor",
                "Stdev",
                "InOutStatus",
                "Notes",
            ]
        ]

    def _blend_third_party(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.config.projection_sources:
            return df
        frames: List[pd.DataFrame] = []
        for idx, path in enumerate(self.config.projection_sources):
            frame = pd.read_csv(path)
            if "DFS_ID" not in frame.columns or "Projection" not in frame.columns:
                raise ValueError(f"Projection source {path} missing DFS_ID/Projection columns")
            rename_map = {
                "Projection": f"Projection_src{idx}",
                "Ceiling": f"Ceiling_src{idx}",
                "Floor": f"Floor_src{idx}",
                "Stdev": f"Stdev_src{idx}",
            }
            subset_cols = [col for col in rename_map if col in frame.columns]
            subset = frame[["DFS_ID", *subset_cols]].rename(columns={col: rename_map[col] for col in subset_cols})
            frames.append(subset)
        if not frames:
            return df
        merged = df.copy()
        for frame in frames:
            merged = merged.merge(frame, on="DFS_ID", how="left")
        weights = self.config.blend_weights or [1.0 / len(frames)] * len(frames)
        weight_sum = float(sum(weights))
        base_weight = max(0.0, 1.0 - weight_sum)
        total_weight = weight_sum + base_weight
        projection_mix = base_weight * merged["Projection"]
        ceiling_mix = base_weight * merged["Ceiling"]
        floor_mix = base_weight * merged["Floor"]
        stdev_mix = base_weight * merged["Stdev"]
        for idx, weight in enumerate(weights):
            proj_series = merged.get(f"Projection_src{idx}", merged["Projection"]).fillna(
                merged["Projection"]
            )
            ceiling_series = merged.get(f"Ceiling_src{idx}", merged["Ceiling"]).fillna(
                merged["Ceiling"]
            )
            floor_series = merged.get(f"Floor_src{idx}", merged["Floor"]).fillna(
                merged["Floor"]
            )
            stdev_series = merged.get(f"Stdev_src{idx}", merged["Stdev"]).fillna(
                merged["Stdev"]
            )
            projection_mix += weight * proj_series
            ceiling_mix += weight * ceiling_series
            floor_mix += weight * floor_series
            stdev_mix += weight * stdev_series
        merged["Projection"] = projection_mix / total_weight
        merged["Ceiling"] = ceiling_mix / total_weight
        merged["Floor"] = floor_mix / total_weight
        merged["Stdev"] = stdev_mix / total_weight
        return merged

    def _apply_uncertainty(self, df: pd.DataFrame) -> pd.DataFrame:
        z_hi = _normal_ppf(self.config.ceiling_quantile)
        z_lo = _normal_ppf(self.config.floor_quantile)
        df = df.copy()
        stdev = df["Stdev"].replace([np.nan, np.inf], self.config.stdev_floor)
        stdev = np.maximum(stdev, self.config.stdev_floor)
        df["Stdev"] = stdev
        df["Ceiling"] = np.maximum(df["Projection"] + z_hi * stdev, df["Projection"])  # monotone
        df["Floor"] = np.clip(df["Projection"] + z_lo * stdev, 0, None)
        df["Projection"] = df["Projection"].clip(lower=0)
        return df

    def _compute_diagnostics(self, df: pd.DataFrame) -> pd.DataFrame:
        diagnostics = df.copy()
        diagnostics["CeilingGap"] = diagnostics["Ceiling"] - diagnostics["Projection"]
        diagnostics["FloorBuffer"] = diagnostics["Projection"] - diagnostics["Floor"]
        grouped = (
            diagnostics.groupby("Position")
            .agg(
                Players=("DFS_ID", "count"),
                MeanProjection=("Projection", "mean"),
                MedianProjection=("Projection", "median"),
                MeanCeiling=("Ceiling", "mean"),
                MeanFloor=("Floor", "mean"),
                MeanStdev=("Stdev", "mean"),
                AvgCeilingGap=("CeilingGap", "mean"),
                AvgFloorBuffer=("FloorBuffer", "mean"),
            )
            .reset_index()
        )
        return grouped
