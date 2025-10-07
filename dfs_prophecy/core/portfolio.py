from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd

from dfs_prophecy.config import PortfolioConfig


@dataclass
class PortfolioResult:
    assignments: pd.DataFrame
    exposure_summary: pd.DataFrame


class PortfolioOptimizer:
    def __init__(self, config: PortfolioConfig, rng: np.random.Generator) -> None:
        self.config = config
        self.rng = rng

    def optimize(self, lineup_metrics: pd.DataFrame, contest_id: str = "NFL_MME") -> PortfolioResult:
        if lineup_metrics.empty:
            raise ValueError("Lineup metrics required for portfolio optimization")
        max_entries_bankroll = (
            int(self.config.bankroll // self.config.entry_fee)
            if self.config.entry_fee
            else len(lineup_metrics)
        )
        max_entries_config = self.config.max_total_entries or len(lineup_metrics)
        max_entries = max(0, min(max_entries_bankroll, max_entries_config))
        if max_entries == 0:
            empty = pd.DataFrame(
                columns=
                [
                    "Lineup_ID",
                    "Contest_ID",
                    "Entries_Assigned",
                    "EV",
                    "ROI",
                    "Prob_Cash",
                    "Prob_Top0.1%",
                    "CVaR@{:.0%}".format(self.config.cvar_alpha),
                    "Kelly",
                    "Sharpe",
                    "Score",
                    "Notes",
                ]
            )
            return PortfolioResult(empty, empty)

        eligible = lineup_metrics[lineup_metrics["CVaR"] >= self.config.min_cvar]
        if eligible.empty:
            eligible = lineup_metrics.copy()
        eligible = eligible.copy()
        def _score(row: pd.Series) -> float:
            ev = float(row.get("EV", 0.0))
            cvar = float(row.get("CVaR", 0.0))
            upside = float(row.get("Prob_Top0.1Pct", row.get("Prob_Top1Pct", 0.0)))
            sharpe = float(row.get("Sharpe", 0.0))
            penalty = max(0.0, self.config.min_cvar - cvar)
            return ev - self.config.risk_aversion * penalty + upside * self.config.entry_fee + 0.1 * sharpe

        eligible["Score"] = eligible.apply(_score, axis=1)
        eligible = eligible.sort_values(["Score", "ROI", "EV"], ascending=False)
        assignments = []
        total_assigned = 0
        for _, row in eligible.iterrows():
            remaining = max_entries - total_assigned
            if remaining <= 0:
                break
            kelly_fraction = float(row.get("Kelly", 0.0))
            kelly_entries = int(math.floor(kelly_fraction * self.config.kelly_multiplier * max_entries))
            entries_for_lineup = max(
                self.config.min_allocation_per_lineup,
                kelly_entries,
            )
            entries_for_lineup = min(
                entries_for_lineup,
                remaining,
                self.config.max_exposure_per_lineup,
            )
            if entries_for_lineup <= 0 and row.get("EV", 0.0) > 0:
                entries_for_lineup = min(remaining, self.config.min_allocation_per_lineup)
            if entries_for_lineup <= 0:
                continue
            assignments.append(
                {
                    "Lineup_ID": row["Lineup_ID"],
                    "Contest_ID": contest_id,
                    "Entries_Assigned": entries_for_lineup,
                    "EV": row["EV"],
                    "ROI": row["ROI"],
                    "Prob_Cash": row.get("Prob_Cash", np.nan),
                    "Prob_Top0.1%": row.get("Prob_Top0.1Pct", row.get("Prob_Top1Pct", np.nan)),
                    "CVaR@{:.0%}".format(self.config.cvar_alpha): row.get("CVaR", np.nan),
                    "Kelly": row.get("Kelly", np.nan),
                    "Sharpe": row.get("Sharpe", np.nan),
                    "Score": row.get("Score", np.nan),
                    "Notes": "Kelly-scaled allocation",
                }
            )
            total_assigned += entries_for_lineup
        assignment_df = pd.DataFrame(assignments)
        if assignment_df.empty:
            exposure_summary = pd.DataFrame(columns=["Lineup_ID", "Entries_Assigned"])
        else:
            exposure_summary = (
                assignment_df.groupby("Lineup_ID")["Entries_Assigned"].sum().reset_index()
            )
        return PortfolioResult(assignment_df, exposure_summary)
