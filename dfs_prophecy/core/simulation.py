from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from dfs_prophecy.config import SimulationConfig


@dataclass
class SimulationResult:
    lineup_metrics: pd.DataFrame
    contest_metrics: pd.DataFrame


class ContestSimulator:
    def __init__(self, config: SimulationConfig, rng: np.random.Generator) -> None:
        self.config = config
        self.rng = rng

    def run(self, lineups: pd.DataFrame, projections: pd.DataFrame) -> SimulationResult:
        if lineups.empty:
            raise ValueError("No lineups provided for simulation")
        merged = self._merge_lineups(lineups, projections)
        scenario_scores = self._simulate_scores(merged)
        payout_table = self._build_payout_table()
        lineup_metrics = self._compute_lineup_metrics(lineups, scenario_scores, payout_table)
        contest_metrics = self._aggregate_contest(lineup_metrics)
        return SimulationResult(lineup_metrics, contest_metrics)

    def _merge_lineups(self, lineups: pd.DataFrame, projections: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        indexed = projections.set_index("DFS_ID")
        data: Dict[str, pd.DataFrame] = {}
        for _, row in lineups.iterrows():
            players = row["Player_IDs"]
            df = pd.DataFrame(
                {
                    "DFS_ID": players,
                    "Mean": indexed.loc[players, "Projection"].values,
                    "Stdev": indexed.loc[players, "Stdev"].values,
                    "Team": indexed.loc[players, "Team"].values,
                    "Opponent": indexed.loc[players, "Opponent"].values,
                    "Position": indexed.loc[players, "Position"].values,
                }
            )
            data[row["Lineup_ID"]] = df
        return data

    def _simulate_scores(self, lineup_tables: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        trials = self.config.num_trials
        lineup_scores = {}
        for lineup_id, table in lineup_tables.items():
            means = table["Mean"].to_numpy()
            stdevs = table["Stdev"].to_numpy()
            cov = np.diag(stdevs ** 2)
            teams = table["Team"].tolist()
            opponents = table["Opponent"].tolist()
            positions = table["Position"].tolist()
            for i in range(len(means)):
                for j in range(i + 1, len(means)):
                    corr = 0.0
                    if teams[i] and teams[j] and teams[i] == teams[j]:
                        corr = 0.25
                        if positions[i].startswith("QB") or positions[j].startswith("QB"):
                            corr += 0.1
                    elif teams[i] and opponents[j] and teams[i] == opponents[j]:
                        corr = 0.1
                    elif teams[j] and opponents[i] and teams[j] == opponents[i]:
                        corr = 0.1
                    cov[i, j] = corr * stdevs[i] * stdevs[j]
                    cov[j, i] = cov[i, j]
            samples = self.rng.multivariate_normal(mean=means, cov=cov, size=trials)
            lineup_scores[lineup_id] = samples.sum(axis=1)
        return pd.DataFrame(lineup_scores)

    def _compute_lineup_metrics(
        self,
        lineups: pd.DataFrame,
        scenario_scores: pd.DataFrame,
        payout_table: pd.DataFrame,
    ) -> pd.DataFrame:
        metrics: List[Dict[str, float]] = []
        entry_fee = self.config.entry_fee
        contest_size = self.config.contest_size
        payout_ranks = payout_table["Rank"].to_numpy()
        payout_values = payout_table["Payout"].to_numpy()
        signature_counts = lineups.groupby("Dup_Signature")["Lineup_ID"].count().to_dict()
        dup_lookup = (
            lineups.set_index("Lineup_ID")["Dup_Signature"].map(signature_counts).to_dict()
        )
        own_lookup = (
            lineups.set_index("Lineup_ID")["Sum_Own"].to_dict()
            if "Sum_Own" in lineups.columns
            else {}
        )
        lineup_size = len(lineups.iloc[0]["Player_IDs"]) if not lineups.empty else 0
        for lineup_id in lineups["Lineup_ID"]:
            scores = scenario_scores[lineup_id].to_numpy()
            mean_score = float(scores.mean())
            score_std = float(scores.std(ddof=1))
            percentiles = self._score_percentiles(scores)
            ranks = np.clip(
                np.floor((1 - percentiles) * (contest_size - 1)) + 1,
                1,
                contest_size,
            )
            payouts = np.interp(ranks, payout_ranks, payout_values)
            profits = payouts - entry_fee
            ev_profit = float(profits.mean())
            profit_std = float(profits.std(ddof=1))
            tail = max(1, int(math.ceil(self.config.cvar_alpha * len(profits))))
            worst = np.sort(profits)[:tail]
            cvar = float(worst.mean())
            prob_cash = float((payouts > 0).mean())
            prob_top1 = float((ranks <= max(1, contest_size * 0.01)).mean())
            prob_top_point_one = float((ranks <= max(1, contest_size * 0.001)).mean())
            mean_rank = float(ranks.mean())
            kelly_denom = float(abs(worst.min()) + entry_fee)
            kelly = float(np.clip(ev_profit / kelly_denom if kelly_denom else 0, 0, 1))
            sharpe = float(ev_profit / (profit_std + 1e-6))
            expected_dups = 0.0
            if lineup_id in own_lookup and lineup_size:
                avg_own = max(own_lookup[lineup_id] / lineup_size, 1.0)
                expected_dups = float(
                    (avg_own / 100.0) ** lineup_size * self.config.contest_size
                )
            metrics.append(
                {
                    "Lineup_ID": lineup_id,
                    "MeanScore": mean_score,
                    "ScoreStdev": score_std,
                    "EV": ev_profit,
                    "ROI": ev_profit / entry_fee if entry_fee else 0.0,
                    "Prob_Cash": prob_cash,
                    "Prob_Top1Pct": prob_top1,
                    "Prob_Top0.1Pct": prob_top_point_one,
                    "CVaR": cvar,
                    "MeanRank": mean_rank,
                    "FinishPercentile": 1 - mean_rank / contest_size,
                    "Kelly": kelly,
                    "Sharpe": sharpe,
                    "DupGroupSize": float(dup_lookup.get(lineup_id, 1)),
                    "ExpectedFieldDup": expected_dups,
                }
            )
        return pd.DataFrame(metrics)

    def _aggregate_contest(self, lineup_metrics: pd.DataFrame) -> pd.DataFrame:
        entry_fee = self.config.entry_fee
        total_entries = len(lineup_metrics)
        total_cost = total_entries * entry_fee
        total_profit = lineup_metrics["EV"].sum()
        roi = total_profit / total_cost if total_cost else 0.0
        contest_df = pd.DataFrame(
            {
                "Entries": [total_entries],
                "TotalCost": [total_cost],
                "TotalEV": [total_profit],
                "ROI": [roi],
                "AvgProbCash": [lineup_metrics["Prob_Cash"].mean() if not lineup_metrics.empty else 0.0],
                "AvgSharpe": [lineup_metrics["Sharpe"].mean() if not lineup_metrics.empty else 0.0],
            }
        )
        return contest_df

    def _build_payout_table(self) -> pd.DataFrame:
        if self.config.payout_curve and Path(self.config.payout_curve).exists():
            table = pd.read_csv(self.config.payout_curve)
        else:
            table = self._synthetic_payout_table()
        columns = {col.lower(): col for col in table.columns}
        rank_col = columns.get("rank") or columns.get("place") or columns.get("position")
        payout_col = columns.get("payout") or columns.get("prize") or columns.get("amount")
        if not rank_col or not payout_col:
            raise ValueError("Payout table must include rank and payout columns")
        payout_table = table[[rank_col, payout_col]].rename(
            columns={rank_col: "Rank", payout_col: "Payout"}
        )
        payout_table = payout_table.sort_values("Rank").reset_index(drop=True)
        if payout_table.iloc[-1]["Rank"] < self.config.contest_size:
            payout_table = pd.concat(
                [
                    payout_table,
                    pd.DataFrame(
                        {"Rank": [self.config.contest_size], "Payout": [0.0]}
                    ),
                ],
                ignore_index=True,
            )
        if payout_table.iloc[0]["Rank"] > 1:
            payout_table = pd.concat(
                [pd.DataFrame({"Rank": [1], "Payout": [payout_table.iloc[0]["Payout"]]}), payout_table],
                ignore_index=True,
            )
        return payout_table

    def _synthetic_payout_table(self) -> pd.DataFrame:
        size = self.config.contest_size
        prize_pool = self.config.entry_fee * size * (1 - self.config.rake)
        paid_spots = max(1, min(size, int(size * 0.2)))
        ranks = np.arange(1, paid_spots + 1)
        decay = np.power(np.linspace(1.0, 0.15, paid_spots), 1.2)
        payouts = prize_pool * decay / decay.sum()
        payout_table = pd.DataFrame({"Rank": ranks, "Payout": payouts})
        payout_table.loc[payout_table["Payout"] < self.config.entry_fee, "Payout"] = (
            self.config.entry_fee
        )
        payout_table = pd.concat(
            [
                payout_table,
                pd.DataFrame({"Rank": [size], "Payout": [0.0]}),
            ],
            ignore_index=True,
        )
        return payout_table

    def _score_percentiles(self, scores: np.ndarray) -> np.ndarray:
        mean = float(scores.mean())
        std = float(scores.std(ddof=1))
        if std == 0:
            return np.full_like(scores, 0.5, dtype=float)
        z = (scores - mean) / std
        return 0.5 * (1 + math.erf(z / math.sqrt(2)))
