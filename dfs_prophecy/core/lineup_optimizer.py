from __future__ import annotations

import itertools
import math
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from dfs_prophecy.config import LineupConfig


LINEUP_COLUMNS = [
    "Lineup_ID",
    "Player_IDs",
    "Total_Salary",
    "Proj_Pts",
    "Sum_Own",
    "Leverage",
    "Stack_Tags",
    "Corr_Score",
    "Dup_Signature",
]


@dataclass
class LineupResult:
    lineups: pd.DataFrame
    summary: pd.DataFrame


class LineupGenerator:
    def __init__(self, config: LineupConfig, rng: np.random.Generator) -> None:
        self.config = config
        self.rng = rng
        self._correlation_matrix: Optional[pd.DataFrame] = None
        if self.config.sport.lower() != "nfl":
            self.config.force_primary_stack = False
            self.config.force_bring_back = False
        if config.correlation_matrix:
            try:
                corr = pd.read_csv(config.correlation_matrix)
            except FileNotFoundError:
                corr = None
            if corr is not None:
                if "DFS_ID" in corr.columns:
                    corr = corr.set_index("DFS_ID")
                if set(corr.columns) == set(corr.index):
                    self._correlation_matrix = corr

    def build(self, projections: pd.DataFrame, ownership: pd.DataFrame) -> LineupResult:
        merged = projections.merge(ownership, on="DFS_ID", how="left")
        merged["Est_Own"] = merged["Est_Own"].fillna(merged["Est_Own"].median())
        merged["Est_Own"].fillna(0.08, inplace=True)
        exposure_limits = self._exposure_limits(merged)
        valid_lineups: List[Dict[str, object]] = []
        salary_cap = self.config.salary_cap
        roster_structure = self._expand_roster_positions(self.config.positions)
        exposure_counts: Dict[str, int] = {}
        attempts = 0
        max_attempts = max(self.config.num_lineups * 50, 10000)
        while len(valid_lineups) < self.config.num_lineups and attempts < max_attempts:
            attempts += 1
            lineup_players = self._sample_lineup(
                merged, roster_structure, exposure_counts, exposure_limits
            )
            if lineup_players is None:
                continue
            lineup_df = merged.loc[merged["DFS_ID"].isin(lineup_players)]
            if lineup_df["Salary"].sum() > salary_cap:
                continue
            if not self._validate_lineup(
                lineup_df,
                lineup_players,
                valid_lineups,
                exposure_counts,
                exposure_limits,
            ):
                continue
            dup_signature = ",".join(sorted(lineup_players))
            if self.config.max_dup_signature is not None:
                seen = sum(1 for l in valid_lineups if l["Dup_Signature"] == dup_signature)
                if seen >= self.config.max_dup_signature:
                    continue
            sum_own = lineup_df["Est_Own"].sum()
            proj_pts = lineup_df["Projection"].sum()
            leverage = proj_pts - self.config.leverage_weight * sum_own
            corr_score = self._estimate_correlation(lineup_df)
            tags = self._stack_tags(lineup_df)
            valid_lineups.append(
                {
                    "Lineup_ID": f"LU{len(valid_lineups)+1:05d}",
                    "Player_IDs": lineup_players,
                    "Total_Salary": float(lineup_df["Salary"].sum()),
                    "Proj_Pts": float(proj_pts),
                    "Sum_Own": float(sum_own),
                    "Leverage": float(leverage),
                    "Stack_Tags": tags,
                    "Corr_Score": float(corr_score),
                    "Dup_Signature": dup_signature,
                }
            )
            for pid in lineup_players:
                exposure_counts[pid] = exposure_counts.get(pid, 0) + 1
        lineups_df = pd.DataFrame(valid_lineups, columns=LINEUP_COLUMNS)
        summary = self._exposure_summary(lineups_df)
        return LineupResult(lineups_df, summary)

    def _sample_lineup(
        self,
        player_table: pd.DataFrame,
        roster_structure: Sequence[str],
        exposure_counts: Dict[str, int],
        exposure_limits: Dict[str, int],
    ) -> Optional[List[str]]:
        choices: List[str] = []
        remaining = roster_structure.copy()
        pool = player_table.copy()
        for pos in remaining:
            eligible = self._eligible_players(pool, pos)
            if eligible.empty:
                return None
            weights = eligible["Projection"] - self.config.ownership_weight * eligible["Est_Own"]
            weights = np.maximum(weights, 0.01)
            weights = self._apply_exposure_penalty(
                eligible, weights, exposure_counts, exposure_limits
            )
            if weights.sum() <= 0:
                return None
            weights = weights / weights.sum()
            selection = self.rng.choice(eligible["DFS_ID"], p=weights)
            choices.append(selection)
            pool = pool[pool["DFS_ID"] != selection]
        return choices

    def _expand_roster_positions(self, positions: Dict[str, int]) -> List[str]:
        roster: List[str] = []
        for pos, count in positions.items():
            roster.extend([pos] * count)
        return roster

    def _eligible_players(self, pool: pd.DataFrame, position: str) -> pd.DataFrame:
        if position.upper() == "FLEX":
            eligible = pool[pool["Position"].isin(self.config.flex_positions)]
        else:
            eligible = pool[pool["Position"].str.contains(position)]
        return eligible

    def _apply_exposure_penalty(
        self,
        eligible: pd.DataFrame,
        base_weights: pd.Series,
        exposure_counts: Dict[str, int],
        exposure_limits: Dict[str, int],
    ) -> pd.Series:
        if not exposure_limits:
            return base_weights
        penalties = []
        for pid in eligible["DFS_ID"]:
            limit = exposure_limits.get(pid)
            used = exposure_counts.get(pid, 0)
            if limit is None or limit <= 0:
                penalties.append(1.0)
                continue
            remaining = max(limit - used, 0)
            penalty = remaining / limit
            penalty = penalty ** (1 - self.config.exposure_smoothing)
            penalties.append(penalty)
        penalties_series = pd.Series(penalties, index=eligible.index)
        adjusted = base_weights * penalties_series.clip(lower=0.0)
        return adjusted

    def _validate_lineup(
        self,
        lineup_df: pd.DataFrame,
        lineup_players: List[str],
        existing_lineups: List[Dict[str, object]],
        exposure_counts: Dict[str, int],
        exposure_limits: Dict[str, int],
    ) -> bool:
        if self.config.max_players_per_team is not None:
            team_counts = lineup_df["Team"].value_counts()
            if (team_counts > self.config.max_players_per_team).any():
                return False
        if not self._validate_uniques(lineup_players, existing_lineups):
            return False
        if not self._validate_exposure(lineup_players, exposure_counts, exposure_limits):
            return False
        if self.config.force_primary_stack or self.config.force_bring_back:
            tags = self._stack_tags(lineup_df)
            if self.config.force_primary_stack and "QB+" not in tags and any(
                lineup_df["Position"].str.startswith("QB")
            ):
                return False
            if self.config.force_bring_back and "BringBack" not in tags and any(
                lineup_df["Position"].str.startswith("QB")
            ):
                return False
        return True

    def _validate_uniques(
        self, lineup_players: List[str], existing_lineups: List[Dict[str, object]]
    ) -> bool:
        required = max(self.config.min_uniques, 1)
        candidate = set(lineup_players)
        for lineup in existing_lineups:
            current = set(lineup["Player_IDs"])
            uniques = len(candidate - current)
            if uniques < required:
                return False
        return True

    def _validate_exposure(
        self,
        lineup_players: List[str],
        exposure_counts: Dict[str, int],
        exposure_limits: Dict[str, int],
    ) -> bool:
        if not exposure_limits:
            return True
        for pid in lineup_players:
            limit = exposure_limits.get(pid)
            if limit is None:
                continue
            if exposure_counts.get(pid, 0) + 1 > limit:
                return False
        return True

    def _exposure_limits(self, player_table: pd.DataFrame) -> Dict[str, int]:
        if self.config.max_exposure_per_player is None:
            return {}
        limit = max(
            1, math.ceil(self.config.max_exposure_per_player * self.config.num_lineups)
        )
        return {pid: limit for pid in player_table["DFS_ID"]}

    def _estimate_correlation(self, lineup_df: pd.DataFrame) -> float:
        if self._correlation_matrix is not None:
            ids = [pid for pid in lineup_df["DFS_ID"] if pid in self._correlation_matrix.index]
            if len(ids) >= 2:
                sub = self._correlation_matrix.loc[ids, ids]
                tril = sub.values[np.tril_indices(len(ids), k=-1)]
                tril = tril[~np.isnan(tril)]
                if len(tril):
                    return float(np.mean(tril))
        if "Team" not in lineup_df.columns or "Opponent" not in lineup_df.columns:
            return 0.0
        same_team_pairs = 0
        opp_pairs = 0
        total_pairs = 0
        for (_, a), (_, b) in itertools.combinations(lineup_df.iterrows(), 2):
            total_pairs += 1
            if a["Team"] == b["Team"]:
                same_team_pairs += 1
            if a["Team"] == b["Opponent"]:
                opp_pairs += 1
        if total_pairs == 0:
            return 0.0
        return 0.6 * same_team_pairs / total_pairs + 0.4 * opp_pairs / total_pairs

    def _stack_tags(self, lineup_df: pd.DataFrame) -> str:
        if "Position" not in lineup_df.columns:
            return ""
        qb_team = None
        for _, row in lineup_df.iterrows():
            if row["Position"].startswith("QB"):
                qb_team = row["Team"]
                break
        if qb_team is None:
            return ""
        pass_catchers = lineup_df[(lineup_df["Team"] == qb_team) & lineup_df["Position"].str.startswith(("WR", "TE"))]
        bring_backs = lineup_df[(lineup_df["Opponent"] == qb_team) & lineup_df["Position"].str.startswith(("WR", "TE", "RB"))]
        tags: List[str] = []
        if len(pass_catchers) >= 2:
            tags.append(f"QB+{len(pass_catchers)}")
        if len(bring_backs) >= 1:
            tags.append(f"BringBack{len(bring_backs)}")
        return "+".join(tags)

    def _exposure_summary(self, lineups_df: pd.DataFrame) -> pd.DataFrame:
        if lineups_df.empty:
            return pd.DataFrame(columns=["DFS_ID", "Exposure", "Pct"])
        player_counts: Counter[str] = Counter()
        for players in lineups_df["Player_IDs"]:
            player_counts.update(players)
        summary = pd.DataFrame(
            {
                "DFS_ID": list(player_counts.keys()),
                "Exposure": list(player_counts.values()),
            }
        )
        summary["Pct"] = summary["Exposure"] / len(lineups_df)
        return summary.sort_values("Pct", ascending=False).reset_index(drop=True)
