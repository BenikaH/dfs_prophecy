from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

from dfs_prophecy.config import PipelineConfig
from dfs_prophecy.core.lineup_optimizer import LineupGenerator
from dfs_prophecy.core.portfolio import PortfolioOptimizer
from dfs_prophecy.core.projections import ProjectionGenerator
from dfs_prophecy.core.simulation import ContestSimulator
from dfs_prophecy.utils.randomness import RNGManager


@dataclass
class PipelineArtifacts:
    projection_path: Path
    lineup_path: Path
    simulation_path: Path
    portfolio_path: Path


class PipelineRunner:
    def __init__(self, config: PipelineConfig, config_path: Optional[Path] = None) -> None:
        self.config = config
        self.config_path = config_path
        self.rng = RNGManager(config.seed)
        # Ensure downstream components inherit sport/entry fee defaults
        if getattr(self.config.lineup, "sport", None) is None:
            self.config.lineup.sport = self.config.projection.sport
        else:
            self.config.lineup.sport = self.config.projection.sport
        if not getattr(self.config.portfolio, "entry_fee", None):
            self.config.portfolio.entry_fee = self.config.simulation.entry_fee

    def run(
        self,
        projection_sources: Iterable[Path],
        ownership_sources: Iterable[Path],
        output_dir: Optional[Path] = None,
    ) -> PipelineArtifacts:
        output_base = (output_dir or self.config.output_dir).joinpath(
            datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        )
        output_base.mkdir(parents=True, exist_ok=True)

        projection_generator = ProjectionGenerator(self.config.projection, self.rng.np_random)
        projection_result = projection_generator.generate(projection_sources)
        projection_path = output_base / "projections.csv"
        projection_result.data.to_csv(projection_path, index=False)
        projection_result.diagnostics.to_csv(output_base / "projection_diagnostics.csv", index=False)

        ownership_df = self._load_ownership(ownership_sources)
        lineup_generator = LineupGenerator(self.config.lineup, self.rng.np_random)
        lineup_result = lineup_generator.build(projection_result.data, ownership_df)
        lineup_path = output_base / "lineups.csv"
        lineup_result.lineups.to_csv(lineup_path, index=False)
        lineup_result.summary.to_csv(output_base / "lineup_exposure.csv", index=False)

        simulator = ContestSimulator(self.config.simulation, self.rng.np_random)
        simulation_result = simulator.run(lineup_result.lineups, projection_result.data)
        simulation_path = output_base / "simulation_results.csv"
        simulation_result.lineup_metrics.to_csv(simulation_path, index=False)
        simulation_result.contest_metrics.to_csv(output_base / "contest_summary.csv", index=False)

        optimizer = PortfolioOptimizer(self.config.portfolio, self.rng.np_random)
        portfolio_result = optimizer.optimize(simulation_result.lineup_metrics)
        portfolio_path = output_base / "portfolio_ranked.csv"
        portfolio_result.assignments.to_csv(portfolio_path, index=False)
        portfolio_result.exposure_summary.to_csv(output_base / "portfolio_exposure.csv", index=False)

        self._write_metadata(output_base)

        return PipelineArtifacts(
            projection_path=projection_path,
            lineup_path=lineup_path,
            simulation_path=simulation_path,
            portfolio_path=portfolio_path,
        )

    def _write_metadata(self, output_dir: Path) -> None:
        metadata = {
            "config_seed": self.config.seed,
            "config_path": str(self.config_path) if self.config_path else None,
        }
        pd.Series(metadata).to_json(output_dir / "metadata.json", indent=2)

    def _load_ownership(self, sources: Iterable[Path]) -> pd.DataFrame:
        frames = []
        for path in sources:
            frame = pd.read_csv(path)
            if "Source" not in frame.columns:
                frame["Source"] = path.stem
            frames.append(frame[["DFS_ID", "Est_Own", "Source"]])
        if not frames:
            raise ValueError("At least one ownership source required")
        combined = pd.concat(frames, ignore_index=True)
        ownership = (
            combined.groupby("DFS_ID")["Est_Own"].mean().reset_index()
        )
        return ownership
