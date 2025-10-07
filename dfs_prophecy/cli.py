from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

from dfs_prophecy.config import PipelineConfig, load_config
from dfs_prophecy.core.pipeline import PipelineRunner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the DFS Prophecy pipeline")
    parser.add_argument("--config", type=Path, default=None, help="Path to YAML config")
    parser.add_argument(
        "--projection-sources",
        type=Path,
        nargs="+",
        required=True,
        help="CSV files with projection features",
    )
    parser.add_argument(
        "--ownership-sources",
        type=Path,
        nargs="+",
        required=True,
        help="CSV files containing ownership estimates",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional override for output directory",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    runner = PipelineRunner(config, args.config)
    artifacts = runner.run(args.projection_sources, args.ownership_sources, args.output_dir)
    print("Pipeline complete. Artifacts:")
    print(f"  Projections: {artifacts.projection_path}")
    print(f"  Lineups: {artifacts.lineup_path}")
    print(f"  Simulation: {artifacts.simulation_path}")
    print(f"  Portfolio: {artifacts.portfolio_path}")


if __name__ == "__main__":
    main()
