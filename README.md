# DFS Prophecy

DFS Prophecy provides an end-to-end Daily Fantasy Sports (DFS) experimentation pipeline that spans projections, lineup generation, contest simulation, and portfolio allocation. The repository is structured so that each stage can be benchmarked independently or chained together into a reproducible workflow.

## Features

- **Player projections** blend feature-derived baselines with optional third-party feeds while applying injury, pace, and usage adjustments to produce calibrated mean/floor/ceiling ranges for NFL (default), NBA, and PGA.
- **Lineup generator** samples contest-legal rosters with stacking enforcement, salary/ownership/leverage objectives, correlation scoring, and configurable exposure limits.
- **Contest simulator** injects team-level correlation, maps simulated outcomes to realistic payout curves, and reports score moments, ROI, CVaR, duplication heuristics, and tail probabilities.
- **Portfolio optimizer** Kelly-scales the simulated EV subject to bankroll, CVaR, and exposure caps, producing contest assignments ranked by a configurable risk score.
- **Configuration first** design with YAML/CLI ergonomics, seeded randomness, and timestamped artifact folders.
- **Sample data + script** to showcase the end-to-end flow without needing live data services.

## Installation

```bash
pip install -e .
```

The project depends on Python 3.10+ and only requires `numpy` and `pandas` for the base workflow. Optional extras such as `ray`, `lightgbm`, or `xgboost` can be installed via `pip install -e .[ray]` or `pip install -e .[ml]`.

## Running the Sample Pipeline

A minimal NFL example is included under `data/sample_inputs/`. Execute the end-to-end pipeline with:

```bash
python scripts/run_sample_pipeline.py
```

This command reads `configs/default.yaml`, generates projections from `data/sample_inputs/nfl_features.csv`, combines ownership from `data/sample_inputs/nfl_ownership.csv`, and writes all artifacts to a timestamped directory under `runs/`.

### Direct CLI Usage

You can also call the CLI directly:

```bash
python -m dfs_prophecy.cli \
  --config configs/default.yaml \
  --projection-sources data/sample_inputs/nfl_features.csv \
  --ownership-sources data/sample_inputs/nfl_ownership.csv
```

### Outputs

Each run produces the following artifacts:

- `projections.csv` – Site-ready projections with calibrated mean/floor/ceiling/stdev plus injury-aware notes.
- `projection_diagnostics.csv` – Aggregated diagnostics by position including average ceiling gaps and floor buffers.
- `lineups.csv` – Generated lineups with salary, ownership, leverage, stacking, correlation scores, and duplication signatures.
- `lineup_exposure.csv` – Exposure summary by player.
- `simulation_results.csv` – Monte Carlo metrics per lineup, including mean score, ROI, CVaR, Sharpe, duplication heuristics, and tail probabilities.
- `contest_summary.csv` – Aggregated contest-level EV/ROI snapshot plus average risk statistics.
- `portfolio_ranked.csv` – Portfolio assignments with Kelly sizing, Sharpe, risk score, and CVaR gating.
- `portfolio_exposure.csv` – Entry count per lineup.
- `metadata.json` – Run metadata linking back to the configuration.

## Extending the Pipeline

- Extend `dfs_prophecy/core/projections.py` with model-based estimators, blending external sources, and calibration logic.
- Replace `dfs_prophecy/core/lineup_optimizer.py` with an ILP or heuristic search to reach 100k-lineup scale; the class is designed for dependency injection of better samplers.
- Upgrade `dfs_prophecy/core/simulation.py` to incorporate structured covariance matrices, adversarial fields, and payout curve inputs.
- Enhance `dfs_prophecy/core/portfolio.py` with knapsack solvers, CVaR optimization, or diversification constraints.

## Reproducibility Notes

- Global randomness is coordinated through `PipelineConfig.seed` and `RNGManager`.
- Configuration files live under `configs/` and can be versioned per slate.
- All artifacts are emitted into `runs/<UTC timestamp>/` to keep experiments isolated.

## License

MIT
