from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


def _parse_value(raw: str) -> Any:
    raw = raw.strip()
    if not raw:
        return ""
    if raw.lower() in {"true", "false"}:
        return raw.lower() == "true"
    try:
        if "." in raw:
            return float(raw)
        return int(raw)
    except ValueError:
        return raw


def _simple_yaml_load(text: str) -> Dict[str, Any]:
    root: Dict[str, Any] = {}
    stack = [(0, root)]
    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        key, _, remainder = line.strip().partition(":")
        value = remainder.strip()
        while stack and indent < stack[-1][0]:
            stack.pop()
        current = stack[-1][1]
        if value:
            current[key] = _parse_value(value)
        else:
            new_dict: Dict[str, Any] = {}
            current[key] = new_dict
            stack.append((indent + 2, new_dict))
    return root


@dataclass
class ProjectionConfig:
    sport: str = "nfl"
    site: str = "dk"
    projection_sources: List[Path] = field(default_factory=list)
    ownership_sources: List[Path] = field(default_factory=list)
    blend_weights: Optional[List[float]] = None
    ceiling_quantile: float = 0.9
    floor_quantile: float = 0.1
    stdev_floor: float = 1.5
    wr_spike_multiplier: float = 1.3
    qb_floor_boost: float = 1.05
    injury_multipliers: Dict[str, float] = field(
        default_factory=lambda: {
            "Out": 0.0,
            "Doubtful": 0.4,
            "Questionable": 0.85,
            "Probable": 1.0,
        }
    )
    historical_regression: float = 0.15


@dataclass
class LineupConfig:
    sport: str = "nfl"
    num_lineups: int = 500
    salary_cap: int = 50000
    positions: Dict[str, int] = field(default_factory=lambda: {
        "QB": 1,
        "RB": 2,
        "WR": 3,
        "TE": 1,
        "DST": 1,
        "FLEX": 1,
    })
    flex_positions: List[str] = field(default_factory=lambda: ["RB", "WR", "TE"])
    max_players_per_team: Optional[int] = None
    min_uniques: int = 2
    max_dup_signature: Optional[int] = None
    leverage_weight: float = 0.15
    ownership_weight: float = 0.1
    max_exposure_per_player: Optional[float] = None
    exposure_smoothing: float = 0.15
    force_primary_stack: bool = True
    force_bring_back: bool = True
    correlation_matrix: Optional[Path] = None


@dataclass
class SimulationConfig:
    num_trials: int = 5000
    bankroll: float = 1000.0
    rake: float = 0.15
    contest_size: int = 100000
    max_entries: int = 150
    entry_fee: float = 20.0
    payout_curve: Optional[Path] = None
    cvar_alpha: float = 0.05


@dataclass
class PortfolioConfig:
    bankroll: float = 1000.0
    cvar_alpha: float = 0.05
    min_cvar: float = -100.0
    max_exposure_per_lineup: int = 10
    entry_fee: float = 20.0
    kelly_multiplier: float = 1.0
    risk_aversion: float = 0.6
    min_allocation_per_lineup: int = 1
    max_total_entries: Optional[int] = None


@dataclass
class PipelineConfig:
    seed: int = 7
    output_dir: Path = Path("runs")
    projection: ProjectionConfig = field(default_factory=ProjectionConfig)
    lineup: LineupConfig = field(default_factory=LineupConfig)
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    portfolio: PortfolioConfig = field(default_factory=PortfolioConfig)

    @staticmethod
    def from_yaml(path: Path) -> "PipelineConfig":
        data = _simple_yaml_load(path.read_text())
        return PipelineConfig(
            seed=data.get("seed", 7),
            output_dir=Path(data.get("output_dir", "runs")),
            projection=ProjectionConfig(**data.get("projection", {})),
            lineup=LineupConfig(**data.get("lineup", {})),
            simulation=SimulationConfig(**data.get("simulation", {})),
            portfolio=PortfolioConfig(**data.get("portfolio", {})),
        )


def load_config(path: Optional[Path] = None) -> PipelineConfig:
    if path is None:
        return PipelineConfig()
    return PipelineConfig.from_yaml(path)


def dump_default_config(path: Path) -> None:
    config = PipelineConfig()
    lines = [
        f"seed: {config.seed}",
        f"output_dir: {config.output_dir}",
        "projection:",
        f"  sport: {config.projection.sport}",
        f"  site: {config.projection.site}",
        "lineup:",
        f"  num_lineups: {config.lineup.num_lineups}",
        f"  salary_cap: {config.lineup.salary_cap}",
        "simulation:",
        f"  num_trials: {config.simulation.num_trials}",
        f"  entry_fee: {config.simulation.entry_fee}",
        "portfolio:",
        f"  bankroll: {config.portfolio.bankroll}",
        f"  cvar_alpha: {config.portfolio.cvar_alpha}",
    ]
    path.write_text("\n".join(lines) + "\n")
