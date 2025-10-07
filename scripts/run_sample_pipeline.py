from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dfs_prophecy.cli import main as cli_main


if __name__ == "__main__":
    args = [
        "--config",
        "configs/default.yaml",
        "--projection-sources",
        "data/sample_inputs/nfl_features.csv",
        "--ownership-sources",
        "data/sample_inputs/nfl_ownership.csv",
    ]
    sys.argv = ["run_sample_pipeline.py", *args]
    cli_main()
