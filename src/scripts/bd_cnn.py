from __future__ import annotations

import sys
from pathlib import Path

from classical.runner import run_cli

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

if __name__ == "__main__":
    run_cli(
        dataset_config_path=PROJECT_ROOT / "configs" / "dataset" / "big.yaml",
        model_config_path=PROJECT_ROOT / "configs" / "model" / "lenet.yaml",
        project_root=PROJECT_ROOT,
    )
