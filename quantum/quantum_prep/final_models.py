import json
import shutil
import sys
from pathlib import Path

import pandas as pd

from src.analysis.aggregate import load_runs

OUT = Path("artifacts/final_models")


def median_val_seed(runs: pd.DataFrame) -> pd.Series:
    r = runs.sort_values(["best_val_auc", "seed"]).reset_index(drop=True)
    return r.iloc[(len(r) - 1) // 2]


def freeze(sweep: Path) -> dict:
    runs = load_runs(sweep).drop_duplicates(["source", "seed"])
    chosen = {}
    for domain, g in runs.groupby("source"):
        pick = median_val_seed(g)
        run_dir = Path(pick.run_dir)
        meta = json.loads((run_dir / "metrics.json").read_text())
        dest = OUT / domain  # type: ignore
        dest.mkdir(parents=True, exist_ok=True)
        shutil.copy(run_dir / "model.pt", dest / "model.pt")
        info = {
            "source": domain,
            "seed": int(pick.seed),
            "best_val_auc": float(pick.best_val_auc),
            "val_auc_median_rule": "lower median of best_val_auc over seeds",
            "n_seeds": int(len(g)),
            "val_auc_min_max": [
                float(g.best_val_auc.min()),
                float(g.best_val_auc.max()),
            ],
            "norm": meta["norm"],
            "commit": meta["commit"],
            "src_run_dir": str(run_dir),
        }
        (dest / "info.json").write_text(json.dumps(info, indent=2))
        chosen[domain] = info
    return chosen


if __name__ == "__main__":
    for d, i in freeze(Path(sys.argv[1])).items():
        print(d, "seed", i["seed"], "val_auc", round(i["best_val_auc"], 3), i["norm"])
