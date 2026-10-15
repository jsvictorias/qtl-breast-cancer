import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def load_runs(sweep_dir: Path) -> pd.DataFrame:
    rows = []
    for f in sweep_dir.rglob("metrics.json"):
        r = json.loads(f.read_text())
        for target, m in r["targets"].items():
            rows.append(
                {
                    "source": r["source"],
                    "seed": r["seed"],
                    "target": target,
                    "commit": r["commit"],
                    "best_epoch": r.get("best_epoch"),
                    "best_val_auc": r.get("best_val_auc"),
                    "run_dir": str(f.parent),
                    **m,
                }
            )
    return pd.DataFrame(rows)


def load_preds(run_dir: Path | str, target: str, split: str = "test"):
    name = f"preds_{target}.npz" if split == "test" else f"preds_val_{target}.npz"
    z = np.load(Path(run_dir) / name)
    return z["probs"], z["labels"]


if __name__ == "__main__":
    sweep = Path(sys.argv[1])
    df = load_runs(sweep)
    df.to_csv(sweep / "results_tidy.csv", index=False)
    matrix = (
        df.groupby(["source", "target"])["roc_auc"]
        .agg(["mean", "std"])
        .unstack("target")
    )
    print(matrix.round(3))
