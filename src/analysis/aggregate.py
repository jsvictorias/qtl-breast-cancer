import json
import sys
from pathlib import Path

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
                    **m,
                }
            )
    return pd.DataFrame(rows)


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
