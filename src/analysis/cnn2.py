import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from src.analysis.aggregate import load_preds
from src.modeling.engine import compute_metrics

DOMAINS = ["small", "medium", "big"]
METRICS = ["roc_auc", "pr_auc", "recall_malignant", "macro_f1", "balanced_acc"]
N_TRAIN = {"small": 427, "medium": 1313, "big": 1567}


def load_tidy(sweep: Path, threshold: float = 0.5) -> pd.DataFrame:
    rows = []
    for f in sorted(sweep.rglob("metrics.json")):
        r = json.loads(f.read_text())
        for tgt in r["targets"]:
            probs, labels = load_preds(f.parent, tgt, "test")
            rows.append(
                {
                    "source": r["source"],
                    "target": tgt,
                    "seed": r["seed"],
                    "best_epoch": r["best_epoch"],
                    "best_val_auc": r["best_val_auc"],
                    "train_fraction": r.get("train_fraction", 1.0),
                    "n_train": r.get("n_train", N_TRAIN[r["source"]]),
                    "run_dir": str(f.parent),
                    **compute_metrics(probs, labels, threshold),
                }
            )
    return pd.DataFrame(rows)


def mean_sd_ci(df: pd.DataFrame, by: list[str], metrics=METRICS) -> pd.DataFrame:
    out = []
    for key, g in df.groupby(by):
        key = key if isinstance(key, tuple) else (key,)
        row = dict(zip(by, key, strict=True))
        n = len(g)
        row["n"] = n
        for m in metrics:
            sd = g[m].std(ddof=1)
            row[f"{m}_mean"] = g[m].mean()
            row[f"{m}_sd"] = sd
            row[f"{m}_ci95"] = stats.t.ppf(0.975, n - 1) * sd / np.sqrt(n)
        out.append(row)
    return pd.DataFrame(out)


def fmt_table(summary: pd.DataFrame, metrics=METRICS) -> pd.DataFrame:
    t = summary[
        [c for c in summary.columns if not c.endswith(("_mean", "_sd", "_ci95"))]
    ].copy()
    for m in metrics:
        t[m] = [
            f"{a:.3f} ± {s:.3f} (IC ±{c:.3f})"
            for a, s, c in zip(
                summary[f"{m}_mean"],
                summary[f"{m}_sd"],
                summary[f"{m}_ci95"],
                strict=True,
            )
        ]
    return t
