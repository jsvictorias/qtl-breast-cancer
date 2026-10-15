import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    recall_score,
    roc_curve,
)

from src.analysis.aggregate import load_preds, load_runs


def youden_threshold(probs: np.ndarray, labels: np.ndarray) -> float:
    fpr, tpr, thr = roc_curve(labels, probs)
    j = tpr - fpr
    t = thr[int(np.argmax(j))]
    return float(min(t, 1.0))  # roc_curve prepends inf


def threshold_metrics(probs, labels, thr) -> dict:
    p = (probs >= thr).astype(int)
    return {
        "macro_f1": float(f1_score(labels, p, average="macro")),
        "recall_malignant": float(recall_score(labels, p, pos_label=1)),
        "balanced_acc": float(balanced_accuracy_score(labels, p)),
    }


def calibration_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for r in df.itertuples():
        vp, vl = load_preds(r.run_dir, r.target, "val")  # type: ignore
        tp, tl = load_preds(r.run_dir, r.target, "test")  # type: ignore
        thr = youden_threshold(vp, vl)
        for rule, t in (("fixed_0.5", 0.5), ("calibrated", thr)):
            rows.append(
                {
                    "source": r.source,
                    "target": r.target,
                    "seed": r.seed,
                    "rule": rule,
                    "threshold": t,
                    "roc_auc": r.roc_auc,
                    "pr_auc": r.pr_auc,
                    **threshold_metrics(tp, tl, t),
                }
            )
    return pd.DataFrame(rows)


if __name__ == "__main__":
    runs = load_runs(Path(sys.argv[1]))
    tab = calibration_table(runs)
    summary = (
        tab.groupby(["source", "target", "rule"])[
            ["threshold", "macro_f1", "recall_malignant", "balanced_acc", "roc_auc"]
        ]
        .mean()
        .round(3)
    )
    pd.set_option("display.width", 200)
    print(summary.to_string())
    if len(sys.argv) > 2:
        tab.to_csv(sys.argv[2], index=False)
