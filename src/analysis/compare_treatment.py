import sys
from pathlib import Path

import pandas as pd
from scipy.stats import wilcoxon

from src.analysis.aggregate import load_runs

METRICS = ["roc_auc", "pr_auc"]


def paired_table(raw: pd.DataFrame, treated: pd.DataFrame) -> pd.DataFrame:
    keys = ["source", "target", "seed"]
    seeds = sorted(set(raw.seed) & set(treated.seed))
    m = raw[raw.seed.isin(seeds)].merge(
        treated[treated.seed.isin(seeds)], on=keys, suffixes=("_raw", "_trt")
    )
    rows = []
    for (src, tgt), g in m.groupby(["source", "target"]):
        for met in METRICS:
            d = g[f"{met}_trt"] - g[f"{met}_raw"]
            stat = wilcoxon(d, alternative="two-sided") if d.any() else None
            rows.append(
                {
                    "source": src,
                    "target": tgt,
                    "metric": met,
                    "n_seeds": len(g),
                    "raw_mean": g[f"{met}_raw"].mean(),
                    "trt_mean": g[f"{met}_trt"].mean(),
                    "delta_mean": d.mean(),
                    "wins_trt": int((d > 0).sum()),
                    "p_value": stat.pvalue if stat else 1.0,  # type: ignore
                }
            )
    out = pd.DataFrame(rows)
    order = out.p_value.sort_values().index
    m_tests = len(out)
    adj, prev = {}, 0.0
    for rank, idx in enumerate(order):
        prev = max(prev, min(1.0, out.p_value[idx] * (m_tests - rank)))
        adj[idx] = prev
    out["p_holm"] = pd.Series(adj)
    return out


if __name__ == "__main__":
    raw = load_runs(Path(sys.argv[1]))
    treated = load_runs(Path(sys.argv[2]))
    res = paired_table(raw, treated)
    pd.set_option("display.width", 200)
    print(res.round(4).to_string(index=False))
    if len(sys.argv) > 3:
        res.to_csv(sys.argv[3], index=False)
