from collections.abc import Callable

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from quantum.encoding import data
from quantum.encoding.encoders import ENCODERS, load_inputs
from quantum.encoding.properties import CONFIGS, label
from quantum.encoding.simulate import expectations, kernel, probabilities, states
from quantum.quantum_prep.extract import DOMAINS

REPORT = data.REPORT
C_SVM = (0.1, 1.0, 10.0, 100.0)
C_LOGISTIC = (0.01, 0.1, 1.0, 10.0, 100.0)
N_BOOT = 1000

Scorer = Callable[[str, str], np.ndarray]  # (domain, split) -> decision scores


def fast_auc(y: np.ndarray, s: np.ndarray) -> np.ndarray:
    r = rankdata(s, axis=1)
    pos = y.sum(1)
    neg = y.shape[1] - pos
    return ((r * y).sum(1) - pos * (pos + 1) / 2) / (pos * neg)


def bootstrap_auc_ci(
    y: np.ndarray, s: np.ndarray, n_boot: int = N_BOOT, seed: int = 0
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(y), (n_boot, len(y)))
    yb = y[idx]
    ok = (yb.sum(1) > 0) & (yb.sum(1) < len(y))
    aucs = fast_auc(yb[ok], s[idx][ok])
    lo, hi = np.quantile(aucs, [0.025, 0.975])
    return float(lo), float(hi)


def evaluate_reader(
    fit: Callable[[dict], Scorer],
    grid: list[dict],
    y: dict[tuple[str, str], np.ndarray],
    source: str,
    meta: dict,
) -> list[dict]:
    best = None
    for params in grid:
        scorer = fit(params)
        val = roc_auc_score(y[(source, "val")], scorer(source, "val"))
        if best is None or val > best[0]:
            best = (val, params, scorer)
    val, params, scorer = best  # type: ignore
    rows = []
    for target in DOMAINS:
        yt, st = y[(target, "test")], scorer(target, "test")
        lo, hi = bootstrap_auc_ci(yt, st)
        rows.append(
            meta
            | {
                "source": source,
                "target": target,
                "params": ",".join(f"{k}={v}" for k, v in params.items()),
                "val_auc": val,
                "test_auc": roc_auc_score(yt, st),
                "test_pr_auc": average_precision_score(yt, st),
                "ci_low": lo,
                "ci_high": hi,
                "n_test": len(yt),
            }
        )
    return rows


def kernel_reader(psi: dict, y: dict, source: str, meta: dict) -> list[dict]:
    train = psi[(source, "train")]
    gram = kernel(train)

    def fit(p):
        svc = SVC(kernel="precomputed", class_weight="balanced", C=p["C"])
        svc.fit(gram, y[(source, "train")])
        return lambda d, s: svc.decision_function(kernel(psi[(d, s)], train))

    return evaluate_reader(fit, [{"C": c} for c in C_SVM], y, source, meta)


def measure_features(state_batch: np.ndarray) -> np.ndarray:
    z, zz = expectations(probabilities(state_batch))
    return np.hstack([z, zz])


def logistic_reader(feats: dict, y: dict, source: str, meta: dict) -> list[dict]:
    scaler = StandardScaler().fit(feats[(source, "train")])

    def fit(p):
        clf = LogisticRegression(class_weight="balanced", C=p["C"], max_iter=5000)
        clf.fit(scaler.transform(feats[(source, "train")]), y[(source, "train")])
        return lambda d, s: clf.decision_function(scaler.transform(feats[(d, s)]))

    return evaluate_reader(fit, [{"C": c} for c in C_LOGISTIC], y, source, meta)


def evaluate_config(name: str, cfg: dict, source: str) -> list[dict]:
    spec = ENCODERS[name]
    x, y = {}, {}
    for d in DOMAINS:
        for s in ("train", "val", "test"):
            x[(d, s)], y[(d, s)] = load_inputs(name, source, d, s)
    psi = {key: states(name, v, **cfg) for key, v in x.items()}
    meta = {
        "layer": data.LAYER,
        "k": spec.k or 84,
        "encoder": label(name, cfg),
    }
    rows = kernel_reader(psi, y, source, meta | {"reader": "kernel"})
    feats = {key: measure_features(v) for key, v in psi.items()}
    rows += logistic_reader(feats, y, source, meta | {"reader": "measure"})
    return rows


def run() -> pd.DataFrame:
    rows = []
    for name, cfg in CONFIGS:
        for source in DOMAINS:
            rows += evaluate_config(name, cfg, source)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    pd.set_option("display.width", 220)
    REPORT.mkdir(parents=True, exist_ok=True)
    df = run()
    df.to_csv(REPORT / "eval_encoders.csv", index=False)
    df["pair"] = df.source.str[0] + ">" + df.target.str[0]
    for reader, g in df.groupby("reader"):
        print(f"\nTest AUC, reader = {reader} (rows: encoder; columns: source>target)")
        print(g.pivot(index="encoder", columns="pair", values="test_auc").round(3))
    print("\nMean val AUC (source) / mean test AUC (all 9 pairs)")
    print(
        df.groupby(["encoder", "reader"], sort=False)[["val_auc", "test_auc"]]
        .mean()
        .round(3)
    )
