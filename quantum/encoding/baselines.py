import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from quantum.encoding import data
from quantum.encoding.evaluate import (
    C_LOGISTIC,
    C_SVM,
    REPORT,
    bootstrap_auc_ci,
    evaluate_reader,
)
from quantum.quantum_prep.extract import DOMAINS, FINAL, SPLITS
from src.modeling.lenet import LeNet5

GAMMA_MULT = (0.1, 1.0, 10.0)
HEAD_DIR = Path("multirun/frozen_head_penultimate_k8_h4")
CNN_SUMMARY = Path("reports/cnn_2/summary_source_target.csv")
INPUTS = ("pc8", "pc16", "amp84")


def load_input(kind: str, source: str, domain: str, split: str):
    if kind == "amp84":
        a = data.load_amplitudes(source, domain, split)
        return a.V[:, : a.n_features], a.y
    a = data.load_angles(source, domain, split, int(kind[2:]))
    return a.X, a.y


def input_reader(kind: str, source: str, reader: str) -> list[dict]:
    x, y = {}, {}
    for d in DOMAINS:
        for s in SPLITS:
            x[(d, s)], y[(d, s)] = load_input(kind, source, d, s)
    train, ytr = x[(source, "train")], y[(source, "train")]
    meta = {"layer": data.LAYER, "k": train.shape[1], "encoder": kind}
    meta |= {"reader": reader, "ci_kind": "bootstrap_test"}
    if reader == "logreg":
        scaler = StandardScaler().fit(train)
        xs = {key: scaler.transform(v) for key, v in x.items()}

        def fit(p):
            clf = LogisticRegression(class_weight="balanced", C=p["C"], max_iter=5000)
            clf.fit(xs[(source, "train")], ytr)
            return lambda d, s: clf.decision_function(xs[(d, s)])

        grid = [{"C": c} for c in C_LOGISTIC]
    else:  # RBF on the raw inputs, i.e. exactly what the encoder receives
        g0 = 1.0 / (train.shape[1] * train.var())

        def fit(p):
            svc = SVC(kernel="rbf", class_weight="balanced", C=p["C"], gamma=p["gamma"])
            svc.fit(train, ytr)
            return lambda d, s: svc.decision_function(x[(d, s)])

        grid = [{"C": c, "gamma": g0 * m} for c in C_SVM for m in GAMMA_MULT]
    return evaluate_reader(fit, grid, y, source, meta)


def head_rows(source: str) -> list[dict]:
    runs = [
        json.loads(p.read_text())
        for p in sorted((HEAD_DIR / f"src_{source}").glob("seed*/metrics.json"))
    ]
    best = max(runs, key=lambda r: r["best_val_auc"])
    base = {"layer": data.LAYER, "k": 8, "encoder": "pc8", "source": source}
    rows = []
    for target in DOMAINS:
        auc = np.array([r["targets"][target]["roc_auc"] for r in runs])
        pr = np.array([r["targets"][target]["pr_auc"] for r in runs])
        lo, hi = np.quantile(auc, [0.025, 0.975])
        rows.append(
            base
            | {
                "target": target,
                "reader": "head41_mean",
                "ci_kind": "seeds_2.5_97.5",
                "val_auc": float(np.mean([r["best_val_auc"] for r in runs])),
                "test_auc": float(auc.mean()),
                "test_pr_auc": float(pr.mean()),
                "ci_low": float(lo),
                "ci_high": float(hi),
            }
        )
        rows.append(
            base
            | {
                "target": target,
                "reader": "head41_valbest",
                "ci_kind": "none",
                "val_auc": best["best_val_auc"],
                "test_auc": best["targets"][target]["roc_auc"],
                "test_pr_auc": best["targets"][target]["pr_auc"],
            }
        )
    return rows


def lenet_rows(source: str) -> list[dict]:
    model = LeNet5()
    model.load_state_dict(torch.load(FINAL / source / "model.pt", map_location="cpu"))
    model.eval()
    raw = np.load(data.FEATS / f"{source}_{data.LAYER}_raw" / "raw.npz")

    def logit(d, s):  # last Linear on the saved penultimate activations
        with torch.inference_mode():
            h = torch.tensor(raw[f"h_{d}_{s}"], dtype=torch.float32)
            return model.classifier[-1](h).squeeze(1).numpy()

    base = {"layer": "-", "k": 84, "encoder": "lenet", "source": source}
    val = roc_auc_score(raw[f"y_{source}_val"], logit(source, "val"))
    ref = pd.read_csv(CNN_SUMMARY).query("source == @source").set_index("target")
    rows = []
    for target in DOMAINS:
        y, s = raw[f"y_{target}_test"], logit(target, "test")
        lo, hi = bootstrap_auc_ci(y, s)
        rows.append(
            base
            | {
                "target": target,
                "reader": "lenet_final",
                "ci_kind": "bootstrap_test",
                "val_auc": val,
                "test_auc": roc_auc_score(y, s),
                "test_pr_auc": average_precision_score(y, s),
                "ci_low": lo,
                "ci_high": hi,
            }
        )
        r = ref.loc[target]
        rows.append(
            base
            | {
                "target": target,
                "reader": "lenet_ref20",
                "ci_kind": "seeds_ci95",
                "test_auc": r.roc_auc_mean,
                "test_pr_auc": r.pr_auc_mean,
                "ci_low": r.roc_auc_mean - r.roc_auc_ci95,
                "ci_high": r.roc_auc_mean + r.roc_auc_ci95,
            }
        )
    return rows


def run() -> pd.DataFrame:
    rows = []
    for source in DOMAINS:
        for kind in INPUTS:
            for reader in ("logreg", "rbf"):
                rows += input_reader(kind, source, reader)
        rows += head_rows(source) + lenet_rows(source)
    return pd.DataFrame(rows)


# which baseline input each encoder is compared against
SAME_INPUT = {
    "angle": "pc8",
    "zz_r1_linear": "pc8",
    "zz_r2_circular": "pc8",
    "dense": "pc16",
    "amplitude": "amp84",
}


def compare(enc: pd.DataFrame, base: pd.DataFrame) -> pd.DataFrame:
    """Quantum kernel/measure vs the RBF SVM and logistic regression, same input."""
    rows = []
    for e, kind in SAME_INPUT.items():
        for reader in ("kernel", "measure"):
            q = enc[(enc.encoder == e) & (enc.reader == reader)]
            for ref in ("rbf", "logreg"):
                b = base[(base.encoder == kind) & (base.reader == ref)]
                m = q.merge(b, on=["source", "target"], suffixes=("_q", "_b"))
                d = m.test_auc_q - m.test_auc_b
                rows.append(
                    {
                        "encoder": e,
                        "quantum_reader": reader,
                        "vs": f"{ref}({kind})",
                        "auc_quantum": m.test_auc_q.mean(),
                        "auc_baseline": m.test_auc_b.mean(),
                        "mean_diff": d.mean(),
                        "wins": f"{(d > 0).sum()}/{len(d)}",
                    }
                )
    return pd.DataFrame(rows)


if __name__ == "__main__":
    pd.set_option("display.width", 220)
    REPORT.mkdir(parents=True, exist_ok=True)
    base = run()
    base.to_csv(REPORT / "eval_baselines.csv", index=False)
    base["pair"] = base.source.str[0] + ">" + base.target.str[0]
    base["row"] = base.encoder + "/" + base.reader
    print("Baselines, test AUC (rows: input/reader; columns: source>target)")
    piv = base.pivot(index="row", columns="pair", values="test_auc")
    piv["mean"] = piv.mean(axis=1)
    print(piv.round(3).to_string())
    enc_path = REPORT / "eval_encoders.csv"
    if enc_path.exists():
        cmp = compare(pd.read_csv(enc_path), base)
        cmp.to_csv(REPORT / "quantum_vs_baselines.csv", index=False)
        print("\nQuantum vs baseline on the same input (mean over the 9 pairs)")
        print(cmp.round(3).to_string(index=False))
