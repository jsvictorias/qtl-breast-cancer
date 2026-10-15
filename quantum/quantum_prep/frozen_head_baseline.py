import copy
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch import nn

from src.modeling.engine import compute_metrics
from src.modeling.utils import set_seed

DOMAINS = ["small", "medium", "big"]
FEATS = Path("artifacts/features")


def make_head(k: int, hidden: int) -> nn.Module:
    return nn.Sequential(nn.Linear(k, hidden), nn.ReLU(), nn.Linear(hidden, 1))


def run_seed(
    source, seed, k, layer, hidden, out, epochs=200, patience=20, lr=1e-2, bs=48
):
    d = FEATS / f"{source}_{layer}_k{k}"
    z, pca = np.load(d / "feats.npz"), np.load(d / "pca.npz")
    scale = np.sqrt(pca["explained_variance"]) + 1e-8
    X = lambda dom, s: torch.tensor(z[f"z_{dom}_{s}"] / scale, dtype=torch.float32)  # noqa: E731
    Y = lambda dom, s: z[f"y_{dom}_{s}"]  # noqa: E731

    set_seed(seed)
    head = make_head(k, hidden)
    ytr = torch.tensor(Y(source, "train"), dtype=torch.float32)
    pos_w = (len(ytr) - ytr.sum()) / ytr.sum()
    crit = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    opt = torch.optim.Adam(head.parameters(), lr=lr)
    xtr, g = X(source, "train"), torch.Generator().manual_seed(seed)

    def prob(x):
        head.eval()
        with torch.inference_mode():
            return torch.sigmoid(head(x).squeeze(1)).numpy()

    best, best_ep, wait, state = -1.0, 0, 0, None
    for ep in range(1, epochs + 1):
        head.train()
        perm = torch.randperm(len(xtr), generator=g)
        for i in range(0, len(xtr), bs):
            idx = perm[i : i + bs]
            opt.zero_grad()
            crit(head(xtr[idx]).squeeze(1), ytr[idx]).backward()
            opt.step()
        auc = roc_auc_score(Y(source, "val"), prob(X(source, "val")))
        if auc > best:
            best, best_ep, wait, state = auc, ep, 0, copy.deepcopy(head.state_dict())
        else:
            wait += 1
            if wait >= patience:
                break
    head.load_state_dict(state)  # type: ignore

    out.mkdir(parents=True, exist_ok=True)
    res = {
        "source": source,
        "seed": seed,
        "commit": "n/a",
        "best_epoch": best_ep,
        "best_val_auc": best,
        "k": k,
        "layer": layer,
        "hidden": hidden,
        "n_head_params": sum(p.numel() for p in head.parameters()),
        "targets": {},
        "val_targets": {},
    }
    for dom in DOMAINS:
        for split, key, name in (
            ("test", "targets", f"preds_{dom}"),
            ("val", "val_targets", f"preds_val_{dom}"),
        ):
            p, y = prob(X(dom, split)), Y(dom, split)
            res[key][dom] = compute_metrics(p, y)
            np.savez(out / f"{name}.npz", probs=p, labels=y)
    (out / "metrics.json").write_text(json.dumps(res, indent=2))
    return res


if __name__ == "__main__":
    k = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    layer = sys.argv[2] if len(sys.argv) > 2 else "penultimate"
    hidden = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    n_seeds = int(sys.argv[4]) if len(sys.argv) > 4 else 20
    root = Path(f"multirun/frozen_head_{layer}_k{k}_h{hidden}")
    for src in DOMAINS:
        aucs = []
        for seed in range(n_seeds):
            r = run_seed(
                src, seed, k, layer, hidden, root / f"src_{src}" / f"seed{seed}"
            )
            aucs.append(r["targets"][src]["roc_auc"])
        print(
            src,
            "in-domain test AUC",
            f"{np.mean(aucs):.3f}±{np.std(aucs):.3f}",
            "head params",
            r["n_head_params"],
        )  # type: ignore
