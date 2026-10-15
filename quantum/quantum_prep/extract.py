import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from torchvision.transforms import v2

from src.data_pipeline.ultrasound_dataset import UltrasoundDataset
from src.modeling.lenet import LeNet5, embed

DOMAINS = ["small", "medium", "big"]
SPLITS = ["train", "val", "test"]
IMAGES = "data/processed/normalized_64.npz"
META = "data/processed/normalized_64_meta.csv"
FINAL = Path("artifacts/final_models")
OUT = Path("artifacts/features")


@torch.inference_mode()
def extract_split(model, mean, std, domain, split, layer):
    ds = UltrasoundDataset(IMAGES, META, domain, split, v2.Normalize([mean], [std]))
    xs, ys = [], []
    for i in range(0, len(ds), 256):
        batch = [ds[j] for j in range(i, min(i + 256, len(ds)))]
        x = torch.stack([b[0] for b in batch])
        xs.append(embed(model, x, layer).numpy())
        ys.append(np.array([b[1].item() for b in batch]))
    return np.concatenate(xs), np.concatenate(ys).astype(int)


def run(source: str, n_components: int, layer: str) -> dict:
    info = json.loads((FINAL / source / "info.json").read_text())
    model = LeNet5()
    model.load_state_dict(torch.load(FINAL / source / "model.pt", map_location="cpu"))
    model.eval()
    mean, std = info["norm"]["mean"], info["norm"]["std"]

    feats = {
        (d, s): extract_split(model, mean, std, d, s, layer)
        for d in DOMAINS
        for s in SPLITS
    }
    xtr, ytr = feats[(source, "train")]
    mu, sd = xtr.mean(0), xtr.std(0) + 1e-8
    pca = PCA(n_components, random_state=0).fit((xtr - mu) / sd)
    proj = {k: pca.transform((x - mu) / sd) for k, (x, _) in feats.items()}

    ztr = proj[(source, "train")]
    lo, hi = np.percentile(ztr, 1, axis=0), np.percentile(ztr, 99, axis=0)

    out = OUT / f"{source}_{layer}_k{n_components}"
    out.mkdir(parents=True, exist_ok=True)
    np.savez(
        out / "pca.npz",
        mu=mu,
        sd=sd,
        components=pca.components_,
        explained_variance=pca.explained_variance_,
        explained_variance_ratio=pca.explained_variance_ratio_,
        lo=lo,
        hi=hi,
    )
    np.savez(
        out / "feats.npz",
        **{f"z_{d}_{s}": proj[(d, s)] for d, s in feats},  # type: ignore
        **{f"y_{d}_{s}": feats[(d, s)][1] for d, s in feats},
    )

    clf = LogisticRegression(max_iter=2000, class_weight="balanced").fit(ztr, ytr)
    auc = {
        d: float(
            roc_auc_score(
                feats[(d, "test")][1], clf.predict_proba(proj[(d, "test")])[:, 1]
            )
        )
        for d in DOMAINS
    }
    meta = {
        "source": source,
        "layer": layer,
        "n_components": n_components,
        "n_features_in": int(xtr.shape[1]),
        "explained_variance_ratio_sum": float(pca.explained_variance_ratio_.sum()),
        "linear_probe_test_auc": auc,
        "model_seed": info["seed"],
    }
    (out / "info.json").write_text(json.dumps(meta, indent=2))
    return meta


if __name__ == "__main__":
    k = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    layer = sys.argv[2] if len(sys.argv) > 2 else "feature_maps"
    for src in DOMAINS:
        m = run(src, k, layer)
        print(
            src,
            f"in={m['n_features_in']} k={k}",
            f"var={m['explained_variance_ratio_sum']:.2f}",
            {d: round(a, 3) for d, a in m["linear_probe_test_auc"].items()},
        )
