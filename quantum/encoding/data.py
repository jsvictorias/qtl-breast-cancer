import json
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
import torch

from quantum.quantum_prep.extract import DOMAINS, FINAL, SPLITS, extract_split
from src.modeling.lenet import LeNet5

FEATS = Path("artifacts/features")
REPORT = Path("reports/a2_encoding")
LAYER = "penultimate"
N_QUBITS = 8
AMP_DIM = 2**N_QUBITS
CLIP_SIGMA = 5.0


class Angles(NamedTuple):
    X: np.ndarray  # (N, k) in [0, 1]
    y: np.ndarray  # (N,) in {0, 1}
    below: np.ndarray  # (k,) fraction of values clipped at 0
    above: np.ndarray  # (k,) fraction of values clipped at 1


class Amplitudes(NamedTuple):
    V: np.ndarray  # (N, 256) standardized activations, zero-padded
    y: np.ndarray  # (N,)
    n_features: int  # 84 informative columns, the rest is padding


def feature_dir(source: str, k: int, layer: str = LAYER) -> Path:
    return FEATS / f"{source}_{layer}_k{k}"


def load_angles(
    source: str, domain: str, split: str, k: int = 8, layer: str = LAYER
) -> Angles:
    """Components of ``domain``/``split`` (``source``'s PCA) rescaled to [0, 1]."""
    d = feature_dir(source, k, layer)
    z, pca = np.load(d / "feats.npz"), np.load(d / "pca.npz")
    raw = (z[f"z_{domain}_{split}"] - pca["lo"]) / (pca["hi"] - pca["lo"])
    return Angles(
        X=np.clip(raw, 0.0, 1.0),
        y=z[f"y_{domain}_{split}"].astype(int),
        below=(raw < 0).mean(0),
        above=(raw > 1).mean(0),
    )


def _raw_path(source: str) -> Path:
    return FEATS / f"{source}_{LAYER}_raw" / "raw.npz"


def build_raw_features(source: str) -> Path:
    """Penultimate activations (84) of every domain/split from the source's CNN."""
    info = json.loads((FINAL / source / "info.json").read_text())
    model = LeNet5()
    model.load_state_dict(torch.load(FINAL / source / "model.pt", map_location="cpu"))
    model.eval()
    mean, std = info["norm"]["mean"], info["norm"]["std"]
    out = {}
    for d in DOMAINS:
        for s in SPLITS:
            h, y = extract_split(model, mean, std, d, s, LAYER)
            out[f"h_{d}_{s}"], out[f"y_{d}_{s}"] = h, y
    path = _raw_path(source)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **out)
    return path


def dead_units(source: str) -> np.ndarray:
    """Boolean mask of penultimate units that are always 0 on the source train set."""
    return np.load(feature_dir(source, 8) / "pca.npz")["sd"] < 1e-6


def load_amplitudes(source: str, domain: str, split: str) -> Amplitudes:
    path = _raw_path(source)
    if not path.exists():
        build_raw_features(source)
    raw = np.load(path)
    # mu/sd of the 84 features on the source training set (same in every k)
    pca = np.load(feature_dir(source, 8) / "pca.npz")
    z = (raw[f"h_{domain}_{split}"] - pca["mu"]) / pca["sd"]
    z = np.clip(z, -CLIP_SIGMA, CLIP_SIGMA)
    V = np.zeros((len(z), AMP_DIM))
    V[:, : z.shape[1]] = z
    return Amplitudes(
        V=V, y=raw[f"y_{domain}_{split}"].astype(int), n_features=z.shape[1]
    )


def saturation_report(ks: tuple[int, ...] = (8, 16)) -> pd.DataFrame:
    """Fraction of clipped values for every (source, k, domain, split)."""
    rows = []
    for source in DOMAINS:
        for k in ks:
            for domain in DOMAINS:
                for split in SPLITS:
                    a = load_angles(source, domain, split, k)
                    rows.append(
                        {
                            "source": source,
                            "k": k,
                            "domain": domain,
                            "split": split,
                            "n": len(a.y),
                            "below": a.below.mean(),
                            "above": a.above.mean(),
                            "saturated": a.below.mean() + a.above.mean(),
                            "worst_component": (a.below + a.above).max(),
                        }
                    )
    return pd.DataFrame(rows)


if __name__ == "__main__":
    rep = saturation_report()
    REPORT.mkdir(parents=True, exist_ok=True)
    rep.to_csv(REPORT / "saturation.csv", index=False)
    pd.set_option("display.width", 200)
    test = rep[rep.split == "test"].pivot_table(
        index=["source", "k"], columns="domain", values="saturated"
    )
    print("Fraction of clipped values on the test split (columns = target domain)")
    print(test.round(3).to_string())
    for s in DOMAINS:
        a = load_amplitudes(s, s, "train")
        norms = np.linalg.norm(a.V, axis=1)
        print(
            f"amplitude {s}: V{a.V.shape}, norm min/median {norms.min():.2f}/"
            f"{np.median(norms):.2f}, zero rows {(norms == 0).sum()}, "
            f"dead units {dead_units(s).sum()}/84"
        )
