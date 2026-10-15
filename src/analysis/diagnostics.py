from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.data_pipeline.ultrasound_dataset import PROJECT_ROOT

DOMAINS = ["small", "medium", "big"]
PROCESSED = PROJECT_ROOT / "data" / "processed"


def per_sample_errors(
    sweep: Path, meta: pd.DataFrame, seeds: range, threshold: float = 0.5
) -> pd.DataFrame:
    rows = []
    for src in DOMAINS:
        for tgt in DOMAINS:
            stack = np.stack(
                [
                    np.load(sweep / f"src_{src}" / f"seed{s}" / f"preds_{tgt}.npz")[
                        "probs"
                    ]
                    for s in seeds
                ]
            )
            test = meta[(meta.dataset == tgt) & (meta.split == "test")]
            labels = test.label_id.to_numpy()
            wrong = (stack >= threshold).astype(int) != labels[None, :]
            rows.append(
                pd.DataFrame(
                    {
                        "source": src,
                        "target": tgt,
                        "sample_id": test.sample_id.to_numpy(),
                        "label_id": labels,
                        "mean_prob": stack.mean(0),
                        "error_rate": wrong.mean(0),
                    }
                )
            )
    return pd.concat(rows, ignore_index=True)


def always_wrong_malignant(err: pd.DataFrame) -> dict[str, set]:
    sel = err[(err.source == err.target) & (err.label_id == 1) & (err.error_rate == 1)]
    return {d: set(sel[sel.source == d].sample_id) for d in DOMAINS}


def error_feature_corr(err: pd.DataFrame, feats: pd.DataFrame) -> pd.DataFrame:
    cols = ["snr_db", "speckle_index", "roi_mean", "roi_std", "roi_entropy_bits"]
    m = err[(err.source == err.target) & (err.label_id == 1)].merge(
        feats[["sample_id", *cols]], on="sample_id"
    )
    out = {}
    for d in DOMAINS:
        g = m[m.source == d]
        out[d] = {c: spearmanr(g[c], g.error_rate, nan_policy="omit")[0] for c in cols}
    return pd.DataFrame(out)


def domain_identifiability(
    images: np.ndarray, meta: pd.DataFrame, n_pca: int = 50, seed: int = 0
) -> dict:
    x = images.reshape(len(images), -1).astype("float32") / 255.0
    y = meta.dataset.map({d: i for i, d in enumerate(DOMAINS)}).to_numpy()
    tr, te = (meta.split == "train").to_numpy(), (meta.split == "test").to_numpy()

    def fit_score(feat):
        clf = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, class_weight="balanced"),
        )
        clf.fit(feat[tr], y[tr])
        return float(balanced_accuracy_score(y[te], clf.predict(feat[te])))

    pca = make_pipeline(StandardScaler(), PCA(n_pca, random_state=seed))
    pca.fit(x[tr])
    stats = np.stack(
        [x.mean(1), x.std(1), np.percentile(x, 10, 1), np.percentile(x, 90, 1)], 1
    )
    return {
        "pixels_pca": fit_score(pca.transform(x)),
        "intensity_stats_only": fit_score(stats),
    }


def load_stage_images() -> dict[str, np.ndarray]:
    z0 = np.load(PROCESSED / "_step0_cache.npz")
    z1 = np.load(PROCESSED / "normalized_64.npz")
    return {
        "raw64": np.clip(z0["raw64"], 0, 255).astype("uint8"),
        "crop64": np.clip(z0["crop64"], 0, 255).astype("uint8"),
        "final": z1["images"],
    }


def check_alignment() -> pd.DataFrame:
    m0 = pd.read_csv(PROCESSED / "_step0_cache_meta.csv")
    m1 = pd.read_csv(PROCESSED / "normalized_64_meta.csv")
    assert (m0.sample_id.to_numpy() == m1.sample_id.to_numpy()).all()
    return m1.merge(m0[["sample_id", "text_found", "text_frac"]], on="sample_id")


def sample_grid(images, meta, per_domain: int = 12, seed: int = 0, only_text=None):
    rng = np.random.default_rng(seed)
    out = {}
    for d in DOMAINS:
        idx = meta.index[meta.dataset == d]
        if only_text is not None:
            idx = idx[meta.loc[idx, "text_found"].to_numpy() == only_text]
        out[d] = rng.choice(idx, size=min(per_domain, len(idx)), replace=False)
    return out


def plot_grid(imgs: dict, meta, idx_by_domain: dict, path: Path, title: str):
    matplotlib.use("Agg")

    n = len(next(iter(idx_by_domain.values())))
    fig, axes = plt.subplots(2 * len(DOMAINS), n, figsize=(1.4 * n, 2.9 * len(DOMAINS)))
    for r, d in enumerate(DOMAINS):
        for c, i in enumerate(idx_by_domain[d]):
            for k, stage in enumerate(("raw64", "final")):
                ax = axes[2 * r + k, c]
                ax.imshow(imgs[stage][i], cmap="gray", vmin=0, vmax=255)
                ax.set_xticks([])
                ax.set_yticks([])
                if k == 0:
                    ax.set_title(f"y={int(meta.label_id[i])}", fontsize=7)
                if c == 0:
                    ax.set_ylabel(f"{d}\n{stage}", fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    out_dir = PROJECT_ROOT / "reports" / "eda_3"
    out_dir.mkdir(parents=True, exist_ok=True)
    meta, imgs = check_alignment(), load_stage_images()
    plot_grid(
        imgs,
        meta,
        sample_grid(imgs["final"], meta, 10),
        out_dir / "grid_random.png",
        "Random sample: raw64 (top) vs final (bottom)",
    )
    plot_grid(
        imgs,
        meta,
        sample_grid(imgs["final"], meta, 10, only_text=True),
        out_dir / "grid_text_removed.png",
        "Images where text was removed",
    )
    print("wrote", out_dir / "grid_random.png", out_dir / "grid_text_removed.png")
