import numpy as np
import pandas as pd

from quantum.encoding import data
from quantum.encoding.encoders import ENCODERS, load_inputs, resources
from quantum.encoding.simulate import kernel, states
from quantum.quantum_prep.extract import DOMAINS

REPORT = data.REPORT
SPLIT = "val"
N_PERM = 200
SHOTS_REF = 1024

CONFIGS: list[tuple[str, dict]] = [
    ("angle", {}),
    ("dense", {}),
    ("zz", {"reps": 1, "entanglement": "linear"}),
    ("zz", {"reps": 2, "entanglement": "circular"}),
    ("amplitude", {}),
]


def label(name: str, cfg: dict) -> str:
    if name != "zz":
        return name
    return f"zz_r{cfg['reps']}_{cfg['entanglement']}"


def centered_alignment(
    gram: np.ndarray, y: np.ndarray, n_perm: int = 0, seed: int = 0
) -> tuple[float, float]:
    """Centered kernel-target alignment"""
    n = len(y)
    h = np.eye(n) - 1.0 / n
    kc = h @ gram @ h
    denom = np.linalg.norm(kc) * 1.0

    def kta(yy):
        yc = yy - yy.mean()
        return float(yc @ kc @ yc / (denom * (yc @ yc))) if denom > 0 else 0.0

    s = 2.0 * y - 1.0
    null = np.nan
    if n_perm:
        rng = np.random.default_rng(seed)
        null = float(
            np.quantile([kta(rng.permutation(s)) for _ in range(n_perm)], 0.95)
        )
    return kta(s), null


def effective_rank(gram: np.ndarray) -> float:
    ev = np.clip(np.linalg.eigvalsh(gram), 0, None)
    p = ev / ev.sum()
    p = p[p > 1e-12]
    return float(np.exp(-(p * np.log(p)).sum()))


def single_qubit_entropy(psi: np.ndarray) -> np.ndarray:
    n_q = int(np.log2(psi.shape[1]))
    t = psi.reshape((len(psi),) + (2,) * n_q)
    out = np.empty((len(psi), n_q))
    for j in range(n_q):
        m = np.moveaxis(t, 1 + (n_q - 1 - j), 1).reshape(len(psi), 2, -1)
        rho = np.einsum("nik,njk->nij", m, m.conj())
        out[:, j] = 1 - np.real(np.einsum("nij,nji->n", rho, rho))
    return out


def properties(
    name: str,
    cfg: dict,
    source: str,
    domain: str,
    layer: str = data.LAYER,
    k: int | None = None,
) -> dict:
    x, y = load_inputs(name, source, domain, SPLIT, layer, k)
    psi = states(name, x, **cfg)
    gram = kernel(psi)
    off = ~np.eye(len(y), dtype=bool)
    same = y[:, None] == y[None, :]
    within, between = gram[off & same].mean(), gram[off & ~same].mean()
    kta, null95 = centered_alignment(gram, y, N_PERM)
    spec = ENCODERS[name]
    sat = np.nan
    if spec.input_kind == "angles":
        a = data.load_angles(source, domain, SPLIT, k or spec.k, layer)  # type: ignore
        sat = float(a.below.mean() + a.above.mean())
    return {
        "layer": layer,
        "k": (k or spec.k) or 84,
        "encoder": label(name, cfg),
        "source": source,
        "domain": domain,
        "split": SPLIT,
        "n": len(y),
        "sat": sat,
        "k_mean": gram[off].mean(),
        "k_std": gram[off].std(),
        "snr_1024": gram[off].std() * np.sqrt(SHOTS_REF),
        "k_within": within,
        "k_between": between,
        "k_gap": within - between,
        "kta": kta,
        "kta_null95": null95,
        "eff_rank": effective_rank(gram),
        "entangle": single_qubit_entropy(psi).mean(),
    }


def run() -> pd.DataFrame:
    rows = []
    for name, cfg in CONFIGS:
        for source in DOMAINS:
            for domain in DOMAINS:
                rows.append(properties(name, cfg, source, domain))
    return pd.DataFrame(rows)


def cost_table() -> pd.DataFrame:
    return pd.DataFrame(
        [resources(n, **c) | {"label": label(n, c)} for n, c in CONFIGS]
    )


if __name__ == "__main__":
    pd.set_option("display.width", 220)
    REPORT.mkdir(parents=True, exist_ok=True)
    df = run()
    df.to_csv(REPORT / "properties.csv", index=False)
    cost_table().to_csv(REPORT / "resources.csv", index=False)
    print(f"Split: {SPLIT}. Mean over the 3x3 (source, domain) pairs")
    cols = ["sat", "k_mean", "k_std", "snr_1024", "k_gap", "kta", "kta_null95"]
    cols += ["eff_rank", "entangle"]
    print(df.groupby("encoder", sort=False)[cols].mean().round(3).to_string())
    print("\nCentered KTA per (source, domain)")
    piv = df.pivot_table(index="encoder", columns=["source", "domain"], values="kta")
    print(piv.round(3).to_string())
    print()
    print(cost_table().to_string(index=False))
