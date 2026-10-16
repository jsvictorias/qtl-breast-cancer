import sys

import numpy as np
import pandas as pd
from sklearn.svm import SVC

from quantum.encoding.backends import make_noise_model, noisy_density_matrix
from quantum.encoding.encoders import ENCODERS, build_circuit, load_inputs, resources
from quantum.encoding.evaluate import (
    C_SVM,
    REPORT,
    evaluate_reader,
    logistic_reader,
)
from quantum.encoding.grid import POINTS, Point
from quantum.encoding.simulate import (
    expectations,
    kernel,
    noisy_probabilities_batch,
    probabilities,
    states,
)
from quantum.quantum_prep.extract import DOMAINS

SHOTS = 1024
SPLITS = ("train", "val", "test")
N_FIDELITY = 10


def sample_probabilities(p: np.ndarray, shots: int, rng: np.random.Generator):
    """Outcome frequencies of shots measurements of every row of p."""
    p = np.clip(p, 0, None)
    p = p / p.sum(1, keepdims=True)
    return np.stack([rng.multinomial(shots, row) for row in p]) / shots


def shot_kernel(a, b=None, shots=SHOTS, rng=None) -> np.ndarray:
    """Kernel estimated with shots shots per entry (symmetric if b is None)."""
    rng = rng or np.random.default_rng(0)
    k = np.clip(kernel(a, b), 0, 1)
    est = rng.binomial(shots, k) / shots
    if b is None:
        est = np.triu(est, 1)
        est = est + est.T
        np.fill_diagonal(est, 1.0)
    return est


def measure_from_probabilities(p: np.ndarray) -> np.ndarray:
    z, zz = expectations(p)
    return np.hstack([z, zz])


def _inputs(p: Point, source: str):
    x, y = {}, {}
    for d in DOMAINS:
        for s in SPLITS:
            x[(d, s)], y[(d, s)] = load_inputs(p.name, source, d, s, p.layer, p.k)
    return x, y


def _meta(p: Point, reader: str) -> dict:
    spec = ENCODERS[p.name]
    return {
        "layer": p.layer,
        "k": (p.k or spec.k) or 84,
        "n_qubits": p.config.get("n_qubits", 8),
        "encoder": p.id.split("/")[1],
        "point": p.id,
        "reader": reader,
        "shots": SHOTS,
    }


def kernel_shots_rows(p: Point, source: str, seed: int = 0) -> list[dict]:
    x, y = _inputs(p, source)
    psi = {key: states(p.name, v, **p.config) for key, v in x.items()}
    rng = np.random.default_rng(seed)
    train = psi[(source, "train")]
    gram = shot_kernel(train, None, SHOTS, rng)
    cross = {key: shot_kernel(v, train, SHOTS, rng) for key, v in psi.items()}

    def fit(prm):
        svc = SVC(kernel="precomputed", class_weight="balanced", C=prm["C"])
        svc.fit(gram, y[(source, "train")])
        return lambda d, s: svc.decision_function(cross[(d, s)])

    grid = [{"C": c} for c in C_SVM]
    return evaluate_reader(fit, grid, y, source, _meta(p, "kernel_shots"))


def measure_rows(p: Point, source: str, noise: bool, seed: int = 0) -> list[dict]:
    x, y = _inputs(p, source)
    nm = make_noise_model() if noise else None
    rng = np.random.default_rng(seed)
    feats = {}
    for i, (key, v) in enumerate(x.items()):
        if noise:
            prob = noisy_probabilities_batch(
                p.name,
                v,
                nm,  # type: ignore
                SHOTS,
                seed + 1000 * i,
                **p.config,  # type: ignore
            )
        else:
            prob = sample_probabilities(
                probabilities(states(p.name, v, **p.config)), SHOTS, rng
            )
        feats[key] = measure_from_probabilities(prob)
    reader = "measure_noise" if noise else "measure_shots"
    return logistic_reader(feats, y, source, _meta(p, reader))


def fidelity_rows(p: Point) -> list[dict]:
    nm = make_noise_model()
    rows = []
    for source in DOMAINS:
        x, _ = load_inputs(p.name, source, source, "val", p.layer, p.k)
        x = x[:N_FIDELITY]
        psi = states(p.name, x, **p.config)
        fid = [
            float(
                np.real(
                    psi[i].conj()
                    @ noisy_density_matrix(build_circuit(p.name, v, **p.config), nm)
                    @ psi[i]
                )
            )
            for i, v in enumerate(x)
        ]
        rows.append(
            {
                "point": p.id,
                "source": source,
                "fidelity": np.mean(fid),
                "fidelity_min": np.min(fid),
            }
        )
    return rows


def noise_points() -> list[Point]:
    return [p for p in POINTS if p.layer == "penultimate" and p.name != "amplitude"]


def run_shots() -> pd.DataFrame:
    rows = []
    for p in POINTS:
        print("shots", p.id, flush=True)
        for source in DOMAINS:
            rows += kernel_shots_rows(p, source)
            rows += measure_rows(p, source, noise=False)
    return pd.DataFrame(rows)


def run_noise() -> pd.DataFrame:
    rows = []
    for p in noise_points():
        print("noise", p.id, flush=True)
        for source in DOMAINS:
            rows += measure_rows(p, source, noise=True)
    return pd.DataFrame(rows)


def run_fidelity() -> pd.DataFrame:
    pts = [p for p in POINTS if p.layer == "penultimate"]
    rows = []
    for p in pts:
        print("fidelity", p.id, flush=True)
        r = pd.DataFrame(fidelity_rows(p))
        res = resources(p.name, **p.config)
        rows.append(
            {
                "point": p.id,
                "fidelity": r.fidelity.mean(),
                "fidelity_min": r.fidelity_min.min(),
                "qubits": res["qubits"],
                "depth": res["depth"],
                "cx": res["cx"],
            }
        )
    return pd.DataFrame(rows)


def summary(ideal: pd.DataFrame, extra: pd.DataFrame) -> pd.DataFrame:
    """Mean val/test AUC over the 9 pairs: ideal vs shots vs noise, per point."""
    a = pd.concat([ideal.assign(shots=np.inf), extra], ignore_index=True)
    a["reader"] = a.reader.replace(
        {"kernel": "kernel_exact", "measure": "measure_exact"}
    )
    g = a.groupby(["point", "reader"])[["val_auc", "test_auc"]].mean()
    return g.unstack("reader")  # type: ignore


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 200)
    REPORT.mkdir(parents=True, exist_ok=True)
    if what in ("shots", "all"):
        run_shots().to_csv(REPORT / "noise_shots.csv", index=False)
    if what in ("noise", "all"):
        run_noise().to_csv(REPORT / "noise_aer.csv", index=False)
    if what in ("fidelity", "all"):
        run_fidelity().to_csv(REPORT / "noise_fidelity.csv", index=False)
    parts = [REPORT / f for f in ("noise_shots.csv", "noise_aer.csv")]
    ideal_p = REPORT / "eval_grid.csv"
    if ideal_p.exists() and any(p.exists() for p in parts):
        extra = pd.concat([pd.read_csv(p) for p in parts if p.exists()])
        s = summary(pd.read_csv(ideal_p), extra)
        s.to_csv(REPORT / "noise_summary.csv")
        print("\nMean test AUC over the 9 pairs (exact vs 1024 shots vs Aer noise)")
        print(s["test_auc"].round(3).to_string())
    fid_p = REPORT / "noise_fidelity.csv"
    if fid_p.exists():
        print("\nState fidelity under the noise model")
        print(pd.read_csv(fid_p).round(3).to_string(index=False))
