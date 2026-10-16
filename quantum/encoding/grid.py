import sys
from dataclasses import dataclass

import pandas as pd

from quantum.encoding import baselines
from quantum.encoding.evaluate import REPORT, evaluate_config
from quantum.encoding.properties import label, properties
from quantum.quantum_prep.extract import DOMAINS

LAYERS = ("penultimate", "feature_maps")
ZZ_CONFIGS = (
    {"reps": 1, "entanglement": "linear"},
    {"reps": 2, "entanglement": "circular"},
)


@dataclass(frozen=True)
class Point:
    layer: str
    k: int | None
    name: str
    cfg: tuple

    @property
    def config(self) -> dict:
        return dict(self.cfg)

    @property
    def id(self) -> str:
        return f"{self.layer}/{label(self.name, self.config)}/k{self.k or 84}"


def _pt(layer, k, name, **cfg) -> Point:
    return Point(layer, k, name, tuple(sorted(cfg.items())))


def declared_points() -> list[Point]:
    pts = []
    for layer in LAYERS:
        for k in (4, 8):
            pts.append(_pt(layer, k, "angle", n_qubits=k))
        for k in (4, 8):
            for zz in ZZ_CONFIGS:
                pts.append(_pt(layer, k, "zz", n_qubits=k, **zz))
        for k in (8, 16):
            pts.append(_pt(layer, k, "dense", n_qubits=k // 2))
    pts.append(_pt("penultimate", None, "amplitude"))
    return pts


POINTS = declared_points()


def baseline_inputs() -> list[tuple[str, str]]:
    """(layer, input kind) pairs the baselines must cover."""
    ks = sorted({p.k for p in POINTS if p.k})
    return [(layer, f"pc{k}") for layer in LAYERS for k in ks] + [
        ("penultimate", "amp84")
    ]


def run_eval() -> pd.DataFrame:
    rows = []
    for p in POINTS:
        print("eval", p.id, flush=True)
        for source in DOMAINS:
            for r in evaluate_config(p.name, p.config, source, p.layer, p.k):
                rows.append(r | {"point": p.id})
    return pd.DataFrame(rows)


def run_properties() -> pd.DataFrame:
    rows = []
    for p in POINTS:
        print("properties", p.id, flush=True)
        for source in DOMAINS:
            for domain in DOMAINS:
                r = properties(p.name, p.config, source, domain, p.layer, p.k)
                rows.append(r | {"point": p.id})
    return pd.DataFrame(rows)


def run_baselines() -> pd.DataFrame:
    rows = []
    for source in DOMAINS:
        for layer, kind in baseline_inputs():
            for reader in ("logreg", "rbf"):
                rows += baselines.input_reader(kind, source, reader, layer)
        for layer in LAYERS:
            rows += baselines.head_rows(source, layer)
        rows += baselines.lenet_rows(source)
    return pd.DataFrame(rows)


def compare(enc: pd.DataFrame, base: pd.DataFrame) -> pd.DataFrame:
    """Each point vs the RBF SVM and logistic regression on its own input."""
    out = []
    for point, q in enc.groupby("point", sort=False):
        kind = "amp84" if q.encoder.iloc[0] == "amplitude" else f"pc{q.k.iloc[0]}"
        for reader, qr in q.groupby("reader"):
            for ref in ("rbf", "logreg"):
                b = base[
                    (base.layer == q.layer.iloc[0])
                    & (base.encoder == kind)
                    & (base.reader == ref)
                ]
                m = qr.merge(b, on=["source", "target"], suffixes=("_q", "_b"))
                d = m.test_auc_q - m.test_auc_b
                out.append(
                    {
                        "point": point,
                        "reader": reader,
                        "vs": f"{ref}({kind})",
                        "val_quantum": m.val_auc_q.mean(),
                        "val_baseline": m.val_auc_b.mean(),
                        "test_quantum": m.test_auc_q.mean(),
                        "test_baseline": m.test_auc_b.mean(),
                        "test_diff": d.mean(),
                        "wins": f"{(d > 0).sum()}/{len(d)}",
                    }
                )
    return pd.DataFrame(out)


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 200)
    REPORT.mkdir(parents=True, exist_ok=True)
    if what in ("eval", "all"):
        enc = run_eval()
        enc.to_csv(REPORT / "eval_grid.csv", index=False)
    if what in ("baselines", "all"):
        run_baselines().to_csv(REPORT / "eval_grid_baselines.csv", index=False)
    if what in ("properties", "all"):
        run_properties().to_csv(REPORT / "properties_grid.csv", index=False)
    enc_p, base_p = REPORT / "eval_grid.csv", REPORT / "eval_grid_baselines.csv"
    if enc_p.exists() and base_p.exists():
        cmp = compare(pd.read_csv(enc_p), pd.read_csv(base_p))
        cmp.to_csv(REPORT / "grid_vs_baselines.csv", index=False)
        print("\nGrid vs baseline on the same input (mean over the 9 pairs)")
        print(cmp.round(3).to_string(index=False))
