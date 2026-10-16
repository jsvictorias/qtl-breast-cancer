import json
from pathlib import Path

import pandas as pd

from quantum.encoding.encoders import resources
from quantum.encoding.evaluate import REPORT
from quantum.encoding.grid import POINTS, Point

PROBE_MARGIN = 0.02
SHOT_DROP_MAX = 0.02
FIDELITY_MIN = 0.85
N_FINALISTS = 2
TIE_DECIMALS = 2
SHOT_READER = {"kernel": "kernel_shots", "measure": "measure_shots"}


def mean_val(df: pd.DataFrame, by: list[str]) -> pd.Series:
    """Source-val AUC averaged over the three sources (one value per source)."""
    per_source = df.groupby(by + ["source"]).val_auc.first()
    return per_source.groupby(by).mean()


def input_kind(p: Point) -> str:
    return "amp84" if p.name == "amplitude" else f"pc{p.k}"


def twin_id(point_id: str) -> str:
    """The penultimate point with the same circuit as a feature_maps point."""
    return point_id.replace("feature_maps/", "penultimate/", 1)


def cost(p: Point) -> dict:
    r = resources(p.name, **p.config)
    return {"depth": r["depth"], "cx": r["cx"], "qubits": r["qubits"]}


def build_ranking(
    ideal: pd.DataFrame,
    base: pd.DataFrame,
    shots: pd.DataFrame,
    fidelity: pd.DataFrame,
    points: list[Point] = POINTS,
) -> pd.DataFrame:
    val = mean_val(ideal, ["point", "reader"]).unstack("reader")
    shot_val = mean_val(shots, ["point", "reader"]).unstack("reader")
    probe = mean_val(base[base.reader == "logreg"], ["layer", "encoder"])
    fid = fidelity.set_index("point").fidelity
    rows = []
    for p in points:
        best = val.loc[p.id].idxmax()
        v = float(val.loc[p.id, best])  # type: ignore
        v_probe = float(probe.loc[(p.layer, input_kind(p))])
        v_shot = shot_val.loc[p.id].get(SHOT_READER[best], float("nan"))  # type: ignore
        f = fid.get(p.id, fid.get(twin_id(p.id), float("nan")))
        rows.append(
            {
                "point": p.id,
                "layer": p.layer,
                "k": p.k or 84,
                "encoder": p.id.split("/")[1],
                "best_reader": best,
                "val_auc": v,
                "val_probe": v_probe,
                "ok_probe": v >= v_probe - PROBE_MARGIN,
                "val_shots": float(v_shot),  # type: ignore
                "shot_drop": v - float(v_shot),  # type: ignore
                "fidelity": float(f),
            }
            | cost(p)
        )
    r = pd.DataFrame(rows)
    r["ok_shots"] = r.shot_drop <= SHOT_DROP_MAX
    r["ok_fidelity"] = r.fidelity >= FIDELITY_MIN
    r["eligible"] = r.ok_probe & r.ok_shots & r.ok_fidelity
    r["val_key"] = r.val_auc.round(TIE_DECIMALS)
    r = r.sort_values(
        ["eligible", "val_key", "depth", "cx", "qubits"],
        ascending=[False, False, True, True, True],
        kind="stable",
    ).reset_index(drop=True)
    r["rank"] = r.index + 1
    return r


def select_finalists(ranking: pd.DataFrame, n: int = N_FINALISTS) -> pd.DataFrame:
    return ranking[ranking.eligible].head(n).reset_index(drop=True)


def save_finalists(
    finalists: pd.DataFrame, points: list[Point] = POINTS, out: Path = REPORT
) -> None:
    by_id = {p.id: p for p in points}
    items = []
    for r in finalists.itertuples():
        p = by_id[r.point]  # type: ignore
        items.append(
            {
                "point": p.id,
                "layer": p.layer,
                "k": p.k,
                "encoder": p.name,
                "config": p.config,
                "reader": r.best_reader,
                "val_auc": r.val_auc,
                "depth": r.depth,
                "cx": r.cx,
                "qubits": r.qubits,
            }
        )
    (out / "finalists.json").write_text(json.dumps(items, indent=2))


def final_test_report(
    ideal: pd.DataFrame, base: pd.DataFrame, finalists
) -> pd.DataFrame:
    """Test AUC (3x3, with bootstrap CI) of each finalist and its baselines."""
    cols = ["source", "target", "test_auc", "ci_low", "ci_high"]
    rows = []
    for f in finalists.itertuples():
        q = ideal[(ideal.point == f.point) & (ideal.reader == f.best_reader)]
        rows.append(
            q[cols].assign(model=f"{f.point} [{f.best_reader}]", kind="quantum")
        )
        p = next(p for p in POINTS if p.id == f.point)
        for ref in ("rbf", "logreg"):
            b = base[
                (base.layer == f.layer)
                & (base.encoder == input_kind(p))
                & (base.reader == ref)
            ]
            rows.append(
                b[cols].assign(model=f"{ref}({input_kind(p)})", kind="baseline")
            )
    for layer in sorted(set(finalists.layer)):
        h = base[(base.reader == "head41_mean") & (base.layer == layer)]
        rows.append(h[cols].assign(model=f"head41_mean ({layer})", kind="reference"))
    lenet = base[base.reader == "lenet_final"]
    rows.append(lenet[cols].assign(model="lenet_final", kind="reference"))
    return pd.concat(rows, ignore_index=True)
