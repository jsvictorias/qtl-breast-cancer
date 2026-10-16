import json

import numpy as np
import pandas as pd

from quantum.encoding.decision import (
    FIDELITY_MIN,
    PROBE_MARGIN,
    SHOT_DROP_MAX,
    build_ranking,
    final_test_report,
    input_kind,
    save_finalists,
    select_finalists,
    twin_id,
)
from quantum.encoding.grid import POINTS

BY_ID = {p.id: p for p in POINTS}


def _frames(val: dict, probe=0.80, shot_drop=0.0, fid=0.99):
    """Synthetic result tables: one reader ("kernel") with the given val AUCs."""
    ideal, shots, base, fids = [], [], [], []
    for pid, v in val.items():
        p = BY_ID[pid]
        for s in ("small", "medium", "big"):
            ideal.append({"point": pid, "reader": "kernel", "source": s, "val_auc": v})
            shots.append(
                {
                    "point": pid,
                    "reader": "kernel_shots",
                    "source": s,
                    "val_auc": v - shot_drop,
                }
            )
        fids.append({"point": twin_id(pid), "fidelity": fid})
        for s in ("small", "medium", "big"):
            base.append(
                {
                    "layer": p.layer,
                    "encoder": input_kind(p),
                    "reader": "logreg",
                    "source": s,
                    "val_auc": probe,
                }
            )
    return (
        pd.DataFrame(ideal),
        pd.DataFrame(base),
        pd.DataFrame(shots),
        pd.DataFrame(fids).drop_duplicates("point"),
    )


A, D, Z = "penultimate/angle/k8", "penultimate/dense/k8", "penultimate/zz_r1_linear/k8"


def test_ranking_orders_by_val_auc_and_marks_everything_eligible():
    ideal, base, shots, fid = _frames({A: 0.80, D: 0.85, Z: 0.90})
    r = build_ranking(ideal, base, shots, fid, [BY_ID[i] for i in (A, D, Z)])
    assert list(r.point) == [Z, D, A] and r.eligible.all()


def test_val_ties_at_two_decimals_are_broken_by_depth_then_cx():
    ideal, base, shots, fid = _frames({A: 0.8201, D: 0.8249, Z: 0.8210})
    r = build_ranking(ideal, base, shots, fid, [BY_ID[i] for i in (A, D, Z)])
    # all round to 0.82 -> angle (depth 1) < dense (2) < zz (23)
    assert list(r.point) == [A, D, Z]


def test_probe_margin_excludes_points_below_the_linear_probe():
    ideal, base, shots, fid = _frames({A: 0.80 - PROBE_MARGIN - 0.01, D: 0.80})
    r = build_ranking(ideal, base, shots, fid, [BY_ID[A], BY_ID[D]])
    assert r.set_index("point").eligible.to_dict() == {A: False, D: True}


def test_shot_drop_and_fidelity_thresholds():
    ideal, base, shots, fid = _frames({A: 0.8}, shot_drop=SHOT_DROP_MAX + 0.01)
    assert not build_ranking(ideal, base, shots, fid, [BY_ID[A]]).eligible[0]
    ideal, base, shots, fid = _frames({A: 0.8}, fid=FIDELITY_MIN - 0.01)
    r = build_ranking(ideal, base, shots, fid, [BY_ID[A]])
    assert not r.ok_fidelity[0] and r.ok_probe[0] and r.ok_shots[0]


def test_feature_maps_inherit_the_fidelity_of_the_penultimate_twin():
    fm = "feature_maps/angle/k8"
    assert twin_id(fm) == A
    ideal, base, shots, fid = _frames({fm: 0.8}, fid=0.9)
    r = build_ranking(ideal, base, shots, fid, [BY_ID[fm]])
    assert np.isclose(r.fidelity[0], 0.9)


def test_finalists_are_the_top_two_eligible_only():
    ideal, base, shots, fid = _frames({A: 0.70, D: 0.85, Z: 0.90})
    r = build_ranking(ideal, base, shots, fid, [BY_ID[i] for i in (A, D, Z)])
    r.loc[r.point == Z, "eligible"] = False
    f = select_finalists(r.sort_values("eligible", ascending=False, kind="stable"))
    assert len(f) <= 2 and Z not in set(f.point)


def test_input_kind_matches_the_baseline_inputs():
    assert input_kind(BY_ID[A]) == "pc8"
    assert input_kind(BY_ID["penultimate/dense/k16"]) == "pc16"
    assert input_kind(BY_ID["penultimate/amplitude/k84"]) == "amp84"


def test_final_test_report_has_quantum_baselines_and_references():
    cols = {
        "source": "small",
        "target": "small",
        "test_auc": 0.7,
        "ci_low": 0.6,
        "ci_high": 0.8,
    }
    ideal = pd.DataFrame([cols | {"point": A, "reader": "kernel"}])
    base = pd.DataFrame(
        [
            cols | {"layer": "penultimate", "encoder": "pc8", "reader": r}
            for r in ("rbf", "logreg", "head41_mean", "lenet_final")
        ]
    )
    finalists = pd.DataFrame(
        [{"point": A, "layer": "penultimate", "best_reader": "kernel"}]
    )
    rep = final_test_report(ideal, base, finalists)
    assert set(rep.kind) == {"quantum", "baseline", "reference"} and len(rep) == 5


def test_save_finalists_writes_what_the_a3_needs(tmp_path):
    ideal, base, shots, fid = _frames({A: 0.80, D: 0.85})
    r = build_ranking(ideal, base, shots, fid, [BY_ID[A], BY_ID[D]])
    save_finalists(select_finalists(r), out=tmp_path)
    saved = json.loads((tmp_path / "finalists.json").read_text())
    assert [f["point"] for f in saved] == [D, A]
    assert saved[0]["config"] == {"n_qubits": 4} or "n_qubits" in saved[0]["config"]
    assert {"layer", "k", "encoder", "reader", "depth", "cx", "qubits"} <= set(saved[0])
