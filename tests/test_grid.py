from pathlib import Path

import pytest

from quantum.encoding.encoders import n_features
from quantum.encoding.grid import (
    LAYERS,
    POINTS,
    baseline_inputs,
    compare,
)


def test_the_declared_grid_has_17_unique_points():
    assert len(POINTS) == 17
    assert len({p.id for p in POINTS}) == 17


def test_qubit_count_matches_the_input_width_of_every_point():
    for p in POINTS:
        n = p.config.get("n_qubits", 8)
        if p.name != "amplitude":
            assert n_features(p.name, n) == p.k, p.id


def test_amplitude_is_only_on_the_penultimate_layer():
    amp = [p for p in POINTS if p.name == "amplitude"]
    assert [p.layer for p in amp] == ["penultimate"]
    assert {p.layer for p in POINTS} == set(LAYERS)


def test_baselines_cover_every_input_of_the_grid():
    have = set(baseline_inputs())
    for p in POINTS:
        kind = "amp84" if p.name == "amplitude" else f"pc{p.k}"
        assert (p.layer, kind) in have, p.id


def test_compare_uses_the_baseline_of_the_same_layer_and_width():
    import pandas as pd

    enc = pd.DataFrame(
        {
            "point": "p",
            "layer": "feature_maps",
            "k": 4,
            "encoder": "angle",
            "reader": "kernel",
            "source": ["a", "b"],
            "target": ["a", "b"],
            "val_auc": 0.8,
            "test_auc": [0.8, 0.6],
        }
    )
    base = pd.DataFrame(
        {
            "layer": ["feature_maps", "feature_maps", "penultimate"] * 2,
            "encoder": ["pc4", "pc8", "pc4"] * 2,
            "reader": "rbf",
            "source": ["a"] * 3 + ["b"] * 3,
            "target": ["a"] * 3 + ["b"] * 3,
            "val_auc": 0.7,
            "test_auc": [0.7, 0.99, 0.99] * 2,
        }
    )
    row = compare(enc, base).query("vs == 'rbf(pc4)'").iloc[0]
    assert abs(row.test_baseline - 0.7) < 1e-12 and row.wins == "1/2"


@pytest.mark.skipif(
    not Path("artifacts/features/big_feature_maps_k16/feats.npz").exists(),
    reason="run the three extra quantum_prep.extract calls listed in grid.py",
)
def test_every_declared_input_exists_on_disk():
    for p in POINTS:
        if p.k:
            for s in ("small", "medium", "big"):
                assert Path(f"artifacts/features/{s}_{p.layer}_k{p.k}").exists()
