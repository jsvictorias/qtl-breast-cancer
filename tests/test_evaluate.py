from pathlib import Path

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from quantum.encoding.evaluate import (
    bootstrap_auc_ci,
    evaluate_config,
    evaluate_reader,
    fast_auc,
)

RNG = np.random.default_rng(5)


def test_fast_auc_matches_sklearn_including_ties():
    y = RNG.integers(0, 2, (6, 80))
    s = np.round(RNG.normal(size=(6, 80)) + y, 1)  # rounding creates ties
    ref = [roc_auc_score(a, b) for a, b in zip(y, s, strict=True)]
    assert np.allclose(fast_auc(y, s), ref)


def test_bootstrap_interval_contains_the_point_estimate_and_shrinks_with_n():
    def ci(n):
        y = RNG.integers(0, 2, n)
        s = RNG.normal(size=n) + y
        lo, hi = bootstrap_auc_ci(y, s)
        assert lo <= roc_auc_score(y, s) <= hi
        return hi - lo

    assert ci(2000) < ci(100)


def test_bootstrap_is_reproducible():
    y = RNG.integers(0, 2, 100)
    s = RNG.normal(size=100) + y
    assert bootstrap_auc_ci(y, s, seed=1) == bootstrap_auc_ci(y, s, seed=1)


def test_the_grid_point_is_chosen_by_source_validation_only():
    # scorer "good" is perfect on source val but bad on every test split
    y = {
        (d, s): RNG.integers(0, 2, 60)
        for d in ("small", "medium", "big")
        for s in ("train", "val", "test")
    }

    def fit(p):
        if p["C"] == 1:
            return lambda d, s: y[(d, s)] + 0.0 if s == "val" else -y[(d, s)] + 0.0
        return lambda d, s: RNG.normal(size=60)

    rows = evaluate_reader(fit, [{"C": 0}, {"C": 1}], y, "small", {"reader": "x"})
    assert all(r["params"] == "C=1" and r["val_auc"] == 1.0 for r in rows)
    assert all(r["test_auc"] < 0.01 for r in rows)  # test never used for choosing


@pytest.mark.skipif(
    not Path("artifacts/features/big_penultimate_k16/feats.npz").exists(),
    reason="run quantum_prep.extract first",
)
def test_end_to_end_rows_for_one_config():
    rows = evaluate_config("angle", {}, "small")
    assert len(rows) == 6  # 3 targets x 2 readers
    assert {r["reader"] for r in rows} == {"kernel", "measure"}
    for r in rows:
        assert r["ci_low"] <= r["test_auc"] <= r["ci_high"]
        assert 0 <= r["test_pr_auc"] <= 1
    own = [r for r in rows if r["target"] == "small"]
    assert all(r["test_auc"] > 0.55 for r in own)
