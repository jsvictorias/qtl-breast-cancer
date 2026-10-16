from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from quantum.encoding.baselines import (
    SAME_INPUT,
    compare,
    head_rows,
    input_reader,
    lenet_rows,
    load_input,
)

pytestmark = pytest.mark.skipif(
    not Path("multirun/frozen_head_penultimate_k8_h4/src_big").exists()
    or not Path("artifacts/features/big_penultimate_k16/feats.npz").exists(),
    reason="run quantum_prep (extract k=8,16 and frozen_head_baseline) first",
)


def test_inputs_have_the_dimension_the_encoder_receives():
    assert load_input("pc8", "big", "medium", "val")[0].shape[1] == 8
    assert load_input("pc16", "big", "medium", "val")[0].shape[1] == 16
    assert load_input("amp84", "big", "medium", "val")[0].shape[1] == 84


@pytest.mark.parametrize("reader", ["logreg", "rbf"])
def test_input_readers_return_one_row_per_target(reader):
    rows = input_reader("pc8", "small", reader)
    assert [r["target"] for r in rows] == ["small", "medium", "big"]
    assert all(r["ci_low"] <= r["test_auc"] <= r["ci_high"] for r in rows)
    assert rows[0]["test_auc"] > 0.55  # in-domain


def test_head_baseline_summarizes_20_seeds_with_41_parameters():
    rows = head_rows("big")
    mean = [r for r in rows if r["reader"] == "head41_mean"]
    assert len(mean) == 3 and all(
        r["ci_low"] <= r["test_auc"] <= r["ci_high"] for r in mean
    )
    in_domain = next(r for r in mean if r["target"] == "big")
    assert 0.80 < in_domain["test_auc"] < 0.87  # big>big, ~0.834


def test_final_lenet_matches_the_reference_sweep_in_domain():
    rows = {(r["source"], r["target"], r["reader"]): r for r in lenet_rows("big")}
    final = rows[("big", "big", "lenet_final")]["test_auc"]
    ref = rows[("big", "big", "lenet_ref20")]["test_auc"]
    assert abs(final - ref) < 0.03


def test_compare_pairs_each_encoder_with_its_own_input():
    enc = pd.DataFrame(
        {
            "encoder": "angle",
            "reader": "kernel",
            "source": ["a", "b"],
            "target": ["a", "b"],
            "test_auc": [0.8, 0.6],
        }
    )
    base = pd.DataFrame(
        {
            "encoder": "pc8",
            "reader": "rbf",
            "source": ["a", "b"],
            "target": ["a", "b"],
            "test_auc": [0.7, 0.7],
        }
    )
    row = compare(enc, base).query("encoder == 'angle' and vs == 'rbf(pc8)'").iloc[0]
    assert np.isclose(row.mean_diff, 0.0) and row.wins == "1/2"
    assert set(SAME_INPUT.values()) == {"pc8", "pc16", "amp84"}
