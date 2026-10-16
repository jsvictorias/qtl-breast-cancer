from pathlib import Path

import numpy as np
import pytest

from quantum.encoding.properties import (
    CONFIGS,
    centered_alignment,
    effective_rank,
    label,
    properties,
    single_qubit_entropy,
)
from quantum.encoding.simulate import states

RNG = np.random.default_rng(3)


def test_perfectly_aligned_kernel_has_alignment_close_to_one():
    y = np.repeat([0, 1], 20)
    s = 2.0 * y - 1
    gram = np.outer(s, s) + 2.0  # centered version is exactly y_c y_c^T
    kta, _ = centered_alignment(gram, y)
    assert kta > 0.99


def test_random_labels_fall_inside_the_permutation_null():
    x = RNG.random((60, 8))
    gram = np.exp(-((x[:, None] - x[None]) ** 2).sum(-1))
    y = RNG.integers(0, 2, 60)
    kta, null95 = centered_alignment(gram, y, n_perm=200)
    assert kta < null95 * 3 and null95 > 0


def test_alignment_of_the_identity_kernel_is_small():
    y = RNG.integers(0, 2, 40)
    kta, _ = centered_alignment(np.eye(40), y)
    assert abs(kta) < 0.2


def test_effective_rank_of_identity_and_of_rank_one():
    assert np.isclose(effective_rank(np.eye(10)), 10.0)
    v = RNG.normal(size=(10, 1))
    assert np.isclose(effective_rank(v @ v.T), 1.0, atol=1e-6)


def test_angle_is_a_product_state_and_zz_is_entangled():
    x = RNG.random((6, 8))
    assert np.allclose(single_qubit_entropy(states("angle", x)), 0, atol=1e-10)
    assert np.allclose(
        single_qubit_entropy(states("dense", RNG.random((6, 16)))), 0, atol=1e-10
    )
    assert (
        single_qubit_entropy(states("zz", x, reps=2, entanglement="circular")).mean()
        > 0.05
    )


def test_entropy_of_a_bell_pair_is_one_half():
    psi = np.zeros((1, 256), dtype=complex)
    psi[0, 0] = psi[0, 3] = 1 / np.sqrt(2)  # |00> + |11> on qubits 0 and 1
    e = single_qubit_entropy(psi)[0]
    assert np.allclose(e[:2], 0.5) and np.allclose(e[2:], 0)


def test_labels_are_unique():
    labels = [label(n, c) for n, c in CONFIGS]
    assert len(set(labels)) == len(labels)


@pytest.mark.skipif(
    not Path("artifacts/features/big_penultimate_k16/feats.npz").exists(),
    reason="run quantum_prep.extract first",
)
def test_properties_row_is_complete_and_consistent():
    r = properties("zz", {"reps": 1, "entanglement": "linear"}, "big", "medium")
    assert r["split"] == "val" and r["n"] > 0
    assert 0 <= r["k_mean"] <= 1 and r["k_std"] > 0
    assert np.isclose(r["k_gap"], r["k_within"] - r["k_between"])
    assert -1 <= r["kta"] <= 1 and r["eff_rank"] >= 1
