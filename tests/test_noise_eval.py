from pathlib import Path

import numpy as np
import pytest

from quantum.encoding.backends import make_noise_model
from quantum.encoding.grid import POINTS
from quantum.encoding.noise_eval import (
    fidelity_rows,
    kernel_shots_rows,
    measure_from_probabilities,
    measure_rows,
    noise_points,
    sample_probabilities,
    shot_kernel,
)
from quantum.encoding.simulate import (
    noisy_probabilities,
    noisy_probabilities_batch,
    probabilities,
    states,
)

RNG = np.random.default_rng(9)
ready = Path("artifacts/features/big_feature_maps_k16/feats.npz").exists()


def test_sampled_frequencies_sum_to_one_and_approach_the_probabilities():
    p = probabilities(states("angle", RNG.random((3, 8))))
    f = sample_probabilities(p, 200_000, np.random.default_rng(0))
    assert np.allclose(f.sum(1), 1) and np.abs(f - p).max() < 0.01


def test_shot_kernel_is_symmetric_unbiased_with_unit_diagonal():
    a = states("zz", RNG.random((30, 8)))
    exact = np.abs(a.conj() @ a.T) ** 2
    est = shot_kernel(a, None, 1024, np.random.default_rng(1))
    assert np.allclose(est, est.T) and np.allclose(np.diag(est), 1)
    off = ~np.eye(30, dtype=bool)
    assert abs((est - exact)[off].mean()) < 0.005
    assert (est - exact)[off].std() < 0.02  # ~ sqrt(K(1-K)/1024)


def test_shot_kernel_rectangular_shape():
    a, b = states("angle", RNG.random((5, 8))), states("angle", RNG.random((7, 8)))
    assert shot_kernel(a, b).shape == (5, 7)


def test_batched_noisy_sampling_matches_the_exact_noisy_distribution():
    x = RNG.random((3, 8))
    nm = make_noise_model(p_readout=0)  # the density matrix has no readout error
    exact = noisy_probabilities("angle", x, nm)  # density matrix
    sampled = noisy_probabilities_batch("angle", x, nm, shots=20_000)
    assert np.allclose(sampled.sum(1), 1) and np.abs(sampled - exact).max() < 0.02


def test_readout_error_moves_about_eight_percent_of_the_mass():
    # |0...0> read with 1% flip per qubit: P(00000000) ~ 0.99^8 = 0.92
    nm = make_noise_model(p1=0, p2=0, p_readout=1e-2)
    p = noisy_probabilities_batch("angle", np.zeros((1, 8)), nm, shots=20_000)
    assert abs(p[0, 0] - 0.99**8) < 0.01


def test_measure_features_have_n_plus_pairs_columns():
    p = probabilities(states("angle", RNG.random((4, 4)), n_qubits=4))
    assert measure_from_probabilities(p).shape == (4, 4 + 6)


def test_noise_is_run_on_the_eight_penultimate_non_amplitude_points():
    pts = noise_points()
    assert len(pts) == 8 and all(p.layer == "penultimate" for p in pts)
    assert all(p.name != "amplitude" for p in pts)


@pytest.mark.skipif(not ready, reason="run the grid extractions first")
def test_fidelity_orders_by_circuit_depth():
    pts = {p.id: p for p in POINTS}
    ang = np.mean([r["fidelity"] for r in fidelity_rows(pts["penultimate/angle/k8"])])
    zz = np.mean(
        [r["fidelity"] for r in fidelity_rows(pts["penultimate/zz_r2_circular/k8"])]
    )
    assert 0.5 < zz < ang <= 1.0  # 32 CX at 1% error ~ 0.75


@pytest.mark.skipif(not ready, reason="run the grid extractions first")
def test_end_to_end_rows_for_one_point():
    p = next(q for q in POINTS if q.id == "penultimate/angle/k4")
    for rows in (kernel_shots_rows(p, "small"), measure_rows(p, "small", noise=False)):
        assert len(rows) == 3 and all(r["shots"] == 1024 for r in rows)
        assert all(r["ci_low"] <= r["test_auc"] <= r["ci_high"] for r in rows)
