from pathlib import Path

import numpy as np
import pytest

from quantum.encoding.data import (
    AMP_DIM,
    dead_units,
    load_amplitudes,
    load_angles,
)

pytestmark = pytest.mark.skipif(
    not Path("artifacts/features/big_penultimate_k16/feats.npz").exists(),
    reason="run quantum_prep.extract (k=8 and k=16, penultimate) first",
)


@pytest.mark.parametrize("k", [8, 16])
def test_angles_are_in_unit_interval_with_k_columns(k):
    a = load_angles("big", "medium", "test", k)
    assert a.X.shape[1] == k
    assert a.X.min() >= 0.0 and a.X.max() <= 1.0
    assert set(np.unique(a.y)) <= {0, 1}


def test_source_train_saturation_matches_the_percentile_definition():
    # lo/hi are the 1st/99th percentiles of the source train set -> ~1% on each side
    a = load_angles("big", "big", "train", 8)
    assert np.all(a.below < 0.02) and np.all(a.above < 0.02)
    assert 0.005 < a.below.mean() < 0.015


def test_rescaling_uses_the_source_lo_hi():
    pca = np.load("artifacts/features/small_penultimate_k8/pca.npz")
    z = np.load("artifacts/features/small_penultimate_k8/feats.npz")
    a = load_angles("small", "big", "val", 8)
    expected = np.clip((z["z_big_val"] - pca["lo"]) / (pca["hi"] - pca["lo"]), 0, 1)
    assert np.allclose(a.X, expected)


def test_destination_saturates_more_than_source_train():
    own = load_angles("small", "small", "train", 8)
    other = load_angles("small", "big", "test", 8)
    assert (other.below + other.above).mean() > (own.below + own.above).mean()


def test_amplitude_vectors_are_padded_standardized_and_nonzero():
    a = load_amplitudes("medium", "medium", "train")
    assert a.V.shape[1] == AMP_DIM and a.n_features == 84
    assert np.all(a.V[:, a.n_features :] == 0)
    assert np.isfinite(a.V).all()
    assert (np.linalg.norm(a.V, axis=1) > 0).all()
    # z-score on the source train set (clip at +-5 only trims the tails); the
    # always-zero ("dead") ReLU units have no variance, so they are left out
    m = a.V[:, : a.n_features][:, ~dead_units("medium")]
    assert abs(m.mean()) < 0.1 and 0.85 < m.std() < 1.05


def test_dead_units_stay_zero_in_other_domains():
    dead = dead_units("big")
    assert 0 < dead.sum() < 84
    a = load_amplitudes("big", "small", "test")
    assert np.all(a.V[:, : a.n_features][:, dead] == 0)


def test_amplitude_and_angles_use_the_same_samples_in_the_same_order():
    a = load_amplitudes("big", "small", "test")
    b = load_angles("big", "small", "test", 8)
    assert len(a.y) == len(b.y) and (a.y == b.y).all()
