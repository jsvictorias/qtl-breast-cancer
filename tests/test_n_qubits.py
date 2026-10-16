import numpy as np
import pytest

from quantum.encoding.backends import exact_state
from quantum.encoding.encoders import build_circuit, n_features, resources
from quantum.encoding.simulate import expectations, kernel, probabilities, states

RNG = np.random.default_rng(21)
N = 4

CONFIGS = [
    ("angle", {}),
    ("dense", {}),
    ("zz", {"reps": 1, "entanglement": "linear"}),
    ("zz", {"reps": 2, "entanglement": "circular"}),
    ("amplitude", {}),
]


def _batch(name, n=4):
    d = n_features(name, N)
    return RNG.normal(size=(n, d)) if name == "amplitude" else RNG.random((n, d))


def test_feature_widths_scale_with_the_number_of_qubits():
    assert [n_features(e, N) for e in ("angle", "dense", "zz", "amplitude")] == [
        4,
        8,
        4,
        16,
    ]


@pytest.mark.parametrize("name,cfg", CONFIGS)
def test_four_qubit_batch_states_match_qiskit(name, cfg):
    X = _batch(name)
    S = states(name, X, n_qubits=N, **cfg)
    assert S.shape == (4, 2**N)
    for x, s in zip(X, S, strict=True):
        ref = exact_state(build_circuit(name, x, n_qubits=N, **cfg))
        assert np.isclose(abs(np.vdot(ref, s)) ** 2, 1.0)


def test_expectation_shapes_follow_the_number_of_qubits():
    z, zz = expectations(probabilities(states("angle", _batch("angle"), n_qubits=N)))
    assert z.shape == (4, 4) and zz.shape == (4, 6)


def test_four_qubit_kernel_is_a_valid_fidelity_matrix():
    K = kernel(states("zz", _batch("zz", 6), n_qubits=N))
    assert np.allclose(np.diag(K), 1) and np.allclose(K, K.T)


def test_resources_report_the_number_of_qubits_and_fewer_gates():
    r4, r8 = resources("zz", n_qubits=4), resources("zz")
    assert r4["qubits"] == 4 and r8["qubits"] == 8 and r4["cx"] < r8["cx"]


def test_wrong_width_for_the_qubit_count_is_rejected():
    with pytest.raises(ValueError):
        states("angle", np.zeros((2, 8)), n_qubits=N)
    with pytest.raises(ValueError):
        build_circuit("dense", np.zeros(16), n_qubits=N)
