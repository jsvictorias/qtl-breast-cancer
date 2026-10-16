import numpy as np
import pytest
from qiskit.quantum_info import SparsePauliOp, Statevector

from quantum.encoding.backends import N_QUBITS, exact_state, make_noise_model
from quantum.encoding.encoders import ENCODERS, build_circuit
from quantum.encoding.simulate import (
    PAIRS,
    expectations,
    kernel,
    noisy_probabilities,
    probabilities,
    states,
)

RNG = np.random.default_rng(11)

CONFIGS = [
    ("angle", {}),
    ("dense", {}),
    ("zz", {"reps": 1, "entanglement": "linear"}),
    ("zz", {"reps": 2, "entanglement": "circular"}),
    ("zz", {"reps": 2, "entanglement": "full"}),
    ("amplitude", {}),
]


def _batch(name, n=5):
    d = ENCODERS[name].n_features
    return RNG.normal(size=(n, d)) if name == "amplitude" else RNG.random((n, d))


@pytest.mark.parametrize("name,cfg", CONFIGS)
def test_batch_states_match_qiskit(name, cfg):
    X = _batch(name)
    S = states(name, X, **cfg)
    for x, s in zip(X, S, strict=True):
        ref = exact_state(build_circuit(name, x, **cfg))
        assert np.isclose(abs(np.vdot(ref, s)) ** 2, 1.0)


@pytest.mark.parametrize("name,cfg", CONFIGS)
def test_states_are_normalized(name, cfg):
    S = states(name, _batch(name), **cfg)
    assert np.allclose(np.linalg.norm(S, axis=1), 1.0)


@pytest.mark.parametrize("name,cfg", CONFIGS)
def test_kernel_is_a_fidelity_gram_matrix(name, cfg):
    S = states(name, _batch(name, 8), **cfg)
    K = kernel(S)
    assert np.allclose(np.diag(K), 1.0)
    assert np.allclose(K, K.T)
    assert K.min() >= -1e-12 and K.max() <= 1 + 1e-12
    assert np.linalg.eigvalsh(K).min() > -1e-9  # positive semi-definite


def test_cross_kernel_has_shape_a_by_b():
    A, B = states("angle", _batch("angle", 4)), states("angle", _batch("angle", 6))
    assert kernel(A, B).shape == (4, 6)


def test_angle_kernel_is_analytical():
    X = RNG.random((6, 8))
    K = kernel(states("angle", X))
    d = X[:, None, :] - X[None, :, :]
    assert np.allclose(K, np.prod(np.cos(np.pi * d / 2) ** 2, axis=2))


def test_angle_expectations_are_analytical():
    X = RNG.random((6, 8))
    z, zz = expectations(probabilities(states("angle", X)))
    assert np.allclose(z, np.cos(np.pi * X))
    for p, (i, j) in enumerate(PAIRS):
        assert np.allclose(zz[:, p], z[:, i] * z[:, j])


@pytest.mark.parametrize("name,cfg", [c for c in CONFIGS if c[0] == "zz"][:2])
def test_expectations_match_qiskit_pauli_convention(name, cfg):
    x = _batch(name, 1)[0]
    sv = Statevector(build_circuit(name, x, **cfg))
    z, zz = expectations(probabilities(states(name, x[None], **cfg)))
    for i in range(N_QUBITS):
        label = ["I"] * N_QUBITS
        label[N_QUBITS - 1 - i] = "Z"  # Qiskit labels print qubit 0 last
        assert np.isclose(sv.expectation_value(SparsePauliOp("".join(label))), z[0, i])
    for p in (0, 9, 27):
        i, j = PAIRS[p]
        label = ["I"] * N_QUBITS
        label[N_QUBITS - 1 - i] = label[N_QUBITS - 1 - j] = "Z"
        assert np.isclose(sv.expectation_value(SparsePauliOp("".join(label))), zz[0, p])


def test_zz_kernel_differs_from_angle_kernel():
    X = RNG.random((6, 8))
    assert not np.allclose(
        kernel(states("zz", X)), kernel(states("angle", X)), atol=1e-3
    )


def test_amplitude_kernel_is_squared_cosine_similarity():
    V = RNG.normal(size=(6, 256))
    Vn = V / np.linalg.norm(V, axis=1, keepdims=True)
    assert np.allclose(kernel(states("amplitude", V)), (Vn @ Vn.T) ** 2)


@pytest.mark.parametrize(
    "name,bad", [("angle", np.zeros((2, 7))), ("angle", np.full((2, 8), 1.5))]
)
def test_invalid_batches_are_rejected(name, bad):
    with pytest.raises(ValueError):
        states(name, bad)


def test_amplitude_rejects_zero_rows():
    V = _batch("amplitude", 3)
    V[1] = 0
    with pytest.raises(ValueError):
        states("amplitude", V)


def test_noiseless_noisy_path_recovers_the_ideal_probabilities():
    X = _batch("zz", 2)
    nm = make_noise_model(0, 0, 0)
    P = noisy_probabilities("zz", X, nm)
    assert np.allclose(P, probabilities(states("zz", X)), atol=1e-8)


def test_noise_perturbs_probabilities_and_shots_sum_to_one():
    X = _batch("angle", 2)
    ideal = probabilities(states("angle", X))
    noisy = noisy_probabilities("angle", X, make_noise_model())
    assert np.allclose(noisy.sum(1), 1.0) and not np.allclose(noisy, ideal, atol=1e-4)
    sampled = noisy_probabilities("angle", X, make_noise_model(), shots=4096)
    assert np.allclose(sampled.sum(1), 1.0)
    # sampling error of 4096 shots is ~1/sqrt(4096) per outcome
    assert np.abs(sampled - noisy).max() < 0.05
