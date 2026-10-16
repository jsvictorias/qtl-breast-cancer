import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.circuit.library import ZZFeatureMap

from quantum.encoding.backends import (
    N_QUBITS,
    exact_state,
    make_noise_model,
    noisy_density_matrix,
)
from quantum.encoding.encoders import (
    AMP_DIM,
    ENCODERS,
    amplitude_encoder,
    angle_encoder,
    build_circuit,
    dense_angle_encoder,
    resources,
    zz_encoder,
)

RNG = np.random.default_rng(7)


def fidelity(a, b):
    return abs(np.vdot(a, b)) ** 2  # insensitive to a global phase


def product_state(qubit_states):
    """Kron of single-qubit states; qubit 0 is the least significant -> last factor."""
    psi = np.array([1.0 + 0j])
    for s in qubit_states:  # qubit 0 first
        psi = np.kron(s, psi)
    return psi


def _inputs(name):
    spec = ENCODERS[name]
    if name == "amplitude":
        return RNG.normal(size=spec.n_features)
    return RNG.random(spec.n_features)


@pytest.mark.parametrize("name", list(ENCODERS))
def test_state_is_normalized(name):
    psi = exact_state(build_circuit(name, _inputs(name)))
    assert psi.shape == (AMP_DIM,)
    assert np.isclose(np.linalg.norm(psi), 1.0)


@pytest.mark.parametrize("name", list(ENCODERS))
def test_deterministic(name):
    x = _inputs(name)
    a = exact_state(build_circuit(name, x))
    b = exact_state(build_circuit(name, x))
    assert np.array_equal(a, b)


def test_angle_matches_the_analytical_product_state():
    x = RNG.random(N_QUBITS)
    qs = [np.array([np.cos(np.pi * t / 2), np.sin(np.pi * t / 2)]) for t in x]
    psi = exact_state(angle_encoder(x))
    assert np.isclose(fidelity(psi, product_state(qs)), 1.0)


def test_angle_limit_cases():
    assert np.isclose(abs(exact_state(angle_encoder(np.zeros(8)))[0]) ** 2, 1.0)
    assert np.isclose(abs(exact_state(angle_encoder(np.ones(8)))[-1]) ** 2, 1.0)


def test_angle_is_periodic_in_theta():
    qc1, qc2 = QuantumCircuit(1), QuantumCircuit(1)
    qc1.ry(0.7, 0)
    qc2.ry(0.7 + 2 * np.pi, 0)
    assert np.isclose(fidelity(exact_state(qc1), exact_state(qc2)), 1.0)


def test_dense_matches_the_analytical_product_state():
    x = RNG.random(2 * N_QUBITS)
    qs = []
    for q in range(N_QUBITS):
        th, ph = np.pi * x[2 * q], np.pi * x[2 * q + 1]
        qs.append(
            np.array(
                [
                    np.cos(th / 2) * np.exp(-1j * ph / 2),
                    np.sin(th / 2) * np.exp(1j * ph / 2),
                ]
            )
        )
    psi = exact_state(dense_angle_encoder(x))
    assert np.isclose(fidelity(psi, product_state(qs)), 1.0)


@pytest.mark.filterwarnings("ignore::DeprecationWarning")  # ZZFeatureMap is deprecated
@pytest.mark.parametrize("reps", [1, 2])
@pytest.mark.parametrize("ent", ["linear", "circular", "full"])
def test_zz_matches_qiskit_zzfeaturemap(reps, ent):
    x = RNG.random(N_QUBITS)
    ref = ZZFeatureMap(N_QUBITS, reps=reps, entanglement=ent)
    ref = ref.assign_parameters(np.pi * x)
    mine = exact_state(zz_encoder(x, reps=reps, entanglement=ent))
    assert np.isclose(fidelity(mine, exact_state(ref)), 1.0)


def test_zz_with_zero_repetitions_is_the_ground_state():
    psi = exact_state(zz_encoder(RNG.random(N_QUBITS), reps=0))
    assert np.isclose(abs(psi[0]) ** 2, 1.0)


def test_zz_single_repetition_starts_from_the_uniform_superposition_at_x_zero():
    # x = 0: RZ(0) and RZ(2 pi^2) per pair only add phases -> populations stay uniform
    psi = exact_state(zz_encoder(np.zeros(N_QUBITS), reps=1))
    assert np.allclose(abs(psi) ** 2, 1 / AMP_DIM)


def test_zz_is_entangling_while_angle_is_not():
    def purity_q0(qc):
        psi = exact_state(qc).reshape(2 ** (N_QUBITS - 1), 2)  # last axis = qubit 0
        rho = psi.T @ psi.conj()
        return np.real(np.trace(rho @ rho))

    x = np.full(N_QUBITS, 0.3)
    assert np.isclose(purity_q0(angle_encoder(x)), 1.0)
    assert purity_q0(zz_encoder(x, reps=2)) < 0.99


def test_amplitude_state_is_the_normalized_input():
    v = RNG.normal(size=AMP_DIM)
    psi = exact_state(amplitude_encoder(v))
    assert np.isclose(fidelity(psi, v / np.linalg.norm(v)), 1.0)


def test_amplitude_ignores_the_norm_of_the_input():
    v = RNG.normal(size=AMP_DIM)
    a = exact_state(amplitude_encoder(v))
    b = exact_state(amplitude_encoder(5.0 * v))
    assert np.isclose(fidelity(a, b), 1.0)


def test_amplitude_rejects_the_zero_vector():
    with pytest.raises(ValueError):
        amplitude_encoder(np.zeros(AMP_DIM))


@pytest.mark.parametrize(
    "name,bad",
    [("angle", np.full(8, 1.2)), ("angle", np.zeros(7)), ("dense", np.zeros(8))],
)
def test_invalid_inputs_are_rejected(name, bad):
    with pytest.raises(ValueError):
        build_circuit(name, bad)


def test_resource_counts_match_the_formulas():
    assert resources("angle")["cx"] == 0 and resources("angle")["depth"] == 1
    assert resources("dense")["one_qubit"] == 16 and resources("dense")["cx"] == 0
    # ZZ: 2 CX per pair per repetition; 7 pairs (linear), 8 (circular), 28 (full)
    assert resources("zz", reps=1, entanglement="linear")["cx"] == 2 * 7
    assert resources("zz", reps=2, entanglement="circular")["cx"] == 2 * 8 * 2
    assert resources("zz", reps=1, entanglement="full")["cx"] == 2 * 28
    # amplitude is far deeper than the others
    assert resources("amplitude")["cx"] > 100


@pytest.mark.parametrize("name", ["angle", "dense", "zz"])
def test_each_encoder_runs_under_the_noise_model(name):
    x = RNG.random(ENCODERS[name].n_features)
    qc = build_circuit(name, x)
    rho = noisy_density_matrix(qc, make_noise_model(p1=1e-3, p2=1e-2, p_readout=0))
    psi = exact_state(qc)
    fid = np.real(psi.conj() @ rho @ psi)
    assert np.isclose(np.trace(rho).real, 1.0, atol=1e-8)
    assert 0.5 < fid <= 1.0 + 1e-8
