import numpy as np
import pytest
from qiskit import QuantumCircuit

from quantum.encoding.backends import (
    N_QUBITS,
    exact_state,
    make_noise_model,
    noisy_counts,
    noisy_density_matrix,
)


def _ry_circuit(angles):
    qc = QuantumCircuit(N_QUBITS)
    for q, a in enumerate(angles):
        qc.ry(a, q)
    return qc


def test_exact_state_is_normalized_and_has_256_amplitudes():
    psi = exact_state(_ry_circuit(np.linspace(0.1, 3.0, N_QUBITS)))
    assert psi.shape == (2**N_QUBITS,)
    assert np.isclose(np.linalg.norm(psi), 1.0)


def test_ry_pi_on_all_qubits_gives_all_ones_state():
    psi = exact_state(_ry_circuit([np.pi] * N_QUBITS))
    assert np.isclose(abs(psi[-1]) ** 2, 1.0)  # |11111111>


def test_qubit_zero_is_the_least_significant_bit():
    qc = QuantumCircuit(N_QUBITS)
    qc.x(0)
    psi = exact_state(qc)
    assert np.isclose(abs(psi[1]) ** 2, 1.0)  # index 1 = ...00001


def test_density_matrix_without_noise_matches_the_ideal_state():
    qc = _ry_circuit(np.linspace(0.2, 2.5, N_QUBITS))
    psi = exact_state(qc)
    rho = noisy_density_matrix(qc, make_noise_model(0, 0, 0))
    fidelity = np.real(psi.conj() @ rho @ psi)
    assert np.isclose(fidelity, 1.0, atol=1e-8)


def test_gate_noise_lowers_fidelity_and_keeps_trace_one():
    qc = _ry_circuit(np.linspace(0.2, 2.5, N_QUBITS))
    for q in range(N_QUBITS - 1):
        qc.cx(q, q + 1)
    psi = exact_state(qc)
    rho = noisy_density_matrix(qc, make_noise_model(p1=1e-2, p2=5e-2, p_readout=0))
    fidelity = np.real(psi.conj() @ rho @ psi)
    assert np.isclose(np.trace(rho).real, 1.0, atol=1e-8)
    assert 0.5 < fidelity < 1.0


def test_noisy_counts_sum_to_shots_and_are_reproducible():
    qc = _ry_circuit([np.pi] * N_QUBITS)
    nm = make_noise_model()
    a = noisy_counts(qc, nm, shots=512, seed=1)
    b = noisy_counts(qc, nm, shots=512, seed=1)
    assert sum(a.values()) == 512
    assert a == b
    assert max(a, key=a.get) == "1" * N_QUBITS


@pytest.mark.parametrize("n", [N_QUBITS])
def test_register_size_is_eight(n):
    assert n == 8
