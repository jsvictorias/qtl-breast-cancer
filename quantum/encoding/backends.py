"""Simulation backends for the encoders (Qiskit, 8 qubits).

- ``exact_state``: ideal statevector (no noise, no shots).
- ``make_noise_model``: depolarizing + readout noise (a stand-in for real hardware).
- ``noisy_density_matrix``: density matrix of a circuit under gate noise (no shots).
- ``noisy_counts``: shot-based sampling under gate and readout noise.

Qiskit uses little-endian ordering: qubit 0 is the *rightmost* bit of a basis-state
label and the least significant bit of the statevector index.
"""

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import DensityMatrix, Statevector
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, ReadoutError, depolarizing_error

N_QUBITS = 8
BASIS_1Q = ["rx", "ry", "rz", "h", "x", "sx", "id"]
BASIS_2Q = ["cx"]
BASIS = BASIS_1Q + BASIS_2Q


def exact_state(circuit: QuantumCircuit) -> np.ndarray:
    """Ideal state as a complex vector of length 2**n."""
    return Statevector(circuit).data


def make_noise_model(
    p1: float = 1e-3, p2: float = 1e-2, p_readout: float = 1e-2
) -> NoiseModel:
    """Depolarizing error on 1- and 2-qubit gates plus symmetric readout error."""
    nm = NoiseModel(basis_gates=BASIS)
    if p1 > 0:
        nm.add_all_qubit_quantum_error(depolarizing_error(p1, 1), BASIS_1Q[:-1])
    if p2 > 0:
        nm.add_all_qubit_quantum_error(depolarizing_error(p2, 2), BASIS_2Q)
    if p_readout > 0:
        nm.add_all_qubit_readout_error(
            ReadoutError([[1 - p_readout, p_readout], [p_readout, 1 - p_readout]])
        )
    return nm


def _to_basis(circuit: QuantumCircuit) -> QuantumCircuit:
    return transpile(
        circuit, basis_gates=BASIS, optimization_level=0, seed_transpiler=0
    )


def noisy_density_matrix(
    circuit: QuantumCircuit, noise_model: NoiseModel
) -> np.ndarray:
    """Density matrix (2**n x 2**n) after gate noise; readout noise is not included."""
    qc = _to_basis(circuit)
    qc.save_density_matrix()
    sim = AerSimulator(method="density_matrix", noise_model=noise_model)
    data = sim.run(qc).result().data()["density_matrix"]
    return np.asarray(DensityMatrix(data).data)


def noisy_counts(
    circuit: QuantumCircuit, noise_model: NoiseModel, shots: int = 1024, seed: int = 0
) -> dict[str, int]:
    """Measured bitstring counts (gate + readout noise, finite shots)."""
    qc = _to_basis(circuit)
    qc.measure_all()
    sim = AerSimulator(noise_model=noise_model, seed_simulator=seed)
    return sim.run(qc, shots=shots).result().get_counts()
