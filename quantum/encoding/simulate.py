from functools import cache
from itertools import combinations

import numpy as np
from qiskit import transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel

from quantum.encoding.backends import (
    BASIS,
    N_QUBITS,
    noisy_counts,
    noisy_density_matrix,
)
from quantum.encoding.encoders import _pairs, build_circuit, n_features


@cache
def zsign(n: int) -> np.ndarray:
    """Z eigenvalue (+1/-1) of every qubit on every basis state, shape (2^n, n)."""
    idx = np.arange(2**n)
    return 1 - 2 * ((idx[:, None] >> np.arange(n)) & 1)


@cache
def hadamard(n: int) -> np.ndarray:
    h1 = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
    out = np.ones((1, 1))
    for _ in range(n):
        out = np.kron(h1, out)
    return out


def pairs_of(n: int) -> list[tuple[int, int]]:
    return list(combinations(range(n), 2))


ZSIGN = zsign(N_QUBITS)
PAIRS = pairs_of(N_QUBITS)


def _qubit_states(theta: np.ndarray, phi: np.ndarray | None) -> np.ndarray:
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    if phi is None:
        return np.stack([c, s], axis=-1).astype(complex)  # (N, n, 2)
    return np.stack([c * np.exp(-0.5j * phi), s * np.exp(0.5j * phi)], axis=-1)


def _product_states(q: np.ndarray) -> np.ndarray:
    """Kron of single-qubit states."""
    psi = np.ones((len(q), 1), dtype=complex)
    for j in range(q.shape[1]):
        psi = (q[:, j, :, None] * psi[:, None, :]).reshape(len(q), -1)
    return psi


def _zz_phase(u: np.ndarray, entanglement: str) -> np.ndarray:
    n = u.shape[1]
    z = zsign(n)
    theta = -(u @ z.T.astype(float))
    for j, l in _pairs(n, entanglement):
        theta -= (np.pi - u[:, [j]]) * (np.pi - u[:, [l]]) * (z[:, j] * z[:, l])
    return np.exp(1j * theta)


def states(name: str, x: np.ndarray, **cfg) -> np.ndarray:
    n = cfg.get("n_qubits", N_QUBITS)
    x = np.atleast_2d(np.asarray(x, dtype=float))
    n_feat = n_features(name, n)
    if x.shape[1] != n_feat:
        raise ValueError(f"{name}: expected {n_feat} features, got {x.shape[1]}")
    if name != "amplitude" and (x.min() < 0 or x.max() > 1):
        raise ValueError("angle inputs must lie in [0, 1]")
    if name == "angle":
        return _product_states(_qubit_states(np.pi * x, None))
    if name == "dense":
        return _product_states(_qubit_states(np.pi * x[:, 0::2], np.pi * x[:, 1::2]))
    if name == "zz":
        reps, ent = cfg.get("reps", 1), cfg.get("entanglement", "linear")
        phase = _zz_phase(np.pi * x, ent)
        psi = np.zeros((len(x), 2**n), dtype=complex)
        psi[:, 0] = 1.0
        for _ in range(reps):
            psi = phase * (psi @ hadamard(n).T)
        return psi
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    if (norm < 1e-12).any():
        raise ValueError("amplitude encoding is undefined for the zero vector")
    return (x / norm).astype(complex)


def kernel(a: np.ndarray, b: np.ndarray | None = None) -> np.ndarray:
    b = a if b is None else b
    return np.abs(a.conj() @ b.T) ** 2


def probabilities(psi: np.ndarray) -> np.ndarray:
    return np.abs(psi) ** 2


def expectations(p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """<Z_i> (N, n) and <Z_i Z_j> (N, n(n-1)/2, pairs ``pairs_of(n)``)."""
    n = int(np.log2(p.shape[1]))
    z_s = zsign(n)
    z = p @ z_s
    zz = p @ np.stack([z_s[:, i] * z_s[:, j] for i, j in pairs_of(n)], axis=1)
    return z, zz


def noisy_probabilities(
    name: str,
    x_batch: np.ndarray,
    noise_model: NoiseModel,
    shots: int | None = None,
    seed: int = 0,
    **cfg,
) -> np.ndarray:
    n = cfg.get("n_qubits", N_QUBITS)
    x_batch = np.atleast_2d(x_batch)
    p = np.zeros((len(x_batch), 2**n))
    for i, x in enumerate(x_batch):
        qc = build_circuit(name, x, **cfg)
        if shots is None:
            p[i] = np.real(np.diag(noisy_density_matrix(qc, noise_model)))
        else:
            for bits, c in noisy_counts(qc, noise_model, shots, seed + i).items():
                p[i, int(bits, 2)] = c / shots
    return p


def noisy_probabilities_batch(
    name: str,
    x_batch: np.ndarray,
    noise_model: NoiseModel,
    shots: int = 1024,
    seed: int = 0,
    chunk: int = 250,
    **cfg,
) -> np.ndarray:
    """Sampled outcome probabilities under noise; circuits are run in batches."""
    n = cfg.get("n_qubits", N_QUBITS)
    x_batch = np.atleast_2d(x_batch)
    sim = AerSimulator(noise_model=noise_model, seed_simulator=seed)
    p = np.zeros((len(x_batch), 2**n))
    for start in range(0, len(x_batch), chunk):
        circuits = []
        for x in x_batch[start : start + chunk]:
            qc = build_circuit(name, x, **cfg)
            qc.measure_all()
            circuits.append(qc)
        circuits = transpile(
            circuits, basis_gates=BASIS, optimization_level=0, seed_transpiler=0
        )
        result = sim.run(circuits, shots=shots).result()
        for i in range(len(circuits)):
            for bits, c in result.get_counts(i).items():
                p[start + i, int(bits, 2)] = c / shots
    return p
