from itertools import combinations

import numpy as np
from qiskit_aer.noise import NoiseModel

from quantum.encoding.backends import N_QUBITS, noisy_counts, noisy_density_matrix
from quantum.encoding.encoders import AMP_DIM, _pairs, build_circuit

_IDX = np.arange(AMP_DIM)
ZSIGN = 1 - 2 * ((_IDX[:, None] >> np.arange(N_QUBITS)) & 1)
PAIRS = list(combinations(range(N_QUBITS), 2))
_H1 = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
_HN = np.ones((1, 1))
for _ in range(N_QUBITS):
    _HN = np.kron(_H1, _HN)


def _qubit_states(theta: np.ndarray, phi: np.ndarray | None) -> np.ndarray:
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    if phi is None:
        return np.stack([c, s], axis=-1).astype(complex)  # (N, n, 2)
    return np.stack([c * np.exp(-0.5j * phi), s * np.exp(0.5j * phi)], axis=-1)


def _product_states(q: np.ndarray) -> np.ndarray:
    """Kron of single-qubit states."""
    psi = np.ones((len(q), 1), dtype=complex)
    for j in range(N_QUBITS):
        psi = (q[:, j, :, None] * psi[:, None, :]).reshape(len(q), -1)
    return psi


def _zz_phase(u: np.ndarray, entanglement: str) -> np.ndarray:
    theta = -(u @ ZSIGN.T.astype(float))
    for j, l in _pairs(N_QUBITS, entanglement):
        theta -= (np.pi - u[:, [j]]) * (np.pi - u[:, [l]]) * (ZSIGN[:, j] * ZSIGN[:, l])
    return np.exp(1j * theta)


def states(name: str, x: np.ndarray, **cfg) -> np.ndarray:
    x = np.atleast_2d(np.asarray(x, dtype=float))
    n_feat = {"angle": 8, "dense": 16, "zz": 8, "amplitude": AMP_DIM}[name]
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
        psi = np.zeros((len(x), AMP_DIM), dtype=complex)
        psi[:, 0] = 1.0
        for _ in range(reps):
            psi = phase * (psi @ _HN.T)
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
    z = p @ ZSIGN
    zz = p @ np.stack([ZSIGN[:, i] * ZSIGN[:, j] for i, j in PAIRS], axis=1)
    return z, zz


def noisy_probabilities(
    name: str,
    x_batch: np.ndarray,
    noise_model: NoiseModel,
    shots: int | None = None,
    seed: int = 0,
    **cfg,
) -> np.ndarray:
    x_batch = np.atleast_2d(x_batch)
    p = np.zeros((len(x_batch), AMP_DIM))
    for n, x in enumerate(x_batch):
        qc = build_circuit(name, x, **cfg)
        if shots is None:
            p[n] = np.real(np.diag(noisy_density_matrix(qc, noise_model)))
        else:
            for bits, c in noisy_counts(qc, noise_model, shots, seed + n).items():
                p[n, int(bits, 2)] = c / shots
    return p
