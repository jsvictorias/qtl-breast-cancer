from collections.abc import Callable
from dataclasses import dataclass
from itertools import combinations

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import StatePreparation

from quantum.encoding import data
from quantum.encoding.backends import BASIS, N_QUBITS

AMP_DIM = 2**N_QUBITS


def _check(x: np.ndarray, n: int, unit_interval: bool) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.shape != (n,):
        raise ValueError(f"expected shape ({n},), got {x.shape}")
    if unit_interval and (x.min() < 0 or x.max() > 1):
        raise ValueError("angle inputs must lie in [0, 1]")
    return x


def angle_encoder(x: np.ndarray) -> QuantumCircuit:
    """One RY per qubit: 8 features, depth 1, no entanglement."""
    x = _check(x, N_QUBITS, True)
    qc = QuantumCircuit(N_QUBITS, name="angle")
    for q in range(N_QUBITS):
        qc.ry(np.pi * x[q], q)
    return qc


def dense_angle_encoder(x: np.ndarray) -> QuantumCircuit:
    """Two features per qubit: RY"""
    x = _check(x, 2 * N_QUBITS, True)
    qc = QuantumCircuit(N_QUBITS, name="dense")
    for q in range(N_QUBITS):
        qc.ry(np.pi * x[2 * q], q)
        qc.rz(np.pi * x[2 * q + 1], q)
    return qc


def _pairs(n: int, entanglement: str) -> list[tuple[int, int]]:
    if entanglement == "linear":
        return [(i, i + 1) for i in range(n - 1)]
    if entanglement == "circular":
        return [(i, i + 1) for i in range(n - 1)] + [(n - 1, 0)]
    if entanglement == "full":
        return list(combinations(range(n), 2))
    raise ValueError(f"unknown entanglement: {entanglement}")


def zz_encoder(
    x: np.ndarray, reps: int = 1, entanglement: str = "linear"
) -> QuantumCircuit:
    x = _check(x, N_QUBITS, True)
    u = np.pi * x
    qc = QuantumCircuit(N_QUBITS, name=f"zz_r{reps}_{entanglement}")
    pairs = _pairs(N_QUBITS, entanglement)
    for _ in range(reps):
        for q in range(N_QUBITS):
            qc.h(q)
            qc.rz(2 * u[q], q)
        for j, l in pairs:
            qc.cx(j, l)
            qc.rz(2 * (np.pi - u[j]) * (np.pi - u[l]), l)
            qc.cx(j, l)
    return qc


def amplitude_encoder(v: np.ndarray, eps: float = 1e-12) -> QuantumCircuit:
    v = _check(v, AMP_DIM, False)
    norm = np.linalg.norm(v)
    if norm < eps:
        raise ValueError("amplitude encoding is undefined for the zero vector")
    qc = QuantumCircuit(N_QUBITS, name="amplitude")
    qc.append(StatePreparation(v / norm), range(N_QUBITS))  # type: ignore
    return qc


@dataclass(frozen=True)
class EncoderSpec:
    name: str
    label: str
    n_features: int
    input_kind: str
    k: int | None
    state: str
    build: Callable[..., QuantumCircuit]


ENCODERS: dict[str, EncoderSpec] = {
    "angle": EncoderSpec(
        "angle",
        "Angle (RY)",
        8,
        "angles",
        8,
        "|psi> = tensor_i RY(pi x_i)|0>",
        angle_encoder,
    ),
    "dense": EncoderSpec(
        "dense",
        "Dense angle (RY+RZ)",
        16,
        "angles",
        16,
        "|psi> = tensor_i RZ(pi x_2i+1) RY(pi x_2i)|0>",
        dense_angle_encoder,
    ),
    "zz": EncoderSpec(
        "zz",
        "ZZ / IQP",
        8,
        "angles",
        8,
        "|psi> = [exp(i sum_jl phi_jl Z_j Z_l) exp(i sum_j phi_j Z_j) H^n]^r |0>",
        zz_encoder,
    ),
    "amplitude": EncoderSpec(
        "amplitude",
        "Amplitude",
        AMP_DIM,
        "amplitude",
        None,
        "|psi> = sum_j v_j |j> / ||v||",
        amplitude_encoder,
    ),
}


def build_circuit(name: str, x: np.ndarray, **cfg) -> QuantumCircuit:
    return ENCODERS[name].build(x, **cfg)


def load_inputs(
    name: str, source: str, domain: str, split: str
) -> tuple[np.ndarray, np.ndarray]:
    """Encoder inputs ``(X, y)`` for one (source model, domain, split)."""
    spec = ENCODERS[name]
    if spec.input_kind == "amplitude":
        a = data.load_amplitudes(source, domain, split)
        return a.V, a.y
    a = data.load_angles(source, domain, split, spec.k)  # type: ignore
    return a.X, a.y


def resources(name: str, x: np.ndarray | None = None, **cfg) -> dict:
    """Gate counts after compiling to the basis used by the noise model."""
    spec = ENCODERS[name]
    if x is None:
        rng = np.random.default_rng(0)
        x = (
            rng.normal(size=spec.n_features)
            if name == "amplitude"
            else rng.random(spec.n_features)
        )
    qc = transpile(
        spec.build(x, **cfg), basis_gates=BASIS, optimization_level=0, seed_transpiler=0
    )
    ops = qc.count_ops()
    return {
        "encoder": name,
        "config": ",".join(f"{k}={v}" for k, v in sorted(cfg.items())),
        "qubits": N_QUBITS,
        "features": spec.n_features,
        "depth": qc.depth(),
        "size": qc.size(),
        "cx": int(ops.get("cx", 0)),
        "one_qubit": int(sum(v for g, v in ops.items() if g != "cx")),
    }
