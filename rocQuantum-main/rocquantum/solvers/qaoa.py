"""Experimental QAOA helpers built on the canonical rocq runtime."""

from __future__ import annotations

from collections.abc import Iterable as IterableABC
from numbers import Real
from typing import Iterable, Sequence, Tuple

import numpy as np

try:
    import rocq
except ImportError:  # pragma: no cover - import contract is tested without native bindings.
    rocq = None  # type: ignore


WeightedEdge = Tuple[int, int, float]


def _validate_positive_integer(value, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be a positive integer.")
    value = int(value)
    if value <= 0:
        raise ValueError(f"{name} must be positive.")
    return value


def _validate_qubit_index(value, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"QAOA edge {name} endpoint must be an integer qubit index.")
    return int(value)


def _validate_finite_weight(value) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError("QAOA edge weights must be finite numeric values.")
    weight = float(value)
    if not np.isfinite(weight):
        raise ValueError("QAOA edge weights must be finite.")
    return weight


def _validate_parameter_vector(
    parameters,
    expected_params: int,
    layers: int,
    label: str = "QAOA",
) -> np.ndarray:
    if isinstance(parameters, (str, bytes)) or isinstance(parameters, (bool, np.bool_)):
        raise ValueError(f"{label} must be finite numeric values.")
    try:
        raw_params = np.asarray(parameters, dtype=object).reshape(-1)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be finite numeric values.") from exc
    params = []
    for value in raw_params:
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
            raise ValueError(f"{label} must be finite numeric values.")
        params.append(float(value))
    params = np.asarray(params, dtype=float)
    if params.size != expected_params:
        if label == "initial_params":
            raise ValueError(
                f"initial_params must contain {expected_params} values for {layers} QAOA layer(s)."
            )
        raise ValueError(f"QAOA expects {expected_params} parameters for {layers} layer(s).")
    if not np.all(np.isfinite(params)):
        raise ValueError(f"{label} must be finite.")
    return params


def _normalize_edges(edges: Iterable[Sequence[float]]) -> list[WeightedEdge]:
    if isinstance(edges, (str, bytes)) or not isinstance(edges, IterableABC):
        raise ValueError("QAOA edges must be an iterable of (u, v) or (u, v, weight).")
    normalized: list[WeightedEdge] = []
    for edge in edges:
        if isinstance(edge, (str, bytes)) or not isinstance(edge, IterableABC):
            raise ValueError("QAOA edges must be (u, v) or (u, v, weight).")
        edge_tuple = tuple(edge)
        if len(edge_tuple) == 2:
            u, v = edge_tuple
            weight = 1.0
        elif len(edge_tuple) == 3:
            u, v, weight = edge_tuple
        else:
            raise ValueError("QAOA edges must be (u, v) or (u, v, weight).")
        normalized.append(
            (
                _validate_qubit_index(u, "source"),
                _validate_qubit_index(v, "target"),
                _validate_finite_weight(weight),
            )
        )
    return normalized


def _canonical_maxcut_edges(num_qubits: int, edges: Iterable[Sequence[float]]) -> list[WeightedEdge]:
    num_qubits = _validate_positive_integer(num_qubits, "num_qubits")
    combined: dict[tuple[int, int], float] = {}
    for u, v, weight in _normalize_edges(edges):
        if u < 0 or v < 0 or u >= num_qubits or v >= num_qubits or u == v:
            raise ValueError("QAOA edge endpoints must be distinct valid qubit indices.")
        key = (min(u, v), max(u, v))
        combined[key] = combined.get(key, 0.0) + float(weight)

    return [
        (u, v, weight)
        for (u, v), weight in combined.items()
        if abs(weight) > 1.0e-15
    ]


def make_maxcut_qaoa_kernel(num_qubits: int, edges: Iterable[Sequence[float]], layers: int = 1):
    """Create an experimental MaxCut-style QAOA ansatz kernel.

    The returned kernel expects a flat parameter vector ordered as
    ``[gamma_0, ..., gamma_{p-1}, beta_0, ..., beta_{p-1}]``.
    """
    if rocq is None:
        raise RuntimeError("Canonical 'rocq' package is required to build QAOA kernels.")
    num_qubits = _validate_positive_integer(num_qubits, "num_qubits")
    layers = _validate_positive_integer(layers, "layers")

    normalized_edges = _canonical_maxcut_edges(num_qubits, edges)

    @rocq.kernel
    def qaoa_ansatz(parameters):
        params = _validate_parameter_vector(parameters, 2 * layers, layers)

        q = rocq.qvec(num_qubits)
        for qubit in range(num_qubits):
            rocq.h(q[qubit])

        gammas = params[:layers]
        betas = params[layers:]
        for layer in range(layers):
            gamma = float(gammas[layer])
            beta = float(betas[layer])
            for u, v, weight in normalized_edges:
                rocq.cnot(q[u], q[v])
                rocq.rz(-gamma * weight, q[v])
                rocq.cnot(q[u], q[v])
            for qubit in range(num_qubits):
                rocq.rx(2.0 * beta, q[qubit])

    return qaoa_ansatz


def maxcut_cost_operator(num_qubits: int, edges: Iterable[Sequence[float]]):
    """Return the Pauli-Z cost Hamiltonian for the experimental MaxCut helper."""
    if rocq is None:
        raise RuntimeError("Canonical 'rocq' package is required to build QAOA operators.")
    from rocq.operator import PauliOperator

    operator = None
    for u, v, weight in _canonical_maxcut_edges(num_qubits, edges):
        term = 0.5 * float(weight) * (
            PauliOperator("I") - PauliOperator(f"Z{int(u)} Z{int(v)}")
        )
        operator = term if operator is None else operator + term

    if operator is None:
        return PauliOperator("I", coefficient=0.0)
    return operator


def solve_maxcut_qaoa(
    num_qubits: int,
    edges: Iterable[Sequence[float]],
    layers: int = 1,
    initial_params=None,
    optimizer=None,
    backend: str = "state_vector",
):
    """Run the experimental MaxCut QAOA helper through ``VQE_Solver``."""
    num_qubits = _validate_positive_integer(num_qubits, "num_qubits")
    layers = _validate_positive_integer(layers, "layers")

    normalized_edges = _canonical_maxcut_edges(num_qubits, edges)
    expected_params = 2 * layers
    if initial_params is None:
        params = np.zeros(expected_params, dtype=float)
    else:
        params = _validate_parameter_vector(
            initial_params,
            expected_params,
            layers,
            label="initial_params",
        )

    ansatz = make_maxcut_qaoa_kernel(num_qubits, normalized_edges, layers=layers)
    cost_operator = maxcut_cost_operator(num_qubits, normalized_edges)
    optimization_operator = -cost_operator

    from .vqe_solver import VQE_Solver

    solver = VQE_Solver(optimizer=optimizer, backend=backend)
    result = solver.solve(optimization_operator, ansatz, num_qubits, initial_params=params)
    optimal_cut_value = -float(result["optimal_energy"])
    intermediate_cut_values = [
        {
            "parameters": entry["parameters"],
            "cut_value": -float(entry["energy"]),
        }
        for entry in result.get("intermediate_results", [])
    ]
    result.update(
        {
            "ansatz": ansatz,
            "cost_operator": cost_operator,
            "optimization_operator": optimization_operator,
            "optimization_direction": "maximize_cut_value",
            "optimal_cut_value": optimal_cut_value,
            "intermediate_cut_values": intermediate_cut_values,
            "normalized_edges": normalized_edges,
            "layers": layers,
            "num_qubits": num_qubits,
            "backend": backend,
        }
    )
    return result
