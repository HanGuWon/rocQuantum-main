"""Experimental QAOA helpers built on the canonical rocq runtime."""

from __future__ import annotations

from typing import Iterable, Sequence, Tuple

import numpy as np

try:
    import rocq
except ImportError:  # pragma: no cover - import contract is tested without native bindings.
    rocq = None  # type: ignore


WeightedEdge = Tuple[int, int, float]


def _normalize_edges(edges: Iterable[Sequence[float]]) -> list[WeightedEdge]:
    normalized: list[WeightedEdge] = []
    for edge in edges:
        if len(edge) == 2:
            u, v = edge
            weight = 1.0
        elif len(edge) == 3:
            u, v, weight = edge
        else:
            raise ValueError("QAOA edges must be (u, v) or (u, v, weight).")
        normalized.append((int(u), int(v), float(weight)))
    return normalized


def make_maxcut_qaoa_kernel(num_qubits: int, edges: Iterable[Sequence[float]], layers: int = 1):
    """Create an experimental MaxCut-style QAOA ansatz kernel.

    The returned kernel expects a flat parameter vector ordered as
    ``[gamma_0, ..., gamma_{p-1}, beta_0, ..., beta_{p-1}]``.
    """
    if rocq is None:
        raise RuntimeError("Canonical 'rocq' package is required to build QAOA kernels.")
    if num_qubits <= 0:
        raise ValueError("num_qubits must be positive.")
    if layers <= 0:
        raise ValueError("layers must be positive.")

    normalized_edges = _normalize_edges(edges)
    for u, v, _ in normalized_edges:
        if u < 0 or v < 0 or u >= num_qubits or v >= num_qubits or u == v:
            raise ValueError("QAOA edge endpoints must be distinct valid qubit indices.")

    @rocq.kernel
    def qaoa_ansatz(parameters):
        params = np.asarray(parameters, dtype=float)
        if params.size != 2 * layers:
            raise ValueError(f"QAOA expects {2 * layers} parameters for {layers} layer(s).")

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
                rocq.rz(2.0 * gamma * weight, q[v])
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
    for u, v, weight in _normalize_edges(edges):
        if u < 0 or v < 0 or u >= num_qubits or v >= num_qubits or u == v:
            raise ValueError("QAOA edge endpoints must be distinct valid qubit indices.")
        term = PauliOperator(f"Z{int(u)} Z{int(v)}", coefficient=-0.5 * float(weight))
        operator = term if operator is None else operator + term

    if operator is None:
        return PauliOperator("I", coefficient=0.0)
    return operator
