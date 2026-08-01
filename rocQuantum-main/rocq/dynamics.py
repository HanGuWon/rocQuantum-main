"""Small-system CPU reference implementation of CUDA-Q-style dynamics.

This module provides a correctness oracle for closed-system Schrödinger and
open-system Lindblad evolution.  It intentionally does not claim ROCm
acceleration or CUDA-Q's distributed dynamics backend.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from numbers import Integral, Real
from typing import Callable, Iterable, Optional, Sequence, Tuple

import numpy as np

from .operator import QuantumOperator, operator_to_matrix


class IntermediateResultSave(str, Enum):
    """Controls whether scheduled intermediate states are retained."""

    NONE = "none"
    EXPECTATION_VALUE = "expectation_value"
    # Backward-compatible spelling used by the first rocq reference layer.
    EXPECTATION_VALUES = "expectation_values"
    # Legacy rocq-only mode: retain state and expectation histories.
    STATE = "state"
    ALL = "all"


class Schedule:
    """A finite, strictly increasing sequence of physical evolution times."""

    def __init__(self, steps: Iterable[float], parameter_names: Sequence[str] = ()):
        if isinstance(steps, (str, bytes)):
            raise TypeError("Schedule steps must be a sequence of finite real times.")
        try:
            raw_steps = list(steps)
        except TypeError as exc:
            raise TypeError(
                "Schedule steps must be a sequence of finite real times."
            ) from exc
        if len(raw_steps) < 2:
            raise ValueError("Schedule must contain at least two time points.")

        normalized_steps = []
        for step in raw_steps:
            if isinstance(step, bool) or not isinstance(step, Real):
                raise ValueError("Schedule steps must be finite real numbers.")
            value = float(step)
            if not math.isfinite(value):
                raise ValueError("Schedule steps must be finite.")
            normalized_steps.append(value)
        if any(
            current <= previous
            for previous, current in zip(normalized_steps, normalized_steps[1:])
        ):
            raise ValueError("Schedule steps must be strictly increasing.")

        if isinstance(parameter_names, (str, bytes)):
            raw_names = [parameter_names]
        else:
            try:
                raw_names = list(parameter_names)
            except TypeError as exc:
                raise TypeError("Schedule parameter_names must be a sequence of strings.") from exc
        if any(not isinstance(name, str) or not name for name in raw_names):
            raise ValueError("Schedule parameter_names must contain non-empty strings.")
        if len(set(raw_names)) != len(raw_names):
            raise ValueError("Schedule parameter_names must be unique.")

        self._steps = np.asarray(normalized_steps, dtype=float)
        self._parameter_names = tuple(raw_names)

    @property
    def steps(self) -> np.ndarray:
        return self._steps.copy()

    @property
    def parameter_names(self) -> Tuple[str, ...]:
        return self._parameter_names

    def __len__(self) -> int:
        return int(self._steps.size)

    def __iter__(self):
        return iter(self._steps.tolist())


@dataclass(frozen=True)
class EvolveResult:
    """Result of a CPU reference dynamics evolution."""

    final_state: np.ndarray
    times: np.ndarray
    expectation_values: np.ndarray
    intermediate_states: Tuple[np.ndarray, ...]
    state_kind: str

    def expectation(self, observable_index: int = 0) -> np.ndarray:
        if isinstance(observable_index, bool) or not isinstance(observable_index, Integral):
            raise ValueError("observable_index must be a non-negative integer.")
        index = int(observable_index)
        if index < 0 or index >= self.expectation_values.shape[1]:
            raise IndexError("observable_index is out of range.")
        return self.expectation_values[:, index].copy()


def _positive_integer(value, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be a positive integer.")
    normalized = int(value)
    if normalized <= 0:
        raise ValueError(f"{name} must be positive.")
    return normalized


def _power_of_two_dimension(value, name: str) -> int:
    dimension = _positive_integer(value, name)
    if dimension & (dimension - 1):
        raise ValueError(f"{name} must be a power of two.")
    return dimension


def _finite_complex_matrix(value, label: str) -> np.ndarray:
    try:
        matrix = np.asarray(value, dtype=np.complex128)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a finite numeric square matrix.") from exc
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.shape[0] == 0:
        raise ValueError(f"{label} must be a non-empty square matrix.")
    if not np.all(np.isfinite(matrix.real)) or not np.all(np.isfinite(matrix.imag)):
        raise ValueError(f"{label} must contain only finite values.")
    return np.ascontiguousarray(matrix, dtype=np.complex128)


def _matrix_for(value, num_qubits: int, label: str) -> np.ndarray:
    if isinstance(value, QuantumOperator):
        matrix = operator_to_matrix(value, num_qubits=num_qubits)
    else:
        matrix = _finite_complex_matrix(value, label)
    dimension = 1 << num_qubits
    if matrix.shape != (dimension, dimension):
        raise ValueError(
            f"{label} dimension must be {(dimension, dimension)} for {num_qubits} qubits."
        )
    return np.asarray(matrix, dtype=np.complex128)


def _infer_num_qubits(hamiltonian, initial_state) -> int:
    if isinstance(hamiltonian, QuantumOperator):
        dimension = operator_to_matrix(hamiltonian).shape[0]
        return int(round(math.log2(dimension)))
    if not callable(hamiltonian):
        matrix = _finite_complex_matrix(hamiltonian, "hamiltonian")
        dimension = _power_of_two_dimension(matrix.shape[0], "hamiltonian dimension")
        return int(round(math.log2(dimension)))
    if initial_state is not None:
        state = np.asarray(initial_state)
        if state.ndim == 1:
            dimension = _power_of_two_dimension(state.size, "initial_state dimension")
        elif state.ndim == 2 and state.shape[0] == state.shape[1]:
            dimension = _power_of_two_dimension(
                state.shape[0],
                "initial_state dimension",
            )
        else:
            raise ValueError(
                "initial_state must be a state vector or square density matrix."
            )
        return int(round(math.log2(dimension)))
    raise ValueError(
        "num_qubits is required when hamiltonian is callable and initial_state is omitted."
    )


def _validate_hermitian(matrix: np.ndarray, label: str) -> np.ndarray:
    if not np.allclose(matrix, matrix.conj().T, rtol=1e-10, atol=1e-12):
        raise ValueError(f"{label} must be Hermitian.")
    return matrix


def _normalize_initial_state(initial_state, dimension: int):
    if initial_state is None:
        state = np.zeros(dimension, dtype=np.complex128)
        state[0] = 1.0
        return state, "state_vector"

    try:
        state = np.asarray(initial_state, dtype=np.complex128)
    except (TypeError, ValueError) as exc:
        raise ValueError("initial_state must contain finite numeric values.") from exc
    if not np.all(np.isfinite(state.real)) or not np.all(np.isfinite(state.imag)):
        raise ValueError("initial_state must contain only finite values.")

    if state.ndim == 1:
        if state.shape != (dimension,):
            raise ValueError(f"initial_state vector must have length {dimension}.")
        norm = float(np.vdot(state, state).real)
        if not math.isclose(norm, 1.0, rel_tol=1e-9, abs_tol=1e-10):
            raise ValueError("initial_state vector must have unit norm.")
        return np.ascontiguousarray(state), "state_vector"

    if state.ndim == 2:
        if state.shape != (dimension, dimension):
            raise ValueError(
                f"initial_state density matrix must have shape {(dimension, dimension)}."
            )
        if not np.allclose(state, state.conj().T, rtol=1e-9, atol=1e-10):
            raise ValueError("initial_state density matrix must be Hermitian.")
        trace = complex(np.trace(state))
        if not math.isclose(trace.real, 1.0, rel_tol=1e-9, abs_tol=1e-10) or abs(trace.imag) > 1e-10:
            raise ValueError("initial_state density matrix must have trace one.")
        if float(np.min(np.linalg.eigvalsh(state))) < -1e-9:
            raise ValueError("initial_state density matrix must be positive semidefinite.")
        return np.ascontiguousarray(state), "density_matrix"

    raise ValueError("initial_state must be a state vector or density matrix.")


def _unitary_step(hamiltonian: np.ndarray, delta_t: float) -> np.ndarray:
    eigenvalues, eigenvectors = np.linalg.eigh(hamiltonian)
    phases = np.exp(-1j * float(delta_t) * eigenvalues)
    return (eigenvectors * phases) @ eigenvectors.conj().T


def _lindblad_derivative(
    density: np.ndarray,
    hamiltonian: np.ndarray,
    collapse_matrices: Sequence[np.ndarray],
) -> np.ndarray:
    derivative = -1j * (hamiltonian @ density - density @ hamiltonian)
    for collapse in collapse_matrices:
        adjoint_product = collapse.conj().T @ collapse
        derivative += collapse @ density @ collapse.conj().T
        derivative -= 0.5 * (adjoint_product @ density + density @ adjoint_product)
    return derivative


def _rk4_lindblad_step(
    density: np.ndarray,
    start_time: float,
    step: float,
    hamiltonian_at: Callable[[float], np.ndarray],
    collapse_matrices: Sequence[np.ndarray],
) -> np.ndarray:
    h0 = hamiltonian_at(start_time)
    hm = hamiltonian_at(start_time + 0.5 * step)
    h1 = hamiltonian_at(start_time + step)
    k1 = _lindblad_derivative(density, h0, collapse_matrices)
    k2 = _lindblad_derivative(density + 0.5 * step * k1, hm, collapse_matrices)
    k3 = _lindblad_derivative(density + 0.5 * step * k2, hm, collapse_matrices)
    k4 = _lindblad_derivative(density + step * k3, h1, collapse_matrices)
    updated = density + (step / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    updated = 0.5 * (updated + updated.conj().T)
    trace = complex(np.trace(updated))
    if not math.isfinite(trace.real) or not math.isfinite(trace.imag) or abs(trace) <= 1e-15:
        raise RuntimeError("Lindblad integration produced an invalid density-matrix trace.")
    return updated / trace


def _normalize_store_mode(value) -> IntermediateResultSave:
    if isinstance(value, bool):
        return IntermediateResultSave.ALL if value else IntermediateResultSave.NONE
    if isinstance(value, IntermediateResultSave):
        return value
    if isinstance(value, str):
        try:
            return IntermediateResultSave(value.lower())
        except ValueError as exc:
            raise ValueError(
                "store_intermediate_results must be none, expectation_value, "
                "expectation_values, state, or all."
            ) from exc
    raise ValueError(
        "store_intermediate_results must be an IntermediateResultSave value or string."
    )


def evolve(
    hamiltonian,
    schedule,
    *,
    num_qubits: Optional[int] = None,
    initial_state=None,
    collapse_operators: Optional[Sequence] = None,
    observables: Optional[Sequence] = None,
    store_intermediate_results=IntermediateResultSave.NONE,
    max_step: float = 0.05,
) -> EvolveResult:
    """Evolve a small qubit system on the CPU.

    ``hamiltonian`` may be a constant ``QuantumOperator``/dense matrix or a
    callable ``H(t)``.  Collapse operators switch the solver to a fourth-order
    Runge--Kutta Lindblad path; otherwise Hermitian eigendecomposition gives an
    exact unitary for each piecewise-midpoint interval.
    """

    resolved_schedule = schedule if isinstance(schedule, Schedule) else Schedule(schedule)
    resolved_qubits = (
        _infer_num_qubits(hamiltonian, initial_state)
        if num_qubits is None
        else _positive_integer(num_qubits, "num_qubits")
    )
    dimension = 1 << resolved_qubits

    if isinstance(max_step, bool) or not isinstance(max_step, Real):
        raise ValueError("max_step must be a positive finite real number.")
    normalized_max_step = float(max_step)
    if not math.isfinite(normalized_max_step) or normalized_max_step <= 0.0:
        raise ValueError("max_step must be positive and finite.")

    def hamiltonian_at(time: float) -> np.ndarray:
        value = hamiltonian(time) if callable(hamiltonian) else hamiltonian
        return _validate_hermitian(
            _matrix_for(value, resolved_qubits, "hamiltonian"),
            "hamiltonian",
        )

    collapse_values = [] if collapse_operators is None else list(collapse_operators)
    collapse_matrices = [
        _matrix_for(value, resolved_qubits, "collapse operator")
        for value in collapse_values
    ]
    observable_values = [] if observables is None else list(observables)
    observable_matrices = [
        _validate_hermitian(
            _matrix_for(value, resolved_qubits, "observable"),
            "observable",
        )
        for value in observable_values
    ]

    state, state_kind = _normalize_initial_state(initial_state, dimension)
    if collapse_matrices and state_kind == "state_vector":
        state = np.outer(state, state.conj())
        state_kind = "density_matrix"

    store_mode = _normalize_store_mode(store_intermediate_results)
    retain_state_history = store_mode in {
        IntermediateResultSave.STATE,
        IntermediateResultSave.ALL,
    }
    retain_expectation_history = store_mode is not IntermediateResultSave.NONE
    retained_states = [state.copy()] if retain_state_history else []
    times = resolved_schedule.steps
    expectations = np.zeros(
        (
            len(resolved_schedule) if retain_expectation_history else 1,
            len(observable_matrices),
        ),
        dtype=np.complex128,
    )

    def record_expectations(time_index: int) -> None:
        for observable_index, observable in enumerate(observable_matrices):
            if state_kind == "state_vector":
                value = np.vdot(state, observable @ state)
            else:
                value = np.trace(state @ observable)
            expectations[time_index, observable_index] = value

    if retain_expectation_history:
        record_expectations(0)
    for time_index, (start, stop) in enumerate(zip(times, times[1:]), start=1):
        interval = float(stop - start)
        substeps = max(1, int(math.ceil(interval / normalized_max_step)))
        step = interval / substeps
        current_time = float(start)
        for _ in range(substeps):
            if state_kind == "state_vector":
                h_mid = hamiltonian_at(current_time + 0.5 * step)
                state = _unitary_step(h_mid, step) @ state
            elif collapse_matrices:
                state = _rk4_lindblad_step(
                    state,
                    current_time,
                    step,
                    hamiltonian_at,
                    collapse_matrices,
                )
            else:
                h_mid = hamiltonian_at(current_time + 0.5 * step)
                unitary = _unitary_step(h_mid, step)
                state = unitary @ state @ unitary.conj().T
            current_time += step
        if retain_expectation_history:
            record_expectations(time_index)
        if retain_state_history:
            retained_states.append(state.copy())

    if not retain_expectation_history:
        record_expectations(0)
    if not retain_state_history:
        retained_states.append(state.copy())

    result_times = times if retain_expectation_history else times[-1:]

    return EvolveResult(
        final_state=np.asarray(state, dtype=np.complex128).copy(),
        times=result_times.copy(),
        expectation_values=expectations.copy(),
        intermediate_states=tuple(retained_states),
        state_kind=state_kind,
    )


def evolve_async(
    hamiltonian,
    schedule,
    *,
    num_qubits: Optional[int] = None,
    initial_state=None,
    collapse_operators: Optional[Sequence] = None,
    observables: Optional[Sequence] = None,
    store_intermediate_results=IntermediateResultSave.NONE,
    max_step: float = 0.05,
    qpu_id: int = 0,
    executor=None,
):
    """Submit :func:`evolve` to the canonical host-side async executor.

    This provides CUDA-Q-style result ergonomics but is deliberately a host
    thread-pool future, not a HIP-stream or multi-QPU dynamics scheduler.
    """

    from .kernel import _submit_async, _validate_qpu_id

    _validate_qpu_id(qpu_id)
    return _submit_async(
        lambda: evolve(
            hamiltonian,
            schedule,
            num_qubits=num_qubits,
            initial_state=initial_state,
            collapse_operators=collapse_operators,
            observables=observables,
            store_intermediate_results=store_intermediate_results,
            max_step=max_step,
        ),
        executor=executor,
    )


__all__ = [
    "EvolveResult",
    "IntermediateResultSave",
    "Schedule",
    "evolve",
    "evolve_async",
]
