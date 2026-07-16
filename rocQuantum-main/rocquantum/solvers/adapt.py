"""Single-process host reference implementation of ADAPT-VQE."""

from __future__ import annotations

import inspect

import numpy as np

try:
    import rocq
    from rocq.kernel import QuantumKernel
    from rocq.operator import QuantumOperator
except ImportError:  # pragma: no cover - import-only environments.
    rocq = None  # type: ignore
    QuantumKernel = None  # type: ignore
    QuantumOperator = None  # type: ignore

from .qaoa import (
    _dense_pauli_word,
    _operator_qubit_count,
    _operator_terms,
    _validate_boolean,
    _validate_finite_real,
    _validate_positive_integer,
)
from .vqe_solver import VQE_Solver, _normalize_optimizer, _parameter_vector


def _positive_finite(value, name: str) -> float:
    normalized = _validate_finite_real(value, name)
    if normalized <= 0.0:
        raise ValueError(f"{name} must be positive.")
    return normalized


def _normalize_dynamic_start(value: str) -> str:
    if not isinstance(value, str):
        raise ValueError("dynamic_start must be 'warm' or 'cold'.")
    normalized = value.strip().lower()
    if normalized not in {"warm", "cold"}:
        raise ValueError("dynamic_start must be 'warm' or 'cold'.")
    return normalized


def _normalize_pool(pool) -> list:
    if isinstance(pool, (str, bytes)):
        raise ValueError("pool must be a non-empty sequence of QuantumOperator values.")
    try:
        operators = list(pool)
    except TypeError as exc:
        raise ValueError(
            "pool must be a non-empty sequence of QuantumOperator values."
        ) from exc
    if not operators:
        raise ValueError("pool must contain at least one operator.")
    if QuantumOperator is not None and any(
        not isinstance(operator, QuantumOperator) for operator in operators
    ):
        raise ValueError("pool entries must be rocq.operator.QuantumOperator values.")
    for operator in operators:
        terms = _operator_terms(operator, "pool operator", exclude_identity=True)
        if not terms:
            raise ValueError("pool operators must contain a non-identity Pauli term.")
    return operators


def _normalize_initial_preparer(initial_state_kernel):
    if initial_state_kernel is None:
        return None
    candidate = (
        initial_state_kernel._func
        if QuantumKernel is not None and isinstance(initial_state_kernel, QuantumKernel)
        else initial_state_kernel
    )
    if not callable(candidate):
        raise ValueError("initial_state_kernel must be callable or None.")
    try:
        signature = inspect.signature(candidate)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "initial_state_kernel must expose a one-register callable signature."
        ) from exc
    positional = [
        parameter
        for parameter in signature.parameters.values()
        if parameter.kind
        in {
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        }
    ]
    has_varargs = any(
        parameter.kind == inspect.Parameter.VAR_POSITIONAL
        for parameter in signature.parameters.values()
    )
    has_required_keyword_only = any(
        parameter.kind == inspect.Parameter.KEYWORD_ONLY
        and parameter.default is inspect.Parameter.empty
        for parameter in signature.parameters.values()
    )
    if len(positional) != 1 or has_varargs or has_required_keyword_only:
        raise ValueError(
            "initial_state_kernel must accept exactly one qvec argument and "
            "must not allocate a separate register."
        )
    return candidate


def _resolve_adapt_num_qubits(hamiltonian, pool, num_qubits=None) -> int:
    required = _operator_qubit_count(hamiltonian)
    for operator in pool:
        required = max(required, _operator_qubit_count(operator))
    if num_qubits is None:
        if required <= 0:
            raise ValueError("num_qubits could not be inferred from the Hamiltonian and pool.")
        return required
    resolved = _validate_positive_integer(num_qubits, "num_qubits")
    if required > resolved:
        raise ValueError("num_qubits is too small for the Hamiltonian or pool indices.")
    return resolved


def _make_adapt_ansatz(
    initial_preparer,
    selected_operators,
    *,
    num_qubits: int,
):
    if rocq is None:
        raise RuntimeError("Canonical 'rocq' package is required to build ADAPT-VQE kernels.")

    generators = []
    for operator in selected_operators:
        generators.append(
            [
                (coefficient, _dense_pauli_word(paulis, num_qubits))
                for coefficient, paulis in _operator_terms(
                    operator,
                    "selected operator",
                    exclude_identity=True,
                )
            ]
        )

    @rocq.kernel
    def adapt_ansatz(parameters):
        params = _parameter_vector(parameters, "ADAPT-VQE parameters")
        if params.size != len(generators):
            raise ValueError(
                "ADAPT-VQE parameters must match the selected operator count."
            )
        q = rocq.qvec(num_qubits)
        if initial_preparer is not None:
            initial_preparer(q)
        for parameter, terms in zip(params, generators):
            for coefficient, word in terms:
                # CUDA-Q convention: exp_pauli(theta, P) = exp(+i theta P).
                rocq.exp_pauli(float(parameter) * coefficient, q, word)

    return adapt_ansatz


def adapt_vqe(
    initial_state_kernel,
    spin_op,
    pool,
    *,
    optimizer="cobyla",
    gradient="central_difference",
    max_iter: int = 20,
    grad_norm_tol: float = 1.0e-5,
    grad_norm_diff_tol=None,
    energy_diff_tol: float = 1.0e-8,
    threshold_energy=None,
    initial_theta: float = 0.0,
    dynamic_start: str = "warm",
    finite_difference_step: float = 1.0e-5,
    shots=None,
    backend: str = "state_vector",
    num_qubits=None,
    tol: float = 1.0e-6,
    optimizer_max_iterations=None,
    optimizer_options=None,
    verbose: bool = False,
):
    """Run host ADAPT-VQE with finite-difference pool gradients.

    The reference implementation intentionally uses finite differences for
    candidate generators.  A weighted or multi-term generator does not in
    general obey the simple two-point parameter-shift rule.
    """

    if shots is not None:
        raise NotImplementedError(
            "shots-based ADAPT-VQE is not supported by the current rocq.observe contract."
        )
    if not isinstance(gradient, str) or gradient not in {
        "central_difference",
        "finite_difference",
        "finite_diff",
    }:
        raise ValueError("ADAPT-VQE gradient must be 'central_difference'.")
    max_iter = _validate_positive_integer(max_iter, "max_iter")
    grad_norm_tol = _positive_finite(grad_norm_tol, "grad_norm_tol")
    if grad_norm_diff_tol is not None:
        grad_norm_diff_tol = _positive_finite(
            grad_norm_diff_tol, "grad_norm_diff_tol"
        )
    energy_diff_tol = _positive_finite(energy_diff_tol, "energy_diff_tol")
    finite_difference_step = _positive_finite(
        finite_difference_step, "finite_difference_step"
    )
    initial_theta = _validate_finite_real(initial_theta, "initial_theta")
    if threshold_energy is not None:
        threshold_energy = _validate_finite_real(
            threshold_energy, "threshold_energy"
        )
    dynamic_start = _normalize_dynamic_start(dynamic_start)
    verbose = _validate_boolean(verbose, "verbose")

    if QuantumOperator is not None and not isinstance(spin_op, QuantumOperator):
        raise ValueError("spin_op must be a rocq.operator.QuantumOperator.")
    _operator_terms(spin_op, "spin_op")
    pool = _normalize_pool(pool)
    initial_preparer = _normalize_initial_preparer(initial_state_kernel)
    resolved_num_qubits = _resolve_adapt_num_qubits(spin_op, pool, num_qubits)
    normalized_optimizer = _normalize_optimizer(
        optimizer,
        tol=tol,
        max_iterations=optimizer_max_iterations,
        optimizer_options=optimizer_options,
    )
    solver = VQE_Solver(
        optimizer=normalized_optimizer,
        backend=backend,
        verbose=verbose,
    )

    selected = []
    parameters = np.asarray([], dtype=float)
    base_ansatz = _make_adapt_ansatz(
        initial_preparer,
        selected,
        num_qubits=resolved_num_qubits,
    )
    current_energy = solver.evaluate_energy(
        spin_op,
        base_ansatz,
        resolved_num_qubits,
        parameters,
    )
    if threshold_energy is not None and current_energy <= threshold_energy:
        return float(current_energy), parameters.copy(), []
    previous_gradient_norm = None

    for _ in range(max_iter):
        candidate_gradients = []
        for candidate in pool:
            candidate_ansatz = _make_adapt_ansatz(
                initial_preparer,
                selected + [candidate],
                num_qubits=resolved_num_qubits,
            )
            plus = np.append(parameters, finite_difference_step)
            minus = np.append(parameters, -finite_difference_step)
            plus_energy = solver.evaluate_energy(
                spin_op,
                candidate_ansatz,
                resolved_num_qubits,
                plus,
            )
            minus_energy = solver.evaluate_energy(
                spin_op,
                candidate_ansatz,
                resolved_num_qubits,
                minus,
            )
            candidate_gradients.append(
                (plus_energy - minus_energy) / (2.0 * finite_difference_step)
            )

        gradient_norm = float(np.linalg.norm(candidate_gradients))
        if gradient_norm <= grad_norm_tol:
            break
        if (
            grad_norm_diff_tol is not None
            and previous_gradient_norm is not None
            and abs(gradient_norm - previous_gradient_norm) <= grad_norm_diff_tol
        ):
            break

        selected_index = int(np.argmax(np.abs(candidate_gradients)))
        selected.append(pool[selected_index])
        if dynamic_start == "warm":
            initial_parameters = np.append(parameters, initial_theta)
        else:
            initial_parameters = np.zeros(len(selected), dtype=float)

        ansatz = _make_adapt_ansatz(
            initial_preparer,
            selected,
            num_qubits=resolved_num_qubits,
        )
        solution = solver.solve(
            spin_op,
            ansatz,
            resolved_num_qubits,
            initial_params=initial_parameters,
        )
        next_energy = float(solution["optimal_energy"])
        parameters = solution["optimal_parameters"].copy()
        previous_gradient_norm = gradient_norm

        if threshold_energy is not None and next_energy <= threshold_energy:
            current_energy = next_energy
            break
        if abs(current_energy - next_energy) <= energy_diff_tol:
            current_energy = next_energy
            break
        current_energy = next_energy

    return float(current_energy), parameters.copy(), list(selected)
