"""Experimental QAOA helpers built on the canonical rocq runtime."""

from __future__ import annotations

from collections.abc import Iterable as IterableABC, Mapping
from dataclasses import dataclass
from numbers import Real
from typing import Iterable, Sequence, Tuple

import numpy as np

try:
    import rocq
except ImportError:  # pragma: no cover - import contract is tested without native bindings.
    rocq = None  # type: ignore


WeightedEdge = Tuple[int, int, float]


@dataclass(frozen=True)
class QAOAResult:
    """Result of generic QAOA optimization and final-state sampling."""

    optimal_value: float
    optimal_parameters: tuple[float, ...]
    optimal_config: dict[str, int]

    def __post_init__(self):
        value = _validate_finite_real(self.optimal_value, "optimal_value")
        parameters = tuple(
            _validate_finite_real(parameter, "optimal_parameters")
            for parameter in self.optimal_parameters
        )
        if not isinstance(self.optimal_config, Mapping):
            raise ValueError("optimal_config must be a sampling-count mapping.")
        counts = {}
        for bitstring, count in self.optimal_config.items():
            if (
                not isinstance(bitstring, str)
                or not bitstring
                or any(bit not in "01" for bit in bitstring)
            ):
                raise ValueError("optimal_config keys must be binary strings.")
            if isinstance(count, (bool, np.bool_)) or not isinstance(count, (int, np.integer)):
                raise ValueError("optimal_config counts must be non-negative integers.")
            integer_count = int(count)
            if integer_count < 0:
                raise ValueError("optimal_config counts must be non-negative integers.")
            counts[bitstring] = integer_count
        sample_counts = rocq.SampleResult(counts) if rocq is not None else counts
        object.__setattr__(self, "optimal_value", value)
        object.__setattr__(self, "optimal_parameters", parameters)
        object.__setattr__(self, "optimal_config", sample_counts)

    def __iter__(self):
        """Iterate in CUDA-QX's ``value, parameters, config`` order."""

        return iter(
            (self.optimal_value, self.optimal_parameters, self.optimal_config)
        )

    def __len__(self) -> int:
        return 3

    def __getitem__(self, index):
        return (
            self.optimal_value,
            self.optimal_parameters,
            self.optimal_config,
        )[index]


def _validate_positive_integer(value, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be a positive integer.")
    value = int(value)
    if value <= 0:
        raise ValueError(f"{name} must be positive.")
    return value


def _validate_nonnegative_integer(value, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be a non-negative integer.")
    value = int(value)
    if value < 0:
        raise ValueError(f"{name} must be non-negative.")
    return value


def _validate_boolean(value, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a boolean.")
    return value


def _validate_finite_real(value, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number.")
    normalized = float(value)
    if not np.isfinite(normalized):
        raise ValueError(f"{name} must be finite.")
    return normalized


def _validate_optional_cost_operator(cost_operator):
    if cost_operator is None:
        return None
    if rocq is None:
        raise RuntimeError("Canonical 'rocq' package is required to validate QAOA cost operators.")
    from rocq.operator import QuantumOperator

    if not isinstance(cost_operator, QuantumOperator):
        raise ValueError("cost_operator must be a rocq.operator.QuantumOperator or None.")
    return cost_operator


def _operator_terms(operator, label: str, *, exclude_identity: bool = False):
    operator = _validate_optional_cost_operator(operator)
    if operator is None:
        return []
    from rocq.operator import iter_pauli_terms

    normalized = []
    for coefficient, paulis in iter_pauli_terms(operator):
        coefficient = complex(coefficient)
        if not np.isfinite(coefficient.real) or not np.isfinite(coefficient.imag):
            raise ValueError(f"{label} coefficients must be finite.")
        if abs(coefficient.imag) > 1.0e-12:
            raise ValueError(f"{label} must have real Pauli coefficients.")
        canonical_paulis = tuple(
            sorted((str(pauli).upper(), int(qubit)) for pauli, qubit in paulis)
        )
        if exclude_identity and not canonical_paulis:
            continue
        normalized.append((float(coefficient.real), canonical_paulis))
    if not normalized:
        raise ValueError(f"{label} must contain at least one supported Pauli term.")
    return normalized


def _operator_qubit_count(operator) -> int:
    if operator is None:
        return 0
    maximum = -1
    for _, paulis in _operator_terms(operator, "operator"):
        for _, qubit in paulis:
            maximum = max(maximum, int(qubit))
    return maximum + 1


def _resolve_qaoa_num_qubits(problem_operator, reference_operator, num_qubits=None) -> int:
    required = max(
        _operator_qubit_count(problem_operator),
        _operator_qubit_count(reference_operator),
    )
    if num_qubits is None:
        if required <= 0:
            raise ValueError(
                "num_qubits is required when the Hamiltonians contain only identity terms."
            )
        return required
    resolved = _validate_positive_integer(num_qubits, "num_qubits")
    if required > resolved:
        raise ValueError("num_qubits is too small for the Hamiltonian qubit indices.")
    return resolved


def _default_mixer(num_qubits: int):
    from rocq.operator import PauliOperator

    mixer = None
    for qubit in range(num_qubits):
        term = PauliOperator(f"X{qubit}")
        mixer = term if mixer is None else mixer + term
    return mixer


def get_num_qaoa_parameters(
    cost_operator=None,
    layers: int = 1,
    *,
    reference_operator=None,
    full_parameterization: bool = False,
    counterdiabatic: bool = False,
    num_qubits=None,
) -> int:
    """Return the official shared/full/CD QAOA parameter-vector length.

    The legacy MaxCut helper remains the default ``2 * layers`` contract.  A
    full parameterization assigns a parameter to every problem term and every
    non-identity reference term in each layer.  Counterdiabatic QAOA adds one
    ``RY`` angle per qubit and layer.
    """

    _validate_optional_cost_operator(cost_operator)
    _validate_optional_cost_operator(reference_operator)
    layers = _validate_positive_integer(layers, "layers")
    full_parameterization = _validate_boolean(
        full_parameterization, "full_parameterization"
    )
    counterdiabatic = _validate_boolean(counterdiabatic, "counterdiabatic")

    if full_parameterization:
        if cost_operator is None:
            raise ValueError("cost_operator is required for full_parameterization.")
        resolved_qubits = _resolve_qaoa_num_qubits(
            cost_operator, reference_operator, num_qubits
        )
        mixer = (
            reference_operator
            if reference_operator is not None
            else _default_mixer(resolved_qubits)
        )
        problem_terms = _operator_terms(cost_operator, "cost_operator")
        reference_terms = _operator_terms(
            mixer, "reference_operator", exclude_identity=True
        )
        count = layers * (len(problem_terms) + len(reference_terms))
    else:
        count = 2 * layers

    if counterdiabatic:
        resolved_qubits = _resolve_qaoa_num_qubits(
            cost_operator, reference_operator, num_qubits
        )
        count += layers * resolved_qubits
    return count


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


def _edge_entry_from_mapping(edge_key, weight):
    if isinstance(edge_key, (str, bytes)) or not isinstance(edge_key, IterableABC):
        raise ValueError("QAOA edge-weight mapping keys must be (u, v) pairs.")
    edge_tuple = tuple(edge_key)
    if len(edge_tuple) != 2:
        raise ValueError("QAOA edge-weight mapping keys must be (u, v) pairs.")
    return edge_tuple[0], edge_tuple[1], weight


def _normalize_edges(edges: Iterable[Sequence[float]]) -> list[WeightedEdge]:
    if isinstance(edges, (str, bytes)) or not isinstance(edges, IterableABC):
        raise ValueError("QAOA edges must be an iterable of (u, v) or (u, v, weight).")
    normalized: list[WeightedEdge] = []
    edge_entries = (
        (_edge_entry_from_mapping(edge_key, weight) for edge_key, weight in edges.items())
        if isinstance(edges, Mapping)
        else edges
    )
    for edge in edge_entries:
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
    expected_params = get_num_qaoa_parameters(layers=layers)

    @rocq.kernel
    def qaoa_ansatz(parameters):
        params = _validate_parameter_vector(parameters, expected_params, layers)

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


def _dense_pauli_word(paulis, num_qubits: int) -> str:
    word = ["I"] * num_qubits
    for pauli, qubit in paulis:
        if qubit < 0 or qubit >= num_qubits:
            raise ValueError("Pauli term contains a qubit outside num_qubits.")
        word[int(qubit)] = str(pauli).upper()
    return "".join(word)


def _make_generic_qaoa_kernel(
    problem_hamiltonian,
    reference_hamiltonian,
    *,
    num_qubits: int,
    layers: int,
    full_parameterization: bool,
    counterdiabatic: bool,
):
    if rocq is None:
        raise RuntimeError("Canonical 'rocq' package is required to build QAOA kernels.")

    problem_terms = _operator_terms(problem_hamiltonian, "problem_hamiltonian")
    reference_terms = _operator_terms(
        reference_hamiltonian,
        "reference_hamiltonian",
        exclude_identity=True,
    )
    problem_words = [
        (coefficient, _dense_pauli_word(paulis, num_qubits))
        for coefficient, paulis in problem_terms
    ]
    reference_words = [
        (coefficient, _dense_pauli_word(paulis, num_qubits))
        for coefficient, paulis in reference_terms
    ]
    expected_parameters = get_num_qaoa_parameters(
        problem_hamiltonian,
        layers,
        reference_operator=reference_hamiltonian,
        full_parameterization=full_parameterization,
        counterdiabatic=counterdiabatic,
        num_qubits=num_qubits,
    )

    @rocq.kernel
    def qaoa_ansatz(parameters):
        params = _validate_parameter_vector(
            parameters,
            expected_parameters,
            layers,
        )
        q = rocq.qvec(num_qubits)
        for qubit in range(num_qubits):
            rocq.h(q[qubit])

        parameter_index = 0
        for _ in range(layers):
            if full_parameterization:
                for coefficient, word in problem_words:
                    rocq.exp_pauli(
                        float(params[parameter_index]) * coefficient,
                        q,
                        word,
                    )
                    parameter_index += 1
            else:
                gamma = float(params[parameter_index])
                parameter_index += 1
                for coefficient, word in problem_words:
                    rocq.exp_pauli(gamma * coefficient, q, word)

            if full_parameterization:
                for coefficient, word in reference_words:
                    rocq.exp_pauli(
                        float(params[parameter_index]) * coefficient,
                        q,
                        word,
                    )
                    parameter_index += 1
            else:
                beta = float(params[parameter_index])
                parameter_index += 1
                for coefficient, word in reference_words:
                    rocq.exp_pauli(beta * coefficient, q, word)

            if counterdiabatic:
                for qubit in range(num_qubits):
                    rocq.ry(float(params[parameter_index]), q[qubit])
                    parameter_index += 1

    return qaoa_ansatz


def _parse_qaoa_call(
    positional,
    *,
    reference_hamiltonian,
    num_layers,
    initial_parameters,
):
    if not positional:
        layers = 1 if num_layers is None else num_layers
        return reference_hamiltonian, layers, initial_parameters

    if num_layers is not None or initial_parameters is not None:
        raise TypeError(
            "num_layers and initial_parameters must not be provided both "
            "positionally and by keyword."
        )

    from rocq.operator import QuantumOperator

    if len(positional) == 2:
        if isinstance(positional[0], QuantumOperator):
            raise TypeError(
                "custom-reference QAOA requires (reference_hamiltonian, "
                "num_layers, initial_parameters)."
            )
        return reference_hamiltonian, positional[0], positional[1]
    if len(positional) == 3 and isinstance(positional[0], QuantumOperator):
        if reference_hamiltonian is not None:
            raise TypeError("reference_hamiltonian was provided more than once.")
        return positional[0], positional[1], positional[2]
    raise TypeError(
        "qaoa expects (problem, num_layers, initial_parameters) or "
        "(problem, reference, num_layers, initial_parameters)."
    )


def qaoa(
    problem_hamiltonian,
    *args,
    reference_hamiltonian=None,
    num_layers=None,
    initial_parameters=None,
    optimizer="cobyla",
    shots: int = 1000,
    full_parameterization: bool = False,
    counterdiabatic: bool = False,
    backend: str = "state_vector",
    num_qubits=None,
    tol: float = 1.0e-6,
    max_iterations=None,
    optimizer_options=None,
    verbose: bool = False,
) -> QAOAResult:
    """Optimize an arbitrary real Pauli-sum QAOA problem.

    ``rocq.gates.exp_pauli`` follows CUDA-Q's ``exp(+i theta P)`` convention;
    every QAOA problem/mixer coefficient is multiplied directly into ``theta``.
    """

    if rocq is None:
        raise RuntimeError("Canonical 'rocq' package is required to execute QAOA.")
    reference_hamiltonian, layers, initial_parameters = _parse_qaoa_call(
        args,
        reference_hamiltonian=reference_hamiltonian,
        num_layers=num_layers,
        initial_parameters=initial_parameters,
    )
    layers = _validate_positive_integer(layers, "num_layers")
    shots = _validate_positive_integer(shots, "shots")
    full_parameterization = _validate_boolean(
        full_parameterization, "full_parameterization"
    )
    counterdiabatic = _validate_boolean(counterdiabatic, "counterdiabatic")
    verbose = _validate_boolean(verbose, "verbose")

    _operator_terms(problem_hamiltonian, "problem_hamiltonian")
    resolved_num_qubits = _resolve_qaoa_num_qubits(
        problem_hamiltonian,
        reference_hamiltonian,
        num_qubits,
    )
    if reference_hamiltonian is None:
        reference_hamiltonian = _default_mixer(resolved_num_qubits)
    _operator_terms(
        reference_hamiltonian,
        "reference_hamiltonian",
        exclude_identity=True,
    )

    parameter_count = get_num_qaoa_parameters(
        problem_hamiltonian,
        layers,
        reference_operator=reference_hamiltonian,
        full_parameterization=full_parameterization,
        counterdiabatic=counterdiabatic,
        num_qubits=resolved_num_qubits,
    )
    if initial_parameters is None:
        parameters = np.zeros(parameter_count, dtype=float)
    else:
        parameters = _validate_parameter_vector(
            initial_parameters,
            parameter_count,
            layers,
            label="initial_parameters",
        )

    ansatz = _make_generic_qaoa_kernel(
        problem_hamiltonian,
        reference_hamiltonian,
        num_qubits=resolved_num_qubits,
        layers=layers,
        full_parameterization=full_parameterization,
        counterdiabatic=counterdiabatic,
    )
    from .vqe_solver import VQE_Solver, _normalize_optimizer

    normalized_optimizer = _normalize_optimizer(
        optimizer,
        tol=tol,
        max_iterations=max_iterations,
        optimizer_options=optimizer_options,
    )
    solution = VQE_Solver(
        optimizer=normalized_optimizer,
        backend=backend,
        verbose=verbose,
    ).solve(
        problem_hamiltonian,
        ansatz,
        resolved_num_qubits,
        initial_params=parameters,
    )
    optimal_parameters = tuple(
        float(value) for value in solution["optimal_parameters"]
    )
    counts = rocq.sample(
        ansatz,
        shots,
        np.asarray(optimal_parameters, dtype=float),
        backend=backend,
    )
    return QAOAResult(
        optimal_value=solution["optimal_energy"],
        optimal_parameters=optimal_parameters,
        optimal_config=counts,
    )


def get_operator_pool(name: str, *, num_qubits=None):
    """Generate the CUDA-QX QAOA ADAPT operator pool."""

    if not isinstance(name, str) or not name:
        raise ValueError("operator pool name must be a non-empty string.")
    if name != "qaoa":
        raise ValueError("only the 'qaoa' operator pool is currently supported.")
    if num_qubits is None:
        raise ValueError("num_qubits is required for the qaoa operator pool.")
    count = _validate_nonnegative_integer(num_qubits, "num_qubits")
    if rocq is None:
        raise RuntimeError("Canonical 'rocq' package is required to build operator pools.")
    from rocq.operator import PauliOperator

    pool = [PauliOperator(f"X{qubit}") for qubit in range(count)]
    pool.extend(PauliOperator(f"Y{qubit}") for qubit in range(count))
    pair_words = ("XX", "YY", "YZ", "ZY", "XY", "YX", "XZ", "ZX")
    for left_pauli, right_pauli in pair_words:
        for left in range(count):
            for right in range(left + 1, count):
                pool.append(
                    PauliOperator(
                        f"{left_pauli}{left} {right_pauli}{right}"
                    )
                )
    return pool


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
    expected_params = get_num_qaoa_parameters(layers=layers)
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
            "parameter_count": expected_params,
            "layers": layers,
            "num_qubits": num_qubits,
            "backend": backend,
        }
    )
    return result
