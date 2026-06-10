from __future__ import annotations

import numpy as np

from qiskit.primitives import BaseEstimatorV2
from qiskit.primitives.containers import DataBin, EstimatorPub, PrimitiveResult, PubResult
from qiskit.primitives.primitive_job import PrimitiveJob


def _sparse_observable_terms(observable, num_qubits: int):
    terms = []
    for paulis, indices, coeff in observable.as_paulis().to_sparse_list():
        label = ["I"] * int(num_qubits)
        for pauli, qubit in zip(str(paulis).upper(), indices):
            if int(qubit) >= int(num_qubits):
                raise ValueError("Observable acts on more qubits than the circuit.")
            label[int(num_qubits) - 1 - int(qubit)] = pauli
        terms.append((complex(coeff), "".join(label)))
    return terms


def _observable_terms(observable, num_qubits: int | None = None):
    try:
        from qiskit.quantum_info import Pauli, SparsePauliOp
    except ImportError as exc:
        raise ImportError("Qiskit quantum_info is required for native expectation estimation.") from exc

    if isinstance(observable, SparsePauliOp):
        return [(complex(coeff), pauli.to_label()) for coeff, pauli in zip(observable.coeffs, observable.paulis)]

    if isinstance(observable, Pauli):
        return [(1.0 + 0.0j, observable.to_label())]

    if isinstance(observable, str):
        return [(1.0 + 0.0j, observable)]

    if isinstance(observable, dict):
        return [(complex(coeff), str(label)) for label, coeff in observable.items()]

    if isinstance(observable, tuple) and len(observable) == 2 and isinstance(observable[0], str):
        label, coeff = observable
        return [(complex(coeff), label)]

    if isinstance(observable, list):
        return [(complex(coeff), str(label)) for label, coeff in observable]

    if hasattr(observable, "as_paulis") and hasattr(observable, "to_sparse_list"):
        if num_qubits is None:
            raise ValueError("SparseObservable conversion requires the circuit qubit count.")
        return _sparse_observable_terms(observable, int(num_qubits))

    if hasattr(observable, "to_sparse_pauli_op"):
        return _observable_terms(observable.to_sparse_pauli_op(), num_qubits=num_qubits)

    raise TypeError(f"Unsupported Qiskit observable type: {type(observable)!r}")


def _dense_operator_matrix(observable, num_qubits: int):
    try:
        from qiskit.quantum_info import Operator
    except ImportError as exc:
        raise ImportError("Qiskit quantum_info is required for native expectation estimation.") from exc

    if not isinstance(observable, Operator):
        return None

    matrix = np.ascontiguousarray(np.asarray(observable.data, dtype=np.complex128))
    expected_dim = 1 << int(num_qubits)
    if matrix.shape != (expected_dim, expected_dim):
        raise ValueError(
            "Dense Qiskit Operator observables must act on all circuit qubits. "
            "Use SparsePauliOp/Pauli for padded or partial observables."
        )
    return matrix


def _label_to_runtime_term(label: str, num_qubits: int):
    if len(label) > num_qubits:
        raise ValueError("Observable acts on more qubits than the circuit.")

    # Qiskit labels are big-endian strings; rocQuantum targets are qubit indices.
    padded = "I" * (num_qubits - len(label)) + label.upper()
    paulis = []
    targets = []
    for qubit, pauli in enumerate(reversed(padded)):
        if pauli == "I":
            continue
        if pauli not in {"X", "Y", "Z"}:
            raise ValueError("Pauli labels may only contain I, X, Y, or Z.")
        paulis.append(pauli)
        targets.append(qubit)
    return "".join(paulis), targets


def _canonical_observable_label(label: str, num_qubits: int):
    normalized_label = str(label).upper()
    if len(normalized_label) > int(num_qubits):
        raise ValueError("Observable acts on more qubits than the circuit.")
    return "I" * (int(num_qubits) - len(normalized_label)) + normalized_label


def _combine_observable_terms(terms, num_qubits: int):
    combined = {}
    for coeff, label in terms:
        normalized_label = _canonical_observable_label(label, int(num_qubits))
        combined[normalized_label] = combined.get(normalized_label, 0.0 + 0.0j) + complex(coeff)
    return [
        (coeff, label)
        for label, coeff in sorted(combined.items())
        if abs(coeff) > 1e-15
    ]


def _observable_signature(observable, num_qubits: int):
    return tuple(
        _combine_observable_terms(
            _observable_terms(observable, num_qubits=int(num_qubits)),
            int(num_qubits),
        )
    )


def _observable_plan(observable, num_qubits: int):
    matrix = _dense_operator_matrix(observable, int(num_qubits))
    if matrix is not None:
        cache_key = (
            "matrix",
            matrix.shape,
            tuple(complex(value) for value in matrix.reshape(-1)),
        )
        return cache_key, ("matrix", matrix)

    signature = _observable_signature(observable, int(num_qubits))
    return ("pauli", signature), ("pauli", signature)


def _estimate_combined_observable_terms(runtime, terms, num_qubits: int) -> float:
    result = 0.0 + 0.0j
    for coeff, label in terms:
        pauli_string, targets = _label_to_runtime_term(label, int(num_qubits))
        if not targets:
            result += coeff
        else:
            result += coeff * runtime.expectation_pauli_string(pauli_string, targets)

    real_result = np.real_if_close(result)
    if np.iscomplexobj(real_result):
        raise ValueError("Observable expectation has a non-negligible imaginary component.")
    return float(real_result)


def _estimate_observable_plan(runtime, plan, num_qubits: int) -> float:
    kind, payload = plan
    if kind == "pauli":
        return _estimate_combined_observable_terms(runtime, payload, int(num_qubits))

    if kind == "matrix":
        result = runtime.expectation_matrix(payload, list(range(int(num_qubits))))
        real_result = np.real_if_close(result)
        if np.iscomplexobj(real_result):
            raise ValueError("Observable expectation has a non-negligible imaginary component.")
        return float(real_result)

    raise TypeError(f"Unsupported observable plan kind: {kind!r}")


def estimate_observable(runtime, observable, num_qubits: int) -> float:
    _, plan = _observable_plan(observable, int(num_qubits))
    return _estimate_observable_plan(runtime, plan, int(num_qubits))


def estimate_pauli_observable(runtime, observable, num_qubits: int) -> float:
    return _estimate_combined_observable_terms(
        runtime,
        _observable_signature(observable, int(num_qubits)),
        int(num_qubits),
    )


def _index_for_shape(index, shape):
    if not shape:
        return ()

    offset = len(index) - len(shape)
    return tuple(0 if dim == 1 else index[offset + axis] for axis, dim in enumerate(shape))


class RocQuantumEstimator(BaseEstimatorV2):
    """Native Qiskit EstimatorV2 backed by rocQuantum Pauli expectations."""

    def __init__(self, backend, *, default_precision: float = 0.0):
        self._backend = backend
        self._default_precision = float(default_precision)

    def run(self, pubs, *, precision: float | None = None):
        target_precision = self._default_precision if precision is None else float(precision)
        coerced_pubs = [EstimatorPub.coerce(pub, target_precision) for pub in pubs]
        job = PrimitiveJob(self._run, coerced_pubs)
        job._submit()
        return job

    def _run(self, pubs):
        results = [self._run_pub(pub) for pub in pubs]
        return PrimitiveResult(results, metadata={"version": 2})

    def _run_pub(self, pub):
        evs = np.empty(pub.shape, dtype=float)
        stds = np.zeros(pub.shape, dtype=float)
        bound_circuits = np.asarray(pub.parameter_values.bind_all(pub.circuit), dtype=object)
        indices_by_parameter = {}

        for index in np.ndindex(pub.shape):
            parameter_index = _index_for_shape(index, pub.parameter_values.shape)
            indices_by_parameter.setdefault(parameter_index, []).append(index)

        for parameter_index, indices in indices_by_parameter.items():
            circuit = bound_circuits[parameter_index]
            self._backend._apply_circuit(circuit, include_global_phase=False)
            observable_cache = {}
            for index in indices:
                observable_index = _index_for_shape(index, pub.observables.shape)
                observable = pub.observables[observable_index]
                cache_key, plan = _observable_plan(observable, int(circuit.num_qubits))
                if cache_key not in observable_cache:
                    observable_cache[cache_key] = _estimate_observable_plan(
                        self._backend._runtime,
                        plan,
                        circuit.num_qubits,
                    )
                evs[index] = observable_cache[cache_key]

        if pub.shape == ():
            evs = evs[()]
            stds = stds[()]

        return PubResult(
            DataBin(evs=evs, stds=stds, shape=pub.shape),
            metadata={
                "target_precision": pub.precision,
                "shots": 0,
                "circuit_metadata": getattr(pub.circuit, "metadata", None) or {},
                "native": True,
            },
        )
