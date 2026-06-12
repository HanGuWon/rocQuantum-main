from __future__ import annotations

from collections.abc import Mapping
from numbers import Real

import numpy as np

from qiskit.primitives import BaseEstimatorV2
from qiskit.primitives.containers import BindingsArray, DataBin, EstimatorPub, PrimitiveResult, PubResult
from qiskit.primitives.containers.observables_array import object_array
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


def _dense_operator_payload(observable):
    try:
        from qiskit.quantum_info import Operator
    except ImportError as exc:
        raise ImportError("Qiskit quantum_info is required for native expectation estimation.") from exc

    if isinstance(observable, Operator):
        return observable, None

    if isinstance(observable, tuple) and len(observable) == 2 and isinstance(observable[0], Operator):
        return observable[0], observable[1]

    if isinstance(observable, Mapping) and isinstance(observable.get("operator"), Operator):
        targets = observable.get("targets", observable.get("qargs"))
        if targets is not None:
            return observable["operator"], targets

    return None


def _normalize_dense_operator_targets(targets, target_qubits: int, num_qubits: int):
    try:
        normalized = tuple(int(qubit) for qubit in targets)
    except TypeError as exc:
        raise TypeError("Dense Qiskit Operator targets must be an iterable of qubit indices.") from exc

    if len(normalized) != int(target_qubits):
        raise ValueError("Dense Qiskit Operator explicit targets must match matrix qubit count.")
    if len(set(normalized)) != len(normalized):
        raise ValueError("Dense Qiskit Operator explicit targets must be unique.")
    if any(qubit < 0 or qubit >= int(num_qubits) for qubit in normalized):
        raise ValueError("Dense Qiskit Operator explicit targets must be circuit qubit indices.")
    return list(normalized)


def _dense_operator_plan(observable, num_qubits: int):
    payload = _dense_operator_payload(observable)
    if payload is None:
        return None
    observable, explicit_targets = payload

    matrix = np.ascontiguousarray(np.asarray(observable.data, dtype=np.complex128))
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Dense Qiskit Operator observables must be square matrices.")

    dimension = int(matrix.shape[0])
    if dimension == 0 or dimension & (dimension - 1):
        raise ValueError("Dense Qiskit Operator dimension must be a power of two.")

    target_qubits = dimension.bit_length() - 1
    if target_qubits == 0:
        raise ValueError("Dense Qiskit Operator observables must act on at least one qubit.")
    if target_qubits > int(num_qubits):
        raise ValueError("Dense Qiskit Operator observable acts on more qubits than the circuit.")

    expected_dim = 1 << target_qubits
    if matrix.shape != (expected_dim, expected_dim):
        raise ValueError(
            "Dense Qiskit Operator dimension must equal 2^k for some k <= circuit qubits."
        )

    if explicit_targets is not None:
        return matrix, _normalize_dense_operator_targets(explicit_targets, target_qubits, int(num_qubits))

    input_dims = tuple(int(dim) for dim in observable.input_dims())
    output_dims = tuple(int(dim) for dim in observable.output_dims())
    if input_dims != output_dims:
        raise ValueError("Dense Qiskit Operator observables must have matching input/output dimensions.")
    if len(input_dims) > int(num_qubits):
        raise ValueError("Dense Qiskit Operator dimension metadata exceeds circuit qubits.")
    if any(dim not in {1, 2} for dim in input_dims):
        raise ValueError("Dense Qiskit Operator dimensions must describe qubit or trivial axes.")

    targets = [
        qubit
        for qubit, dim in enumerate(reversed(input_dims))
        if dim == 2
    ]
    if len(targets) != target_qubits:
        raise ValueError("Dense Qiskit Operator dimension metadata does not match matrix size.")

    return matrix, targets


def _is_dense_operator_observable(observable) -> bool:
    try:
        return _dense_operator_payload(observable) is not None
    except ImportError:
        return False


def _dense_operator_object_array(observables, *, copy: bool):
    if _is_dense_operator_observable(observables):
        observable_array = np.empty((), dtype=object)
        observable_array[()] = observables
        return observable_array.copy() if copy else observable_array
    return object_array(observables, copy=copy)


def _contains_dense_operator_observable(observables) -> bool:
    if _is_dense_operator_observable(observables):
        return True
    try:
        observable_array = object_array(observables, copy=False)
    except (TypeError, ValueError):
        return _is_dense_operator_observable(observables)
    return any(_is_dense_operator_observable(observable) for observable in observable_array.flat)


def _validate_precision(precision: float | None) -> None:
    if precision is None:
        return
    if not isinstance(precision, Real):
        raise TypeError(f"precision must be a real number, not {type(precision)}.")
    if precision < 0:
        raise ValueError("precision must be non-negative")


class _DenseOperatorObservablesArray:
    def __init__(self, observables):
        self._array = _dense_operator_object_array(observables, copy=True)
        self._shape = self._array.shape

    @property
    def shape(self):
        return self._shape

    def __getitem__(self, index):
        return self._array[index]


class _DenseOperatorEstimatorPub:
    def __init__(self, circuit, observables, parameter_values=None, precision=None):
        self.circuit = circuit
        self.observables = _DenseOperatorObservablesArray(observables)
        self.parameter_values = parameter_values or BindingsArray()
        self.precision = precision
        self.shape = np.broadcast_shapes(self.observables.shape, self.parameter_values.shape)

    @classmethod
    def coerce(cls, pub, precision: float | None = None):
        _validate_precision(precision)
        if isinstance(pub, EstimatorPub):
            return pub
        if len(pub) not in [2, 3, 4]:
            raise ValueError(f"The length of pub must be 2, 3 or 4, but length {len(pub)} is given.")

        circuit = pub[0]
        parameter_values = None
        if len(pub) > 2 and pub[2] is not None:
            values = pub[2]
            if not isinstance(values, (BindingsArray, Mapping)):
                values = {tuple(circuit.parameters): values}
            parameter_values = BindingsArray.coerce(values)

        if len(pub) > 3 and pub[3] is not None:
            precision = pub[3]
            _validate_precision(precision)

        return cls(circuit, pub[1], parameter_values=parameter_values, precision=precision)


def _coerce_estimator_pub(pub, precision: float | None):
    if isinstance(pub, EstimatorPub):
        return pub

    try:
        pub_len = len(pub)
    except TypeError:
        return EstimatorPub.coerce(pub, precision)

    if pub_len in [2, 3, 4] and _contains_dense_operator_observable(pub[1]):
        return _DenseOperatorEstimatorPub.coerce(pub, precision)
    return EstimatorPub.coerce(pub, precision)


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
    dense_plan = _dense_operator_plan(observable, int(num_qubits))
    if dense_plan is not None:
        matrix, targets = dense_plan
        cache_key = (
            "matrix",
            tuple(targets),
            matrix.shape,
            tuple(complex(value) for value in matrix.reshape(-1)),
        )
        return cache_key, ("matrix", (matrix, tuple(targets)))

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


def _estimate_combined_observable_terms_batch(runtime, terms, num_qubits: int) -> np.ndarray:
    result = np.zeros(runtime.batch_size(), dtype=np.complex128)
    for coeff, label in terms:
        pauli_string, targets = _label_to_runtime_term(label, int(num_qubits))
        if not targets:
            result += coeff
        else:
            result += coeff * runtime.expectation_pauli_string_batch(pauli_string, targets)

    real_result = np.real_if_close(result)
    if np.iscomplexobj(real_result):
        raise ValueError("Observable expectation has a non-negligible imaginary component.")
    return np.asarray(real_result, dtype=float)


def _estimate_observable_plan(runtime, plan, num_qubits: int) -> float:
    kind, payload = plan
    if kind == "pauli":
        return _estimate_combined_observable_terms(runtime, payload, int(num_qubits))

    if kind == "matrix":
        matrix, targets = payload
        result = runtime.expectation_matrix(matrix, targets)
        real_result = np.real_if_close(result)
        if np.iscomplexobj(real_result):
            raise ValueError("Observable expectation has a non-negligible imaginary component.")
        return float(real_result)

    raise TypeError(f"Unsupported observable plan kind: {kind!r}")


def _estimate_observable_plan_batch(runtime, plan, num_qubits: int) -> np.ndarray:
    kind, payload = plan
    if kind == "pauli":
        return _estimate_combined_observable_terms_batch(runtime, payload, int(num_qubits))

    if kind == "matrix":
        matrix, targets = payload
        result = runtime.expectation_matrix_batch(matrix, targets)
        real_result = np.real_if_close(result)
        if np.iscomplexobj(real_result):
            raise ValueError("Observable expectation has a non-negligible imaginary component.")
        return np.asarray(real_result, dtype=float)

    raise TypeError(f"Unsupported observable plan kind: {kind!r}")


def estimate_observable(runtime, observable, num_qubits: int) -> float:
    _, plan = _observable_plan(observable, int(num_qubits))
    return _estimate_observable_plan(runtime, plan, int(num_qubits))


def estimate_observable_batch(runtime, observable, num_qubits: int) -> np.ndarray:
    _, plan = _observable_plan(observable, int(num_qubits))
    return _estimate_observable_plan_batch(runtime, plan, int(num_qubits))


def estimate_pauli_observable(runtime, observable, num_qubits: int) -> float:
    return _estimate_combined_observable_terms(
        runtime,
        _observable_signature(observable, int(num_qubits)),
        int(num_qubits),
    )


def estimate_pauli_observable_batch(runtime, observable, num_qubits: int) -> np.ndarray:
    return _estimate_combined_observable_terms_batch(
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
        coerced_pubs = [_coerce_estimator_pub(pub, target_precision) for pub in pubs]
        job = PrimitiveJob(self._run, coerced_pubs)
        job._submit()
        return job

    def _run(self, pubs):
        results = [self._run_pub(pub) for pub in pubs]
        return PrimitiveResult(results, metadata={"version": 2})

    def _run_pub(self, pub):
        batched_result = self._try_run_pub_batched_parameters(pub)
        if batched_result is not None:
            return batched_result

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

    def _try_run_pub_batched_parameters(self, pub):
        parameter_shape = pub.parameter_values.shape
        if not parameter_shape:
            return None

        bound_circuits = np.asarray(pub.parameter_values.bind_all(pub.circuit), dtype=object)
        parameter_indices = list(np.ndindex(parameter_shape))
        if len(parameter_indices) <= 1:
            return None

        circuits = [bound_circuits[index] for index in parameter_indices]

        try:
            self._backend._apply_circuit_batch(circuits, include_global_phase=False)
        except (NotImplementedError, RuntimeError, TypeError, ValueError):
            return None

        observable_indices = sorted(
            {
                _index_for_shape(index, pub.observables.shape)
                for index in np.ndindex(pub.shape)
            }
        )
        observable_cache_keys = {}
        observable_values_by_cache = {}
        for observable_index in observable_indices:
            observable = pub.observables[observable_index]
            try:
                cache_key, plan = _observable_plan(observable, int(pub.circuit.num_qubits))
                if cache_key not in observable_values_by_cache:
                    observable_values_by_cache[cache_key] = _estimate_observable_plan_batch(
                        self._backend._runtime,
                        plan,
                        int(pub.circuit.num_qubits),
                    )
                observable_cache_keys[observable_index] = cache_key
            except (NotImplementedError, RuntimeError, TypeError, ValueError):
                return None

        parameter_offsets = {index: offset for offset, index in enumerate(parameter_indices)}
        evs = np.empty(pub.shape, dtype=float)
        stds = np.zeros(pub.shape, dtype=float)
        for index in np.ndindex(pub.shape):
            parameter_index = _index_for_shape(index, parameter_shape)
            observable_index = _index_for_shape(index, pub.observables.shape)
            cache_key = observable_cache_keys[observable_index]
            evs[index] = observable_values_by_cache[cache_key][parameter_offsets[parameter_index]]

        return PubResult(
            DataBin(evs=evs, stds=stds, shape=pub.shape),
            metadata={
                "target_precision": pub.precision,
                "shots": 0,
                "circuit_metadata": getattr(pub.circuit, "metadata", None) or {},
                "native": True,
                "batched_parameters": True,
            },
        )
