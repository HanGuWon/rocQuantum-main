# pennylane_rocq/rocq_device.py
import pennylane as qml
from pennylane.operation import Operation
import numpy as np

try:
    from pennylane import QubitDevice
except ImportError:
    from pennylane.devices import QubitDevice

from rocquantum.framework_runtime import (
    RocQuantumRuntime,
    matrix_to_little_endian_wires,
    sample_rows_from_statevector,
    samples_to_binary_rows,
    statevector_to_little_endian_wires,
)

try:
    import rocquantum_bind
except ImportError:
    rocquantum_bind = None

# Mapping from PennyLane operation names to rocQuantum names
PENNYLANE_TO_ROCQ_GATES = {
    "PauliX": "X", "PauliY": "Y", "PauliZ": "Z",
    "Hadamard": "H", "S": "S", "T": "T",
    "CNOT": "CNOT", "CZ": "CZ", "SWAP": "SWAP",
}

NATIVE_PARAMETRIC_OPS = {"RX", "RY", "RZ", "CRX", "CRY", "CRZ"}
MATRIX_OPS = {
    "QubitUnitary",
    "ControlledQubitUnitary",
    "PhaseShift", "ControlledPhaseShift",
    "CPhaseShift00", "CPhaseShift01", "CPhaseShift10",
    "CH", "CY", "CCZ", "CRot",
    "MultiControlledX", "MultiRZ", "PauliRot",
    "IsingXX", "IsingYY", "IsingZZ", "IsingXY",
    "PSWAP", "ISWAP", "SISWAP", "SQISW", "ECR",
    "SingleExcitation", "SingleExcitationPlus", "SingleExcitationMinus",
    "DoubleExcitation", "DoubleExcitationPlus", "DoubleExcitationMinus",
    "OrbitalRotation", "FermionicSWAP",
    "Toffoli", "CSWAP",
}
DECOMPOSED_OPS = {
    "BasisEmbedding",
    "DiagonalQubitUnitary",
    "GlobalPhase",
    "GroverOperator",
    "Permute",
    "QFT",
    "QubitCarry",
    "QubitSum",
    "SelectPauliRot",
}
PENNYLANE_PAULI_TO_CHAR = {
    "Identity": "I",
    "PauliX": "X",
    "PauliY": "Y",
    "PauliZ": "Z",
}
PAULI_PRODUCTS = {
    ("I", "I"): (1.0 + 0.0j, "I"),
    ("I", "X"): (1.0 + 0.0j, "X"),
    ("I", "Y"): (1.0 + 0.0j, "Y"),
    ("I", "Z"): (1.0 + 0.0j, "Z"),
    ("X", "I"): (1.0 + 0.0j, "X"),
    ("X", "X"): (1.0 + 0.0j, "I"),
    ("X", "Y"): (0.0 + 1.0j, "Z"),
    ("X", "Z"): (0.0 - 1.0j, "Y"),
    ("Y", "I"): (1.0 + 0.0j, "Y"),
    ("Y", "X"): (0.0 - 1.0j, "Z"),
    ("Y", "Y"): (1.0 + 0.0j, "I"),
    ("Y", "Z"): (0.0 + 1.0j, "X"),
    ("Z", "I"): (1.0 + 0.0j, "Z"),
    ("Z", "X"): (0.0 + 1.0j, "Y"),
    ("Z", "Y"): (0.0 - 1.0j, "X"),
    ("Z", "Z"): (1.0 + 0.0j, "I"),
}


def _pauli_string_from_observable(observable, wire_map):
    if observable.name in PENNYLANE_PAULI_TO_CHAR:
        if observable.name == "Identity":
            return "", []
        if len(observable.wires) != 1:
            return None
        return PENNYLANE_PAULI_TO_CHAR[observable.name], [wire_map[observable.wires[0]]]

    if observable.name not in {"Prod", "Tensor"}:
        return None

    paulis = []
    targets = []
    seen_targets = set()
    for operand in getattr(observable, "operands", None) or getattr(observable, "obs", ()):
        term = _pauli_string_from_observable(operand, wire_map)
        if term is None:
            return None
        operand_paulis, operand_targets = term
        for pauli, target in zip(operand_paulis, operand_targets):
            if target in seen_targets:
                return None
            seen_targets.add(target)
            paulis.append(pauli)
            targets.append(target)
    return "".join(paulis), targets


def _projector_terms_from_observable(observable, wire_map):
    if observable.name != "Projector" or not getattr(observable, "parameters", None):
        return None

    bits = np.asarray(observable.parameters[0], dtype=int).reshape(-1)
    wires = list(observable.wires)
    if len(bits) != len(wires):
        return None
    if not np.all((bits == 0) | (bits == 1)):
        return None

    terms = [(1.0 + 0.0j, "", [])]
    for bit, wire in zip(bits, wires):
        target = wire_map[wire]
        sign = 1.0 if int(bit) == 0 else -1.0
        expanded_terms = []
        for coeff, pauli_string, targets in terms:
            expanded_terms.append((0.5 * coeff, pauli_string, list(targets)))
            expanded_terms.append((0.5 * sign * coeff, pauli_string + "Z", list(targets) + [target]))
        terms = expanded_terms
    return terms


def _pauli_terms_from_observable(observable, wire_map):
    if observable.name == "Hadamard":
        if len(observable.wires) != 1:
            return None
        target = wire_map[observable.wires[0]]
        coeff = 1.0 / np.sqrt(2.0)
        return [(coeff, "X", [target]), (coeff, "Z", [target])]

    projector_terms = _projector_terms_from_observable(observable, wire_map)
    if projector_terms is not None:
        return projector_terms

    term = _pauli_string_from_observable(observable, wire_map)
    if term is not None:
        pauli_string, targets = term
        return [(1.0, pauli_string, targets)]

    if observable.name == "SProd":
        base_terms = _pauli_terms_from_observable(observable.base, wire_map)
        if base_terms is None:
            return None
        return [(observable.scalar * coeff, pauli_string, targets)
                for coeff, pauli_string, targets in base_terms]

    if observable.name == "Sum":
        terms = []
        for operand in getattr(observable, "operands", ()):
            operand_terms = _pauli_terms_from_observable(operand, wire_map)
            if operand_terms is None:
                return None
            terms.extend(operand_terms)
        return terms

    if observable.name in {"LinearCombination", "Hamiltonian"} and callable(getattr(observable, "terms", None)):
        coeffs, observables = observable.terms()
        terms = []
        for coeff, sub_observable in zip(coeffs, observables):
            sub_terms = _pauli_terms_from_observable(sub_observable, wire_map)
            if sub_terms is None:
                return None
            terms.extend((coeff * sub_coeff, pauli_string, targets)
                         for sub_coeff, pauli_string, targets in sub_terms)
        return terms

    return None


def _term_to_pauli_map(pauli_string, targets):
    return {int(target): pauli for pauli, target in zip(pauli_string, targets)}


def _pauli_map_to_term(pauli_map):
    targets = sorted(pauli_map)
    return "".join(pauli_map[target] for target in targets), targets


def _canonical_pauli_term(term):
    coeff, pauli_string, targets = term
    pauli_map = _term_to_pauli_map(pauli_string, targets)
    canonical_paulis, canonical_targets = _pauli_map_to_term(pauli_map)
    return complex(coeff), canonical_paulis, tuple(canonical_targets)


def _combine_pauli_terms(terms):
    combined = {}
    for term in terms:
        coeff, pauli_string, targets = _canonical_pauli_term(term)
        key = (pauli_string, targets)
        combined[key] = combined.get(key, 0.0 + 0.0j) + coeff

    return [
        (coeff, pauli_string, list(targets))
        for (pauli_string, targets), coeff in combined.items()
        if abs(coeff) > 1e-15
    ]


def _multiply_pauli_terms(left, right):
    left_coeff, left_paulis, left_targets = left
    right_coeff, right_paulis, right_targets = right
    left_map = _term_to_pauli_map(left_paulis, left_targets)
    right_map = _term_to_pauli_map(right_paulis, right_targets)

    coeff = complex(left_coeff) * complex(right_coeff)
    product_map = {}
    for target in sorted(set(left_map) | set(right_map)):
        phase, pauli = PAULI_PRODUCTS[(left_map.get(target, "I"), right_map.get(target, "I"))]
        coeff *= phase
        if pauli != "I":
            product_map[target] = pauli

    pauli_string, targets = _pauli_map_to_term(product_map)
    return coeff, pauli_string, targets


def _pauli_square_terms(terms):
    combined_terms = _combine_pauli_terms(terms)
    return _combine_pauli_terms(
        _multiply_pauli_terms(left, right)
        for left in combined_terms
        for right in combined_terms
    )


def _evaluate_pauli_terms(runtime, terms):
    result = 0.0 + 0.0j
    for coeff, pauli_string, targets in _combine_pauli_terms(terms):
        if not targets:
            result += coeff
        else:
            result += coeff * runtime.expectation_pauli_string(pauli_string, targets)
    return result


def _evaluate_pauli_terms_batch(runtime, terms):
    result = np.zeros(runtime.batch_size(), dtype=np.complex128)
    for coeff, pauli_string, targets in _combine_pauli_terms(terms):
        if not targets:
            result += coeff
        else:
            result += coeff * runtime.expectation_pauli_string_batch(pauli_string, targets)
    return result


def _real_measurement_result(value, measurement_name):
    real_value = np.real_if_close(value)
    if np.iscomplexobj(real_value):
        raise ValueError(f"{measurement_name} has a non-negligible imaginary component.")
    return float(real_value)


def _sparse_hamiltonian_moments(state, observable, wire_order):
    sparse_matrix = observable.sparse_matrix(wire_order=wire_order, format="csr")
    h_state = sparse_matrix.dot(state)
    mean = np.vdot(state, h_state)
    second_moment = np.vdot(h_state, h_state)
    return mean, second_moment


def _native_sparse_hamiltonian_moments(runtime, observable, wire_order):
    sparse_matrix = observable.sparse_matrix(wire_order=wire_order, format="csr")
    return runtime.sparse_hamiltonian_moments(
        sparse_matrix.data,
        sparse_matrix.indices,
        sparse_matrix.indptr,
        sparse_matrix.shape,
    )


def _native_sparse_hamiltonian_moments_batch(runtime, payload):
    _, data, indices, indptr, shape = payload
    return runtime.sparse_hamiltonian_moments_batch(data, indices, indptr, shape)


def _hermitian_matrix_and_targets(observable, wire_map):
    if observable.name != "Hermitian":
        return None
    matrix = matrix_to_little_endian_wires(qml.matrix(observable))
    targets = [wire_map[wire] for wire in observable.wires]
    return matrix, targets


def _native_hermitian_expectation(runtime, observable, wire_map):
    components = _hermitian_matrix_and_targets(observable, wire_map)
    if components is None:
        raise NotImplementedError
    matrix, targets = components
    return runtime.expectation_matrix(matrix, targets)


def _observable_batch_payload(observable, wire_map, wire_order=None):
    if observable is None:
        return None

    terms = _pauli_terms_from_observable(observable, wire_map)
    if terms is not None:
        return "pauli", _combine_pauli_terms(terms)

    if observable.name == "SparseHamiltonian":
        sparse_matrix = observable.sparse_matrix(wire_order=wire_order or tuple(wire_map.keys()), format="csr")
        return (
            "sparse",
            np.asarray(sparse_matrix.data, dtype=np.complex128),
            np.asarray(sparse_matrix.indices, dtype=np.int64),
            np.asarray(sparse_matrix.indptr, dtype=np.int64),
            tuple(int(dim) for dim in sparse_matrix.shape),
        )

    components = _hermitian_matrix_and_targets(observable, wire_map)
    if components is not None:
        matrix, targets = components
        return "matrix", matrix, tuple(targets)

    return None


def _observable_batch_payload_matches(observable, wire_map, reference_payload, wire_order=None):
    current = _observable_batch_payload(observable, wire_map, wire_order=wire_order)
    if current is None or current[0] != reference_payload[0]:
        return False
    if reference_payload[0] == "pauli":
        return current[1] == reference_payload[1]
    if reference_payload[0] == "sparse":
        return (
            current[4] == reference_payload[4]
            and np.array_equal(current[1], reference_payload[1])
            and np.array_equal(current[2], reference_payload[2])
            and np.array_equal(current[3], reference_payload[3])
        )
    return current[2] == reference_payload[2] and np.array_equal(current[1], reference_payload[1])


def _evaluate_observable_batch_payload(runtime, payload):
    kind = payload[0]
    if kind == "pauli":
        return _evaluate_pauli_terms_batch(runtime, payload[1])
    if kind == "matrix":
        return runtime.expectation_matrix_batch(payload[1], payload[2])
    if kind == "sparse":
        means, _ = _native_sparse_hamiltonian_moments_batch(runtime, payload)
        return means
    raise TypeError(f"Unsupported observable batch payload kind: {kind!r}")


def _shot_count(shots):
    total_shots = getattr(shots, "total_shots", None)
    if total_shots is not None:
        return int(total_shots)
    return int(shots)


def _has_partitioned_shots(shots):
    return bool(getattr(shots, "has_partitioned_shots", False))


def _sample_result_from_rows(rows):
    rows = np.asarray(rows, dtype=int)
    if rows.ndim == 2 and rows.shape[1] == 1:
        return rows[:, 0]
    return rows


def _counts_result_from_rows(rows):
    counts = {}
    for row in np.asarray(rows, dtype=int):
        key = "".join(str(int(bit)) for bit in np.asarray(row).reshape(-1))
        counts[key] = counts.get(key, 0) + 1
    return counts


def _basis_state_bits(op, num_wires, op_name="BasisState"):
    if not getattr(op, "parameters", None):
        raise ValueError(f"{op_name} requires a computational basis vector.")
    bits = np.asarray(op.parameters[0], dtype=int).reshape(-1)
    if len(bits) != int(num_wires):
        raise ValueError(f"{op_name} length must match the number of target wires.")
    if not np.all((bits == 0) | (bits == 1)):
        raise ValueError(f"{op_name} entries must be 0 or 1.")
    return bits


def _control_value_is_one(value):
    if isinstance(value, str):
        return value in {"1", "True", "true"}
    return bool(value)


def _native_mcx_wire_indices(op, wire_map):
    control_wires = list(getattr(op, "control_wires", ()))
    target_wires = list(getattr(op, "target_wires", ()))
    control_values = getattr(op, "control_values", None)
    if control_values is None:
        control_values = getattr(op, "hyperparameters", {}).get("control_values")

    if len(target_wires) != 1 or not control_wires or control_values is None:
        return None
    if len(control_values) != len(control_wires):
        return None

    return [wire_map[wire] for wire in control_wires + target_wires]


def _mcx_control_values(op):
    control_values = getattr(op, "control_values", None)
    if control_values is None:
        control_values = getattr(op, "hyperparameters", {}).get("control_values")
    return list(control_values or ())


def _apply_mcx_with_control_values(runtime, wire_indices, control_values):
    if len(wire_indices) < 2:
        raise ValueError("MultiControlledX requires at least one control wire and one target wire.")
    if len(control_values) != len(wire_indices) - 1:
        raise ValueError("MultiControlledX control_values length does not match control wires.")

    open_controls = [
        wire_index
        for wire_index, value in zip(wire_indices[:-1], control_values)
        if not _control_value_is_one(value)
    ]
    flipped = []
    try:
        for wire_index in open_controls:
            runtime.apply_operation("X", [wire_index])
            flipped.append(wire_index)
        runtime.apply_operation("MCX", wire_indices)
        for wire_index in reversed(flipped):
            runtime.apply_operation("X", [wire_index])
        flipped.clear()
    finally:
        for wire_index in reversed(flipped):
            runtime.apply_operation("X", [wire_index])


def _controlled_qubit_unitary_components(op, wire_map):
    control_wires = list(getattr(op, "control_wires", ()))
    target_wires = list(getattr(op, "target_wires", ()))
    control_values = getattr(op, "control_values", None)
    if control_values is None:
        control_values = getattr(op, "hyperparameters", {}).get("control_values")

    if not control_wires or not target_wires or control_values is None:
        return None
    if len(control_values) != len(control_wires):
        return None

    base = getattr(op, "base", None)
    if base is None:
        base = getattr(op, "hyperparameters", {}).get("base")
    if base is None:
        return None

    try:
        matrix = matrix_to_little_endian_wires(qml.matrix(base))
    except (TypeError, ValueError, RuntimeError):
        return None
    expected_dimension = 1 << len(target_wires)
    if matrix.shape != (expected_dimension, expected_dimension):
        return None

    controls = [wire_map[wire] for wire in control_wires]
    targets = [wire_map[wire] for wire in target_wires]
    return matrix, controls, targets, list(control_values)


def _apply_controlled_qubit_unitary(runtime, matrix, controls, targets, control_values):
    open_controls = [
        wire_index
        for wire_index, value in zip(controls, control_values)
        if not _control_value_is_one(value)
    ]
    flipped = []
    try:
        for wire_index in open_controls:
            runtime.apply_operation("X", [wire_index])
            flipped.append(wire_index)
        runtime.apply_controlled_matrix(matrix, controls, targets)
        for wire_index in reversed(flipped):
            runtime.apply_operation("X", [wire_index])
        flipped.clear()
    finally:
        for wire_index in reversed(flipped):
            runtime.apply_operation("X", [wire_index])


def _parameter_value_matches(left, right):
    try:
        left_array = np.asarray(left)
        right_array = np.asarray(right)
        if left_array.shape or right_array.shape:
            return left_array.shape == right_array.shape and np.allclose(left_array, right_array)
    except (TypeError, ValueError):
        pass

    comparison = left == right
    if isinstance(comparison, np.ndarray):
        return bool(np.all(comparison))
    return bool(comparison)


def _parameter_lists_match(left, right):
    if len(left) != len(right):
        return False
    return all(_parameter_value_matches(left_value, right_value) for left_value, right_value in zip(left, right))


def _apply_native_or_matrix(runtime, native_name, wire_indices, op):
    try:
        runtime.apply_operation(native_name, wire_indices)
    except (NotImplementedError, RuntimeError, TypeError, ValueError):
        matrix = matrix_to_little_endian_wires(qml.matrix(op))
        runtime.apply_operation(
            "QubitUnitary",
            wire_indices,
            matrix=matrix,
        )


def _apply_multirz(runtime, wire_indices, theta):
    if not wire_indices:
        raise ValueError("MultiRZ requires at least one wire.")
    if len(wire_indices) == 1:
        runtime.apply_operation("RZ", [wire_indices[0]], [theta])
        return

    target = wire_indices[-1]
    controls = list(wire_indices[:-1])
    for control in controls:
        runtime.apply_operation("CNOT", [control, target])
    runtime.apply_operation("RZ", [target], [theta])
    for control in reversed(controls):
        runtime.apply_operation("CNOT", [control, target])


def _apply_multirz_batch(runtime, wire_indices, thetas):
    if not wire_indices:
        raise ValueError("MultiRZ requires at least one wire.")
    if len(wire_indices) == 1:
        runtime.apply_operation_batch("RZ", [wire_indices[0]], thetas)
        return

    target = wire_indices[-1]
    controls = list(wire_indices[:-1])
    for control in controls:
        runtime.apply_operation("CNOT", [control, target])
    runtime.apply_operation_batch("RZ", [target], thetas)
    for control in reversed(controls):
        runtime.apply_operation("CNOT", [control, target])


def _paulirot_pauli_word(op):
    return str(getattr(op, "hyperparameters", {}).get("pauli_word", ""))


def _paulirot_active_terms(wire_indices, pauli_word):
    if len(pauli_word) != len(wire_indices):
        raise ValueError("PauliRot pauli_word length must match its wires.")
    return [
        (wire_index, pauli)
        for wire_index, pauli in zip(wire_indices, pauli_word)
        if pauli != "I"
    ]


def _apply_paulirot_basis_change(runtime, active_terms, inverse=False):
    for wire_index, pauli in active_terms:
        if pauli == "X":
            runtime.apply_operation("H", [wire_index])
        elif pauli == "Y":
            angle = -np.pi / 2 if inverse else np.pi / 2
            runtime.apply_operation("RX", [wire_index], [angle])
        elif pauli != "Z":
            raise ValueError(f"Unsupported PauliRot pauli word character {pauli!r}.")


def _apply_paulirot(runtime, wire_indices, theta, pauli_word):
    active_terms = _paulirot_active_terms(wire_indices, pauli_word)
    if not active_terms:
        if wire_indices:
            _apply_global_phase(runtime, wire_indices[0], -theta / 2)
        return

    _apply_paulirot_basis_change(runtime, active_terms)
    _apply_multirz(runtime, [wire_index for wire_index, _ in active_terms], theta)
    _apply_paulirot_basis_change(runtime, active_terms, inverse=True)


def _apply_paulirot_batch(runtime, wire_indices, thetas, pauli_word):
    active_terms = _paulirot_active_terms(wire_indices, pauli_word)
    if not active_terms:
        return

    _apply_paulirot_basis_change(runtime, active_terms)
    _apply_multirz_batch(runtime, [wire_index for wire_index, _ in active_terms], thetas)
    _apply_paulirot_basis_change(runtime, active_terms, inverse=True)


def _supports_native_phase_decomposition(runtime):
    simulator = runtime.simulator
    has_gate_dispatch = callable(getattr(simulator, "apply_gate", None))
    has_matrix_dispatch = callable(getattr(simulator, "apply_matrix", None)) or callable(
        getattr(simulator, "ApplyGate", None)
    )
    return has_gate_dispatch and has_matrix_dispatch


def _supports_native_parametric_decomposition(runtime):
    return callable(getattr(runtime.simulator, "apply_gate", None))


def _supports_native_gate_decomposition(runtime):
    return callable(getattr(runtime.simulator, "apply_gate", None))


def _apply_global_phase(runtime, wire_index, phase):
    if not getattr(runtime, "preserve_global_phase", True):
        return
    factor = np.exp(1j * phase)
    matrix = np.array([[factor, 0.0], [0.0, factor]], dtype=np.complex128)
    runtime.apply_operation("QubitUnitary", [wire_index], matrix=matrix)


def _apply_global_phase_operation(runtime, wire_indices, theta, fallback_wire=None):
    if not getattr(runtime, "preserve_global_phase", True):
        return
    if wire_indices:
        wire_index = wire_indices[0]
    elif fallback_wire is not None:
        wire_index = fallback_wire
    else:
        raise ValueError("GlobalPhase requires at least one device wire for runtime dispatch.")
    _apply_global_phase(runtime, wire_index, -theta)


def _apply_phase_shift(runtime, wire_indices, theta):
    if len(wire_indices) != 1:
        raise ValueError("PhaseShift requires exactly one wire.")

    wire_index = wire_indices[0]
    try:
        runtime.apply_operation("P", [wire_index], [theta])
        return
    except (NotImplementedError, RuntimeError, TypeError, ValueError):
        pass
    _apply_global_phase(runtime, wire_index, 0.5 * theta)
    runtime.apply_operation("RZ", [wire_index], [theta])


def _apply_phase_shift_batch(runtime, wire_indices, thetas):
    if len(wire_indices) != 1:
        raise ValueError("PhaseShift requires exactly one wire.")

    try:
        runtime.apply_operation_batch("P", [wire_indices[0]], thetas)
        return
    except (NotImplementedError, RuntimeError, TypeError, ValueError):
        pass
    runtime.apply_operation_batch("RZ", [wire_indices[0]], thetas)


def _apply_sx(runtime, wire_indices):
    if len(wire_indices) != 1:
        raise ValueError("SX requires exactly one wire.")

    wire_index = wire_indices[0]
    _apply_global_phase(runtime, wire_index, np.pi / 4)
    runtime.apply_operation("RX", [wire_index], [np.pi / 2])


def _apply_controlled_phase_shift(runtime, wire_indices, theta):
    if len(wire_indices) != 2:
        raise ValueError("ControlledPhaseShift requires exactly two wires.")

    control, target = wire_indices
    try:
        runtime.apply_operation("CP", [control, target], [theta])
        return
    except (NotImplementedError, RuntimeError, TypeError, ValueError):
        pass
    _apply_global_phase(runtime, control, 0.25 * theta)
    runtime.apply_operation("RZ", [control], [0.5 * theta])
    runtime.apply_operation("CNOT", [control, target])
    runtime.apply_operation("RZ", [target], [-0.5 * theta])
    runtime.apply_operation("CNOT", [control, target])
    runtime.apply_operation("RZ", [target], [0.5 * theta])


def _apply_controlled_phase_shift_batch(runtime, wire_indices, thetas):
    if len(wire_indices) != 2:
        raise ValueError("ControlledPhaseShift requires exactly two wires.")

    control, target = wire_indices
    try:
        runtime.apply_operation_batch("CP", [control, target], thetas)
        return
    except (NotImplementedError, RuntimeError, TypeError, ValueError):
        pass
    half_thetas = [0.5 * theta for theta in thetas]
    runtime.apply_operation_batch("RZ", [control], half_thetas)
    runtime.apply_operation("CNOT", [control, target])
    runtime.apply_operation_batch("RZ", [target], [-theta for theta in half_thetas])
    runtime.apply_operation("CNOT", [control, target])
    runtime.apply_operation_batch("RZ", [target], half_thetas)


def _gray_code(rank):
    if rank == 0:
        return np.array([0], dtype=int)
    code = np.array([0, 1], dtype=int)
    for index in range(1, rank):
        code = np.concatenate([code, code[::-1] + 2**index])
    return code


def _normalized_walsh_hadamard(values):
    transformed = np.asarray(values, dtype=float).copy()
    width = transformed.shape[-1]
    step = 1
    while step < width:
        for start in range(0, width, 2 * step):
            left = transformed[..., start:start + step].copy()
            right = transformed[..., start + step:start + 2 * step].copy()
            transformed[..., start:start + step] = left + right
            transformed[..., start + step:start + 2 * step] = left - right
        step *= 2
    return transformed / width


def _uniform_rz_thetas(angles):
    angles = np.asarray(angles, dtype=float)
    control_count = int(np.log2(angles.shape[-1]))
    transformed = _normalized_walsh_hadamard(angles)
    return transformed[..., _gray_code(control_count)]


def _apply_uniform_rz(runtime, control_indices, target, angles):
    thetas = _uniform_rz_thetas(angles)
    tolerance = np.finfo(np.asarray(thetas).dtype).eps
    if not np.any(np.abs(thetas) > tolerance):
        return
    if not control_indices:
        runtime.apply_operation("RZ", [target], [thetas[0]])
        return

    code = _gray_code(len(control_indices))
    control_indexes = np.log2(code ^ np.roll(code, -1)).astype(int)
    for theta, control_index in zip(thetas, control_indexes):
        if abs(theta) > tolerance:
            runtime.apply_operation("RZ", [target], [theta])
        runtime.apply_operation("CNOT", [control_indices[control_index], target])


def _apply_uniform_rz_batch(runtime, control_indices, target, angles_by_batch):
    thetas_by_batch = _uniform_rz_thetas(angles_by_batch)
    tolerance = np.finfo(np.asarray(thetas_by_batch).dtype).eps
    if not np.any(np.abs(thetas_by_batch) > tolerance):
        return
    if not control_indices:
        runtime.apply_operation_batch("RZ", [target], thetas_by_batch[:, 0])
        return

    code = _gray_code(len(control_indices))
    control_indexes = np.log2(code ^ np.roll(code, -1)).astype(int)
    for theta_column, control_index in zip(thetas_by_batch.T, control_indexes):
        if np.any(np.abs(theta_column) > tolerance):
            runtime.apply_operation_batch("RZ", [target], theta_column)
        runtime.apply_operation("CNOT", [control_indices[control_index], target])


def _diagonal_unitary_phases(diagonal, wire_count):
    phases = np.angle(np.asarray(diagonal, dtype=np.complex128).reshape(-1))
    expected = 1 << wire_count
    if phases.shape[-1] != expected:
        raise ValueError("DiagonalQubitUnitary diagonal length must match its wires.")
    return phases


def _apply_diagonal_phases(runtime, wire_indices, phases):
    if len(wire_indices) == 1:
        mean = 0.5 * (phases[0] + phases[1])
        diff = phases[1] - phases[0]
        _apply_global_phase_operation(runtime, wire_indices, -mean)
        runtime.apply_operation("RZ", [wire_indices[0]], [diff])
        return

    means = 0.5 * (phases[::2] + phases[1::2])
    diffs = phases[1::2] - phases[::2]
    _apply_diagonal_phases(runtime, wire_indices[:-1], means)
    _apply_uniform_rz(runtime, wire_indices[:-1], wire_indices[-1], diffs)


def _apply_diagonal_phases_batch(runtime, wire_indices, phases_by_batch):
    if len(wire_indices) == 1:
        means = 0.5 * (phases_by_batch[:, 0] + phases_by_batch[:, 1])
        diffs = phases_by_batch[:, 1] - phases_by_batch[:, 0]
        if getattr(runtime, "preserve_global_phase", True):
            if not np.allclose(means, means[0]):
                raise NotImplementedError("Batched GlobalPhase sweeps require measurement-only execution.")
            _apply_global_phase_operation(runtime, wire_indices, -means[0])
        runtime.apply_operation_batch("RZ", [wire_indices[0]], diffs)
        return

    means = 0.5 * (phases_by_batch[:, ::2] + phases_by_batch[:, 1::2])
    diffs = phases_by_batch[:, 1::2] - phases_by_batch[:, ::2]
    _apply_diagonal_phases_batch(runtime, wire_indices[:-1], means)
    _apply_uniform_rz_batch(runtime, wire_indices[:-1], wire_indices[-1], diffs)


def _apply_diagonal_qubit_unitary(runtime, wire_indices, diagonal):
    if not wire_indices:
        raise ValueError("DiagonalQubitUnitary requires at least one wire.")
    phases = _diagonal_unitary_phases(diagonal, len(wire_indices))
    _apply_diagonal_phases(runtime, wire_indices, phases)


def _apply_diagonal_qubit_unitary_batch(runtime, wire_indices, diagonals):
    if not wire_indices:
        raise ValueError("DiagonalQubitUnitary requires at least one wire.")
    phases_by_batch = np.vstack(
        [_diagonal_unitary_phases(diagonal, len(wire_indices)) for diagonal in diagonals]
    )
    _apply_diagonal_phases_batch(runtime, wire_indices, phases_by_batch)


def _select_pauli_rot_axis(op):
    return str(getattr(op, "hyperparameters", {}).get("rot_axis", "Z")).upper()


def _apply_select_pauli_rot(runtime, wire_indices, angles, rot_axis):
    if len(wire_indices) < 2:
        raise ValueError("SelectPauliRot requires at least one control wire and one target wire.")
    control_indices = wire_indices[:-1]
    target = wire_indices[-1]
    expected = 1 << len(control_indices)
    if np.asarray(angles).reshape(-1).shape[-1] != expected:
        raise ValueError("SelectPauliRot angle count must match its control wires.")

    if rot_axis == "X":
        runtime.apply_operation("H", [target])
        _apply_uniform_rz(runtime, control_indices, target, angles)
        runtime.apply_operation("H", [target])
    elif rot_axis == "Y":
        runtime.apply_operation("SDG", [target])
        runtime.apply_operation("H", [target])
        _apply_uniform_rz(runtime, control_indices, target, angles)
        runtime.apply_operation("H", [target])
        runtime.apply_operation("S", [target])
    elif rot_axis == "Z":
        _apply_uniform_rz(runtime, control_indices, target, angles)
    else:
        raise ValueError("SelectPauliRot rot_axis must be X, Y, or Z.")


def _apply_select_pauli_rot_batch(runtime, wire_indices, angles_by_batch, rot_axis):
    if len(wire_indices) < 2:
        raise ValueError("SelectPauliRot requires at least one control wire and one target wire.")
    control_indices = wire_indices[:-1]
    target = wire_indices[-1]
    angle_array = np.asarray(angles_by_batch, dtype=float)
    expected = 1 << len(control_indices)
    if angle_array.ndim != 2 or angle_array.shape[-1] != expected:
        raise ValueError("SelectPauliRot batched angles must match its control wires.")

    if rot_axis == "X":
        runtime.apply_operation("H", [target])
        _apply_uniform_rz_batch(runtime, control_indices, target, angle_array)
        runtime.apply_operation("H", [target])
    elif rot_axis == "Y":
        runtime.apply_operation("SDG", [target])
        runtime.apply_operation("H", [target])
        _apply_uniform_rz_batch(runtime, control_indices, target, angle_array)
        runtime.apply_operation("H", [target])
        runtime.apply_operation("S", [target])
    elif rot_axis == "Z":
        _apply_uniform_rz_batch(runtime, control_indices, target, angle_array)
    else:
        raise ValueError("SelectPauliRot rot_axis must be X, Y, or Z.")


def _permute_permutation(op):
    return tuple(getattr(op, "hyperparameters", {}).get("permutation", ()))


def _apply_basis_embedding(runtime, wire_indices, bits):
    if len(bits) != len(wire_indices):
        raise ValueError("BasisEmbedding length must match the number of target wires.")
    for bit, wire_index in zip(bits, wire_indices):
        if int(bit):
            runtime.apply_operation("X", [wire_index])


def _apply_permute(runtime, wire_indices, op):
    permutation = _permute_permutation(op)
    wire_labels = list(getattr(op, "wires", ()))
    if len(permutation) != len(wire_indices) or len(wire_labels) != len(wire_indices):
        raise ValueError("Permute permutation length must match its wires.")

    working_order = list(wire_labels)
    for current_index, target_label in enumerate(permutation):
        if working_order[current_index] == target_label:
            continue
        swap_index = working_order.index(target_label)
        runtime.apply_operation("SWAP", [wire_indices[current_index], wire_indices[swap_index]])
        working_order[current_index], working_order[swap_index] = (
            working_order[swap_index],
            working_order[current_index],
        )


def _apply_qft(runtime, wire_indices):
    if not wire_indices:
        raise ValueError("QFT requires at least one wire.")

    for target_position, target in enumerate(wire_indices):
        runtime.apply_operation("H", [target])
        for control_position in range(target_position + 1, len(wire_indices)):
            control = wire_indices[control_position]
            angle = np.pi / (2 ** (control_position - target_position))
            _apply_controlled_phase_shift(runtime, [control, target], angle)

    for left_position in range(len(wire_indices) // 2):
        runtime.apply_operation(
            "SWAP",
            [wire_indices[left_position], wire_indices[-left_position - 1]],
        )


def _apply_qubit_sum(runtime, wire_indices):
    if len(wire_indices) != 3:
        raise ValueError("QubitSum requires exactly three wires.")

    first, second, output = wire_indices
    runtime.apply_operation("CNOT", [second, output])
    runtime.apply_operation("CNOT", [first, output])


def _apply_qubit_carry(runtime, wire_indices):
    if len(wire_indices) != 4:
        raise ValueError("QubitCarry requires exactly four wires.")

    first, second, third, output = wire_indices
    runtime.apply_operation("MCX", [second, third, output])
    runtime.apply_operation("CNOT", [second, third])
    runtime.apply_operation("MCX", [first, third, output])


def _apply_grover_operator(runtime, wire_indices):
    if len(wire_indices) < 2:
        raise ValueError("GroverOperator requires at least two wires.")

    target = wire_indices[-1]
    controls = wire_indices[:-1]
    for wire_index in controls:
        runtime.apply_operation("H", [wire_index])
    runtime.apply_operation("Z", [target])
    _apply_mcx_with_control_values(runtime, wire_indices, [False] * len(controls))
    runtime.apply_operation("Z", [target])
    for wire_index in controls:
        runtime.apply_operation("H", [wire_index])
    _apply_global_phase_operation(runtime, wire_indices, np.pi)


def _apply_controlled_phase_variant(runtime, wire_indices, theta, control_state):
    if len(control_state) != len(wire_indices):
        raise ValueError("Controlled phase control_state must match the wire count.")

    flipped_wires = [
        wire_index
        for wire_index, state_bit in zip(wire_indices, control_state)
        if state_bit == "0"
    ]
    for wire_index in flipped_wires:
        runtime.apply_operation("X", [wire_index])
    _apply_controlled_phase_shift(runtime, wire_indices, theta)
    for wire_index in reversed(flipped_wires):
        runtime.apply_operation("X", [wire_index])


def _apply_controlled_phase_variant_batch(runtime, wire_indices, thetas, control_state):
    if len(control_state) != len(wire_indices):
        raise ValueError("Controlled phase control_state must match the wire count.")

    flipped_wires = [
        wire_index
        for wire_index, state_bit in zip(wire_indices, control_state)
        if state_bit == "0"
    ]
    for wire_index in flipped_wires:
        runtime.apply_operation("X", [wire_index])
    _apply_controlled_phase_shift_batch(runtime, wire_indices, thetas)
    for wire_index in reversed(flipped_wires):
        runtime.apply_operation("X", [wire_index])


def _apply_isingxx(runtime, wire_indices, theta):
    if len(wire_indices) != 2:
        raise ValueError("IsingXX requires exactly two wires.")

    control, target = wire_indices
    runtime.apply_operation("CNOT", [control, target])
    runtime.apply_operation("RX", [control], [theta])
    runtime.apply_operation("CNOT", [control, target])


def _apply_isingxx_batch(runtime, wire_indices, thetas):
    if len(wire_indices) != 2:
        raise ValueError("IsingXX requires exactly two wires.")

    control, target = wire_indices
    runtime.apply_operation("CNOT", [control, target])
    runtime.apply_operation_batch("RX", [control], thetas)
    runtime.apply_operation("CNOT", [control, target])


def _apply_isingyy(runtime, wire_indices, theta):
    if len(wire_indices) != 2:
        raise ValueError("IsingYY requires exactly two wires.")

    for wire_index in wire_indices:
        runtime.apply_operation("RX", [wire_index], [np.pi / 2])
    _apply_multirz(runtime, wire_indices, theta)
    for wire_index in wire_indices:
        runtime.apply_operation("RX", [wire_index], [-np.pi / 2])


def _apply_isingyy_batch(runtime, wire_indices, thetas):
    if len(wire_indices) != 2:
        raise ValueError("IsingYY requires exactly two wires.")

    for wire_index in wire_indices:
        runtime.apply_operation("RX", [wire_index], [np.pi / 2])
    _apply_multirz_batch(runtime, wire_indices, thetas)
    for wire_index in wire_indices:
        runtime.apply_operation("RX", [wire_index], [-np.pi / 2])


def _apply_isingxy(runtime, wire_indices, theta):
    if len(wire_indices) != 2:
        raise ValueError("IsingXY requires exactly two wires.")

    left, right = wire_indices
    runtime.apply_operation("H", [left])
    _apply_cy(runtime, wire_indices)
    runtime.apply_operation("RY", [left], [theta / 2])
    runtime.apply_operation("RX", [right], [-theta / 2])
    _apply_cy(runtime, wire_indices)
    runtime.apply_operation("H", [left])


def _apply_isingxy_batch(runtime, wire_indices, thetas):
    if len(wire_indices) != 2:
        raise ValueError("IsingXY requires exactly two wires.")

    left, right = wire_indices
    half_thetas = [theta / 2 for theta in thetas]
    runtime.apply_operation("H", [left])
    _apply_cy(runtime, wire_indices)
    runtime.apply_operation_batch("RY", [left], half_thetas)
    runtime.apply_operation_batch("RX", [right], [-theta for theta in half_thetas])
    _apply_cy(runtime, wire_indices)
    runtime.apply_operation("H", [left])


def _apply_cy(runtime, wire_indices):
    if len(wire_indices) != 2:
        raise ValueError("CY requires exactly two wires.")

    control, target = wire_indices
    runtime.apply_operation("SDG", [target])
    runtime.apply_operation("CNOT", [control, target])
    runtime.apply_operation("S", [target])


def _apply_ccz(runtime, wire_indices):
    if len(wire_indices) != 3:
        raise ValueError("CCZ requires exactly three wires.")

    control_a, control_b, target = wire_indices
    runtime.apply_operation("H", [target])
    runtime.apply_operation("MCX", [control_a, control_b, target])
    runtime.apply_operation("H", [target])


def _apply_ch(runtime, wire_indices):
    if len(wire_indices) != 2:
        raise ValueError("CH requires exactly two wires.")

    control, target = wire_indices
    runtime.apply_operation("RY", [target], [np.pi / 4])
    runtime.apply_operation("CNOT", [control, target])
    runtime.apply_operation("RY", [target], [-np.pi / 4])


def _apply_iswap(runtime, wire_indices):
    if len(wire_indices) != 2:
        raise ValueError("ISWAP requires exactly two wires.")

    left, right = wire_indices
    runtime.apply_operation("S", [left])
    runtime.apply_operation("S", [right])
    runtime.apply_operation("H", [left])
    runtime.apply_operation("CNOT", [left, right])
    runtime.apply_operation("CNOT", [right, left])
    runtime.apply_operation("H", [right])


def _apply_pswap(runtime, wire_indices, phi):
    if len(wire_indices) != 2:
        raise ValueError("PSWAP requires exactly two wires.")

    left, right = wire_indices
    runtime.apply_operation("SWAP", [left, right])
    runtime.apply_operation("CNOT", [left, right])
    _apply_phase_shift(runtime, [right], phi)
    runtime.apply_operation("CNOT", [left, right])


def _apply_pswap_batch(runtime, wire_indices, phis):
    if len(wire_indices) != 2:
        raise ValueError("PSWAP requires exactly two wires.")

    left, right = wire_indices
    runtime.apply_operation("SWAP", [left, right])
    runtime.apply_operation("CNOT", [left, right])
    _apply_phase_shift_batch(runtime, [right], phis)
    runtime.apply_operation("CNOT", [left, right])


def _apply_siswap(runtime, wire_indices):
    if len(wire_indices) != 2:
        raise ValueError("SISWAP requires exactly two wires.")

    left, right = wire_indices
    _apply_sx(runtime, [left])
    runtime.apply_operation("RZ", [left], [np.pi / 2])
    runtime.apply_operation("CNOT", [left, right])
    _apply_sx(runtime, [left])
    runtime.apply_operation("RZ", [left], [7 * np.pi / 4])
    _apply_sx(runtime, [left])
    runtime.apply_operation("RZ", [left], [np.pi / 2])
    _apply_sx(runtime, [right])
    runtime.apply_operation("RZ", [right], [7 * np.pi / 4])
    runtime.apply_operation("CNOT", [left, right])
    _apply_sx(runtime, [left])
    _apply_sx(runtime, [right])


def _apply_ecr(runtime, wire_indices):
    if len(wire_indices) != 2:
        raise ValueError("ECR requires exactly two wires.")

    control, target = wire_indices
    runtime.apply_operation("Z", [control])
    runtime.apply_operation("CNOT", [control, target])
    _apply_sx(runtime, [target])
    runtime.apply_operation("RX", [control], [np.pi / 2])
    runtime.apply_operation("RY", [control], [np.pi / 2])
    runtime.apply_operation("RX", [control], [np.pi / 2])


def _apply_single_excitation(runtime, wire_indices, theta):
    if len(wire_indices) != 2:
        raise ValueError("SingleExcitation requires exactly two wires.")

    left, right = wire_indices
    runtime.apply_operation("H", [left])
    runtime.apply_operation("CNOT", [left, right])
    runtime.apply_operation("RY", [left], [-theta / 2])
    runtime.apply_operation("RY", [right], [-theta / 2])
    runtime.apply_operation("CNOT", [left, right])
    runtime.apply_operation("H", [left])


def _apply_single_excitation_batch(runtime, wire_indices, thetas):
    if len(wire_indices) != 2:
        raise ValueError("SingleExcitation requires exactly two wires.")

    left, right = wire_indices
    half_thetas = [-theta / 2 for theta in thetas]
    runtime.apply_operation("H", [left])
    runtime.apply_operation("CNOT", [left, right])
    runtime.apply_operation_batch("RY", [left], half_thetas)
    runtime.apply_operation_batch("RY", [right], half_thetas)
    runtime.apply_operation("CNOT", [left, right])
    runtime.apply_operation("H", [left])


def _apply_single_excitation_phase_variant(runtime, wire_indices, theta, sign):
    if len(wire_indices) != 2:
        raise ValueError("SingleExcitation phase variants require exactly two wires.")

    left, right = wire_indices
    half_theta = theta / 2
    runtime.apply_operation("H", [right])
    runtime.apply_operation("CNOT", [right, left])
    runtime.apply_operation("RY", [left], [half_theta])
    runtime.apply_operation("RY", [right], [half_theta])
    _apply_cy(runtime, [right, left])
    runtime.apply_operation("S", [right])
    runtime.apply_operation("H", [right])
    runtime.apply_operation("RZ", [right], [-sign * half_theta])
    runtime.apply_operation("CNOT", [left, right])
    _apply_global_phase(runtime, left, sign * theta / 4)


def _apply_single_excitation_phase_variant_batch(runtime, wire_indices, thetas, sign):
    if len(wire_indices) != 2:
        raise ValueError("SingleExcitation phase variants require exactly two wires.")

    left, right = wire_indices
    half_thetas = [theta / 2 for theta in thetas]
    runtime.apply_operation("H", [right])
    runtime.apply_operation("CNOT", [right, left])
    runtime.apply_operation_batch("RY", [left], half_thetas)
    runtime.apply_operation_batch("RY", [right], half_thetas)
    _apply_cy(runtime, [right, left])
    runtime.apply_operation("S", [right])
    runtime.apply_operation("H", [right])
    runtime.apply_operation_batch("RZ", [right], [-sign * theta for theta in half_thetas])
    runtime.apply_operation("CNOT", [left, right])


def _apply_crot(runtime, wire_indices, phi, theta, omega):
    if len(wire_indices) != 2:
        raise ValueError("CRot requires exactly two wires.")

    control, target = wire_indices
    runtime.apply_operation("RZ", [target], [(phi - omega) / 2])
    runtime.apply_operation("CNOT", [control, target])
    runtime.apply_operation("RZ", [target], [-(phi + omega) / 2])
    runtime.apply_operation("RY", [target], [-theta / 2])
    runtime.apply_operation("CNOT", [control, target])
    runtime.apply_operation("RY", [target], [theta / 2])
    runtime.apply_operation("RZ", [target], [omega])


def _apply_rot_batch(runtime, wire_indices, params_by_op):
    if len(wire_indices) != 1:
        raise ValueError("Rot requires exactly one wire.")

    target = wire_indices[0]
    runtime.apply_operation_batch("RZ", [target], [params[0] for params in params_by_op])
    runtime.apply_operation_batch("RY", [target], [params[1] for params in params_by_op])
    runtime.apply_operation_batch("RZ", [target], [params[2] for params in params_by_op])


def _apply_crot_batch(runtime, wire_indices, params_by_op):
    if len(wire_indices) != 2:
        raise ValueError("CRot requires exactly two wires.")

    control, target = wire_indices
    phis = [params[0] for params in params_by_op]
    thetas = [params[1] for params in params_by_op]
    omegas = [params[2] for params in params_by_op]
    runtime.apply_operation_batch("RZ", [target], [(phi - omega) / 2 for phi, omega in zip(phis, omegas)])
    runtime.apply_operation("CNOT", [control, target])
    runtime.apply_operation_batch("RZ", [target], [-(phi + omega) / 2 for phi, omega in zip(phis, omegas)])
    runtime.apply_operation_batch("RY", [target], [-theta / 2 for theta in thetas])
    runtime.apply_operation("CNOT", [control, target])
    runtime.apply_operation_batch("RY", [target], [theta / 2 for theta in thetas])
    runtime.apply_operation_batch("RZ", [target], omegas)


def _apply_double_excitation(runtime, wire_indices, theta):
    if len(wire_indices) != 4:
        raise ValueError("DoubleExcitation requires exactly four wires.")

    first, second, third, fourth = wire_indices
    angle = theta / 8
    runtime.apply_operation("CNOT", [third, fourth])
    runtime.apply_operation("CNOT", [first, third])
    runtime.apply_operation("H", [fourth])
    runtime.apply_operation("H", [first])
    runtime.apply_operation("CNOT", [third, fourth])
    runtime.apply_operation("CNOT", [first, second])
    runtime.apply_operation("RY", [second], [angle])
    runtime.apply_operation("RY", [first], [-angle])
    runtime.apply_operation("CNOT", [first, fourth])
    runtime.apply_operation("H", [fourth])
    runtime.apply_operation("CNOT", [fourth, second])
    runtime.apply_operation("RY", [second], [angle])
    runtime.apply_operation("RY", [first], [-angle])
    runtime.apply_operation("CNOT", [third, second])
    runtime.apply_operation("CNOT", [third, first])
    runtime.apply_operation("RY", [second], [-angle])
    runtime.apply_operation("RY", [first], [angle])
    runtime.apply_operation("CNOT", [fourth, second])
    runtime.apply_operation("H", [fourth])
    runtime.apply_operation("CNOT", [first, fourth])
    runtime.apply_operation("RY", [second], [-angle])
    runtime.apply_operation("RY", [first], [angle])
    runtime.apply_operation("CNOT", [first, second])
    runtime.apply_operation("CNOT", [third, first])
    runtime.apply_operation("H", [first])
    runtime.apply_operation("H", [fourth])
    runtime.apply_operation("CNOT", [first, third])
    runtime.apply_operation("CNOT", [third, fourth])


def _apply_double_excitation_batch(runtime, wire_indices, thetas):
    if len(wire_indices) != 4:
        raise ValueError("DoubleExcitation requires exactly four wires.")

    first, second, third, fourth = wire_indices
    angles = [theta / 8 for theta in thetas]
    negative_angles = [-angle for angle in angles]
    runtime.apply_operation("CNOT", [third, fourth])
    runtime.apply_operation("CNOT", [first, third])
    runtime.apply_operation("H", [fourth])
    runtime.apply_operation("H", [first])
    runtime.apply_operation("CNOT", [third, fourth])
    runtime.apply_operation("CNOT", [first, second])
    runtime.apply_operation_batch("RY", [second], angles)
    runtime.apply_operation_batch("RY", [first], negative_angles)
    runtime.apply_operation("CNOT", [first, fourth])
    runtime.apply_operation("H", [fourth])
    runtime.apply_operation("CNOT", [fourth, second])
    runtime.apply_operation_batch("RY", [second], angles)
    runtime.apply_operation_batch("RY", [first], negative_angles)
    runtime.apply_operation("CNOT", [third, second])
    runtime.apply_operation("CNOT", [third, first])
    runtime.apply_operation_batch("RY", [second], negative_angles)
    runtime.apply_operation_batch("RY", [first], angles)
    runtime.apply_operation("CNOT", [fourth, second])
    runtime.apply_operation("H", [fourth])
    runtime.apply_operation("CNOT", [first, fourth])
    runtime.apply_operation_batch("RY", [second], negative_angles)
    runtime.apply_operation_batch("RY", [first], angles)
    runtime.apply_operation("CNOT", [first, second])
    runtime.apply_operation("CNOT", [third, first])
    runtime.apply_operation("H", [first])
    runtime.apply_operation("H", [fourth])
    runtime.apply_operation("CNOT", [first, third])
    runtime.apply_operation("CNOT", [third, fourth])


def _apply_double_excitation_phase_variant(runtime, wire_indices, theta, sign):
    if len(wire_indices) != 4:
        raise ValueError("DoubleExcitation phase variants require exactly four wires.")

    alpha = sign * theta / 2
    _apply_global_phase(runtime, wire_indices[0], 7 * alpha / 8)
    for relative_wires, coefficient in (
        ((0, 1), 1),
        ((0, 2), -1),
        ((0, 3), -1),
        ((1, 2), -1),
        ((1, 3), -1),
        ((2, 3), 1),
        ((0, 1, 2, 3), 1),
    ):
        _apply_multirz(
            runtime,
            [wire_indices[index] for index in relative_wires],
            coefficient * alpha / 4,
        )
    _apply_double_excitation(runtime, wire_indices, theta)


def _apply_double_excitation_phase_variant_batch(runtime, wire_indices, thetas, sign):
    if len(wire_indices) != 4:
        raise ValueError("DoubleExcitation phase variants require exactly four wires.")

    alphas = [sign * theta / 2 for theta in thetas]
    for relative_wires, coefficient in (
        ((0, 1), 1),
        ((0, 2), -1),
        ((0, 3), -1),
        ((1, 2), -1),
        ((1, 3), -1),
        ((2, 3), 1),
        ((0, 1, 2, 3), 1),
    ):
        _apply_multirz_batch(
            runtime,
            [wire_indices[index] for index in relative_wires],
            [coefficient * alpha / 4 for alpha in alphas],
        )
    _apply_double_excitation_batch(runtime, wire_indices, thetas)


def _apply_fermionic_swap(runtime, wire_indices, theta):
    if len(wire_indices) != 2:
        raise ValueError("FermionicSWAP requires exactly two wires.")

    left, right = wire_indices
    half_theta = theta / 2
    runtime.apply_operation("H", [left])
    runtime.apply_operation("H", [right])
    _apply_multirz(runtime, wire_indices, half_theta)
    runtime.apply_operation("H", [left])
    runtime.apply_operation("H", [right])
    runtime.apply_operation("RX", [left], [np.pi / 2])
    runtime.apply_operation("RX", [right], [np.pi / 2])
    _apply_multirz(runtime, wire_indices, half_theta)
    runtime.apply_operation("RX", [left], [-np.pi / 2])
    runtime.apply_operation("RX", [right], [-np.pi / 2])
    runtime.apply_operation("RZ", [left], [half_theta])
    runtime.apply_operation("RZ", [right], [half_theta])
    _apply_global_phase(runtime, left, half_theta)


def _apply_fermionic_swap_batch(runtime, wire_indices, thetas):
    if len(wire_indices) != 2:
        raise ValueError("FermionicSWAP requires exactly two wires.")

    left, right = wire_indices
    half_thetas = [theta / 2 for theta in thetas]
    runtime.apply_operation("H", [left])
    runtime.apply_operation("H", [right])
    _apply_multirz_batch(runtime, wire_indices, half_thetas)
    runtime.apply_operation("H", [left])
    runtime.apply_operation("H", [right])
    runtime.apply_operation("RX", [left], [np.pi / 2])
    runtime.apply_operation("RX", [right], [np.pi / 2])
    _apply_multirz_batch(runtime, wire_indices, half_thetas)
    runtime.apply_operation("RX", [left], [-np.pi / 2])
    runtime.apply_operation("RX", [right], [-np.pi / 2])
    runtime.apply_operation_batch("RZ", [left], half_thetas)
    runtime.apply_operation_batch("RZ", [right], half_thetas)


def _apply_orbital_rotation(runtime, wire_indices, theta):
    if len(wire_indices) != 4:
        raise ValueError("OrbitalRotation requires exactly four wires.")

    first, second, third, fourth = wire_indices
    _apply_fermionic_swap(runtime, [second, third], np.pi)
    _apply_single_excitation(runtime, [first, second], theta)
    _apply_single_excitation(runtime, [third, fourth], theta)
    _apply_fermionic_swap(runtime, [second, third], np.pi)


def _apply_orbital_rotation_batch(runtime, wire_indices, thetas):
    if len(wire_indices) != 4:
        raise ValueError("OrbitalRotation requires exactly four wires.")

    first, second, third, fourth = wire_indices
    _apply_fermionic_swap(runtime, [second, third], np.pi)
    _apply_single_excitation_batch(runtime, [first, second], thetas)
    _apply_single_excitation_batch(runtime, [third, fourth], thetas)
    _apply_fermionic_swap(runtime, [second, third], np.pi)


def _apply_static_batch_operation(runtime, gate_name, wire_indices, params, op=None, wire_map=None):
    if gate_name == "QubitUnitary" and len(params) == 1:
        matrix = matrix_to_little_endian_wires(params[0])
        runtime.apply_operation("QubitUnitary", wire_indices, matrix=matrix)
        return True

    if gate_name == "ControlledQubitUnitary" and op is not None and wire_map is not None:
        payload = _controlled_qubit_unitary_components(op, wire_map)
        if payload is None:
            return False
        _apply_controlled_qubit_unitary(runtime, *payload)
        return True

    if gate_name in PENNYLANE_TO_ROCQ_GATES and not params:
        runtime.apply_operation(PENNYLANE_TO_ROCQ_GATES[gate_name], wire_indices)
        return True

    if gate_name == "QFT" and not params:
        _apply_qft(runtime, wire_indices)
        return True

    if gate_name == "GlobalPhase" and len(params) == 1:
        _apply_global_phase_operation(runtime, wire_indices, params[0])
        return True

    if gate_name == "DiagonalQubitUnitary" and len(params) == 1:
        _apply_diagonal_qubit_unitary(runtime, wire_indices, params[0])
        return True

    if gate_name == "SelectPauliRot" and len(params) == 1 and op is not None:
        _apply_select_pauli_rot(runtime, wire_indices, params[0], _select_pauli_rot_axis(op))
        return True

    if gate_name == "BasisEmbedding" and len(params) == 1:
        bits = np.asarray(params[0], dtype=int).reshape(-1)
        _apply_basis_embedding(runtime, wire_indices, bits)
        return True

    if gate_name == "Permute" and not params and op is not None:
        _apply_permute(runtime, wire_indices, op)
        return True

    if gate_name == "QubitSum" and not params:
        _apply_qubit_sum(runtime, wire_indices)
        return True

    if gate_name == "QubitCarry" and not params:
        _apply_qubit_carry(runtime, wire_indices)
        return True

    if gate_name == "GroverOperator" and not params:
        _apply_grover_operator(runtime, wire_indices)
        return True

    if gate_name == "Toffoli" and not params:
        runtime.apply_operation("MCX", wire_indices)
        return True

    if gate_name == "CSWAP" and not params:
        runtime.apply_operation("CSWAP", wire_indices)
        return True

    if gate_name == "CY" and not params:
        if not _supports_native_gate_decomposition(runtime):
            raise NotImplementedError
        _apply_cy(runtime, wire_indices)
        return True

    if gate_name == "CH" and not params:
        if not _supports_native_gate_decomposition(runtime):
            raise NotImplementedError
        _apply_ch(runtime, wire_indices)
        return True

    if gate_name == "CCZ" and not params:
        if not _supports_native_gate_decomposition(runtime):
            raise NotImplementedError
        _apply_ccz(runtime, wire_indices)
        return True

    if gate_name == "ISWAP" and not params:
        if not _supports_native_gate_decomposition(runtime):
            raise NotImplementedError
        _apply_iswap(runtime, wire_indices)
        return True

    if gate_name in {"SISWAP", "SQISW"} and not params:
        if not _supports_native_phase_decomposition(runtime):
            raise NotImplementedError
        _apply_siswap(runtime, wire_indices)
        return True

    if gate_name == "ECR" and not params:
        if not _supports_native_phase_decomposition(runtime):
            raise NotImplementedError
        _apply_ecr(runtime, wire_indices)
        return True

    if gate_name == "PSWAP" and len(params) == 1:
        if not _supports_native_phase_decomposition(runtime):
            raise NotImplementedError
        _apply_pswap(runtime, wire_indices, params[0])
        return True

    if gate_name == "IsingXY" and len(params) == 1:
        if not _supports_native_parametric_decomposition(runtime):
            raise NotImplementedError
        _apply_isingxy(runtime, wire_indices, params[0])
        return True

    if gate_name == "SingleExcitation" and len(params) == 1:
        if not _supports_native_parametric_decomposition(runtime):
            raise NotImplementedError
        _apply_single_excitation(runtime, wire_indices, params[0])
        return True

    if gate_name in {"SingleExcitationPlus", "SingleExcitationMinus"} and len(params) == 1:
        if not _supports_native_phase_decomposition(runtime):
            raise NotImplementedError
        sign = 1 if gate_name == "SingleExcitationPlus" else -1
        _apply_single_excitation_phase_variant(runtime, wire_indices, params[0], sign)
        return True

    if gate_name == "DoubleExcitation" and len(params) == 1:
        if not _supports_native_parametric_decomposition(runtime):
            raise NotImplementedError
        _apply_double_excitation(runtime, wire_indices, params[0])
        return True

    if gate_name in {"DoubleExcitationPlus", "DoubleExcitationMinus"} and len(params) == 1:
        if not _supports_native_phase_decomposition(runtime):
            raise NotImplementedError
        sign = 1 if gate_name == "DoubleExcitationPlus" else -1
        _apply_double_excitation_phase_variant(runtime, wire_indices, params[0], sign)
        return True

    if gate_name == "FermionicSWAP" and len(params) == 1:
        if not _supports_native_phase_decomposition(runtime):
            raise NotImplementedError
        _apply_fermionic_swap(runtime, wire_indices, params[0])
        return True

    if gate_name == "OrbitalRotation" and len(params) == 1:
        if not _supports_native_phase_decomposition(runtime):
            raise NotImplementedError
        _apply_orbital_rotation(runtime, wire_indices, params[0])
        return True

    if gate_name in {"CPhaseShift00", "CPhaseShift01", "CPhaseShift10"} and len(params) == 1:
        if not _supports_native_phase_decomposition(runtime):
            raise NotImplementedError
        control_state = gate_name[len("CPhaseShift"):]
        _apply_controlled_phase_variant(runtime, wire_indices, params[0], control_state)
        return True

    return False


class RocQDevice(QubitDevice):
    name = "rocQuantum Simulator Device"
    short_name = "rocquantum.qpu"
    author = "rocQuantum contributors"
    version = "0.1.0"
    pennylane_requires = ">=0.30"

    operations = (
        set(PENNYLANE_TO_ROCQ_GATES.keys())
        | NATIVE_PARAMETRIC_OPS
        | MATRIX_OPS
        | DECOMPOSED_OPS
        | {"BasisState", "StatePrep", "Rot"}
    )
    observables = {
        "PauliX", "PauliY", "PauliZ", "Hadamard", "Identity",
        "Hermitian", "Projector", "SparseHamiltonian",
        "Counts", "State",
        "Prod", "Tensor", "SProd", "Sum", "LinearCombination", "Hamiltonian",
    }

    @classmethod
    def capabilities(cls):
        capabilities = dict(super().capabilities())
        capabilities.update(
            {
                "returns_state": True,
                "returns_probs": True,
                "supports_finite_shots": True,
            }
        )
        return capabilities

    def __init__(self, wires, shots=None, **kwargs):
        super().__init__(wires=wires, shots=shots)
        self.sim = None
        self._state = None
        self._skip_diagonalizing_rotations = False
        self._diagonalizing_rotations_applied = False
        self._capture_pre_rotated_state = False
        self._pre_rotated_state = None
        self._preserve_global_phase = True
        self.reset()

    def reset(self, batch_size=1):
        try:
            self._runtime = RocQuantumRuntime.from_bindings(
                len(self.wires),
                binding_module=rocquantum_bind,
                batch_size=int(batch_size),
            )
        except ImportError as exc:
            raise ImportError(
                "The 'rocquantum_bind' module is not installed. "
                "Build and install rocQuantum with ROCQUANTUM_BUILD_BINDINGS=ON before creating "
                "a PennyLane rocQuantum device."
            ) from exc
        self.sim = self._runtime.simulator
        self._runtime.preserve_global_phase = self._preserve_global_phase
        self._state = None
        self._pre_rotated_state = None
        self._diagonalizing_rotations_applied = False

    def _circuit_preserves_global_phase(self, circuit):
        measurements = getattr(circuit, "measurements", ())
        if not measurements:
            return True

        return any(
            measurement.__class__.__name__ in {"StateMP", "AmplitudeMP"}
            for measurement in measurements
        )

    def _analytic_measurements_use_native_pauli(self, circuit):
        if self.shots is not None:
            return False

        measurements = getattr(circuit, "measurements", ())
        if not measurements:
            return False

        for measurement in measurements:
            if measurement.__class__.__name__ not in {"ExpectationMP", "VarianceMP"}:
                return False
            observable = getattr(measurement, "obs", None)
            if observable is None:
                return False
            if observable.name == "Hermitian":
                continue
            if _pauli_terms_from_observable(observable, self.wire_map) is None:
                return False
        return True

    def _get_diagonalizing_gates(self, circuit):
        if self._skip_diagonalizing_rotations:
            return []
        return super()._get_diagonalizing_gates(circuit)

    def execute_and_gradients(self, circuits, method="jacobian", **kwargs):
        if method != "adjoint_jacobian":
            return super().execute_and_gradients(circuits, method=method, **kwargs)

        previous = self._capture_pre_rotated_state
        self._capture_pre_rotated_state = True
        try:
            return super().execute_and_gradients(circuits, method=method, **kwargs)
        finally:
            self._capture_pre_rotated_state = previous

    def execute(self, circuit, **kwargs):
        skip_rotations = self._analytic_measurements_use_native_pauli(circuit)
        preserve_global_phase = self._circuit_preserves_global_phase(circuit)
        previous = self._skip_diagonalizing_rotations
        previous_global_phase = self._preserve_global_phase
        self._skip_diagonalizing_rotations = skip_rotations
        self._preserve_global_phase = preserve_global_phase
        try:
            return super().execute(circuit, **kwargs)
        finally:
            self._skip_diagonalizing_rotations = previous
            self._preserve_global_phase = previous_global_phase
            if getattr(self, "_runtime", None) is not None:
                self._runtime.preserve_global_phase = previous_global_phase

    def batch_execute(self, circuits, **kwargs):
        circuits = list(circuits)
        try:
            result = self._try_execute_batched_parameter_circuits(circuits)
            if result is not None:
                return result
        except (NotImplementedError, RuntimeError, TypeError, ValueError):
            self.reset()
        return super().batch_execute(circuits, **kwargs)

    def _try_execute_batched_parameter_circuits(self, circuits):
        if len(circuits) <= 1:
            return None

        reference_ops = list(getattr(circuits[0], "operations", ()))
        reference_measurements = list(getattr(circuits[0], "measurements", ()))
        if not reference_measurements:
            return None

        measurement_names = [measurement.__class__.__name__ for measurement in reference_measurements]
        finite_shots = self.shots is not None
        if finite_shots:
            if (
                len(reference_measurements) != 1
                or _has_partitioned_shots(self.shots)
                or measurement_names[0] not in {"SampleMP", "CountsMP"}
            ):
                return None
            if measurement_names[0] == "CountsMP" and getattr(reference_measurements[0], "all_outcomes", False):
                return None
        elif all(name in {"ExpectationMP", "VarianceMP", "ProbabilityMP"} for name in measurement_names):
            pass
        else:
            return None

        reference_measurement_specs = []
        for measurement_name, measurement in zip(measurement_names, reference_measurements):
            if measurement_name in {"ExpectationMP", "VarianceMP"}:
                observable = getattr(measurement, "obs", None)
                payload = _observable_batch_payload(observable, self.wire_map, wire_order=self.wires)
                if payload is None:
                    return None
                reference_measurement_specs.append((measurement_name, payload))
            elif measurement_name in {"ProbabilityMP", "SampleMP", "CountsMP"}:
                reference_measurement_specs.append((measurement_name, tuple(measurement.wires)))
            else:
                reference_measurement_specs.append((measurement_name, None))

        for circuit in circuits:
            if len(getattr(circuit, "operations", ())) != len(reference_ops):
                return None
            measurements = list(getattr(circuit, "measurements", ()))
            if len(measurements) != len(reference_measurements):
                return None
            current_names = [measurement.__class__.__name__ for measurement in measurements]
            if current_names != measurement_names:
                return None
            for measurement, (measurement_name, reference_payload) in zip(measurements, reference_measurement_specs):
                if measurement_name in {"ExpectationMP", "VarianceMP"}:
                    if not _observable_batch_payload_matches(
                        getattr(measurement, "obs", None),
                        self.wire_map,
                        reference_payload,
                        wire_order=self.wires,
                    ):
                        return None
                elif measurement_name in {"ProbabilityMP", "SampleMP", "CountsMP"} and (
                    tuple(measurement.wires) != reference_payload
                ):
                    return None
            if measurement_names[0] == "CountsMP" and getattr(measurements[0], "all_outcomes", False):
                return None

        self.reset(batch_size=len(circuits))
        self._runtime.preserve_global_phase = False
        for op_index, reference_op in enumerate(reference_ops):
            gate_name = reference_op.name
            wire_indices = [self.wire_map[w] for w in reference_op.wires]
            ops = [list(getattr(circuit, "operations", ()))[op_index] for circuit in circuits]
            if any(op.name != gate_name or [self.wire_map[w] for w in op.wires] != wire_indices for op in ops):
                return None

            if gate_name == "PauliRot" and any(
                _paulirot_pauli_word(op) != _paulirot_pauli_word(reference_op) for op in ops[1:]
            ):
                return None

            if gate_name == "SelectPauliRot" and any(
                _select_pauli_rot_axis(op) != _select_pauli_rot_axis(reference_op) for op in ops[1:]
            ):
                return None

            if gate_name == "Permute" and any(
                _permute_permutation(op) != _permute_permutation(reference_op) for op in ops[1:]
            ):
                return None

            if gate_name == "BasisState":
                if op_index != 0:
                    return None
                try:
                    bits_by_op = [tuple(_basis_state_bits(op, len(wire_indices))) for op in ops]
                except (TypeError, ValueError):
                    return None
                if any(bits != bits_by_op[0] for bits in bits_by_op[1:]):
                    return None
                for bit, wire_index in zip(bits_by_op[0], wire_indices):
                    if int(bit):
                        self._runtime.apply_operation("X", [wire_index])
                continue

            if gate_name == "StatePrep":
                if op_index != 0 or len(wire_indices) != self.num_wires:
                    return None
                statevectors = []
                for op in ops:
                    params = list(getattr(op, "parameters", []))
                    if len(params) != 1:
                        return None
                    statevectors.append(statevector_to_little_endian_wires(params[0]))
                self._runtime.set_statevectors(statevectors)
                continue

            params_by_op = [list(getattr(op, "parameters", [])) for op in ops]
            if gate_name in {"RX", "RY", "RZ", "CRX", "CRY", "CRZ"} and all(
                len(params) == 1 for params in params_by_op
            ):
                self._runtime.apply_operation_batch(gate_name, wire_indices, [params[0] for params in params_by_op])
                continue

            if gate_name == "GlobalPhase" and all(len(params) == 1 for params in params_by_op):
                continue

            if gate_name == "BasisEmbedding" and all(len(params) == 1 for params in params_by_op):
                try:
                    bits_by_op = [
                        tuple(np.asarray(params[0], dtype=int).reshape(-1))
                        for params in params_by_op
                    ]
                except (TypeError, ValueError):
                    return None
                if any(bits != bits_by_op[0] for bits in bits_by_op[1:]):
                    return None
                _apply_basis_embedding(self._runtime, wire_indices, bits_by_op[0])
                continue

            if gate_name == "DiagonalQubitUnitary" and all(len(params) == 1 for params in params_by_op):
                diagonals = [params[0] for params in params_by_op]
                _apply_diagonal_qubit_unitary_batch(self._runtime, wire_indices, diagonals)
                continue

            if gate_name == "SelectPauliRot" and all(len(params) == 1 for params in params_by_op):
                angles_by_batch = [params[0] for params in params_by_op]
                _apply_select_pauli_rot_batch(
                    self._runtime,
                    wire_indices,
                    angles_by_batch,
                    _select_pauli_rot_axis(reference_op),
                )
                continue

            if gate_name in {
                "PhaseShift", "ControlledPhaseShift", "MultiRZ", "IsingXX", "IsingYY", "IsingZZ",
                "IsingXY", "PauliRot", "PSWAP", "SingleExcitation", "SingleExcitationPlus", "SingleExcitationMinus",
                "DoubleExcitation", "DoubleExcitationPlus", "DoubleExcitationMinus", "FermionicSWAP",
                "OrbitalRotation", "CPhaseShift00", "CPhaseShift01", "CPhaseShift10",
            } and all(
                len(params) == 1 for params in params_by_op
            ):
                thetas = [params[0] for params in params_by_op]
                if gate_name == "PhaseShift":
                    _apply_phase_shift_batch(self._runtime, wire_indices, thetas)
                elif gate_name == "ControlledPhaseShift":
                    _apply_controlled_phase_shift_batch(self._runtime, wire_indices, thetas)
                elif gate_name == "MultiRZ":
                    _apply_multirz_batch(self._runtime, wire_indices, thetas)
                elif gate_name == "IsingXX":
                    _apply_isingxx_batch(self._runtime, wire_indices, thetas)
                elif gate_name == "IsingYY":
                    _apply_isingyy_batch(self._runtime, wire_indices, thetas)
                elif gate_name == "IsingZZ":
                    _apply_multirz_batch(self._runtime, wire_indices, thetas)
                elif gate_name == "IsingXY":
                    _apply_isingxy_batch(self._runtime, wire_indices, thetas)
                elif gate_name == "PauliRot":
                    _apply_paulirot_batch(self._runtime, wire_indices, thetas, _paulirot_pauli_word(reference_op))
                elif gate_name == "PSWAP":
                    _apply_pswap_batch(self._runtime, wire_indices, thetas)
                elif gate_name == "SingleExcitation":
                    _apply_single_excitation_batch(self._runtime, wire_indices, thetas)
                elif gate_name in {"SingleExcitationPlus", "SingleExcitationMinus"}:
                    sign = 1 if gate_name == "SingleExcitationPlus" else -1
                    _apply_single_excitation_phase_variant_batch(self._runtime, wire_indices, thetas, sign)
                elif gate_name == "DoubleExcitation":
                    _apply_double_excitation_batch(self._runtime, wire_indices, thetas)
                elif gate_name in {"DoubleExcitationPlus", "DoubleExcitationMinus"}:
                    sign = 1 if gate_name == "DoubleExcitationPlus" else -1
                    _apply_double_excitation_phase_variant_batch(self._runtime, wire_indices, thetas, sign)
                elif gate_name == "FermionicSWAP":
                    _apply_fermionic_swap_batch(self._runtime, wire_indices, thetas)
                elif gate_name.startswith("CPhaseShift"):
                    _apply_controlled_phase_variant_batch(
                        self._runtime,
                        wire_indices,
                        thetas,
                        gate_name[len("CPhaseShift"):],
                    )
                else:
                    _apply_orbital_rotation_batch(self._runtime, wire_indices, thetas)
                continue

            if gate_name == "Rot" and all(len(params) == 3 for params in params_by_op):
                _apply_rot_batch(self._runtime, wire_indices, params_by_op)
                continue

            if gate_name == "CRot" and all(len(params) == 3 for params in params_by_op):
                _apply_crot_batch(self._runtime, wire_indices, params_by_op)
                continue

            if any(not _parameter_lists_match(params, params_by_op[0]) for params in params_by_op[1:]):
                return None
            try:
                if _apply_static_batch_operation(
                    self._runtime,
                    gate_name,
                    wire_indices,
                    params_by_op[0],
                    op=reference_op,
                    wire_map=self.wire_map,
                ):
                    continue
            except (NotImplementedError, RuntimeError, TypeError, ValueError):
                return None
            if gate_name in PENNYLANE_TO_ROCQ_GATES and not params_by_op[0]:
                self._runtime.apply_operation(PENNYLANE_TO_ROCQ_GATES[gate_name], wire_indices)
                continue
            if gate_name in NATIVE_PARAMETRIC_OPS:
                self._runtime.apply_operation(gate_name, wire_indices, params_by_op[0])
                continue
            return None

        if all(name in {"ExpectationMP", "VarianceMP", "ProbabilityMP"} for name in measurement_names):
            batched_values = []
            for measurement_name, payload in reference_measurement_specs:
                if measurement_name == "ProbabilityMP":
                    probability_targets = (
                        None
                        if not payload
                        else [self.wire_map[wire] for wire in payload]
                    )
                    batched_values.append(self._runtime.probabilities_batch(probability_targets))
                    continue

                if measurement_name == "VarianceMP" and payload[0] == "sparse":
                    means, second_moments = _native_sparse_hamiltonian_moments_batch(self._runtime, payload)
                    batched_values.append(second_moments - means * means)
                    continue

                means = _evaluate_observable_batch_payload(self._runtime, payload)
                if measurement_name == "VarianceMP":
                    if payload[0] == "pauli":
                        second_payload = ("pauli", _pauli_square_terms(payload[1]))
                    else:
                        second_payload = ("matrix", payload[1] @ payload[1], payload[2])
                    second_moments = _evaluate_observable_batch_payload(self._runtime, second_payload)
                    batched_values.append(second_moments - means * means)
                else:
                    batched_values.append(means)
            if len(batched_values) == 1:
                if measurement_names[0] == "ProbabilityMP":
                    return tuple(np.asarray(row, dtype=float) for row in batched_values[0])
                return tuple(
                    _real_measurement_result(value, measurement_names[0])
                    for value in batched_values[0]
                )
            return [
                tuple(
                    (
                        np.asarray(values[batch_index], dtype=float)
                        if measurement_name == "ProbabilityMP"
                        else np.asarray(_real_measurement_result(values[batch_index], measurement_name))
                    )
                    for values, measurement_name in zip(batched_values, measurement_names)
                )
                for batch_index in range(len(circuits))
            ]

        if measurement_names[0] in {"SampleMP", "CountsMP"}:
            shots = _shot_count(self.shots)
            _, reference_probability_wires = reference_measurement_specs[0]
            measure_targets = (
                list(range(len(self.wires)))
                if not reference_probability_wires
                else [self.wire_map[wire] for wire in reference_probability_wires]
            )
            raw_samples_batch = self._runtime.measure_batch(measure_targets, shots)
            results = []
            for raw_samples in raw_samples_batch:
                rows = samples_to_binary_rows(raw_samples, len(measure_targets))
                if measurement_names[0] == "SampleMP":
                    results.append(_sample_result_from_rows(rows))
                else:
                    results.append(_counts_result_from_rows(rows))
            return results

        return None

    def apply(self, operations: list[Operation], rotations=None, **kwargs):
        operation_applied = False
        circuit_ops = list(operations)
        rotation_ops = list(rotations or [])
        self._diagonalizing_rotations_applied = bool(rotation_ops)
        self._runtime.preserve_global_phase = self._preserve_global_phase
        for op_index, op in enumerate(circuit_ops + rotation_ops):
            if op_index == len(circuit_ops):
                self._capture_adjoint_reference_state(flush=True)
            gate_name = op.name
            wire_indices = [self.wire_map[w] for w in op.wires]
            if gate_name == "BasisState":
                if operation_applied:
                    raise ValueError("BasisState is only supported as an initial state preparation.")
                bits = _basis_state_bits(op, len(wire_indices))
                for bit, wire_index in zip(bits, wire_indices):
                    if int(bit):
                        self._runtime.apply_operation("X", [wire_index])
                operation_applied = True
            elif gate_name == "StatePrep":
                if operation_applied:
                    raise ValueError("StatePrep is only supported as an initial state preparation.")
                statevector = getattr(op, "parameters", [None])[0]
                if len(wire_indices) == self.num_wires and statevector is not None:
                    try:
                        self._runtime.set_statevector(
                            statevector_to_little_endian_wires(statevector)
                        )
                    except (NotImplementedError, RuntimeError, TypeError, ValueError):
                        matrix = matrix_to_little_endian_wires(qml.matrix(op))
                        self._runtime.apply_operation(
                            "QubitUnitary",
                            wire_indices,
                            matrix=matrix,
                        )
                else:
                    matrix = matrix_to_little_endian_wires(qml.matrix(op))
                    self._runtime.apply_operation(
                        "QubitUnitary",
                        wire_indices,
                        matrix=matrix,
                    )
                operation_applied = True
            elif gate_name in PENNYLANE_TO_ROCQ_GATES:
                self._runtime.apply_operation(PENNYLANE_TO_ROCQ_GATES[gate_name], wire_indices)
                operation_applied = True
            elif gate_name in NATIVE_PARAMETRIC_OPS:
                try:
                    self._runtime.apply_operation(gate_name, wire_indices, getattr(op, "parameters", []))
                except NotImplementedError:
                    matrix = matrix_to_little_endian_wires(qml.matrix(op))
                    self._runtime.apply_operation(
                        "QubitUnitary",
                        wire_indices,
                        matrix=matrix,
                    )
                operation_applied = True
            elif gate_name == "Rot":
                try:
                    phi, theta, omega = getattr(op, "parameters", [])
                    self._runtime.apply_operation("RZ", wire_indices, [phi])
                    self._runtime.apply_operation("RY", wire_indices, [theta])
                    self._runtime.apply_operation("RZ", wire_indices, [omega])
                except NotImplementedError:
                    matrix = matrix_to_little_endian_wires(qml.matrix(op))
                    self._runtime.apply_operation(
                        "QubitUnitary",
                        wire_indices,
                        matrix=matrix,
                    )
                operation_applied = True
            elif gate_name == "Toffoli":
                _apply_native_or_matrix(self._runtime, "MCX", wire_indices, op)
                operation_applied = True
            elif gate_name == "CSWAP":
                _apply_native_or_matrix(self._runtime, "CSWAP", wire_indices, op)
                operation_applied = True
            elif gate_name == "MultiControlledX":
                native_wire_indices = _native_mcx_wire_indices(op, self.wire_map)
                if native_wire_indices is None:
                    matrix = matrix_to_little_endian_wires(qml.matrix(op))
                    self._runtime.apply_operation(
                        gate_name,
                        wire_indices,
                        matrix=matrix,
                    )
                else:
                    try:
                        if not _supports_native_gate_decomposition(self._runtime):
                            raise NotImplementedError
                        _apply_mcx_with_control_values(
                            self._runtime,
                            native_wire_indices,
                            _mcx_control_values(op),
                        )
                    except (NotImplementedError, RuntimeError, TypeError, ValueError):
                        matrix = matrix_to_little_endian_wires(qml.matrix(op))
                        self._runtime.apply_operation(
                            gate_name,
                            wire_indices,
                            matrix=matrix,
                        )
                operation_applied = True
            elif gate_name == "MultiRZ":
                try:
                    (theta,) = getattr(op, "parameters", [])
                    if not _supports_native_parametric_decomposition(self._runtime):
                        raise NotImplementedError
                    _apply_multirz(self._runtime, wire_indices, theta)
                except (NotImplementedError, RuntimeError, TypeError, ValueError):
                    matrix = matrix_to_little_endian_wires(qml.matrix(op))
                    self._runtime.apply_operation(
                        gate_name,
                        wire_indices,
                        matrix=matrix,
                    )
                operation_applied = True
            elif gate_name == "PauliRot":
                try:
                    (theta,) = getattr(op, "parameters", [])
                    if not _supports_native_parametric_decomposition(self._runtime):
                        raise NotImplementedError
                    _apply_paulirot(self._runtime, wire_indices, theta, _paulirot_pauli_word(op))
                except (NotImplementedError, RuntimeError, TypeError, ValueError):
                    matrix = matrix_to_little_endian_wires(qml.matrix(op))
                    self._runtime.apply_operation(
                        gate_name,
                        wire_indices,
                        matrix=matrix,
                    )
                operation_applied = True
            elif gate_name == "IsingZZ":
                try:
                    (theta,) = getattr(op, "parameters", [])
                    if not _supports_native_parametric_decomposition(self._runtime):
                        raise NotImplementedError
                    _apply_multirz(self._runtime, wire_indices, theta)
                except (NotImplementedError, RuntimeError, TypeError, ValueError):
                    matrix = matrix_to_little_endian_wires(qml.matrix(op))
                    self._runtime.apply_operation(
                        gate_name,
                        wire_indices,
                        matrix=matrix,
                    )
                operation_applied = True
            elif gate_name == "IsingXX":
                try:
                    (theta,) = getattr(op, "parameters", [])
                    if not _supports_native_parametric_decomposition(self._runtime):
                        raise NotImplementedError
                    _apply_isingxx(self._runtime, wire_indices, theta)
                except (NotImplementedError, RuntimeError, TypeError, ValueError):
                    matrix = matrix_to_little_endian_wires(qml.matrix(op))
                    self._runtime.apply_operation(
                        gate_name,
                        wire_indices,
                        matrix=matrix,
                    )
                operation_applied = True
            elif gate_name == "IsingYY":
                try:
                    (theta,) = getattr(op, "parameters", [])
                    if not _supports_native_parametric_decomposition(self._runtime):
                        raise NotImplementedError
                    _apply_isingyy(self._runtime, wire_indices, theta)
                except (NotImplementedError, RuntimeError, TypeError, ValueError):
                    matrix = matrix_to_little_endian_wires(qml.matrix(op))
                    self._runtime.apply_operation(
                        gate_name,
                        wire_indices,
                        matrix=matrix,
                    )
                operation_applied = True
            elif gate_name == "IsingXY":
                try:
                    (theta,) = getattr(op, "parameters", [])
                    if not _supports_native_parametric_decomposition(self._runtime):
                        raise NotImplementedError
                    _apply_isingxy(self._runtime, wire_indices, theta)
                except (NotImplementedError, RuntimeError, TypeError, ValueError):
                    matrix = matrix_to_little_endian_wires(qml.matrix(op))
                    self._runtime.apply_operation(
                        gate_name,
                        wire_indices,
                        matrix=matrix,
                    )
                operation_applied = True
            elif gate_name == "CY":
                try:
                    if not _supports_native_gate_decomposition(self._runtime):
                        raise NotImplementedError
                    _apply_cy(self._runtime, wire_indices)
                except (NotImplementedError, RuntimeError, TypeError, ValueError):
                    matrix = matrix_to_little_endian_wires(qml.matrix(op))
                    self._runtime.apply_operation(
                        gate_name,
                        wire_indices,
                        matrix=matrix,
                    )
                operation_applied = True
            elif gate_name == "CH":
                try:
                    if not _supports_native_gate_decomposition(self._runtime):
                        raise NotImplementedError
                    _apply_ch(self._runtime, wire_indices)
                except (NotImplementedError, RuntimeError, TypeError, ValueError):
                    matrix = matrix_to_little_endian_wires(qml.matrix(op))
                    self._runtime.apply_operation(
                        gate_name,
                        wire_indices,
                        matrix=matrix,
                    )
                operation_applied = True
            elif gate_name == "ISWAP":
                try:
                    if not _supports_native_gate_decomposition(self._runtime):
                        raise NotImplementedError
                    _apply_iswap(self._runtime, wire_indices)
                except (NotImplementedError, RuntimeError, TypeError, ValueError):
                    matrix = matrix_to_little_endian_wires(qml.matrix(op))
                    self._runtime.apply_operation(
                        gate_name,
                        wire_indices,
                        matrix=matrix,
                    )
                operation_applied = True
            elif gate_name == "PSWAP":
                try:
                    (phi,) = getattr(op, "parameters", [])
                    if not _supports_native_phase_decomposition(self._runtime):
                        raise NotImplementedError
                    _apply_pswap(self._runtime, wire_indices, phi)
                except (NotImplementedError, RuntimeError, TypeError, ValueError):
                    matrix = matrix_to_little_endian_wires(qml.matrix(op))
                    self._runtime.apply_operation(
                        gate_name,
                        wire_indices,
                        matrix=matrix,
                    )
                operation_applied = True
            elif gate_name in {"SISWAP", "SQISW"}:
                try:
                    if not _supports_native_phase_decomposition(self._runtime):
                        raise NotImplementedError
                    _apply_siswap(self._runtime, wire_indices)
                except (NotImplementedError, RuntimeError, TypeError, ValueError):
                    matrix = matrix_to_little_endian_wires(qml.matrix(op))
                    self._runtime.apply_operation(
                        gate_name,
                        wire_indices,
                        matrix=matrix,
                    )
                operation_applied = True
            elif gate_name == "ECR":
                try:
                    if not _supports_native_phase_decomposition(self._runtime):
                        raise NotImplementedError
                    _apply_ecr(self._runtime, wire_indices)
                except (NotImplementedError, RuntimeError, TypeError, ValueError):
                    matrix = matrix_to_little_endian_wires(qml.matrix(op))
                    self._runtime.apply_operation(
                        gate_name,
                        wire_indices,
                        matrix=matrix,
                    )
                operation_applied = True
            elif gate_name == "SingleExcitation":
                try:
                    (theta,) = getattr(op, "parameters", [])
                    if not _supports_native_parametric_decomposition(self._runtime):
                        raise NotImplementedError
                    _apply_single_excitation(self._runtime, wire_indices, theta)
                except (NotImplementedError, RuntimeError, TypeError, ValueError):
                    matrix = matrix_to_little_endian_wires(qml.matrix(op))
                    self._runtime.apply_operation(
                        gate_name,
                        wire_indices,
                        matrix=matrix,
                    )
                operation_applied = True
            elif gate_name in {"SingleExcitationPlus", "SingleExcitationMinus"}:
                try:
                    (theta,) = getattr(op, "parameters", [])
                    if not _supports_native_phase_decomposition(self._runtime):
                        raise NotImplementedError
                    sign = 1 if gate_name == "SingleExcitationPlus" else -1
                    _apply_single_excitation_phase_variant(self._runtime, wire_indices, theta, sign)
                except (NotImplementedError, RuntimeError, TypeError, ValueError):
                    matrix = matrix_to_little_endian_wires(qml.matrix(op))
                    self._runtime.apply_operation(
                        gate_name,
                        wire_indices,
                        matrix=matrix,
                    )
                operation_applied = True
            elif gate_name == "CRot":
                try:
                    phi, theta, omega = getattr(op, "parameters", [])
                    if not _supports_native_parametric_decomposition(self._runtime):
                        raise NotImplementedError
                    _apply_crot(self._runtime, wire_indices, phi, theta, omega)
                except (NotImplementedError, RuntimeError, TypeError, ValueError):
                    matrix = matrix_to_little_endian_wires(qml.matrix(op))
                    self._runtime.apply_operation(
                        gate_name,
                        wire_indices,
                        matrix=matrix,
                    )
                operation_applied = True
            elif gate_name == "DoubleExcitation":
                try:
                    (theta,) = getattr(op, "parameters", [])
                    if not _supports_native_parametric_decomposition(self._runtime):
                        raise NotImplementedError
                    _apply_double_excitation(self._runtime, wire_indices, theta)
                except (NotImplementedError, RuntimeError, TypeError, ValueError):
                    matrix = matrix_to_little_endian_wires(qml.matrix(op))
                    self._runtime.apply_operation(
                        gate_name,
                        wire_indices,
                        matrix=matrix,
                    )
                operation_applied = True
            elif gate_name in {"DoubleExcitationPlus", "DoubleExcitationMinus"}:
                try:
                    (theta,) = getattr(op, "parameters", [])
                    if not _supports_native_phase_decomposition(self._runtime):
                        raise NotImplementedError
                    sign = 1 if gate_name == "DoubleExcitationPlus" else -1
                    _apply_double_excitation_phase_variant(self._runtime, wire_indices, theta, sign)
                except (NotImplementedError, RuntimeError, TypeError, ValueError):
                    matrix = matrix_to_little_endian_wires(qml.matrix(op))
                    self._runtime.apply_operation(
                        gate_name,
                        wire_indices,
                        matrix=matrix,
                    )
                operation_applied = True
            elif gate_name == "FermionicSWAP":
                try:
                    (theta,) = getattr(op, "parameters", [])
                    if not _supports_native_phase_decomposition(self._runtime):
                        raise NotImplementedError
                    _apply_fermionic_swap(self._runtime, wire_indices, theta)
                except (NotImplementedError, RuntimeError, TypeError, ValueError):
                    matrix = matrix_to_little_endian_wires(qml.matrix(op))
                    self._runtime.apply_operation(
                        gate_name,
                        wire_indices,
                        matrix=matrix,
                    )
                operation_applied = True
            elif gate_name == "OrbitalRotation":
                try:
                    (theta,) = getattr(op, "parameters", [])
                    if not _supports_native_phase_decomposition(self._runtime):
                        raise NotImplementedError
                    _apply_orbital_rotation(self._runtime, wire_indices, theta)
                except (NotImplementedError, RuntimeError, TypeError, ValueError):
                    matrix = matrix_to_little_endian_wires(qml.matrix(op))
                    self._runtime.apply_operation(
                        gate_name,
                        wire_indices,
                        matrix=matrix,
                    )
                operation_applied = True
            elif gate_name == "CCZ":
                try:
                    if not _supports_native_gate_decomposition(self._runtime):
                        raise NotImplementedError
                    _apply_ccz(self._runtime, wire_indices)
                except (NotImplementedError, RuntimeError, TypeError, ValueError):
                    matrix = matrix_to_little_endian_wires(qml.matrix(op))
                    self._runtime.apply_operation(
                        gate_name,
                        wire_indices,
                        matrix=matrix,
                    )
                operation_applied = True
            elif gate_name == "PhaseShift":
                try:
                    (theta,) = getattr(op, "parameters", [])
                    if not _supports_native_phase_decomposition(self._runtime):
                        raise NotImplementedError
                    _apply_phase_shift(self._runtime, wire_indices, theta)
                except (NotImplementedError, RuntimeError, TypeError, ValueError):
                    matrix = matrix_to_little_endian_wires(qml.matrix(op))
                    self._runtime.apply_operation(
                        gate_name,
                        wire_indices,
                        matrix=matrix,
                    )
                operation_applied = True
            elif gate_name == "ControlledPhaseShift":
                try:
                    (theta,) = getattr(op, "parameters", [])
                    if not _supports_native_phase_decomposition(self._runtime):
                        raise NotImplementedError
                    _apply_controlled_phase_shift(self._runtime, wire_indices, theta)
                except (NotImplementedError, RuntimeError, TypeError, ValueError):
                    matrix = matrix_to_little_endian_wires(qml.matrix(op))
                    self._runtime.apply_operation(
                        gate_name,
                        wire_indices,
                        matrix=matrix,
                    )
                operation_applied = True
            elif gate_name == "ControlledQubitUnitary":
                native_components = _controlled_qubit_unitary_components(op, self.wire_map)
                if native_components is None:
                    matrix = matrix_to_little_endian_wires(qml.matrix(op))
                    self._runtime.apply_operation(
                        gate_name,
                        wire_indices,
                        matrix=matrix,
                    )
                else:
                    matrix, controls, targets, control_values = native_components
                    try:
                        _apply_controlled_qubit_unitary(
                            self._runtime,
                            matrix,
                            controls,
                            targets,
                            control_values,
                        )
                    except (NotImplementedError, RuntimeError, TypeError, ValueError):
                        matrix = matrix_to_little_endian_wires(qml.matrix(op))
                        self._runtime.apply_operation(
                            gate_name,
                            wire_indices,
                            matrix=matrix,
                        )
                operation_applied = True
            elif gate_name in {"CPhaseShift00", "CPhaseShift01", "CPhaseShift10"}:
                try:
                    (theta,) = getattr(op, "parameters", [])
                    if not _supports_native_phase_decomposition(self._runtime):
                        raise NotImplementedError
                    control_state = gate_name[len("CPhaseShift"):]
                    _apply_controlled_phase_variant(self._runtime, wire_indices, theta, control_state)
                except (NotImplementedError, RuntimeError, TypeError, ValueError):
                    matrix = matrix_to_little_endian_wires(qml.matrix(op))
                    self._runtime.apply_operation(
                        gate_name,
                        wire_indices,
                        matrix=matrix,
                    )
                operation_applied = True
            elif gate_name == "QFT":
                try:
                    _apply_qft(self._runtime, wire_indices)
                except (NotImplementedError, RuntimeError, TypeError, ValueError):
                    matrix = matrix_to_little_endian_wires(qml.matrix(op))
                    self._runtime.apply_operation(
                        gate_name,
                        wire_indices,
                        matrix=matrix,
                    )
                operation_applied = True
            elif gate_name == "GlobalPhase":
                try:
                    (theta,) = getattr(op, "parameters", [])
                    fallback_wire = self.wire_map[self.wires[0]] if len(self.wires) else None
                    _apply_global_phase_operation(
                        self._runtime,
                        wire_indices,
                        theta,
                        fallback_wire=fallback_wire,
                    )
                except (NotImplementedError, RuntimeError, TypeError, ValueError):
                    matrix = matrix_to_little_endian_wires(qml.matrix(op))
                    self._runtime.apply_operation(
                        gate_name,
                        wire_indices,
                        matrix=matrix,
                    )
                operation_applied = True
            elif gate_name == "DiagonalQubitUnitary":
                try:
                    (diagonal,) = getattr(op, "parameters", [])
                    _apply_diagonal_qubit_unitary(self._runtime, wire_indices, diagonal)
                except (NotImplementedError, RuntimeError, TypeError, ValueError):
                    matrix = matrix_to_little_endian_wires(qml.matrix(op))
                    self._runtime.apply_operation(
                        gate_name,
                        wire_indices,
                        matrix=matrix,
                    )
                operation_applied = True
            elif gate_name == "SelectPauliRot":
                try:
                    (angles,) = getattr(op, "parameters", [])
                    _apply_select_pauli_rot(
                        self._runtime,
                        wire_indices,
                        angles,
                        _select_pauli_rot_axis(op),
                    )
                except (NotImplementedError, RuntimeError, TypeError, ValueError):
                    matrix = matrix_to_little_endian_wires(qml.matrix(op))
                    self._runtime.apply_operation(
                        gate_name,
                        wire_indices,
                        matrix=matrix,
                    )
                operation_applied = True
            elif gate_name == "BasisEmbedding":
                try:
                    bits = _basis_state_bits(op, len(wire_indices), op_name="BasisEmbedding")
                    _apply_basis_embedding(self._runtime, wire_indices, bits)
                except (NotImplementedError, RuntimeError, TypeError, ValueError):
                    matrix = matrix_to_little_endian_wires(qml.matrix(op))
                    self._runtime.apply_operation(
                        gate_name,
                        wire_indices,
                        matrix=matrix,
                    )
                operation_applied = True
            elif gate_name == "Permute":
                try:
                    _apply_permute(self._runtime, wire_indices, op)
                except (NotImplementedError, RuntimeError, TypeError, ValueError):
                    matrix = matrix_to_little_endian_wires(qml.matrix(op))
                    self._runtime.apply_operation(
                        gate_name,
                        wire_indices,
                        matrix=matrix,
                    )
                operation_applied = True
            elif gate_name == "QubitSum":
                try:
                    _apply_qubit_sum(self._runtime, wire_indices)
                except (NotImplementedError, RuntimeError, TypeError, ValueError):
                    matrix = matrix_to_little_endian_wires(qml.matrix(op))
                    self._runtime.apply_operation(
                        gate_name,
                        wire_indices,
                        matrix=matrix,
                    )
                operation_applied = True
            elif gate_name == "QubitCarry":
                try:
                    _apply_qubit_carry(self._runtime, wire_indices)
                except (NotImplementedError, RuntimeError, TypeError, ValueError):
                    matrix = matrix_to_little_endian_wires(qml.matrix(op))
                    self._runtime.apply_operation(
                        gate_name,
                        wire_indices,
                        matrix=matrix,
                    )
                operation_applied = True
            elif gate_name == "GroverOperator":
                try:
                    _apply_grover_operator(self._runtime, wire_indices)
                except (NotImplementedError, RuntimeError, TypeError, ValueError):
                    matrix = matrix_to_little_endian_wires(qml.matrix(op))
                    self._runtime.apply_operation(
                        gate_name,
                        wire_indices,
                        matrix=matrix,
                    )
                operation_applied = True
            elif gate_name in MATRIX_OPS:
                matrix = matrix_to_little_endian_wires(qml.matrix(op))
                self._runtime.apply_operation(
                    gate_name,
                    wire_indices,
                    matrix=matrix,
                )
                operation_applied = True
            else:
                raise NotImplementedError(f"Operation {gate_name} not supported.")
        execute = getattr(self.sim, "Execute", None)
        if callable(execute):
            execute()
        self._state = None
        if len(rotation_ops) == 0:
            self._capture_adjoint_reference_state(flush=False)

    def _capture_adjoint_reference_state(self, *, flush):
        if not self._capture_pre_rotated_state:
            return
        if flush:
            execute = getattr(self.sim, "Execute", None)
            if callable(execute):
                execute()
            self._state = None
        self._pre_rotated_state = self._reshape(self._ensure_state(), [2] * self.num_wires)

    def _apply_operation(self, state, operation):
        if operation.name == "Identity":
            return state
        matrix = np.asarray(qml.matrix(operation), dtype=self.C_DTYPE)
        return self._apply_unitary(state, matrix, operation.wires)

    def _apply_unitary(self, state, mat, wires):
        device_wires = list(self.map_wires(wires))
        matrix = np.asarray(mat, dtype=self.C_DTYPE).reshape([2] * len(device_wires) * 2)
        axes = (list(range(len(device_wires), 2 * len(device_wires))), device_wires)
        contracted = np.tensordot(matrix, state, axes=axes)
        unused_axes = [idx for idx in range(self.num_wires) if idx not in device_wires]
        permutation = list(device_wires) + unused_axes
        inverse_permutation = np.argsort(permutation)
        return np.transpose(contracted, inverse_permutation)

    def _ensure_state(self):
        if self._state is None:
            self._state = self._runtime.statevector()
        return self._state

    @property
    def state(self):
        return self._ensure_state()

    def generate_samples(self):
        if self.shots is None:
            raise ValueError("shots must be set before generating samples.")

        all_wires = list(range(len(self.wires)))
        shots = _shot_count(self.shots)
        measure = getattr(self.sim, "measure", None)
        if callable(measure):
            raw_samples = self._runtime.measure(all_wires, shots)
            return samples_to_binary_rows(raw_samples, len(all_wires))

        return sample_rows_from_statevector(self._ensure_state(), shots)

    def expval(self, observable, shot_range=None, bin_size=None):
        if self.shots is not None or shot_range is not None or bin_size is not None:
            return super().expval(observable, shot_range=shot_range, bin_size=bin_size)
        if self._diagonalizing_rotations_applied:
            return super().expval(observable, shot_range=shot_range, bin_size=bin_size)

        if observable.name == "SparseHamiltonian":
            try:
                mean, _ = _native_sparse_hamiltonian_moments(self._runtime, observable, self.wires)
            except NotImplementedError:
                mean, _ = _sparse_hamiltonian_moments(self._ensure_state(), observable, self.wires)
            return _real_measurement_result(mean, "Expectation value")

        if observable.name == "Hermitian":
            try:
                mean = _native_hermitian_expectation(self._runtime, observable, self.wire_map)
            except (NotImplementedError, RuntimeError):
                return super().expval(observable, shot_range=shot_range, bin_size=bin_size)
            return _real_measurement_result(mean, "Expectation value")

        terms = _pauli_terms_from_observable(observable, self.wire_map)
        if terms is None:
            return super().expval(observable, shot_range=shot_range, bin_size=bin_size)

        return _real_measurement_result(_evaluate_pauli_terms(self._runtime, terms), "Expectation value")

    def var(self, observable, shot_range=None, bin_size=None):
        if self.shots is not None or shot_range is not None or bin_size is not None:
            return super().var(observable, shot_range=shot_range, bin_size=bin_size)
        if self._diagonalizing_rotations_applied:
            return super().var(observable, shot_range=shot_range, bin_size=bin_size)

        if observable.name == "SparseHamiltonian":
            try:
                mean, second_moment = _native_sparse_hamiltonian_moments(
                    self._runtime,
                    observable,
                    self.wires,
                )
            except NotImplementedError:
                mean, second_moment = _sparse_hamiltonian_moments(self._ensure_state(), observable, self.wires)
            return _real_measurement_result(second_moment - mean * mean, "Variance")

        if observable.name == "Hermitian":
            try:
                matrix, targets = _hermitian_matrix_and_targets(observable, self.wire_map)
                mean = self._runtime.expectation_matrix(matrix, targets)
                second_moment = self._runtime.expectation_matrix(matrix @ matrix, targets)
            except (NotImplementedError, RuntimeError):
                return super().var(observable, shot_range=shot_range, bin_size=bin_size)
            return _real_measurement_result(second_moment - mean * mean, "Variance")

        terms = _pauli_terms_from_observable(observable, self.wire_map)
        if terms is None:
            return super().var(observable, shot_range=shot_range, bin_size=bin_size)

        mean = _evaluate_pauli_terms(self._runtime, terms)
        second_moment = _evaluate_pauli_terms(self._runtime, _pauli_square_terms(terms))
        return _real_measurement_result(second_moment - mean * mean, "Variance")

    def analytic_probability(self, wires=None):
        requested_labels = list(self.wires if wires is None or len(wires) == 0 else getattr(wires, "labels", wires))
        wire_indices = [self.wire_map[wire] for wire in requested_labels]
        has_native_probabilities = callable(getattr(self.sim, "probabilities", None)) or callable(
            getattr(self.sim, "Probabilities", None)
        )
        if has_native_probabilities:
            try:
                return self._runtime.probabilities(wire_indices)
            except NotImplementedError:
                pass

        all_probs = np.abs(self._ensure_state()) ** 2
        if wires is None or len(wires) == 0: return all_probs
        requested_wires = set(getattr(wires, "labels", wires))
        wires_to_trace = [i for i, w in enumerate(self.wires) if w not in requested_wires]
        if not wires_to_trace: return all_probs
        return self.marginal_prob(all_probs, wires_to_trace)
