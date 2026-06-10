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
    "MultiControlledX", "MultiRZ",
    "IsingXX", "IsingYY", "IsingZZ", "IsingXY",
    "PSWAP", "ISWAP", "SISWAP", "SQISW", "ECR",
    "SingleExcitation", "SingleExcitationPlus", "SingleExcitationMinus",
    "DoubleExcitation", "DoubleExcitationPlus", "DoubleExcitationMinus",
    "OrbitalRotation", "FermionicSWAP",
    "Toffoli", "CSWAP",
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


def _shot_count(shots):
    total_shots = getattr(shots, "total_shots", None)
    if total_shots is not None:
        return int(total_shots)
    return int(shots)


def _basis_state_bits(op, num_wires):
    if not getattr(op, "parameters", None):
        raise ValueError("BasisState requires a computational basis vector.")
    bits = np.asarray(op.parameters[0], dtype=int).reshape(-1)
    if len(bits) != int(num_wires):
        raise ValueError("BasisState length must match the number of target wires.")
    if not np.all((bits == 0) | (bits == 1)):
        raise ValueError("BasisState entries must be 0 or 1.")
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


def _apply_phase_shift(runtime, wire_indices, theta):
    if len(wire_indices) != 1:
        raise ValueError("PhaseShift requires exactly one wire.")

    wire_index = wire_indices[0]
    _apply_global_phase(runtime, wire_index, 0.5 * theta)
    runtime.apply_operation("RZ", [wire_index], [theta])


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
    _apply_global_phase(runtime, control, 0.25 * theta)
    runtime.apply_operation("RZ", [control], [0.5 * theta])
    runtime.apply_operation("CNOT", [control, target])
    runtime.apply_operation("RZ", [target], [-0.5 * theta])
    runtime.apply_operation("CNOT", [control, target])
    runtime.apply_operation("RZ", [target], [0.5 * theta])


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


def _apply_isingxx(runtime, wire_indices, theta):
    if len(wire_indices) != 2:
        raise ValueError("IsingXX requires exactly two wires.")

    control, target = wire_indices
    runtime.apply_operation("CNOT", [control, target])
    runtime.apply_operation("RX", [control], [theta])
    runtime.apply_operation("CNOT", [control, target])


def _apply_isingyy(runtime, wire_indices, theta):
    if len(wire_indices) != 2:
        raise ValueError("IsingYY requires exactly two wires.")

    for wire_index in wire_indices:
        runtime.apply_operation("RX", [wire_index], [np.pi / 2])
    _apply_multirz(runtime, wire_indices, theta)
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


def _apply_orbital_rotation(runtime, wire_indices, theta):
    if len(wire_indices) != 4:
        raise ValueError("OrbitalRotation requires exactly four wires.")

    first, second, third, fourth = wire_indices
    _apply_fermionic_swap(runtime, [second, third], np.pi)
    _apply_single_excitation(runtime, [first, second], theta)
    _apply_single_excitation(runtime, [third, fourth], theta)
    _apply_fermionic_swap(runtime, [second, third], np.pi)


class RocQDevice(QubitDevice):
    name = "rocQuantum Simulator Device"
    short_name = "rocquantum.qpu"
    author = "rocQuantum contributors"
    version = "0.1.0"
    pennylane_requires = ">=0.30"

    operations = set(PENNYLANE_TO_ROCQ_GATES.keys()) | NATIVE_PARAMETRIC_OPS | MATRIX_OPS | {"BasisState", "StatePrep", "Rot"}
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
        self._preserve_global_phase = True
        self.reset()

    def reset(self):
        try:
            self._runtime = RocQuantumRuntime.from_bindings(len(self.wires), binding_module=rocquantum_bind)
        except ImportError as exc:
            raise ImportError(
                "The 'rocquantum_bind' module is not installed. "
                "Build and install rocQuantum with ROCQUANTUM_BUILD_BINDINGS=ON before creating "
                "a PennyLane rocQuantum device."
            ) from exc
        self.sim = self._runtime.simulator
        self._runtime.preserve_global_phase = self._preserve_global_phase
        self._state = None
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
            if observable is None or _pauli_terms_from_observable(observable, self.wire_map) is None:
                return False
        return True

    def _get_diagonalizing_gates(self, circuit):
        if self._skip_diagonalizing_rotations:
            return []
        return super()._get_diagonalizing_gates(circuit)

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

    def apply(self, operations: list[Operation], rotations=None, **kwargs):
        operation_applied = False
        rotation_ops = list(rotations or [])
        self._diagonalizing_rotations_applied = bool(rotation_ops)
        self._runtime.preserve_global_phase = self._preserve_global_phase
        for op in list(operations) + rotation_ops:
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
