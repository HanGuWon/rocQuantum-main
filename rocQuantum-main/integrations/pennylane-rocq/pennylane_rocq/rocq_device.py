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
    sample_rows_from_statevector,
    samples_to_binary_rows,
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

NATIVE_PARAMETRIC_OPS = {"RX", "RY", "RZ"}
MATRIX_OPS = {"QubitUnitary"}
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


def _pauli_terms_from_observable(observable, wire_map):
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
    return [_multiply_pauli_terms(left, right) for left in terms for right in terms]


def _evaluate_pauli_terms(runtime, terms):
    result = 0.0 + 0.0j
    for coeff, pauli_string, targets in terms:
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


class RocQDevice(QubitDevice):
    name = "rocQuantum Simulator Device"
    short_name = "rocquantum.qpu"
    author = "rocQuantum contributors"
    version = "0.1.0"
    pennylane_requires = ">=0.30"

    operations = set(PENNYLANE_TO_ROCQ_GATES.keys()) | {"QubitUnitary", "RX", "RY", "RZ", "Rot"}
    observables = {
        "PauliX", "PauliY", "PauliZ", "Identity",
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
        self._state = None

    def apply(self, operations: list[Operation], rotations=None, **kwargs):
        for op in list(operations):
            gate_name = op.name
            wire_indices = [self.wire_map[w] for w in op.wires]
            if gate_name in PENNYLANE_TO_ROCQ_GATES:
                self._runtime.apply_operation(PENNYLANE_TO_ROCQ_GATES[gate_name], wire_indices)
            elif gate_name in NATIVE_PARAMETRIC_OPS:
                try:
                    self._runtime.apply_operation(gate_name, wire_indices, getattr(op, "parameters", []))
                except NotImplementedError:
                    matrix = qml.matrix(op)
                    self._runtime.apply_operation(
                        "QubitUnitary",
                        wire_indices,
                        matrix=matrix.astype(np.complex128),
                    )
            elif gate_name == "Rot":
                try:
                    phi, theta, omega = getattr(op, "parameters", [])
                    self._runtime.apply_operation("RZ", wire_indices, [phi])
                    self._runtime.apply_operation("RY", wire_indices, [theta])
                    self._runtime.apply_operation("RZ", wire_indices, [omega])
                except NotImplementedError:
                    matrix = qml.matrix(op)
                    self._runtime.apply_operation(
                        "QubitUnitary",
                        wire_indices,
                        matrix=matrix.astype(np.complex128),
                    )
            elif gate_name in MATRIX_OPS:
                matrix = qml.matrix(op)
                self._runtime.apply_operation(
                    gate_name,
                    wire_indices,
                    matrix=matrix.astype(np.complex128),
                )
            else:
                raise NotImplementedError(f"Operation {gate_name} not supported.")
        execute = getattr(self.sim, "Execute", None)
        if callable(execute):
            execute()
        self._state = self._runtime.statevector()

    @property
    def state(self):
        return self._state

    def generate_samples(self):
        if self.shots is None:
            raise ValueError("shots must be set before generating samples.")

        all_wires = list(range(len(self.wires)))
        measure = getattr(self.sim, "measure", None)
        if callable(measure):
            raw_samples = self._runtime.measure(all_wires, int(self.shots))
            return samples_to_binary_rows(raw_samples, len(all_wires))

        return sample_rows_from_statevector(self._state, int(self.shots))

    def expval(self, observable, shot_range=None, bin_size=None):
        if self.shots is not None or shot_range is not None or bin_size is not None:
            return super().expval(observable, shot_range=shot_range, bin_size=bin_size)

        terms = _pauli_terms_from_observable(observable, self.wire_map)
        if terms is None:
            return super().expval(observable, shot_range=shot_range, bin_size=bin_size)

        return _real_measurement_result(_evaluate_pauli_terms(self._runtime, terms), "Expectation value")

    def var(self, observable, shot_range=None, bin_size=None):
        if self.shots is not None or shot_range is not None or bin_size is not None:
            return super().var(observable, shot_range=shot_range, bin_size=bin_size)

        terms = _pauli_terms_from_observable(observable, self.wire_map)
        if terms is None:
            return super().var(observable, shot_range=shot_range, bin_size=bin_size)

        mean = _evaluate_pauli_terms(self._runtime, terms)
        second_moment = _evaluate_pauli_terms(self._runtime, _pauli_square_terms(terms))
        return _real_measurement_result(second_moment - mean * mean, "Variance")

    def analytic_probability(self, wires=None):
        if self._state is None: return None
        all_probs = np.abs(self._state) ** 2
        if wires is None or len(wires) == 0: return all_probs
        requested_wires = set(getattr(wires, "labels", wires))
        wires_to_trace = [i for i, w in enumerate(self.wires) if w not in requested_wires]
        if not wires_to_trace: return all_probs
        return self.marginal_prob(all_probs, wires_to_trace)
