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

        result = 0.0 + 0.0j
        for coeff, pauli_string, targets in terms:
            if not targets:
                result += coeff
            else:
                result += coeff * self._runtime.expectation_pauli_string(pauli_string, targets)
        return float(np.real_if_close(result).real)

    def analytic_probability(self, wires=None):
        if self._state is None: return None
        all_probs = np.abs(self._state) ** 2
        if wires is None: return all_probs
        wires_to_trace = [i for i, w in enumerate(self.wires) if w not in wires]
        return self.marginal_prob(all_probs, wires_to_trace)
