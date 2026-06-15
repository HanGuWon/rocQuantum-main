# cirq-rocm/cirq_rocm/roc_quantum_simulator.py
import cirq
import numpy as np

from rocquantum.framework_runtime import RocQuantumRuntime, matrix_to_little_endian_wires

try:
    import rocquantum_bind
except ImportError:
    rocquantum_bind = None

def _rotation_gate_name(gate):
    if isinstance(gate, cirq.Rx):
        return "RX"
    if isinstance(gate, cirq.Ry):
        return "RY"
    if isinstance(gate, cirq.Rz):
        return "RZ"
    return None


def _rotation_angle(gate):
    angle = getattr(gate, "_rads", None)
    if angle is not None:
        return float(angle)
    exponent = getattr(gate, "exponent", None)
    if exponent is None:
        raise TypeError(f"Unable to extract a rotation angle from {gate!r}.")
    return float(exponent) * np.pi


def _fixed_gate_name(gate):
    fixed_gates = (
        (cirq.X, "X"),
        (cirq.Y, "Y"),
        (cirq.Z, "Z"),
        (cirq.H, "H"),
        (cirq.S, "S"),
        (cirq.S**-1, "SDG"),
        (cirq.T, "T"),
        (cirq.T**-1, "TDG"),
        (cirq.CNOT, "CNOT"),
        (cirq.CZ, "CZ"),
        (cirq.SWAP, "SWAP"),
        (cirq.TOFFOLI, "TOFFOLI"),
        (cirq.FREDKIN, "CSWAP"),
    )
    for cirq_gate, rocq_name in fixed_gates:
        if gate == cirq_gate:
            return rocq_name
    return None


def _samples_to_bits(raw_samples, width):
    return np.array(
        [[(int(sample) >> bit) & 1 for bit in range(width)] for sample in raw_samples],
        dtype=np.int8,
    )


class RocQuantumSimulator(cirq.SimulatesFinalState, cirq.SimulatesSamples):
    def _require_binding(self):
        if rocquantum_bind is None:
            raise ImportError(
                "The 'rocquantum_bind' module is not installed. "
                "Build rocQuantum with native Python bindings before executing Cirq circuits."
            )
        return rocquantum_bind

    def _run(self, circuit, param_resolver, repetitions):
        qubit_order = sorted(circuit.all_qubits())
        runtime, q_map = self._execute_circuit(circuit, qubit_order)
        state_vector = None
        full_outcomes = None
        measurements = {}

        for op in circuit.all_operations():
            if isinstance(op.gate, cirq.MeasurementGate):
                key = op.gate.key
                indices = [q_map[q] for q in op.qubits]
                try:
                    raw_samples = runtime.measure(indices, repetitions)
                    values = _samples_to_bits(raw_samples, len(indices))
                except NotImplementedError:
                    if state_vector is None:
                        state_vector = runtime.statevector()
                        probs = np.abs(state_vector) ** 2
                        full_outcomes = np.random.choice(len(probs), size=repetitions, p=probs)
                    values = (full_outcomes[:, np.newaxis] >> indices) & 1
                measurements[key] = values
        return measurements

    def _execute_circuit(self, circuit, qubit_order):
        binding = self._require_binding()
        q_map = {q: i for i, q in enumerate(qubit_order)}
        runtime = RocQuantumRuntime.from_bindings(len(q_map), binding_module=binding)
        for op in circuit.all_operations():
            if isinstance(op.gate, cirq.MeasurementGate):
                continue
            indices = [q_map[q] for q in op.qubits]
            fixed_gate = _fixed_gate_name(op.gate)
            if fixed_gate is not None:
                runtime.apply_operation(fixed_gate, indices)
            elif _rotation_gate_name(op.gate) is not None:
                try:
                    runtime.apply_operation(_rotation_gate_name(op.gate), indices, [_rotation_angle(op.gate)])
                except NotImplementedError:
                    matrix = matrix_to_little_endian_wires(cirq.unitary(op))
                    runtime.apply_matrix(matrix, indices)
            elif isinstance(op.gate, cirq.MatrixGate):
                matrix = matrix_to_little_endian_wires(cirq.unitary(op))
                runtime.apply_matrix(matrix, indices)
            else:
                raise TypeError(f"Unsupported gate: {op.gate}")
        execute = getattr(runtime.simulator, "Execute", None)
        if callable(execute):
            execute()
        return runtime, q_map

    def _get_final_statevector(self, circuit, qubit_order):
        runtime, _ = self._execute_circuit(circuit, qubit_order)
        return runtime.statevector()

    def _perform_final_state_simulation(self, circuit, qubit_order, initial_state):
        if not np.allclose(initial_state[0], 1):
            raise ValueError("Only simulation from |0> state is supported.")
        state_vector = self._get_final_statevector(circuit, qubit_order)
        return cirq.StateVectorTrialResult(params={}, measurements={}, final_simulator_state=state_vector)
