from __future__ import annotations

import math
import os
import sys
import types

import numpy as np
import pytest

try:
    import tomllib
except ModuleNotFoundError:
    tomllib = None

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


class _FakeQuantumSimulator:
    instances = []
    enable_sparse_moments = False
    enable_probabilities = False
    enable_matrix_expectation = False
    measure_qubit_results = []

    def __init__(self, num_qubits, batch_size=1):
        self._num_qubits = int(num_qubits)
        self._batch_size = int(batch_size)
        self._measured = False
        self.ops = []
        self.batch_ops = []
        self.matrices = []
        self.controlled_matrices = []
        self.statevectors = []
        self.measurements = []
        self.measure_qubits = []
        self.total_measurements = []
        self.total_measure_qubits = []
        self.expectations = []
        self.batch_expectations = []
        self.matrix_expectations = []
        self.matrix_batch_expectations = []
        self.probability_requests = []
        self.sparse_moments = []
        self.sparse_batch_moments = []
        self.reset_qubits = []
        self.total_reset_qubits = []
        self.statevector_reads = 0
        self.total_gate_applications = 0
        self.total_matrix_applications = 0
        _FakeQuantumSimulator.instances.append(self)

    def reset(self):
        self.ops.clear()
        self.batch_ops.clear()
        self.matrices.clear()
        self.controlled_matrices.clear()
        self.statevectors.clear()
        self.measurements.clear()
        self.measure_qubits.clear()
        self.expectations.clear()
        self.batch_expectations.clear()
        self.matrix_expectations.clear()
        self.matrix_batch_expectations.clear()
        self.probability_requests.clear()
        self.sparse_moments.clear()
        self.sparse_batch_moments.clear()
        self.reset_qubits.clear()
        self.statevector_reads = 0
        self._measured = False

    def num_qubits(self):
        return self._num_qubits

    def batch_size(self):
        return self._batch_size

    def apply_gate(self, name, targets, params=None):
        self.total_gate_applications += 1
        self.ops.append((name, tuple(targets), tuple(params or ())))

    def apply_gate_batch(self, name, targets, params_by_batch):
        self.total_gate_applications += int(self._batch_size)
        self.batch_ops.append((name, tuple(targets), tuple(float(param) for param in params_by_batch)))

    def apply_matrix(self, matrix, targets):
        self.total_matrix_applications += 1
        self.matrices.append((np.asarray(matrix, dtype=np.complex128), tuple(targets)))

    def apply_controlled_matrix(self, matrix, controls, targets):
        self.controlled_matrices.append(
            (
                np.asarray(matrix, dtype=np.complex128),
                tuple(controls),
                tuple(targets),
            )
        )

    def set_statevector(self, statevector):
        self.statevectors.append(np.asarray(statevector, dtype=np.complex128))

    def set_statevectors(self, statevectors):
        self.statevectors.append(
            np.asarray(statevectors, dtype=np.complex128).reshape(self._batch_size, 1 << self._num_qubits)
        )

    def measure(self, qubits, shots):
        self.measurements.append((tuple(qubits), int(shots)))
        self.total_measurements.append((tuple(qubits), int(shots)))
        self._measured = True
        if self._num_qubits == 1 and len(qubits) == 1:
            return [0 for _ in range(int(shots))]
        high = (1 << len(qubits)) - 1
        return [0 if shot % 2 == 0 else high for shot in range(int(shots))]

    def measure_qubit(self, target):
        target = int(target)
        self.measure_qubits.append(target)
        self.total_measure_qubits.append(target)
        self._measured = True
        if _FakeQuantumSimulator.measure_qubit_results:
            return int(_FakeQuantumSimulator.measure_qubit_results.pop(0))
        return 0

    def reset_qubit(self, target):
        self.reset_qubits.append(int(target))
        self.total_reset_qubits.append(int(target))

    def get_statevector(self):
        self.statevector_reads += 1
        return self._peek_statevector()

    def _peek_statevector(self):
        state = np.zeros(1 << self._num_qubits, dtype=np.complex128)
        if self._measured:
            state[0] = 1.0
            return state
        if self._num_qubits == 2:
            state[0] = 1.0 / math.sqrt(2.0)
            state[3] = 1.0 / math.sqrt(2.0)
        else:
            state[0] = 1.0
        return state

    def expectation_matrix(self, matrix, targets):
        if not self.enable_matrix_expectation:
            raise NotImplementedError("native matrix expectation disabled")
        matrix = np.asarray(matrix, dtype=np.complex128)
        targets = tuple(int(target) for target in targets)
        self.matrix_expectations.append((matrix, targets))
        return self._expectation_matrix_value(matrix, targets)

    def expectation_matrix_batch(self, matrix, targets):
        if not self.enable_matrix_expectation:
            raise NotImplementedError("native matrix expectation batch disabled")
        matrix = np.asarray(matrix, dtype=np.complex128)
        targets = tuple(int(target) for target in targets)
        self.matrix_batch_expectations.append((matrix, targets))
        value = self._expectation_matrix_value(matrix, targets)
        return np.full(self._batch_size, value, dtype=np.complex128)

    def _expectation_matrix_value(self, matrix, targets):
        state = self._peek_statevector()
        dimension = 1 << len(targets)
        result = 0.0 + 0.0j
        for row_index, amplitude in enumerate(state):
            row_target = 0
            base_index = int(row_index)
            for output_bit, target in enumerate(targets):
                mask = 1 << target
                if row_index & mask:
                    row_target |= 1 << output_bit
                base_index &= ~mask
            for col_target in range(dimension):
                col_index = base_index
                for output_bit, target in enumerate(targets):
                    if (col_target >> output_bit) & 1:
                        col_index |= 1 << target
                result += np.conj(amplitude) * matrix[row_target, col_target] * state[col_index]
        return result

    def probabilities(self, qubits):
        if not self.enable_probabilities:
            raise NotImplementedError("native probabilities disabled")
        qubits = tuple(int(qubit) for qubit in qubits)
        self.probability_requests.append(qubits)
        dimension = 1 << len(qubits)
        weights = np.arange(1, dimension + 1, dtype=float)
        return weights / np.sum(weights)

    def probabilities_batch(self, qubits):
        qubits = tuple(int(qubit) for qubit in qubits)
        self.probability_requests.append(qubits)
        dimension = 1 << len(qubits)
        weights = np.arange(1, dimension + 1, dtype=float)
        probabilities = weights / np.sum(weights)
        return np.tile(probabilities, (self._batch_size, 1))

    def expectation_pauli_string(self, pauli_string, targets):
        self.expectations.append((pauli_string, tuple(targets)))
        if pauli_string == "Z" and tuple(targets) == (0,):
            return 0.5
        if pauli_string == "X" and tuple(targets) == (0,):
            return 0.5
        if pauli_string == "XZ" and tuple(targets) == (0, 1):
            return 0.25
        if pauli_string == "XX" and tuple(targets) == (0, 1):
            return 0.25
        return 0.0

    def expectation_pauli_string_batch(self, pauli_string, targets):
        self.batch_expectations.append((pauli_string, tuple(targets)))
        if pauli_string == "Z" and tuple(targets) == (0,):
            return np.full(self._batch_size, 0.5, dtype=float)
        if pauli_string == "X" and tuple(targets) == (0,):
            return np.full(self._batch_size, 0.5, dtype=float)
        return np.zeros(self._batch_size, dtype=float)

    def sparse_hamiltonian_moments(self, data, indices, indptr, shape):
        if not self.enable_sparse_moments:
            raise NotImplementedError("native sparse Hamiltonian moments disabled")
        self.sparse_moments.append(
            (
                np.asarray(data, dtype=np.complex128),
                np.asarray(indices, dtype=np.int64),
                np.asarray(indptr, dtype=np.int64),
                tuple(shape),
            )
        )
        return 1.0 + 0.0j, 1.0 + 0.0j

    def sparse_hamiltonian_moments_batch(self, data, indices, indptr, shape):
        if not self.enable_sparse_moments:
            raise NotImplementedError("native sparse Hamiltonian moments batch disabled")
        self.sparse_batch_moments.append(
            (
                np.asarray(data, dtype=np.complex128),
                np.asarray(indices, dtype=np.int64),
                np.asarray(indptr, dtype=np.int64),
                tuple(shape),
            )
        )
        return (
            np.ones(self._batch_size, dtype=np.complex128),
            np.ones(self._batch_size, dtype=np.complex128),
        )

    def ApplyGate(self, *args):
        self.ops.append(("legacy", args, ()))

    def Execute(self):
        pass

    def GetStateVector(self):
        return self.get_statevector()


def _install_fake_binding(monkeypatch):
    fake = types.ModuleType("rocquantum_bind")
    fake.QuantumSimulator = _FakeQuantumSimulator
    fake.QSim = _FakeQuantumSimulator
    monkeypatch.setitem(sys.modules, "rocquantum_bind", fake)
    _FakeQuantumSimulator.instances.clear()
    _FakeQuantumSimulator.enable_sparse_moments = False
    _FakeQuantumSimulator.enable_probabilities = False
    _FakeQuantumSimulator.enable_matrix_expectation = False
    _FakeQuantumSimulator.measure_qubit_results = []
    return fake


def test_qiskit_backend_returns_job_and_fixed_width_counts(monkeypatch):
    pytest.importorskip("qiskit")
    _install_fake_binding(monkeypatch)

    from qiskit import QuantumCircuit
    from qiskit_rocquantum_provider import RocQuantumProvider

    provider = RocQuantumProvider()
    backend = provider.get_backend("rocq_simulator")

    circuit = QuantumCircuit(2, 2)
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.measure([0, 1], [0, 1])

    job = backend.run(circuit, shots=6)
    result = job.result()

    assert job.done()
    assert result.success
    assert result.get_counts() == {"00": 3, "11": 3}
    assert _FakeQuantumSimulator.instances[-1].measurements == [((0, 1), 6)]
    assert _FakeQuantumSimulator.instances[-1].statevector_reads == 0


def test_qiskit_backend_deduplicates_repeated_terminal_measurements(monkeypatch):
    pytest.importorskip("qiskit")
    _install_fake_binding(monkeypatch)

    from qiskit import QuantumCircuit
    from qiskit_rocquantum_provider import RocQuantumProvider

    backend = RocQuantumProvider().get_backend("rocq_simulator")
    circuit = QuantumCircuit(2, 2)
    circuit.h(0)
    circuit.measure(0, 0)
    circuit.measure(0, 1)

    result = backend.run(circuit, shots=6, statevector=False).result()

    assert result.get_counts() == {"00": 3, "11": 3}
    assert _FakeQuantumSimulator.instances[-1].measurements == [((0,), 6)]


def test_qiskit_backend_returns_statevector_before_sampling(monkeypatch):
    pytest.importorskip("qiskit")
    _install_fake_binding(monkeypatch)

    from qiskit import QuantumCircuit
    from qiskit_rocquantum_provider import RocQuantumProvider

    provider = RocQuantumProvider()
    backend = provider.get_backend("rocq_simulator")

    circuit = QuantumCircuit(2)
    circuit.h(0)
    circuit.cx(0, 1)

    state = backend.run(circuit, shots=4, statevector=True).result().get_statevector()
    expected = np.array([1.0 / math.sqrt(2.0), 0.0, 0.0, 1.0 / math.sqrt(2.0)])

    np.testing.assert_allclose(state, expected)
    assert _FakeQuantumSimulator.instances[-1].measurements == [((0, 1), 4)]


def test_qiskit_backend_ignores_save_statevector_marker(monkeypatch):
    pytest.importorskip("qiskit")
    _install_fake_binding(monkeypatch)

    from qiskit import QuantumCircuit
    from qiskit.circuit import Instruction
    from qiskit_rocquantum_provider import RocQuantumProvider

    provider = RocQuantumProvider()
    backend = provider.get_backend("rocq_simulator")

    circuit = QuantumCircuit(1)
    circuit.h(0)
    circuit.append(Instruction("save_statevector", 0, 0, []), [], [])

    backend.run(circuit, shots=1).result().get_statevector()

    assert _FakeQuantumSimulator.instances[-1].ops == [("H", (0,), ())]


def test_qiskit_backend_can_skip_statevector_readback_for_sampling(monkeypatch):
    pytest.importorskip("qiskit")
    _install_fake_binding(monkeypatch)

    from qiskit import QuantumCircuit
    from qiskit_rocquantum_provider import RocQuantumProvider

    backend = RocQuantumProvider().get_backend("rocq_simulator")
    circuit = QuantumCircuit(2, 2)
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.measure([0, 1], [0, 1])

    result = backend.run(circuit, shots=6, statevector=False).result()

    assert result.get_counts() == {"00": 3, "11": 3}
    assert _FakeQuantumSimulator.instances[-1].measurements == [((0, 1), 6)]
    assert _FakeQuantumSimulator.instances[-1].statevector_reads == 0


def test_qiskit_backend_can_skip_sampling_for_statevector_only(monkeypatch):
    pytest.importorskip("qiskit")
    _install_fake_binding(monkeypatch)

    from qiskit import QuantumCircuit
    from qiskit_rocquantum_provider import RocQuantumProvider

    backend = RocQuantumProvider().get_backend("rocq_simulator")
    circuit = QuantumCircuit(2)
    circuit.h(0)
    circuit.cx(0, 1)

    result = backend.run(circuit, sampling=False, statevector=True).result()
    state = result.get_statevector()

    expected = np.array([1.0 / math.sqrt(2.0), 0.0, 0.0, 1.0 / math.sqrt(2.0)])
    np.testing.assert_allclose(state, expected)
    assert result.results[0].shots == 0
    assert _FakeQuantumSimulator.instances[-1].measurements == []
    assert _FakeQuantumSimulator.instances[-1].statevector_reads == 1


def test_framework_runtime_converts_full_statevectors_to_little_endian_order():
    from rocquantum.framework_runtime import statevector_to_little_endian_wires

    state = np.array([0, 1, 2, 3], dtype=np.complex128)

    np.testing.assert_allclose(
        statevector_to_little_endian_wires(state),
        np.array([0, 2, 1, 3], dtype=np.complex128),
    )


def test_qiskit_backend_samples_mid_circuit_measurement_trajectory(monkeypatch):
    pytest.importorskip("qiskit")
    _install_fake_binding(monkeypatch)
    _FakeQuantumSimulator.measure_qubit_results = [1]

    from qiskit import QuantumCircuit
    from qiskit_rocquantum_provider import RocQuantumProvider

    backend = RocQuantumProvider().get_backend("rocq_simulator")
    circuit = QuantumCircuit(1, 1)
    circuit.measure(0, 0)
    circuit.x(0)

    result = backend.run(circuit, shots=1, memory=True, statevector=False).result()

    assert result.get_counts() == {"1": 1}
    assert result.get_memory() == ["1"]
    sim = _FakeQuantumSimulator.instances[-1]
    assert sim.measure_qubits == [0]
    assert sim.ops == [("X", (0,), ())]


def test_qiskit_backend_allows_initial_reset_as_noop(monkeypatch):
    pytest.importorskip("qiskit")
    _install_fake_binding(monkeypatch)

    from qiskit import QuantumCircuit
    from qiskit_rocquantum_provider import RocQuantumProvider

    backend = RocQuantumProvider().get_backend("rocq_simulator")
    circuit = QuantumCircuit(1)
    circuit.reset(0)
    circuit.x(0)

    assert "reset" in set(backend.target.operation_names)

    backend.run(circuit, sampling=False, statevector=True).result()

    assert _FakeQuantumSimulator.instances[-1].ops == [("X", (0,), ())]


def test_qiskit_backend_samples_runtime_reset_shot_by_shot(monkeypatch):
    pytest.importorskip("qiskit")
    _install_fake_binding(monkeypatch)

    from qiskit import QuantumCircuit
    from qiskit_rocquantum_provider import RocQuantumProvider

    backend = RocQuantumProvider().get_backend("rocq_simulator")
    circuit = QuantumCircuit(2, 2)
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.reset(0)
    circuit.measure([0, 1], [0, 1])

    result = backend.run(circuit, shots=4, statevector=False).result()

    assert result.success
    assert _FakeQuantumSimulator.instances[-1].total_reset_qubits == [0, 0, 0, 0]
    assert _FakeQuantumSimulator.instances[-1].total_measurements == [((0, 1), 1)] * 4


def test_qiskit_backend_rejects_statevector_for_runtime_reset(monkeypatch):
    pytest.importorskip("qiskit")
    _install_fake_binding(monkeypatch)

    from qiskit import QuantumCircuit
    from qiskit_rocquantum_provider import RocQuantumProvider

    backend = RocQuantumProvider().get_backend("rocq_simulator")
    circuit = QuantumCircuit(1)
    circuit.h(0)
    circuit.reset(0)

    with pytest.raises(ValueError, match="single statevector"):
        backend.run(circuit, statevector=True).result()


def test_qiskit_backend_samples_if_else_conditioned_gate(monkeypatch):
    pytest.importorskip("qiskit")
    _install_fake_binding(monkeypatch)
    _FakeQuantumSimulator.measure_qubit_results = [1, 1]

    from qiskit import QuantumCircuit
    from qiskit_rocquantum_provider import RocQuantumProvider

    circuit = QuantumCircuit(2, 2)
    if not hasattr(circuit, "if_test"):
        pytest.skip("Qiskit version does not expose QuantumCircuit.if_test")

    circuit.measure(0, 0)
    with circuit.if_test((circuit.clbits[0], True)):
        circuit.x(1)
    circuit.measure(1, 1)

    backend = RocQuantumProvider().get_backend("rocq_simulator")
    result = backend.run(circuit, shots=1, memory=True, statevector=False).result()

    assert result.get_counts() == {"11": 1}
    assert result.get_memory() == ["11"]
    sim = _FakeQuantumSimulator.instances[-1]
    assert sim.ops == [("X", (1,), ())]
    assert sim.measure_qubits == [0, 1]
    assert sim.measurements == []


def test_qiskit_backend_samples_switch_case_conditioned_gate(monkeypatch):
    pytest.importorskip("qiskit")
    _install_fake_binding(monkeypatch)
    _FakeQuantumSimulator.measure_qubit_results = [1, 1]

    from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
    from qiskit_rocquantum_provider import RocQuantumProvider

    qreg = QuantumRegister(2, "q")
    creg = ClassicalRegister(2, "c")
    circuit = QuantumCircuit(qreg, creg)
    if not hasattr(circuit, "switch"):
        pytest.skip("Qiskit version does not expose QuantumCircuit.switch")

    circuit.measure(qreg[0], creg[0])
    with circuit.switch(creg) as case:
        with case(1):
            circuit.x(qreg[1])
        with case(2):
            circuit.z(qreg[1])
        with case(case.DEFAULT):
            circuit.h(qreg[1])
    circuit.measure(qreg[1], creg[1])

    backend = RocQuantumProvider().get_backend("rocq_simulator")
    result = backend.run(circuit, shots=1, memory=True, statevector=False).result()

    assert result.get_counts() == {"11": 1}
    assert result.get_memory() == ["11"]
    sim = _FakeQuantumSimulator.instances[-1]
    assert sim.ops == [("X", (1,), ())]
    assert sim.measure_qubits == [0, 1]
    assert sim.measurements == []


def test_qiskit_backend_samples_switch_case_default_branch(monkeypatch):
    pytest.importorskip("qiskit")
    _install_fake_binding(monkeypatch)
    _FakeQuantumSimulator.measure_qubit_results = [0, 0]

    from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
    from qiskit_rocquantum_provider import RocQuantumProvider

    qreg = QuantumRegister(2, "q")
    creg = ClassicalRegister(2, "c")
    circuit = QuantumCircuit(qreg, creg)
    if not hasattr(circuit, "switch"):
        pytest.skip("Qiskit version does not expose QuantumCircuit.switch")

    circuit.measure(qreg[0], creg[0])
    with circuit.switch(creg) as case:
        with case(1):
            circuit.x(qreg[1])
        with case(case.DEFAULT):
            circuit.h(qreg[1])
    circuit.measure(qreg[1], creg[1])

    backend = RocQuantumProvider().get_backend("rocq_simulator")
    result = backend.run(circuit, shots=1, memory=True, statevector=False).result()

    assert result.get_counts() == {"00": 1}
    assert result.get_memory() == ["00"]
    sim = _FakeQuantumSimulator.instances[-1]
    assert sim.ops == [("H", (1,), ())]
    assert sim.measure_qubits == [0, 1]
    assert sim.measurements == []


def test_qiskit_backend_samples_static_for_loop(monkeypatch):
    pytest.importorskip("qiskit")
    _install_fake_binding(monkeypatch)
    _FakeQuantumSimulator.measure_qubit_results = [0, 0]

    from qiskit import QuantumCircuit
    from qiskit_rocquantum_provider import RocQuantumProvider

    circuit = QuantumCircuit(2, 2)
    if not hasattr(circuit, "for_loop"):
        pytest.skip("Qiskit version does not expose QuantumCircuit.for_loop")

    circuit.measure(0, 0)
    with circuit.for_loop(range(2)):
        circuit.x(1)
    circuit.measure(1, 1)

    backend = RocQuantumProvider().get_backend("rocq_simulator")
    result = backend.run(circuit, shots=1, memory=True, statevector=False).result()

    assert result.get_counts() == {"00": 1}
    assert result.get_memory() == ["00"]
    sim = _FakeQuantumSimulator.instances[-1]
    assert sim.ops == [("X", (1,), ()), ("X", (1,), ())]
    assert sim.measure_qubits == [0, 1]
    assert sim.measurements == []


def test_qiskit_backend_samples_parameterized_for_loop(monkeypatch):
    pytest.importorskip("qiskit")
    _install_fake_binding(monkeypatch)
    _FakeQuantumSimulator.measure_qubit_results = [0, 0]

    from qiskit import QuantumCircuit
    from qiskit.circuit import Parameter
    from qiskit_rocquantum_provider import RocQuantumProvider

    circuit = QuantumCircuit(1, 1)
    if not hasattr(circuit, "for_loop"):
        pytest.skip("Qiskit version does not expose QuantumCircuit.for_loop")

    loop_parameter = Parameter("i")
    with circuit.for_loop(range(2), loop_parameter):
        circuit.rx(loop_parameter, 0)
    circuit.measure(0, 0)

    backend = RocQuantumProvider().get_backend("rocq_simulator")
    result = backend.run(circuit, shots=1, memory=True, statevector=False).result()

    assert result.get_counts() == {"0": 1}
    assert result.get_memory() == ["0"]
    sim = _FakeQuantumSimulator.instances[-1]
    assert sim.ops == [("RX", (0,), (0.0,)), ("RX", (0,), (1.0,))]
    assert sim.measure_qubits == [0]
    assert sim.measurements == []


def test_qiskit_backend_samples_break_loop_in_for_loop(monkeypatch):
    pytest.importorskip("qiskit")
    _install_fake_binding(monkeypatch)
    _FakeQuantumSimulator.measure_qubit_results = [0, 0]

    from qiskit import QuantumCircuit
    from qiskit_rocquantum_provider import RocQuantumProvider

    circuit = QuantumCircuit(2, 2)
    if not hasattr(circuit, "for_loop") or not hasattr(circuit, "break_loop"):
        pytest.skip("Qiskit version does not expose loop-control operations")

    with circuit.for_loop(range(3)):
        circuit.x(0)
        circuit.break_loop()
        circuit.x(1)
    circuit.measure([0, 1], [0, 1])

    backend = RocQuantumProvider().get_backend("rocq_simulator")
    result = backend.run(circuit, shots=1, memory=True, statevector=False).result()

    assert result.get_counts() == {"00": 1}
    assert result.get_memory() == ["00"]
    sim = _FakeQuantumSimulator.instances[-1]
    assert sim.ops == [("X", (0,), ())]
    assert sim.measure_qubits == [0, 1]
    assert sim.measurements == []


def test_qiskit_backend_samples_continue_loop_in_for_loop(monkeypatch):
    pytest.importorskip("qiskit")
    _install_fake_binding(monkeypatch)
    _FakeQuantumSimulator.measure_qubit_results = [0, 0]

    from qiskit import QuantumCircuit
    from qiskit_rocquantum_provider import RocQuantumProvider

    circuit = QuantumCircuit(2, 2)
    if not hasattr(circuit, "for_loop") or not hasattr(circuit, "continue_loop"):
        pytest.skip("Qiskit version does not expose loop-control operations")

    with circuit.for_loop(range(3)):
        circuit.x(0)
        circuit.continue_loop()
        circuit.y(1)
    circuit.measure([0, 1], [0, 1])

    backend = RocQuantumProvider().get_backend("rocq_simulator")
    result = backend.run(circuit, shots=1, memory=True, statevector=False).result()

    assert result.get_counts() == {"00": 1}
    assert result.get_memory() == ["00"]
    sim = _FakeQuantumSimulator.instances[-1]
    assert sim.ops == [("X", (0,), ()), ("X", (0,), ()), ("X", (0,), ())]
    assert sim.measure_qubits == [0, 1]
    assert sim.measurements == []


def test_qiskit_backend_rejects_statevector_for_if_else(monkeypatch):
    pytest.importorskip("qiskit")
    _install_fake_binding(monkeypatch)

    from qiskit import QuantumCircuit
    from qiskit_rocquantum_provider import RocQuantumProvider

    circuit = QuantumCircuit(1, 1)
    if not hasattr(circuit, "if_test"):
        pytest.skip("Qiskit version does not expose QuantumCircuit.if_test")

    circuit.measure(0, 0)
    with circuit.if_test((circuit.clbits[0], True)):
        circuit.x(0)

    backend = RocQuantumProvider().get_backend("rocq_simulator")
    with pytest.raises(ValueError, match="shot-trajectory"):
        backend.run(circuit, statevector=True).result()


def test_qiskit_backend_samples_while_loop_until_condition_false(monkeypatch):
    pytest.importorskip("qiskit")
    _install_fake_binding(monkeypatch)
    _FakeQuantumSimulator.measure_qubit_results = [1, 0, 1]

    from qiskit import QuantumCircuit
    from qiskit_rocquantum_provider import RocQuantumProvider

    circuit = QuantumCircuit(2, 2)
    if not hasattr(circuit, "while_loop"):
        pytest.skip("Qiskit version does not expose QuantumCircuit.while_loop")

    circuit.measure(0, 0)
    with circuit.while_loop((circuit.clbits[0], True)):
        circuit.x(1)
        circuit.measure(0, 0)
    circuit.measure(1, 1)

    backend = RocQuantumProvider().get_backend("rocq_simulator")
    result = backend.run(circuit, shots=1, memory=True, statevector=False).result()

    assert result.get_counts() == {"10": 1}
    assert result.get_memory() == ["10"]
    sim = _FakeQuantumSimulator.instances[-1]
    assert sim.ops == [("X", (1,), ())]
    assert sim.measure_qubits == [0, 0, 1]
    assert sim.measurements == []


def test_qiskit_backend_limits_while_loop_iterations(monkeypatch):
    pytest.importorskip("qiskit")
    _install_fake_binding(monkeypatch)
    _FakeQuantumSimulator.measure_qubit_results = [1]

    from qiskit import QuantumCircuit
    from qiskit_rocquantum_provider import RocQuantumProvider

    circuit = QuantumCircuit(1, 1)
    if not hasattr(circuit, "while_loop"):
        pytest.skip("Qiskit version does not expose QuantumCircuit.while_loop")

    circuit.measure(0, 0)
    with circuit.while_loop((circuit.clbits[0], True)):
        circuit.x(0)

    backend = RocQuantumProvider().get_backend("rocq_simulator")
    with pytest.raises(RuntimeError, match="max_dynamic_loop_iterations"):
        backend.run(
            circuit,
            shots=1,
            memory=True,
            statevector=False,
            max_dynamic_loop_iterations=2,
        ).result()


def test_qiskit_backend_applies_global_phase_for_statevector(monkeypatch):
    pytest.importorskip("qiskit")
    _install_fake_binding(monkeypatch)

    from qiskit import QuantumCircuit
    from qiskit_rocquantum_provider import RocQuantumProvider

    backend = RocQuantumProvider().get_backend("rocq_simulator")
    circuit = QuantumCircuit(1)
    circuit.global_phase = math.pi / 3
    circuit.h(0)

    backend.run(circuit, sampling=False, statevector=True).result()

    matrix, targets = _FakeQuantumSimulator.instances[-1].matrices[0]
    phase = np.exp(1j * math.pi / 3)
    np.testing.assert_allclose(matrix, np.array([[phase, 0.0], [0.0, phase]], dtype=np.complex128))
    assert targets == (0,)
    assert _FakeQuantumSimulator.instances[-1].ops == [("H", (0,), ())]


def test_qiskit_backend_skips_global_phase_for_sampling_only(monkeypatch):
    pytest.importorskip("qiskit")
    _install_fake_binding(monkeypatch)

    from qiskit import QuantumCircuit
    from qiskit_rocquantum_provider import RocQuantumProvider

    backend = RocQuantumProvider().get_backend("rocq_simulator")
    circuit = QuantumCircuit(1, 1)
    circuit.global_phase = math.pi / 3
    circuit.h(0)
    circuit.measure(0, 0)

    backend.run(circuit, statevector=False).result()

    assert _FakeQuantumSimulator.instances[-1].matrices == []
    assert _FakeQuantumSimulator.instances[-1].ops == [("H", (0,), ())]


def test_qiskit_backend_dispatches_controlled_rotations_natively(monkeypatch):
    pytest.importorskip("qiskit")
    _install_fake_binding(monkeypatch)

    from qiskit import QuantumCircuit
    from qiskit_rocquantum_provider import RocQuantumProvider

    backend = RocQuantumProvider().get_backend("rocq_simulator")
    circuit = QuantumCircuit(2)
    circuit.crx(0.1, 0, 1)
    circuit.cry(0.2, 0, 1)
    circuit.crz(0.3, 0, 1)

    assert {"crx", "cry", "crz"}.issubset(set(backend.target.operation_names))

    backend.run(circuit, shots=1).result()

    assert _FakeQuantumSimulator.instances[-1].ops == [
        ("CRX", (0, 1), (0.1,)),
        ("CRY", (0, 1), (0.2,)),
        ("CRZ", (0, 1), (0.3,)),
    ]


def test_qiskit_target_supports_transpile_native_phase_and_matrix_fallback_gates(monkeypatch):
    pytest.importorskip("qiskit")
    _install_fake_binding(monkeypatch)

    from qiskit import QuantumCircuit, transpile
    from qiskit_rocquantum_provider import RocQuantumProvider

    backend = RocQuantumProvider().get_backend("rocq_simulator")
    circuit = QuantumCircuit(2)
    circuit.sx(0)
    circuit.sxdg(1)
    circuit.p(0.2, 0)
    circuit.tdg(0)
    circuit.cp(0.3, 0, 1)
    circuit.rxx(0.4, 0, 1)
    circuit.ryy(0.5, 0, 1)
    circuit.rzz(0.6, 0, 1)
    circuit.rzx(0.65, 0, 1)
    circuit.u(0.7, 0.8, 0.9, 1)

    assert backend.target.num_qubits >= 2
    assert {"sx", "tdg", "p", "cp", "rxx", "ryy", "rzz", "rzx", "u"}.issubset(
        set(backend.target.operation_names)
    )

    transpiled = transpile(circuit, backend)
    assert transpiled.num_qubits == 2

    backend.run(circuit, shots=1).result()

    sim = _FakeQuantumSimulator.instances[-1]
    assert sim.ops == [
        ("RX", (0,), (np.pi / 2,)),
        ("RX", (1,), (-np.pi / 2,)),
        ("P", (0,), (0.2,)),
        ("TDG", (0,), ()),
        ("CP", (0, 1), (0.3,)),
        ("CNOT", (0, 1), ()),
        ("RX", (0,), (0.4,)),
        ("CNOT", (0, 1), ()),
        ("RX", (0,), (np.pi / 2,)),
        ("RX", (1,), (np.pi / 2,)),
        ("CNOT", (0, 1), ()),
        ("RZ", (1,), (0.5,)),
        ("CNOT", (0, 1), ()),
        ("RX", (0,), (-np.pi / 2,)),
        ("RX", (1,), (-np.pi / 2,)),
        ("CNOT", (0, 1), ()),
        ("RZ", (1,), (0.6,)),
        ("CNOT", (0, 1), ()),
        ("H", (1,), ()),
        ("CNOT", (0, 1), ()),
        ("RZ", (1,), (0.65,)),
        ("CNOT", (0, 1), ()),
        ("H", (1,), ()),
        ("RZ", (1,), (0.9,)),
        ("RY", (1,), (0.7,)),
        ("RZ", (1,), (0.8,)),
    ]
    assert sim.matrices == []


def test_qiskit_pauli_evolution_decomposes_single_pauli_string_natively(monkeypatch):
    pytest.importorskip("qiskit")
    _install_fake_binding(monkeypatch)

    from qiskit import QuantumCircuit
    from qiskit.circuit.library import PauliEvolutionGate
    from qiskit.quantum_info import SparsePauliOp
    from qiskit_rocquantum_provider import RocQuantumProvider

    backend = RocQuantumProvider().get_backend("rocq_simulator")
    circuit = QuantumCircuit(3)
    circuit.append(
        PauliEvolutionGate(SparsePauliOp.from_list([("XYZ", 1.0)]), time=0.2),
        [0, 1, 2],
    )

    backend.run(circuit, sampling=False).result()

    sim = _FakeQuantumSimulator.instances[-1]
    assert sim.ops == [
        ("RX", (1,), (np.pi / 2,)),
        ("H", (2,), ()),
        ("CNOT", (0, 2), ()),
        ("CNOT", (1, 2), ()),
        ("RZ", (2,), (0.4,)),
        ("CNOT", (1, 2), ()),
        ("CNOT", (0, 2), ()),
        ("RX", (1,), (-np.pi / 2,)),
        ("H", (2,), ()),
    ]
    assert sim.matrices == []


def test_qiskit_pauli_evolution_single_pauli_uses_direct_rotation(monkeypatch):
    pytest.importorskip("qiskit")
    _install_fake_binding(monkeypatch)

    from qiskit import QuantumCircuit
    from qiskit.circuit.library import PauliEvolutionGate
    from qiskit.quantum_info import SparsePauliOp
    from qiskit_rocquantum_provider import RocQuantumProvider

    backend = RocQuantumProvider().get_backend("rocq_simulator")
    circuit = QuantumCircuit(3)
    circuit.append(PauliEvolutionGate(SparsePauliOp.from_list([("XII", 0.5)]), time=0.2), [0, 1, 2])
    circuit.append(PauliEvolutionGate(SparsePauliOp.from_list([("IYI", -1.5)]), time=0.2), [0, 1, 2])
    circuit.append(PauliEvolutionGate(SparsePauliOp.from_list([("IIZ", 2.0)]), time=0.2), [0, 1, 2])

    backend.run(circuit, sampling=False).result()

    sim = _FakeQuantumSimulator.instances[-1]
    assert sim.ops == [
        ("RX", (2,), (0.2,)),
        ("RY", (1,), (-0.6000000000000001,)),
        ("RZ", (0,), (0.8,)),
    ]
    assert sim.matrices == []


def test_qiskit_pauli_evolution_decomposes_commuting_sum_natively(monkeypatch):
    pytest.importorskip("qiskit")
    _install_fake_binding(monkeypatch)

    from qiskit import QuantumCircuit
    from qiskit.circuit.library import PauliEvolutionGate
    from qiskit.quantum_info import SparsePauliOp
    from qiskit_rocquantum_provider import RocQuantumProvider

    backend = RocQuantumProvider().get_backend("rocq_simulator")
    circuit = QuantumCircuit(2)
    circuit.append(
        PauliEvolutionGate(
            SparsePauliOp.from_list([("ZI", 0.5), ("IZ", -1.5)]),
            time=0.2,
        ),
        [0, 1],
    )

    backend.run(circuit, sampling=False).result()

    sim = _FakeQuantumSimulator.instances[-1]
    assert sim.ops == [
        ("RZ", (1,), (0.2,)),
        ("RZ", (0,), (-0.6000000000000001,)),
    ]
    assert sim.matrices == []


def test_qiskit_pauli_evolution_preserves_identity_global_phase_for_statevector(monkeypatch):
    pytest.importorskip("qiskit")
    _install_fake_binding(monkeypatch)

    from qiskit import QuantumCircuit
    from qiskit.circuit.library import PauliEvolutionGate
    from qiskit.quantum_info import SparsePauliOp
    from qiskit_rocquantum_provider import RocQuantumProvider

    backend = RocQuantumProvider().get_backend("rocq_simulator")
    circuit = QuantumCircuit(1)
    circuit.append(PauliEvolutionGate(SparsePauliOp.from_list([("I", 2.0)]), time=0.3), [0])

    backend.run(circuit, sampling=False, statevector=True).result()

    sim = _FakeQuantumSimulator.instances[-1]
    assert sim.ops == []
    assert [(matrix.shape, targets) for matrix, targets in sim.matrices] == [((2, 2), (0,))]
    np.testing.assert_allclose(sim.matrices[0][0], np.eye(2) * np.exp(-0.6j))


def test_qiskit_phase_decomposition_preserves_statevector_global_phase(monkeypatch):
    pytest.importorskip("qiskit")
    _install_fake_binding(monkeypatch)

    from qiskit import QuantumCircuit
    from qiskit_rocquantum_provider import RocQuantumProvider

    backend = RocQuantumProvider().get_backend("rocq_simulator")
    circuit = QuantumCircuit(2)
    circuit.p(0.2, 0)
    circuit.tdg(1)
    circuit.cp(0.3, 0, 1)

    backend.run(circuit, sampling=False, statevector=True).result()

    sim = _FakeQuantumSimulator.instances[-1]
    assert sim.ops == [
        ("P", (0,), (0.2,)),
        ("TDG", (1,), ()),
        ("CP", (0, 1), (0.3,)),
    ]
    assert sim.matrices == []


def test_qiskit_single_qubit_decompositions_preserve_statevector_global_phase(monkeypatch):
    pytest.importorskip("qiskit")
    _install_fake_binding(monkeypatch)

    from qiskit import QuantumCircuit
    from qiskit_rocquantum_provider import RocQuantumProvider

    backend = RocQuantumProvider().get_backend("rocq_simulator")
    circuit = QuantumCircuit(2)
    circuit.sx(0)
    circuit.sxdg(1)
    circuit.u(0.7, 0.8, 0.9, 1)

    backend.run(circuit, sampling=False, statevector=True).result()

    sim = _FakeQuantumSimulator.instances[-1]
    assert sim.ops == [
        ("RX", (0,), (np.pi / 2,)),
        ("RX", (1,), (-np.pi / 2,)),
        ("RZ", (1,), (0.9,)),
        ("RY", (1,), (0.7,)),
        ("RZ", (1,), (0.8,)),
    ]
    assert [(matrix.shape, targets) for matrix, targets in sim.matrices] == [
        ((2, 2), (0,)),
        ((2, 2), (1,)),
        ((2, 2), (1,)),
    ]


def test_qiskit_native_multi_control_and_matrix_fallback_gates(monkeypatch):
    pytest.importorskip("qiskit")
    _install_fake_binding(monkeypatch)

    from qiskit import QuantumCircuit
    from qiskit.circuit.library import (
        CCXGate,
        CCZGate,
        CHGate,
        CSwapGate,
        CYGate,
        DCXGate,
        ECRGate,
        RCCXGate,
        RC3XGate,
        iSwapGate,
    )
    from qiskit_rocquantum_provider import RocQuantumProvider

    backend = RocQuantumProvider().get_backend("rocq_simulator")
    circuit = QuantumCircuit(4)
    circuit.append(CHGate(), [0, 1])
    circuit.append(CYGate(), [0, 1])
    circuit.append(CCXGate(), [0, 1, 2])
    circuit.append(CCZGate(), [0, 1, 2])
    circuit.append(CSwapGate(), [0, 1, 2])
    circuit.append(ECRGate(), [0, 1])
    circuit.append(iSwapGate(), [0, 1])
    circuit.append(DCXGate(), [0, 1])
    circuit.append(RCCXGate(), [0, 1, 2])
    circuit.append(RC3XGate(), [0, 1, 2, 3])

    assert {
        "ccx", "ccz", "ch", "cswap", "cy",
        "dcx", "ecr", "iswap", "rccx", "rcccx",
    }.issubset(set(backend.target.operation_names))

    backend.run(circuit, shots=1, statevector=False).result()

    sim = _FakeQuantumSimulator.instances[-1]
    assert sim.ops == [
        ("RY", (1,), (np.pi / 4,)),
        ("CNOT", (0, 1), ()),
        ("RY", (1,), (-np.pi / 4,)),
        ("SDG", (1,), ()),
        ("CNOT", (0, 1), ()),
        ("S", (1,), ()),
        ("MCX", (0, 1, 2), ()),
        ("H", (2,), ()),
        ("MCX", (0, 1, 2), ()),
        ("H", (2,), ()),
        ("CSWAP", (0, 1, 2), ()),
        ("S", (0,), ()),
        ("RX", (1,), (np.pi / 2,)),
        ("CNOT", (0, 1), ()),
        ("X", (0,), ()),
        ("S", (0,), ()),
        ("S", (1,), ()),
        ("H", (0,), ()),
        ("CNOT", (0, 1), ()),
        ("CNOT", (1, 0), ()),
        ("H", (1,), ()),
        ("CNOT", (0, 1), ()),
        ("CNOT", (1, 0), ()),
        ("H", (2,), ()),
        ("RZ", (2,), (np.pi / 4,)),
        ("CNOT", (1, 2), ()),
        ("RZ", (2,), (-np.pi / 4,)),
        ("CNOT", (0, 2), ()),
        ("RZ", (2,), (np.pi / 4,)),
        ("CNOT", (1, 2), ()),
        ("RZ", (2,), (-np.pi / 4,)),
        ("H", (2,), ()),
        ("H", (3,), ()),
        ("RZ", (3,), (np.pi / 4,)),
        ("CNOT", (2, 3), ()),
        ("RZ", (3,), (-np.pi / 4,)),
        ("H", (3,), ()),
        ("CNOT", (0, 3), ()),
        ("RZ", (3,), (np.pi / 4,)),
        ("CNOT", (1, 3), ()),
        ("RZ", (3,), (-np.pi / 4,)),
        ("CNOT", (0, 3), ()),
        ("RZ", (3,), (np.pi / 4,)),
        ("CNOT", (1, 3), ()),
        ("RZ", (3,), (-np.pi / 4,)),
        ("H", (3,), ()),
        ("RZ", (3,), (np.pi / 4,)),
        ("CNOT", (2, 3), ()),
        ("RZ", (3,), (-np.pi / 4,)),
        ("H", (3,), ()),
    ]
    assert sim.matrices == []


def test_qiskit_backend_dispatches_general_mcx_natively(monkeypatch):
    pytest.importorskip("qiskit")
    _install_fake_binding(monkeypatch)

    from qiskit import QuantumCircuit
    from qiskit_rocquantum_provider import RocQuantumProvider

    backend = RocQuantumProvider().get_backend("rocq_simulator")
    circuit = QuantumCircuit(4)
    circuit.mcx([0, 1, 2], 3)

    assert "mcx" in set(backend.target.operation_names)

    backend.run(circuit, sampling=False).result()

    sim = _FakeQuantumSimulator.instances[-1]
    assert sim.ops == [("MCX", (0, 1, 2, 3), ())]
    assert sim.matrices == []


def test_qiskit_backend_decomposes_open_control_x_y_z_and_h_natively(monkeypatch):
    pytest.importorskip("qiskit")
    _install_fake_binding(monkeypatch)

    from qiskit import QuantumCircuit
    from qiskit.circuit.library import HGate, XGate, YGate, ZGate
    from qiskit_rocquantum_provider import RocQuantumProvider

    backend = RocQuantumProvider().get_backend("rocq_simulator")
    circuit = QuantumCircuit(4)
    circuit.append(XGate().control(2, ctrl_state="01"), [0, 1, 2])
    circuit.append(HGate().control(1, ctrl_state="0"), [0, 1])
    circuit.append(XGate().control(3, ctrl_state="101"), [0, 1, 2, 3])
    circuit.append(YGate().control(1, ctrl_state="0"), [0, 1])
    circuit.append(ZGate().control(1, ctrl_state="0"), [0, 1])
    circuit.append(ZGate().control(2, ctrl_state="01"), [0, 1, 2])

    backend.run(circuit, sampling=False).result()

    sim = _FakeQuantumSimulator.instances[-1]
    assert sim.ops == [
        ("X", (1,), ()),
        ("MCX", (0, 1, 2), ()),
        ("X", (1,), ()),
        ("X", (0,), ()),
        ("RY", (1,), (np.pi / 4,)),
        ("CNOT", (0, 1), ()),
        ("RY", (1,), (-np.pi / 4,)),
        ("X", (0,), ()),
        ("X", (1,), ()),
        ("MCX", (0, 1, 2, 3), ()),
        ("X", (1,), ()),
        ("X", (0,), ()),
        ("SDG", (1,), ()),
        ("CNOT", (0, 1), ()),
        ("S", (1,), ()),
        ("X", (0,), ()),
        ("X", (0,), ()),
        ("CZ", (0, 1), ()),
        ("X", (0,), ()),
        ("X", (1,), ()),
        ("H", (2,), ()),
        ("MCX", (0, 1, 2), ()),
        ("H", (2,), ()),
        ("X", (1,), ()),
    ]
    assert sim.matrices == []


def test_qiskit_backend_decomposes_open_control_parametric_gates_natively(monkeypatch):
    pytest.importorskip("qiskit")
    _install_fake_binding(monkeypatch)

    from qiskit import QuantumCircuit
    from qiskit.circuit.library import PhaseGate, RXGate, RYGate, RZGate
    from qiskit_rocquantum_provider import RocQuantumProvider

    backend = RocQuantumProvider().get_backend("rocq_simulator")
    circuit = QuantumCircuit(2)
    circuit.append(RXGate(0.2).control(1, ctrl_state="0"), [0, 1])
    circuit.append(RYGate(0.3).control(1, ctrl_state="0"), [0, 1])
    circuit.append(RZGate(0.4).control(1, ctrl_state="0"), [0, 1])
    circuit.append(PhaseGate(0.5).control(1, ctrl_state="0"), [0, 1])

    backend.run(circuit, sampling=False).result()

    sim = _FakeQuantumSimulator.instances[-1]
    assert sim.ops == [
        ("X", (0,), ()),
        ("CRX", (0, 1), (0.2,)),
        ("X", (0,), ()),
        ("X", (0,), ()),
        ("CRY", (0, 1), (0.3,)),
        ("X", (0,), ()),
        ("X", (0,), ()),
        ("CRZ", (0, 1), (0.4,)),
        ("X", (0,), ()),
        ("X", (0,), ()),
        ("CP", (0, 1), (0.5,)),
        ("X", (0,), ()),
    ]
    assert sim.matrices == []


def test_qiskit_open_control_phase_preserves_statevector_global_phase(monkeypatch):
    pytest.importorskip("qiskit")
    _install_fake_binding(monkeypatch)

    from qiskit import QuantumCircuit
    from qiskit.circuit.library import PhaseGate
    from qiskit_rocquantum_provider import RocQuantumProvider

    backend = RocQuantumProvider().get_backend("rocq_simulator")
    circuit = QuantumCircuit(2)
    circuit.append(PhaseGate(0.5).control(1, ctrl_state="0"), [0, 1])

    backend.run(circuit, sampling=False, statevector=True).result()

    sim = _FakeQuantumSimulator.instances[-1]
    assert sim.ops == [
        ("X", (0,), ()),
        ("CP", (0, 1), (0.5,)),
        ("X", (0,), ()),
    ]
    assert sim.matrices == []


def test_qiskit_backend_runs_direct_state_preparation(monkeypatch):
    pytest.importorskip("qiskit")
    _install_fake_binding(monkeypatch)

    from qiskit import QuantumCircuit
    from qiskit_rocquantum_provider import RocQuantumProvider

    backend = RocQuantumProvider().get_backend("rocq_simulator")
    circuit = QuantumCircuit(1)
    circuit.prepare_state([0.0, 1.0], 0)

    backend.run(circuit, sampling=False).result()

    sim = _FakeQuantumSimulator.instances[-1]
    np.testing.assert_allclose(sim.statevectors[0], np.array([0.0, 1.0], dtype=np.complex128))
    assert sim.ops == []
    assert sim.matrices == []


def test_qiskit_state_preparation_after_operation_stays_matrix(monkeypatch):
    pytest.importorskip("qiskit")
    _install_fake_binding(monkeypatch)

    from qiskit import QuantumCircuit
    from qiskit_rocquantum_provider import RocQuantumProvider

    backend = RocQuantumProvider().get_backend("rocq_simulator")
    circuit = QuantumCircuit(1)
    circuit.h(0)
    circuit.prepare_state([0.0, 1.0], 0)

    backend.run(circuit, sampling=False).result()

    sim = _FakeQuantumSimulator.instances[-1]
    assert sim.statevectors == []
    assert sim.ops == [("H", (0,), ())]
    assert [targets for _, targets in sim.matrices] == [(0,)]


def test_qiskit_backend_runs_initial_initialize_as_state_preparation(monkeypatch):
    pytest.importorskip("qiskit")
    _install_fake_binding(monkeypatch)

    from qiskit import QuantumCircuit
    from qiskit_rocquantum_provider import RocQuantumProvider

    backend = RocQuantumProvider().get_backend("rocq_simulator")
    circuit = QuantumCircuit(1)
    circuit.initialize([0.0, 1.0], 0)

    backend.run(circuit, sampling=False).result()

    sim = _FakeQuantumSimulator.instances[-1]
    np.testing.assert_allclose(sim.statevectors[0], np.array([0.0, 1.0], dtype=np.complex128))
    assert sim.ops == []
    assert sim.matrices == []


def test_qiskit_backend_rejects_initialize_after_operation(monkeypatch):
    pytest.importorskip("qiskit")
    _install_fake_binding(monkeypatch)

    from qiskit import QuantumCircuit
    from qiskit_rocquantum_provider import RocQuantumProvider

    backend = RocQuantumProvider().get_backend("rocq_simulator")
    circuit = QuantumCircuit(1)
    circuit.x(0)
    circuit.initialize([1.0, 0.0], 0)

    with pytest.raises(ValueError, match="initialize before a qubit has been operated on"):
        backend.run(circuit).result()


def test_qiskit_backend_runs_direct_unitary_without_parameter_normalization(monkeypatch):
    pytest.importorskip("qiskit")
    _install_fake_binding(monkeypatch)

    from qiskit import QuantumCircuit
    from qiskit_rocquantum_provider import RocQuantumProvider

    unitary = np.array([[0, 1], [1, 0]], dtype=np.complex128)
    backend = RocQuantumProvider().get_backend("rocq_simulator")
    circuit = QuantumCircuit(1)
    circuit.unitary(unitary, [0])

    backend.run(circuit, sampling=False).result()

    matrix, targets = _FakeQuantumSimulator.instances[-1].matrices[0]
    np.testing.assert_allclose(matrix, unitary)
    assert targets == (0,)


def test_qiskit_backend_runs_pauli_gate_natively(monkeypatch):
    pytest.importorskip("qiskit")
    _install_fake_binding(monkeypatch)

    from qiskit import QuantumCircuit
    from qiskit.circuit.library import PauliGate
    from qiskit_rocquantum_provider import RocQuantumProvider

    gate = PauliGate("XIZYZ")
    backend = RocQuantumProvider().get_backend("rocq_simulator")
    circuit = QuantumCircuit(5)
    circuit.append(gate, [0, 1, 2, 3, 4])

    backend.run(circuit, sampling=False).result()

    sim = _FakeQuantumSimulator.instances[-1]
    assert sim.ops == [("Z", (0,), ()), ("Y", (1,), ()), ("Z", (2,), ()), ("X", (4,), ())]
    assert sim.matrices == []


def test_qiskit_backend_decomposes_r_gate_natively(monkeypatch):
    pytest.importorskip("qiskit")
    _install_fake_binding(monkeypatch)

    from qiskit import QuantumCircuit
    from qiskit_rocquantum_provider import RocQuantumProvider

    backend = RocQuantumProvider().get_backend("rocq_simulator")
    assert "r" in set(backend.target.operation_names)
    circuit = QuantumCircuit(1)
    circuit.r(0.3, 0.4, 0)

    backend.run(circuit, sampling=False).result()

    sim = _FakeQuantumSimulator.instances[-1]
    assert sim.ops == [
        ("RZ", (0,), (np.pi / 2 - 0.4,)),
        ("RY", (0,), (0.3,)),
        ("RZ", (0,), (0.4 - np.pi / 2,)),
    ]
    assert sim.matrices == []


def test_qiskit_backend_decomposes_xxplusyy_and_xxminusyy_natively(monkeypatch):
    pytest.importorskip("qiskit")
    _install_fake_binding(monkeypatch)

    from qiskit import QuantumCircuit
    from qiskit.circuit.library import XXMinusYYGate, XXPlusYYGate
    from qiskit_rocquantum_provider import RocQuantumProvider

    backend = RocQuantumProvider().get_backend("rocq_simulator")
    assert {"xx_plus_yy", "xx_minus_yy"}.issubset(set(backend.target.operation_names))
    circuit = QuantumCircuit(2)
    circuit.append(XXPlusYYGate(0.4, 0.2), [0, 1])
    circuit.append(XXMinusYYGate(0.6, 0.3), [0, 1])

    backend.run(circuit, sampling=False).result()

    sim = _FakeQuantumSimulator.instances[-1]
    assert sim.matrices == []
    assert ("CNOT", (1, 0), ()) in sim.ops
    assert ("CNOT", (0, 1), ()) in sim.ops
    assert ("RZ", (0,), (0.2,)) in sim.ops
    assert ("RZ", (0,), (-0.2,)) in sim.ops
    assert ("RZ", (1,), (-0.3,)) in sim.ops
    assert ("RZ", (1,), (0.3,)) in sim.ops
    assert ("RY", (1,), (-0.2,)) in sim.ops
    assert ("RY", (0,), (-0.2,)) in sim.ops
    assert ("RY", (0,), (0.3,)) in sim.ops
    assert ("RY", (1,), (-0.3,)) in sim.ops


def test_qiskit_backend_uses_native_controlled_matrix_for_generic_controlled_unitary(monkeypatch):
    pytest.importorskip("qiskit")
    _install_fake_binding(monkeypatch)

    from qiskit import QuantumCircuit
    from qiskit.circuit.library import UnitaryGate
    from qiskit_rocquantum_provider import RocQuantumProvider

    base = np.array([[1.0, 0.0], [0.0, 1.0j]], dtype=np.complex128)
    controlled = UnitaryGate(base).control(1)
    backend = RocQuantumProvider().get_backend("rocq_simulator")
    circuit = QuantumCircuit(2)
    circuit.append(controlled, [0, 1])

    backend.run(circuit, sampling=False).result()

    sim = _FakeQuantumSimulator.instances[-1]
    assert sim.matrices == []
    assert len(sim.controlled_matrices) == 1
    matrix, controls, targets = sim.controlled_matrices[0]
    np.testing.assert_allclose(matrix, base)
    assert controls == (0,)
    assert targets == (1,)


def test_qiskit_provider_exposes_backend_primitives(monkeypatch):
    pytest.importorskip("qiskit")
    _install_fake_binding(monkeypatch)

    from qiskit.primitives import BackendEstimatorV2, BackendSamplerV2
    from qiskit_rocquantum_provider import RocQuantumEstimator, RocQuantumProvider, RocQuantumSampler

    provider = RocQuantumProvider()

    assert isinstance(provider.get_sampler(), RocQuantumSampler)
    assert isinstance(provider.get_sampler(native=False), BackendSamplerV2)
    assert isinstance(provider.get_estimator(), RocQuantumEstimator)
    assert isinstance(provider.get_estimator(native=False), BackendEstimatorV2)


def test_qiskit_provider_primitives_run_with_backend(monkeypatch):
    pytest.importorskip("qiskit")
    _install_fake_binding(monkeypatch)

    from qiskit import QuantumCircuit
    from qiskit.quantum_info import SparsePauliOp
    from qiskit_rocquantum_provider import RocQuantumProvider

    provider = RocQuantumProvider()

    sampler_circuit = QuantumCircuit(1, 1)
    sampler_circuit.h(0)
    sampler_circuit.measure(0, 0)
    sampler_pub = provider.get_sampler().run([sampler_circuit], shots=4).result()[0]

    assert sampler_pub.data.c.num_shots == 4
    assert sampler_pub.data.c.get_counts() == {"0": 4}
    assert sampler_pub.metadata["native"] is True

    estimator_circuit = QuantumCircuit(1)
    observable = SparsePauliOp.from_list([("Z", 1.0)])
    estimator_pub = provider.get_estimator().run(
        [(estimator_circuit, observable)],
    ).result()[0]

    assert float(estimator_pub.data.evs) == pytest.approx(0.5)
    assert float(estimator_pub.data.stds) == pytest.approx(0.0)
    assert estimator_pub.metadata["native"] is True
    assert estimator_pub.metadata["shots"] == 0
    assert _FakeQuantumSimulator.instances[-1].expectations == [("Z", (0,))]


def test_qiskit_native_sampler_deduplicates_repeated_terminal_measurements(monkeypatch):
    pytest.importorskip("qiskit")
    _install_fake_binding(monkeypatch)

    from qiskit import QuantumCircuit
    from qiskit_rocquantum_provider import RocQuantumProvider

    circuit = QuantumCircuit(2, 2)
    circuit.h(0)
    circuit.measure(0, 0)
    circuit.measure(0, 1)

    result = RocQuantumProvider().get_sampler().run([circuit], shots=6).result()[0]

    assert result.data.c.get_counts() == {"00": 3, "11": 3}
    assert _FakeQuantumSimulator.instances[-1].measurements == [((0,), 6)]


def test_qiskit_native_sampler_binds_parameter_values(monkeypatch):
    pytest.importorskip("qiskit")
    _install_fake_binding(monkeypatch)

    from qiskit import QuantumCircuit
    from qiskit.circuit import Parameter
    from qiskit_rocquantum_provider import RocQuantumProvider

    theta = Parameter("theta")
    circuit = QuantumCircuit(1, 1)
    circuit.ry(theta, 0)
    circuit.measure(0, 0)

    result = RocQuantumProvider().get_sampler().run(
        [(circuit, [0.1, 0.2])],
        shots=3,
    ).result()[0]

    assert result.data.c.shape == (2,)
    assert result.data.c.num_shots == 3
    counts = [result.data.c[idx].get_counts() for idx in range(2)]
    assert [sum(row.values()) for row in counts] == [3, 3]
    assert all(set(row).issubset({"0", "1"}) for row in counts)
    assert result.metadata["batched_parameters"] is True
    sim = _FakeQuantumSimulator.instances[-1]
    assert sim.batch_size() == 2
    assert sim.ops == []
    assert sim.batch_ops == [("RY", (0,), (0.1, 0.2))]
    assert sim.measurements == []
    assert sim.probability_requests == [(0,)]


def test_qiskit_native_sampler_batches_controlled_parameter_values(monkeypatch):
    pytest.importorskip("qiskit")
    _install_fake_binding(monkeypatch)

    from qiskit import QuantumCircuit
    from qiskit.circuit import Parameter
    from qiskit_rocquantum_provider import RocQuantumProvider

    theta = Parameter("theta")
    circuit = QuantumCircuit(2, 1)
    circuit.crx(theta, 0, 1)
    circuit.measure(1, 0)

    result = RocQuantumProvider().get_sampler().run(
        [(circuit, [0.1, 0.2])],
        shots=3,
    ).result()[0]

    assert result.data.c.shape == (2,)
    counts = [result.data.c[idx].get_counts() for idx in range(2)]
    assert [sum(row.values()) for row in counts] == [3, 3]
    assert all(set(row).issubset({"0", "1"}) for row in counts)
    assert result.metadata["batched_parameters"] is True
    sim = _FakeQuantumSimulator.instances[-1]
    assert sim.batch_size() == 2
    assert sim.ops == []
    assert sim.batch_ops == [("CRX", (0, 1), (0.1, 0.2))]
    assert sim.measurements == []
    assert sim.probability_requests == [(1,)]


def test_qiskit_native_sampler_batches_static_native_decompositions(monkeypatch):
    pytest.importorskip("qiskit")
    _install_fake_binding(monkeypatch)

    from qiskit import QuantumCircuit
    from qiskit.circuit import Parameter
    from qiskit_rocquantum_provider import RocQuantumProvider

    theta = Parameter("theta")
    circuit = QuantumCircuit(2, 1)
    circuit.ry(theta, 0)
    circuit.sx(1)
    circuit.ch(0, 1)
    circuit.measure(1, 0)

    result = RocQuantumProvider().get_sampler().run(
        [(circuit, [0.1, 0.2])],
        shots=3,
    ).result()[0]

    assert result.data.c.shape == (2,)
    assert result.metadata["batched_parameters"] is True
    sim = _FakeQuantumSimulator.instances[-1]
    assert sim.batch_size() == 2
    assert sim.batch_ops == [("RY", (0,), (0.1, 0.2))]
    assert sim.ops == [
        ("RX", (1,), (np.pi / 2,)),
        ("RY", (1,), (np.pi / 4,)),
        ("CNOT", (0, 1), ()),
        ("RY", (1,), (-np.pi / 4,)),
    ]
    assert sim.matrices == []
    assert sim.measurements == []
    assert sim.probability_requests == [(1,)]


def test_runtime_measure_batch_falls_back_to_batch_probabilities():
    from rocquantum.framework_runtime import RocQuantumRuntime

    class _BatchProbabilitySimulator:
        def __init__(self):
            self.probability_requests = []

        def batch_size(self):
            return 2

        def probabilities_batch(self, qubits):
            self.probability_requests.append(tuple(qubits))
            return np.array([[1.0, 0.0], [0.0, 1.0]], dtype=float)

    sim = _BatchProbabilitySimulator()
    runtime = RocQuantumRuntime(sim)

    samples = runtime.measure_batch([0], 4)

    np.testing.assert_array_equal(samples, np.array([[0, 0, 0, 0], [1, 1, 1, 1]], dtype=np.int64))
    assert sim.probability_requests == [(0,)]


def test_runtime_measure_batch_prefers_native_binding():
    from rocquantum.framework_runtime import RocQuantumRuntime

    class _NativeBatchMeasureSimulator:
        def __init__(self):
            self.calls = []

        def batch_size(self):
            return 2

        def measure_batch(self, qubits, shots):
            self.calls.append((tuple(qubits), int(shots)))
            return np.array([[0, 1, 0], [1, 0, 1]], dtype=np.int64)

    sim = _NativeBatchMeasureSimulator()
    runtime = RocQuantumRuntime(sim)

    np.testing.assert_array_equal(
        runtime.measure_batch([0], 3),
        np.array([[0, 1, 0], [1, 0, 1]], dtype=np.int64),
    )
    assert sim.calls == [((0,), 3)]


def test_qiskit_native_estimator_binds_parameter_values(monkeypatch):
    pytest.importorskip("qiskit")
    _install_fake_binding(monkeypatch)

    from qiskit import QuantumCircuit
    from qiskit.circuit import Parameter
    from qiskit.quantum_info import SparsePauliOp
    from qiskit_rocquantum_provider import RocQuantumProvider

    theta = Parameter("theta")
    circuit = QuantumCircuit(1)
    circuit.ry(theta, 0)
    observable = SparsePauliOp.from_list([("Z", 1.0)])

    result = RocQuantumProvider().get_estimator().run(
        [(circuit, observable, [0.1, 0.2])],
    ).result()[0]

    np.testing.assert_allclose(result.data.evs, np.array([0.5, 0.5]))
    np.testing.assert_allclose(result.data.stds, np.array([0.0, 0.0]))
    assert result.metadata["batched_parameters"] is True
    sim = _FakeQuantumSimulator.instances[-1]
    assert sim.batch_size() == 2
    assert sim.ops == []
    assert sim.batch_ops == [("RY", (0,), (0.1, 0.2))]
    assert sim.batch_expectations == [("Z", (0,))]


def test_qiskit_native_estimator_batches_initial_state_preparation(monkeypatch):
    pytest.importorskip("qiskit")
    _install_fake_binding(monkeypatch)

    from qiskit import QuantumCircuit
    from qiskit.circuit import Parameter
    from qiskit.quantum_info import SparsePauliOp
    from qiskit_rocquantum_provider import RocQuantumProvider

    theta = Parameter("theta")
    circuit = QuantumCircuit(1)
    circuit.prepare_state([0.0, 1.0], [0])
    circuit.ry(theta, 0)
    observable = SparsePauliOp.from_list([("Z", 1.0)])

    result = RocQuantumProvider().get_estimator().run(
        [(circuit, observable, [0.1, 0.2])],
    ).result()[0]

    np.testing.assert_allclose(result.data.evs, np.array([0.5, 0.5]))
    assert result.metadata["batched_parameters"] is True
    sim = _FakeQuantumSimulator.instances[-1]
    assert sim.batch_size() == 2
    np.testing.assert_allclose(
        sim.statevectors[0],
        np.array([[0.0, 1.0], [0.0, 1.0]], dtype=np.complex128),
    )
    assert sim.batch_ops == [("RY", (0,), (0.1, 0.2))]
    assert sim.matrices == []
    assert sim.batch_expectations == [("Z", (0,))]


def test_qiskit_native_estimator_keeps_fixed_unitaries_batched(monkeypatch):
    pytest.importorskip("qiskit")
    _install_fake_binding(monkeypatch)

    from qiskit import QuantumCircuit
    from qiskit.circuit import Parameter
    from qiskit.circuit.library import UnitaryGate
    from qiskit.quantum_info import SparsePauliOp
    from qiskit_rocquantum_provider import RocQuantumProvider

    theta = Parameter("theta")
    unitary = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
    controlled = np.array([[1.0, 0.0], [0.0, 1.0j]], dtype=np.complex128)
    circuit = QuantumCircuit(2)
    circuit.unitary(unitary, [0])
    circuit.append(UnitaryGate(controlled).control(1), [0, 1])
    circuit.ry(theta, 1)
    observable = SparsePauliOp.from_list([("IZ", 1.0)])

    result = RocQuantumProvider().get_estimator().run(
        [(circuit, observable, [0.2, 0.8])],
    ).result()[0]

    np.testing.assert_allclose(result.data.evs, np.array([0.5, 0.5]))
    assert result.metadata["batched_parameters"] is True
    sim = _FakeQuantumSimulator.instances[-1]
    assert sim.batch_size() == 2
    assert len(sim.matrices) == 1
    matrix, targets = sim.matrices[0]
    np.testing.assert_allclose(matrix, unitary)
    assert targets == (0,)
    assert len(sim.controlled_matrices) == 1
    matrix, controls, targets = sim.controlled_matrices[0]
    np.testing.assert_allclose(matrix, controlled)
    assert controls == (0,)
    assert targets == (1,)
    assert sim.batch_ops == [("RY", (1,), (0.2, 0.8))]
    assert sim.batch_expectations == [("Z", (0,))]


def test_qiskit_native_estimator_keeps_fixed_pauli_gates_batched(monkeypatch):
    pytest.importorskip("qiskit")
    _install_fake_binding(monkeypatch)

    from qiskit import QuantumCircuit
    from qiskit.circuit import Parameter
    from qiskit.circuit.library import PauliGate
    from qiskit.quantum_info import SparsePauliOp
    from qiskit_rocquantum_provider import RocQuantumProvider

    theta = Parameter("theta")
    gate = PauliGate("XZ")
    circuit = QuantumCircuit(2)
    circuit.append(gate, [0, 1])
    circuit.ry(theta, 1)
    observable = SparsePauliOp.from_list([("IZ", 1.0)])

    result = RocQuantumProvider().get_estimator().run(
        [(circuit, observable, [0.2, 0.8])],
    ).result()[0]

    np.testing.assert_allclose(result.data.evs, np.array([0.5, 0.5]))
    assert result.metadata["batched_parameters"] is True
    sim = _FakeQuantumSimulator.instances[-1]
    assert sim.batch_size() == 2
    assert sim.ops == [("Z", (0,), ()), ("X", (1,), ())]
    assert sim.matrices == []
    assert sim.batch_ops == [("RY", (1,), (0.2, 0.8))]
    assert sim.batch_expectations == [("Z", (0,))]


def test_qiskit_native_estimator_batches_controlled_parameter_values(monkeypatch):
    pytest.importorskip("qiskit")
    _install_fake_binding(monkeypatch)

    from qiskit import QuantumCircuit
    from qiskit.circuit import Parameter
    from qiskit.quantum_info import SparsePauliOp
    from qiskit_rocquantum_provider import RocQuantumProvider

    theta = Parameter("theta")
    circuit = QuantumCircuit(2)
    circuit.crx(theta, 0, 1)
    observable = SparsePauliOp.from_list([("IZ", 1.0)])

    result = RocQuantumProvider().get_estimator().run(
        [(circuit, observable, [0.1, 0.2])],
    ).result()[0]

    np.testing.assert_allclose(result.data.evs, np.array([0.5, 0.5]))
    assert result.metadata["batched_parameters"] is True
    sim = _FakeQuantumSimulator.instances[-1]
    assert sim.batch_size() == 2
    assert sim.batch_ops == [("CRX", (0, 1), (0.1, 0.2))]
    assert sim.batch_expectations == [("Z", (0,))]


def test_qiskit_native_estimator_batches_open_controlled_parameter_values(monkeypatch):
    pytest.importorskip("qiskit")
    _install_fake_binding(monkeypatch)

    from qiskit import QuantumCircuit
    from qiskit.circuit import Parameter
    from qiskit.circuit.library import CPhaseGate, CRXGate, CRYGate, CRZGate
    from qiskit.quantum_info import SparsePauliOp
    from qiskit_rocquantum_provider import RocQuantumProvider

    theta = Parameter("theta")
    circuit = QuantumCircuit(2)
    circuit.append(CRXGate(theta, ctrl_state=0), [0, 1])
    circuit.append(CRYGate(theta, ctrl_state=0), [0, 1])
    circuit.append(CRZGate(theta, ctrl_state=0), [0, 1])
    circuit.append(CPhaseGate(theta, ctrl_state=0), [0, 1])
    observable = SparsePauliOp.from_list([("IZ", 1.0)])

    result = RocQuantumProvider().get_estimator().run(
        [(circuit, observable, [0.1, 0.2])],
    ).result()[0]

    np.testing.assert_allclose(result.data.evs, np.array([0.5, 0.5]))
    assert result.metadata["batched_parameters"] is True
    sim = _FakeQuantumSimulator.instances[-1]
    assert sim.batch_size() == 2
    assert sim.ops == [("X", (0,), ())] * 8
    assert sim.batch_ops == [
        ("CRX", (0, 1), (0.1, 0.2)),
        ("CRY", (0, 1), (0.1, 0.2)),
        ("CRZ", (0, 1), (0.1, 0.2)),
        ("CP", (0, 1), (0.1, 0.2)),
    ]
    assert sim.matrices == []
    assert sim.batch_expectations == [("Z", (0,))]


def test_qiskit_native_estimator_batches_u_gate_parameters(monkeypatch):
    pytest.importorskip("qiskit")
    _install_fake_binding(monkeypatch)

    from qiskit import QuantumCircuit
    from qiskit.circuit import Parameter
    from qiskit.quantum_info import SparsePauliOp
    from qiskit_rocquantum_provider import RocQuantumProvider

    theta = Parameter("theta")
    phi = Parameter("phi")
    lam = Parameter("lam")
    circuit = QuantumCircuit(1)
    circuit.u(theta, phi, lam, 0)
    observable = SparsePauliOp.from_list([("Z", 1.0)])

    result = RocQuantumProvider().get_estimator().run(
        [(circuit, observable, {(theta, phi, lam): [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]})],
    ).result()[0]

    np.testing.assert_allclose(result.data.evs, np.array([0.5, 0.5]))
    assert result.metadata["batched_parameters"] is True
    sim = _FakeQuantumSimulator.instances[-1]
    assert sim.batch_size() == 2
    assert sim.batch_ops == [
        ("RZ", (0,), (0.3, 0.6)),
        ("RY", (0,), (0.1, 0.4)),
        ("RZ", (0,), (0.2, 0.5)),
    ]
    assert sim.batch_expectations == [("Z", (0,))]


def test_qiskit_native_estimator_batches_r_gate_parameters(monkeypatch):
    pytest.importorskip("qiskit")
    _install_fake_binding(monkeypatch)

    from qiskit import QuantumCircuit
    from qiskit.circuit import Parameter
    from qiskit.quantum_info import SparsePauliOp
    from qiskit_rocquantum_provider import RocQuantumProvider

    theta = Parameter("theta")
    phi = Parameter("phi")
    circuit = QuantumCircuit(1)
    circuit.r(theta, phi, 0)
    observable = SparsePauliOp.from_list([("Z", 1.0)])

    result = RocQuantumProvider().get_estimator().run(
        [(circuit, observable, {(theta, phi): [[0.1, 0.2], [0.4, 0.5]]})],
    ).result()[0]

    np.testing.assert_allclose(result.data.evs, np.array([0.5, 0.5]))
    assert result.metadata["batched_parameters"] is True
    sim = _FakeQuantumSimulator.instances[-1]
    assert sim.batch_size() == 2
    assert len(sim.batch_ops) == 3
    assert sim.batch_ops[0][0:2] == ("RZ", (0,))
    np.testing.assert_allclose(sim.batch_ops[0][2], (np.pi / 2 - 0.2, np.pi / 2 - 0.5))
    assert sim.batch_ops[1] == ("RY", (0,), (0.1, 0.4))
    assert sim.batch_ops[2][0:2] == ("RZ", (0,))
    np.testing.assert_allclose(sim.batch_ops[2][2], (0.2 - np.pi / 2, 0.5 - np.pi / 2))
    assert sim.matrices == []
    assert sim.batch_expectations == [("Z", (0,))]


def test_qiskit_native_estimator_batches_xxplusyy_and_xxminusyy(monkeypatch):
    pytest.importorskip("qiskit")
    _install_fake_binding(monkeypatch)

    from qiskit import QuantumCircuit
    from qiskit.circuit import Parameter
    from qiskit.circuit.library import XXMinusYYGate, XXPlusYYGate
    from qiskit.quantum_info import SparsePauliOp
    from qiskit_rocquantum_provider import RocQuantumProvider

    theta = Parameter("theta")
    beta = Parameter("beta")
    circuit = QuantumCircuit(2)
    circuit.append(XXPlusYYGate(theta, beta), [0, 1])
    circuit.append(XXMinusYYGate(theta, beta), [0, 1])
    observable = SparsePauliOp.from_list([("IZ", 1.0)])

    result = RocQuantumProvider().get_estimator().run(
        [(circuit, observable, {(theta, beta): [[0.2, 0.1], [0.8, 0.4]]})],
    ).result()[0]

    np.testing.assert_allclose(result.data.evs, np.array([0.5, 0.5]))
    assert result.metadata["batched_parameters"] is True
    sim = _FakeQuantumSimulator.instances[-1]
    assert sim.batch_size() == 2
    assert sim.matrices == []
    assert sim.batch_ops[0] == ("RZ", (0,), (0.1, 0.4))
    assert sim.batch_ops[1] == ("RY", (1,), (-0.1, -0.4))
    assert sim.batch_ops[2] == ("RY", (0,), (-0.1, -0.4))
    assert sim.batch_ops[3] == ("RZ", (0,), (-0.1, -0.4))
    assert sim.batch_ops[4] == ("RZ", (1,), (-0.1, -0.4))
    assert sim.batch_ops[5] == ("RY", (0,), (0.1, 0.4))
    assert sim.batch_ops[6] == ("RY", (1,), (-0.1, -0.4))
    assert sim.batch_ops[7] == ("RZ", (1,), (0.1, 0.4))
    assert sim.batch_expectations == [("Z", (0,))]


def test_qiskit_native_estimator_batches_pauli_evolution_time(monkeypatch):
    pytest.importorskip("qiskit")
    _install_fake_binding(monkeypatch)

    from qiskit import QuantumCircuit
    from qiskit.circuit import Parameter
    from qiskit.circuit.library import PauliEvolutionGate
    from qiskit.quantum_info import SparsePauliOp
    from qiskit_rocquantum_provider import RocQuantumProvider

    time = Parameter("time")
    circuit = QuantumCircuit(2)
    circuit.append(
        PauliEvolutionGate(SparsePauliOp.from_list([("ZZ", 0.5)]), time=time),
        [0, 1],
    )
    observable = SparsePauliOp.from_list([("ZI", 1.0)])

    result = RocQuantumProvider().get_estimator().run(
        [(circuit, observable, [0.1, 0.2])],
    ).result()[0]

    np.testing.assert_allclose(result.data.evs, np.array([0.0, 0.0]))
    assert result.metadata["batched_parameters"] is True
    sim = _FakeQuantumSimulator.instances[-1]
    assert sim.batch_size() == 2
    assert sim.ops == [("CNOT", (0, 1), ()), ("CNOT", (0, 1), ())]
    assert sim.batch_ops == [("RZ", (1,), (0.1, 0.2))]
    assert sim.matrices == []
    assert sim.batch_expectations == [("Z", (1,))]


def test_qiskit_native_estimator_batches_parameters_with_observables(monkeypatch):
    pytest.importorskip("qiskit")
    _install_fake_binding(monkeypatch)

    from qiskit import QuantumCircuit
    from qiskit.circuit import Parameter
    from qiskit.quantum_info import SparsePauliOp
    from qiskit_rocquantum_provider import RocQuantumProvider

    theta = Parameter("theta")
    circuit = QuantumCircuit(1)
    circuit.ry(theta, 0)
    observables = [
        SparsePauliOp.from_list([("Z", 1.0)]),
        SparsePauliOp.from_list([("X", 1.0)]),
    ]

    result = RocQuantumProvider().get_estimator().run(
        [(circuit, observables, [0.1, 0.2])],
    ).result()[0]

    np.testing.assert_allclose(result.data.evs, np.array([0.5, 0.5]))
    np.testing.assert_allclose(result.data.stds, np.array([0.0, 0.0]))
    assert result.metadata["batched_parameters"] is True
    sim = _FakeQuantumSimulator.instances[-1]
    assert sim.batch_size() == 2
    assert sim.ops == []
    assert sim.batch_ops == [("RY", (0,), (0.1, 0.2))]
    assert sim.batch_expectations == [("Z", (0,)), ("X", (0,))]


def test_qiskit_native_estimator_batches_parametric_values(monkeypatch):
    pytest.importorskip("qiskit")
    _install_fake_binding(monkeypatch)

    from qiskit import QuantumCircuit
    from qiskit.circuit import Parameter
    from qiskit.quantum_info import SparsePauliOp
    from qiskit_rocquantum_provider import RocQuantumProvider

    theta = Parameter("theta")
    circuit = QuantumCircuit(2)
    circuit.p(theta, 0)
    circuit.cp(theta, 0, 1)
    circuit.rxx(theta, 0, 1)
    circuit.ryy(theta, 0, 1)
    circuit.rzz(theta, 0, 1)
    circuit.rzx(theta, 0, 1)
    observable = SparsePauliOp.from_list([("IZ", 1.0)])

    result = RocQuantumProvider().get_estimator().run(
        [(circuit, observable, [0.1, 0.2])],
    ).result()[0]

    np.testing.assert_allclose(result.data.evs, np.array([0.5, 0.5]))
    assert result.metadata["batched_parameters"] is True
    sim = _FakeQuantumSimulator.instances[-1]
    assert sim.batch_size() == 2
    assert ("P", (0,), (0.1, 0.2)) in sim.batch_ops
    assert ("CP", (0, 1), (0.1, 0.2)) in sim.batch_ops
    assert ("RX", (0,), (0.1, 0.2)) in sim.batch_ops
    assert sim.batch_ops.count(("RZ", (1,), (0.1, 0.2))) == 3
    assert sim.matrices == []
    assert sim.batch_expectations == [("Z", (0,))]


def test_qiskit_pauli_estimator_combines_batched_expectations(monkeypatch):
    pytest.importorskip("qiskit")
    _install_fake_binding(monkeypatch)

    from qiskit.quantum_info import SparsePauliOp
    from qiskit_rocquantum_provider.estimator import estimate_pauli_observable_batch

    class BatchRuntime:
        def __init__(self):
            self.calls = []

        def batch_size(self):
            return 2

        def expectation_pauli_string_batch(self, pauli_string, targets):
            self.calls.append((pauli_string, tuple(targets)))
            if pauli_string == "Z" and tuple(targets) == (0,):
                return np.array([0.25, -0.5], dtype=float)
            raise AssertionError((pauli_string, targets))

    runtime = BatchRuntime()
    observable = SparsePauliOp.from_list([("Z", 2.0), ("I", 1.0)])

    np.testing.assert_allclose(
        estimate_pauli_observable_batch(runtime, observable, 1),
        np.array([1.5, 0.0], dtype=float),
    )
    assert runtime.calls == [("Z", (0,))]


def test_qiskit_estimator_dense_operator_batch_helper_uses_matrix_batch(monkeypatch):
    pytest.importorskip("qiskit")
    _install_fake_binding(monkeypatch)

    from qiskit.quantum_info import Operator
    from qiskit_rocquantum_provider.estimator import estimate_observable_batch

    class BatchRuntime:
        def __init__(self):
            self.calls = []

        def expectation_matrix_batch(self, matrix, targets):
            self.calls.append((np.asarray(matrix, dtype=np.complex128), tuple(targets)))
            return np.array([1.0 + 0.0j, -1.0 + 0.0j], dtype=np.complex128)

    runtime = BatchRuntime()
    observable = Operator(np.diag([1.0, -1.0]).astype(np.complex128))

    np.testing.assert_allclose(
        estimate_observable_batch(runtime, observable, 1),
        np.array([1.0, -1.0], dtype=float),
    )
    assert runtime.calls[0][1] == (0,)
    np.testing.assert_allclose(runtime.calls[0][0], np.diag([1.0, -1.0]))


def test_qiskit_estimator_partial_dense_operator_uses_matrix_targets(monkeypatch):
    pytest.importorskip("qiskit")
    _install_fake_binding(monkeypatch)

    from qiskit.quantum_info import Operator
    from qiskit_rocquantum_provider.estimator import estimate_observable

    class Runtime:
        def __init__(self):
            self.calls = []

        def expectation_matrix(self, matrix, targets):
            self.calls.append((np.asarray(matrix, dtype=np.complex128), tuple(targets)))
            return 0.25 + 0.0j

    runtime = Runtime()
    observable = Operator(np.diag([1.0, -0.5]).astype(np.complex128))

    assert estimate_observable(runtime, observable, 2) == pytest.approx(0.25)
    assert runtime.calls[0][1] == (0,)
    np.testing.assert_allclose(runtime.calls[0][0], np.diag([1.0, -0.5]))


def test_qiskit_native_estimator_accepts_dense_operator(monkeypatch):
    pytest.importorskip("qiskit")
    _install_fake_binding(monkeypatch)
    monkeypatch.setattr(_FakeQuantumSimulator, "enable_matrix_expectation", True)

    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Operator
    from qiskit_rocquantum_provider import RocQuantumProvider

    circuit = QuantumCircuit(1)
    observable = Operator(np.diag([1.0, -1.0]).astype(np.complex128))

    result = RocQuantumProvider().get_estimator().run(
        [(circuit, observable)],
    ).result()[0]

    assert float(result.data.evs) == pytest.approx(1.0)
    assert result.metadata["native"] is True
    sim = _FakeQuantumSimulator.instances[-1]
    assert sim.matrix_expectations[0][1] == (0,)
    assert sim.expectations == []
    assert sim.statevector_reads == 0


def test_qiskit_native_estimator_accepts_partial_dense_operator(monkeypatch):
    pytest.importorskip("qiskit")
    _install_fake_binding(monkeypatch)
    monkeypatch.setattr(_FakeQuantumSimulator, "enable_matrix_expectation", True)

    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Operator
    from qiskit_rocquantum_provider import RocQuantumProvider

    circuit = QuantumCircuit(2)
    observable = Operator(np.diag([1.0, -0.5]).astype(np.complex128))

    result = RocQuantumProvider().get_estimator().run(
        [(circuit, observable)],
    ).result()[0]

    assert float(result.data.evs) == pytest.approx(0.25)
    sim = _FakeQuantumSimulator.instances[-1]
    assert sim.matrix_expectations[0][1] == (0,)
    assert sim.expectations == []
    assert sim.statevector_reads == 0


def test_qiskit_native_estimator_batches_dense_operator_parameters(monkeypatch):
    pytest.importorskip("qiskit")
    _install_fake_binding(monkeypatch)
    monkeypatch.setattr(_FakeQuantumSimulator, "enable_matrix_expectation", True)

    from qiskit import QuantumCircuit
    from qiskit.circuit import Parameter
    from qiskit.quantum_info import Operator
    from qiskit_rocquantum_provider import RocQuantumProvider

    theta = Parameter("theta")
    circuit = QuantumCircuit(1)
    circuit.ry(theta, 0)
    observable = Operator(np.diag([1.0, -1.0]).astype(np.complex128))

    result = RocQuantumProvider().get_estimator().run(
        [(circuit, observable, [0.1, 0.2])],
    ).result()[0]

    np.testing.assert_allclose(result.data.evs, np.array([1.0, 1.0], dtype=float))
    assert result.metadata["batched_parameters"] is True
    sim = _FakeQuantumSimulator.instances[-1]
    assert sim.batch_size() == 2
    assert sim.batch_ops == [("RY", (0,), (0.1, 0.2))]
    assert sim.matrix_batch_expectations[0][1] == (0,)
    assert sim.matrix_expectations == []


def test_qiskit_native_estimator_reuses_bound_circuit_for_observable_batch(monkeypatch):
    pytest.importorskip("qiskit")
    _install_fake_binding(monkeypatch)

    from qiskit import QuantumCircuit
    from qiskit.quantum_info import SparsePauliOp
    from qiskit_rocquantum_provider import RocQuantumProvider

    circuit = QuantumCircuit(1)
    circuit.h(0)
    observables = [
        SparsePauliOp.from_list([("Z", 1.0)]),
        SparsePauliOp.from_list([("X", 1.0)]),
    ]

    result = RocQuantumProvider().get_estimator().run(
        [(circuit, observables)],
    ).result()[0]

    np.testing.assert_allclose(result.data.evs, np.array([0.5, 0.5]))
    sim = _FakeQuantumSimulator.instances[-1]
    assert sim.ops == [("H", (0,), ())]
    assert sim.total_gate_applications == 1
    assert sim.expectations == [("Z", (0,)), ("X", (0,))]


def test_qiskit_native_estimator_reuses_duplicate_observable_batch_terms(monkeypatch):
    pytest.importorskip("qiskit")
    _install_fake_binding(monkeypatch)

    from qiskit import QuantumCircuit
    from qiskit.quantum_info import SparsePauliOp
    from qiskit_rocquantum_provider import RocQuantumProvider

    circuit = QuantumCircuit(1)
    circuit.h(0)
    observables = [
        SparsePauliOp.from_list([("Z", 0.5), ("Z", 0.7)]),
        SparsePauliOp.from_list([("Z", 1.2)]),
        SparsePauliOp.from_list([("X", 1.0)]),
    ]

    result = RocQuantumProvider().get_estimator().run(
        [(circuit, observables)],
    ).result()[0]

    np.testing.assert_allclose(result.data.evs, np.array([0.6, 0.6, 0.5]))
    sim = _FakeQuantumSimulator.instances[-1]
    assert sim.ops == [("H", (0,), ())]
    assert sim.total_gate_applications == 1
    assert sim.expectations == [("Z", (0,)), ("X", (0,))]


def test_qiskit_provider_estimates_dense_operator_natively(monkeypatch):
    pytest.importorskip("qiskit")
    _install_fake_binding(monkeypatch)
    monkeypatch.setattr(_FakeQuantumSimulator, "enable_matrix_expectation", True)

    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Operator
    from qiskit_rocquantum_provider import RocQuantumProvider

    provider = RocQuantumProvider()
    circuit = QuantumCircuit(2)
    observable = Operator(np.diag([1.0, -1.0, -1.0, 1.0]).astype(np.complex128))

    assert provider.estimate_expectation(circuit, observable) == pytest.approx(1.0)
    sim = _FakeQuantumSimulator.instances[-1]
    assert sim.matrix_expectations[0][1] == (0, 1)
    assert sim.expectations == []
    assert sim.statevector_reads == 0


def test_qiskit_provider_estimates_partial_dense_operator_natively(monkeypatch):
    pytest.importorskip("qiskit")
    _install_fake_binding(monkeypatch)
    monkeypatch.setattr(_FakeQuantumSimulator, "enable_matrix_expectation", True)

    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Operator
    from qiskit_rocquantum_provider import RocQuantumProvider

    provider = RocQuantumProvider()
    circuit = QuantumCircuit(2)
    observable = Operator(np.diag([1.0, -0.5]).astype(np.complex128))

    assert provider.estimate_expectation(circuit, observable) == pytest.approx(0.25)
    sim = _FakeQuantumSimulator.instances[-1]
    assert sim.matrix_expectations[0][1] == (0,)
    assert sim.expectations == []
    assert sim.statevector_reads == 0


def test_qiskit_provider_estimates_sparse_pauli_observable_natively(monkeypatch):
    pytest.importorskip("qiskit")
    _install_fake_binding(monkeypatch)

    from qiskit import QuantumCircuit
    from qiskit.quantum_info import SparsePauliOp
    from qiskit_rocquantum_provider import RocQuantumProvider

    provider = RocQuantumProvider()
    circuit = QuantumCircuit(2)
    circuit.h(0)

    observable = SparsePauliOp.from_list([
        ("IZ", 1.2),
        ("ZX", -0.5),
        ("II", 0.25),
    ])

    assert provider.estimate_expectation(circuit, observable) == pytest.approx(0.725)
    assert _FakeQuantumSimulator.instances[-1].expectations == [
        ("Z", (0,)),
        ("XZ", (0, 1)),
    ]


def test_qiskit_provider_combines_duplicate_pauli_terms(monkeypatch):
    pytest.importorskip("qiskit")
    _install_fake_binding(monkeypatch)

    from qiskit import QuantumCircuit
    from qiskit.quantum_info import SparsePauliOp
    from qiskit_rocquantum_provider import RocQuantumProvider

    provider = RocQuantumProvider()
    circuit = QuantumCircuit(1)
    observable = SparsePauliOp.from_list([
        ("Z", 0.5),
        ("Z", 0.7),
        ("I", 0.25),
    ])

    assert provider.estimate_expectation(circuit, observable) == pytest.approx(0.85)
    assert _FakeQuantumSimulator.instances[-1].expectations == [("Z", (0,))]


def test_qiskit_provider_combines_padded_duplicate_pauli_labels(monkeypatch):
    pytest.importorskip("qiskit")
    _install_fake_binding(monkeypatch)

    from qiskit import QuantumCircuit
    from qiskit_rocquantum_provider import RocQuantumProvider

    provider = RocQuantumProvider()
    circuit = QuantumCircuit(2)
    observable = [("Z", 0.5), ("IZ", 0.7), ("II", 0.25)]

    assert provider.estimate_expectation(circuit, observable) == pytest.approx(0.85)
    assert _FakeQuantumSimulator.instances[-1].expectations == [("Z", (0,))]


def test_qiskit_provider_estimates_sparse_observable_natively(monkeypatch):
    pytest.importorskip("qiskit")
    _install_fake_binding(monkeypatch)

    from qiskit import QuantumCircuit
    from qiskit.quantum_info import SparseObservable
    from qiskit_rocquantum_provider import RocQuantumProvider

    provider = RocQuantumProvider()
    circuit = QuantumCircuit(2)
    observable = SparseObservable.from_list([
        ("IZ", 1.2),
        ("XX", -0.5),
    ])

    assert provider.estimate_expectation(circuit, observable) == pytest.approx(0.475)
    assert _FakeQuantumSimulator.instances[-1].expectations == [
        ("Z", (0,)),
        ("XX", (0, 1)),
    ]


def test_pennylane_plugin_aliases_load_with_real_pennylane(monkeypatch):
    pytest.importorskip("pennylane")
    _install_fake_binding(monkeypatch)
    for name in list(sys.modules):
        if name.startswith("pennylane_rocq"):
            sys.modules.pop(name)

    import pennylane as qml

    dev = qml.device("lightning.rocq", wires=2)

    @qml.qnode(dev)
    def circuit():
        qml.Hadamard(wires=0)
        qml.CNOT(wires=[0, 1])
        return qml.state()

    state = circuit()
    expected = np.array([1.0 / math.sqrt(2.0), 0.0, 0.0, 1.0 / math.sqrt(2.0)])

    np.testing.assert_allclose(state, expected)

    alias_dev = qml.device("rocq.pennylane", wires=1)
    assert alias_dev.short_name == "rocq.pennylane"

    rocm_alias = qml.device("lightning.rocm", wires=1)
    assert rocm_alias.short_name == "lightning.rocm"


def test_pennylane_expval_uses_native_pauli_expectation(monkeypatch):
    pytest.importorskip("pennylane")
    _install_fake_binding(monkeypatch)
    for name in list(sys.modules):
        if name.startswith("pennylane_rocq"):
            sys.modules.pop(name)

    import pennylane as qml

    dev = qml.device("lightning.rocq", wires=1)

    @qml.qnode(dev)
    def circuit():
        qml.RY(0.123, wires=0)
        return qml.expval(qml.PauliZ(0))

    assert circuit() == 0.5
    sim = _FakeQuantumSimulator.instances[-1]
    assert sim.expectations == [("Z", (0,))]
    assert sim.statevector_reads == 0


def test_pennylane_var_uses_native_pauli_expectation(monkeypatch):
    pytest.importorskip("pennylane")
    _install_fake_binding(monkeypatch)
    for name in list(sys.modules):
        if name.startswith("pennylane_rocq"):
            sys.modules.pop(name)

    import pennylane as qml

    dev = qml.device("lightning.rocq", wires=1)

    @qml.qnode(dev)
    def circuit():
        qml.RY(0.123, wires=0)
        return qml.var(qml.PauliZ(0))

    assert circuit() == pytest.approx(0.75)
    sim = _FakeQuantumSimulator.instances[-1]
    assert sim.expectations == [("Z", (0,))]
    assert sim.statevector_reads == 0


def test_pennylane_pauli_terms_combine_batched_expectations(monkeypatch):
    pytest.importorskip("pennylane")
    _install_fake_binding(monkeypatch)
    for name in list(sys.modules):
        if name.startswith("pennylane_rocq"):
            sys.modules.pop(name)

    from pennylane_rocq.rocq_device import _evaluate_pauli_terms_batch

    class BatchRuntime:
        def __init__(self):
            self.calls = []

        def batch_size(self):
            return 2

        def expectation_pauli_string_batch(self, pauli_string, targets):
            self.calls.append((pauli_string, tuple(targets)))
            if pauli_string == "Z" and tuple(targets) == (0,):
                return np.array([0.25, -0.5], dtype=float)
            raise AssertionError((pauli_string, targets))

    runtime = BatchRuntime()

    np.testing.assert_allclose(
        _evaluate_pauli_terms_batch(
            runtime,
            [
                (1.0, "Z", [0]),
                (0.5, "Z", [0]),
                (2.0, "", []),
            ],
        ),
        np.array([2.375, 1.25], dtype=np.complex128),
    )
    assert runtime.calls == [("Z", (0,))]


def test_pennylane_batch_execute_uses_batched_parametric_gate(monkeypatch):
    pytest.importorskip("pennylane")
    _install_fake_binding(monkeypatch)
    for name in list(sys.modules):
        if name.startswith("pennylane_rocq"):
            sys.modules.pop(name)

    import pennylane as qml

    dev = qml.device("lightning.rocq", wires=1)
    circuits = [
        qml.tape.QuantumScript([qml.RY(0.1, wires=0)], [qml.expval(qml.PauliZ(0))]),
        qml.tape.QuantumScript([qml.RY(0.2, wires=0)], [qml.expval(qml.PauliZ(0))]),
    ]

    assert dev.batch_execute(circuits) == pytest.approx((0.5, 0.5))
    sim = _FakeQuantumSimulator.instances[-1]
    assert sim.batch_size() == 2
    assert sim.batch_ops == [("RY", (0,), (0.1, 0.2))]
    assert sim.batch_expectations == [("Z", (0,))]


def test_pennylane_batch_execute_uses_batched_basis_state_sweep(monkeypatch):
    pytest.importorskip("pennylane")
    _install_fake_binding(monkeypatch)
    for name in list(sys.modules):
        if name.startswith("pennylane_rocq"):
            sys.modules.pop(name)

    import pennylane as qml

    dev = qml.device("lightning.rocq", wires=2)
    circuits = [
        qml.tape.QuantumScript(
            [qml.BasisState(np.array([1, 0]), wires=[0, 1]), qml.RY(0.1, wires=1)],
            [qml.expval(qml.PauliZ(0))],
        ),
        qml.tape.QuantumScript(
            [qml.BasisState(np.array([1, 0]), wires=[0, 1]), qml.RY(0.2, wires=1)],
            [qml.expval(qml.PauliZ(0))],
        ),
    ]

    assert dev.batch_execute(circuits) == pytest.approx((0.5, 0.5))
    sim = _FakeQuantumSimulator.instances[-1]
    assert sim.batch_size() == 2
    assert sim.ops == [("X", (0,), ())]
    assert sim.batch_ops == [("RY", (1,), (0.1, 0.2))]
    assert sim.matrices == []
    assert sim.batch_expectations == [("Z", (0,))]


def test_pennylane_batch_execute_uses_batched_stateprep_sweep(monkeypatch):
    pytest.importorskip("pennylane")
    _install_fake_binding(monkeypatch)
    for name in list(sys.modules):
        if name.startswith("pennylane_rocq"):
            sys.modules.pop(name)

    import pennylane as qml

    dev = qml.device("lightning.rocq", wires=1)
    circuits = [
        qml.tape.QuantumScript(
            [qml.StatePrep(np.array([0.0, 1.0], dtype=np.complex128), wires=[0]), qml.RY(0.1, wires=0)],
            [qml.expval(qml.PauliZ(0))],
        ),
        qml.tape.QuantumScript(
            [qml.StatePrep(np.array([0.0, 1.0], dtype=np.complex128), wires=[0]), qml.RY(0.2, wires=0)],
            [qml.expval(qml.PauliZ(0))],
        ),
    ]

    assert dev.batch_execute(circuits) == pytest.approx((0.5, 0.5))
    sim = _FakeQuantumSimulator.instances[-1]
    assert sim.batch_size() == 2
    np.testing.assert_allclose(
        sim.statevectors[0],
        np.array([[0.0, 1.0], [0.0, 1.0]], dtype=np.complex128),
    )
    assert sim.batch_ops == [("RY", (0,), (0.1, 0.2))]
    assert sim.matrices == []
    assert sim.batch_expectations == [("Z", (0,))]


def test_pennylane_batch_execute_keeps_static_native_decompositions_batched(monkeypatch):
    pytest.importorskip("pennylane")
    _install_fake_binding(monkeypatch)
    for name in list(sys.modules):
        if name.startswith("pennylane_rocq"):
            sys.modules.pop(name)

    import pennylane as qml

    dev = qml.device("lightning.rocq", wires=2)
    circuits = [
        qml.tape.QuantumScript(
            [qml.RY(0.1, wires=0), qml.CH(wires=[0, 1]), qml.ECR(wires=[0, 1])],
            [qml.expval(qml.PauliZ(0))],
        ),
        qml.tape.QuantumScript(
            [qml.RY(0.2, wires=0), qml.CH(wires=[0, 1]), qml.ECR(wires=[0, 1])],
            [qml.expval(qml.PauliZ(0))],
        ),
    ]

    assert dev.batch_execute(circuits) == pytest.approx((0.5, 0.5))
    sim = _FakeQuantumSimulator.instances[-1]
    assert sim.batch_size() == 2
    assert sim.batch_ops == [("RY", (0,), (0.1, 0.2))]
    assert sim.ops == [
        ("RY", (1,), (np.pi / 4,)),
        ("CNOT", (0, 1), ()),
        ("RY", (1,), (-np.pi / 4,)),
        ("Z", (0,), ()),
        ("CNOT", (0, 1), ()),
        ("RX", (1,), (np.pi / 2,)),
        ("RX", (0,), (np.pi / 2,)),
        ("RY", (0,), (np.pi / 2,)),
        ("RX", (0,), (np.pi / 2,)),
    ]
    assert sim.matrices == []
    assert sim.batch_expectations == [("Z", (0,))]


def test_pennylane_batch_execute_keeps_static_qft_batched(monkeypatch):
    pytest.importorskip("pennylane")
    _install_fake_binding(monkeypatch)
    for name in list(sys.modules):
        if name.startswith("pennylane_rocq"):
            sys.modules.pop(name)

    import pennylane as qml

    dev = qml.device("lightning.rocq", wires=2)
    circuits = [
        qml.tape.QuantumScript(
            [qml.QFT(wires=[0, 1]), qml.RY(0.1, wires=0)],
            [qml.expval(qml.PauliZ(0))],
        ),
        qml.tape.QuantumScript(
            [qml.QFT(wires=[0, 1]), qml.RY(0.2, wires=0)],
            [qml.expval(qml.PauliZ(0))],
        ),
    ]

    assert dev.batch_execute(circuits) == pytest.approx((0.5, 0.5))
    sim = _FakeQuantumSimulator.instances[-1]
    assert sim.batch_size() == 2
    assert sim.ops == [
        ("H", (0,), ()),
        ("CP", (1, 0), (np.pi / 2,)),
        ("H", (1,), ()),
        ("SWAP", (0, 1), ()),
    ]
    assert sim.batch_ops == [("RY", (0,), (0.1, 0.2))]
    assert sim.matrices == []
    assert sim.batch_expectations == [("Z", (0,))]


def test_pennylane_batch_execute_skips_global_phase_sweeps(monkeypatch):
    pytest.importorskip("pennylane")
    _install_fake_binding(monkeypatch)
    for name in list(sys.modules):
        if name.startswith("pennylane_rocq"):
            sys.modules.pop(name)

    import pennylane as qml

    dev = qml.device("lightning.rocq", wires=2)
    circuits = [
        qml.tape.QuantumScript(
            [qml.GlobalPhase(0.1, wires=[0, 1]), qml.RY(0.2, wires=0)],
            [qml.expval(qml.PauliZ(0))],
        ),
        qml.tape.QuantumScript(
            [qml.GlobalPhase(0.9, wires=[0, 1]), qml.RY(0.8, wires=0)],
            [qml.expval(qml.PauliZ(0))],
        ),
    ]

    assert dev.batch_execute(circuits) == pytest.approx((0.5, 0.5))
    sim = _FakeQuantumSimulator.instances[-1]
    assert sim.batch_size() == 2
    assert sim.ops == []
    assert sim.batch_ops == [("RY", (0,), (0.2, 0.8))]
    assert sim.matrices == []
    assert sim.batch_expectations == [("Z", (0,))]


def test_pennylane_batch_execute_batches_diagonal_qubit_unitary_sweeps(monkeypatch):
    pytest.importorskip("pennylane")
    _install_fake_binding(monkeypatch)
    for name in list(sys.modules):
        if name.startswith("pennylane_rocq"):
            sys.modules.pop(name)

    import pennylane as qml

    first_diagonal = np.exp(1j * np.array([0.1, 0.3, 0.6, 1.2]))
    second_diagonal = np.exp(1j * np.array([0.2, 0.5, 0.9, 1.4]))
    dev = qml.device("lightning.rocq", wires=2)
    circuits = [
        qml.tape.QuantumScript(
            [qml.DiagonalQubitUnitary(first_diagonal, wires=[0, 1]), qml.RY(0.2, wires=0)],
            [qml.expval(qml.PauliZ(0))],
        ),
        qml.tape.QuantumScript(
            [qml.DiagonalQubitUnitary(second_diagonal, wires=[0, 1]), qml.RY(0.8, wires=0)],
            [qml.expval(qml.PauliZ(0))],
        ),
    ]

    assert dev.batch_execute(circuits) == pytest.approx((0.5, 0.5))
    sim = _FakeQuantumSimulator.instances[-1]
    assert sim.batch_size() == 2
    assert [op[:2] for op in sim.batch_ops] == [
        ("RZ", (0,)),
        ("RZ", (1,)),
        ("RZ", (1,)),
        ("RY", (0,)),
    ]
    np.testing.assert_allclose(sim.batch_ops[0][2], (0.7, 0.8))
    np.testing.assert_allclose(sim.batch_ops[1][2], (0.4, 0.4))
    np.testing.assert_allclose(sim.batch_ops[2][2], (-0.2, -0.1))
    np.testing.assert_allclose(sim.batch_ops[3][2], (0.2, 0.8))
    assert sim.ops == [("CNOT", (0, 1), ()), ("CNOT", (0, 1), ())]
    assert sim.matrices == []
    assert sim.batch_expectations == [("Z", (0,))]


def test_pennylane_batch_execute_batches_select_pauli_rot_sweeps(monkeypatch):
    pytest.importorskip("pennylane")
    _install_fake_binding(monkeypatch)
    for name in list(sys.modules):
        if name.startswith("pennylane_rocq"):
            sys.modules.pop(name)

    import pennylane as qml

    dev = qml.device("lightning.rocq", wires=2)
    circuits = [
        qml.tape.QuantumScript(
            [
                qml.SelectPauliRot(
                    np.array([0.25, 1.25]),
                    control_wires=[0],
                    target_wire=1,
                    rot_axis="X",
                ),
                qml.RY(0.2, wires=0),
            ],
            [qml.expval(qml.PauliZ(0))],
        ),
        qml.tape.QuantumScript(
            [
                qml.SelectPauliRot(
                    np.array([0.75, 1.75]),
                    control_wires=[0],
                    target_wire=1,
                    rot_axis="X",
                ),
                qml.RY(0.8, wires=0),
            ],
            [qml.expval(qml.PauliZ(0))],
        ),
    ]

    assert dev.batch_execute(circuits) == pytest.approx((0.5, 0.5))
    sim = _FakeQuantumSimulator.instances[-1]
    assert sim.batch_size() == 2
    assert sim.ops == [
        ("H", (1,), ()),
        ("CNOT", (0, 1), ()),
        ("CNOT", (0, 1), ()),
        ("H", (1,), ()),
    ]
    assert [op[:2] for op in sim.batch_ops] == [
        ("RZ", (1,)),
        ("RZ", (1,)),
        ("RY", (0,)),
    ]
    np.testing.assert_allclose(sim.batch_ops[0][2], (0.75, 1.25))
    np.testing.assert_allclose(sim.batch_ops[1][2], (-0.5, -0.5))
    np.testing.assert_allclose(sim.batch_ops[2][2], (0.2, 0.8))
    assert sim.matrices == []
    assert sim.batch_expectations == [("Z", (0,))]


def test_pennylane_batch_execute_keeps_permutation_templates_batched(monkeypatch):
    pytest.importorskip("pennylane")
    _install_fake_binding(monkeypatch)
    for name in list(sys.modules):
        if name.startswith("pennylane_rocq"):
            sys.modules.pop(name)

    import pennylane as qml

    dev = qml.device("lightning.rocq", wires=3)
    circuits = [
        qml.tape.QuantumScript(
            [
                qml.BasisEmbedding(np.array([1, 0, 1]), wires=[0, 1, 2]),
                qml.Permute([2, 0, 1], wires=[0, 1, 2]),
                qml.RY(0.2, wires=0),
            ],
            [qml.expval(qml.PauliZ(0))],
        ),
        qml.tape.QuantumScript(
            [
                qml.BasisEmbedding(np.array([1, 0, 1]), wires=[0, 1, 2]),
                qml.Permute([2, 0, 1], wires=[0, 1, 2]),
                qml.RY(0.8, wires=0),
            ],
            [qml.expval(qml.PauliZ(0))],
        ),
    ]

    assert dev.batch_execute(circuits) == pytest.approx((0.5, 0.5))
    sim = _FakeQuantumSimulator.instances[-1]
    assert sim.batch_size() == 2
    assert sim.ops == [
        ("X", (0,), ()),
        ("X", (2,), ()),
        ("SWAP", (0, 2), ()),
        ("SWAP", (1, 2), ()),
    ]
    assert sim.batch_ops == [("RY", (0,), (0.2, 0.8))]
    assert sim.matrices == []
    assert sim.batch_expectations == [("Z", (0,))]


def test_pennylane_batch_execute_batches_controlled_sequence_sweeps(monkeypatch):
    pytest.importorskip("pennylane")
    _install_fake_binding(monkeypatch)
    for name in list(sys.modules):
        if name.startswith("pennylane_rocq"):
            sys.modules.pop(name)

    import pennylane as qml

    dev = qml.device("lightning.rocq", wires=3)
    circuits = [
        qml.tape.QuantumScript(
            [qml.ControlledSequence(qml.RX(0.2, wires=2), control=[0, 1])],
            [qml.expval(qml.PauliZ(0))],
        ),
        qml.tape.QuantumScript(
            [qml.ControlledSequence(qml.RX(0.8, wires=2), control=[0, 1])],
            [qml.expval(qml.PauliZ(0))],
        ),
    ]

    assert dev.batch_execute(circuits) == pytest.approx((0.5, 0.5))
    sim = _FakeQuantumSimulator.instances[-1]
    assert sim.batch_size() == 2
    assert sim.batch_ops == [
        ("CRX", (0, 2), (0.4, 1.6)),
        ("CRX", (1, 2), (0.2, 0.8)),
    ]
    assert sim.ops == []
    assert sim.matrices == []
    assert sim.batch_expectations == [("Z", (0,))]


def test_pennylane_batch_execute_batches_select_sweeps(monkeypatch):
    pytest.importorskip("pennylane")
    _install_fake_binding(monkeypatch)
    for name in list(sys.modules):
        if name.startswith("pennylane_rocq"):
            sys.modules.pop(name)

    import pennylane as qml

    dev = qml.device("lightning.rocq", wires=2)
    circuits = [
        qml.tape.QuantumScript(
            [qml.Select([qml.RX(0.2, wires=1), qml.RY(0.3, wires=1)], control=[0])],
            [qml.expval(qml.PauliZ(0))],
        ),
        qml.tape.QuantumScript(
            [qml.Select([qml.RX(0.8, wires=1), qml.RY(0.9, wires=1)], control=[0])],
            [qml.expval(qml.PauliZ(0))],
        ),
    ]

    assert dev.batch_execute(circuits) == pytest.approx((0.5, 0.5))
    sim = _FakeQuantumSimulator.instances[-1]
    assert sim.batch_size() == 2
    assert sim.ops == [("X", (0,), ()), ("X", (0,), ())]
    assert sim.batch_ops == [
        ("CRX", (0, 1), (0.2, 0.8)),
        ("CRY", (0, 1), (0.3, 0.9)),
    ]
    assert sim.matrices == []
    assert sim.batch_expectations == [("Z", (0,))]


def test_pennylane_batch_execute_keeps_arithmetic_templates_batched(monkeypatch):
    pytest.importorskip("pennylane")
    _install_fake_binding(monkeypatch)
    for name in list(sys.modules):
        if name.startswith("pennylane_rocq"):
            sys.modules.pop(name)

    import pennylane as qml

    dev = qml.device("lightning.rocq", wires=4)
    circuits = [
        qml.tape.QuantumScript(
            [
                qml.QubitSum(wires=[0, 1, 2]),
                qml.QubitCarry(wires=[0, 1, 2, 3]),
                qml.RY(0.2, wires=0),
            ],
            [qml.expval(qml.PauliZ(0))],
        ),
        qml.tape.QuantumScript(
            [
                qml.QubitSum(wires=[0, 1, 2]),
                qml.QubitCarry(wires=[0, 1, 2, 3]),
                qml.RY(0.8, wires=0),
            ],
            [qml.expval(qml.PauliZ(0))],
        ),
    ]

    assert dev.batch_execute(circuits) == pytest.approx((0.5, 0.5))
    sim = _FakeQuantumSimulator.instances[-1]
    assert sim.batch_size() == 2
    assert sim.ops == [
        ("CNOT", (1, 2), ()),
        ("CNOT", (0, 2), ()),
        ("MCX", (1, 2, 3), ()),
        ("CNOT", (1, 2), ()),
        ("MCX", (0, 2, 3), ()),
    ]
    assert sim.batch_ops == [("RY", (0,), (0.2, 0.8))]
    assert sim.matrices == []
    assert sim.batch_expectations == [("Z", (0,))]


def test_pennylane_batch_execute_keeps_grover_operator_batched(monkeypatch):
    pytest.importorskip("pennylane")
    _install_fake_binding(monkeypatch)
    for name in list(sys.modules):
        if name.startswith("pennylane_rocq"):
            sys.modules.pop(name)

    import pennylane as qml

    dev = qml.device("lightning.rocq", wires=3)
    circuits = [
        qml.tape.QuantumScript(
            [qml.GroverOperator(wires=[0, 1, 2]), qml.RY(0.2, wires=0)],
            [qml.expval(qml.PauliZ(0))],
        ),
        qml.tape.QuantumScript(
            [qml.GroverOperator(wires=[0, 1, 2]), qml.RY(0.8, wires=0)],
            [qml.expval(qml.PauliZ(0))],
        ),
    ]

    assert dev.batch_execute(circuits) == pytest.approx((0.5, 0.5))
    sim = _FakeQuantumSimulator.instances[-1]
    assert sim.batch_size() == 2
    assert sim.ops == [
        ("H", (0,), ()),
        ("H", (1,), ()),
        ("Z", (2,), ()),
        ("X", (0,), ()),
        ("X", (1,), ()),
        ("MCX", (0, 1, 2), ()),
        ("X", (1,), ()),
        ("X", (0,), ()),
        ("Z", (2,), ()),
        ("H", (0,), ()),
        ("H", (1,), ()),
    ]
    assert sim.batch_ops == [("RY", (0,), (0.2, 0.8))]
    assert sim.matrices == []
    assert sim.batch_expectations == [("Z", (0,))]


def test_pennylane_batch_execute_keeps_static_unitaries_batched(monkeypatch):
    pytest.importorskip("pennylane")
    _install_fake_binding(monkeypatch)
    for name in list(sys.modules):
        if name.startswith("pennylane_rocq"):
            sys.modules.pop(name)

    import pennylane as qml

    unitary = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
    controlled = np.array([[1.0, 0.0], [0.0, 1.0j]], dtype=np.complex128)
    dev = qml.device("lightning.rocq", wires=2)
    circuits = [
        qml.tape.QuantumScript(
            [
                qml.QubitUnitary(unitary, wires=[0]),
                qml.ControlledQubitUnitary(controlled, wires=[0, 1], control_values=[False]),
                qml.RY(0.2, wires=1),
            ],
            [qml.expval(qml.PauliZ(0))],
        ),
        qml.tape.QuantumScript(
            [
                qml.QubitUnitary(unitary.copy(), wires=[0]),
                qml.ControlledQubitUnitary(controlled.copy(), wires=[0, 1], control_values=[False]),
                qml.RY(0.8, wires=1),
            ],
            [qml.expval(qml.PauliZ(0))],
        ),
    ]

    assert dev.batch_execute(circuits) == pytest.approx((0.5, 0.5))
    sim = _FakeQuantumSimulator.instances[-1]
    assert sim.batch_size() == 2
    assert len(sim.matrices) == 1
    matrix, targets = sim.matrices[0]
    np.testing.assert_allclose(matrix, unitary)
    assert targets == (0,)
    assert sim.ops == [("X", (0,), ()), ("X", (0,), ())]
    assert len(sim.controlled_matrices) == 1
    matrix, controls, targets = sim.controlled_matrices[0]
    np.testing.assert_allclose(matrix, controlled)
    assert controls == (0,)
    assert targets == (1,)
    assert sim.batch_ops == [("RY", (1,), (0.2, 0.8))]
    assert sim.batch_expectations == [("Z", (0,))]


def test_pennylane_batch_execute_keeps_static_block_encode_batched(monkeypatch):
    pytest.importorskip("pennylane")
    _install_fake_binding(monkeypatch)
    for name in list(sys.modules):
        if name.startswith("pennylane_rocq"):
            sys.modules.pop(name)

    import pennylane as qml
    from rocquantum.framework_runtime import matrix_to_little_endian_wires

    block = np.array([[0.2, 0.3], [0.4, 0.1]], dtype=np.complex128)
    dev = qml.device("lightning.rocq", wires=2)
    circuits = [
        qml.tape.QuantumScript(
            [qml.BlockEncode(block, wires=[0, 1]), qml.RY(0.2, wires=0)],
            [qml.expval(qml.PauliZ(0))],
        ),
        qml.tape.QuantumScript(
            [qml.BlockEncode(block.copy(), wires=[0, 1]), qml.RY(0.8, wires=0)],
            [qml.expval(qml.PauliZ(0))],
        ),
    ]

    assert dev.batch_execute(circuits) == pytest.approx((0.5, 0.5))
    sim = _FakeQuantumSimulator.instances[-1]
    assert sim.batch_size() == 2
    assert len(sim.matrices) == 1
    matrix, targets = sim.matrices[0]
    np.testing.assert_allclose(
        matrix,
        matrix_to_little_endian_wires(qml.matrix(qml.BlockEncode(block, wires=[0, 1]))),
    )
    assert targets == (0, 1)
    assert sim.batch_ops == [("RY", (0,), (0.2, 0.8))]
    assert sim.batch_expectations == [("Z", (0,))]


def test_pennylane_batch_execute_uses_batched_multiple_expvals(monkeypatch):
    pytest.importorskip("pennylane")
    _install_fake_binding(monkeypatch)
    for name in list(sys.modules):
        if name.startswith("pennylane_rocq"):
            sys.modules.pop(name)

    import pennylane as qml

    dev = qml.device("lightning.rocq", wires=1)
    circuits = [
        qml.tape.QuantumScript(
            [qml.RY(0.1, wires=0)],
            [qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliX(0))],
        ),
        qml.tape.QuantumScript(
            [qml.RY(0.2, wires=0)],
            [qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliX(0))],
        ),
    ]

    results = dev.batch_execute(circuits)

    assert len(results) == 2
    for result in results:
        np.testing.assert_allclose(result, (0.5, 0.5))
    sim = _FakeQuantumSimulator.instances[-1]
    assert sim.batch_size() == 2
    assert sim.batch_ops == [("RY", (0,), (0.1, 0.2))]
    assert sim.batch_expectations == [("Z", (0,)), ("X", (0,))]


def test_pennylane_batch_execute_uses_batched_variance(monkeypatch):
    pytest.importorskip("pennylane")
    _install_fake_binding(monkeypatch)
    for name in list(sys.modules):
        if name.startswith("pennylane_rocq"):
            sys.modules.pop(name)

    import pennylane as qml

    dev = qml.device("lightning.rocq", wires=1)
    circuits = [
        qml.tape.QuantumScript(
            [qml.RY(0.1, wires=0)],
            [qml.expval(qml.PauliZ(0)), qml.var(qml.PauliZ(0))],
        ),
        qml.tape.QuantumScript(
            [qml.RY(0.2, wires=0)],
            [qml.expval(qml.PauliZ(0)), qml.var(qml.PauliZ(0))],
        ),
    ]

    results = dev.batch_execute(circuits)

    assert len(results) == 2
    for result in results:
        np.testing.assert_allclose(result, (0.5, 0.75))
    sim = _FakeQuantumSimulator.instances[-1]
    assert sim.batch_size() == 2
    assert sim.batch_ops == [("RY", (0,), (0.1, 0.2))]
    assert sim.batch_expectations == [("Z", (0,)), ("Z", (0,))]


def test_pennylane_batch_execute_uses_batched_hermitian_expval(monkeypatch):
    pytest.importorskip("pennylane")
    _install_fake_binding(monkeypatch)
    monkeypatch.setattr(_FakeQuantumSimulator, "enable_matrix_expectation", True)
    for name in list(sys.modules):
        if name.startswith("pennylane_rocq"):
            sys.modules.pop(name)

    import pennylane as qml

    dev = qml.device("lightning.rocq", wires=1)
    observable = qml.Hermitian(np.diag([1.0, -1.0]).astype(np.complex128), wires=0)
    circuits = [
        qml.tape.QuantumScript([qml.RY(0.1, wires=0)], [qml.expval(observable)]),
        qml.tape.QuantumScript([qml.RY(0.2, wires=0)], [qml.expval(observable)]),
    ]

    assert dev.batch_execute(circuits) == pytest.approx((1.0, 1.0))
    sim = _FakeQuantumSimulator.instances[-1]
    assert sim.batch_size() == 2
    assert sim.batch_ops == [("RY", (0,), (0.1, 0.2))]
    assert sim.matrix_batch_expectations[0][1] == (0,)
    np.testing.assert_allclose(sim.matrix_batch_expectations[0][0], np.diag([1.0, -1.0]))
    assert sim.matrix_expectations == []
    assert sim.batch_expectations == []
    assert sim.statevector_reads == 0


def test_pennylane_batch_execute_uses_batched_hermitian_variance(monkeypatch):
    pytest.importorskip("pennylane")
    _install_fake_binding(monkeypatch)
    monkeypatch.setattr(_FakeQuantumSimulator, "enable_matrix_expectation", True)
    for name in list(sys.modules):
        if name.startswith("pennylane_rocq"):
            sys.modules.pop(name)

    import pennylane as qml

    dev = qml.device("lightning.rocq", wires=1)
    observable = qml.Hermitian(np.diag([1.0, -1.0]).astype(np.complex128), wires=0)
    circuits = [
        qml.tape.QuantumScript([qml.RY(0.1, wires=0)], [qml.var(observable)]),
        qml.tape.QuantumScript([qml.RY(0.2, wires=0)], [qml.var(observable)]),
    ]

    assert dev.batch_execute(circuits) == pytest.approx((0.0, 0.0))
    sim = _FakeQuantumSimulator.instances[-1]
    assert sim.batch_size() == 2
    assert sim.batch_ops == [("RY", (0,), (0.1, 0.2))]
    assert len(sim.matrix_batch_expectations) == 2
    np.testing.assert_allclose(sim.matrix_batch_expectations[0][0], np.diag([1.0, -1.0]))
    np.testing.assert_allclose(sim.matrix_batch_expectations[1][0], np.eye(2))
    assert sim.statevector_reads == 0


def test_pennylane_batch_execute_uses_batched_mixed_analytic_measurements(monkeypatch):
    pytest.importorskip("pennylane")
    _install_fake_binding(monkeypatch)
    for name in list(sys.modules):
        if name.startswith("pennylane_rocq"):
            sys.modules.pop(name)

    import pennylane as qml

    dev = qml.device("lightning.rocq", wires=1)
    circuits = [
        qml.tape.QuantumScript(
            [qml.RY(0.1, wires=0)],
            [qml.expval(qml.PauliZ(0)), qml.probs(wires=[0])],
        ),
        qml.tape.QuantumScript(
            [qml.RY(0.2, wires=0)],
            [qml.expval(qml.PauliZ(0)), qml.probs(wires=[0])],
        ),
    ]

    results = dev.batch_execute(circuits)

    assert len(results) == 2
    for expval, probabilities in results:
        assert expval == pytest.approx(0.5)
        np.testing.assert_allclose(probabilities, np.array([1 / 3, 2 / 3]))
    sim = _FakeQuantumSimulator.instances[-1]
    assert sim.batch_size() == 2
    assert sim.batch_ops == [("RY", (0,), (0.1, 0.2))]
    assert sim.batch_expectations == [("Z", (0,))]
    assert sim.probability_requests == [(0,)]


def test_pennylane_batch_execute_uses_batched_multiple_probabilities(monkeypatch):
    pytest.importorskip("pennylane")
    _install_fake_binding(monkeypatch)
    for name in list(sys.modules):
        if name.startswith("pennylane_rocq"):
            sys.modules.pop(name)

    import pennylane as qml

    dev = qml.device("lightning.rocq", wires=2)
    circuits = [
        qml.tape.QuantumScript(
            [qml.CRX(0.1, wires=[0, 1])],
            [qml.probs(wires=[0]), qml.probs(wires=[1])],
        ),
        qml.tape.QuantumScript(
            [qml.CRX(0.2, wires=[0, 1])],
            [qml.probs(wires=[0]), qml.probs(wires=[1])],
        ),
    ]

    results = dev.batch_execute(circuits)

    assert len(results) == 2
    for left_probs, right_probs in results:
        np.testing.assert_allclose(left_probs, np.array([1 / 3, 2 / 3]))
        np.testing.assert_allclose(right_probs, np.array([1 / 3, 2 / 3]))
    sim = _FakeQuantumSimulator.instances[-1]
    assert sim.batch_size() == 2
    assert sim.batch_ops == [("CRX", (0, 1), (0.1, 0.2))]
    assert sim.probability_requests == [(0,), (1,)]


def test_pennylane_batch_execute_uses_batched_controlled_parametric_gate(monkeypatch):
    pytest.importorskip("pennylane")
    _install_fake_binding(monkeypatch)
    for name in list(sys.modules):
        if name.startswith("pennylane_rocq"):
            sys.modules.pop(name)

    import pennylane as qml

    dev = qml.device("lightning.rocq", wires=2)
    circuits = [
        qml.tape.QuantumScript([qml.CRX(0.1, wires=[0, 1])], [qml.expval(qml.PauliZ(0))]),
        qml.tape.QuantumScript([qml.CRX(0.2, wires=[0, 1])], [qml.expval(qml.PauliZ(0))]),
    ]

    assert dev.batch_execute(circuits) == pytest.approx((0.5, 0.5))
    sim = _FakeQuantumSimulator.instances[-1]
    assert sim.batch_size() == 2
    assert sim.batch_ops == [("CRX", (0, 1), (0.1, 0.2))]
    assert sim.batch_expectations == [("Z", (0,))]


def test_pennylane_batch_execute_uses_batched_rot_and_crot(monkeypatch):
    pytest.importorskip("pennylane")
    _install_fake_binding(monkeypatch)
    for name in list(sys.modules):
        if name.startswith("pennylane_rocq"):
            sys.modules.pop(name)

    import pennylane as qml

    dev = qml.device("lightning.rocq", wires=2)
    circuits = [
        qml.tape.QuantumScript(
            [
                qml.Rot(0.1, 0.2, 0.3, wires=0),
                qml.CRot(0.4, 0.5, 0.6, wires=[0, 1]),
            ],
            [qml.expval(qml.PauliZ(0))],
        ),
        qml.tape.QuantumScript(
            [
                qml.Rot(0.7, 0.8, 0.9, wires=0),
                qml.CRot(1.0, 1.1, 1.2, wires=[0, 1]),
            ],
            [qml.expval(qml.PauliZ(0))],
        ),
    ]

    assert dev.batch_execute(circuits) == pytest.approx((0.5, 0.5))
    sim = _FakeQuantumSimulator.instances[-1]
    assert sim.batch_size() == 2
    assert sim.batch_ops == [
        ("RZ", (0,), (0.1, 0.7)),
        ("RY", (0,), (0.2, 0.8)),
        ("RZ", (0,), (0.3, 0.9)),
        ("RZ", (1,), (-0.09999999999999998, -0.09999999999999998)),
        ("RZ", (1,), (-0.5, -1.1)),
        ("RY", (1,), (-0.25, -0.55)),
        ("RY", (1,), (0.25, 0.55)),
        ("RZ", (1,), (0.6, 1.2)),
    ]
    assert sim.ops == [("CNOT", (0, 1), ()), ("CNOT", (0, 1), ())]
    assert sim.batch_expectations == [("Z", (0,))]


def test_pennylane_batch_execute_uses_batched_parametric_gates(monkeypatch):
    pytest.importorskip("pennylane")
    _install_fake_binding(monkeypatch)
    for name in list(sys.modules):
        if name.startswith("pennylane_rocq"):
            sys.modules.pop(name)

    import pennylane as qml

    dev = qml.device("lightning.rocq", wires=2)
    circuits = [
        qml.tape.QuantumScript(
            [
                qml.PhaseShift(0.1, wires=0),
                qml.ControlledPhaseShift(0.2, wires=[0, 1]),
                qml.IsingXX(0.3, wires=[0, 1]),
                qml.IsingYY(0.4, wires=[0, 1]),
                qml.IsingZZ(0.5, wires=[0, 1]),
            ],
            [qml.expval(qml.PauliZ(0))],
        ),
        qml.tape.QuantumScript(
            [
                qml.PhaseShift(0.6, wires=0),
                qml.ControlledPhaseShift(0.7, wires=[0, 1]),
                qml.IsingXX(0.8, wires=[0, 1]),
                qml.IsingYY(0.9, wires=[0, 1]),
                qml.IsingZZ(1.0, wires=[0, 1]),
            ],
            [qml.expval(qml.PauliZ(0))],
        ),
    ]

    assert dev.batch_execute(circuits) == pytest.approx((0.5, 0.5))
    sim = _FakeQuantumSimulator.instances[-1]
    assert sim.batch_size() == 2
    assert ("P", (0,), (0.1, 0.6)) in sim.batch_ops
    assert ("CP", (0, 1), (0.2, 0.7)) in sim.batch_ops
    assert ("RX", (0,), (0.3, 0.8)) in sim.batch_ops
    assert ("RZ", (1,), (0.4, 0.9)) in sim.batch_ops
    assert ("RZ", (1,), (0.5, 1.0)) in sim.batch_ops
    assert sim.matrices == []
    assert sim.batch_expectations == [("Z", (0,))]


def test_pennylane_batch_execute_uses_batched_extended_decompositions(monkeypatch):
    pytest.importorskip("pennylane")
    _install_fake_binding(monkeypatch)
    for name in list(sys.modules):
        if name.startswith("pennylane_rocq"):
            sys.modules.pop(name)

    import pennylane as qml

    dev = qml.device("lightning.rocq", wires=4)
    circuits = [
        qml.tape.QuantumScript(
            [
                qml.MultiRZ(0.1, wires=[0, 1, 2]),
                qml.IsingXY(0.2, wires=[0, 1]),
                qml.SingleExcitation(0.3, wires=[0, 1]),
                qml.DoubleExcitation(0.4, wires=[0, 1, 2, 3]),
                qml.FermionicSWAP(0.5, wires=[0, 1]),
                qml.OrbitalRotation(0.6, wires=[0, 1, 2, 3]),
            ],
            [qml.expval(qml.PauliZ(0))],
        ),
        qml.tape.QuantumScript(
            [
                qml.MultiRZ(0.7, wires=[0, 1, 2]),
                qml.IsingXY(0.8, wires=[0, 1]),
                qml.SingleExcitation(0.9, wires=[0, 1]),
                qml.DoubleExcitation(1.0, wires=[0, 1, 2, 3]),
                qml.FermionicSWAP(1.1, wires=[0, 1]),
                qml.OrbitalRotation(1.2, wires=[0, 1, 2, 3]),
            ],
            [qml.expval(qml.PauliZ(0))],
        ),
    ]

    assert dev.batch_execute(circuits) == pytest.approx((0.5, 0.5))
    sim = _FakeQuantumSimulator.instances[-1]
    assert sim.batch_size() == 2
    assert ("RZ", (2,), (0.1, 0.7)) in sim.batch_ops
    assert ("RY", (0,), (0.1, 0.4)) in sim.batch_ops
    assert ("RX", (1,), (-0.1, -0.4)) in sim.batch_ops
    assert ("RY", (0,), (-0.15, -0.45)) in sim.batch_ops
    assert ("RY", (1,), (-0.15, -0.45)) in sim.batch_ops
    assert ("RY", (1,), (0.05, 0.125)) in sim.batch_ops
    assert ("RZ", (0,), (0.25, 0.55)) in sim.batch_ops
    assert ("RZ", (1,), (0.25, 0.55)) in sim.batch_ops
    assert ("RY", (0,), (-0.3, -0.6)) in sim.batch_ops
    assert ("RY", (3,), (-0.3, -0.6)) in sim.batch_ops
    assert sim.matrices == []
    assert sim.batch_expectations == [("Z", (0,))]


def test_pennylane_batch_execute_uses_batched_paulirot_sweep(monkeypatch):
    pytest.importorskip("pennylane")
    _install_fake_binding(monkeypatch)
    for name in list(sys.modules):
        if name.startswith("pennylane_rocq"):
            sys.modules.pop(name)

    import pennylane as qml

    dev = qml.device("lightning.rocq", wires=3)
    circuits = [
        qml.tape.QuantumScript(
            [qml.PauliRot(0.2, "XYZ", wires=[0, 1, 2])],
            [qml.expval(qml.PauliZ(0))],
        ),
        qml.tape.QuantumScript(
            [qml.PauliRot(0.8, "XYZ", wires=[0, 1, 2])],
            [qml.expval(qml.PauliZ(0))],
        ),
    ]

    assert dev.batch_execute(circuits) == pytest.approx((0.5, 0.5))
    sim = _FakeQuantumSimulator.instances[-1]
    assert sim.batch_size() == 2
    assert sim.ops == [
        ("H", (0,), ()),
        ("RX", (1,), (np.pi / 2,)),
        ("CNOT", (0, 2), ()),
        ("CNOT", (1, 2), ()),
        ("CNOT", (1, 2), ()),
        ("CNOT", (0, 2), ()),
        ("H", (0,), ()),
        ("RX", (1,), (-np.pi / 2,)),
    ]
    assert sim.batch_ops == [("RZ", (2,), (0.2, 0.8))]
    assert sim.matrices == []
    assert sim.batch_expectations == [("Z", (0,))]


def test_pennylane_batch_execute_uses_batched_phase_excitation_variants(monkeypatch):
    pytest.importorskip("pennylane")
    _install_fake_binding(monkeypatch)
    for name in list(sys.modules):
        if name.startswith("pennylane_rocq"):
            sys.modules.pop(name)

    import pennylane as qml

    dev = qml.device("lightning.rocq", wires=4)
    circuits = [
        qml.tape.QuantumScript(
            [
                qml.CPhaseShift00(0.1, wires=[0, 1]),
                qml.CPhaseShift01(0.2, wires=[0, 1]),
                qml.CPhaseShift10(0.3, wires=[0, 1]),
                qml.PSWAP(0.4, wires=[0, 1]),
                qml.SingleExcitationPlus(0.5, wires=[0, 1]),
                qml.SingleExcitationMinus(0.6, wires=[0, 1]),
                qml.DoubleExcitationPlus(0.7, wires=[0, 1, 2, 3]),
                qml.DoubleExcitationMinus(0.8, wires=[0, 1, 2, 3]),
            ],
            [qml.expval(qml.PauliZ(0))],
        ),
        qml.tape.QuantumScript(
            [
                qml.CPhaseShift00(0.9, wires=[0, 1]),
                qml.CPhaseShift01(1.0, wires=[0, 1]),
                qml.CPhaseShift10(1.1, wires=[0, 1]),
                qml.PSWAP(1.2, wires=[0, 1]),
                qml.SingleExcitationPlus(1.3, wires=[0, 1]),
                qml.SingleExcitationMinus(1.4, wires=[0, 1]),
                qml.DoubleExcitationPlus(1.5, wires=[0, 1, 2, 3]),
                qml.DoubleExcitationMinus(1.6, wires=[0, 1, 2, 3]),
            ],
            [qml.expval(qml.PauliZ(0))],
        ),
    ]

    assert dev.batch_execute(circuits) == pytest.approx((0.5, 0.5))
    sim = _FakeQuantumSimulator.instances[-1]
    assert sim.batch_size() == 2
    assert ("CP", (0, 1), (0.1, 0.9)) in sim.batch_ops
    assert ("CP", (0, 1), (0.2, 1.0)) in sim.batch_ops
    assert ("CP", (0, 1), (0.3, 1.1)) in sim.batch_ops
    assert ("P", (1,), (0.4, 1.2)) in sim.batch_ops
    assert ("RY", (0,), (0.25, 0.65)) in sim.batch_ops
    assert ("RZ", (1,), (-0.25, -0.65)) in sim.batch_ops
    assert ("RY", (0,), (0.3, 0.7)) in sim.batch_ops
    assert ("RZ", (1,), (0.3, 0.7)) in sim.batch_ops
    assert ("RZ", (1,), (0.0875, 0.1875)) in sim.batch_ops
    assert ("RZ", (1,), (-0.1, -0.2)) in sim.batch_ops
    assert ("RY", (1,), (0.0875, 0.1875)) in sim.batch_ops
    assert ("RY", (1,), (0.1, 0.2)) in sim.batch_ops
    assert sim.matrices == []
    assert sim.batch_expectations == [("Z", (0,))]


def test_pennylane_batch_execute_uses_batched_probabilities(monkeypatch):
    pytest.importorskip("pennylane")
    _install_fake_binding(monkeypatch)
    for name in list(sys.modules):
        if name.startswith("pennylane_rocq"):
            sys.modules.pop(name)

    import pennylane as qml

    dev = qml.device("lightning.rocq", wires=1)
    circuits = [
        qml.tape.QuantumScript([qml.RY(0.1, wires=0)], [qml.probs(wires=[0])]),
        qml.tape.QuantumScript([qml.RY(0.2, wires=0)], [qml.probs(wires=[0])]),
    ]

    results = dev.batch_execute(circuits)

    assert len(results) == 2
    np.testing.assert_allclose(results[0], np.array([1 / 3, 2 / 3]))
    np.testing.assert_allclose(results[1], np.array([1 / 3, 2 / 3]))
    sim = _FakeQuantumSimulator.instances[-1]
    assert sim.batch_size() == 2
    assert sim.batch_ops == [("RY", (0,), (0.1, 0.2))]
    assert sim.probability_requests == [(0,)]


def test_pennylane_batch_execute_uses_batched_samples(monkeypatch):
    pytest.importorskip("pennylane")
    _install_fake_binding(monkeypatch)
    for name in list(sys.modules):
        if name.startswith("pennylane_rocq"):
            sys.modules.pop(name)

    import pennylane as qml
    from pennylane.exceptions import PennyLaneDeprecationWarning

    with pytest.warns(PennyLaneDeprecationWarning):
        dev = qml.device("lightning.rocq", wires=1, shots=4)
    circuits = [
        qml.tape.QuantumScript([qml.RY(0.1, wires=0)], [qml.sample(wires=[0])]),
        qml.tape.QuantumScript([qml.RY(0.2, wires=0)], [qml.sample(wires=[0])]),
    ]

    results = dev.batch_execute(circuits)

    assert len(results) == 2
    assert all(result.shape == (4,) for result in results)
    assert all(np.isin(result, [0, 1]).all() for result in results)
    sim = _FakeQuantumSimulator.instances[-1]
    assert sim.batch_size() == 2
    assert sim.batch_ops == [("RY", (0,), (0.1, 0.2))]
    assert sim.measurements == []
    assert sim.probability_requests == [(0,)]


def test_pennylane_batch_execute_uses_batched_counts(monkeypatch):
    pytest.importorskip("pennylane")
    _install_fake_binding(monkeypatch)
    for name in list(sys.modules):
        if name.startswith("pennylane_rocq"):
            sys.modules.pop(name)

    import pennylane as qml
    from pennylane.exceptions import PennyLaneDeprecationWarning

    with pytest.warns(PennyLaneDeprecationWarning):
        dev = qml.device("lightning.rocq", wires=1, shots=4)
    circuits = [
        qml.tape.QuantumScript([qml.RY(0.1, wires=0)], [qml.counts(wires=[0])]),
        qml.tape.QuantumScript([qml.RY(0.2, wires=0)], [qml.counts(wires=[0])]),
    ]

    results = dev.batch_execute(circuits)

    assert len(results) == 2
    assert all(sum(int(value) for value in result.values()) == 4 for result in results)
    assert all(set(str(key) for key in result).issubset({"0", "1"}) for result in results)
    sim = _FakeQuantumSimulator.instances[-1]
    assert sim.batch_size() == 2
    assert sim.batch_ops == [("RY", (0,), (0.1, 0.2))]
    assert sim.measurements == []
    assert sim.probability_requests == [(0,)]


def test_pennylane_paulix_expval_skips_diagonalizing_rotation(monkeypatch):
    pytest.importorskip("pennylane")
    _install_fake_binding(monkeypatch)
    for name in list(sys.modules):
        if name.startswith("pennylane_rocq"):
            sys.modules.pop(name)

    import pennylane as qml

    dev = qml.device("lightning.rocq", wires=1)

    @qml.qnode(dev)
    def circuit():
        return qml.expval(qml.PauliX(0))

    assert circuit() == pytest.approx(0.5)
    sim = _FakeQuantumSimulator.instances[-1]
    assert sim.ops == []
    assert sim.expectations == [("X", (0,))]
    assert sim.statevector_reads == 0


def test_pennylane_hermitian_observable_uses_statevector_fallback(monkeypatch):
    pytest.importorskip("pennylane")
    _install_fake_binding(monkeypatch)
    for name in list(sys.modules):
        if name.startswith("pennylane_rocq"):
            sys.modules.pop(name)

    import pennylane as qml

    dev = qml.device("lightning.rocq", wires=1)

    @qml.qnode(dev)
    def hermitian_circuit():
        return qml.expval(qml.Hermitian(np.eye(2), wires=0))

    assert hermitian_circuit() == pytest.approx(1.0)
    sim = _FakeQuantumSimulator.instances[-1]
    assert sim.expectations == []
    assert sim.statevector_reads == 1


def test_pennylane_hermitian_expval_prefers_native_matrix_expectation(monkeypatch):
    pytest.importorskip("pennylane")
    _install_fake_binding(monkeypatch)
    monkeypatch.setattr(_FakeQuantumSimulator, "enable_matrix_expectation", True)
    for name in list(sys.modules):
        if name.startswith("pennylane_rocq"):
            sys.modules.pop(name)

    import pennylane as qml

    dev = qml.device("lightning.rocq", wires=1)
    observable = qml.Hermitian(np.diag([1.0, -1.0]).astype(np.complex128), wires=0)

    @qml.qnode(dev)
    def hermitian_circuit():
        return qml.expval(observable)

    assert hermitian_circuit() == pytest.approx(1.0)
    sim = _FakeQuantumSimulator.instances[-1]
    assert sim.expectations == []
    assert sim.statevector_reads == 0
    assert len(sim.matrix_expectations) == 1
    np.testing.assert_allclose(sim.matrix_expectations[0][0], np.diag([1.0, -1.0]))
    assert sim.matrix_expectations[0][1] == (0,)


def test_pennylane_hermitian_var_prefers_native_matrix_expectation(monkeypatch):
    pytest.importorskip("pennylane")
    _install_fake_binding(monkeypatch)
    monkeypatch.setattr(_FakeQuantumSimulator, "enable_matrix_expectation", True)
    for name in list(sys.modules):
        if name.startswith("pennylane_rocq"):
            sys.modules.pop(name)

    import pennylane as qml

    dev = qml.device("lightning.rocq", wires=1)
    observable = qml.Hermitian(np.diag([1.0, -1.0]).astype(np.complex128), wires=0)

    @qml.qnode(dev)
    def hermitian_circuit():
        return qml.var(observable)

    assert hermitian_circuit() == pytest.approx(0.0)
    sim = _FakeQuantumSimulator.instances[-1]
    assert sim.expectations == []
    assert sim.statevector_reads == 0
    assert len(sim.matrix_expectations) == 2
    np.testing.assert_allclose(sim.matrix_expectations[0][0], np.diag([1.0, -1.0]))
    np.testing.assert_allclose(sim.matrix_expectations[1][0], np.eye(2))


def test_pennylane_projector_expval_uses_native_z_projector_terms(monkeypatch):
    pytest.importorskip("pennylane")
    _install_fake_binding(monkeypatch)
    for name in list(sys.modules):
        if name.startswith("pennylane_rocq"):
            sys.modules.pop(name)

    import pennylane as qml

    dev = qml.device("lightning.rocq", wires=2)

    @qml.qnode(dev)
    def projector_circuit():
        return qml.expval(qml.Projector([1, 0], wires=[0, 1]))

    assert projector_circuit() == pytest.approx(0.125)
    sim = _FakeQuantumSimulator.instances[-1]
    assert sim.expectations == [
        ("Z", (1,)),
        ("Z", (0,)),
        ("ZZ", (0, 1)),
    ]
    assert sim.statevector_reads == 0


def test_pennylane_projector_var_uses_native_z_projector_terms(monkeypatch):
    pytest.importorskip("pennylane")
    _install_fake_binding(monkeypatch)
    for name in list(sys.modules):
        if name.startswith("pennylane_rocq"):
            sys.modules.pop(name)

    import pennylane as qml

    dev = qml.device("lightning.rocq", wires=1)

    @qml.qnode(dev)
    def projector_var_circuit():
        return qml.var(qml.Projector([0], wires=0))

    assert projector_var_circuit() == pytest.approx(0.1875)
    sim = _FakeQuantumSimulator.instances[-1]
    assert sim.expectations == [
        ("Z", (0,)),
        ("Z", (0,)),
    ]
    assert sim.statevector_reads == 0


def test_pennylane_sparse_hamiltonian_uses_sparse_statevector_fallback(monkeypatch):
    pytest.importorskip("pennylane")
    sp = pytest.importorskip("scipy.sparse")
    _install_fake_binding(monkeypatch)
    for name in list(sys.modules):
        if name.startswith("pennylane_rocq"):
            sys.modules.pop(name)

    import pennylane as qml

    hamiltonian_matrix = sp.csr_matrix(np.diag([1.0, -1.0]).astype(np.complex128))
    observable = qml.SparseHamiltonian(hamiltonian_matrix, wires=[0])
    dev = qml.device("lightning.rocq", wires=1)

    @qml.qnode(dev)
    def expval_circuit():
        return qml.expval(observable)

    @qml.qnode(dev)
    def var_circuit():
        return qml.var(observable)

    assert expval_circuit() == pytest.approx(1.0)
    expval_sim = _FakeQuantumSimulator.instances[-1]
    assert expval_sim.expectations == []
    assert expval_sim.statevector_reads == 1

    assert var_circuit() == pytest.approx(0.0)
    var_sim = _FakeQuantumSimulator.instances[-1]
    assert var_sim.expectations == []
    assert var_sim.statevector_reads == 1


def test_pennylane_sparse_hamiltonian_prefers_native_csr_moments(monkeypatch):
    pytest.importorskip("pennylane")
    sp = pytest.importorskip("scipy.sparse")
    _install_fake_binding(monkeypatch)
    monkeypatch.setattr(_FakeQuantumSimulator, "enable_sparse_moments", True)
    for name in list(sys.modules):
        if name.startswith("pennylane_rocq"):
            sys.modules.pop(name)

    import pennylane as qml

    hamiltonian_matrix = sp.csr_matrix(np.diag([1.0, -1.0]).astype(np.complex128))
    observable = qml.SparseHamiltonian(hamiltonian_matrix, wires=[0])
    dev = qml.device("lightning.rocq", wires=1)

    @qml.qnode(dev)
    def expval_circuit():
        return qml.expval(observable)

    @qml.qnode(dev)
    def var_circuit():
        return qml.var(observable)

    assert expval_circuit() == pytest.approx(1.0)
    expval_sim = _FakeQuantumSimulator.instances[-1]
    assert expval_sim.statevector_reads == 0
    assert len(expval_sim.sparse_moments) == 1

    assert var_circuit() == pytest.approx(0.0)
    var_sim = _FakeQuantumSimulator.instances[-1]
    assert var_sim.statevector_reads == 0
    assert len(var_sim.sparse_moments) == 1

    data, indices, indptr, shape = var_sim.sparse_moments[0]
    np.testing.assert_allclose(data, np.array([1.0, -1.0], dtype=np.complex128))
    np.testing.assert_array_equal(indices, np.array([0, 1], dtype=np.int64))
    np.testing.assert_array_equal(indptr, np.array([0, 1, 2], dtype=np.int64))
    assert shape == (2, 2)


def test_pennylane_sparse_hamiltonian_batch_uses_native_csr_moments(monkeypatch):
    pytest.importorskip("pennylane")
    sp = pytest.importorskip("scipy.sparse")
    _install_fake_binding(monkeypatch)
    monkeypatch.setattr(_FakeQuantumSimulator, "enable_sparse_moments", True)
    for name in list(sys.modules):
        if name.startswith("pennylane_rocq"):
            sys.modules.pop(name)

    import pennylane as qml

    hamiltonian_matrix = sp.csr_matrix(np.diag([1.0, -1.0]).astype(np.complex128))
    observable = qml.SparseHamiltonian(hamiltonian_matrix, wires=[0])
    dev = qml.device("lightning.rocq", wires=1)
    tapes = [
        qml.tape.QuantumScript([qml.RY(0.1, wires=0)], [qml.expval(observable)]),
        qml.tape.QuantumScript([qml.RY(0.2, wires=0)], [qml.expval(observable)]),
    ]

    results = dev.batch_execute(tapes)

    assert results == pytest.approx((1.0, 1.0))
    sim = _FakeQuantumSimulator.instances[-1]
    assert sim.batch_size() == 2
    assert sim.batch_ops == [("RY", (0,), (0.1, 0.2))]
    assert len(sim.sparse_batch_moments) == 1
    assert sim.sparse_moments == []
    assert sim.statevector_reads == 0


def test_pennylane_sparse_hamiltonian_batch_variance_uses_native_csr_moments(monkeypatch):
    pytest.importorskip("pennylane")
    sp = pytest.importorskip("scipy.sparse")
    _install_fake_binding(monkeypatch)
    monkeypatch.setattr(_FakeQuantumSimulator, "enable_sparse_moments", True)
    for name in list(sys.modules):
        if name.startswith("pennylane_rocq"):
            sys.modules.pop(name)

    import pennylane as qml

    hamiltonian_matrix = sp.csr_matrix(np.diag([1.0, -1.0]).astype(np.complex128))
    observable = qml.SparseHamiltonian(hamiltonian_matrix, wires=[0])
    dev = qml.device("lightning.rocq", wires=1)
    tapes = [
        qml.tape.QuantumScript([qml.RY(0.1, wires=0)], [qml.var(observable)]),
        qml.tape.QuantumScript([qml.RY(0.2, wires=0)], [qml.var(observable)]),
    ]

    results = dev.batch_execute(tapes)

    assert results == pytest.approx((0.0, 0.0))
    sim = _FakeQuantumSimulator.instances[-1]
    assert sim.batch_size() == 2
    assert sim.batch_ops == [("RY", (0,), (0.1, 0.2))]
    assert len(sim.sparse_batch_moments) == 1
    assert sim.sparse_moments == []
    assert sim.statevector_reads == 0


def test_pennylane_hadamard_observable_uses_native_pauli_terms(monkeypatch):
    pytest.importorskip("pennylane")
    _install_fake_binding(monkeypatch)
    for name in list(sys.modules):
        if name.startswith("pennylane_rocq"):
            sys.modules.pop(name)

    import pennylane as qml

    dev = qml.device("lightning.rocq", wires=1)

    @qml.qnode(dev)
    def circuit():
        return qml.expval(qml.Hadamard(0))

    assert circuit() == pytest.approx(1 / math.sqrt(2))
    sim = _FakeQuantumSimulator.instances[-1]
    assert sim.ops == []
    assert sim.expectations == [("X", (0,)), ("Z", (0,))]
    assert sim.statevector_reads == 0


def test_pennylane_rot_dispatches_native_rotation_sequence(monkeypatch):
    pytest.importorskip("pennylane")
    _install_fake_binding(monkeypatch)
    for name in list(sys.modules):
        if name.startswith("pennylane_rocq"):
            sys.modules.pop(name)

    import pennylane as qml

    dev = qml.device("lightning.rocq", wires=1)

    @qml.qnode(dev)
    def circuit():
        qml.Rot(0.1, 0.2, 0.3, wires=0)
        return qml.expval(qml.PauliZ(0))

    circuit()
    sim = _FakeQuantumSimulator.instances[-1]

    assert sim.ops == [
        ("RZ", (0,), (0.1,)),
        ("RY", (0,), (0.2,)),
        ("RZ", (0,), (0.3,)),
    ]
    assert sim.matrices == []


def test_pennylane_expval_skips_unobservable_global_phase_matrices(monkeypatch):
    pytest.importorskip("pennylane")
    _install_fake_binding(monkeypatch)
    for name in list(sys.modules):
        if name.startswith("pennylane_rocq"):
            sys.modules.pop(name)

    import pennylane as qml

    dev = qml.device("lightning.rocq", wires=2)

    @qml.qnode(dev)
    def circuit():
        qml.PhaseShift(0.1, wires=0)
        qml.ControlledPhaseShift(0.2, wires=[0, 1])
        return qml.expval(qml.PauliZ(0))

    circuit()
    sim = _FakeQuantumSimulator.instances[-1]

    assert sim.ops == [
        ("P", (0,), (0.1,)),
        ("CP", (0, 1), (0.2,)),
    ]
    assert sim.matrices == []
    assert sim.statevector_reads == 0


def test_pennylane_controlled_rotations_dispatch_natively(monkeypatch):
    pytest.importorskip("pennylane")
    _install_fake_binding(monkeypatch)
    for name in list(sys.modules):
        if name.startswith("pennylane_rocq"):
            sys.modules.pop(name)

    import pennylane as qml

    dev = qml.device("lightning.rocq", wires=2)

    @qml.qnode(dev)
    def circuit():
        qml.CRX(0.1, wires=[0, 1])
        qml.CRY(0.2, wires=[0, 1])
        qml.CRZ(0.3, wires=[0, 1])
        return qml.expval(qml.PauliZ(0))

    circuit()
    sim = _FakeQuantumSimulator.instances[-1]

    assert sim.ops[:3] == [
        ("CRX", (0, 1), (0.1,)),
        ("CRY", (0, 1), (0.2,)),
        ("CRZ", (0, 1), (0.3,)),
    ]
    assert sim.matrices == []


def test_pennylane_crot_decomposes_natively(monkeypatch):
    pytest.importorskip("pennylane")
    _install_fake_binding(monkeypatch)
    for name in list(sys.modules):
        if name.startswith("pennylane_rocq"):
            sys.modules.pop(name)

    import pennylane as qml

    dev = qml.device("lightning.rocq", wires=2)

    @qml.qnode(dev)
    def circuit():
        qml.CRot(0.7, 0.8, 0.9, wires=[0, 1])
        return qml.state()

    circuit()
    sim = _FakeQuantumSimulator.instances[-1]

    assert sim.ops == [
        ("RZ", (1,), ((0.7 - 0.9) / 2,)),
        ("CNOT", (0, 1), ()),
        ("RZ", (1,), (-(0.7 + 0.9) / 2,)),
        ("RY", (1,), (-0.8 / 2,)),
        ("CNOT", (0, 1), ()),
        ("RY", (1,), (0.8 / 2,)),
        ("RZ", (1,), (0.9,)),
    ]
    assert sim.matrices == []


def test_pennylane_basis_state_prepares_with_native_x_gates(monkeypatch):
    pytest.importorskip("pennylane")
    _install_fake_binding(monkeypatch)
    for name in list(sys.modules):
        if name.startswith("pennylane_rocq"):
            sys.modules.pop(name)

    import pennylane as qml

    dev = qml.device("lightning.rocq", wires=3)

    @qml.qnode(dev)
    def circuit():
        qml.BasisState(np.array([1, 0, 1]), wires=[0, 1, 2])
        return qml.state()

    circuit()

    assert _FakeQuantumSimulator.instances[-1].ops == [
        ("X", (0,), ()),
        ("X", (2,), ()),
    ]


def test_pennylane_stateprep_dispatches_initial_matrix(monkeypatch):
    pytest.importorskip("pennylane")
    _install_fake_binding(monkeypatch)
    for name in list(sys.modules):
        if name.startswith("pennylane_rocq"):
            sys.modules.pop(name)

    import pennylane as qml

    state = np.array([0.0, 1.0], dtype=np.complex128)
    dev = qml.device("lightning.rocq", wires=1)

    @qml.qnode(dev)
    def circuit():
        qml.StatePrep(state, wires=[0])
        return qml.state()

    circuit()
    sim = _FakeQuantumSimulator.instances[-1]

    assert sim.ops == []
    assert sim.matrices == []
    np.testing.assert_allclose(sim.statevectors[0], state)


def test_pennylane_stateprep_after_operation_is_rejected(monkeypatch):
    pytest.importorskip("pennylane")
    _install_fake_binding(monkeypatch)
    for name in list(sys.modules):
        if name.startswith("pennylane_rocq"):
            sys.modules.pop(name)

    import pennylane as qml

    dev = qml.device("lightning.rocq", wires=1)

    @qml.qnode(dev)
    def circuit():
        qml.Hadamard(wires=0)
        qml.StatePrep(np.array([1.0, 0.0], dtype=np.complex128), wires=[0])
        return qml.state()

    with pytest.raises(ValueError, match="StatePrep is only supported as an initial state preparation"):
        circuit()


def test_pennylane_global_phase_uses_single_wire_phase_matrix(monkeypatch):
    pytest.importorskip("pennylane")
    _install_fake_binding(monkeypatch)
    for name in list(sys.modules):
        if name.startswith("pennylane_rocq"):
            sys.modules.pop(name)

    import pennylane as qml

    dev = qml.device("lightning.rocq", wires=2)

    @qml.qnode(dev)
    def circuit():
        qml.GlobalPhase(0.3)
        return qml.state()

    circuit()
    sim = _FakeQuantumSimulator.instances[-1]

    assert sim.ops == []
    assert len(sim.matrices) == 1
    matrix, targets = sim.matrices[0]
    np.testing.assert_allclose(matrix, np.eye(2, dtype=np.complex128) * np.exp(-0.3j))
    assert targets == (0,)


def test_pennylane_diagonal_qubit_unitary_decomposes_natively(monkeypatch):
    pytest.importorskip("pennylane")
    _install_fake_binding(monkeypatch)
    for name in list(sys.modules):
        if name.startswith("pennylane_rocq"):
            sys.modules.pop(name)

    import pennylane as qml

    diagonal = np.exp(1j * np.array([0.1, 0.3, 0.6, 1.2]))
    dev = qml.device("lightning.rocq", wires=2)

    @qml.qnode(dev)
    def circuit():
        qml.DiagonalQubitUnitary(diagonal, wires=[0, 1])
        return qml.state()

    circuit()
    sim = _FakeQuantumSimulator.instances[-1]

    assert [op[:2] for op in sim.ops] == [
        ("RZ", (0,)),
        ("RZ", (1,)),
        ("CNOT", (0, 1)),
        ("RZ", (1,)),
        ("CNOT", (0, 1)),
    ]
    np.testing.assert_allclose(sim.ops[0][2], (0.7,))
    np.testing.assert_allclose(sim.ops[1][2], (0.4,))
    np.testing.assert_allclose(sim.ops[3][2], (-0.2,))
    assert len(sim.matrices) == 1
    matrix, targets = sim.matrices[0]
    np.testing.assert_allclose(matrix, np.eye(2, dtype=np.complex128) * np.exp(0.55j))
    assert targets == (0,)


def test_pennylane_select_pauli_rot_decomposes_natively(monkeypatch):
    pytest.importorskip("pennylane")
    _install_fake_binding(monkeypatch)
    for name in list(sys.modules):
        if name.startswith("pennylane_rocq"):
            sys.modules.pop(name)

    import pennylane as qml

    dev = qml.device("lightning.rocq", wires=2)

    @qml.qnode(dev)
    def circuit():
        qml.SelectPauliRot(np.array([0.25, 1.25]), control_wires=[0], target_wire=1, rot_axis="Z")
        qml.SelectPauliRot(np.array([0.5, 1.5]), control_wires=[0], target_wire=1, rot_axis="X")
        qml.SelectPauliRot(np.array([0.75, 1.75]), control_wires=[0], target_wire=1, rot_axis="Y")
        return qml.state()

    circuit()
    sim = _FakeQuantumSimulator.instances[-1]

    assert [op[:2] for op in sim.ops] == [
        ("RZ", (1,)),
        ("CNOT", (0, 1)),
        ("RZ", (1,)),
        ("CNOT", (0, 1)),
        ("H", (1,)),
        ("RZ", (1,)),
        ("CNOT", (0, 1)),
        ("RZ", (1,)),
        ("CNOT", (0, 1)),
        ("H", (1,)),
        ("SDG", (1,)),
        ("H", (1,)),
        ("RZ", (1,)),
        ("CNOT", (0, 1)),
        ("RZ", (1,)),
        ("CNOT", (0, 1)),
        ("H", (1,)),
        ("S", (1,)),
    ]
    np.testing.assert_allclose(sim.ops[0][2], (0.75,))
    np.testing.assert_allclose(sim.ops[2][2], (-0.5,))
    np.testing.assert_allclose(sim.ops[5][2], (1.0,))
    np.testing.assert_allclose(sim.ops[7][2], (-0.5,))
    np.testing.assert_allclose(sim.ops[12][2], (1.25,))
    np.testing.assert_allclose(sim.ops[14][2], (-0.5,))
    assert sim.matrices == []


def test_pennylane_permutation_templates_decompose_natively(monkeypatch):
    pytest.importorskip("pennylane")
    _install_fake_binding(monkeypatch)
    for name in list(sys.modules):
        if name.startswith("pennylane_rocq"):
            sys.modules.pop(name)

    import pennylane as qml

    dev = qml.device("lightning.rocq", wires=3)

    @qml.qnode(dev)
    def circuit():
        qml.BasisEmbedding(np.array([1, 0, 1]), wires=[0, 1, 2])
        qml.Permute([2, 0, 1], wires=[0, 1, 2])
        return qml.state()

    circuit()
    sim = _FakeQuantumSimulator.instances[-1]

    assert sim.ops == [
        ("X", (0,), ()),
        ("X", (2,), ()),
        ("SWAP", (0, 2), ()),
        ("SWAP", (1, 2), ()),
    ]
    assert sim.matrices == []


def test_pennylane_controlled_sequence_decomposes_natively(monkeypatch):
    pytest.importorskip("pennylane")
    _install_fake_binding(monkeypatch)
    for name in list(sys.modules):
        if name.startswith("pennylane_rocq"):
            sys.modules.pop(name)

    import pennylane as qml

    dev = qml.device("lightning.rocq", wires=3)

    @qml.qnode(dev)
    def circuit():
        qml.ControlledSequence(qml.RX(0.2, wires=2), control=[0, 1])
        qml.ControlledSequence(qml.PhaseShift(0.3, wires=2), control=[0, 1])
        qml.ControlledSequence(qml.Hadamard(wires=2), control=[0, 1])
        return qml.state()

    circuit()
    sim = _FakeQuantumSimulator.instances[-1]

    assert sim.ops[:6] == [
        ("CRX", (0, 2), (0.4,)),
        ("CRX", (1, 2), (0.2,)),
        ("CP", (0, 2), (0.6,)),
        ("CP", (1, 2), (0.3,)),
        ("RY", (2,), (np.pi / 4,)),
        ("CNOT", (1, 2), ()),
    ]
    assert sim.ops[6:] == [
        ("RY", (2,), (-np.pi / 4,)),
    ]
    assert sim.matrices == []


def test_pennylane_select_decomposes_natively(monkeypatch):
    pytest.importorskip("pennylane")
    _install_fake_binding(monkeypatch)
    for name in list(sys.modules):
        if name.startswith("pennylane_rocq"):
            sys.modules.pop(name)

    import pennylane as qml

    dev = qml.device("lightning.rocq", wires=3)

    @qml.qnode(dev)
    def circuit():
        qml.Select([qml.PauliX(wires=1), qml.PauliZ(wires=1)], control=[0])
        qml.Select([qml.SWAP(wires=[1, 2]), qml.SWAP(wires=[1, 2])], control=[0])
        return qml.state()

    circuit()
    sim = _FakeQuantumSimulator.instances[-1]

    assert sim.ops == [
        ("X", (0,), ()),
        ("CNOT", (0, 1), ()),
        ("X", (0,), ()),
        ("CZ", (0, 1), ()),
        ("X", (0,), ()),
        ("CSWAP", (0, 1, 2), ()),
        ("X", (0,), ()),
        ("CSWAP", (0, 1, 2), ()),
    ]
    assert sim.matrices == []
    assert sim.controlled_matrices == []


def test_pennylane_partial_select_decomposes_natively(monkeypatch):
    pytest.importorskip("pennylane")
    _install_fake_binding(monkeypatch)
    for name in list(sys.modules):
        if name.startswith("pennylane_rocq"):
            sys.modules.pop(name)

    import pennylane as qml

    dev = qml.device("lightning.rocq", wires=6)

    @qml.qnode(dev)
    def circuit():
        qml.Select(
            [qml.PauliX(wires=3), qml.PauliX(wires=4), qml.PauliX(wires=5)],
            control=[0, 1, 2],
            partial=True,
        )
        return qml.state()

    circuit()
    sim = _FakeQuantumSimulator.instances[-1]

    assert sim.ops == [
        ("X", (1,), ()),
        ("X", (2,), ()),
        ("MCX", (1, 2, 3), ()),
        ("X", (2,), ()),
        ("X", (1,), ()),
        ("CNOT", (2, 4), ()),
        ("CNOT", (1, 5), ()),
    ]
    assert sim.matrices == []
    assert sim.controlled_matrices == []


def test_pennylane_prep_sel_prep_decomposes_partial_select_natively(monkeypatch):
    pytest.importorskip("pennylane")
    _install_fake_binding(monkeypatch)
    for name in list(sys.modules):
        if name.startswith("pennylane_rocq"):
            sys.modules.pop(name)

    import pennylane as qml

    dev = qml.device("lightning.rocq", wires=2)

    @qml.qnode(dev)
    def circuit():
        lcu = 0.25 * qml.X(1) + 0.75 * qml.Z(1)
        qml.PrepSelPrep(lcu, control=[0])
        return qml.state()

    circuit()
    sim = _FakeQuantumSimulator.instances[-1]

    assert sim.ops == [
        ("RY", (0,), (2 * np.pi / 3,)),
        ("X", (0,), ()),
        ("CNOT", (0, 1), ()),
        ("X", (0,), ()),
        ("CZ", (0, 1), ()),
        ("RY", (0,), (-2 * np.pi / 3,)),
    ]
    assert sim.matrices == []
    assert sim.controlled_matrices == []


def test_pennylane_prep_sel_prep_signed_coefficients_decompose_controlled_phase_natively(monkeypatch):
    pytest.importorskip("pennylane")
    _install_fake_binding(monkeypatch)
    for name in list(sys.modules):
        if name.startswith("pennylane_rocq"):
            sys.modules.pop(name)

    import pennylane as qml

    cases = [
        (-0.25 * qml.X(1) + 0.75 * qml.Z(1), np.pi),
        ((0.25j) * qml.X(1) + 0.75 * qml.Z(1), np.pi / 2),
    ]

    for lcu, phase_angle in cases:
        dev = qml.device("lightning.rocq", wires=2)

        @qml.qnode(dev)
        def circuit():
            qml.PrepSelPrep(lcu, control=[0])
            return qml.state()

        circuit()
        sim = _FakeQuantumSimulator.instances[-1]

        assert [op[:2] for op in sim.ops] == [
            ("RY", (0,)),
            ("X", (0,)),
            ("CNOT", (0, 1)),
            ("X", (0,)),
            ("X", (0,)),
            ("P", (0,)),
            ("X", (0,)),
            ("CZ", (0, 1)),
            ("RY", (0,)),
        ]
        np.testing.assert_allclose(sim.ops[5][2], (phase_angle,))
        assert sim.matrices == []
        assert sim.controlled_matrices == []


def test_pennylane_select_global_phase_uses_small_controlled_matrix(monkeypatch):
    pytest.importorskip("pennylane")
    _install_fake_binding(monkeypatch)
    for name in list(sys.modules):
        if name.startswith("pennylane_rocq"):
            sys.modules.pop(name)

    import pennylane as qml

    dev = qml.device("lightning.rocq", wires=3)

    @qml.qnode(dev)
    def circuit():
        qml.Select(
            [
                qml.I(2),
                qml.I(2),
                qml.I(2),
                qml.GlobalPhase(0.4, wires=[2]),
            ],
            control=[0, 1],
        )
        return qml.state()

    circuit()
    sim = _FakeQuantumSimulator.instances[-1]

    assert sim.ops == []
    assert sim.matrices == []
    assert len(sim.controlled_matrices) == 1
    matrix, controls, targets = sim.controlled_matrices[0]
    np.testing.assert_allclose(matrix, np.diag([1.0, np.exp(-0.4j)]))
    assert controls == (0,)
    assert targets == (1,)


def test_pennylane_qrom_select_product_decomposes_natively(monkeypatch):
    pytest.importorskip("pennylane")
    _install_fake_binding(monkeypatch)
    for name in list(sys.modules):
        if name.startswith("pennylane_rocq"):
            sys.modules.pop(name)

    import pennylane as qml

    dev = qml.device("lightning.rocq", wires=6)

    @qml.qnode(dev)
    def circuit():
        qml.QROM(
            np.array([[0, 1], [1, 0], [1, 1], [0, 0]], dtype=int),
            control_wires=[0, 1],
            target_wires=[2, 3],
            work_wires=[4, 5],
            clean=False,
        )
        return qml.state()

    circuit()
    sim = _FakeQuantumSimulator.instances[-1]

    assert sim.ops == [
        ("X", (0,), ()),
        ("CNOT", (0, 3), ()),
        ("CNOT", (0, 4), ()),
        ("X", (0,), ()),
        ("CNOT", (0, 2), ()),
        ("CNOT", (0, 3), ()),
        ("CSWAP", (1, 2, 4), ()),
        ("CSWAP", (1, 3, 5), ()),
    ]
    assert sim.matrices == []
    assert sim.controlled_matrices == []


def test_pennylane_fable_decomposes_natively(monkeypatch):
    pytest.importorskip("pennylane")
    _install_fake_binding(monkeypatch)
    for name in list(sys.modules):
        if name.startswith("pennylane_rocq"):
            sys.modules.pop(name)

    import pennylane as qml

    dev = qml.device("lightning.rocq", wires=3)

    @qml.qnode(dev)
    def circuit():
        qml.FABLE(np.array([[0.5, 0.25], [0.125, 0.75]]), wires=[0, 1, 2])
        return qml.state()

    circuit()
    sim = _FakeQuantumSimulator.instances[-1]

    assert [op[:2] for op in sim.ops] == [
        ("H", (1,)),
        ("RY", (0,)),
        ("CNOT", (2, 0)),
        ("RY", (0,)),
        ("CNOT", (1, 0)),
        ("RY", (0,)),
        ("CNOT", (2, 0)),
        ("RY", (0,)),
        ("CNOT", (1, 0)),
        ("SWAP", (1, 2)),
        ("H", (1,)),
    ]
    assert sim.matrices == []
    assert sim.controlled_matrices == []


def test_pennylane_common_gates_use_native_toffoli_and_matrix_fallback(monkeypatch):
    pytest.importorskip("pennylane")
    _install_fake_binding(monkeypatch)
    for name in list(sys.modules):
        if name.startswith("pennylane_rocq"):
            sys.modules.pop(name)

    import pennylane as qml

    dev = qml.device("lightning.rocq", wires=3)

    @qml.qnode(dev)
    def circuit():
        qml.PhaseShift(0.1, wires=0)
        qml.ControlledPhaseShift(0.2, wires=[0, 1])
        qml.IsingXX(0.3, wires=[0, 1])
        qml.IsingYY(0.4, wires=[0, 1])
        qml.IsingZZ(0.5, wires=[0, 1])
        qml.IsingXY(0.6, wires=[0, 1])
        qml.CRot(0.7, 0.8, 0.9, wires=[0, 1])
        qml.Toffoli(wires=[0, 1, 2])
        return qml.state()

    circuit()
    sim = _FakeQuantumSimulator.instances[-1]

    assert sim.ops == [
        ("P", (0,), (0.1,)),
        ("CP", (0, 1), (0.2,)),
        ("CNOT", (0, 1), ()),
        ("RX", (0,), (0.3,)),
        ("CNOT", (0, 1), ()),
        ("RX", (0,), (np.pi / 2,)),
        ("RX", (1,), (np.pi / 2,)),
        ("CNOT", (0, 1), ()),
        ("RZ", (1,), (0.4,)),
        ("CNOT", (0, 1), ()),
        ("RX", (0,), (-np.pi / 2,)),
        ("RX", (1,), (-np.pi / 2,)),
        ("CNOT", (0, 1), ()),
        ("RZ", (1,), (0.5,)),
        ("CNOT", (0, 1), ()),
        ("H", (0,), ()),
        ("SDG", (1,), ()),
        ("CNOT", (0, 1), ()),
        ("S", (1,), ()),
        ("RY", (0,), (0.3,)),
        ("RX", (1,), (-0.3,)),
        ("SDG", (1,), ()),
        ("CNOT", (0, 1), ()),
        ("S", (1,), ()),
        ("H", (0,), ()),
        ("RZ", (1,), ((0.7 - 0.9) / 2,)),
        ("CNOT", (0, 1), ()),
        ("RZ", (1,), (-(0.7 + 0.9) / 2,)),
        ("RY", (1,), (-0.8 / 2,)),
        ("CNOT", (0, 1), ()),
        ("RY", (1,), (0.8 / 2,)),
        ("RZ", (1,), (0.9,)),
        ("MCX", (0, 1, 2), ()),
    ]
    assert sim.matrices == []


def test_pennylane_matrix_fallback_converts_wire_order_for_rocquantum(monkeypatch):
    pytest.importorskip("pennylane")
    _install_fake_binding(monkeypatch)
    for name in list(sys.modules):
        if name.startswith("pennylane_rocq"):
            sys.modules.pop(name)

    import pennylane as qml

    dev = qml.device("lightning.rocq", wires=2)
    pennylane_cnot_matrix = qml.matrix(qml.CNOT(wires=[0, 1]))

    @qml.qnode(dev)
    def circuit():
        qml.QubitUnitary(pennylane_cnot_matrix, wires=[0, 1])
        return qml.state()

    circuit()
    matrix, targets = _FakeQuantumSimulator.instances[-1].matrices[0]

    expected_local_little_endian = np.array(
        [
            [1, 0, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0],
            [0, 1, 0, 0],
        ],
        dtype=np.complex128,
    )
    assert targets == (0, 1)
    np.testing.assert_allclose(matrix, expected_local_little_endian)


def test_pennylane_block_encode_dispatches_dense_matrix(monkeypatch):
    pytest.importorskip("pennylane")
    _install_fake_binding(monkeypatch)
    for name in list(sys.modules):
        if name.startswith("pennylane_rocq"):
            sys.modules.pop(name)

    import pennylane as qml
    from rocquantum.framework_runtime import matrix_to_little_endian_wires

    block = np.array([[0.2, 0.3], [0.4, 0.1]], dtype=np.complex128)
    dev = qml.device("lightning.rocq", wires=2)

    @qml.qnode(dev)
    def circuit():
        qml.BlockEncode(block, wires=[0, 1])
        return qml.state()

    circuit()
    sim = _FakeQuantumSimulator.instances[-1]

    assert sim.ops == []
    assert len(sim.matrices) == 1
    matrix, targets = sim.matrices[0]
    np.testing.assert_allclose(
        matrix,
        matrix_to_little_endian_wires(qml.matrix(qml.BlockEncode(block, wires=[0, 1]))),
    )
    assert targets == (0, 1)


def test_pennylane_sparse_block_encode_uses_sparse_statevector_fallback(monkeypatch):
    pytest.importorskip("pennylane")
    sparse = pytest.importorskip("scipy.sparse")
    _install_fake_binding(monkeypatch)
    for name in list(sys.modules):
        if name.startswith("pennylane_rocq"):
            sys.modules.pop(name)

    import pennylane as qml
    from rocquantum.framework_runtime import (
        apply_sparse_matrix_to_statevector,
        sparse_matrix_to_little_endian_wires,
    )

    block = sparse.csr_matrix(np.array([[0.2, 0.0], [0.0, 0.3]], dtype=np.complex128))
    dev = qml.device("lightning.rocq", wires=2)

    @qml.qnode(dev)
    def circuit():
        qml.BlockEncode(block, wires=[0, 1])
        return qml.state()

    circuit()
    sim = _FakeQuantumSimulator.instances[-1]

    assert sim.ops == []
    assert sim.matrices == []
    assert len(sim.statevectors) == 1
    sparse_matrix = sparse_matrix_to_little_endian_wires(qml.BlockEncode(block, wires=[0, 1]).sparse_matrix())
    np.testing.assert_allclose(
        sim.statevectors[0],
        apply_sparse_matrix_to_statevector(
            np.array([1.0 / np.sqrt(2.0), 0.0, 0.0, 1.0 / np.sqrt(2.0)], dtype=np.complex128),
            sparse_matrix.data,
            sparse_matrix.indices,
            sparse_matrix.indptr,
            sparse_matrix.shape,
            [0, 1],
            2,
        ),
    )


def test_pennylane_controlled_qubit_unitary_uses_native_controlled_matrix(monkeypatch):
    pytest.importorskip("pennylane")
    _install_fake_binding(monkeypatch)
    for name in list(sys.modules):
        if name.startswith("pennylane_rocq"):
            sys.modules.pop(name)

    import pennylane as qml

    base = np.array([[1.0, 0.0], [0.0, 1.0j]], dtype=np.complex128)
    dev = qml.device("lightning.rocq", wires=2)

    @qml.qnode(dev)
    def circuit():
        qml.ControlledQubitUnitary(base, wires=[0, 1])
        return qml.state()

    circuit()
    sim = _FakeQuantumSimulator.instances[-1]

    assert sim.matrices == []
    assert len(sim.controlled_matrices) == 1
    matrix, controls, targets = sim.controlled_matrices[0]
    np.testing.assert_allclose(matrix, base)
    assert controls == (0,)
    assert targets == (1,)


def test_pennylane_open_controlled_qubit_unitary_flips_around_native_matrix(monkeypatch):
    pytest.importorskip("pennylane")
    _install_fake_binding(monkeypatch)
    for name in list(sys.modules):
        if name.startswith("pennylane_rocq"):
            sys.modules.pop(name)

    import pennylane as qml

    base = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
    dev = qml.device("lightning.rocq", wires=2)

    @qml.qnode(dev)
    def circuit():
        qml.ControlledQubitUnitary(base, wires=[0, 1], control_values=[False])
        return qml.state()

    circuit()
    sim = _FakeQuantumSimulator.instances[-1]

    assert sim.matrices == []
    assert sim.ops == [("X", (0,), ()), ("X", (0,), ())]
    assert len(sim.controlled_matrices) == 1
    matrix, controls, targets = sim.controlled_matrices[0]
    np.testing.assert_allclose(matrix, base)
    assert controls == (0,)
    assert targets == (1,)


def test_pennylane_extended_gates_use_native_multi_control_and_matrix_fallback(monkeypatch):
    pytest.importorskip("pennylane")
    _install_fake_binding(monkeypatch)
    for name in list(sys.modules):
        if name.startswith("pennylane_rocq"):
            sys.modules.pop(name)

    import pennylane as qml

    dev = qml.device("lightning.rocq", wires=4)

    @qml.qnode(dev)
    def circuit():
        qml.CH(wires=[0, 1])
        qml.CY(wires=[0, 1])
        qml.CCZ(wires=[0, 1, 2])
        qml.CPhaseShift00(0.1, wires=[0, 1])
        qml.CPhaseShift01(0.2, wires=[0, 1])
        qml.CPhaseShift10(0.3, wires=[0, 1])
        qml.MultiControlledX(wires=[0, 1, 2])
        qml.Toffoli(wires=[0, 1, 2])
        qml.CSWAP(wires=[0, 1, 2])
        qml.MultiRZ(0.4, wires=[0, 1, 2])
        qml.PSWAP(0.5, wires=[0, 1])
        qml.ISWAP(wires=[0, 1])
        qml.SISWAP(wires=[0, 1])
        qml.SQISW(wires=[0, 1])
        qml.ECR(wires=[0, 1])
        qml.SingleExcitation(0.6, wires=[0, 1])
        qml.DoubleExcitation(0.7, wires=[0, 1, 2, 3])
        qml.OrbitalRotation(0.8, wires=[0, 1, 2, 3])
        qml.FermionicSWAP(0.9, wires=[0, 1])
        return qml.state()

    circuit()
    sim = _FakeQuantumSimulator.instances[-1]

    assert sim.ops == [
        ("RY", (1,), (np.pi / 4,)),
        ("CNOT", (0, 1), ()),
        ("RY", (1,), (-np.pi / 4,)),
        ("SDG", (1,), ()),
        ("CNOT", (0, 1), ()),
        ("S", (1,), ()),
        ("H", (2,), ()),
        ("MCX", (0, 1, 2), ()),
        ("H", (2,), ()),
        ("X", (0,), ()),
        ("X", (1,), ()),
        ("CP", (0, 1), (0.1,)),
        ("X", (1,), ()),
        ("X", (0,), ()),
        ("X", (0,), ()),
        ("CP", (0, 1), (0.2,)),
        ("X", (0,), ()),
        ("X", (1,), ()),
        ("CP", (0, 1), (0.3,)),
        ("X", (1,), ()),
        ("MCX", (0, 1, 2), ()),
        ("MCX", (0, 1, 2), ()),
        ("CSWAP", (0, 1, 2), ()),
        ("CNOT", (0, 2), ()),
        ("CNOT", (1, 2), ()),
        ("RZ", (2,), (0.4,)),
        ("CNOT", (1, 2), ()),
        ("CNOT", (0, 2), ()),
        ("SWAP", (0, 1), ()),
        ("CNOT", (0, 1), ()),
        ("P", (1,), (0.5,)),
        ("CNOT", (0, 1), ()),
        ("S", (0,), ()),
        ("S", (1,), ()),
        ("H", (0,), ()),
        ("CNOT", (0, 1), ()),
        ("CNOT", (1, 0), ()),
        ("H", (1,), ()),
        ("RX", (0,), (np.pi / 2,)),
        ("RZ", (0,), (np.pi / 2,)),
        ("CNOT", (0, 1), ()),
        ("RX", (0,), (np.pi / 2,)),
        ("RZ", (0,), (7 * np.pi / 4,)),
        ("RX", (0,), (np.pi / 2,)),
        ("RZ", (0,), (np.pi / 2,)),
        ("RX", (1,), (np.pi / 2,)),
        ("RZ", (1,), (7 * np.pi / 4,)),
        ("CNOT", (0, 1), ()),
        ("RX", (0,), (np.pi / 2,)),
        ("RX", (1,), (np.pi / 2,)),
        ("RX", (0,), (np.pi / 2,)),
        ("RZ", (0,), (np.pi / 2,)),
        ("CNOT", (0, 1), ()),
        ("RX", (0,), (np.pi / 2,)),
        ("RZ", (0,), (7 * np.pi / 4,)),
        ("RX", (0,), (np.pi / 2,)),
        ("RZ", (0,), (np.pi / 2,)),
        ("RX", (1,), (np.pi / 2,)),
        ("RZ", (1,), (7 * np.pi / 4,)),
        ("CNOT", (0, 1), ()),
        ("RX", (0,), (np.pi / 2,)),
        ("RX", (1,), (np.pi / 2,)),
        ("Z", (0,), ()),
        ("CNOT", (0, 1), ()),
        ("RX", (1,), (np.pi / 2,)),
        ("RX", (0,), (np.pi / 2,)),
        ("RY", (0,), (np.pi / 2,)),
        ("RX", (0,), (np.pi / 2,)),
        ("H", (0,), ()),
        ("CNOT", (0, 1), ()),
        ("RY", (0,), (-0.3,)),
        ("RY", (1,), (-0.3,)),
        ("CNOT", (0, 1), ()),
        ("H", (0,), ()),
        ("CNOT", (2, 3), ()),
        ("CNOT", (0, 2), ()),
        ("H", (3,), ()),
        ("H", (0,), ()),
        ("CNOT", (2, 3), ()),
        ("CNOT", (0, 1), ()),
        ("RY", (1,), (0.0875,)),
        ("RY", (0,), (-0.0875,)),
        ("CNOT", (0, 3), ()),
        ("H", (3,), ()),
        ("CNOT", (3, 1), ()),
        ("RY", (1,), (0.0875,)),
        ("RY", (0,), (-0.0875,)),
        ("CNOT", (2, 1), ()),
        ("CNOT", (2, 0), ()),
        ("RY", (1,), (-0.0875,)),
        ("RY", (0,), (0.0875,)),
        ("CNOT", (3, 1), ()),
        ("H", (3,), ()),
        ("CNOT", (0, 3), ()),
        ("RY", (1,), (-0.0875,)),
        ("RY", (0,), (0.0875,)),
        ("CNOT", (0, 1), ()),
        ("CNOT", (2, 0), ()),
        ("H", (0,), ()),
        ("H", (3,), ()),
        ("CNOT", (0, 2), ()),
        ("CNOT", (2, 3), ()),
        ("H", (1,), ()),
        ("H", (2,), ()),
        ("CNOT", (1, 2), ()),
        ("RZ", (2,), (np.pi / 2,)),
        ("CNOT", (1, 2), ()),
        ("H", (1,), ()),
        ("H", (2,), ()),
        ("RX", (1,), (np.pi / 2,)),
        ("RX", (2,), (np.pi / 2,)),
        ("CNOT", (1, 2), ()),
        ("RZ", (2,), (np.pi / 2,)),
        ("CNOT", (1, 2), ()),
        ("RX", (1,), (-np.pi / 2,)),
        ("RX", (2,), (-np.pi / 2,)),
        ("RZ", (1,), (np.pi / 2,)),
        ("RZ", (2,), (np.pi / 2,)),
        ("H", (0,), ()),
        ("CNOT", (0, 1), ()),
        ("RY", (0,), (-0.4,)),
        ("RY", (1,), (-0.4,)),
        ("CNOT", (0, 1), ()),
        ("H", (0,), ()),
        ("H", (2,), ()),
        ("CNOT", (2, 3), ()),
        ("RY", (2,), (-0.4,)),
        ("RY", (3,), (-0.4,)),
        ("CNOT", (2, 3), ()),
        ("H", (2,), ()),
        ("H", (1,), ()),
        ("H", (2,), ()),
        ("CNOT", (1, 2), ()),
        ("RZ", (2,), (np.pi / 2,)),
        ("CNOT", (1, 2), ()),
        ("H", (1,), ()),
        ("H", (2,), ()),
        ("RX", (1,), (np.pi / 2,)),
        ("RX", (2,), (np.pi / 2,)),
        ("CNOT", (1, 2), ()),
        ("RZ", (2,), (np.pi / 2,)),
        ("CNOT", (1, 2), ()),
        ("RX", (1,), (-np.pi / 2,)),
        ("RX", (2,), (-np.pi / 2,)),
        ("RZ", (1,), (np.pi / 2,)),
        ("RZ", (2,), (np.pi / 2,)),
        ("H", (0,), ()),
        ("H", (1,), ()),
        ("CNOT", (0, 1), ()),
        ("RZ", (1,), (0.45,)),
        ("CNOT", (0, 1), ()),
        ("H", (0,), ()),
        ("H", (1,), ()),
        ("RX", (0,), (np.pi / 2,)),
        ("RX", (1,), (np.pi / 2,)),
        ("CNOT", (0, 1), ()),
        ("RZ", (1,), (0.45,)),
        ("CNOT", (0, 1), ()),
        ("RX", (0,), (-np.pi / 2,)),
        ("RX", (1,), (-np.pi / 2,)),
        ("RZ", (0,), (0.45,)),
        ("RZ", (1,), (0.45,)),
    ]
    assert [targets for _, targets in sim.matrices] == [
        (0,),
        (0,),
        (0,),
        (1,),
        (0,),
        (1,),
        (0,),
        (0,),
        (0,),
        (1,),
        (0,),
        (1,),
        (1,),
        (1,),
        (1,),
        (0,),
    ]


def test_pennylane_single_excitation_phase_variants_decompose_natively(monkeypatch):
    pytest.importorskip("pennylane")
    _install_fake_binding(monkeypatch)
    for name in list(sys.modules):
        if name.startswith("pennylane_rocq"):
            sys.modules.pop(name)

    import pennylane as qml

    dev = qml.device("lightning.rocq", wires=2)

    @qml.qnode(dev)
    def circuit():
        qml.SingleExcitationPlus(0.6, wires=[0, 1])
        qml.SingleExcitationMinus(0.8, wires=[0, 1])
        return qml.state()

    circuit()
    sim = _FakeQuantumSimulator.instances[-1]

    assert sim.ops == [
        ("H", (1,), ()),
        ("CNOT", (1, 0), ()),
        ("RY", (0,), (0.3,)),
        ("RY", (1,), (0.3,)),
        ("SDG", (0,), ()),
        ("CNOT", (1, 0), ()),
        ("S", (0,), ()),
        ("S", (1,), ()),
        ("H", (1,), ()),
        ("RZ", (1,), (-0.3,)),
        ("CNOT", (0, 1), ()),
        ("H", (1,), ()),
        ("CNOT", (1, 0), ()),
        ("RY", (0,), (0.4,)),
        ("RY", (1,), (0.4,)),
        ("SDG", (0,), ()),
        ("CNOT", (1, 0), ()),
        ("S", (0,), ()),
        ("S", (1,), ()),
        ("H", (1,), ()),
        ("RZ", (1,), (0.4,)),
        ("CNOT", (0, 1), ()),
    ]
    assert [targets for _, targets in sim.matrices] == [(0,), (0,)]
    np.testing.assert_allclose(sim.matrices[0][0], np.eye(2) * np.exp(0.15j))
    np.testing.assert_allclose(sim.matrices[1][0], np.eye(2) * np.exp(-0.2j))


def test_pennylane_double_excitation_phase_variants_decompose_natively(monkeypatch):
    pytest.importorskip("pennylane")
    _install_fake_binding(monkeypatch)
    for name in list(sys.modules):
        if name.startswith("pennylane_rocq"):
            sys.modules.pop(name)

    import pennylane as qml

    def phase_ops(theta, sign):
        alpha = sign * theta / 2
        ops = []
        for wires, coefficient in (
            ((0, 1), 1),
            ((0, 2), -1),
            ((0, 3), -1),
            ((1, 2), -1),
            ((1, 3), -1),
            ((2, 3), 1),
        ):
            control, target = wires
            ops.extend(
                [
                    ("CNOT", (control, target), ()),
                    ("RZ", (target,), (coefficient * alpha / 4,)),
                    ("CNOT", (control, target), ()),
                ]
            )
        ops.extend(
            [
                ("CNOT", (0, 3), ()),
                ("CNOT", (1, 3), ()),
                ("CNOT", (2, 3), ()),
                ("RZ", (3,), (alpha / 4,)),
                ("CNOT", (2, 3), ()),
                ("CNOT", (1, 3), ()),
                ("CNOT", (0, 3), ()),
            ]
        )
        return ops

    dev = qml.device("lightning.rocq", wires=4)

    @qml.qnode(dev)
    def circuit():
        qml.DoubleExcitationPlus(0.8, wires=[0, 1, 2, 3])
        qml.DoubleExcitationMinus(0.6, wires=[0, 1, 2, 3])
        return qml.state()

    circuit()
    sim = _FakeQuantumSimulator.instances[-1]

    assert sim.ops[:25] == phase_ops(0.8, 1)
    assert sim.ops[25:29] == [
        ("CNOT", (2, 3), ()),
        ("CNOT", (0, 2), ()),
        ("H", (3,), ()),
        ("H", (0,), ()),
    ]
    assert sim.ops[53:78] == phase_ops(0.6, -1)
    assert sim.ops[78:82] == [
        ("CNOT", (2, 3), ()),
        ("CNOT", (0, 2), ()),
        ("H", (3,), ()),
        ("H", (0,), ()),
    ]
    assert [targets for _, targets in sim.matrices] == [(0,), (0,)]
    np.testing.assert_allclose(sim.matrices[0][0], np.eye(2) * np.exp(1j * 7 * 0.8 / 16))
    np.testing.assert_allclose(sim.matrices[1][0], np.eye(2) * np.exp(-1j * 7 * 0.6 / 16))


def test_pennylane_qft_uses_native_controlled_phase_decomposition(monkeypatch):
    pytest.importorskip("pennylane")
    _install_fake_binding(monkeypatch)
    for name in list(sys.modules):
        if name.startswith("pennylane_rocq"):
            sys.modules.pop(name)

    import pennylane as qml

    dev = qml.device("lightning.rocq", wires=2)

    @qml.qnode(dev)
    def circuit():
        qml.QFT(wires=[0, 1])
        return qml.state()

    circuit()
    sim = _FakeQuantumSimulator.instances[-1]

    assert sim.ops == [
        ("H", (0,), ()),
        ("CP", (1, 0), (np.pi / 2,)),
        ("H", (1,), ()),
        ("SWAP", (0, 1), ()),
    ]
    assert sim.matrices == []


def test_pennylane_arithmetic_templates_decompose_natively(monkeypatch):
    pytest.importorskip("pennylane")
    _install_fake_binding(monkeypatch)
    for name in list(sys.modules):
        if name.startswith("pennylane_rocq"):
            sys.modules.pop(name)

    import pennylane as qml

    dev = qml.device("lightning.rocq", wires=4)

    @qml.qnode(dev)
    def circuit():
        qml.QubitSum(wires=[0, 1, 2])
        qml.QubitCarry(wires=[0, 1, 2, 3])
        return qml.state()

    circuit()
    sim = _FakeQuantumSimulator.instances[-1]

    assert sim.ops == [
        ("CNOT", (1, 2), ()),
        ("CNOT", (0, 2), ()),
        ("MCX", (1, 2, 3), ()),
        ("CNOT", (1, 2), ()),
        ("MCX", (0, 2, 3), ()),
    ]
    assert sim.matrices == []


def test_pennylane_grover_operator_decomposes_natively(monkeypatch):
    pytest.importorskip("pennylane")
    _install_fake_binding(monkeypatch)
    for name in list(sys.modules):
        if name.startswith("pennylane_rocq"):
            sys.modules.pop(name)

    import pennylane as qml

    dev = qml.device("lightning.rocq", wires=3)

    @qml.qnode(dev)
    def circuit():
        qml.GroverOperator(wires=[0, 1, 2])
        return qml.state()

    circuit()
    sim = _FakeQuantumSimulator.instances[-1]

    assert sim.ops == [
        ("H", (0,), ()),
        ("H", (1,), ()),
        ("Z", (2,), ()),
        ("X", (0,), ()),
        ("X", (1,), ()),
        ("MCX", (0, 1, 2), ()),
        ("X", (1,), ()),
        ("X", (0,), ()),
        ("Z", (2,), ()),
        ("H", (0,), ()),
        ("H", (1,), ()),
    ]
    assert len(sim.matrices) == 1
    matrix, targets = sim.matrices[0]
    np.testing.assert_allclose(matrix, -np.eye(2, dtype=np.complex128))
    assert targets == (0,)


def test_pennylane_isingzz_dispatches_native_cnot_chain(monkeypatch):
    pytest.importorskip("pennylane")
    _install_fake_binding(monkeypatch)
    for name in list(sys.modules):
        if name.startswith("pennylane_rocq"):
            sys.modules.pop(name)

    import pennylane as qml

    dev = qml.device("lightning.rocq", wires=2)

    @qml.qnode(dev)
    def circuit():
        qml.IsingZZ(0.7, wires=[0, 1])
        return qml.state()

    circuit()
    sim = _FakeQuantumSimulator.instances[-1]

    assert sim.ops == [
        ("CNOT", (0, 1), ()),
        ("RZ", (1,), (0.7,)),
        ("CNOT", (0, 1), ()),
    ]
    assert sim.matrices == []


def test_pennylane_isingxx_isingyy_dispatch_native_sequences(monkeypatch):
    pytest.importorskip("pennylane")
    _install_fake_binding(monkeypatch)
    for name in list(sys.modules):
        if name.startswith("pennylane_rocq"):
            sys.modules.pop(name)

    import pennylane as qml

    dev = qml.device("lightning.rocq", wires=2)

    @qml.qnode(dev)
    def circuit():
        qml.IsingXX(0.2, wires=[0, 1])
        qml.IsingYY(0.3, wires=[0, 1])
        return qml.state()

    circuit()
    sim = _FakeQuantumSimulator.instances[-1]

    assert sim.ops == [
        ("CNOT", (0, 1), ()),
        ("RX", (0,), (0.2,)),
        ("CNOT", (0, 1), ()),
        ("RX", (0,), (np.pi / 2,)),
        ("RX", (1,), (np.pi / 2,)),
        ("CNOT", (0, 1), ()),
        ("RZ", (1,), (0.3,)),
        ("CNOT", (0, 1), ()),
        ("RX", (0,), (-np.pi / 2,)),
        ("RX", (1,), (-np.pi / 2,)),
    ]
    assert sim.matrices == []


def test_pennylane_multirz_dispatches_native_cnot_chain(monkeypatch):
    pytest.importorskip("pennylane")
    _install_fake_binding(monkeypatch)
    for name in list(sys.modules):
        if name.startswith("pennylane_rocq"):
            sys.modules.pop(name)

    import pennylane as qml

    dev = qml.device("lightning.rocq", wires=3)

    @qml.qnode(dev)
    def circuit():
        qml.MultiRZ(0.7, wires=[0, 1, 2])
        return qml.state()

    circuit()
    sim = _FakeQuantumSimulator.instances[-1]

    assert sim.ops == [
        ("CNOT", (0, 2), ()),
        ("CNOT", (1, 2), ()),
        ("RZ", (2,), (0.7,)),
        ("CNOT", (1, 2), ()),
        ("CNOT", (0, 2), ()),
    ]
    assert sim.matrices == []


def test_pennylane_paulirot_uses_native_multirz_decomposition(monkeypatch):
    pytest.importorskip("pennylane")
    _install_fake_binding(monkeypatch)
    for name in list(sys.modules):
        if name.startswith("pennylane_rocq"):
            sys.modules.pop(name)

    import pennylane as qml

    dev = qml.device("lightning.rocq", wires=3)

    @qml.qnode(dev)
    def circuit():
        qml.PauliRot(0.2, "XYZ", wires=[0, 1, 2])
        return qml.state()

    circuit()
    sim = _FakeQuantumSimulator.instances[-1]

    assert ("RZ", (2,), (0.2,)) in sim.ops
    assert ("CNOT", (0, 2), ()) in sim.ops
    assert ("CNOT", (1, 2), ()) in sim.ops
    assert sim.matrices == []


def test_pennylane_multicontrolledx_all_one_controls_dispatches_natively(monkeypatch):
    pytest.importorskip("pennylane")
    _install_fake_binding(monkeypatch)
    for name in list(sys.modules):
        if name.startswith("pennylane_rocq"):
            sys.modules.pop(name)

    import pennylane as qml

    dev = qml.device("lightning.rocq", wires=4)

    @qml.qnode(dev)
    def circuit():
        qml.MultiControlledX(wires=[0, 1, 2, 3])
        return qml.state()

    circuit()
    sim = _FakeQuantumSimulator.instances[-1]

    assert sim.ops == [("MCX", (0, 1, 2, 3), ())]
    assert sim.matrices == []


def test_pennylane_multicontrolledx_nondefault_controls_decomposes_natively(monkeypatch):
    pytest.importorskip("pennylane")
    _install_fake_binding(monkeypatch)
    for name in list(sys.modules):
        if name.startswith("pennylane_rocq"):
            sys.modules.pop(name)

    import pennylane as qml

    dev = qml.device("lightning.rocq", wires=3)

    @qml.qnode(dev)
    def circuit():
        qml.MultiControlledX(wires=[0, 1, 2], control_values=[True, False])
        return qml.state()

    circuit()
    sim = _FakeQuantumSimulator.instances[-1]

    assert sim.ops == [
        ("X", (1,), ()),
        ("MCX", (0, 1, 2), ()),
        ("X", (1,), ()),
    ]
    assert sim.matrices == []


def test_pennylane_parameter_shift_gradient_pipeline_runs(monkeypatch):
    pytest.importorskip("pennylane")
    _install_fake_binding(monkeypatch)
    for name in list(sys.modules):
        if name.startswith("pennylane_rocq"):
            sys.modules.pop(name)

    import pennylane as qml
    from pennylane import numpy as pnp

    dev = qml.device("lightning.rocq", wires=1)

    @qml.qnode(dev)
    def circuit(theta):
        qml.RY(theta, wires=0)
        return qml.expval(qml.PauliZ(0))

    theta = pnp.array(0.123, requires_grad=True)

    assert qml.grad(circuit)(theta) == pytest.approx(0.0)


def test_pennylane_explicit_adjoint_gradient_uses_captured_state(monkeypatch):
    pytest.importorskip("pennylane")

    class _RYPreservingSimulator(_FakeQuantumSimulator):
        def __init__(self, num_qubits):
            super().__init__(num_qubits)
            self._state = np.zeros(1 << self._num_qubits, dtype=np.complex128)
            self._state[0] = 1.0

        def reset(self):
            super().reset()
            self._state = np.zeros(1 << self._num_qubits, dtype=np.complex128)
            self._state[0] = 1.0

        def apply_gate(self, name, targets, params=None):
            super().apply_gate(name, targets, params)
            if name != "RY" or tuple(targets) != (0,):
                return
            theta = float((params or ())[0])
            c = math.cos(theta / 2.0)
            s = math.sin(theta / 2.0)
            old_state = self._state.copy()
            self._state[0] = c * old_state[0] - s * old_state[1]
            self._state[1] = s * old_state[0] + c * old_state[1]

        def _peek_statevector(self):
            return self._state.copy()

        def expectation_pauli_string(self, pauli_string, targets):
            self.expectations.append((pauli_string, tuple(targets)))
            if pauli_string == "Z" and tuple(targets) == (0,):
                return float(abs(self._state[0]) ** 2 - abs(self._state[1]) ** 2)
            return super().expectation_pauli_string(pauli_string, targets)

    fake = _install_fake_binding(monkeypatch)
    fake.QuantumSimulator = _RYPreservingSimulator
    fake.QSim = _RYPreservingSimulator
    for name in list(sys.modules):
        if name.startswith("pennylane_rocq"):
            sys.modules.pop(name)

    import pennylane as qml
    from pennylane import numpy as pnp
    from pennylane.devices import ExecutionConfig

    dev = qml.device("lightning.rocq", wires=1)
    assert dev.supports_derivatives(ExecutionConfig(gradient_method="adjoint"))
    assert not dev.supports_derivatives(ExecutionConfig(gradient_method="device"))

    @qml.qnode(dev, diff_method="adjoint")
    def circuit(theta):
        qml.RY(theta, wires=0)
        return qml.expval(qml.PauliZ(0))

    theta = pnp.array(0.321, requires_grad=True)

    assert circuit(theta) == pytest.approx(math.cos(float(theta)))
    assert qml.grad(circuit)(theta) == pytest.approx(-math.sin(float(theta)))
    sim = _RYPreservingSimulator.instances[-1]
    assert sim.expectations == [("Z", (0,))]
    assert sim.statevector_reads == 1


def test_pennylane_finite_shot_sample_and_counts_use_native_measure(monkeypatch):
    pytest.importorskip("pennylane")
    _install_fake_binding(monkeypatch)
    for name in list(sys.modules):
        if name.startswith("pennylane_rocq"):
            sys.modules.pop(name)

    import pennylane as qml
    from pennylane.exceptions import PennyLaneDeprecationWarning

    with pytest.warns(PennyLaneDeprecationWarning):
        sample_dev = qml.device("lightning.rocq", wires=2, shots=4)

    @qml.qnode(sample_dev)
    def sample_circuit():
        qml.Hadamard(wires=0)
        qml.CNOT(wires=[0, 1])
        return qml.sample(wires=[0, 1])

    samples = sample_circuit()
    np.testing.assert_array_equal(
        samples,
        np.array([[0, 0], [1, 1], [0, 0], [1, 1]], dtype=int),
    )
    sample_sim = _FakeQuantumSimulator.instances[-1]
    assert sample_sim.measurements == [((0, 1), 4)]
    assert sample_sim.statevector_reads == 0

    with pytest.warns(PennyLaneDeprecationWarning):
        counts_dev = qml.device("lightning.rocq", wires=2, shots=4)

    @qml.qnode(counts_dev)
    def counts_circuit():
        qml.Hadamard(wires=0)
        qml.CNOT(wires=[0, 1])
        return qml.counts(wires=[0, 1])

    counts = {str(key): int(value) for key, value in counts_circuit().items()}

    assert counts == {"00": 2, "11": 2}
    counts_sim = _FakeQuantumSimulator.instances[-1]
    assert counts_sim.measurements == [((0, 1), 4)]
    assert counts_sim.statevector_reads == 0


def test_pennylane_finite_shot_probs_uses_native_measure(monkeypatch):
    pytest.importorskip("pennylane")
    _install_fake_binding(monkeypatch)
    for name in list(sys.modules):
        if name.startswith("pennylane_rocq"):
            sys.modules.pop(name)

    import pennylane as qml
    from pennylane.exceptions import PennyLaneDeprecationWarning

    with pytest.warns(PennyLaneDeprecationWarning):
        dev = qml.device("lightning.rocq", wires=2, shots=4)

    @qml.qnode(dev)
    def probs_circuit():
        qml.Hadamard(wires=0)
        qml.CNOT(wires=[0, 1])
        return qml.probs(wires=[0, 1])

    np.testing.assert_allclose(probs_circuit(), np.array([0.5, 0.0, 0.0, 0.5]))
    sim = _FakeQuantumSimulator.instances[-1]
    assert sim.measurements == [((0, 1), 4)]
    assert sim.statevector_reads == 0


def test_pennylane_shot_vector_uses_total_native_measure(monkeypatch):
    pytest.importorskip("pennylane")
    _install_fake_binding(monkeypatch)
    for name in list(sys.modules):
        if name.startswith("pennylane_rocq"):
            sys.modules.pop(name)

    import pennylane as qml
    from pennylane.exceptions import PennyLaneDeprecationWarning

    with pytest.warns(PennyLaneDeprecationWarning):
        dev = qml.device("lightning.rocq", wires=1, shots=(2, 3))

    @qml.qnode(dev)
    def circuit():
        return qml.sample(wires=[0])

    first, second = circuit()

    np.testing.assert_array_equal(first, np.array([0, 0], dtype=int))
    np.testing.assert_array_equal(second, np.array([0, 0, 0], dtype=int))
    sim = _FakeQuantumSimulator.instances[-1]
    assert sim.measurements == [((0,), 5)]
    assert sim.statevector_reads == 0


def test_pennylane_probs_returns_full_and_marginal_probabilities(monkeypatch):
    pytest.importorskip("pennylane")
    _install_fake_binding(monkeypatch)
    for name in list(sys.modules):
        if name.startswith("pennylane_rocq"):
            sys.modules.pop(name)

    import pennylane as qml

    dev = qml.device("lightning.rocq", wires=2)

    @qml.qnode(dev)
    def full_probs():
        qml.Hadamard(wires=0)
        qml.CNOT(wires=[0, 1])
        return qml.probs(wires=[0, 1])

    @qml.qnode(dev)
    def marginal_probs():
        qml.Hadamard(wires=0)
        qml.CNOT(wires=[0, 1])
        return qml.probs(wires=[0])

    np.testing.assert_allclose(full_probs(), np.array([0.5, 0.0, 0.0, 0.5]))
    np.testing.assert_allclose(marginal_probs(), np.array([0.5, 0.5]))


def test_pennylane_analytic_probs_prefers_native_probability_hook(monkeypatch):
    pytest.importorskip("pennylane")
    _install_fake_binding(monkeypatch)
    _FakeQuantumSimulator.enable_probabilities = True
    for name in list(sys.modules):
        if name.startswith("pennylane_rocq"):
            sys.modules.pop(name)

    import pennylane as qml

    dev = qml.device("lightning.rocq", wires=2)

    @qml.qnode(dev)
    def full_probs():
        qml.Hadamard(wires=0)
        qml.CNOT(wires=[0, 1])
        return qml.probs(wires=[0, 1])

    @qml.qnode(dev)
    def marginal_probs():
        qml.Hadamard(wires=0)
        qml.CNOT(wires=[0, 1])
        return qml.probs(wires=[0])

    np.testing.assert_allclose(full_probs(), np.array([0.1, 0.2, 0.3, 0.4]))
    full_sim = _FakeQuantumSimulator.instances[-1]
    assert full_sim.probability_requests == [(0, 1)]
    assert full_sim.statevector_reads == 0

    np.testing.assert_allclose(marginal_probs(), np.array([1 / 3, 2 / 3]))
    marginal_sim = _FakeQuantumSimulator.instances[-1]
    assert marginal_sim.probability_requests == [(0,)]
    assert marginal_sim.statevector_reads == 0


def test_runtime_probabilities_falls_back_when_native_reports_not_implemented(monkeypatch):
    from rocquantum.framework_runtime import RocQuantumRuntime

    sim = _FakeQuantumSimulator(2)

    def unavailable_probabilities(qubits):
        sim.probability_requests.append(tuple(qubits))
        raise RuntimeError("hipStateVec error during probabilities (status 5)")

    monkeypatch.setattr(sim, "probabilities", unavailable_probabilities)
    runtime = RocQuantumRuntime(sim)

    np.testing.assert_allclose(runtime.probabilities([0, 1]), np.array([0.5, 0.0, 0.0, 0.5]))
    assert sim.probability_requests == [(0, 1)]
    assert sim.statevector_reads == 1


def test_runtime_can_create_and_read_batched_bindings():
    from rocquantum.framework_runtime import RocQuantumRuntime

    class _BatchedBindingSimulator:
        created = []

        def __init__(self, num_qubits, batch_size=1):
            self._num_qubits = int(num_qubits)
            self._batch_size = int(batch_size)
            self._states = np.zeros((self._batch_size, 1 << self._num_qubits), dtype=np.complex128)
            for batch_index in range(self._batch_size):
                self._states[batch_index, batch_index % self._states.shape[1]] = 1.0
            self.created.append((self._num_qubits, self._batch_size))

        def num_qubits(self):
            return self._num_qubits

        def batch_size(self):
            return self._batch_size

        def get_statevector(self, batch_index=0):
            return self._states[int(batch_index)].copy()

        def get_statevectors(self):
            return self._states.copy()

        def set_statevectors(self, states):
            self._states = np.asarray(states, dtype=np.complex128).reshape(
                self._batch_size,
                1 << self._num_qubits,
            )

        def apply_gate_batch(self, name, targets, params_by_batch):
            self.batch_ops = [(str(name), tuple(targets), tuple(float(param) for param in params_by_batch))]

    fake = types.ModuleType("rocquantum_bind")
    fake.QuantumSimulator = _BatchedBindingSimulator

    runtime = RocQuantumRuntime.from_bindings(2, binding_module=fake, batch_size=3)

    assert _BatchedBindingSimulator.created == [(2, 3)]
    assert runtime.batch_size() == 3
    np.testing.assert_allclose(runtime.statevector(1), np.array([0, 1, 0, 0], dtype=np.complex128))
    np.testing.assert_allclose(
        runtime.statevectors(),
        np.array(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
            ],
            dtype=np.complex128,
        ),
    )

    updated = np.zeros((3, 4), dtype=np.complex128)
    updated[:, 3] = 1.0
    runtime.set_statevectors(updated)
    np.testing.assert_allclose(runtime.statevectors(), updated)
    np.testing.assert_allclose(
        runtime.probabilities_batch([0]),
        np.array(
            [
                [0.0, 1.0],
                [0.0, 1.0],
                [0.0, 1.0],
            ],
            dtype=float,
        ),
    )
    np.testing.assert_allclose(
        runtime.expectation_pauli_string_batch("Z", [0]),
        np.array([-1.0, -1.0, -1.0], dtype=float),
    )
    np.testing.assert_allclose(
        runtime.expectation_matrix_batch(np.diag([1.0, -1.0]).astype(np.complex128), [0]),
        np.array([-1.0, -1.0, -1.0], dtype=np.complex128),
    )
    runtime.apply_operation_batch("ry", [0], [0.1, 0.2, 0.3])
    assert runtime.simulator.batch_ops == [("RY", (0,), (0.1, 0.2, 0.3))]
    runtime.apply_operation_batch("crz", [0, 1], [0.4, 0.5, 0.6])
    assert runtime.simulator.batch_ops == [("CRZ", (0, 1), (0.4, 0.5, 0.6))]
    runtime.apply_operation_batch("p", [0], [0.7, 0.8, 0.9])
    assert runtime.simulator.batch_ops == [("P", (0,), (0.7, 0.8, 0.9))]
    runtime.apply_operation_batch("cp", [0, 1], [1.0, 1.1, 1.2])
    assert runtime.simulator.batch_ops == [("CP", (0, 1), (1.0, 1.1, 1.2))]


def test_runtime_batch_parametric_gate_falls_back_for_equal_angles():
    from rocquantum.framework_runtime import RocQuantumRuntime

    class _EqualAngleSimulator:
        def __init__(self):
            self.ops = []

        def batch_size(self):
            return 3

        def apply_gate(self, name, targets, params):
            self.ops.append((str(name), tuple(targets), tuple(params)))

    sim = _EqualAngleSimulator()
    runtime = RocQuantumRuntime(sim)

    runtime.apply_operation_batch("rz", [1], [0.25, 0.25, 0.25])

    assert sim.ops == [("RZ", (1,), (0.25,))]


def test_pennylane_hamiltonian_expval_sums_native_pauli_terms(monkeypatch):
    pytest.importorskip("pennylane")
    _install_fake_binding(monkeypatch)
    for name in list(sys.modules):
        if name.startswith("pennylane_rocq"):
            sys.modules.pop(name)

    import pennylane as qml

    dev = qml.device("lightning.rocq", wires=2)
    hamiltonian = qml.Hamiltonian(
        [1.2, -0.5, 0.25],
        [qml.PauliZ(0), qml.PauliX(0) @ qml.PauliZ(1), qml.Identity(0)],
    )

    @qml.qnode(dev)
    def circuit():
        qml.Hadamard(wires=0)
        return qml.expval(hamiltonian)

    assert circuit() == pytest.approx(0.725)
    sim = _FakeQuantumSimulator.instances[-1]
    assert sim.expectations == [
        ("Z", (0,)),
        ("XZ", (0, 1)),
    ]
    assert sim.statevector_reads == 0


def test_pennylane_hamiltonian_expval_combines_duplicate_pauli_terms(monkeypatch):
    pytest.importorskip("pennylane")
    _install_fake_binding(monkeypatch)
    for name in list(sys.modules):
        if name.startswith("pennylane_rocq"):
            sys.modules.pop(name)

    import pennylane as qml

    dev = qml.device("lightning.rocq", wires=1)
    hamiltonian = qml.Hamiltonian(
        [0.5, 0.7, 0.25],
        [qml.PauliZ(0), qml.PauliZ(0), qml.Identity(0)],
    )

    @qml.qnode(dev)
    def circuit():
        return qml.expval(hamiltonian)

    assert circuit() == pytest.approx(0.85)
    sim = _FakeQuantumSimulator.instances[-1]
    assert sim.expectations == [("Z", (0,))]
    assert sim.statevector_reads == 0


def test_pennylane_hamiltonian_var_uses_native_pauli_products(monkeypatch):
    pytest.importorskip("pennylane")
    _install_fake_binding(monkeypatch)
    for name in list(sys.modules):
        if name.startswith("pennylane_rocq"):
            sys.modules.pop(name)

    import pennylane as qml

    dev = qml.device("lightning.rocq", wires=2)
    hamiltonian = qml.Hamiltonian(
        [1.2, -0.5, 0.25],
        [qml.PauliZ(0), qml.PauliX(0) @ qml.PauliZ(1), qml.Identity(0)],
    )

    @qml.qnode(dev)
    def circuit():
        qml.Hadamard(wires=0)
        return qml.var(hamiltonian)

    assert circuit() == pytest.approx(1.464375)
    sim = _FakeQuantumSimulator.instances[-1]
    assert sim.expectations == [
        ("Z", (0,)),
        ("XZ", (0, 1)),
        ("Z", (0,)),
        ("XZ", (0, 1)),
    ]
    assert sim.statevector_reads == 0


def test_root_package_declares_framework_entry_points():
    toml_reader = tomllib
    if toml_reader is None:
        toml_reader = pytest.importorskip("tomli")

    pyproject_path = os.path.join(_PROJECT_ROOT, "pyproject.toml")
    with open(pyproject_path, "rb") as f:
        pyproject = toml_reader.load(f)

    packages = pyproject["tool"]["scikit-build"]["wheel"]["packages"]
    assert packages["qiskit_rocquantum_provider"].endswith("qiskit_rocquantum_provider")
    assert packages["pennylane_rocq"].endswith("pennylane_rocq")

    entry_points = pyproject["project"]["entry-points"]
    pennylane_plugins = entry_points["pennylane.plugins"]
    assert pennylane_plugins["lightning.rocq"] == "pennylane_rocq:LightningRocqDevice"
    assert pennylane_plugins["lightning.rocm"] == "pennylane_rocq:LightningRocmDevice"
    assert entry_points["qiskit.providers"]["rocquantum"] == "qiskit_rocquantum_provider:RocQuantumProvider"


def test_statevector_expectation_fallback_handles_y_phase():
    from rocquantum.framework_runtime import expectation_from_statevector

    state = np.array([1.0, 1.0j], dtype=np.complex128) / math.sqrt(2.0)

    assert expectation_from_statevector(state, "Y", [0]) == pytest.approx(1.0)
