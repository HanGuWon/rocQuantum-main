from __future__ import annotations

import cmath
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
    reject_batch_gates = set()

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
        if str(name).upper() in self.reject_batch_gates:
            raise NotImplementedError(f"batch gate {name} disabled")
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

    def get_statevectors(self):
        self.statevector_reads += 1
        return np.tile(self._peek_statevector(), (self._batch_size, 1))

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
        data = np.asarray(data, dtype=np.complex128)
        indices = np.asarray(indices, dtype=np.int64)
        indptr = np.asarray(indptr, dtype=np.int64)
        shape = tuple(shape)
        self.sparse_moments.append(
            (
                data,
                indices,
                indptr,
                shape,
            )
        )
        state = self._peek_statevector()
        h_state = np.zeros_like(state)
        for row in range(int(shape[0])):
            for offset in range(int(indptr[row]), int(indptr[row + 1])):
                h_state[row] += data[offset] * state[int(indices[offset])]
        return np.vdot(state, h_state), np.vdot(h_state, h_state)

    def sparse_hamiltonian_moments_batch(self, data, indices, indptr, shape):
        if not self.enable_sparse_moments:
            raise NotImplementedError("native sparse Hamiltonian moments batch disabled")
        data = np.asarray(data, dtype=np.complex128)
        indices = np.asarray(indices, dtype=np.int64)
        indptr = np.asarray(indptr, dtype=np.int64)
        shape = tuple(shape)
        self.sparse_batch_moments.append(
            (
                data,
                indices,
                indptr,
                shape,
            )
        )
        mean, second_moment = self.sparse_hamiltonian_moments(data, indices, indptr, shape)
        self.sparse_moments.pop()
        return (
            np.full(self._batch_size, mean, dtype=np.complex128),
            np.full(self._batch_size, second_moment, dtype=np.complex128),
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
    _FakeQuantumSimulator.reject_batch_gates = set()
    return fake


def _single_qubit_sparse_x(sp):
    return sp.csr_matrix(np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128))


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


def test_qiskit_backend_batches_compatible_sampling_circuit_lists(monkeypatch):
    pytest.importorskip("qiskit")
    _install_fake_binding(monkeypatch)

    from qiskit import QuantumCircuit
    from qiskit_rocquantum_provider import RocQuantumProvider

    circuits = []
    for angle in (0.1, 0.2):
        circuit = QuantumCircuit(1, 1)
        circuit.ry(angle, 0)
        circuit.measure(0, 0)
        circuits.append(circuit)

    backend = RocQuantumProvider().get_backend("rocq_simulator")
    result = backend.run(circuits, shots=3, statevector=False).result()

    assert len(result.results) == 2
    counts = [result.get_counts(index) for index in range(2)]
    assert [sum(row.values()) for row in counts] == [3, 3]
    assert all(set(row).issubset({"0", "1"}) for row in counts)
    sim = _FakeQuantumSimulator.instances[-1]
    assert sim.batch_size() == 2
    assert sim.ops == []
    assert sim.batch_ops == [("RY", (0,), (0.1, 0.2))]
    assert sim.measurements == []
    assert sim.probability_requests == [(0,)]
    assert sim.statevector_reads == 0


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


def test_qiskit_backend_batches_compatible_statevector_circuit_lists(monkeypatch):
    pytest.importorskip("qiskit")
    _install_fake_binding(monkeypatch)

    from qiskit import QuantumCircuit
    from qiskit_rocquantum_provider import RocQuantumProvider

    circuits = []
    for angle in (0.1, 0.2):
        circuit = QuantumCircuit(1)
        circuit.ry(angle, 0)
        circuits.append(circuit)

    backend = RocQuantumProvider().get_backend("rocq_simulator")
    result = backend.run(circuits, sampling=False, statevector=True).result()

    assert len(result.results) == 2
    for index in range(2):
        np.testing.assert_allclose(
            np.asarray(result.results[index].data.statevector, dtype=np.complex128),
            np.array([1.0, 0.0], dtype=np.complex128),
        )
        assert result.results[index].shots == 0

    sim = _FakeQuantumSimulator.instances[-1]
    assert sim.batch_size() == 2
    assert sim.ops == []
    assert sim.batch_ops == [("RY", (0,), (0.1, 0.2))]
    assert sim.measurements == []
    assert sim.statevector_reads == 1


def test_qiskit_backend_batches_statevector_lists_with_global_phase(monkeypatch):
    pytest.importorskip("qiskit")
    _install_fake_binding(monkeypatch)

    from qiskit import QuantumCircuit
    from qiskit_rocquantum_provider import RocQuantumProvider

    circuits = []
    for angle, phase in ((0.1, 0.25), (0.2, 0.5)):
        circuit = QuantumCircuit(1)
        circuit.global_phase = phase
        circuit.ry(angle, 0)
        circuits.append(circuit)

    backend = RocQuantumProvider().get_backend("rocq_simulator")
    result = backend.run(circuits, sampling=False, statevector=True).result()

    assert len(result.results) == 2
    for index, phase in enumerate((0.25, 0.5)):
        np.testing.assert_allclose(
            np.asarray(result.results[index].data.statevector, dtype=np.complex128),
            np.array([cmath.exp(1j * phase), 0.0], dtype=np.complex128),
        )
    sim = _FakeQuantumSimulator.instances[-1]
    assert sim.batch_size() == 2
    assert sim.batch_ops == [("RY", (0,), (0.1, 0.2))]
    assert sim.statevector_reads == 1


def test_qiskit_backend_batches_statevector_phase_lists_with_native_phase_batch(monkeypatch):
    pytest.importorskip("qiskit")
    _install_fake_binding(monkeypatch)

    from qiskit import QuantumCircuit
    from qiskit_rocquantum_provider import RocQuantumProvider

    circuits = []
    for angle in (0.1, 0.2):
        circuit = QuantumCircuit(1)
        circuit.p(angle, 0)
        circuits.append(circuit)

    backend = RocQuantumProvider().get_backend("rocq_simulator")
    result = backend.run(circuits, sampling=False, statevector=True).result()

    assert len(result.results) == 2
    sim = _FakeQuantumSimulator.instances[-1]
    assert sim.batch_size() == 2
    assert sim.batch_ops == [("P", (0,), (0.1, 0.2))]
    assert sim.statevector_reads == 1


def test_qiskit_backend_does_not_batch_statevector_phase_lists_without_native_phase(monkeypatch):
    pytest.importorskip("qiskit")
    _install_fake_binding(monkeypatch)
    _FakeQuantumSimulator.reject_batch_gates = {"P"}

    from qiskit import QuantumCircuit
    from qiskit_rocquantum_provider import RocQuantumProvider

    circuits = []
    for angle in (0.1, 0.2):
        circuit = QuantumCircuit(1)
        circuit.p(angle, 0)
        circuits.append(circuit)

    before = len(_FakeQuantumSimulator.instances)
    backend = RocQuantumProvider().get_backend("rocq_simulator")
    result = backend.run(circuits, sampling=False, statevector=True).result()

    assert len(result.results) == 2
    sims = _FakeQuantumSimulator.instances[before:]
    assert sims
    assert _FakeQuantumSimulator.instances[-1].batch_size() == 1
    assert _FakeQuantumSimulator.instances[-1].batch_ops == []


def test_qiskit_backend_batches_statevector_controlled_phase_lists_with_native_phase_batch(monkeypatch):
    pytest.importorskip("qiskit")
    _install_fake_binding(monkeypatch)

    from qiskit import QuantumCircuit
    from qiskit_rocquantum_provider import RocQuantumProvider

    circuits = []
    for angle in (0.1, 0.2):
        circuit = QuantumCircuit(2)
        circuit.cp(angle, 0, 1)
        circuits.append(circuit)

    backend = RocQuantumProvider().get_backend("rocq_simulator")
    result = backend.run(circuits, sampling=False, statevector=True).result()

    assert len(result.results) == 2
    sim = _FakeQuantumSimulator.instances[-1]
    assert sim.batch_size() == 2
    assert sim.batch_ops == [("CP", (0, 1), (0.1, 0.2))]
    assert sim.statevector_reads == 1


def test_qiskit_backend_does_not_batch_statevector_controlled_phase_lists_without_native_phase(monkeypatch):
    pytest.importorskip("qiskit")
    _install_fake_binding(monkeypatch)
    _FakeQuantumSimulator.reject_batch_gates = {"CP"}

    from qiskit import QuantumCircuit
    from qiskit_rocquantum_provider import RocQuantumProvider

    circuits = []
    for angle in (0.1, 0.2):
        circuit = QuantumCircuit(2)
        circuit.cp(angle, 0, 1)
        circuits.append(circuit)

    before = len(_FakeQuantumSimulator.instances)
    backend = RocQuantumProvider().get_backend("rocq_simulator")
    result = backend.run(circuits, sampling=False, statevector=True).result()

    assert len(result.results) == 2
    sims = _FakeQuantumSimulator.instances[before:]
    assert sims
    assert _FakeQuantumSimulator.instances[-1].batch_size() == 1
    assert _FakeQuantumSimulator.instances[-1].batch_ops == []


def test_qiskit_backend_batches_statevector_pauli_rotation_lists(monkeypatch):
    pytest.importorskip("qiskit")
    _install_fake_binding(monkeypatch)

    from qiskit import QuantumCircuit
    from qiskit_rocquantum_provider import RocQuantumProvider

    circuits = []
    for angle in (0.1, 0.2):
        circuit = QuantumCircuit(2)
        circuit.rxx(angle, 0, 1)
        circuit.ryy(angle + 0.1, 0, 1)
        circuit.rzz(angle + 0.2, 0, 1)
        circuit.rzx(angle + 0.3, 0, 1)
        circuits.append(circuit)

    backend = RocQuantumProvider().get_backend("rocq_simulator")
    result = backend.run(circuits, sampling=False, statevector=True).result()

    assert len(result.results) == 2
    sim = _FakeQuantumSimulator.instances[-1]
    assert sim.batch_size() == 2
    assert sim.statevector_reads == 1
    assert sim.matrices == []
    assert sim.batch_ops == [
        ("RX", (0,), (0.1, 0.2)),
        ("RZ", (1,), (0.2, 0.30000000000000004)),
        ("RZ", (1,), (0.30000000000000004, 0.4)),
        ("RZ", (1,), (0.4, 0.5)),
    ]


def test_qiskit_backend_batches_statevector_pauli_evolution_lists(monkeypatch):
    pytest.importorskip("qiskit")
    _install_fake_binding(monkeypatch)

    from qiskit import QuantumCircuit
    from qiskit.circuit.library import PauliEvolutionGate
    from qiskit.quantum_info import SparsePauliOp
    from qiskit_rocquantum_provider import RocQuantumProvider

    circuits = []
    for time in (0.1, 0.2):
        circuit = QuantumCircuit(2)
        circuit.append(
            PauliEvolutionGate(SparsePauliOp.from_list([("ZZ", 0.5)]), time=time),
            [0, 1],
        )
        circuits.append(circuit)

    backend = RocQuantumProvider().get_backend("rocq_simulator")
    result = backend.run(circuits, sampling=False, statevector=True).result()

    assert len(result.results) == 2
    sim = _FakeQuantumSimulator.instances[-1]
    assert sim.batch_size() == 2
    assert sim.ops == [("CNOT", (0, 1), ()), ("CNOT", (0, 1), ())]
    assert sim.batch_ops == [("RZ", (1,), (0.1, 0.2))]
    assert sim.statevector_reads == 1


def test_qiskit_backend_batches_statevector_pauli_evolution_identity_terms(monkeypatch):
    pytest.importorskip("qiskit")
    _install_fake_binding(monkeypatch)

    from qiskit import QuantumCircuit
    from qiskit.circuit.library import PauliEvolutionGate
    from qiskit.quantum_info import SparsePauliOp
    from qiskit_rocquantum_provider import RocQuantumProvider

    circuits = []
    for time in (0.1, 0.2):
        circuit = QuantumCircuit(1)
        circuit.append(
            PauliEvolutionGate(SparsePauliOp.from_list([("I", 0.5), ("Z", 1.0)]), time=time),
            [0],
        )
        circuits.append(circuit)

    backend = RocQuantumProvider().get_backend("rocq_simulator")
    result = backend.run(circuits, sampling=False, statevector=True).result()

    assert len(result.results) == 2
    for index, time in enumerate((0.1, 0.2)):
        np.testing.assert_allclose(
            np.asarray(result.results[index].data.statevector, dtype=np.complex128),
            np.array([cmath.exp(-0.5j * time), 0.0], dtype=np.complex128),
        )
    sim = _FakeQuantumSimulator.instances[-1]
    assert sim.batch_size() == 2
    assert sim.batch_ops == [("RZ", (0,), (0.2, 0.4))]
    assert sim.statevector_reads == 1


def test_qiskit_backend_batches_statevector_lists_with_fixed_pauli_and_unitary(monkeypatch):
    pytest.importorskip("qiskit")
    _install_fake_binding(monkeypatch)

    from qiskit import QuantumCircuit
    from qiskit.circuit.library import PauliGate
    from qiskit_rocquantum_provider import RocQuantumProvider

    unitary = np.array([[0, 1], [1, 0]], dtype=np.complex128)
    circuits = []
    for angle in (0.1, 0.2):
        circuit = QuantumCircuit(2)
        circuit.unitary(unitary, [0])
        circuit.append(PauliGate("XZ"), [0, 1])
        circuit.ry(angle, 1)
        circuits.append(circuit)

    backend = RocQuantumProvider().get_backend("rocq_simulator")
    result = backend.run(circuits, sampling=False, statevector=True).result()

    assert len(result.results) == 2
    sim = _FakeQuantumSimulator.instances[-1]
    assert sim.batch_size() == 2
    assert sim.statevector_reads == 1
    assert sim.ops == [("Z", (0,), ()), ("X", (1,), ())]
    matrix, targets = sim.matrices[0]
    np.testing.assert_allclose(matrix, unitary)
    assert targets == (0,)
    assert sim.batch_ops == [("RY", (1,), (0.1, 0.2))]


def test_framework_runtime_converts_full_statevectors_to_little_endian_order():
    from rocquantum.framework_runtime import statevector_to_little_endian_wires

    state = np.array([0, 1, 2, 3], dtype=np.complex128)

    np.testing.assert_allclose(
        statevector_to_little_endian_wires(state),
        np.array([0, 2, 1, 3], dtype=np.complex128),
    )


def test_framework_runtime_rejects_nonfinite_statevector_uploads(monkeypatch):
    _install_fake_binding(monkeypatch)

    from rocquantum.framework_runtime import RocQuantumRuntime, statevector_to_little_endian_wires

    invalid_states = (
        np.array([1.0, np.nan], dtype=np.complex128),
        np.array([1.0, np.inf], dtype=np.complex128),
        np.array([1.0, 1.0j * np.inf], dtype=np.complex128),
        [True, 0.0],
        ["1.0", 0.0],
    )
    for state in invalid_states:
        with pytest.raises(ValueError, match="Statevector amplitudes"):
            statevector_to_little_endian_wires(state)

    runtime = RocQuantumRuntime.from_bindings(1)
    sim = _FakeQuantumSimulator.instances[-1]
    for state in invalid_states:
        with pytest.raises(ValueError, match="Statevector amplitudes"):
            runtime.set_statevector(state)
    assert sim.statevectors == []

    batch_runtime = RocQuantumRuntime.from_bindings(1, batch_size=2)
    batch_sim = _FakeQuantumSimulator.instances[-1]
    invalid_batched_states = (
        np.array([[1.0, 0.0], [np.nan, 0.0]], dtype=np.complex128),
        [[1.0, 0.0], [True, 0.0]],
        [[1.0, 0.0], ["1.0", 0.0]],
    )
    for states in invalid_batched_states:
        with pytest.raises(ValueError, match="Statevector amplitudes"):
            batch_runtime.set_statevectors(states)
    assert batch_sim.statevectors == []


def test_framework_runtime_rejects_nonfinite_matrix_and_sparse_payloads(monkeypatch):
    _install_fake_binding(monkeypatch)

    from rocquantum.framework_runtime import RocQuantumRuntime, matrix_to_little_endian_wires

    invalid_matrix = np.array([[1.0, np.nan], [0.0, 1.0]], dtype=np.complex128)
    with pytest.raises(ValueError, match="Operation matrix"):
        matrix_to_little_endian_wires(invalid_matrix)

    runtime = RocQuantumRuntime.from_bindings(2)
    sim = _FakeQuantumSimulator.instances[-1]
    with pytest.raises(ValueError, match="Operation matrix"):
        runtime.apply_matrix(invalid_matrix, [0])
    assert sim.matrices == []

    with pytest.raises(ValueError, match="Controlled operation matrix"):
        runtime.apply_controlled_matrix(invalid_matrix, [0], [1])
    assert sim.controlled_matrices == []

    _FakeQuantumSimulator.enable_matrix_expectation = True
    with pytest.raises(ValueError, match="Dense expectation matrix"):
        runtime.expectation_matrix(invalid_matrix, [0])
    with pytest.raises(ValueError, match="Dense expectation matrix"):
        runtime.expectation_matrix_batch(invalid_matrix, [0])
    assert sim.matrix_expectations == []
    assert sim.matrix_batch_expectations == []

    with pytest.raises(ValueError, match="Sparse operation CSR data"):
        runtime.apply_sparse_matrix(
            np.array([np.nan], dtype=np.complex128),
            np.array([0], dtype=np.int64),
            np.array([0, 1, 1], dtype=np.int64),
            (2, 2),
            [0],
        )
    assert sim.statevector_reads == 0
    assert sim.statevectors == []

    _FakeQuantumSimulator.enable_sparse_moments = True
    with pytest.raises(ValueError, match="Sparse Hamiltonian CSR data"):
        runtime.sparse_hamiltonian_moments(
            np.array([np.inf], dtype=np.complex128),
            np.array([0], dtype=np.int64),
            np.array([0, 1, 1, 1, 1], dtype=np.int64),
            (4, 4),
        )
    assert sim.sparse_moments == []


def test_framework_runtime_validates_matrix_shapes_and_sparse_csr_before_native_dispatch(monkeypatch):
    _install_fake_binding(monkeypatch)

    from rocquantum.framework_runtime import RocQuantumRuntime

    runtime = RocQuantumRuntime.from_bindings(2)
    sim = _FakeQuantumSimulator.instances[-1]

    invalid_matrix_cases = (
        (np.eye(2, dtype=np.complex128), [0, 1]),
        (np.eye(4, dtype=np.complex128), [0, 0]),
        (np.eye(2, dtype=np.complex128), []),
        (np.eye(2, dtype=np.complex128), [-1]),
        ([[1.0, 0.0], [0.0, True]], [0]),
        ([[1.0, 0.0], [0.0, "1.0"]], [0]),
    )
    for matrix, targets in invalid_matrix_cases:
        with pytest.raises(ValueError):
            runtime.apply_matrix(matrix, targets)
    assert sim.matrices == []

    invalid_controlled_cases = (
        (np.eye(2, dtype=np.complex128), [], [1]),
        (np.eye(2, dtype=np.complex128), [0], [0]),
        (np.eye(2, dtype=np.complex128), [0, 0], [1]),
        (np.eye(4, dtype=np.complex128), [0], [1]),
        ([[1.0, 0.0], [0.0, True]], [0], [1]),
    )
    for matrix, controls, targets in invalid_controlled_cases:
        with pytest.raises(ValueError):
            runtime.apply_controlled_matrix(matrix, controls, targets)
    assert sim.controlled_matrices == []

    class _NativeSparseSimulator(_FakeQuantumSimulator):
        def __init__(self, num_qubits):
            super().__init__(num_qubits)
            self.sparse_applications = []

        def apply_sparse_matrix(self, data, indices, indptr, shape, targets):
            self.sparse_applications.append((data, indices, indptr, shape, targets))

    sparse_sim = _NativeSparseSimulator(2)
    sparse_runtime = RocQuantumRuntime(sparse_sim)
    invalid_sparse_cases = (
        ([True], [0], [0, 1], (2, 2), [0]),
        ([1.0], [True], [0, 1], (2, 2), [0]),
        ([1.0], [-1], [0, 1], (2, 2), [0]),
        ([1.0], [0], [0, True], (2, 2), [0]),
        ([1.0], [0], [0, 1], (4, 4), [0]),
        ([1.0], [0], [0, 1], (2, 2), [0, 0]),
        ([1.0, 2.0], [0], [0, 1], (2, 2), [0]),
        ([1.0], [0], [1, 1], (2, 2), [0]),
        ([1.0], [0], [0, 1, 0], (2, 2), [0]),
        ([1.0], [2], [0, 1], (2, 2), [0]),
    )
    for data, indices, indptr, shape, targets in invalid_sparse_cases:
        with pytest.raises(ValueError):
            sparse_runtime.apply_sparse_matrix(data, indices, indptr, shape, targets)

    assert sparse_sim.sparse_applications == []
    assert sparse_sim.statevector_reads == 0


def test_framework_runtime_revalidates_sparse_hamiltonian_moments_before_native_dispatch(monkeypatch):
    _install_fake_binding(monkeypatch)

    from rocquantum.framework_runtime import (
        RocQuantumRuntime,
        sparse_hamiltonian_moments_from_statevector,
    )

    _FakeQuantumSimulator.enable_sparse_moments = True
    runtime = RocQuantumRuntime.from_bindings(1)
    sim = _FakeQuantumSimulator.instances[-1]

    invalid_sparse_moments = (
        ([True], [0], [0, 1, 1], (2, 2)),
        (["1.0"], [0], [0, 1, 1], (2, 2)),
        ([np.nan], [0], [0, 1, 1], (2, 2)),
        ([1.0], [True], [0, 1, 1], (2, 2)),
        ([1.0], ["0"], [0, 1, 1], (2, 2)),
        ([1.0], [0], [0, True, 1], (2, 2)),
        ([1.0], [0], [0, 1, 1], (True, 2)),
        ([1.0], [0], [0, 1, 1], "22"),
        ([1.0], [2], [0, 1, 1], (2, 2)),
    )

    for data, indices, indptr, shape in invalid_sparse_moments:
        with pytest.raises(ValueError, match="Sparse Hamiltonian"):
            runtime.sparse_hamiltonian_moments(data, indices, indptr, shape)
        with pytest.raises(ValueError, match="Sparse Hamiltonian"):
            sparse_hamiltonian_moments_from_statevector(
                np.array([1.0, 0.0], dtype=np.complex128),
                data,
                indices,
                indptr,
                shape,
            )

    assert sim.sparse_moments == []

    batch_runtime = RocQuantumRuntime.from_bindings(1, batch_size=2)
    batch_sim = _FakeQuantumSimulator.instances[-1]
    for data, indices, indptr, shape in invalid_sparse_moments:
        with pytest.raises(ValueError, match="Sparse Hamiltonian"):
            batch_runtime.sparse_hamiltonian_moments_batch(data, indices, indptr, shape)

    assert batch_sim.sparse_moments == []
    assert batch_sim.sparse_batch_moments == []


def test_framework_runtime_revalidates_pauli_expectation_payloads_before_native_dispatch(monkeypatch):
    _install_fake_binding(monkeypatch)

    from rocquantum.framework_runtime import (
        RocQuantumRuntime,
        expectation_from_statevector,
    )

    runtime = RocQuantumRuntime.from_bindings(2)
    sim = _FakeQuantumSimulator.instances[-1]

    invalid_pauli_payloads = (
        (None, [0]),
        (b"Z", [0]),
        ("A", [0]),
        ("ZZ", [0]),
        ("Z", [0, 0]),
        ("Z", [-1]),
        ("Z", [2]),
        ("Z", [True]),
        ("Z", ["0"]),
    )

    for pauli_string, targets in invalid_pauli_payloads:
        with pytest.raises(ValueError):
            runtime.expectation_pauli_string(pauli_string, targets)
        with pytest.raises(ValueError):
            expectation_from_statevector(
                np.array([1.0, 0.0, 0.0, 0.0], dtype=np.complex128),
                pauli_string,
                targets,
            )

    assert sim.expectations == []

    with pytest.raises(ValueError):
        runtime.expectation_value("A", 0)
    with pytest.raises(ValueError):
        runtime.expectation_value("Z", True)
    with pytest.raises(ValueError):
        runtime.expectation_value("Z", 2)
    assert sim.expectations == []

    batch_runtime = RocQuantumRuntime.from_bindings(2, batch_size=2)
    batch_sim = _FakeQuantumSimulator.instances[-1]
    for pauli_string, targets in invalid_pauli_payloads:
        with pytest.raises(ValueError):
            batch_runtime.expectation_pauli_string_batch(pauli_string, targets)

    assert batch_sim.batch_expectations == []


def test_framework_runtime_revalidates_native_pauli_expectation_results():
    from rocquantum.framework_runtime import (
        RocQuantumRuntime,
        normalize_real_result_scalar,
        normalize_real_result_vector,
    )

    invalid_scalars = (
        True,
        np.bool_(False),
        "0.5",
        b"0.5",
        np.nan,
        np.inf,
        -np.inf,
        0.5 + 0.0j,
        [0.5, 0.25],
    )
    for value in invalid_scalars:
        with pytest.raises(ValueError, match="Pauli expectation"):
            normalize_real_result_scalar(value, "Pauli expectation value")

    invalid_vectors = (
        [0.5, True],
        [0.5, "0.25"],
        [0.5, np.nan],
        [0.5],
        [0.5, 0.25, 0.0],
    )
    for values in invalid_vectors:
        with pytest.raises(ValueError, match="Batched Pauli"):
            normalize_real_result_vector(values, "Batched Pauli expectation values", expected_count=2)

    class _BadNativePauliExpectationSimulator:
        def __init__(self, value):
            self.value = value
            self.value_calls = []
            self.string_calls = []

        def num_qubits(self):
            return 1

        def expectation_value(self, pauli, target):
            self.value_calls.append((pauli, int(target)))
            return self.value

        def expectation_pauli_string(self, pauli_string, targets):
            self.string_calls.append((pauli_string, tuple(targets)))
            return self.value

    for value in invalid_scalars:
        sim = _BadNativePauliExpectationSimulator(value)
        runtime = RocQuantumRuntime(sim)
        with pytest.raises(ValueError, match="Pauli expectation"):
            runtime.expectation_value("Z", 0)
        with pytest.raises(ValueError, match="Pauli expectation"):
            runtime.expectation_pauli_string("Z", [0])
        assert sim.value_calls == [("Z", 0)]
        assert sim.string_calls == [("Z", (0,))]

    class _BadNativeBatchPauliExpectationSimulator:
        def __init__(self, values):
            self.values = values
            self.calls = []

        def num_qubits(self):
            return 1

        def batch_size(self):
            return 2

        def expectation_pauli_string_batch(self, pauli_string, targets):
            self.calls.append((pauli_string, tuple(targets)))
            return self.values

    for values in invalid_vectors:
        sim = _BadNativeBatchPauliExpectationSimulator(values)
        with pytest.raises(ValueError, match="Batched Pauli"):
            RocQuantumRuntime(sim).expectation_pauli_string_batch("Z", [0])
        assert sim.calls == [("Z", (0,))]


def test_framework_runtime_rejects_nonfinite_probability_payloads():
    from rocquantum.framework_runtime import (
        RocQuantumRuntime,
        probabilities_from_statevector,
        sample_indices_batch_from_probabilities,
        sample_indices_from_probabilities,
    )

    with pytest.raises(ValueError, match="Sampler probabilities"):
        sample_indices_from_probabilities([0.5, np.nan], 4)
    with pytest.raises(ValueError, match="Sampler probabilities"):
        sample_indices_from_probabilities([True, 0.0], 4)
    with pytest.raises(ValueError, match="Sampler probabilities"):
        sample_indices_from_probabilities(["0.5", 0.5], 4)
    with pytest.raises(ValueError, match="Batched sampler probabilities"):
        sample_indices_batch_from_probabilities([[1.0, 0.0], [np.inf, 0.0]], 4)
    with pytest.raises(ValueError, match="Batched sampler probabilities"):
        sample_indices_batch_from_probabilities([[1.0, 0.0], [True, 0.0]], 4)
    with pytest.raises(ValueError, match="Statevector amplitudes"):
        probabilities_from_statevector(np.array([1.0, np.nan], dtype=np.complex128))
    with pytest.raises(ValueError, match="Statevector amplitudes"):
        probabilities_from_statevector([True, 0.0])

    class _BadProbabilitySimulator:
        def __init__(self):
            self.statevector_reads = 0

        def batch_size(self):
            return 2

        def probabilities(self, qubits):
            return np.array([0.5, np.nan], dtype=float)

        def probabilities_batch(self, qubits):
            return np.array([[1.0, 0.0], [np.inf, 0.0]], dtype=float)

        def get_statevector(self):
            self.statevector_reads += 1
            return np.array([1.0, 0.0], dtype=np.complex128)

    simulator = _BadProbabilitySimulator()
    runtime = RocQuantumRuntime(simulator)

    with pytest.raises(ValueError, match="Probability vector"):
        runtime.probabilities([0])
    with pytest.raises(ValueError, match="Batched probability vector"):
        runtime.probabilities_batch([0])
    with pytest.raises(ValueError, match="Batched probability vector"):
        runtime.measure_batch([0], 4)
    assert simulator.statevector_reads == 0


def test_framework_runtime_rejects_ambiguous_gate_parameters(monkeypatch):
    _install_fake_binding(monkeypatch)

    from rocquantum.framework_runtime import RocQuantumRuntime, normalize_params

    invalid_values = (True, np.bool_(False), "0.25", b"0.25", np.nan, np.inf, -np.inf, 0.25 + 0.0j)
    for value in invalid_values:
        with pytest.raises((TypeError, ValueError), match="Operation parameters"):
            normalize_params([value])

    np.testing.assert_allclose(normalize_params([np.float64(0.25), 1]), [0.25, 1.0])

    runtime = RocQuantumRuntime.from_bindings(1)
    sim = _FakeQuantumSimulator.instances[-1]
    for value in invalid_values:
        with pytest.raises((TypeError, ValueError), match="Operation parameters"):
            runtime.apply_operation("RX", [0], [value])
    assert sim.ops == []

    batch_runtime = RocQuantumRuntime.from_bindings(1, batch_size=2)
    batch_sim = _FakeQuantumSimulator.instances[-1]
    for value in invalid_values:
        with pytest.raises((TypeError, ValueError), match="Operation parameters"):
            batch_runtime.apply_operation_batch("RX", [0], [0.1, value])
    assert batch_sim.batch_ops == []


def test_framework_runtime_rejects_ambiguous_targets_before_native_dispatch(monkeypatch):
    _install_fake_binding(monkeypatch)

    from rocquantum.framework_runtime import RocQuantumRuntime, normalize_targets

    invalid_targets = (True, np.bool_(False), "0", b"0", 0.0, 0.5, 0.0 + 0.0j)
    for target in invalid_targets:
        with pytest.raises((TypeError, ValueError), match="Operation targets"):
            normalize_targets([target])

    with pytest.raises(ValueError, match="Operation targets"):
        normalize_targets("0")
    assert normalize_targets([np.int64(1), 0]) == [1, 0]

    runtime = RocQuantumRuntime.from_bindings(2)
    sim = _FakeQuantumSimulator.instances[-1]
    for target in invalid_targets:
        with pytest.raises((TypeError, ValueError), match="Operation targets"):
            runtime.apply_operation("X", [target])
        with pytest.raises((TypeError, ValueError), match="Operation targets"):
            runtime.apply_matrix(np.eye(2, dtype=np.complex128), [target])
        with pytest.raises((TypeError, ValueError), match="Operation targets"):
            runtime.measure([target], 1)
    assert sim.ops == []
    assert sim.matrices == []
    assert sim.measurements == []

    batch_runtime = RocQuantumRuntime.from_bindings(2, batch_size=2)
    batch_sim = _FakeQuantumSimulator.instances[-1]
    with pytest.raises((TypeError, ValueError), match="Operation targets"):
        batch_runtime.apply_operation_batch("RX", [0.0], [0.1, 0.2])
    with pytest.raises((TypeError, ValueError), match="Operation targets"):
        batch_runtime.probabilities_batch([True])
    assert batch_sim.batch_ops == []
    assert batch_sim.probability_requests == []


def test_framework_runtime_validates_qubit_subsets_before_native_dispatch(monkeypatch):
    _install_fake_binding(monkeypatch)

    from rocquantum.framework_runtime import (
        RocQuantumRuntime,
        normalize_qubit_subset,
        probabilities_from_statevector,
    )

    assert normalize_qubit_subset([np.int64(1), 0], 2, "Qubit targets") == [1, 0]
    assert normalize_qubit_subset([], 2, "Qubit targets", allow_empty=True) == []
    with pytest.raises(ValueError, match="Qubit targets"):
        normalize_qubit_subset([], 2, "Qubit targets", allow_empty=False)
    with pytest.raises(ValueError, match="Qubit targets"):
        normalize_qubit_subset([0, 0], 2, "Qubit targets")
    with pytest.raises(ValueError, match="Qubit targets"):
        normalize_qubit_subset([-1], 2, "Qubit targets")
    with pytest.raises(ValueError, match="Qubit targets"):
        normalize_qubit_subset([2], 2, "Qubit targets")

    state = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.complex128)
    for probability_qubits in ([0, 0], [-1], [2]):
        with pytest.raises(ValueError, match="Probability qubits"):
            probabilities_from_statevector(state, probability_qubits)
    np.testing.assert_allclose(probabilities_from_statevector(state, []), np.array([1.0, 0.0, 0.0, 0.0]))

    runtime = RocQuantumRuntime.from_bindings(2)
    sim = _FakeQuantumSimulator.instances[-1]
    for measure_qubits in ([], [0, 0], [-1], [2]):
        with pytest.raises(ValueError, match="Measurement qubits"):
            runtime.measure(measure_qubits, 1)
    assert sim.measurements == []

    batch_runtime = RocQuantumRuntime.from_bindings(2, batch_size=2)
    batch_sim = _FakeQuantumSimulator.instances[-1]
    for measure_qubits in ([], [0, 0], [-1], [2]):
        with pytest.raises(ValueError, match="Measurement qubits"):
            batch_runtime.measure_batch(measure_qubits, 1)
    assert batch_sim.measurements == []
    assert batch_sim.probability_requests == []

    _FakeQuantumSimulator.enable_probabilities = True
    probability_runtime = RocQuantumRuntime.from_bindings(2)
    probability_sim = _FakeQuantumSimulator.instances[-1]
    for probability_qubits in ([0, 0], [-1], [2]):
        with pytest.raises(ValueError, match="Probability qubits"):
            probability_runtime.probabilities(probability_qubits)
    assert probability_sim.probability_requests == []

    probability_batch_runtime = RocQuantumRuntime.from_bindings(2, batch_size=2)
    probability_batch_sim = _FakeQuantumSimulator.instances[-1]
    for probability_qubits in ([0, 0], [-1], [2]):
        with pytest.raises(ValueError, match="Probability qubits"):
            probability_batch_runtime.probabilities_batch(probability_qubits)
    assert probability_batch_sim.probability_requests == []


def test_framework_runtime_validates_single_qubit_measure_reset_payloads(monkeypatch):
    _install_fake_binding(monkeypatch)

    from rocquantum.framework_runtime import (
        RocQuantumRuntime,
        normalize_measurement_bit,
        normalize_single_qubit,
    )

    assert normalize_single_qubit(np.int64(1), 2, "Qubit target") == 1
    assert normalize_measurement_bit(np.int64(1)) == 1

    invalid_targets = (True, np.bool_(False), "0", b"0", 0.0, 0.5, -1, 2)
    for target in invalid_targets:
        with pytest.raises(ValueError):
            normalize_single_qubit(target, 2, "Qubit target")

    invalid_bits = (True, np.bool_(False), "1", b"1", 0.0, 0.5, -1, 2)
    for bit in invalid_bits:
        with pytest.raises(ValueError, match="Measurement bit"):
            normalize_measurement_bit(bit)

    runtime = RocQuantumRuntime.from_bindings(2)
    sim = _FakeQuantumSimulator.instances[-1]
    for target in invalid_targets:
        with pytest.raises(ValueError):
            runtime.measure_qubit(target)
        with pytest.raises(ValueError):
            runtime.reset_qubit(target)
    assert sim.measure_qubits == []
    assert sim.reset_qubits == []

    class _BadMeasureQubitResultSimulator:
        def __init__(self, result):
            self.result = result
            self.calls = []

        def num_qubits(self):
            return 1

        def measure_qubit(self, target):
            self.calls.append(int(target))
            return self.result

    for result in invalid_bits:
        sim = _BadMeasureQubitResultSimulator(result)
        with pytest.raises(ValueError, match="Measurement bit"):
            RocQuantumRuntime(sim).measure_qubit(0)
        assert sim.calls == [0]

    valid_sim = _BadMeasureQubitResultSimulator(np.int64(1))
    assert RocQuantumRuntime(valid_sim).measure_qubit(0) == 1
    assert valid_sim.calls == [0]


def test_framework_runtime_rejects_ambiguous_shots_before_native_dispatch(monkeypatch):
    _install_fake_binding(monkeypatch)

    from rocquantum.framework_runtime import (
        RocQuantumRuntime,
        normalize_shots,
        sample_indices_batch_from_probabilities,
        sample_indices_from_probabilities,
        sample_rows_from_statevector,
    )

    invalid_shots = (0, -1, True, np.bool_(False), 1.5, "4", b"4", 1.0 + 0.0j)
    for shots in invalid_shots:
        with pytest.raises(ValueError, match="shots must"):
            normalize_shots(shots)
        with pytest.raises(ValueError, match="shots must"):
            sample_rows_from_statevector(np.array([1.0, 0.0], dtype=np.complex128), shots)
        with pytest.raises(ValueError, match="shots must"):
            sample_indices_from_probabilities([1.0, 0.0], shots)
        with pytest.raises(ValueError, match="shots must"):
            sample_indices_batch_from_probabilities([[1.0, 0.0]], shots)

    assert normalize_shots(np.int64(3)) == 3

    runtime = RocQuantumRuntime.from_bindings(1)
    sim = _FakeQuantumSimulator.instances[-1]
    for shots in invalid_shots:
        with pytest.raises(ValueError, match="shots must"):
            runtime.measure([0], shots)
    assert sim.measurements == []

    batch_runtime = RocQuantumRuntime.from_bindings(1, batch_size=2)
    batch_sim = _FakeQuantumSimulator.instances[-1]
    for shots in invalid_shots:
        with pytest.raises(ValueError, match="shots must"):
            batch_runtime.measure_batch([0], shots)
    assert batch_sim.measurements == []
    assert batch_sim.probability_requests == []


def test_framework_runtime_revalidates_native_sample_payloads(monkeypatch):
    _install_fake_binding(monkeypatch)

    from rocquantum.framework_runtime import (
        RocQuantumRuntime,
        normalize_sample_indices,
        qiskit_memory_from_samples,
        samples_to_binary_rows,
    )

    invalid_samples = (
        [True],
        [np.bool_(False)],
        [0.5],
        ["1"],
        [b"1"],
        [1.0 + 0.0j],
        [-1],
        [2],
    )

    for raw_samples in invalid_samples:
        with pytest.raises(ValueError, match="samples"):
            normalize_sample_indices(raw_samples, 1)
        with pytest.raises(ValueError, match="samples"):
            samples_to_binary_rows(raw_samples, 1)
        with pytest.raises(ValueError, match="samples"):
            qiskit_memory_from_samples(raw_samples, [(0, 0)], 1)

    with pytest.raises(ValueError, match="count"):
        normalize_sample_indices([0], 1, expected_count=2)
    with pytest.raises(ValueError, match="offsets"):
        qiskit_memory_from_samples([0], [(0, 0)], 1, sample_offsets=[True])
    with pytest.raises(ValueError, match="offsets"):
        qiskit_memory_from_samples([0], [(0, 0)], 1, sample_offsets=[])

    class _BadNativeMeasureSimulator:
        def __init__(self, samples):
            self.samples = samples
            self.calls = []

        def measure(self, qubits, shots):
            self.calls.append((tuple(qubits), int(shots)))
            return self.samples

    for raw_samples, shots in tuple((samples, 1) for samples in invalid_samples) + (([0], 2),):
        sim = _BadNativeMeasureSimulator(raw_samples)
        with pytest.raises(ValueError):
            RocQuantumRuntime(sim).measure([0], shots)
        assert sim.calls == [((0,), shots)]

    class _BadNativeBatchMeasureSimulator:
        def __init__(self, samples):
            self.samples = samples
            self.calls = []

        def batch_size(self):
            return 2

        def measure_batch(self, qubits, shots):
            self.calls.append((tuple(qubits), int(shots)))
            return self.samples

    invalid_batch_samples = (
        [[0, 1], [1, 2]],
        [[0, 1], [True, 0]],
        [[0, 1], [1]],
        [0, 1, 1],
    )
    for raw_samples in invalid_batch_samples:
        sim = _BadNativeBatchMeasureSimulator(raw_samples)
        with pytest.raises(ValueError):
            RocQuantumRuntime(sim).measure_batch([0], 2)
        assert sim.calls == [((0,), 2)]


def test_framework_runtime_rejects_ambiguous_binding_dimensions(monkeypatch):
    _install_fake_binding(monkeypatch)

    from rocquantum.framework_runtime import RocQuantumRuntime

    invalid_dimensions = (0, -1, True, np.bool_(False), 1.5, "2", b"2", 1.0 + 0.0j)
    initial_instance_count = len(_FakeQuantumSimulator.instances)

    for num_qubits in invalid_dimensions:
        with pytest.raises(ValueError, match="num_qubits"):
            RocQuantumRuntime.from_bindings(num_qubits)

    for batch_size in invalid_dimensions:
        with pytest.raises(ValueError, match="batch_size"):
            RocQuantumRuntime.from_bindings(1, batch_size=batch_size)

    assert len(_FakeQuantumSimulator.instances) == initial_instance_count

    runtime = RocQuantumRuntime.from_bindings(np.int64(1), batch_size=np.int64(2))
    assert runtime.num_qubits() == 1
    assert runtime.batch_size() == 2


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


def test_qiskit_backend_handles_global_phase_gate_without_empty_matrix(monkeypatch):
    pytest.importorskip("qiskit")
    _install_fake_binding(monkeypatch)

    from qiskit import QuantumCircuit
    from qiskit.circuit.library import GlobalPhaseGate
    from qiskit_rocquantum_provider import RocQuantumProvider

    backend = RocQuantumProvider().get_backend("rocq_simulator")
    assert "global_phase" in set(backend.target.operation_names)

    circuit = QuantumCircuit(1)
    circuit.h(0)
    circuit.append(GlobalPhaseGate(math.pi / 5), [])

    backend.run(circuit, sampling=False, statevector=True).result()

    sim = _FakeQuantumSimulator.instances[-1]
    phase = np.exp(1j * math.pi / 5)
    assert sim.ops == [("H", (0,), ())]
    assert [(matrix.shape, targets) for matrix, targets in sim.matrices] == [((2, 2), (0,))]
    np.testing.assert_allclose(sim.matrices[0][0], np.array([[phase, 0.0], [0.0, phase]]))


def test_qiskit_backend_skips_global_phase_gate_for_sampling_only(monkeypatch):
    pytest.importorskip("qiskit")
    _install_fake_binding(monkeypatch)

    from qiskit import QuantumCircuit
    from qiskit.circuit.library import GlobalPhaseGate
    from qiskit_rocquantum_provider import RocQuantumProvider

    backend = RocQuantumProvider().get_backend("rocq_simulator")
    circuit = QuantumCircuit(1, 1)
    circuit.h(0)
    circuit.measure(0, 0)
    circuit.append(GlobalPhaseGate(math.pi / 5), [])

    backend.run(circuit, statevector=False).result()

    sim = _FakeQuantumSimulator.instances[-1]
    assert sim.ops == [("H", (0,), ())]
    assert sim.matrices == []
    assert sim.measurements == [((0,), 1024)]


def test_qiskit_backend_decomposes_controlled_global_phase_natively(monkeypatch):
    pytest.importorskip("qiskit")
    _install_fake_binding(monkeypatch)

    from qiskit import QuantumCircuit
    from qiskit.circuit.library import GlobalPhaseGate
    from qiskit_rocquantum_provider import RocQuantumProvider

    backend = RocQuantumProvider().get_backend("rocq_simulator")
    assert {"cglobal_phase", "ccglobal_phase", "c3global_phase"}.issubset(
        set(backend.target.operation_names)
    )
    circuit = QuantumCircuit(3)
    circuit.append(GlobalPhaseGate(0.3).control(1, annotated=False), [0])
    circuit.append(GlobalPhaseGate(0.4).control(2, ctrl_state="01", annotated=False), [0, 1])

    backend.run(circuit, sampling=False).result()

    sim = _FakeQuantumSimulator.instances[-1]
    assert ("P", (0,), (0.3,)) in sim.ops
    assert ("X", (1,), ()) in sim.ops
    assert ("RZ", (0,), (0.2,)) in sim.ops
    assert sim.matrices == []
    assert sim.controlled_matrices == []


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


def test_qiskit_backend_decomposes_controlled_two_qubit_rotations_natively(monkeypatch):
    pytest.importorskip("qiskit")
    _install_fake_binding(monkeypatch)

    from qiskit import QuantumCircuit
    from qiskit.circuit.library import RXXGate, RYYGate, RZXGate, RZZGate
    from qiskit_rocquantum_provider import RocQuantumProvider

    backend = RocQuantumProvider().get_backend("rocq_simulator")
    assert {
        "crxx", "ccrxx", "c3rxx",
        "cryy", "ccryy", "c3ryy",
        "crzz", "ccrzz", "c3rzz",
        "crzx", "ccrzx", "c3rzx",
    }.issubset(set(backend.target.operation_names))
    circuit = QuantumCircuit(4)
    circuit.append(RZZGate(0.4).control(1, annotated=False), [0, 1, 2])
    circuit.append(RXXGate(0.6).control(2, annotated=False), [0, 1, 2, 3])
    circuit.append(RYYGate(0.8).control(2, ctrl_state="01", annotated=False), [0, 1, 2, 3])
    circuit.append(RZXGate(1.0).control(1, annotated=False), [0, 1, 2])

    backend.run(circuit, sampling=False).result()

    sim = _FakeQuantumSimulator.instances[-1]
    assert ("RZ", (2,), (0.2,)) in sim.ops
    assert ("RZ", (2,), (-0.2,)) in sim.ops
    assert ("H", (2,), ()) in sim.ops
    assert ("H", (3,), ()) in sim.ops
    assert ("RX", (2,), (np.pi / 2,)) in sim.ops
    assert ("X", (1,), ()) in sim.ops
    assert sim.matrices == []
    assert sim.controlled_matrices == []


def test_qiskit_backend_decomposes_controlled_xx_yy_natively(monkeypatch):
    pytest.importorskip("qiskit")
    _install_fake_binding(monkeypatch)

    from qiskit import QuantumCircuit
    from qiskit.circuit.library import XXMinusYYGate, XXPlusYYGate
    from qiskit_rocquantum_provider import RocQuantumProvider

    backend = RocQuantumProvider().get_backend("rocq_simulator")
    assert {
        "cxx_plus_yy", "ccxx_plus_yy", "c3xx_plus_yy",
        "cxx_minus_yy", "ccxx_minus_yy", "c3xx_minus_yy",
    }.issubset(set(backend.target.operation_names))
    circuit = QuantumCircuit(5)
    circuit.append(XXPlusYYGate(0.6, 0.4).control(1, annotated=False), [0, 1, 2])
    circuit.append(XXMinusYYGate(0.8, 0.5).control(2, ctrl_state="01", annotated=False), [0, 1, 2, 3])
    circuit.append(XXPlusYYGate(1.0, 0.7).control(3, ctrl_state="101", annotated=False), [0, 1, 2, 3, 4])

    backend.run(circuit, sampling=False).result()

    sim = _FakeQuantumSimulator.instances[-1]
    assert ("CRZ", (0, 1), (0.4,)) in sim.ops
    assert ("CP", (0, 2), (-np.pi / 2,)) in sim.ops
    assert ("CRY", (0, 2), (-0.3,)) in sim.ops
    assert ("X", (1,), ()) in sim.ops
    assert sim.matrices == []
    assert sim.controlled_matrices == []


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


def test_qiskit_backend_decomposes_legacy_u_family_natively(monkeypatch):
    pytest.importorskip("qiskit")
    _install_fake_binding(monkeypatch)

    from qiskit import QuantumCircuit
    from qiskit.circuit.library import U1Gate, U2Gate, U3Gate
    from qiskit_rocquantum_provider import RocQuantumProvider

    backend = RocQuantumProvider().get_backend("rocq_simulator")
    assert {"u1", "u2", "u3"}.issubset(set(backend.target.operation_names))

    circuit = QuantumCircuit(1)
    circuit.append(U1Gate(0.4), [0])
    circuit.append(U2Gate(0.2, 0.3), [0])
    circuit.append(U3Gate(0.1, 0.2, 0.3), [0])

    backend.run(circuit, sampling=False).result()

    sim = _FakeQuantumSimulator.instances[-1]
    assert sim.ops == [
        ("P", (0,), (0.4,)),
        ("RZ", (0,), (0.3,)),
        ("RY", (0,), (np.pi / 2,)),
        ("RZ", (0,), (0.2,)),
        ("RZ", (0,), (0.3,)),
        ("RY", (0,), (0.1,)),
        ("RZ", (0,), (0.2,)),
    ]
    assert sim.matrices == []


def test_qiskit_backend_decomposes_legacy_multi_control_u1_natively(monkeypatch):
    pytest.importorskip("qiskit")
    _install_fake_binding(monkeypatch)

    from qiskit import QuantumCircuit
    from qiskit.circuit.library import U1Gate
    from qiskit_rocquantum_provider import RocQuantumProvider

    backend = RocQuantumProvider().get_backend("rocq_simulator")
    assert "mcu1" in set(backend.target.operation_names)
    circuit = QuantumCircuit(4)
    circuit.append(U1Gate(0.4).control(2, annotated=False), [0, 1, 2])
    circuit.append(U1Gate(0.6).control(3, ctrl_state="101", annotated=False), [0, 1, 2, 3])

    backend.run(circuit, sampling=False).result()

    sim = _FakeQuantumSimulator.instances[-1]
    assert ("RZ", (2,), (0.1,)) in sim.ops
    assert ("RZ", (3,), (0.075,)) in sim.ops
    assert ("X", (1,), ()) in sim.ops
    assert sim.matrices == []
    assert sim.controlled_matrices == []


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


def test_qiskit_backend_decomposes_multi_controlled_swap_natively(monkeypatch):
    pytest.importorskip("qiskit")
    _install_fake_binding(monkeypatch)

    from qiskit import QuantumCircuit
    from qiskit.circuit.library import SwapGate
    from qiskit_rocquantum_provider import RocQuantumProvider

    backend = RocQuantumProvider().get_backend("rocq_simulator")
    assert {"ccswap", "c3swap"}.issubset(set(backend.target.operation_names))
    circuit = QuantumCircuit(5)
    circuit.append(SwapGate().control(2, annotated=False), [0, 1, 2, 3])
    circuit.append(SwapGate().control(3, ctrl_state="101", annotated=False), [0, 1, 2, 3, 4])

    backend.run(circuit, shots=1, statevector=False).result()

    sim = _FakeQuantumSimulator.instances[-1]
    assert sim.ops[:3] == [
        ("MCX", (0, 1, 2, 3), ()),
        ("MCX", (0, 1, 3, 2), ()),
        ("MCX", (0, 1, 2, 3), ()),
    ]
    assert ("X", (1,), ()) in sim.ops
    assert ("MCX", (0, 1, 2, 3, 4), ()) in sim.ops
    assert ("MCX", (0, 1, 2, 4, 3), ()) in sim.ops
    assert sim.matrices == []
    assert sim.controlled_matrices == []


def test_qiskit_backend_decomposes_controlled_dcx_natively(monkeypatch):
    pytest.importorskip("qiskit")
    _install_fake_binding(monkeypatch)

    from qiskit import QuantumCircuit
    from qiskit.circuit.library import DCXGate
    from qiskit_rocquantum_provider import RocQuantumProvider

    backend = RocQuantumProvider().get_backend("rocq_simulator")
    assert {"cdcx", "ccdcx", "c3dcx"}.issubset(set(backend.target.operation_names))
    circuit = QuantumCircuit(5)
    circuit.append(DCXGate().control(1, annotated=False), [0, 1, 2])
    circuit.append(DCXGate().control(2, annotated=False), [0, 1, 2, 3])
    circuit.append(DCXGate().control(3, ctrl_state="101", annotated=False), [0, 1, 2, 3, 4])

    backend.run(circuit, shots=1, statevector=False).result()

    sim = _FakeQuantumSimulator.instances[-1]
    assert sim.ops[:4] == [
        ("MCX", (0, 1, 2), ()),
        ("MCX", (0, 2, 1), ()),
        ("MCX", (0, 1, 2, 3), ()),
        ("MCX", (0, 1, 3, 2), ()),
    ]
    assert ("X", (1,), ()) in sim.ops
    assert ("MCX", (0, 1, 2, 3, 4), ()) in sim.ops
    assert ("MCX", (0, 1, 2, 4, 3), ()) in sim.ops
    assert sim.matrices == []
    assert sim.controlled_matrices == []


def test_qiskit_backend_decomposes_controlled_ecr_natively(monkeypatch):
    pytest.importorskip("qiskit")
    _install_fake_binding(monkeypatch)

    from qiskit import QuantumCircuit
    from qiskit.circuit.library import ECRGate
    from qiskit_rocquantum_provider import RocQuantumProvider

    backend = RocQuantumProvider().get_backend("rocq_simulator")
    assert {"cecr", "ccecr", "c3ecr"}.issubset(set(backend.target.operation_names))
    circuit = QuantumCircuit(5)
    circuit.append(ECRGate().control(1, annotated=False), [0, 1, 2])
    circuit.append(ECRGate().control(2, annotated=False), [0, 1, 2, 3])
    circuit.append(ECRGate().control(3, ctrl_state="101", annotated=False), [0, 1, 2, 3, 4])

    backend.run(circuit, shots=1, statevector=False).result()

    sim = _FakeQuantumSimulator.instances[-1]
    assert sim.ops[:4] == [
        ("CP", (0, 1), (np.pi / 2,)),
        ("CRX", (0, 2), (np.pi / 2,)),
        ("MCX", (0, 1, 2), ()),
        ("CNOT", (0, 1), ()),
    ]
    assert ("X", (1,), ()) in sim.ops
    assert ("MCX", (0, 1, 2, 3, 4), ()) in sim.ops
    assert ("MCX", (0, 1, 2, 3), ()) in sim.ops
    assert sim.matrices == []
    assert sim.controlled_matrices == []


def test_qiskit_backend_decomposes_controlled_iswap_natively(monkeypatch):
    pytest.importorskip("qiskit")
    _install_fake_binding(monkeypatch)

    from qiskit import QuantumCircuit
    from qiskit.circuit.library import iSwapGate
    from qiskit_rocquantum_provider import RocQuantumProvider

    backend = RocQuantumProvider().get_backend("rocq_simulator")
    assert {"ciswap", "cciswap", "c3iswap"}.issubset(set(backend.target.operation_names))
    circuit = QuantumCircuit(5)
    circuit.append(iSwapGate().control(1, annotated=False), [0, 1, 2])
    circuit.append(iSwapGate().control(2, annotated=False), [0, 1, 2, 3])
    circuit.append(iSwapGate().control(3, ctrl_state="101", annotated=False), [0, 1, 2, 3, 4])

    backend.run(circuit, shots=1, statevector=False).result()

    sim = _FakeQuantumSimulator.instances[-1]
    assert sim.ops[:9] == [
        ("CP", (0, 1), (np.pi / 2,)),
        ("CP", (0, 2), (np.pi / 2,)),
        ("RY", (1,), (np.pi / 4,)),
        ("CNOT", (0, 1), ()),
        ("RY", (1,), (-np.pi / 4,)),
        ("MCX", (0, 1, 2), ()),
        ("MCX", (0, 2, 1), ()),
        ("RY", (2,), (np.pi / 4,)),
        ("CNOT", (0, 2), ()),
    ]
    assert ("X", (1,), ()) in sim.ops
    assert ("MCX", (0, 1, 2, 3, 4), ()) in sim.ops
    assert ("MCX", (0, 1, 2, 4, 3), ()) in sim.ops
    assert sim.matrices == []
    assert sim.controlled_matrices == []


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


def test_qiskit_backend_decomposes_multi_control_h_y_z_natively(monkeypatch):
    pytest.importorskip("qiskit")
    _install_fake_binding(monkeypatch)

    from qiskit import QuantumCircuit
    from qiskit.circuit.library import HGate, YGate, ZGate
    from qiskit_rocquantum_provider import RocQuantumProvider

    backend = RocQuantumProvider().get_backend("rocq_simulator")
    assert {"cch", "ccy", "c3h", "c3y", "c3z"}.issubset(set(backend.target.operation_names))
    circuit = QuantumCircuit(4)
    circuit.append(ZGate().control(3, annotated=False), [0, 1, 2, 3])
    circuit.append(YGate().control(2, annotated=False), [0, 1, 2])
    circuit.append(HGate().control(2, annotated=False), [0, 1, 2])

    backend.run(circuit, sampling=False).result()

    sim = _FakeQuantumSimulator.instances[-1]
    assert sim.ops == [
        ("H", (3,), ()),
        ("MCX", (0, 1, 2, 3), ()),
        ("H", (3,), ()),
        ("SDG", (2,), ()),
        ("MCX", (0, 1, 2), ()),
        ("S", (2,), ()),
        ("RY", (2,), (np.pi / 4,)),
        ("MCX", (0, 1, 2), ()),
        ("RY", (2,), (-np.pi / 4,)),
    ]
    assert sim.matrices == []
    assert sim.controlled_matrices == []


def test_qiskit_backend_decomposes_open_multi_control_h_y_z_natively(monkeypatch):
    pytest.importorskip("qiskit")
    _install_fake_binding(monkeypatch)

    from qiskit import QuantumCircuit
    from qiskit.circuit.library import HGate, YGate, ZGate
    from qiskit_rocquantum_provider import RocQuantumProvider

    backend = RocQuantumProvider().get_backend("rocq_simulator")
    circuit = QuantumCircuit(4)
    circuit.append(ZGate().control(3, ctrl_state="101", annotated=False), [0, 1, 2, 3])
    circuit.append(YGate().control(2, ctrl_state="01", annotated=False), [0, 1, 2])
    circuit.append(HGate().control(2, ctrl_state="01", annotated=False), [0, 1, 2])

    backend.run(circuit, sampling=False).result()

    sim = _FakeQuantumSimulator.instances[-1]
    assert sim.ops == [
        ("X", (1,), ()),
        ("H", (3,), ()),
        ("MCX", (0, 1, 2, 3), ()),
        ("H", (3,), ()),
        ("X", (1,), ()),
        ("X", (1,), ()),
        ("SDG", (2,), ()),
        ("MCX", (0, 1, 2), ()),
        ("S", (2,), ()),
        ("X", (1,), ()),
        ("X", (1,), ()),
        ("RY", (2,), (np.pi / 4,)),
        ("MCX", (0, 1, 2), ()),
        ("RY", (2,), (-np.pi / 4,)),
        ("X", (1,), ()),
    ]
    assert sim.matrices == []
    assert sim.controlled_matrices == []


def test_qiskit_backend_decomposes_multi_control_sx_natively(monkeypatch):
    pytest.importorskip("qiskit")
    _install_fake_binding(monkeypatch)

    from qiskit import QuantumCircuit
    from qiskit.circuit.library import C3SXGate, SXGate
    from qiskit_rocquantum_provider import RocQuantumProvider

    backend = RocQuantumProvider().get_backend("rocq_simulator")
    assert {"ccsx", "c3sx"}.issubset(set(backend.target.operation_names))
    circuit = QuantumCircuit(4)
    circuit.append(C3SXGate(), [0, 1, 2, 3])
    circuit.append(SXGate().control(2, annotated=False), [0, 1, 2])

    backend.run(circuit, sampling=False).result()

    sim = _FakeQuantumSimulator.instances[-1]
    assert sim.ops[0] == ("H", (3,), ())
    assert sim.ops[-1] == ("H", (2,), ())
    assert ("RZ", (3,), (np.pi / 16,)) in sim.ops
    assert ("RZ", (2,), (np.pi / 8,)) in sim.ops
    assert sim.matrices == []
    assert sim.controlled_matrices == []


def test_qiskit_backend_decomposes_open_multi_control_sx_natively(monkeypatch):
    pytest.importorskip("qiskit")
    _install_fake_binding(monkeypatch)

    from qiskit import QuantumCircuit
    from qiskit.circuit.library import SXGate
    from qiskit_rocquantum_provider import RocQuantumProvider

    backend = RocQuantumProvider().get_backend("rocq_simulator")
    circuit = QuantumCircuit(4)
    circuit.append(SXGate().control(3, ctrl_state="101", annotated=False), [0, 1, 2, 3])

    backend.run(circuit, sampling=False).result()

    sim = _FakeQuantumSimulator.instances[-1]
    assert sim.ops[0] == ("X", (1,), ())
    assert sim.ops[1] == ("H", (3,), ())
    assert sim.ops[-2:] == [("H", (3,), ()), ("X", (1,), ())]
    assert ("RZ", (3,), (np.pi / 16,)) in sim.ops
    assert sim.matrices == []
    assert sim.controlled_matrices == []


def test_qiskit_multi_control_sx_preserves_statevector_global_phase(monkeypatch):
    pytest.importorskip("qiskit")
    _install_fake_binding(monkeypatch)

    from qiskit import QuantumCircuit
    from qiskit.circuit.library import C3SXGate
    from qiskit_rocquantum_provider import RocQuantumProvider

    backend = RocQuantumProvider().get_backend("rocq_simulator")
    circuit = QuantumCircuit(4)
    circuit.append(C3SXGate(), [0, 1, 2, 3])

    backend.run(circuit, sampling=False, statevector=True).result()

    sim = _FakeQuantumSimulator.instances[-1]
    phase = np.exp(1j * np.pi / 32)
    assert sim.ops[0] == ("H", (3,), ())
    assert sim.ops[-1] == ("H", (3,), ())
    assert [(matrix.shape, targets) for matrix, targets in sim.matrices] == [((2, 2), (0,))]
    np.testing.assert_allclose(sim.matrices[0][0], np.array([[phase, 0.0], [0.0, phase]]))
    assert sim.controlled_matrices == []


def test_qiskit_backend_decomposes_controlled_phase_root_gates_natively(monkeypatch):
    pytest.importorskip("qiskit")
    _install_fake_binding(monkeypatch)

    from qiskit import QuantumCircuit
    from qiskit.circuit.library import SdgGate, SGate, TGate, TdgGate
    from qiskit_rocquantum_provider import RocQuantumProvider

    backend = RocQuantumProvider().get_backend("rocq_simulator")
    assert {
        "ccs",
        "c3s",
        "ccsdg",
        "c3sdg",
        "ct",
        "cct",
        "c3t",
        "ctdg",
        "cctdg",
        "c3tdg",
    }.issubset(set(backend.target.operation_names))

    circuit = QuantumCircuit(4)
    circuit.append(TGate().control(1, annotated=False), [0, 1])
    circuit.append(TdgGate().control(1, annotated=False), [0, 1])
    circuit.append(SGate().control(2, annotated=False), [0, 1, 2])
    circuit.append(SdgGate().control(2, annotated=False), [0, 1, 2])
    circuit.append(TGate().control(3, annotated=False), [0, 1, 2, 3])
    circuit.append(TdgGate().control(3, annotated=False), [0, 1, 2, 3])

    backend.run(circuit, sampling=False).result()

    sim = _FakeQuantumSimulator.instances[-1]
    assert sim.ops[:2] == [
        ("CP", (0, 1), (np.pi / 4,)),
        ("CP", (0, 1), (-np.pi / 4,)),
    ]
    assert ("RZ", (2,), (np.pi / 8,)) in sim.ops
    assert ("RZ", (2,), (-np.pi / 8,)) in sim.ops
    assert ("RZ", (3,), (np.pi / 32,)) in sim.ops
    assert ("RZ", (3,), (-np.pi / 32,)) in sim.ops
    assert sim.matrices == []
    assert sim.controlled_matrices == []


def test_qiskit_backend_decomposes_open_controlled_phase_root_gates_natively(monkeypatch):
    pytest.importorskip("qiskit")
    _install_fake_binding(monkeypatch)

    from qiskit import QuantumCircuit
    from qiskit.circuit.library import TGate
    from qiskit_rocquantum_provider import RocQuantumProvider

    backend = RocQuantumProvider().get_backend("rocq_simulator")
    circuit = QuantumCircuit(4)
    circuit.append(TGate().control(3, ctrl_state="101", annotated=False), [0, 1, 2, 3])

    backend.run(circuit, sampling=False).result()

    sim = _FakeQuantumSimulator.instances[-1]
    assert sim.ops[0] == ("X", (1,), ())
    assert sim.ops[-1] == ("X", (1,), ())
    assert ("RZ", (3,), (np.pi / 32,)) in sim.ops
    assert sim.matrices == []
    assert sim.controlled_matrices == []


def test_qiskit_backend_decomposes_multi_controlled_rz_natively(monkeypatch):
    pytest.importorskip("qiskit")
    _install_fake_binding(monkeypatch)

    from qiskit import QuantumCircuit
    from qiskit.circuit.library import RZGate
    from qiskit_rocquantum_provider import RocQuantumProvider

    backend = RocQuantumProvider().get_backend("rocq_simulator")
    assert {"ccrz", "c3rz"}.issubset(set(backend.target.operation_names))
    circuit = QuantumCircuit(4)
    circuit.append(RZGate(0.4).control(2, annotated=False), [0, 1, 2])
    circuit.append(RZGate(0.6).control(3, ctrl_state="101", annotated=False), [0, 1, 2, 3])

    backend.run(circuit, sampling=False).result()

    sim = _FakeQuantumSimulator.instances[-1]
    assert ("RZ", (0,), (-0.1,)) in sim.ops
    assert ("RZ", (2,), (0.1,)) in sim.ops
    assert ("RZ", (3,), (0.075,)) in sim.ops
    assert ("X", (1,), ()) in sim.ops
    assert sim.matrices == []
    assert sim.controlled_matrices == []


def test_qiskit_backend_decomposes_multi_controlled_rx_ry_natively(monkeypatch):
    pytest.importorskip("qiskit")
    _install_fake_binding(monkeypatch)

    from qiskit import QuantumCircuit
    from qiskit.circuit.library import RXGate, RYGate
    from qiskit_rocquantum_provider import RocQuantumProvider

    backend = RocQuantumProvider().get_backend("rocq_simulator")
    assert {"ccrx", "c3rx", "ccry", "c3ry"}.issubset(set(backend.target.operation_names))
    circuit = QuantumCircuit(4)
    circuit.append(RXGate(0.4).control(2, annotated=False), [0, 1, 2])
    circuit.append(RYGate(0.6).control(3, ctrl_state="101", annotated=False), [0, 1, 2, 3])

    backend.run(circuit, sampling=False).result()

    sim = _FakeQuantumSimulator.instances[-1]
    assert sim.ops[0] == ("H", (2,), ())
    assert ("RZ", (2,), (0.1,)) in sim.ops
    assert ("X", (1,), ()) in sim.ops
    assert ("SDG", (3,), ()) in sim.ops
    assert sim.ops[-2:] == [("S", (3,), ()), ("X", (1,), ())]
    assert sim.matrices == []
    assert sim.controlled_matrices == []


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


def test_qiskit_backend_decomposes_multi_controlled_phase_natively(monkeypatch):
    pytest.importorskip("qiskit")
    _install_fake_binding(monkeypatch)

    from qiskit import QuantumCircuit
    from qiskit.circuit.library import MCPhaseGate
    from qiskit_rocquantum_provider import RocQuantumProvider

    backend = RocQuantumProvider().get_backend("rocq_simulator")
    assert "mcphase" in set(backend.target.operation_names)
    circuit = QuantumCircuit(3)
    circuit.append(MCPhaseGate(0.4, 2), [0, 1, 2])

    backend.run(circuit, sampling=False).result()

    sim = _FakeQuantumSimulator.instances[-1]
    assert sim.ops == [
        ("RZ", (0,), (0.1,)),
        ("RZ", (1,), (0.1,)),
        ("RZ", (2,), (0.1,)),
        ("CNOT", (0, 1), ()),
        ("RZ", (1,), (-0.1,)),
        ("CNOT", (0, 1), ()),
        ("CNOT", (0, 2), ()),
        ("RZ", (2,), (-0.1,)),
        ("CNOT", (0, 2), ()),
        ("CNOT", (1, 2), ()),
        ("RZ", (2,), (-0.1,)),
        ("CNOT", (1, 2), ()),
        ("CNOT", (0, 2), ()),
        ("CNOT", (1, 2), ()),
        ("RZ", (2,), (0.1,)),
        ("CNOT", (1, 2), ()),
        ("CNOT", (0, 2), ()),
    ]
    assert sim.matrices == []
    assert sim.controlled_matrices == []


def test_qiskit_open_control_multi_phase_decomposes_natively(monkeypatch):
    pytest.importorskip("qiskit")
    _install_fake_binding(monkeypatch)

    from qiskit import QuantumCircuit
    from qiskit.circuit.library import MCPhaseGate
    from qiskit_rocquantum_provider import RocQuantumProvider

    backend = RocQuantumProvider().get_backend("rocq_simulator")
    circuit = QuantumCircuit(3)
    circuit.append(MCPhaseGate(0.4, 2, ctrl_state="01"), [0, 1, 2])

    backend.run(circuit, sampling=False).result()

    sim = _FakeQuantumSimulator.instances[-1]
    assert sim.ops[0] == ("X", (1,), ())
    assert sim.ops[-1] == ("X", (1,), ())
    assert ("RZ", (2,), (0.1,)) in sim.ops
    assert sim.matrices == []
    assert sim.controlled_matrices == []


def test_qiskit_multi_controlled_phase_preserves_statevector_global_phase(monkeypatch):
    pytest.importorskip("qiskit")
    _install_fake_binding(monkeypatch)

    from qiskit import QuantumCircuit
    from qiskit.circuit.library import MCPhaseGate
    from qiskit_rocquantum_provider import RocQuantumProvider

    backend = RocQuantumProvider().get_backend("rocq_simulator")
    circuit = QuantumCircuit(3)
    circuit.append(MCPhaseGate(0.4, 2), [0, 1, 2])

    backend.run(circuit, sampling=False, statevector=True).result()

    sim = _FakeQuantumSimulator.instances[-1]
    phase = np.exp(1j * 0.4 / 8)
    assert [(matrix.shape, targets) for matrix, targets in sim.matrices] == [((2, 2), (0,))]
    np.testing.assert_allclose(sim.matrices[0][0], np.array([[phase, 0.0], [0.0, phase]]))
    assert sim.controlled_matrices == []


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


def test_qiskit_backend_decomposes_controlled_r_gate_natively(monkeypatch):
    pytest.importorskip("qiskit")
    _install_fake_binding(monkeypatch)

    from qiskit import QuantumCircuit
    from qiskit.circuit.library import RGate
    from qiskit_rocquantum_provider import RocQuantumProvider

    backend = RocQuantumProvider().get_backend("rocq_simulator")
    assert "cr" in set(backend.target.operation_names)
    circuit = QuantumCircuit(2)
    circuit.append(RGate(0.4, 0.2).control(1, annotated=False), [0, 1])
    circuit.append(RGate(0.6, 0.3).control(1, ctrl_state=0, annotated=False), [0, 1])

    backend.run(circuit, sampling=False).result()

    sim = _FakeQuantumSimulator.instances[-1]
    assert sim.ops == [
        ("RZ", (1,), (-0.2,)),
        ("CRX", (0, 1), (0.4,)),
        ("RZ", (1,), (0.2,)),
        ("X", (0,), ()),
        ("RZ", (1,), (-0.3,)),
        ("CRX", (0, 1), (0.6,)),
        ("RZ", (1,), (0.3,)),
        ("X", (0,), ()),
    ]
    assert sim.matrices == []
    assert sim.controlled_matrices == []


def test_qiskit_backend_decomposes_multi_controlled_r_gate_natively(monkeypatch):
    pytest.importorskip("qiskit")
    _install_fake_binding(monkeypatch)

    from qiskit import QuantumCircuit
    from qiskit.circuit.library import RGate
    from qiskit_rocquantum_provider import RocQuantumProvider

    backend = RocQuantumProvider().get_backend("rocq_simulator")
    assert {"ccr", "c3r"}.issubset(set(backend.target.operation_names))
    circuit = QuantumCircuit(4)
    circuit.append(RGate(0.4, 0.2).control(2, annotated=False), [0, 1, 2])
    circuit.append(RGate(0.6, 0.3).control(3, ctrl_state="101", annotated=False), [0, 1, 2, 3])

    backend.run(circuit, sampling=False).result()

    sim = _FakeQuantumSimulator.instances[-1]
    assert sim.ops[0:2] == [("RZ", (2,), (-0.2,)), ("H", (2,), ())]
    assert ("RZ", (2,), (0.2,)) in sim.ops
    assert ("X", (1,), ()) in sim.ops
    assert ("RZ", (3,), (-0.3,)) in sim.ops
    assert ("RZ", (3,), (0.3,)) in sim.ops
    assert sim.ops[-1] == ("X", (1,), ())
    assert sim.matrices == []
    assert sim.controlled_matrices == []


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


def test_qiskit_backend_decomposes_controlled_u_family_natively(monkeypatch):
    pytest.importorskip("qiskit")
    _install_fake_binding(monkeypatch)

    from qiskit import QuantumCircuit
    from qiskit.circuit.library import CSGate, CSdgGate, CSXGate, CU1Gate, CU3Gate, CUGate
    from qiskit_rocquantum_provider import RocQuantumProvider

    backend = RocQuantumProvider().get_backend("rocq_simulator")
    assert {"cs", "csdg", "csx", "cu1", "cu3", "cu"}.issubset(set(backend.target.operation_names))

    circuit = QuantumCircuit(2)
    circuit.append(CSGate(), [0, 1])
    circuit.append(CSdgGate(), [0, 1])
    circuit.append(CSXGate(), [0, 1])
    circuit.append(CU1Gate(0.4), [0, 1])
    circuit.append(CU3Gate(0.2, 0.3, 0.4), [0, 1])
    circuit.append(CUGate(0.2, 0.3, 0.4, 0.5), [0, 1])

    backend.run(circuit, sampling=False).result()

    sim = _FakeQuantumSimulator.instances[-1]
    assert sim.matrices == []
    assert sim.controlled_matrices == []
    assert sim.ops[:5] == [
        ("CP", (0, 1), (np.pi / 2,)),
        ("CP", (0, 1), (-np.pi / 2,)),
        ("P", (0,), (np.pi / 4,)),
        ("CRX", (0, 1), (np.pi / 2,)),
        ("CP", (0, 1), (0.4,)),
    ]
    assert ("P", (0,), (0.5,)) in sim.ops
    assert ("P", (0,), (0.35,)) in sim.ops
    assert ("P", (1,), (0.1,)) in sim.ops
    assert sim.ops.count(("CNOT", (0, 1), ())) == 4
    assert ("RY", (1,), (-0.1,)) in sim.ops
    assert ("RY", (1,), (0.1,)) in sim.ops


def test_qiskit_backend_decomposes_controlled_u2_u3_natively(monkeypatch):
    pytest.importorskip("qiskit")
    _install_fake_binding(monkeypatch)

    from qiskit import QuantumCircuit
    from qiskit.circuit.library import U2Gate, U3Gate
    from qiskit_rocquantum_provider import RocQuantumProvider

    backend = RocQuantumProvider().get_backend("rocq_simulator")
    assert {"cu2", "ccu2", "c3u2", "ccu3", "c3u3"}.issubset(set(backend.target.operation_names))
    circuit = QuantumCircuit(4)
    circuit.append(U2Gate(0.2, 0.3).control(1, annotated=False), [0, 1])
    circuit.append(U2Gate(0.4, 0.5).control(2, annotated=False), [0, 1, 2])
    circuit.append(U3Gate(0.6, 0.7, 0.8).control(2, annotated=False), [0, 1, 2])
    circuit.append(U3Gate(0.9, 1.0, 1.1).control(3, ctrl_state="101", annotated=False), [0, 1, 2, 3])

    backend.run(circuit, sampling=False).result()

    sim = _FakeQuantumSimulator.instances[-1]
    assert ("P", (0,), (0.25,)) in sim.ops
    assert ("CRZ", (0, 1), (0.3,)) in sim.ops
    assert ("CRY", (0, 1), (np.pi / 2,)) in sim.ops
    assert ("X", (1,), ()) in sim.ops
    assert sim.matrices == []
    assert sim.controlled_matrices == []


def test_qiskit_backend_decomposes_multi_controlled_u_natively(monkeypatch):
    pytest.importorskip("qiskit")
    _install_fake_binding(monkeypatch)

    from qiskit import QuantumCircuit
    from qiskit.circuit.library import UGate
    from qiskit_rocquantum_provider import RocQuantumProvider

    backend = RocQuantumProvider().get_backend("rocq_simulator")
    assert {"ccu", "c3u"}.issubset(set(backend.target.operation_names))
    circuit = QuantumCircuit(4)
    circuit.append(UGate(0.2, 0.3, 0.4).control(2, annotated=False), [0, 1, 2])
    circuit.append(UGate(0.5, 0.6, 0.7).control(3, ctrl_state="101", annotated=False), [0, 1, 2, 3])

    backend.run(circuit, sampling=False).result()

    sim = _FakeQuantumSimulator.instances[-1]
    assert ("RZ", (0,), (0.25 * (0.3 + 0.4),)) in sim.ops
    assert ("X", (1,), ()) in sim.ops
    assert sim.matrices == []
    assert sim.controlled_matrices == []


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


def test_qiskit_native_sampler_samples_runtime_reset_shot_by_shot(monkeypatch):
    pytest.importorskip("qiskit")
    _install_fake_binding(monkeypatch)

    from qiskit import QuantumCircuit
    from qiskit_rocquantum_provider import RocQuantumProvider

    circuit = QuantumCircuit(2, 2)
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.reset(0)
    circuit.measure([0, 1], [0, 1])

    result = RocQuantumProvider().get_sampler().run([circuit], shots=4).result()[0]

    assert result.data.c.num_shots == 4
    assert sum(result.data.c.get_counts().values()) == 4
    assert result.metadata["shot_trajectory"] is True
    sim = _FakeQuantumSimulator.instances[-1]
    assert sim.total_reset_qubits == [0, 0, 0, 0]
    assert sim.total_measurements == [((0, 1), 1)] * 4


def test_qiskit_native_sampler_samples_if_else_conditioned_gate(monkeypatch):
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

    result = RocQuantumProvider().get_sampler().run([circuit], shots=1).result()[0]

    assert result.data.c.get_counts() == {"11": 1}
    assert result.metadata["shot_trajectory"] is True
    sim = _FakeQuantumSimulator.instances[-1]
    assert sim.ops == [("X", (1,), ())]
    assert sim.measure_qubits == [0, 1]
    assert sim.measurements == []


def test_qiskit_native_sampler_limits_while_loop_iterations(monkeypatch):
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

    sampler = RocQuantumProvider().get_sampler(max_dynamic_loop_iterations=2)
    with pytest.raises(RuntimeError, match="max_dynamic_loop_iterations"):
        sampler.run([circuit], shots=1).result()


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


def test_qiskit_native_sampler_batches_initial_reset_parameter_values(monkeypatch):
    pytest.importorskip("qiskit")
    _install_fake_binding(monkeypatch)

    from qiskit import QuantumCircuit
    from qiskit.circuit import Parameter
    from qiskit_rocquantum_provider import RocQuantumProvider

    theta = Parameter("theta")
    circuit = QuantumCircuit(1, 1)
    circuit.reset(0)
    circuit.ry(theta, 0)
    circuit.measure(0, 0)

    result = RocQuantumProvider().get_sampler().run(
        [(circuit, [0.1, 0.2])],
        shots=3,
    ).result()[0]

    assert result.data.c.shape == (2,)
    assert result.metadata["batched_parameters"] is True
    sim = _FakeQuantumSimulator.instances[-1]
    assert sim.batch_size() == 2
    assert sim.reset_qubits == []
    assert sim.batch_ops == [("RY", (0,), (0.1, 0.2))]
    assert sim.measurements == []
    assert sim.probability_requests == [(0,)]


def test_qiskit_native_sampler_does_not_batch_mid_circuit_state_preparation(monkeypatch):
    pytest.importorskip("qiskit")
    _install_fake_binding(monkeypatch)

    from qiskit import QuantumCircuit
    from qiskit.circuit import Parameter
    from qiskit_rocquantum_provider import RocQuantumProvider

    theta = Parameter("theta")
    circuit = QuantumCircuit(1, 1)
    circuit.h(0)
    circuit.prepare_state([0.0, 1.0], 0)
    circuit.ry(theta, 0)
    circuit.measure(0, 0)

    result = RocQuantumProvider().get_sampler().run(
        [(circuit, [0.1, 0.2])],
        shots=3,
    ).result()[0]

    assert "batched_parameters" not in result.metadata
    sim = _FakeQuantumSimulator.instances[-1]
    assert sim.batch_size() == 1
    assert sim.statevectors == []
    assert sim.batch_ops == []
    assert [targets for _, targets in sim.matrices] == [(0,)]
    assert sim.ops == [("H", (0,), ()), ("RY", (0,), (0.2,))]


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


def test_qiskit_native_estimator_batches_initial_reset_parameters(monkeypatch):
    pytest.importorskip("qiskit")
    _install_fake_binding(monkeypatch)

    from qiskit import QuantumCircuit
    from qiskit.circuit import Parameter
    from qiskit.quantum_info import SparsePauliOp
    from qiskit_rocquantum_provider import RocQuantumProvider

    theta = Parameter("theta")
    circuit = QuantumCircuit(1)
    circuit.reset(0)
    circuit.ry(theta, 0)
    observable = SparsePauliOp.from_list([("Z", 1.0)])

    result = RocQuantumProvider().get_estimator().run(
        [(circuit, observable, [0.1, 0.2])],
    ).result()[0]

    np.testing.assert_allclose(result.data.evs, np.array([0.5, 0.5]))
    assert result.metadata["batched_parameters"] is True
    sim = _FakeQuantumSimulator.instances[-1]
    assert sim.batch_size() == 2
    assert sim.reset_qubits == []
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


def test_qiskit_native_estimator_does_not_batch_mid_circuit_state_preparation(monkeypatch):
    pytest.importorskip("qiskit")
    _install_fake_binding(monkeypatch)

    from qiskit import QuantumCircuit
    from qiskit.circuit import Parameter
    from qiskit.quantum_info import SparsePauliOp
    from qiskit_rocquantum_provider import RocQuantumProvider

    theta = Parameter("theta")
    circuit = QuantumCircuit(1)
    circuit.h(0)
    circuit.prepare_state([0.0, 1.0], 0)
    circuit.ry(theta, 0)
    observable = SparsePauliOp.from_list([("Z", 1.0)])

    result = RocQuantumProvider().get_estimator().run(
        [(circuit, observable, [0.1, 0.2])],
    ).result()[0]

    assert "batched_parameters" not in result.metadata
    np.testing.assert_allclose(result.data.evs, np.array([0.5, 0.5]))
    sim = _FakeQuantumSimulator.instances[-1]
    assert sim.batch_size() == 1
    assert sim.statevectors == []
    assert sim.batch_ops == []
    assert [targets for _, targets in sim.matrices] == [(0,)]
    assert sim.ops == [("H", (0,), ()), ("RY", (0,), (0.2,))]
    assert sim.expectations == [("Z", (0,))]


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


def test_qiskit_native_estimator_keeps_fixed_multi_control_h_y_z_batched(monkeypatch):
    pytest.importorskip("qiskit")
    _install_fake_binding(monkeypatch)

    from qiskit import QuantumCircuit
    from qiskit.circuit import Parameter
    from qiskit.circuit.library import HGate, YGate, ZGate
    from qiskit.quantum_info import SparsePauliOp
    from qiskit_rocquantum_provider import RocQuantumProvider

    theta = Parameter("theta")
    circuit = QuantumCircuit(4)
    circuit.append(ZGate().control(3, annotated=False), [0, 1, 2, 3])
    circuit.append(YGate().control(2, annotated=False), [0, 1, 2])
    circuit.append(HGate().control(2, annotated=False), [0, 1, 2])
    circuit.ry(theta, 3)
    observable = SparsePauliOp.from_list([("IIIZ", 1.0)])

    result = RocQuantumProvider().get_estimator().run(
        [(circuit, observable, [0.1, 0.2])],
    ).result()[0]

    np.testing.assert_allclose(result.data.evs, np.array([0.5, 0.5]))
    assert result.metadata["batched_parameters"] is True
    sim = _FakeQuantumSimulator.instances[-1]
    assert sim.batch_size() == 2
    assert sim.ops == [
        ("H", (3,), ()),
        ("MCX", (0, 1, 2, 3), ()),
        ("H", (3,), ()),
        ("SDG", (2,), ()),
        ("MCX", (0, 1, 2), ()),
        ("S", (2,), ()),
        ("RY", (2,), (np.pi / 4,)),
        ("MCX", (0, 1, 2), ()),
        ("RY", (2,), (-np.pi / 4,)),
    ]
    assert sim.batch_ops == [("RY", (3,), (0.1, 0.2))]
    assert sim.matrices == []
    assert sim.controlled_matrices == []
    assert sim.batch_expectations == [("Z", (0,))]


def test_qiskit_native_estimator_keeps_fixed_multi_control_sx_batched(monkeypatch):
    pytest.importorskip("qiskit")
    _install_fake_binding(monkeypatch)

    from qiskit import QuantumCircuit
    from qiskit.circuit import Parameter
    from qiskit.circuit.library import C3SXGate
    from qiskit.quantum_info import SparsePauliOp
    from qiskit_rocquantum_provider import RocQuantumProvider

    theta = Parameter("theta")
    circuit = QuantumCircuit(4)
    circuit.append(C3SXGate(), [0, 1, 2, 3])
    circuit.ry(theta, 3)
    observable = SparsePauliOp.from_list([("IIIZ", 1.0)])

    result = RocQuantumProvider().get_estimator().run(
        [(circuit, observable, [0.1, 0.2])],
    ).result()[0]

    np.testing.assert_allclose(result.data.evs, np.array([0.5, 0.5]))
    assert result.metadata["batched_parameters"] is True
    sim = _FakeQuantumSimulator.instances[-1]
    assert sim.batch_size() == 2
    assert sim.ops[0] == ("H", (3,), ())
    assert sim.ops[-1] == ("H", (3,), ())
    assert ("RZ", (3,), (np.pi / 16,)) in sim.ops
    assert sim.batch_ops == [("RY", (3,), (0.1, 0.2))]
    assert sim.matrices == []
    assert sim.controlled_matrices == []
    assert sim.batch_expectations == [("Z", (0,))]


def test_qiskit_native_estimator_keeps_fixed_controlled_phase_roots_batched(monkeypatch):
    pytest.importorskip("qiskit")
    _install_fake_binding(monkeypatch)

    from qiskit import QuantumCircuit
    from qiskit.circuit import Parameter
    from qiskit.circuit.library import TGate
    from qiskit.quantum_info import SparsePauliOp
    from qiskit_rocquantum_provider import RocQuantumProvider

    theta = Parameter("theta")
    circuit = QuantumCircuit(4)
    circuit.append(TGate().control(3, annotated=False), [0, 1, 2, 3])
    circuit.ry(theta, 3)
    observable = SparsePauliOp.from_list([("IIIZ", 1.0)])

    result = RocQuantumProvider().get_estimator().run(
        [(circuit, observable, [0.1, 0.2])],
    ).result()[0]

    np.testing.assert_allclose(result.data.evs, np.array([0.5, 0.5]))
    assert result.metadata["batched_parameters"] is True
    sim = _FakeQuantumSimulator.instances[-1]
    assert sim.batch_size() == 2
    assert ("RZ", (3,), (np.pi / 32,)) in sim.ops
    assert sim.batch_ops == [("RY", (3,), (0.1, 0.2))]
    assert sim.matrices == []
    assert sim.controlled_matrices == []
    assert sim.batch_expectations == [("Z", (0,))]


def test_qiskit_native_estimator_batches_multi_controlled_phase_parameters(monkeypatch):
    pytest.importorskip("qiskit")
    _install_fake_binding(monkeypatch)

    from qiskit import QuantumCircuit
    from qiskit.circuit import Parameter
    from qiskit.circuit.library import MCPhaseGate
    from qiskit.quantum_info import SparsePauliOp
    from qiskit_rocquantum_provider import RocQuantumProvider

    theta = Parameter("theta")
    circuit = QuantumCircuit(3)
    circuit.append(MCPhaseGate(theta, 2), [0, 1, 2])
    observable = SparsePauliOp.from_list([("IIZ", 1.0)])

    result = RocQuantumProvider().get_estimator().run(
        [(circuit, observable, [0.1, 0.2])],
    ).result()[0]

    np.testing.assert_allclose(result.data.evs, np.array([0.5, 0.5]))
    assert result.metadata["batched_parameters"] is True
    sim = _FakeQuantumSimulator.instances[-1]
    assert sim.batch_size() == 2
    assert sim.batch_ops == [
        ("RZ", (0,), (0.025, 0.05)),
        ("RZ", (1,), (0.025, 0.05)),
        ("RZ", (2,), (0.025, 0.05)),
        ("RZ", (1,), (-0.025, -0.05)),
        ("RZ", (2,), (-0.025, -0.05)),
        ("RZ", (2,), (-0.025, -0.05)),
        ("RZ", (2,), (0.025, 0.05)),
    ]
    assert sim.ops == [
        ("CNOT", (0, 1), ()),
        ("CNOT", (0, 1), ()),
        ("CNOT", (0, 2), ()),
        ("CNOT", (0, 2), ()),
        ("CNOT", (1, 2), ()),
        ("CNOT", (1, 2), ()),
        ("CNOT", (0, 2), ()),
        ("CNOT", (1, 2), ()),
        ("CNOT", (1, 2), ()),
        ("CNOT", (0, 2), ()),
    ]
    assert sim.matrices == []
    assert sim.controlled_matrices == []
    assert sim.batch_expectations == [("Z", (0,))]


def test_qiskit_native_estimator_batches_legacy_multi_control_u1_parameters(monkeypatch):
    pytest.importorskip("qiskit")
    _install_fake_binding(monkeypatch)

    from qiskit import QuantumCircuit
    from qiskit.circuit import Parameter
    from qiskit.circuit.library import U1Gate
    from qiskit.quantum_info import SparsePauliOp
    from qiskit_rocquantum_provider import RocQuantumProvider

    theta = Parameter("theta")
    circuit = QuantumCircuit(3)
    circuit.append(U1Gate(theta).control(2, annotated=False), [0, 1, 2])
    observable = SparsePauliOp.from_list([("IIZ", 1.0)])

    result = RocQuantumProvider().get_estimator().run(
        [(circuit, observable, [0.1, 0.2])],
    ).result()[0]

    np.testing.assert_allclose(result.data.evs, np.array([0.5, 0.5]))
    assert result.metadata["batched_parameters"] is True
    sim = _FakeQuantumSimulator.instances[-1]
    assert sim.batch_size() == 2
    assert sim.batch_ops == [
        ("RZ", (0,), (0.025, 0.05)),
        ("RZ", (1,), (0.025, 0.05)),
        ("RZ", (2,), (0.025, 0.05)),
        ("RZ", (1,), (-0.025, -0.05)),
        ("RZ", (2,), (-0.025, -0.05)),
        ("RZ", (2,), (-0.025, -0.05)),
        ("RZ", (2,), (0.025, 0.05)),
    ]
    assert sim.matrices == []
    assert sim.controlled_matrices == []
    assert sim.batch_expectations == [("Z", (0,))]


def test_qiskit_native_estimator_batches_multi_controlled_rz_parameters(monkeypatch):
    pytest.importorskip("qiskit")
    _install_fake_binding(monkeypatch)

    from qiskit import QuantumCircuit
    from qiskit.circuit import Parameter
    from qiskit.circuit.library import RZGate
    from qiskit.quantum_info import SparsePauliOp
    from qiskit_rocquantum_provider import RocQuantumProvider

    theta = Parameter("theta")
    circuit = QuantumCircuit(3)
    circuit.append(RZGate(theta).control(2, annotated=False), [0, 1, 2])
    observable = SparsePauliOp.from_list([("IIZ", 1.0)])

    result = RocQuantumProvider().get_estimator().run(
        [(circuit, observable, [0.1, 0.2])],
    ).result()[0]

    np.testing.assert_allclose(result.data.evs, np.array([0.5, 0.5]))
    assert result.metadata["batched_parameters"] is True
    sim = _FakeQuantumSimulator.instances[-1]
    assert sim.batch_size() == 2
    assert ("RZ", (0,), (-0.025, -0.05)) in sim.batch_ops
    assert ("RZ", (2,), (0.025, 0.05)) in sim.batch_ops
    assert sim.matrices == []
    assert sim.controlled_matrices == []
    assert sim.batch_expectations == [("Z", (0,))]


def test_qiskit_native_estimator_batches_multi_controlled_rx_ry_parameters(monkeypatch):
    pytest.importorskip("qiskit")
    _install_fake_binding(monkeypatch)

    from qiskit import QuantumCircuit
    from qiskit.circuit import Parameter
    from qiskit.circuit.library import RXGate, RYGate
    from qiskit.quantum_info import SparsePauliOp
    from qiskit_rocquantum_provider import RocQuantumProvider

    theta = Parameter("theta")
    phi = Parameter("phi")
    circuit = QuantumCircuit(3)
    circuit.append(RXGate(theta).control(2, annotated=False), [0, 1, 2])
    circuit.append(RYGate(phi).control(2, annotated=False), [0, 1, 2])
    observable = SparsePauliOp.from_list([("IIZ", 1.0)])

    result = RocQuantumProvider().get_estimator().run(
        [
            (
                circuit,
                observable,
                {
                    (theta, phi): [
                        [0.1, 0.3],
                        [0.2, 0.4],
                    ],
                },
            )
        ],
    ).result()[0]

    np.testing.assert_allclose(result.data.evs, np.array([0.5, 0.5]))
    assert result.metadata["batched_parameters"] is True
    sim = _FakeQuantumSimulator.instances[-1]
    assert sim.batch_size() == 2
    assert ("H", (2,), ()) in sim.ops
    assert ("SDG", (2,), ()) in sim.ops
    assert ("S", (2,), ()) in sim.ops
    assert ("RZ", (0,), (-0.025, -0.05)) in sim.batch_ops
    assert ("RZ", (2,), (0.075, 0.1)) in sim.batch_ops
    assert sim.matrices == []
    assert sim.controlled_matrices == []
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


def test_qiskit_native_estimator_batches_controlled_r_parameters(monkeypatch):
    pytest.importorskip("qiskit")
    _install_fake_binding(monkeypatch)

    from qiskit import QuantumCircuit
    from qiskit.circuit import Parameter
    from qiskit.circuit.library import RGate
    from qiskit.quantum_info import SparsePauliOp
    from qiskit_rocquantum_provider import RocQuantumProvider

    theta = Parameter("theta")
    phi = Parameter("phi")
    circuit = QuantumCircuit(2)
    circuit.append(RGate(theta, phi).control(1, annotated=False), [0, 1])
    observable = SparsePauliOp.from_list([("IZ", 1.0)])

    result = RocQuantumProvider().get_estimator().run(
        [
            (
                circuit,
                observable,
                {
                    (theta, phi): [
                        [0.2, 0.3],
                        [0.6, 0.7],
                    ],
                },
            )
        ],
    ).result()[0]

    np.testing.assert_allclose(result.data.evs, np.array([0.5, 0.5]))
    assert result.metadata["batched_parameters"] is True
    sim = _FakeQuantumSimulator.instances[-1]
    assert sim.batch_size() == 2
    assert sim.ops == []
    assert sim.batch_ops == [
        ("RZ", (1,), (-0.3, -0.7)),
        ("CRX", (0, 1), (0.2, 0.6)),
        ("RZ", (1,), (0.3, 0.7)),
    ]
    assert sim.matrices == []
    assert sim.controlled_matrices == []
    assert sim.batch_expectations == [("Z", (0,))]


def test_qiskit_native_estimator_batches_multi_controlled_r_parameters(monkeypatch):
    pytest.importorskip("qiskit")
    _install_fake_binding(monkeypatch)

    from qiskit import QuantumCircuit
    from qiskit.circuit import Parameter
    from qiskit.circuit.library import RGate
    from qiskit.quantum_info import SparsePauliOp
    from qiskit_rocquantum_provider import RocQuantumProvider

    theta = Parameter("theta")
    phi = Parameter("phi")
    circuit = QuantumCircuit(3)
    circuit.append(RGate(theta, phi).control(2, annotated=False), [0, 1, 2])
    observable = SparsePauliOp.from_list([("IIZ", 1.0)])

    result = RocQuantumProvider().get_estimator().run(
        [
            (
                circuit,
                observable,
                {
                    (theta, phi): [
                        [0.2, 0.3],
                        [0.6, 0.7],
                    ],
                },
            )
        ],
    ).result()[0]

    np.testing.assert_allclose(result.data.evs, np.array([0.5, 0.5]))
    assert result.metadata["batched_parameters"] is True
    sim = _FakeQuantumSimulator.instances[-1]
    assert sim.batch_size() == 2
    assert sim.ops[0] == ("H", (2,), ())
    assert sim.ops[-1] == ("H", (2,), ())
    assert ("RZ", (2,), (-0.3, -0.7)) in sim.batch_ops
    assert ("RZ", (2,), (0.3, 0.7)) in sim.batch_ops
    assert ("RZ", (2,), (0.05, 0.15)) in sim.batch_ops
    assert sim.matrices == []
    assert sim.controlled_matrices == []
    assert sim.batch_expectations == [("Z", (0,))]


def test_qiskit_native_estimator_batches_legacy_u_family_parameters(monkeypatch):
    pytest.importorskip("qiskit")
    _install_fake_binding(monkeypatch)

    from qiskit import QuantumCircuit
    from qiskit.circuit import Parameter
    from qiskit.circuit.library import U1Gate, U2Gate, U3Gate
    from qiskit.quantum_info import SparsePauliOp
    from qiskit_rocquantum_provider import RocQuantumProvider

    alpha = Parameter("alpha")
    phi2 = Parameter("phi2")
    lam2 = Parameter("lam2")
    theta3 = Parameter("theta3")
    phi3 = Parameter("phi3")
    lam3 = Parameter("lam3")
    circuit = QuantumCircuit(1)
    circuit.append(U1Gate(alpha), [0])
    circuit.append(U2Gate(phi2, lam2), [0])
    circuit.append(U3Gate(theta3, phi3, lam3), [0])
    observable = SparsePauliOp.from_list([("Z", 1.0)])

    result = RocQuantumProvider().get_estimator().run(
        [
            (
                circuit,
                observable,
                {
                    (alpha, phi2, lam2, theta3, phi3, lam3): [
                        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
                        [0.7, 0.8, 0.9, 1.0, 1.1, 1.2],
                    ],
                },
            )
        ],
    ).result()[0]

    np.testing.assert_allclose(result.data.evs, np.array([0.5, 0.5]))
    assert result.metadata["batched_parameters"] is True
    sim = _FakeQuantumSimulator.instances[-1]
    assert sim.batch_size() == 2
    assert sim.ops == []
    assert sim.batch_ops == [
        ("P", (0,), (0.1, 0.7)),
        ("RZ", (0,), (0.3, 0.9)),
        ("RY", (0,), (np.pi / 2, np.pi / 2)),
        ("RZ", (0,), (0.2, 0.8)),
        ("RZ", (0,), (0.6, 1.2)),
        ("RY", (0,), (0.4, 1.0)),
        ("RZ", (0,), (0.5, 1.1)),
    ]
    assert sim.matrices == []
    assert sim.batch_expectations == [("Z", (0,))]


def test_qiskit_native_estimator_batches_controlled_u2_u3_parameters(monkeypatch):
    pytest.importorskip("qiskit")
    _install_fake_binding(monkeypatch)

    from qiskit import QuantumCircuit
    from qiskit.circuit import Parameter
    from qiskit.circuit.library import U2Gate, U3Gate
    from qiskit.quantum_info import SparsePauliOp
    from qiskit_rocquantum_provider import RocQuantumProvider

    phi2 = Parameter("phi2")
    lam2 = Parameter("lam2")
    theta3 = Parameter("theta3")
    phi3 = Parameter("phi3")
    lam3 = Parameter("lam3")
    circuit = QuantumCircuit(3)
    circuit.append(U2Gate(phi2, lam2).control(1, annotated=False), [0, 2])
    circuit.append(U3Gate(theta3, phi3, lam3).control(2, annotated=False), [0, 1, 2])
    observable = SparsePauliOp.from_list([("IIZ", 1.0)])

    result = RocQuantumProvider().get_estimator().run(
        [
            (
                circuit,
                observable,
                {
                    (phi2, lam2, theta3, phi3, lam3): [
                        [0.2, 0.3, 0.4, 0.5, 0.6],
                        [0.7, 0.8, 0.9, 1.0, 1.1],
                    ],
                },
            )
        ],
    ).result()[0]

    np.testing.assert_allclose(result.data.evs, np.array([0.5, 0.5]))
    assert result.metadata["batched_parameters"] is True
    sim = _FakeQuantumSimulator.instances[-1]
    assert sim.batch_size() == 2
    assert ("P", (0,), (0.25, 0.75)) in sim.batch_ops
    assert ("CRY", (0, 2), (np.pi / 2, np.pi / 2)) in sim.batch_ops
    assert ("RZ", (0,), (0.275, 0.525)) in sim.batch_ops
    assert sim.matrices == []
    assert sim.controlled_matrices == []
    assert sim.batch_expectations == [("Z", (0,))]


def test_qiskit_native_estimator_batches_multi_controlled_u_parameters(monkeypatch):
    pytest.importorskip("qiskit")
    _install_fake_binding(monkeypatch)

    from qiskit import QuantumCircuit
    from qiskit.circuit import Parameter
    from qiskit.circuit.library import UGate
    from qiskit.quantum_info import SparsePauliOp
    from qiskit_rocquantum_provider import RocQuantumProvider

    theta = Parameter("theta")
    phi = Parameter("phi")
    lam = Parameter("lam")
    circuit = QuantumCircuit(3)
    circuit.append(UGate(theta, phi, lam).control(2, annotated=False), [0, 1, 2])
    observable = SparsePauliOp.from_list([("IIZ", 1.0)])

    result = RocQuantumProvider().get_estimator().run(
        [
            (
                circuit,
                observable,
                {
                    (theta, phi, lam): [
                        [0.2, 0.3, 0.4],
                        [0.5, 0.6, 0.7],
                    ],
                },
            )
        ],
    ).result()[0]

    np.testing.assert_allclose(result.data.evs, np.array([0.5, 0.5]))
    assert result.metadata["batched_parameters"] is True
    sim = _FakeQuantumSimulator.instances[-1]
    assert sim.batch_size() == 2
    assert ("RZ", (0,), (0.25 * (0.3 + 0.4), 0.25 * (0.6 + 0.7))) in sim.batch_ops
    assert sim.matrices == []
    assert sim.controlled_matrices == []
    assert sim.batch_expectations == [("Z", (0,))]


def test_qiskit_native_estimator_batches_global_phase_parameters(monkeypatch):
    pytest.importorskip("qiskit")
    _install_fake_binding(monkeypatch)

    from qiskit import QuantumCircuit
    from qiskit.circuit import Parameter
    from qiskit.circuit.library import GlobalPhaseGate
    from qiskit.quantum_info import SparsePauliOp
    from qiskit_rocquantum_provider import RocQuantumProvider

    theta = Parameter("theta")
    circuit = QuantumCircuit(1)
    circuit.append(GlobalPhaseGate(theta), [])
    observable = SparsePauliOp.from_list([("Z", 1.0)])

    result = RocQuantumProvider().get_estimator().run(
        [(circuit, observable, [0.1, 0.2])],
    ).result()[0]

    np.testing.assert_allclose(result.data.evs, np.array([0.5, 0.5]))
    assert result.metadata["batched_parameters"] is True
    sim = _FakeQuantumSimulator.instances[-1]
    assert sim.batch_size() == 2
    assert sim.ops == []
    assert sim.batch_ops == []
    assert sim.matrices == []
    assert sim.batch_expectations == [("Z", (0,))]


def test_qiskit_native_estimator_batches_controlled_global_phase_parameters(monkeypatch):
    pytest.importorskip("qiskit")
    _install_fake_binding(monkeypatch)

    from qiskit import QuantumCircuit
    from qiskit.circuit import Parameter
    from qiskit.circuit.library import GlobalPhaseGate
    from qiskit.quantum_info import SparsePauliOp
    from qiskit_rocquantum_provider import RocQuantumProvider

    theta = Parameter("theta")
    circuit = QuantumCircuit(2)
    circuit.append(GlobalPhaseGate(theta).control(2, annotated=False), [0, 1])
    observable = SparsePauliOp.from_list([("IZ", 1.0)])

    result = RocQuantumProvider().get_estimator().run(
        [(circuit, observable, [0.2, 0.6])],
    ).result()[0]

    np.testing.assert_allclose(result.data.evs, np.array([0.5, 0.5]))
    assert result.metadata["batched_parameters"] is True
    sim = _FakeQuantumSimulator.instances[-1]
    assert sim.batch_size() == 2
    assert ("RZ", (0,), (0.1, 0.3)) in sim.batch_ops
    assert sim.matrices == []
    assert sim.controlled_matrices == []
    assert sim.batch_expectations == [("Z", (0,))]


def test_qiskit_native_estimator_batches_controlled_u_parameters(monkeypatch):
    pytest.importorskip("qiskit")
    _install_fake_binding(monkeypatch)

    from qiskit import QuantumCircuit
    from qiskit.circuit import Parameter
    from qiskit.quantum_info import SparsePauliOp
    from qiskit_rocquantum_provider import RocQuantumProvider

    theta = Parameter("theta")
    phi = Parameter("phi")
    lam = Parameter("lam")
    gamma = Parameter("gamma")
    circuit = QuantumCircuit(2)
    circuit.cu(theta, phi, lam, gamma, 0, 1)
    observable = SparsePauliOp.from_list([("IZ", 1.0)])

    result = RocQuantumProvider().get_estimator().run(
        [
            (
                circuit,
                observable,
                {
                    (theta, phi, lam, gamma): [
                        [0.2, 0.3, 0.4, 0.5],
                        [0.6, 0.7, 0.8, 0.9],
                    ],
                },
            )
        ],
    ).result()[0]

    np.testing.assert_allclose(result.data.evs, np.array([0.5, 0.5]))
    assert result.metadata["batched_parameters"] is True
    sim = _FakeQuantumSimulator.instances[-1]
    assert sim.batch_size() == 2
    assert sim.ops == [("CNOT", (0, 1), ()), ("CNOT", (0, 1), ())]
    assert sim.batch_ops == [
        ("P", (0,), (0.5, 0.9)),
        ("P", (0,), (0.35, 0.75)),
        ("P", (1,), (0.1, 0.3)),
        ("RZ", (1,), (-0.35, -0.75)),
        ("RY", (1,), (-0.1, -0.3)),
        ("RZ", (1,), (0.0, 0.0)),
        ("RZ", (1,), (0.0, 0.0)),
        ("RY", (1,), (0.1, 0.3)),
        ("RZ", (1,), (0.3, 0.7)),
    ]
    assert sim.matrices == []
    assert sim.controlled_matrices == []
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


def test_qiskit_estimator_reuses_scalar_multiple_pauli_readout(monkeypatch):
    pytest.importorskip("qiskit")
    _install_fake_binding(monkeypatch)

    from qiskit import QuantumCircuit
    from qiskit.quantum_info import SparsePauliOp
    from qiskit_rocquantum_provider import RocQuantumProvider

    circuit = QuantumCircuit(1)
    observables = [
        SparsePauliOp.from_list([("Z", 1.0)]),
        SparsePauliOp.from_list([("Z", 2.0)]),
    ]

    result = RocQuantumProvider().get_estimator().run(
        [(circuit, observables)],
    ).result()[0]

    np.testing.assert_allclose(result.data.evs, np.array([0.5, 1.0], dtype=float))
    sim = _FakeQuantumSimulator.instances[-1]
    assert sim.expectations == [("Z", (0,))]
    assert sim.statevector_reads == 0


def test_qiskit_estimator_reuses_scalar_multiple_pauli_batch_readout(monkeypatch):
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
        SparsePauliOp.from_list([("Z", 2.0)]),
    ]

    result = RocQuantumProvider().get_estimator().run(
        [(circuit, observables, [0.1, 0.2])],
    ).result()[0]

    np.testing.assert_allclose(result.data.evs, np.array([0.5, 1.0], dtype=float))
    assert result.metadata["batched_parameters"] is True
    sim = _FakeQuantumSimulator.instances[-1]
    assert sim.batch_size() == 2
    assert sim.batch_ops == [("RY", (0,), (0.1, 0.2))]
    assert sim.batch_expectations == [("Z", (0,))]
    assert sim.statevector_reads == 0


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


def test_qiskit_native_estimator_batches_controlled_two_qubit_rotation_parameters(monkeypatch):
    pytest.importorskip("qiskit")
    _install_fake_binding(monkeypatch)

    from qiskit import QuantumCircuit
    from qiskit.circuit import Parameter
    from qiskit.circuit.library import RXXGate, RZZGate
    from qiskit.quantum_info import SparsePauliOp
    from qiskit_rocquantum_provider import RocQuantumProvider

    theta = Parameter("theta")
    phi = Parameter("phi")
    circuit = QuantumCircuit(4)
    circuit.append(RZZGate(theta).control(1, annotated=False), [0, 1, 2])
    circuit.append(RXXGate(phi).control(2, annotated=False), [0, 1, 2, 3])
    observable = SparsePauliOp.from_list([("IIIZ", 1.0)])

    result = RocQuantumProvider().get_estimator().run(
        [
            (
                circuit,
                observable,
                {
                    (theta, phi): [
                        [0.4, 0.8],
                        [0.6, 1.0],
                    ],
                },
            )
        ],
    ).result()[0]

    np.testing.assert_allclose(result.data.evs, np.array([0.5, 0.5]))
    assert result.metadata["batched_parameters"] is True
    sim = _FakeQuantumSimulator.instances[-1]
    assert sim.batch_size() == 2
    assert ("RZ", (2,), (0.2, 0.3)) in sim.batch_ops
    assert ("RZ", (2,), (-0.2, -0.3)) in sim.batch_ops
    assert ("RZ", (3,), (0.2, 0.25)) in sim.batch_ops
    assert sim.matrices == []
    assert sim.controlled_matrices == []
    assert sim.batch_expectations == [("Z", (0,))]


def test_qiskit_native_estimator_batches_controlled_xx_yy_parameters(monkeypatch):
    pytest.importorskip("qiskit")
    _install_fake_binding(monkeypatch)

    from qiskit import QuantumCircuit
    from qiskit.circuit import Parameter
    from qiskit.circuit.library import XXPlusYYGate
    from qiskit.quantum_info import SparsePauliOp
    from qiskit_rocquantum_provider import RocQuantumProvider

    theta = Parameter("theta")
    beta = Parameter("beta")
    circuit = QuantumCircuit(3)
    circuit.append(XXPlusYYGate(theta, beta).control(1, annotated=False), [0, 1, 2])
    observable = SparsePauliOp.from_list([("IIZ", 1.0)])

    result = RocQuantumProvider().get_estimator().run(
        [
            (
                circuit,
                observable,
                {
                    (theta, beta): [
                        [0.6, 0.4],
                        [0.8, 0.5],
                    ],
                },
            )
        ],
    ).result()[0]

    np.testing.assert_allclose(result.data.evs, np.array([0.5, 0.5]))
    assert result.metadata["batched_parameters"] is True
    sim = _FakeQuantumSimulator.instances[-1]
    assert sim.batch_size() == 2
    assert ("CRZ", (0, 1), (0.4, 0.5)) in sim.batch_ops
    assert ("CRY", (0, 2), (-0.3, -0.4)) in sim.batch_ops
    assert ("CRY", (0, 1), (-0.3, -0.4)) in sim.batch_ops
    assert sim.matrices == []
    assert sim.controlled_matrices == []
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
    matrix = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
    observable = Operator(matrix)

    np.testing.assert_allclose(
        estimate_observable_batch(runtime, observable, 1),
        np.array([1.0, -1.0], dtype=float),
    )
    assert runtime.calls[0][1] == (0,)
    np.testing.assert_allclose(runtime.calls[0][0], matrix)


def test_qiskit_estimator_reuses_scalar_multiple_dense_operator_readout(monkeypatch):
    pytest.importorskip("qiskit")
    _install_fake_binding(monkeypatch)
    monkeypatch.setattr(_FakeQuantumSimulator, "enable_matrix_expectation", True)

    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Operator
    from qiskit_rocquantum_provider import RocQuantumProvider

    circuit = QuantumCircuit(1)
    matrix = np.array([[1.0, 1.0], [1.0, 1.0]], dtype=np.complex128)

    result = RocQuantumProvider().get_estimator().run(
        [(circuit, [Operator(matrix), Operator(2.0 * matrix)])],
    ).result()[0]

    np.testing.assert_allclose(result.data.evs, np.array([1.0, 2.0], dtype=float))
    sim = _FakeQuantumSimulator.instances[-1]
    assert len(sim.matrix_expectations) == 1
    np.testing.assert_allclose(sim.matrix_expectations[0][0], matrix)
    assert sim.matrix_expectations[0][1] == (0,)
    assert sim.statevector_reads == 0


def test_qiskit_estimator_reuses_scalar_multiple_dense_operator_batch_readout(monkeypatch):
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
    matrix = np.array([[1.0, 1.0], [1.0, 1.0]], dtype=np.complex128)

    result = RocQuantumProvider().get_estimator().run(
        [(circuit, [Operator(matrix), Operator(2.0 * matrix)], [0.1, 0.2])],
    ).result()[0]

    np.testing.assert_allclose(result.data.evs, np.array([1.0, 2.0], dtype=float))
    assert result.metadata["batched_parameters"] is True
    sim = _FakeQuantumSimulator.instances[-1]
    assert sim.batch_size() == 2
    assert sim.batch_ops == [("RY", (0,), (0.1, 0.2))]
    assert len(sim.matrix_batch_expectations) == 1
    np.testing.assert_allclose(sim.matrix_batch_expectations[0][0], matrix)
    assert sim.matrix_batch_expectations[0][1] == (0,)
    assert sim.matrix_expectations == []
    assert sim.statevector_reads == 0


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
    matrix = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
    observable = Operator(matrix)

    assert estimate_observable(runtime, observable, 2) == pytest.approx(0.25)
    assert runtime.calls[0][1] == (0,)
    np.testing.assert_allclose(runtime.calls[0][0], matrix)


def test_qiskit_estimator_dense_operator_uses_dimension_metadata_targets(monkeypatch):
    pytest.importorskip("qiskit")
    _install_fake_binding(monkeypatch)

    from qiskit.quantum_info import Operator
    from qiskit_rocquantum_provider.estimator import estimate_observable

    class Runtime:
        def __init__(self):
            self.calls = []

        def expectation_matrix(self, matrix, targets):
            self.calls.append((np.asarray(matrix, dtype=np.complex128), tuple(targets)))
            return 0.75 + 0.0j

    runtime = Runtime()
    matrix = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
    observable = Operator(
        matrix,
        input_dims=(2, 1),
        output_dims=(2, 1),
    )

    assert estimate_observable(runtime, observable, 2) == pytest.approx(0.75)
    assert runtime.calls[0][1] == (1,)
    np.testing.assert_allclose(runtime.calls[0][0], matrix)


def test_qiskit_estimator_dense_operator_accepts_explicit_targets(monkeypatch):
    pytest.importorskip("qiskit")
    _install_fake_binding(monkeypatch)

    from qiskit.quantum_info import Operator
    from qiskit_rocquantum_provider.estimator import estimate_observable, estimate_observable_batch

    class Runtime:
        def __init__(self):
            self.calls = []

        def expectation_matrix(self, matrix, targets):
            self.calls.append(("single", np.asarray(matrix, dtype=np.complex128), tuple(targets)))
            return 0.75 + 0.0j

        def expectation_matrix_batch(self, matrix, targets):
            self.calls.append(("batch", np.asarray(matrix, dtype=np.complex128), tuple(targets)))
            return np.array([0.75 + 0.0j, -0.25 + 0.0j], dtype=np.complex128)

        def batch_size(self):
            return 2

    runtime = Runtime()
    matrix = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
    observable = Operator(matrix)

    assert estimate_observable(runtime, (observable, [1]), 2) == pytest.approx(0.75)
    np.testing.assert_allclose(
        estimate_observable_batch(runtime, {"operator": observable, "qargs": [1]}, 2),
        np.array([0.75, -0.25], dtype=float),
    )
    assert runtime.calls[0][2] == (1,)
    assert runtime.calls[1][2] == (1,)
    np.testing.assert_allclose(runtime.calls[0][1], matrix)


def test_qiskit_estimator_dense_operator_mapping_wrapper_uses_default_targets(monkeypatch):
    pytest.importorskip("qiskit")
    _install_fake_binding(monkeypatch)

    from qiskit.quantum_info import Operator
    from qiskit_rocquantum_provider.estimator import estimate_observable

    class Runtime:
        def __init__(self):
            self.calls = []

        def expectation_matrix(self, matrix, targets):
            self.calls.append((np.asarray(matrix, dtype=np.complex128), tuple(targets)))
            return 0.5 + 0.0j

    runtime = Runtime()
    matrix = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
    observable = Operator(matrix)

    assert estimate_observable(runtime, {"operator": observable}, 1) == pytest.approx(0.5)
    assert runtime.calls[0][1] == (0,)
    np.testing.assert_allclose(runtime.calls[0][0], matrix)


def test_qiskit_estimator_diagonal_dense_operator_uses_pauli_terms(monkeypatch):
    pytest.importorskip("qiskit")
    _install_fake_binding(monkeypatch)

    from qiskit.quantum_info import Operator
    from qiskit_rocquantum_provider.estimator import estimate_observable

    class Runtime:
        def __init__(self):
            self.calls = []

        def expectation_pauli_string(self, pauli_string, targets):
            self.calls.append((pauli_string, tuple(targets)))
            return 0.25

        def expectation_matrix(self, matrix, targets):
            raise AssertionError("diagonal dense operators should lower to Pauli terms")

    runtime = Runtime()
    observable = Operator(np.diag([1.0, -1.0]).astype(np.complex128))

    assert estimate_observable(runtime, observable, 1) == pytest.approx(0.25)
    assert runtime.calls == [("Z", (0,))]


def test_qiskit_estimator_diagonal_dense_operator_batch_uses_pauli_terms(monkeypatch):
    pytest.importorskip("qiskit")
    _install_fake_binding(monkeypatch)

    from qiskit.quantum_info import Operator
    from qiskit_rocquantum_provider.estimator import estimate_observable_batch

    class BatchRuntime:
        def __init__(self):
            self.calls = []

        def batch_size(self):
            return 2

        def expectation_pauli_string_batch(self, pauli_string, targets):
            self.calls.append((pauli_string, tuple(targets)))
            return np.array([0.25, -0.5], dtype=float)

        def expectation_matrix_batch(self, matrix, targets):
            raise AssertionError("diagonal dense operators should lower to Pauli terms")

    runtime = BatchRuntime()
    observable = Operator(np.diag([2.0, -2.0]).astype(np.complex128))

    np.testing.assert_allclose(
        estimate_observable_batch(runtime, observable, 1),
        np.array([0.5, -1.0], dtype=float),
    )
    assert runtime.calls == [("Z", (0,))]


def test_qiskit_estimator_diagonal_dense_operator_respects_explicit_targets(monkeypatch):
    pytest.importorskip("qiskit")
    _install_fake_binding(monkeypatch)

    from qiskit.quantum_info import Operator
    from qiskit_rocquantum_provider.estimator import estimate_observable

    class Runtime:
        def __init__(self):
            self.calls = []

        def expectation_pauli_string(self, pauli_string, targets):
            self.calls.append((pauli_string, tuple(targets)))
            return 0.75

        def expectation_matrix(self, matrix, targets):
            raise AssertionError("diagonal dense operators should lower to Pauli terms")

    runtime = Runtime()
    observable = Operator(np.diag([1.0, -1.0]).astype(np.complex128))

    assert estimate_observable(runtime, (observable, [1]), 2) == pytest.approx(0.75)
    assert runtime.calls == [("Z", (1,))]


def test_qiskit_estimator_two_qubit_diagonal_dense_operator_uses_pauli_terms(monkeypatch):
    pytest.importorskip("qiskit")
    _install_fake_binding(monkeypatch)

    from qiskit.quantum_info import Operator
    from qiskit_rocquantum_provider.estimator import estimate_observable

    class Runtime:
        def __init__(self):
            self.calls = []

        def expectation_pauli_string(self, pauli_string, targets):
            self.calls.append((pauli_string, tuple(targets)))
            if pauli_string == "ZZ" and tuple(targets) == (0, 1):
                return 0.25
            raise AssertionError((pauli_string, targets))

        def expectation_matrix(self, matrix, targets):
            raise AssertionError("diagonal dense operators should lower to Pauli terms")

    runtime = Runtime()
    observable = Operator(np.diag([1.0, -1.0, -1.0, 1.0]).astype(np.complex128))

    assert estimate_observable(runtime, observable, 2) == pytest.approx(0.25)
    assert runtime.calls == [("ZZ", (0, 1))]


def test_qiskit_estimator_dense_operator_rejects_invalid_explicit_targets(monkeypatch):
    pytest.importorskip("qiskit")
    _install_fake_binding(monkeypatch)

    from qiskit.quantum_info import Operator
    from qiskit_rocquantum_provider.estimator import estimate_observable

    class Runtime:
        def expectation_matrix(self, matrix, targets):
            raise AssertionError("invalid target metadata should fail before runtime dispatch")

    observable = Operator(np.diag([1.0, -0.5]).astype(np.complex128))

    with pytest.raises(ValueError, match="explicit targets"):
        estimate_observable(Runtime(), (observable, [0, 0]), 2)


def test_qiskit_estimator_dense_scalar_operator_rejects_nonempty_explicit_targets(monkeypatch):
    pytest.importorskip("qiskit")
    _install_fake_binding(monkeypatch)

    from qiskit.quantum_info import Operator
    from qiskit_rocquantum_provider.estimator import estimate_observable

    class Runtime:
        def expectation_matrix(self, matrix, targets):
            raise AssertionError("scalar observables should fail or fold before runtime dispatch")

    observable = Operator(np.array([[2.0]], dtype=np.complex128))

    assert estimate_observable(Runtime(), (observable, []), 1) == pytest.approx(2.0)
    with pytest.raises(ValueError, match="explicit targets"):
        estimate_observable(Runtime(), (observable, [0]), 1)


def test_qiskit_estimator_dense_identity_operator_folds_without_readout(monkeypatch):
    pytest.importorskip("qiskit")
    _install_fake_binding(monkeypatch)

    from qiskit.quantum_info import Operator
    from qiskit_rocquantum_provider.estimator import estimate_observable

    class Runtime:
        def expectation_pauli_string(self, pauli_string, targets):
            raise AssertionError("dense identity observables should fold before Pauli readout")

        def expectation_matrix(self, matrix, targets):
            raise AssertionError("dense identity observables should fold before matrix readout")

    observable = Operator(2.0 * np.eye(32, dtype=np.complex128))

    assert estimate_observable(Runtime(), observable, 5) == pytest.approx(2.0)


def test_qiskit_estimator_dense_identity_operator_batch_folds_without_readout(monkeypatch):
    pytest.importorskip("qiskit")
    _install_fake_binding(monkeypatch)

    from qiskit.quantum_info import Operator
    from qiskit_rocquantum_provider.estimator import estimate_observable_batch

    class Runtime:
        def batch_size(self):
            return 2

        def expectation_pauli_string_batch(self, pauli_string, targets):
            raise AssertionError("dense identity observables should fold before Pauli readout")

        def expectation_matrix_batch(self, matrix, targets):
            raise AssertionError("dense identity observables should fold before matrix readout")

    observable = Operator(3.0 * np.eye(32, dtype=np.complex128))

    np.testing.assert_allclose(
        estimate_observable_batch(Runtime(), observable, 5),
        np.array([3.0, 3.0], dtype=float),
    )


def test_qiskit_native_estimator_accepts_dense_operator(monkeypatch):
    pytest.importorskip("qiskit")
    _install_fake_binding(monkeypatch)
    monkeypatch.setattr(_FakeQuantumSimulator, "enable_matrix_expectation", True)

    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Operator
    from qiskit_rocquantum_provider import RocQuantumProvider

    circuit = QuantumCircuit(1)
    matrix = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
    observable = Operator(matrix)

    result = RocQuantumProvider().get_estimator().run(
        [(circuit, observable)],
    ).result()[0]

    assert float(result.data.evs) == pytest.approx(0.0)
    assert result.metadata["native"] is True
    sim = _FakeQuantumSimulator.instances[-1]
    assert sim.matrix_expectations[0][1] == (0,)
    np.testing.assert_allclose(sim.matrix_expectations[0][0], matrix)
    assert sim.expectations == []
    assert sim.statevector_reads == 0


def test_qiskit_native_estimator_dense_operator_uses_runtime_statevector_fallback(monkeypatch):
    pytest.importorskip("qiskit")
    _install_fake_binding(monkeypatch)

    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Operator
    from qiskit_rocquantum_provider import RocQuantumProvider

    circuit = QuantumCircuit(1)
    observable = Operator(np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128))

    result = RocQuantumProvider().get_estimator().run(
        [(circuit, observable)],
    ).result()[0]

    assert float(result.data.evs) == pytest.approx(0.0)
    sim = _FakeQuantumSimulator.instances[-1]
    assert sim.matrix_expectations == []
    assert sim.expectations == []
    assert sim.statevector_reads == 1


def test_qiskit_native_estimator_accepts_partial_dense_operator(monkeypatch):
    pytest.importorskip("qiskit")
    _install_fake_binding(monkeypatch)
    monkeypatch.setattr(_FakeQuantumSimulator, "enable_matrix_expectation", True)

    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Operator
    from qiskit_rocquantum_provider import RocQuantumProvider

    circuit = QuantumCircuit(2)
    matrix = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
    observable = Operator(matrix)

    result = RocQuantumProvider().get_estimator().run(
        [(circuit, observable)],
    ).result()[0]

    assert float(result.data.evs) == pytest.approx(0.0)
    sim = _FakeQuantumSimulator.instances[-1]
    assert sim.matrix_expectations[0][1] == (0,)
    np.testing.assert_allclose(sim.matrix_expectations[0][0], matrix)
    assert sim.expectations == []
    assert sim.statevector_reads == 0


def test_qiskit_native_estimator_accepts_explicit_dense_operator_targets(monkeypatch):
    pytest.importorskip("qiskit")
    _install_fake_binding(monkeypatch)
    monkeypatch.setattr(_FakeQuantumSimulator, "enable_matrix_expectation", True)

    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Operator
    from qiskit_rocquantum_provider import RocQuantumProvider

    circuit = QuantumCircuit(2)
    matrix = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
    observable = Operator(matrix)

    result = RocQuantumProvider().get_estimator().run(
        [(circuit, (observable, [1]))],
    ).result()[0]

    assert float(result.data.evs) == pytest.approx(0.0)
    sim = _FakeQuantumSimulator.instances[-1]
    assert sim.matrix_expectations[0][1] == (1,)
    np.testing.assert_allclose(sim.matrix_expectations[0][0], matrix)
    assert sim.expectations == []
    assert sim.statevector_reads == 0


def test_qiskit_native_estimator_accepts_dense_operator_mapping_without_targets(monkeypatch):
    pytest.importorskip("qiskit")
    _install_fake_binding(monkeypatch)
    monkeypatch.setattr(_FakeQuantumSimulator, "enable_matrix_expectation", True)

    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Operator
    from qiskit_rocquantum_provider import RocQuantumProvider

    circuit = QuantumCircuit(1)
    matrix = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
    observable = {"operator": Operator(matrix)}

    result = RocQuantumProvider().get_estimator().run(
        [(circuit, observable)],
    ).result()[0]

    assert float(result.data.evs) == pytest.approx(0.0)
    sim = _FakeQuantumSimulator.instances[-1]
    assert sim.matrix_expectations[0][1] == (0,)
    np.testing.assert_allclose(sim.matrix_expectations[0][0], matrix)
    assert sim.expectations == []
    assert sim.statevector_reads == 0


def test_qiskit_native_estimator_diagonal_dense_operator_uses_pauli_terms(monkeypatch):
    pytest.importorskip("qiskit")
    _install_fake_binding(monkeypatch)

    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Operator
    from qiskit_rocquantum_provider import RocQuantumProvider

    circuit = QuantumCircuit(1)
    observable = Operator(np.diag([1.0, -1.0]).astype(np.complex128))

    result = RocQuantumProvider().get_estimator().run(
        [(circuit, observable)],
    ).result()[0]

    assert float(result.data.evs) == pytest.approx(0.5)
    sim = _FakeQuantumSimulator.instances[-1]
    assert sim.expectations == [("Z", (0,))]
    assert sim.matrix_expectations == []
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
    matrix = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
    observable = Operator(matrix)

    result = RocQuantumProvider().get_estimator().run(
        [(circuit, observable, [0.1, 0.2])],
    ).result()[0]

    np.testing.assert_allclose(result.data.evs, np.array([0.0, 0.0], dtype=float))
    assert result.metadata["batched_parameters"] is True
    sim = _FakeQuantumSimulator.instances[-1]
    assert sim.batch_size() == 2
    assert sim.batch_ops == [("RY", (0,), (0.1, 0.2))]
    assert sim.matrix_batch_expectations[0][1] == (0,)
    np.testing.assert_allclose(sim.matrix_batch_expectations[0][0], matrix)
    assert sim.matrix_expectations == []


def test_qiskit_native_estimator_batches_diagonal_dense_operator_as_pauli_terms(monkeypatch):
    pytest.importorskip("qiskit")
    _install_fake_binding(monkeypatch)

    from qiskit import QuantumCircuit
    from qiskit.circuit import Parameter
    from qiskit.quantum_info import Operator
    from qiskit_rocquantum_provider import RocQuantumProvider

    theta = Parameter("theta")
    circuit = QuantumCircuit(1)
    circuit.ry(theta, 0)
    observable = Operator(np.diag([2.0, -2.0]).astype(np.complex128))

    result = RocQuantumProvider().get_estimator().run(
        [(circuit, observable, [0.1, 0.2])],
    ).result()[0]

    np.testing.assert_allclose(result.data.evs, np.array([1.0, 1.0], dtype=float))
    assert result.metadata["batched_parameters"] is True
    sim = _FakeQuantumSimulator.instances[-1]
    assert sim.batch_size() == 2
    assert sim.batch_ops == [("RY", (0,), (0.1, 0.2))]
    assert sim.batch_expectations == [("Z", (0,))]
    assert sim.matrix_batch_expectations == []
    assert sim.statevector_reads == 0


def test_qiskit_native_estimator_reuses_duplicate_dense_operator_batch(monkeypatch):
    pytest.importorskip("qiskit")
    _install_fake_binding(monkeypatch)
    monkeypatch.setattr(_FakeQuantumSimulator, "enable_matrix_expectation", True)

    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Operator
    from qiskit_rocquantum_provider import RocQuantumProvider

    circuit = QuantumCircuit(1)
    circuit.h(0)
    matrix = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
    observables = [Operator(matrix.copy()), Operator(matrix.copy())]

    result = RocQuantumProvider().get_estimator().run(
        [(circuit, observables)],
    ).result()[0]

    np.testing.assert_allclose(result.data.evs, np.array([0.0, 0.0]))
    sim = _FakeQuantumSimulator.instances[-1]
    assert sim.ops == [("H", (0,), ())]
    assert sim.total_gate_applications == 1
    assert len(sim.matrix_expectations) == 1
    assert sim.matrix_expectations[0][1] == (0,)
    np.testing.assert_allclose(sim.matrix_expectations[0][0], matrix)
    assert sim.expectations == []


def test_qiskit_native_estimator_accepts_dense_scalar_operator_without_readout(monkeypatch):
    pytest.importorskip("qiskit")
    _install_fake_binding(monkeypatch)

    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Operator
    from qiskit_rocquantum_provider import RocQuantumProvider

    circuit = QuantumCircuit(1)
    circuit.h(0)
    observable = Operator(np.array([[2.0]], dtype=np.complex128))

    result = RocQuantumProvider().get_estimator().run(
        [(circuit, observable)],
    ).result()[0]

    assert result.data.evs == pytest.approx(2.0)
    sim = _FakeQuantumSimulator.instances[-1]
    assert sim.ops == [("H", (0,), ())]
    assert sim.matrix_expectations == []
    assert sim.expectations == []


def test_qiskit_native_estimator_accepts_dense_identity_operator_without_readout(monkeypatch):
    pytest.importorskip("qiskit")
    _install_fake_binding(monkeypatch)

    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Operator
    from qiskit_rocquantum_provider import RocQuantumProvider

    circuit = QuantumCircuit(5)
    circuit.h(0)
    observable = Operator(2.0 * np.eye(32, dtype=np.complex128))

    result = RocQuantumProvider().get_estimator().run(
        [(circuit, observable)],
    ).result()[0]

    assert result.data.evs == pytest.approx(2.0)
    sim = _FakeQuantumSimulator.instances[-1]
    assert sim.ops == [("H", (0,), ())]
    assert sim.matrix_expectations == []
    assert sim.expectations == []
    assert sim.statevector_reads == 0


def test_qiskit_native_estimator_batches_dense_scalar_operator_without_readout(monkeypatch):
    pytest.importorskip("qiskit")
    _install_fake_binding(monkeypatch)

    from qiskit import QuantumCircuit
    from qiskit.circuit import Parameter
    from qiskit.quantum_info import Operator
    from qiskit_rocquantum_provider import RocQuantumProvider

    theta = Parameter("theta")
    circuit = QuantumCircuit(1)
    circuit.ry(theta, 0)
    observable = Operator(np.array([[2.0]], dtype=np.complex128))

    result = RocQuantumProvider().get_estimator().run(
        [(circuit, observable, [0.1, 0.2])],
    ).result()[0]

    np.testing.assert_allclose(result.data.evs, np.array([2.0, 2.0]))
    assert result.metadata["batched_parameters"] is True
    sim = _FakeQuantumSimulator.instances[-1]
    assert sim.batch_size() == 2
    assert sim.batch_ops == [("RY", (0,), (0.1, 0.2))]
    assert sim.matrix_batch_expectations == []
    assert sim.batch_expectations == []


def test_qiskit_native_estimator_batches_dense_identity_operator_without_readout(monkeypatch):
    pytest.importorskip("qiskit")
    _install_fake_binding(monkeypatch)

    from qiskit import QuantumCircuit
    from qiskit.circuit import Parameter
    from qiskit.quantum_info import Operator
    from qiskit_rocquantum_provider import RocQuantumProvider

    theta = Parameter("theta")
    circuit = QuantumCircuit(5)
    circuit.ry(theta, 0)
    observable = Operator(3.0 * np.eye(32, dtype=np.complex128))

    result = RocQuantumProvider().get_estimator().run(
        [(circuit, observable, [0.1, 0.2])],
    ).result()[0]

    np.testing.assert_allclose(result.data.evs, np.array([3.0, 3.0]))
    assert result.metadata["batched_parameters"] is True
    sim = _FakeQuantumSimulator.instances[-1]
    assert sim.batch_size() == 2
    assert sim.batch_ops == [("RY", (0,), (0.1, 0.2))]
    assert sim.matrix_batch_expectations == []
    assert sim.batch_expectations == []
    assert sim.statevector_reads == 0


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
    matrix = np.array(
        [
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
        ],
        dtype=np.complex128,
    )
    observable = Operator(matrix)

    assert provider.estimate_expectation(circuit, observable) == pytest.approx(1.0)
    sim = _FakeQuantumSimulator.instances[-1]
    assert sim.matrix_expectations[0][1] == (0, 1)
    np.testing.assert_allclose(sim.matrix_expectations[0][0], matrix)
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
    matrix = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
    observable = Operator(matrix)

    assert provider.estimate_expectation(circuit, observable) == pytest.approx(0.0)
    sim = _FakeQuantumSimulator.instances[-1]
    assert sim.matrix_expectations[0][1] == (0,)
    np.testing.assert_allclose(sim.matrix_expectations[0][0], matrix)
    assert sim.expectations == []
    assert sim.statevector_reads == 0


def test_qiskit_provider_estimates_explicit_partial_dense_operator_natively(monkeypatch):
    pytest.importorskip("qiskit")
    _install_fake_binding(monkeypatch)
    monkeypatch.setattr(_FakeQuantumSimulator, "enable_matrix_expectation", True)

    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Operator
    from qiskit_rocquantum_provider import RocQuantumProvider

    provider = RocQuantumProvider()
    circuit = QuantumCircuit(2)
    matrix = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
    observable = Operator(matrix)

    assert provider.estimate_expectation(
        circuit,
        {"operator": observable, "targets": [1]},
    ) == pytest.approx(0.0)
    sim = _FakeQuantumSimulator.instances[-1]
    assert sim.matrix_expectations[0][1] == (1,)
    np.testing.assert_allclose(sim.matrix_expectations[0][0], matrix)
    assert sim.expectations == []
    assert sim.statevector_reads == 0


def test_qiskit_provider_estimates_diagonal_dense_operator_as_pauli_terms(monkeypatch):
    pytest.importorskip("qiskit")
    _install_fake_binding(monkeypatch)

    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Operator
    from qiskit_rocquantum_provider import RocQuantumProvider

    provider = RocQuantumProvider()
    circuit = QuantumCircuit(1)
    observable = Operator(np.diag([1.0, -1.0]).astype(np.complex128))

    assert provider.estimate_expectation(circuit, observable) == pytest.approx(0.5)
    sim = _FakeQuantumSimulator.instances[-1]
    assert sim.expectations == [("Z", (0,))]
    assert sim.matrix_expectations == []
    assert sim.statevector_reads == 0


def test_qiskit_provider_estimates_dense_identity_operator_without_readout(monkeypatch):
    pytest.importorskip("qiskit")
    _install_fake_binding(monkeypatch)

    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Operator
    from qiskit_rocquantum_provider import RocQuantumProvider

    provider = RocQuantumProvider()
    circuit = QuantumCircuit(5)
    circuit.h(0)
    observable = Operator(2.0 * np.eye(32, dtype=np.complex128))

    assert provider.estimate_expectation(circuit, observable) == pytest.approx(2.0)
    sim = _FakeQuantumSimulator.instances[-1]
    assert sim.ops == [("H", (0,), ())]
    assert sim.matrix_expectations == []
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


def test_pennylane_single_execute_caches_duplicate_analytic_measurements(monkeypatch):
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
        return qml.expval(qml.PauliZ(0)), qml.var(qml.PauliZ(0)), qml.expval(qml.PauliZ(0))

    assert circuit() == pytest.approx((0.5, 0.75, 0.5))
    sim = _FakeQuantumSimulator.instances[-1]
    assert sim.expectations == [("Z", (0,))]
    assert sim.statevector_reads == 0


def test_pennylane_single_execute_caches_duplicate_probabilities(monkeypatch):
    pytest.importorskip("pennylane")
    _install_fake_binding(monkeypatch)
    monkeypatch.setattr(_FakeQuantumSimulator, "enable_probabilities", True)
    for name in list(sys.modules):
        if name.startswith("pennylane_rocq"):
            sys.modules.pop(name)

    import pennylane as qml

    dev = qml.device("lightning.rocq", wires=1)

    @qml.qnode(dev)
    def circuit():
        qml.RY(0.123, wires=0)
        return qml.probs(wires=[0]), qml.probs(wires=[0])

    first, second = circuit()
    np.testing.assert_allclose(first, np.array([1 / 3, 2 / 3]))
    np.testing.assert_allclose(second, np.array([1 / 3, 2 / 3]))
    sim = _FakeQuantumSimulator.instances[-1]
    assert sim.probability_requests == [(0,)]
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


def test_pennylane_batch_execute_caches_duplicate_analytic_measurements(monkeypatch):
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
            [
                qml.expval(qml.PauliZ(0)),
                qml.expval(qml.PauliZ(0)),
                qml.var(qml.PauliZ(0)),
            ],
        ),
        qml.tape.QuantumScript(
            [qml.RY(0.2, wires=0)],
            [
                qml.expval(qml.PauliZ(0)),
                qml.expval(qml.PauliZ(0)),
                qml.var(qml.PauliZ(0)),
            ],
        ),
    ]

    results = dev.batch_execute(circuits)

    assert len(results) == 2
    for result in results:
        assert result == pytest.approx((0.5, 0.5, 0.75))
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


def test_pennylane_batch_execute_keeps_static_controlled_wrappers_batched(monkeypatch):
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
                qml.RY(0.1, wires=0),
                qml.ctrl(qml.Hadamard(wires=2), control=[0, 1]),
                qml.ctrl(qml.SWAP(wires=[2, 3]), control=[0, 1]),
            ],
            [qml.expval(qml.PauliZ(0))],
        ),
        qml.tape.QuantumScript(
            [
                qml.RY(0.2, wires=0),
                qml.ctrl(qml.Hadamard(wires=2), control=[0, 1]),
                qml.ctrl(qml.SWAP(wires=[2, 3]), control=[0, 1]),
            ],
            [qml.expval(qml.PauliZ(0))],
        ),
    ]

    assert dev.batch_execute(circuits) == pytest.approx((0.5, 0.5))
    sim = _FakeQuantumSimulator.instances[-1]
    assert sim.batch_size() == 2
    assert sim.batch_ops == [("RY", (0,), (0.1, 0.2))]
    assert sim.ops == [
        ("RY", (2,), (np.pi / 4,)),
        ("MCX", (0, 1, 2), ()),
        ("RY", (2,), (-np.pi / 4,)),
        ("MCX", (0, 1, 2, 3), ()),
        ("MCX", (0, 1, 3, 2), ()),
        ("MCX", (0, 1, 2, 3), ()),
    ]
    assert sim.matrices == []
    assert sim.controlled_matrices == []
    assert sim.batch_expectations == [("Z", (0,))]


def test_pennylane_batch_execute_batches_parametric_controlled_wrappers(monkeypatch):
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
                qml.ctrl(qml.RX(0.1, wires=2), control=[0, 1]),
                qml.ctrl(qml.PhaseShift(0.2, wires=2), control=[0, 1], control_values=[True, False]),
                qml.ctrl(qml.Rot(0.3, 0.4, 0.5, wires=2), control=[0, 1]),
                qml.ctrl(qml.PSWAP(0.6, wires=[2, 3]), control=[0, 1]),
            ],
            [qml.expval(qml.PauliZ(0))],
        ),
        qml.tape.QuantumScript(
            [
                qml.ctrl(qml.RX(0.7, wires=2), control=[0, 1]),
                qml.ctrl(qml.PhaseShift(0.8, wires=2), control=[0, 1], control_values=[True, False]),
                qml.ctrl(qml.Rot(0.9, 1.0, 1.1, wires=2), control=[0, 1]),
                qml.ctrl(qml.PSWAP(1.2, wires=[2, 3]), control=[0, 1]),
            ],
            [qml.expval(qml.PauliZ(0))],
        ),
    ]

    assert dev.batch_execute(circuits) == pytest.approx((0.5, 0.5))
    sim = _FakeQuantumSimulator.instances[-1]
    assert sim.batch_size() == 2
    assert sim.matrices == []
    assert sim.controlled_matrices == []
    assert sim.batch_expectations == [("Z", (0,))]
    assert len(sim.batch_ops) >= 8
    assert all(name == "RZ" for name, _, _ in sim.batch_ops)
    assert any(params == (-0.025, -0.175) for _, _, params in sim.batch_ops)
    assert any(params == (0.05, 0.2) for _, _, params in sim.batch_ops)
    assert any(params == (0.15, 0.3) for _, _, params in sim.batch_ops)


def test_pennylane_batch_execute_keeps_static_controlled_siswap_batched(monkeypatch):
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
                qml.RY(0.1, wires=0),
                qml.ctrl(qml.SISWAP(wires=[2, 3]), control=[0, 1]),
            ],
            [qml.expval(qml.PauliZ(0))],
        ),
        qml.tape.QuantumScript(
            [
                qml.RY(0.2, wires=0),
                qml.ctrl(qml.SISWAP(wires=[2, 3]), control=[0, 1]),
            ],
            [qml.expval(qml.PauliZ(0))],
        ),
    ]

    assert dev.batch_execute(circuits) == pytest.approx((0.5, 0.5))
    sim = _FakeQuantumSimulator.instances[-1]
    assert sim.batch_size() == 2
    assert sim.batch_ops == [("RY", (0,), (0.1, 0.2))]
    assert ("MCX", (0, 1, 2, 3), ()) in sim.ops
    assert any(name == "RZ" for name, _, _ in sim.ops)
    assert sim.matrices == []
    assert sim.controlled_matrices == []
    assert sim.batch_expectations == [("Z", (0,))]


def test_pennylane_batch_execute_keeps_static_controlled_ecr_batched(monkeypatch):
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
                qml.RY(0.1, wires=0),
                qml.ctrl(qml.ECR(wires=[2, 3]), control=[0, 1]),
            ],
            [qml.expval(qml.PauliZ(0))],
        ),
        qml.tape.QuantumScript(
            [
                qml.RY(0.2, wires=0),
                qml.ctrl(qml.ECR(wires=[2, 3]), control=[0, 1]),
            ],
            [qml.expval(qml.PauliZ(0))],
        ),
    ]

    assert dev.batch_execute(circuits) == pytest.approx((0.5, 0.5))
    sim = _FakeQuantumSimulator.instances[-1]
    assert sim.batch_size() == 2
    assert sim.batch_ops == [("RY", (0,), (0.1, 0.2))]
    assert ("MCX", (0, 1, 2, 3), ()) in sim.ops
    assert any(name == "RZ" for name, _, _ in sim.ops)
    assert sim.matrices == []
    assert sim.controlled_matrices == []
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


def test_pennylane_batch_execute_batches_controlled_global_phase_wrappers(monkeypatch):
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
                qml.ctrl(qml.GlobalPhase(0.2, wires=[]), control=[0, 1], control_values=[True, False]),
                qml.RY(0.4, wires=2),
            ],
            [qml.expval(qml.PauliZ(0))],
        ),
        qml.tape.QuantumScript(
            [
                qml.ctrl(qml.GlobalPhase(0.6, wires=[]), control=[0, 1], control_values=[True, False]),
                qml.RY(0.8, wires=2),
            ],
            [qml.expval(qml.PauliZ(0))],
        ),
    ]

    assert dev.batch_execute(circuits) == pytest.approx((0.5, 0.5))
    sim = _FakeQuantumSimulator.instances[-1]

    assert sim.batch_size() == 2
    assert ("RY", (2,), (0.4, 0.8)) in sim.batch_ops
    assert any(params == (-0.1, -0.3) for _, _, params in sim.batch_ops)
    assert sim.ops[0] == ("X", (1,), ())
    assert sim.ops[-1] == ("X", (1,), ())
    assert sim.matrices == []
    assert sim.controlled_matrices == []
    assert sim.batch_expectations == [("Z", (0,))]


def test_pennylane_batch_execute_keeps_controlled_phase_root_wrappers_native(monkeypatch):
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
                qml.ctrl(qml.S(wires=2), control=[0, 1], control_values=[False, True]),
                qml.ctrl(qml.adjoint(qml.T(wires=2)), control=[0, 1]),
                qml.RY(0.4, wires=2),
            ],
            [qml.expval(qml.PauliZ(0))],
        ),
        qml.tape.QuantumScript(
            [
                qml.ctrl(qml.S(wires=2), control=[0, 1], control_values=[False, True]),
                qml.ctrl(qml.adjoint(qml.T(wires=2)), control=[0, 1]),
                qml.RY(0.8, wires=2),
            ],
            [qml.expval(qml.PauliZ(0))],
        ),
    ]

    assert dev.batch_execute(circuits) == pytest.approx((0.5, 0.5))
    sim = _FakeQuantumSimulator.instances[-1]

    assert ("RY", (2,), (0.4, 0.8)) in sim.batch_ops
    assert any(name == "RZ" for name, _, _ in sim.ops)
    assert any(name == "RZ" and params == (-np.pi / 16,) for name, _, params in sim.ops)
    assert any(name in {"CNOT", "CX"} for name, _, _ in sim.ops)
    assert sim.ops[0] == ("X", (0,), ())
    assert sum(op == ("X", (0,), ()) for op in sim.ops) >= 2
    assert sim.matrices == []
    assert sim.controlled_matrices == []
    assert sim.batch_expectations == [("Z", (0,))]


def test_pennylane_controlled_phase_variant_wrappers_decompose_natively(monkeypatch):
    pytest.importorskip("pennylane")
    _install_fake_binding(monkeypatch)
    for name in list(sys.modules):
        if name.startswith("pennylane_rocq"):
            sys.modules.pop(name)

    import pennylane as qml

    dev = qml.device("lightning.rocq", wires=3)

    @qml.qnode(dev)
    def circuit():
        qml.ctrl(qml.CPhaseShift00(0.3, wires=[1, 2]), control=[0], control_values=[False])
        return qml.expval(qml.PauliZ(0))

    assert circuit() == pytest.approx(0.5)
    sim = _FakeQuantumSimulator.instances[-1]

    assert sim.ops[:3] == [("X", (0,), ()), ("X", (1,), ()), ("X", (2,), ())]
    assert sim.ops[-3:] == [("X", (2,), ()), ("X", (1,), ()), ("X", (0,), ())]
    assert any(name == "RZ" and params == (0.075,) for name, _, params in sim.ops)
    assert sim.matrices == []
    assert sim.controlled_matrices == []
    assert sim.expectations == [("Z", (0,))]


def test_pennylane_batch_execute_batches_controlled_phase_variant_wrappers(monkeypatch):
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
                qml.ctrl(qml.CPhaseShift01(0.2, wires=[1, 2]), control=[0], control_values=[False]),
                qml.RY(0.4, wires=2),
            ],
            [qml.expval(qml.PauliZ(0))],
        ),
        qml.tape.QuantumScript(
            [
                qml.ctrl(qml.CPhaseShift01(0.6, wires=[1, 2]), control=[0], control_values=[False]),
                qml.RY(0.8, wires=2),
            ],
            [qml.expval(qml.PauliZ(0))],
        ),
    ]

    assert dev.batch_execute(circuits) == pytest.approx((0.5, 0.5))
    sim = _FakeQuantumSimulator.instances[-1]

    assert sim.batch_size() == 2
    assert sim.ops[:2] == [("X", (0,), ()), ("X", (1,), ())]
    assert sim.ops[-2:] == [("X", (1,), ()), ("X", (0,), ())]
    assert ("RZ", (0,), (0.05, 0.15)) in sim.batch_ops
    assert ("RY", (2,), (0.4, 0.8)) in sim.batch_ops
    assert sim.matrices == []
    assert sim.controlled_matrices == []
    assert sim.batch_expectations == [("Z", (0,))]


def test_pennylane_controlled_adjoint_phase_root_wrappers_decompose_natively(monkeypatch):
    pytest.importorskip("pennylane")
    _install_fake_binding(monkeypatch)
    for name in list(sys.modules):
        if name.startswith("pennylane_rocq"):
            sys.modules.pop(name)

    import pennylane as qml

    dev = qml.device("lightning.rocq", wires=3)

    @qml.qnode(dev)
    def circuit():
        qml.ctrl(qml.adjoint(qml.S(wires=2)), control=[0, 1], control_values=[True, False])
        qml.ctrl(qml.adjoint(qml.T(wires=2)), control=[0, 1])
        return qml.expval(qml.PauliZ(0))

    assert circuit() == pytest.approx(0.5)
    sim = _FakeQuantumSimulator.instances[-1]

    assert any(name == "RZ" and params == (-np.pi / 8,) for name, _, params in sim.ops)
    assert any(name == "RZ" and params == (-np.pi / 16,) for name, _, params in sim.ops)
    assert any(name in {"CNOT", "CX"} for name, _, _ in sim.ops)
    assert sim.ops[0] == ("X", (1,), ())
    assert sim.matrices == []
    assert sim.controlled_matrices == []
    assert sim.expectations == [("Z", (0,))]


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


def test_pennylane_batch_execute_keeps_adjoint_phase_root_controlled_sequence_native(monkeypatch):
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
                qml.ControlledSequence(qml.adjoint(qml.S(wires=2)), control=[0, 1]),
                qml.RY(0.2, wires=2),
            ],
            [qml.expval(qml.PauliZ(0))],
        ),
        qml.tape.QuantumScript(
            [
                qml.ControlledSequence(qml.adjoint(qml.S(wires=2)), control=[0, 1]),
                qml.RY(0.8, wires=2),
            ],
            [qml.expval(qml.PauliZ(0))],
        ),
    ]

    assert dev.batch_execute(circuits) == pytest.approx((0.5, 0.5))
    sim = _FakeQuantumSimulator.instances[-1]

    assert sim.ops == [
        ("CP", (0, 2), (-np.pi,)),
        ("CP", (1, 2), (-np.pi / 2,)),
    ]
    assert sim.batch_ops == [("RY", (2,), (0.2, 0.8))]
    assert sim.matrices == []
    assert sim.controlled_matrices == []
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


def test_pennylane_batch_execute_keeps_select_product_basis_native_batched(monkeypatch):
    pytest.importorskip("pennylane")
    _install_fake_binding(monkeypatch)
    for name in list(sys.modules):
        if name.startswith("pennylane_rocq"):
            sys.modules.pop(name)

    import pennylane as qml

    selected_product = qml.prod(
        qml.BasisEmbedding(np.array([1, 0]), wires=[1, 2]),
        qml.PauliX(wires=2),
    )
    dev = qml.device("lightning.rocq", wires=3)
    circuits = [
        qml.tape.QuantumScript(
            [
                qml.Select([selected_product, qml.Identity(wires=1)], control=[0]),
                qml.RY(0.2, wires=0),
            ],
            [qml.expval(qml.PauliZ(0))],
        ),
        qml.tape.QuantumScript(
            [
                qml.Select([selected_product, qml.Identity(wires=1)], control=[0]),
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
        ("CNOT", (0, 1), ()),
        ("X", (0,), ()),
        ("X", (0,), ()),
        ("CNOT", (0, 2), ()),
        ("X", (0,), ()),
    ]
    assert sim.batch_ops == [("RY", (0,), (0.2, 0.8))]
    assert sim.matrices == []
    assert sim.controlled_matrices == []
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


def test_pennylane_batch_execute_reuses_scaled_pauli_expval_readout(monkeypatch):
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
            [qml.expval(qml.PauliZ(0)), qml.expval(2.0 * qml.PauliZ(0))],
        ),
        qml.tape.QuantumScript(
            [qml.RY(0.2, wires=0)],
            [qml.expval(qml.PauliZ(0)), qml.expval(2.0 * qml.PauliZ(0))],
        ),
    ]

    results = dev.batch_execute(circuits)

    assert len(results) == 2
    for result in results:
        np.testing.assert_allclose(result, (0.5, 1.0))
    sim = _FakeQuantumSimulator.instances[-1]
    assert sim.batch_size() == 2
    assert sim.batch_ops == [("RY", (0,), (0.1, 0.2))]
    assert sim.batch_expectations == [("Z", (0,))]


def test_pennylane_batch_execute_reuses_reordered_scaled_pauli_sum_readout(monkeypatch):
    pytest.importorskip("pennylane")
    _install_fake_binding(monkeypatch)
    for name in list(sys.modules):
        if name.startswith("pennylane_rocq"):
            sys.modules.pop(name)

    import pennylane as qml

    base_observable = qml.sum(qml.PauliZ(0), qml.PauliX(0))
    scaled_reordered = qml.sum(2.0 * qml.PauliX(0), 2.0 * qml.PauliZ(0))
    dev = qml.device("lightning.rocq", wires=1)
    circuits = [
        qml.tape.QuantumScript(
            [qml.RY(0.1, wires=0)],
            [qml.expval(base_observable), qml.expval(scaled_reordered)],
        ),
        qml.tape.QuantumScript(
            [qml.RY(0.2, wires=0)],
            [qml.expval(base_observable), qml.expval(scaled_reordered)],
        ),
    ]

    results = dev.batch_execute(circuits)

    assert len(results) == 2
    for result in results:
        np.testing.assert_allclose(result, (1.0, 2.0))
    sim = _FakeQuantumSimulator.instances[-1]
    assert sim.batch_size() == 2
    assert sim.batch_ops == [("RY", (0,), (0.1, 0.2))]
    assert sim.batch_expectations == [("X", (0,)), ("Z", (0,))]


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
    assert sim.batch_expectations == [("Z", (0,))]


def test_pennylane_batch_execute_uses_batched_hermitian_expval(monkeypatch):
    pytest.importorskip("pennylane")
    _install_fake_binding(monkeypatch)
    monkeypatch.setattr(_FakeQuantumSimulator, "enable_matrix_expectation", True)
    for name in list(sys.modules):
        if name.startswith("pennylane_rocq"):
            sys.modules.pop(name)

    import pennylane as qml

    dev = qml.device("lightning.rocq", wires=1)
    matrix = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
    observable = qml.Hermitian(matrix, wires=0)
    circuits = [
        qml.tape.QuantumScript([qml.RY(0.1, wires=0)], [qml.expval(observable)]),
        qml.tape.QuantumScript([qml.RY(0.2, wires=0)], [qml.expval(observable)]),
    ]

    assert dev.batch_execute(circuits) == pytest.approx((0.0, 0.0))
    sim = _FakeQuantumSimulator.instances[-1]
    assert sim.batch_size() == 2
    assert sim.batch_ops == [("RY", (0,), (0.1, 0.2))]
    assert sim.matrix_batch_expectations[0][1] == (0,)
    np.testing.assert_allclose(sim.matrix_batch_expectations[0][0], matrix)
    assert sim.matrix_expectations == []
    assert sim.batch_expectations == []
    assert sim.statevector_reads == 0


def test_pennylane_batch_execute_uses_batched_scaled_hermitian_expval(monkeypatch):
    pytest.importorskip("pennylane")
    _install_fake_binding(monkeypatch)
    monkeypatch.setattr(_FakeQuantumSimulator, "enable_matrix_expectation", True)
    for name in list(sys.modules):
        if name.startswith("pennylane_rocq"):
            sys.modules.pop(name)

    import pennylane as qml

    dev = qml.device("lightning.rocq", wires=1)
    matrix = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
    observable = qml.s_prod(2.0, qml.Hermitian(matrix, wires=0))
    circuits = [
        qml.tape.QuantumScript([qml.RY(0.1, wires=0)], [qml.expval(observable)]),
        qml.tape.QuantumScript([qml.RY(0.2, wires=0)], [qml.expval(observable)]),
    ]

    assert dev.batch_execute(circuits) == pytest.approx((0.0, 0.0))
    sim = _FakeQuantumSimulator.instances[-1]
    assert sim.batch_size() == 2
    assert sim.batch_ops == [("RY", (0,), (0.1, 0.2))]
    assert sim.matrix_batch_expectations[0][1] == (0,)
    np.testing.assert_allclose(sim.matrix_batch_expectations[0][0], matrix)
    assert sim.matrix_expectations == []
    assert sim.batch_expectations == []
    assert sim.statevector_reads == 0


def test_pennylane_batch_execute_reuses_scaled_hermitian_expval_readout(monkeypatch):
    pytest.importorskip("pennylane")
    _install_fake_binding(monkeypatch)
    monkeypatch.setattr(_FakeQuantumSimulator, "enable_matrix_expectation", True)
    for name in list(sys.modules):
        if name.startswith("pennylane_rocq"):
            sys.modules.pop(name)

    import pennylane as qml

    dev = qml.device("lightning.rocq", wires=1)
    matrix = np.array([[1.0, 1.0], [1.0, 1.0]], dtype=np.complex128)
    circuits = [
        qml.tape.QuantumScript(
            [qml.RY(0.1, wires=0)],
            [qml.expval(qml.Hermitian(matrix, wires=0)), qml.expval(2.0 * qml.Hermitian(matrix.copy(), wires=0))],
        ),
        qml.tape.QuantumScript(
            [qml.RY(0.2, wires=0)],
            [qml.expval(qml.Hermitian(matrix, wires=0)), qml.expval(2.0 * qml.Hermitian(matrix.copy(), wires=0))],
        ),
    ]

    results = dev.batch_execute(circuits)

    assert len(results) == 2
    for result in results:
        np.testing.assert_allclose(result, (1.0, 2.0))
    sim = _FakeQuantumSimulator.instances[-1]
    assert sim.batch_size() == 2
    assert sim.batch_ops == [("RY", (0,), (0.1, 0.2))]
    assert len(sim.matrix_batch_expectations) == 1
    np.testing.assert_allclose(sim.matrix_batch_expectations[0][0], matrix)
    assert sim.matrix_batch_expectations[0][1] == (0,)
    assert sim.matrix_expectations == []
    assert sim.batch_expectations == []
    assert sim.statevector_reads == 0


def test_pennylane_batch_execute_folds_identity_hermitian_readouts(monkeypatch):
    pytest.importorskip("pennylane")
    _install_fake_binding(monkeypatch)
    for name in list(sys.modules):
        if name.startswith("pennylane_rocq"):
            sys.modules.pop(name)

    import pennylane as qml

    dev = qml.device("lightning.rocq", wires=1)
    observable = qml.s_prod(2.0, qml.Hermitian(np.eye(2, dtype=np.complex128), wires=0))
    circuits = [
        qml.tape.QuantumScript([qml.RY(0.1, wires=0)], [qml.expval(observable), qml.var(observable)]),
        qml.tape.QuantumScript([qml.RY(0.2, wires=0)], [qml.expval(observable), qml.var(observable)]),
    ]

    np.testing.assert_allclose(
        np.asarray(dev.batch_execute(circuits), dtype=float),
        np.array([[2.0, 0.0], [2.0, 0.0]], dtype=float),
    )
    sim = _FakeQuantumSimulator.instances[-1]
    assert sim.batch_size() == 2
    assert sim.batch_ops == [("RY", (0,), (0.1, 0.2))]
    assert sim.matrix_batch_expectations == []
    assert sim.batch_expectations == []
    assert sim.statevector_reads == 0


def test_pennylane_batch_execute_folds_diagonal_hermitian_to_pauli_terms(monkeypatch):
    pytest.importorskip("pennylane")
    _install_fake_binding(monkeypatch)
    for name in list(sys.modules):
        if name.startswith("pennylane_rocq"):
            sys.modules.pop(name)

    import pennylane as qml

    dev = qml.device("lightning.rocq", wires=1)
    observable = qml.s_prod(2.0, qml.Hermitian(np.diag([1.0, -1.0]).astype(np.complex128), wires=0))
    circuits = [
        qml.tape.QuantumScript([qml.RY(0.1, wires=0)], [qml.expval(observable), qml.var(observable)]),
        qml.tape.QuantumScript([qml.RY(0.2, wires=0)], [qml.expval(observable), qml.var(observable)]),
    ]

    np.testing.assert_allclose(
        np.asarray(dev.batch_execute(circuits), dtype=float),
        np.array([[1.0, 3.0], [1.0, 3.0]], dtype=float),
    )
    sim = _FakeQuantumSimulator.instances[-1]
    assert sim.batch_size() == 2
    assert sim.batch_ops == [("RY", (0,), (0.1, 0.2))]
    assert sim.batch_expectations == [("Z", (0,))]
    assert sim.matrix_batch_expectations == []
    assert sim.statevector_reads == 0


def test_pennylane_batch_execute_uses_projector_idempotent_variance(monkeypatch):
    pytest.importorskip("pennylane")
    _install_fake_binding(monkeypatch)
    for name in list(sys.modules):
        if name.startswith("pennylane_rocq"):
            sys.modules.pop(name)

    import pennylane as qml

    dev = qml.device("lightning.rocq", wires=2)
    observable = qml.Projector([1, 0], wires=[0, 1])
    circuits = [
        qml.tape.QuantumScript([qml.RY(0.1, wires=0)], [qml.var(observable)]),
        qml.tape.QuantumScript([qml.RY(0.2, wires=0)], [qml.var(observable)]),
    ]

    np.testing.assert_allclose(
        np.asarray(dev.batch_execute(circuits), dtype=float),
        np.array([0.109375, 0.109375], dtype=float),
    )
    sim = _FakeQuantumSimulator.instances[-1]
    assert sim.batch_size() == 2
    assert sim.batch_ops == [("RY", (0,), (0.1, 0.2))]
    assert sim.batch_expectations == [
        ("Z", (0,)),
        ("Z", (1,)),
        ("ZZ", (0, 1)),
    ]
    assert sim.matrix_batch_expectations == []
    assert sim.statevector_reads == 0


def test_pennylane_batch_execute_uses_scaled_projector_idempotent_variance(monkeypatch):
    pytest.importorskip("pennylane")
    _install_fake_binding(monkeypatch)
    for name in list(sys.modules):
        if name.startswith("pennylane_rocq"):
            sys.modules.pop(name)

    import pennylane as qml

    dev = qml.device("lightning.rocq", wires=2)
    observable = qml.s_prod(2.0, qml.Projector([1, 0], wires=[0, 1]))
    circuits = [
        qml.tape.QuantumScript([qml.RY(0.1, wires=0)], [qml.var(observable)]),
        qml.tape.QuantumScript([qml.RY(0.2, wires=0)], [qml.var(observable)]),
    ]

    np.testing.assert_allclose(
        np.asarray(dev.batch_execute(circuits), dtype=float),
        np.array([0.4375, 0.4375], dtype=float),
    )
    sim = _FakeQuantumSimulator.instances[-1]
    assert sim.batch_size() == 2
    assert sim.batch_ops == [("RY", (0,), (0.1, 0.2))]
    assert sim.batch_expectations == [
        ("Z", (0,)),
        ("Z", (1,)),
        ("ZZ", (0, 1)),
    ]
    assert sim.matrix_batch_expectations == []
    assert sim.statevector_reads == 0


def test_pennylane_batch_execute_uses_batched_mixed_matrix_sum_expval(monkeypatch):
    pytest.importorskip("pennylane")
    _install_fake_binding(monkeypatch)
    monkeypatch.setattr(_FakeQuantumSimulator, "enable_matrix_expectation", True)
    for name in list(sys.modules):
        if name.startswith("pennylane_rocq"):
            sys.modules.pop(name)

    import pennylane as qml

    dev = qml.device("lightning.rocq", wires=1)
    matrix = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
    observable = qml.sum(
        qml.Hermitian(matrix, wires=0),
        2.0 * qml.Hermitian(matrix.copy(), wires=0),
        0.5 * qml.PauliZ(0),
    )
    circuits = [
        qml.tape.QuantumScript([qml.RY(0.1, wires=0)], [qml.expval(observable)]),
        qml.tape.QuantumScript([qml.RY(0.2, wires=0)], [qml.expval(observable)]),
    ]

    assert dev.batch_execute(circuits) == pytest.approx((0.25, 0.25))
    sim = _FakeQuantumSimulator.instances[-1]
    assert sim.batch_size() == 2
    assert sim.batch_ops == [("RY", (0,), (0.1, 0.2))]
    assert sim.batch_expectations == [("Z", (0,))]
    assert len(sim.matrix_batch_expectations) == 1
    assert sim.matrix_batch_expectations[0][1] == (0,)
    np.testing.assert_allclose(sim.matrix_batch_expectations[0][0], matrix)
    assert sim.matrix_expectations == []
    assert sim.statevector_reads == 0


def test_pennylane_batch_execute_reuses_sum_component_matrix_readouts(monkeypatch):
    pytest.importorskip("pennylane")
    _install_fake_binding(monkeypatch)
    monkeypatch.setattr(_FakeQuantumSimulator, "enable_matrix_expectation", True)
    for name in list(sys.modules):
        if name.startswith("pennylane_rocq"):
            sys.modules.pop(name)

    import pennylane as qml

    dev = qml.device("lightning.rocq", wires=1)
    matrix = np.array([[1.0, 1.0], [1.0, 1.0]], dtype=np.complex128)
    observable = qml.sum(qml.PauliZ(0), qml.Hermitian(matrix, wires=0))
    circuits = [
        qml.tape.QuantumScript(
            [qml.RY(0.1, wires=0)],
            [qml.expval(qml.PauliZ(0)), qml.expval(observable)],
        ),
        qml.tape.QuantumScript(
            [qml.RY(0.2, wires=0)],
            [qml.expval(qml.PauliZ(0)), qml.expval(observable)],
        ),
    ]

    results = dev.batch_execute(circuits)

    assert len(results) == 2
    for result in results:
        np.testing.assert_allclose(result, (0.5, 1.5))
    sim = _FakeQuantumSimulator.instances[-1]
    assert sim.batch_size() == 2
    assert sim.batch_ops == [("RY", (0,), (0.1, 0.2))]
    assert sim.batch_expectations == [("Z", (0,))]
    assert len(sim.matrix_batch_expectations) == 1
    np.testing.assert_allclose(sim.matrix_batch_expectations[0][0], matrix)
    assert sim.matrix_batch_expectations[0][1] == (0,)
    assert sim.matrix_expectations == []
    assert sim.statevector_reads == 0


def test_pennylane_batch_execute_uses_batched_mixed_matrix_sum_variance(monkeypatch):
    pytest.importorskip("pennylane")
    _install_fake_binding(monkeypatch)
    monkeypatch.setattr(_FakeQuantumSimulator, "enable_matrix_expectation", True)
    for name in list(sys.modules):
        if name.startswith("pennylane_rocq"):
            sys.modules.pop(name)

    import pennylane as qml

    dev = qml.device("lightning.rocq", wires=1)
    matrix = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
    observable = qml.sum(
        qml.Hermitian(matrix, wires=0),
        2.0 * qml.Hermitian(matrix.copy(), wires=0),
        0.5 * qml.PauliZ(0),
    )
    circuits = [
        qml.tape.QuantumScript([qml.RY(0.1, wires=0)], [qml.var(observable)]),
        qml.tape.QuantumScript([qml.RY(0.2, wires=0)], [qml.var(observable)]),
    ]

    assert dev.batch_execute(circuits) == pytest.approx((9.0, 9.0))
    sim = _FakeQuantumSimulator.instances[-1]
    assert sim.batch_size() == 2
    assert sim.batch_ops == [("RY", (0,), (0.1, 0.2))]
    assert sim.batch_expectations == []
    assert len(sim.matrix_batch_expectations) == 2
    np.testing.assert_allclose(sim.matrix_batch_expectations[0][0], np.array([[1.0, 6.0], [6.0, -1.0]]))
    np.testing.assert_allclose(sim.matrix_batch_expectations[1][0], np.eye(2))
    assert sim.matrix_expectations == []
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
    matrix = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
    observable = qml.Hermitian(matrix, wires=0)
    circuits = [
        qml.tape.QuantumScript([qml.RY(0.1, wires=0)], [qml.var(observable)]),
        qml.tape.QuantumScript([qml.RY(0.2, wires=0)], [qml.var(observable)]),
    ]

    assert dev.batch_execute(circuits) == pytest.approx((1.0, 1.0))
    sim = _FakeQuantumSimulator.instances[-1]
    assert sim.batch_size() == 2
    assert sim.batch_ops == [("RY", (0,), (0.1, 0.2))]
    assert len(sim.matrix_batch_expectations) == 2
    np.testing.assert_allclose(sim.matrix_batch_expectations[0][0], matrix)
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


def test_pennylane_batch_execute_uses_batched_finite_shot_probabilities(monkeypatch):
    pytest.importorskip("pennylane")
    _install_fake_binding(monkeypatch)
    for name in list(sys.modules):
        if name.startswith("pennylane_rocq"):
            sys.modules.pop(name)

    import pennylane as qml
    from pennylane.exceptions import PennyLaneDeprecationWarning

    with pytest.warns(PennyLaneDeprecationWarning):
        dev = qml.device("lightning.rocq", wires=2, shots=8)
    circuits = [
        qml.tape.QuantumScript(
            [qml.CRX(0.1, wires=[0, 1])],
            [qml.probs(wires=[0, 1]), qml.probs(wires=[0])],
        ),
        qml.tape.QuantumScript(
            [qml.CRX(0.2, wires=[0, 1])],
            [qml.probs(wires=[0, 1]), qml.probs(wires=[0])],
        ),
    ]

    results = dev.batch_execute(circuits)

    assert len(results) == 2
    for full_probs, marginal_probs in results:
        assert full_probs.shape == (4,)
        assert marginal_probs.shape == (2,)
        np.testing.assert_allclose(np.sum(full_probs), 1.0)
        np.testing.assert_allclose(np.sum(marginal_probs), 1.0)
        np.testing.assert_allclose(full_probs * 8, np.rint(full_probs * 8))
        np.testing.assert_allclose(marginal_probs * 8, np.rint(marginal_probs * 8))
    sim = _FakeQuantumSimulator.instances[-1]
    assert sim.batch_size() == 2
    assert sim.batch_ops == [("CRX", (0, 1), (0.1, 0.2))]
    assert sim.measurements == []
    assert sim.probability_requests == [(0, 1)]


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


def test_pennylane_batch_execute_uses_batched_counts_all_outcomes(monkeypatch):
    pytest.importorskip("pennylane")
    _install_fake_binding(monkeypatch)
    for name in list(sys.modules):
        if name.startswith("pennylane_rocq"):
            sys.modules.pop(name)

    import pennylane as qml
    from pennylane.exceptions import PennyLaneDeprecationWarning

    with pytest.warns(PennyLaneDeprecationWarning):
        dev = qml.device("lightning.rocq", wires=2, shots=4)
    circuits = [
        qml.tape.QuantumScript([qml.CRX(0.1, wires=[0, 1])], [qml.counts(wires=[0, 1], all_outcomes=True)]),
        qml.tape.QuantumScript([qml.CRX(0.2, wires=[0, 1])], [qml.counts(wires=[0, 1], all_outcomes=True)]),
    ]

    results = dev.batch_execute(circuits)

    assert len(results) == 2
    for result in results:
        assert set(result) == {"00", "01", "10", "11"}
        assert result["00"] + result["01"] + result["10"] + result["11"] == 4
    sim = _FakeQuantumSimulator.instances[-1]
    assert sim.batch_size() == 2
    assert sim.batch_ops == [("CRX", (0, 1), (0.1, 0.2))]
    assert sim.measurements == []
    assert sim.probability_requests == [(0, 1)]


def test_pennylane_batch_execute_uses_batched_multiple_finite_shot_measurements(monkeypatch):
    pytest.importorskip("pennylane")
    _install_fake_binding(monkeypatch)
    for name in list(sys.modules):
        if name.startswith("pennylane_rocq"):
            sys.modules.pop(name)

    import pennylane as qml
    from pennylane.exceptions import PennyLaneDeprecationWarning

    with pytest.warns(PennyLaneDeprecationWarning):
        dev = qml.device("lightning.rocq", wires=2, shots=4)
    circuits = [
        qml.tape.QuantumScript(
            [qml.CRX(0.1, wires=[0, 1])],
            [qml.sample(wires=[0]), qml.counts(wires=[1], all_outcomes=True)],
        ),
        qml.tape.QuantumScript(
            [qml.CRX(0.2, wires=[0, 1])],
            [qml.sample(wires=[0]), qml.counts(wires=[1], all_outcomes=True)],
        ),
    ]

    results = dev.batch_execute(circuits)

    assert len(results) == 2
    for sample_result, count_result in results:
        assert sample_result.shape == (4,)
        assert np.isin(sample_result, [0, 1]).all()
        assert set(count_result) == {"0", "1"}
        assert count_result["0"] + count_result["1"] == 4
    sim = _FakeQuantumSimulator.instances[-1]
    assert sim.batch_size() == 2
    assert sim.batch_ops == [("CRX", (0, 1), (0.1, 0.2))]
    assert sim.measurements == []
    assert sim.probability_requests == [(0, 1)]


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
    observable = qml.Hermitian(np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128), wires=0)

    @qml.qnode(dev)
    def hermitian_circuit():
        return qml.expval(observable)

    assert hermitian_circuit() == pytest.approx(0.0)
    sim = _FakeQuantumSimulator.instances[-1]
    assert sim.expectations == []
    assert sim.statevector_reads == 1


def test_pennylane_hermitian_identity_expval_folds_without_readout(monkeypatch):
    pytest.importorskip("pennylane")
    _install_fake_binding(monkeypatch)
    for name in list(sys.modules):
        if name.startswith("pennylane_rocq"):
            sys.modules.pop(name)

    import pennylane as qml

    dev = qml.device("lightning.rocq", wires=1)
    observable = qml.s_prod(2.0, qml.Hermitian(np.eye(2, dtype=np.complex128), wires=0))

    @qml.qnode(dev)
    def hermitian_identity_circuit():
        qml.RY(0.123, wires=0)
        return qml.expval(observable)

    assert hermitian_identity_circuit() == pytest.approx(2.0)
    sim = _FakeQuantumSimulator.instances[-1]
    assert sim.ops == [("RY", (0,), (0.123,))]
    assert sim.expectations == []
    assert sim.matrix_expectations == []
    assert sim.statevector_reads == 0


def test_pennylane_diagonal_hermitian_expval_uses_native_pauli_terms(monkeypatch):
    pytest.importorskip("pennylane")
    _install_fake_binding(monkeypatch)
    for name in list(sys.modules):
        if name.startswith("pennylane_rocq"):
            sys.modules.pop(name)

    import pennylane as qml

    dev = qml.device("lightning.rocq", wires=1)
    observable = qml.Hermitian(np.diag([1.0, -1.0]).astype(np.complex128), wires=0)

    @qml.qnode(dev)
    def hermitian_circuit():
        return qml.expval(observable)

    assert hermitian_circuit() == pytest.approx(0.5)
    sim = _FakeQuantumSimulator.instances[-1]
    assert sim.expectations == [("Z", (0,))]
    assert sim.matrix_expectations == []
    assert sim.statevector_reads == 0


def test_pennylane_diagonal_hermitian_var_uses_native_pauli_terms(monkeypatch):
    pytest.importorskip("pennylane")
    _install_fake_binding(monkeypatch)
    for name in list(sys.modules):
        if name.startswith("pennylane_rocq"):
            sys.modules.pop(name)

    import pennylane as qml

    dev = qml.device("lightning.rocq", wires=1)
    observable = qml.Hermitian(np.diag([1.0, -1.0]).astype(np.complex128), wires=0)

    @qml.qnode(dev)
    def hermitian_circuit():
        return qml.var(observable)

    assert hermitian_circuit() == pytest.approx(0.75)
    sim = _FakeQuantumSimulator.instances[-1]
    assert sim.expectations == [("Z", (0,))]
    assert sim.matrix_expectations == []
    assert sim.statevector_reads == 0


def test_pennylane_hermitian_var_uses_single_runtime_statevector_fallback(monkeypatch):
    pytest.importorskip("pennylane")
    _install_fake_binding(monkeypatch)
    for name in list(sys.modules):
        if name.startswith("pennylane_rocq"):
            sys.modules.pop(name)

    import pennylane as qml

    dev = qml.device("lightning.rocq", wires=1)
    observable = qml.Hermitian(np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128), wires=0)

    @qml.qnode(dev)
    def hermitian_var_circuit():
        return qml.var(observable)

    assert hermitian_var_circuit() == pytest.approx(1.0)
    sim = _FakeQuantumSimulator.instances[-1]
    assert sim.expectations == []
    assert sim.matrix_expectations == []
    assert sim.statevector_reads == 1


def test_pennylane_hermitian_identity_var_folds_without_readout(monkeypatch):
    pytest.importorskip("pennylane")
    _install_fake_binding(monkeypatch)
    for name in list(sys.modules):
        if name.startswith("pennylane_rocq"):
            sys.modules.pop(name)

    import pennylane as qml

    dev = qml.device("lightning.rocq", wires=1)
    observable = qml.s_prod(3.0, qml.Hermitian(np.eye(2, dtype=np.complex128), wires=0))

    @qml.qnode(dev)
    def hermitian_identity_var_circuit():
        qml.RY(0.123, wires=0)
        return qml.var(observable)

    assert hermitian_identity_var_circuit() == pytest.approx(0.0)
    sim = _FakeQuantumSimulator.instances[-1]
    assert sim.ops == [("RY", (0,), (0.123,))]
    assert sim.expectations == []
    assert sim.matrix_expectations == []
    assert sim.statevector_reads == 0


def test_pennylane_hermitian_expval_prefers_native_matrix_expectation(monkeypatch):
    pytest.importorskip("pennylane")
    _install_fake_binding(monkeypatch)
    monkeypatch.setattr(_FakeQuantumSimulator, "enable_matrix_expectation", True)
    for name in list(sys.modules):
        if name.startswith("pennylane_rocq"):
            sys.modules.pop(name)

    import pennylane as qml

    dev = qml.device("lightning.rocq", wires=1)
    matrix = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
    observable = qml.Hermitian(matrix, wires=0)

    @qml.qnode(dev)
    def hermitian_circuit():
        return qml.expval(observable)

    assert hermitian_circuit() == pytest.approx(0.0)
    sim = _FakeQuantumSimulator.instances[-1]
    assert sim.expectations == []
    assert sim.statevector_reads == 0
    assert len(sim.matrix_expectations) == 1
    np.testing.assert_allclose(sim.matrix_expectations[0][0], matrix)
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
    matrix = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
    observable = qml.Hermitian(matrix, wires=0)

    @qml.qnode(dev)
    def hermitian_circuit():
        return qml.var(observable)

    assert hermitian_circuit() == pytest.approx(1.0)
    sim = _FakeQuantumSimulator.instances[-1]
    assert sim.expectations == []
    assert sim.statevector_reads == 0
    assert len(sim.matrix_expectations) == 2
    np.testing.assert_allclose(sim.matrix_expectations[0][0], matrix)
    np.testing.assert_allclose(sim.matrix_expectations[1][0], np.eye(2))


def test_pennylane_single_execute_caches_hermitian_moments(monkeypatch):
    pytest.importorskip("pennylane")
    _install_fake_binding(monkeypatch)
    monkeypatch.setattr(_FakeQuantumSimulator, "enable_matrix_expectation", True)
    for name in list(sys.modules):
        if name.startswith("pennylane_rocq"):
            sys.modules.pop(name)

    import pennylane as qml

    dev = qml.device("lightning.rocq", wires=1)
    matrix = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
    observable = qml.Hermitian(matrix, wires=0)

    @qml.qnode(dev)
    def hermitian_circuit():
        return qml.expval(observable), qml.var(observable)

    assert hermitian_circuit() == pytest.approx((0.0, 1.0))
    sim = _FakeQuantumSimulator.instances[-1]
    assert sim.expectations == []
    assert sim.statevector_reads == 0
    assert len(sim.matrix_expectations) == 2
    np.testing.assert_allclose(sim.matrix_expectations[0][0], matrix)
    np.testing.assert_allclose(sim.matrix_expectations[1][0], np.eye(2))


def test_pennylane_scaled_hermitian_uses_native_matrix_expectation(monkeypatch):
    pytest.importorskip("pennylane")
    _install_fake_binding(monkeypatch)
    monkeypatch.setattr(_FakeQuantumSimulator, "enable_matrix_expectation", True)
    for name in list(sys.modules):
        if name.startswith("pennylane_rocq"):
            sys.modules.pop(name)

    import pennylane as qml

    dev = qml.device("lightning.rocq", wires=1)
    matrix = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
    observable = 2.0 * qml.Hermitian(matrix, wires=0)

    @qml.qnode(dev)
    def hermitian_circuit():
        return qml.expval(observable), qml.var(observable)

    assert hermitian_circuit() == pytest.approx((0.0, 4.0))
    sim = _FakeQuantumSimulator.instances[-1]
    assert sim.expectations == []
    assert sim.statevector_reads == 0
    assert len(sim.matrix_expectations) == 2
    np.testing.assert_allclose(sim.matrix_expectations[0][0], matrix)
    np.testing.assert_allclose(sim.matrix_expectations[1][0], np.eye(2))


def test_pennylane_single_execute_reuses_scaled_hermitian_expval_readout(monkeypatch):
    pytest.importorskip("pennylane")
    _install_fake_binding(monkeypatch)
    monkeypatch.setattr(_FakeQuantumSimulator, "enable_matrix_expectation", True)
    for name in list(sys.modules):
        if name.startswith("pennylane_rocq"):
            sys.modules.pop(name)

    import pennylane as qml

    dev = qml.device("lightning.rocq", wires=1)
    matrix = np.array([[1.0, 1.0], [1.0, 1.0]], dtype=np.complex128)

    @qml.qnode(dev)
    def hermitian_circuit():
        return qml.expval(qml.Hermitian(matrix, wires=0)), qml.expval(2.0 * qml.Hermitian(matrix.copy(), wires=0))

    assert hermitian_circuit() == pytest.approx((1.0, 2.0))
    sim = _FakeQuantumSimulator.instances[-1]
    assert sim.expectations == []
    assert sim.statevector_reads == 0
    assert len(sim.matrix_expectations) == 1
    np.testing.assert_allclose(sim.matrix_expectations[0][0], matrix)
    assert sim.matrix_expectations[0][1] == (0,)


def test_pennylane_mixed_matrix_sum_expval_uses_native_readouts(monkeypatch):
    pytest.importorskip("pennylane")
    _install_fake_binding(monkeypatch)
    monkeypatch.setattr(_FakeQuantumSimulator, "enable_matrix_expectation", True)
    for name in list(sys.modules):
        if name.startswith("pennylane_rocq"):
            sys.modules.pop(name)

    import pennylane as qml

    dev = qml.device("lightning.rocq", wires=1)
    matrix = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
    sum_observable = qml.sum(
        qml.Hermitian(matrix, wires=0),
        2.0 * qml.Hermitian(matrix.copy(), wires=0),
        0.5 * qml.PauliZ(0),
    )
    hamiltonian_observable = qml.Hamiltonian(
        [1.0, 2.0, 0.5],
        [qml.Hermitian(matrix, wires=0), qml.Hermitian(matrix.copy(), wires=0), qml.PauliZ(0)],
    )

    @qml.qnode(dev)
    def sum_circuit():
        return qml.expval(sum_observable)

    @qml.qnode(dev)
    def hamiltonian_circuit():
        return qml.expval(hamiltonian_observable)

    for circuit in (sum_circuit, hamiltonian_circuit):
        assert circuit() == pytest.approx(0.25)
        sim = _FakeQuantumSimulator.instances[-1]
        assert sim.expectations == [("Z", (0,))]
        assert sim.statevector_reads == 0
        assert len(sim.matrix_expectations) == 1
        np.testing.assert_allclose(sim.matrix_expectations[0][0], matrix)
        assert sim.matrix_expectations[0][1] == (0,)


def test_pennylane_mixed_matrix_sum_elides_merged_zero_readout(monkeypatch):
    pytest.importorskip("pennylane")
    _install_fake_binding(monkeypatch)
    monkeypatch.setattr(_FakeQuantumSimulator, "enable_matrix_expectation", True)
    for name in list(sys.modules):
        if name.startswith("pennylane_rocq"):
            sys.modules.pop(name)

    import pennylane as qml

    dev = qml.device("lightning.rocq", wires=1)
    matrix = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
    observable = qml.sum(
        qml.Hermitian(matrix, wires=0),
        -1.0 * qml.Hermitian(matrix.copy(), wires=0),
        0.5 * qml.PauliZ(0),
    )

    @qml.qnode(dev)
    def circuit():
        return qml.expval(observable)

    assert circuit() == pytest.approx(0.25)
    sim = _FakeQuantumSimulator.instances[-1]
    assert sim.expectations == [("Z", (0,))]
    assert sim.matrix_expectations == []
    assert sim.statevector_reads == 0


def test_pennylane_mixed_matrix_sum_folds_merged_diagonal_readout(monkeypatch):
    pytest.importorskip("pennylane")
    _install_fake_binding(monkeypatch)
    monkeypatch.setattr(_FakeQuantumSimulator, "enable_matrix_expectation", True)
    for name in list(sys.modules):
        if name.startswith("pennylane_rocq"):
            sys.modules.pop(name)

    import pennylane as qml

    dev = qml.device("lightning.rocq", wires=1)
    matrix = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
    diagonal_after_merge = np.array([[0.25, -1.0], [-1.0, -0.25]], dtype=np.complex128)
    observable = qml.sum(
        qml.Hermitian(matrix, wires=0),
        qml.Hermitian(diagonal_after_merge, wires=0),
    )

    @qml.qnode(dev)
    def circuit():
        return qml.expval(observable)

    assert circuit() == pytest.approx(0.125)
    sim = _FakeQuantumSimulator.instances[-1]
    assert sim.expectations == [("Z", (0,))]
    assert sim.matrix_expectations == []
    assert sim.statevector_reads == 0


def test_pennylane_mixed_matrix_sum_variance_uses_native_dense_moments(monkeypatch):
    pytest.importorskip("pennylane")
    _install_fake_binding(monkeypatch)
    monkeypatch.setattr(_FakeQuantumSimulator, "enable_matrix_expectation", True)
    for name in list(sys.modules):
        if name.startswith("pennylane_rocq"):
            sys.modules.pop(name)

    import pennylane as qml

    dev = qml.device("lightning.rocq", wires=1)
    matrix = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
    observable = qml.sum(
        qml.Hermitian(matrix, wires=0),
        2.0 * qml.Hermitian(matrix.copy(), wires=0),
        0.5 * qml.PauliZ(0),
    )

    @qml.qnode(dev)
    def circuit():
        return qml.var(observable)

    assert circuit() == pytest.approx(9.0)
    sim = _FakeQuantumSimulator.instances[-1]
    assert sim.expectations == []
    assert sim.statevector_reads == 0
    assert len(sim.matrix_expectations) == 2
    np.testing.assert_allclose(sim.matrix_expectations[0][0], np.array([[1.0, 6.0], [6.0, -1.0]]))
    np.testing.assert_allclose(sim.matrix_expectations[1][0], 37.0 * np.eye(2))
    assert sim.matrix_expectations[0][1] == (0,)
    assert sim.matrix_expectations[1][1] == (0,)


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

    dev = qml.device("lightning.rocq", wires=2)

    @qml.qnode(dev)
    def projector_var_circuit():
        return qml.var(qml.Projector([1, 0], wires=[0, 1]))

    assert projector_var_circuit() == pytest.approx(0.109375)
    sim = _FakeQuantumSimulator.instances[-1]
    assert sim.expectations == [
        ("Z", (1,)),
        ("Z", (0,)),
        ("ZZ", (0, 1)),
    ]
    assert sim.statevector_reads == 0


def test_pennylane_scaled_projector_var_uses_idempotent_native_terms(monkeypatch):
    pytest.importorskip("pennylane")
    _install_fake_binding(monkeypatch)
    for name in list(sys.modules):
        if name.startswith("pennylane_rocq"):
            sys.modules.pop(name)

    import pennylane as qml

    dev = qml.device("lightning.rocq", wires=2)
    observable = qml.s_prod(2.0, qml.Projector([1, 0], wires=[0, 1]))

    @qml.qnode(dev)
    def projector_var_circuit():
        return qml.var(observable)

    assert projector_var_circuit() == pytest.approx(0.4375)
    sim = _FakeQuantumSimulator.instances[-1]
    assert sim.expectations == [
        ("Z", (1,)),
        ("Z", (0,)),
        ("ZZ", (0, 1)),
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

    hamiltonian_matrix = _single_qubit_sparse_x(sp)
    observable = qml.SparseHamiltonian(hamiltonian_matrix, wires=[0])
    dev = qml.device("lightning.rocq", wires=1)

    @qml.qnode(dev)
    def expval_circuit():
        return qml.expval(observable)

    @qml.qnode(dev)
    def var_circuit():
        return qml.var(observable)

    assert expval_circuit() == pytest.approx(0.0)
    expval_sim = _FakeQuantumSimulator.instances[-1]
    assert expval_sim.expectations == []
    assert expval_sim.statevector_reads == 1

    assert var_circuit() == pytest.approx(1.0)
    var_sim = _FakeQuantumSimulator.instances[-1]
    assert var_sim.expectations == []
    assert var_sim.statevector_reads == 1


def test_pennylane_sparse_hamiltonian_identity_folds_without_readout(monkeypatch):
    pytest.importorskip("pennylane")
    sp = pytest.importorskip("scipy.sparse")
    _install_fake_binding(monkeypatch)
    for name in list(sys.modules):
        if name.startswith("pennylane_rocq"):
            sys.modules.pop(name)

    import pennylane as qml

    identity = sp.eye(2, dtype=np.complex128, format="csr")
    observable = qml.s_prod(2.0, qml.SparseHamiltonian(identity, wires=[0]))
    dev = qml.device("lightning.rocq", wires=1)

    @qml.qnode(dev)
    def sparse_identity_circuit():
        qml.RY(0.123, wires=0)
        return qml.expval(observable), qml.var(observable)

    assert sparse_identity_circuit() == pytest.approx((2.0, 0.0))
    sim = _FakeQuantumSimulator.instances[-1]
    assert sim.ops == [("RY", (0,), (0.123,))]
    assert sim.expectations == []
    assert sim.sparse_moments == []
    assert sim.statevector_reads == 0


def test_pennylane_diagonal_sparse_hamiltonian_uses_native_pauli_terms(monkeypatch):
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

    assert expval_circuit() == pytest.approx(0.5)
    expval_sim = _FakeQuantumSimulator.instances[-1]
    assert expval_sim.expectations == [("Z", (0,))]
    assert expval_sim.sparse_moments == []
    assert expval_sim.statevector_reads == 0

    assert var_circuit() == pytest.approx(0.75)
    var_sim = _FakeQuantumSimulator.instances[-1]
    assert var_sim.expectations == [("Z", (0,))]
    assert var_sim.sparse_moments == []
    assert var_sim.statevector_reads == 0


def test_pennylane_sparse_hamiltonian_prefers_native_csr_moments(monkeypatch):
    pytest.importorskip("pennylane")
    sp = pytest.importorskip("scipy.sparse")
    _install_fake_binding(monkeypatch)
    monkeypatch.setattr(_FakeQuantumSimulator, "enable_sparse_moments", True)
    for name in list(sys.modules):
        if name.startswith("pennylane_rocq"):
            sys.modules.pop(name)

    import pennylane as qml

    hamiltonian_matrix = _single_qubit_sparse_x(sp)
    observable = qml.SparseHamiltonian(hamiltonian_matrix, wires=[0])
    dev = qml.device("lightning.rocq", wires=1)

    @qml.qnode(dev)
    def expval_circuit():
        return qml.expval(observable)

    @qml.qnode(dev)
    def var_circuit():
        return qml.var(observable)

    assert expval_circuit() == pytest.approx(0.0)
    expval_sim = _FakeQuantumSimulator.instances[-1]
    assert expval_sim.statevector_reads == 0
    assert len(expval_sim.sparse_moments) == 1

    assert var_circuit() == pytest.approx(1.0)
    var_sim = _FakeQuantumSimulator.instances[-1]
    assert var_sim.statevector_reads == 0
    assert len(var_sim.sparse_moments) == 1

    data, indices, indptr, shape = var_sim.sparse_moments[0]
    np.testing.assert_allclose(data, np.array([1.0, 1.0], dtype=np.complex128))
    np.testing.assert_array_equal(indices, np.array([1, 0], dtype=np.int64))
    np.testing.assert_array_equal(indptr, np.array([0, 1, 2], dtype=np.int64))
    assert shape == (2, 2)


def test_pennylane_single_execute_caches_sparse_hamiltonian_moments(monkeypatch):
    pytest.importorskip("pennylane")
    sp = pytest.importorskip("scipy.sparse")
    _install_fake_binding(monkeypatch)
    monkeypatch.setattr(_FakeQuantumSimulator, "enable_sparse_moments", True)
    for name in list(sys.modules):
        if name.startswith("pennylane_rocq"):
            sys.modules.pop(name)

    import pennylane as qml

    hamiltonian_matrix = _single_qubit_sparse_x(sp)
    observable = qml.SparseHamiltonian(hamiltonian_matrix, wires=[0])
    dev = qml.device("lightning.rocq", wires=1)

    @qml.qnode(dev)
    def sparse_circuit():
        return qml.expval(observable), qml.var(observable)

    assert sparse_circuit() == pytest.approx((0.0, 1.0))
    sim = _FakeQuantumSimulator.instances[-1]
    assert sim.statevector_reads == 0
    assert len(sim.sparse_moments) == 1
    data, indices, indptr, shape = sim.sparse_moments[0]
    np.testing.assert_allclose(data, np.array([1.0, 1.0], dtype=np.complex128))
    np.testing.assert_array_equal(indices, np.array([1, 0], dtype=np.int64))
    np.testing.assert_array_equal(indptr, np.array([0, 1, 2], dtype=np.int64))
    assert shape == (2, 2)


def test_pennylane_scaled_sparse_hamiltonian_uses_native_csr_moments(monkeypatch):
    pytest.importorskip("pennylane")
    sp = pytest.importorskip("scipy.sparse")
    _install_fake_binding(monkeypatch)
    monkeypatch.setattr(_FakeQuantumSimulator, "enable_sparse_moments", True)
    for name in list(sys.modules):
        if name.startswith("pennylane_rocq"):
            sys.modules.pop(name)

    import pennylane as qml

    hamiltonian_matrix = _single_qubit_sparse_x(sp)
    observable = qml.s_prod(2.0, qml.SparseHamiltonian(hamiltonian_matrix, wires=[0]))
    dev = qml.device("lightning.rocq", wires=1)

    @qml.qnode(dev)
    def sparse_circuit():
        return qml.expval(observable), qml.var(observable)

    assert sparse_circuit() == pytest.approx((0.0, 4.0))
    sim = _FakeQuantumSimulator.instances[-1]
    assert sim.statevector_reads == 0
    assert len(sim.sparse_moments) == 1
    data, indices, indptr, shape = sim.sparse_moments[0]
    np.testing.assert_allclose(data, np.array([1.0, 1.0], dtype=np.complex128))
    np.testing.assert_array_equal(indices, np.array([1, 0], dtype=np.int64))
    np.testing.assert_array_equal(indptr, np.array([0, 1, 2], dtype=np.int64))
    assert shape == (2, 2)


def test_pennylane_single_execute_reuses_scaled_sparse_expval_readout(monkeypatch):
    pytest.importorskip("pennylane")
    sp = pytest.importorskip("scipy.sparse")
    _install_fake_binding(monkeypatch)
    monkeypatch.setattr(_FakeQuantumSimulator, "enable_sparse_moments", True)
    for name in list(sys.modules):
        if name.startswith("pennylane_rocq"):
            sys.modules.pop(name)

    import pennylane as qml

    hamiltonian_matrix = sp.csr_matrix(np.array([[1.0, 1.0], [1.0, 1.0]], dtype=np.complex128))
    observable = qml.SparseHamiltonian(hamiltonian_matrix, wires=[0])
    scaled_observable = 2.0 * qml.SparseHamiltonian(hamiltonian_matrix.copy(), wires=[0])
    dev = qml.device("lightning.rocq", wires=1)

    @qml.qnode(dev)
    def sparse_circuit():
        return qml.expval(observable), qml.expval(scaled_observable)

    assert sparse_circuit() == pytest.approx((1.0, 2.0))
    sim = _FakeQuantumSimulator.instances[-1]
    assert sim.statevector_reads == 0
    assert len(sim.sparse_moments) == 1
    data, indices, indptr, shape = sim.sparse_moments[0]
    np.testing.assert_allclose(data, np.array([1.0, 1.0, 1.0, 1.0], dtype=np.complex128))
    np.testing.assert_array_equal(indices, np.array([0, 1, 0, 1], dtype=np.int64))
    np.testing.assert_array_equal(indptr, np.array([0, 2, 4], dtype=np.int64))
    assert shape == (2, 2)


def test_pennylane_mixed_sparse_sum_expval_uses_native_readouts(monkeypatch):
    pytest.importorskip("pennylane")
    sp = pytest.importorskip("scipy.sparse")
    _install_fake_binding(monkeypatch)
    monkeypatch.setattr(_FakeQuantumSimulator, "enable_sparse_moments", True)
    for name in list(sys.modules):
        if name.startswith("pennylane_rocq"):
            sys.modules.pop(name)

    import pennylane as qml

    hamiltonian_matrix = _single_qubit_sparse_x(sp)
    observable = qml.sum(
        qml.SparseHamiltonian(hamiltonian_matrix, wires=[0]),
        2.0 * qml.SparseHamiltonian(hamiltonian_matrix.copy(), wires=[0]),
        0.5 * qml.PauliZ(0),
    )
    dev = qml.device("lightning.rocq", wires=1)

    @qml.qnode(dev)
    def sparse_circuit():
        return qml.expval(observable)

    assert sparse_circuit() == pytest.approx(0.25)
    sim = _FakeQuantumSimulator.instances[-1]
    assert sim.expectations == [("Z", (0,))]
    assert sim.statevector_reads == 0
    assert len(sim.sparse_moments) == 1
    data, indices, indptr, shape = sim.sparse_moments[0]
    np.testing.assert_allclose(data, np.array([1.0, 1.0], dtype=np.complex128))
    np.testing.assert_array_equal(indices, np.array([1, 0], dtype=np.int64))
    np.testing.assert_array_equal(indptr, np.array([0, 1, 2], dtype=np.int64))
    assert shape == (2, 2)


def test_pennylane_mixed_sparse_sum_elides_merged_zero_readout(monkeypatch):
    pytest.importorskip("pennylane")
    sp = pytest.importorskip("scipy.sparse")
    _install_fake_binding(monkeypatch)
    monkeypatch.setattr(_FakeQuantumSimulator, "enable_sparse_moments", True)
    for name in list(sys.modules):
        if name.startswith("pennylane_rocq"):
            sys.modules.pop(name)

    import pennylane as qml

    hamiltonian_matrix = _single_qubit_sparse_x(sp)
    observable = qml.sum(
        qml.SparseHamiltonian(hamiltonian_matrix, wires=[0]),
        -1.0 * qml.SparseHamiltonian(hamiltonian_matrix.copy(), wires=[0]),
        0.5 * qml.PauliZ(0),
    )
    dev = qml.device("lightning.rocq", wires=1)

    @qml.qnode(dev)
    def sparse_circuit():
        return qml.expval(observable)

    assert sparse_circuit() == pytest.approx(0.25)
    sim = _FakeQuantumSimulator.instances[-1]
    assert sim.expectations == [("Z", (0,))]
    assert sim.sparse_moments == []
    assert sim.statevector_reads == 0


def test_pennylane_mixed_sparse_sum_variance_uses_native_csr_moments(monkeypatch):
    pytest.importorskip("pennylane")
    sp = pytest.importorskip("scipy.sparse")
    _install_fake_binding(monkeypatch)
    monkeypatch.setattr(_FakeQuantumSimulator, "enable_sparse_moments", True)
    for name in list(sys.modules):
        if name.startswith("pennylane_rocq"):
            sys.modules.pop(name)

    import pennylane as qml

    hamiltonian_matrix = _single_qubit_sparse_x(sp)
    sum_observable = qml.sum(qml.SparseHamiltonian(hamiltonian_matrix, wires=[0]), 0.5 * qml.PauliZ(0))
    hamiltonian_observable = qml.Hamiltonian(
        [1.0, 0.5],
        [qml.SparseHamiltonian(hamiltonian_matrix, wires=[0]), qml.PauliZ(0)],
    )
    dev = qml.device("lightning.rocq", wires=1)

    @qml.qnode(dev)
    def sum_circuit():
        return qml.var(sum_observable)

    @qml.qnode(dev)
    def hamiltonian_circuit():
        return qml.var(hamiltonian_observable)

    for circuit in (sum_circuit, hamiltonian_circuit):
        assert circuit() == pytest.approx(1.0)
        sim = _FakeQuantumSimulator.instances[-1]
        assert sim.expectations == []
        assert sim.statevector_reads == 0
        assert len(sim.sparse_moments) == 1
        data, indices, indptr, shape = sim.sparse_moments[0]
        np.testing.assert_allclose(data, np.array([1.0, 2.0, 2.0, -1.0], dtype=np.complex128))
        np.testing.assert_array_equal(indices, np.array([0, 1, 0, 1], dtype=np.int64))
        np.testing.assert_array_equal(indptr, np.array([0, 2, 4], dtype=np.int64))
        assert shape == (2, 2)


def test_pennylane_mixed_dense_sparse_sum_variance_uses_native_csr_moments(monkeypatch):
    pytest.importorskip("pennylane")
    sp = pytest.importorskip("scipy.sparse")
    _install_fake_binding(monkeypatch)
    monkeypatch.setattr(_FakeQuantumSimulator, "enable_matrix_expectation", True)
    monkeypatch.setattr(_FakeQuantumSimulator, "enable_sparse_moments", True)
    for name in list(sys.modules):
        if name.startswith("pennylane_rocq"):
            sys.modules.pop(name)

    import pennylane as qml

    sparse_matrix = _single_qubit_sparse_x(sp)
    dense_matrix = np.array([[0.0, 2.0], [2.0, 0.0]], dtype=np.complex128)
    observable = qml.sum(
        qml.Hermitian(dense_matrix, wires=0),
        qml.SparseHamiltonian(sparse_matrix, wires=[0]),
        0.5 * qml.PauliZ(0),
    )
    dev = qml.device("lightning.rocq", wires=1)

    @qml.qnode(dev)
    def circuit():
        return qml.var(observable)

    assert circuit() == pytest.approx(9.0)
    sim = _FakeQuantumSimulator.instances[-1]
    assert sim.expectations == []
    assert sim.matrix_expectations == []
    assert sim.statevector_reads == 0
    assert len(sim.sparse_moments) == 1
    data, indices, indptr, shape = sim.sparse_moments[0]
    np.testing.assert_allclose(data, np.array([1.0, 6.0, 6.0, -1.0], dtype=np.complex128))
    np.testing.assert_array_equal(indices, np.array([0, 1, 0, 1], dtype=np.int64))
    np.testing.assert_array_equal(indptr, np.array([0, 2, 4], dtype=np.int64))
    assert shape == (2, 2)


def test_pennylane_large_mixed_dense_sum_variance_falls_back_cleanly(monkeypatch):
    pytest.importorskip("pennylane")
    _install_fake_binding(monkeypatch)
    for name in list(sys.modules):
        if name.startswith("pennylane_rocq"):
            sys.modules.pop(name)

    import pennylane as qml

    matrix = np.fliplr(np.eye(32, dtype=np.complex128))
    observable = qml.sum(qml.Hermitian(matrix, wires=range(5)), 0.5 * qml.PauliZ(0))
    dev = qml.device("lightning.rocq", wires=5)

    @qml.qnode(dev)
    def circuit():
        return qml.var(observable)

    assert circuit() == pytest.approx(0.0)
    sim = _FakeQuantumSimulator.instances[-1]
    assert sim.expectations == []
    assert sim.matrix_expectations == []
    assert sim.sparse_moments == []
    assert sim.statevector_reads == 1


def test_pennylane_sparse_hamiltonian_batch_uses_native_csr_moments(monkeypatch):
    pytest.importorskip("pennylane")
    sp = pytest.importorskip("scipy.sparse")
    _install_fake_binding(monkeypatch)
    monkeypatch.setattr(_FakeQuantumSimulator, "enable_sparse_moments", True)
    for name in list(sys.modules):
        if name.startswith("pennylane_rocq"):
            sys.modules.pop(name)

    import pennylane as qml

    hamiltonian_matrix = _single_qubit_sparse_x(sp)
    observable = qml.SparseHamiltonian(hamiltonian_matrix, wires=[0])
    dev = qml.device("lightning.rocq", wires=1)
    tapes = [
        qml.tape.QuantumScript([qml.RY(0.1, wires=0)], [qml.expval(observable)]),
        qml.tape.QuantumScript([qml.RY(0.2, wires=0)], [qml.expval(observable)]),
    ]

    results = dev.batch_execute(tapes)

    assert results == pytest.approx((0.0, 0.0))
    sim = _FakeQuantumSimulator.instances[-1]
    assert sim.batch_size() == 2
    assert sim.batch_ops == [("RY", (0,), (0.1, 0.2))]
    assert len(sim.sparse_batch_moments) == 1
    assert sim.sparse_moments == []
    assert sim.statevector_reads == 0


def test_pennylane_diagonal_sparse_hamiltonian_batch_uses_native_pauli_terms(monkeypatch):
    pytest.importorskip("pennylane")
    sp = pytest.importorskip("scipy.sparse")
    _install_fake_binding(monkeypatch)
    for name in list(sys.modules):
        if name.startswith("pennylane_rocq"):
            sys.modules.pop(name)

    import pennylane as qml

    hamiltonian_matrix = sp.csr_matrix(np.diag([1.0, -1.0]).astype(np.complex128))
    observable = 2.0 * qml.SparseHamiltonian(hamiltonian_matrix, wires=[0])
    dev = qml.device("lightning.rocq", wires=1)
    tapes = [
        qml.tape.QuantumScript([qml.RY(0.1, wires=0)], [qml.expval(observable), qml.var(observable)]),
        qml.tape.QuantumScript([qml.RY(0.2, wires=0)], [qml.expval(observable), qml.var(observable)]),
    ]

    np.testing.assert_allclose(
        np.asarray(dev.batch_execute(tapes), dtype=float),
        np.array([[1.0, 3.0], [1.0, 3.0]], dtype=float),
    )
    sim = _FakeQuantumSimulator.instances[-1]
    assert sim.batch_size() == 2
    assert sim.batch_ops == [("RY", (0,), (0.1, 0.2))]
    assert sim.batch_expectations == [("Z", (0,))]
    assert sim.sparse_batch_moments == []
    assert sim.statevector_reads == 0


def test_pennylane_sparse_hamiltonian_identity_batch_folds_without_readout(monkeypatch):
    pytest.importorskip("pennylane")
    sp = pytest.importorskip("scipy.sparse")
    _install_fake_binding(monkeypatch)
    for name in list(sys.modules):
        if name.startswith("pennylane_rocq"):
            sys.modules.pop(name)

    import pennylane as qml

    identity = sp.eye(2, dtype=np.complex128, format="csr")
    observable = 2.0 * qml.SparseHamiltonian(identity, wires=[0])
    dev = qml.device("lightning.rocq", wires=1)
    tapes = [
        qml.tape.QuantumScript([qml.RY(0.1, wires=0)], [qml.expval(observable), qml.var(observable)]),
        qml.tape.QuantumScript([qml.RY(0.2, wires=0)], [qml.expval(observable), qml.var(observable)]),
    ]

    np.testing.assert_allclose(
        np.asarray(dev.batch_execute(tapes), dtype=float),
        np.array([[2.0, 0.0], [2.0, 0.0]], dtype=float),
    )
    sim = _FakeQuantumSimulator.instances[-1]
    assert sim.batch_size() == 2
    assert sim.batch_ops == [("RY", (0,), (0.1, 0.2))]
    assert sim.sparse_batch_moments == []
    assert sim.batch_expectations == []
    assert sim.statevector_reads == 0


def test_pennylane_scaled_sparse_hamiltonian_batch_uses_native_csr_moments(monkeypatch):
    pytest.importorskip("pennylane")
    sp = pytest.importorskip("scipy.sparse")
    _install_fake_binding(monkeypatch)
    monkeypatch.setattr(_FakeQuantumSimulator, "enable_sparse_moments", True)
    for name in list(sys.modules):
        if name.startswith("pennylane_rocq"):
            sys.modules.pop(name)

    import pennylane as qml

    hamiltonian_matrix = _single_qubit_sparse_x(sp)
    observable = 2.0 * qml.SparseHamiltonian(hamiltonian_matrix, wires=[0])
    dev = qml.device("lightning.rocq", wires=1)
    tapes = [
        qml.tape.QuantumScript([qml.RY(0.1, wires=0)], [qml.expval(observable)]),
        qml.tape.QuantumScript([qml.RY(0.2, wires=0)], [qml.expval(observable)]),
    ]

    results = dev.batch_execute(tapes)

    assert results == pytest.approx((0.0, 0.0))
    sim = _FakeQuantumSimulator.instances[-1]
    assert sim.batch_size() == 2
    assert sim.batch_ops == [("RY", (0,), (0.1, 0.2))]
    assert len(sim.sparse_batch_moments) == 1
    assert sim.sparse_moments == []
    assert sim.statevector_reads == 0
    data, indices, indptr, shape = sim.sparse_batch_moments[0]
    np.testing.assert_allclose(data, np.array([1.0, 1.0], dtype=np.complex128))
    np.testing.assert_array_equal(indices, np.array([1, 0], dtype=np.int64))
    np.testing.assert_array_equal(indptr, np.array([0, 1, 2], dtype=np.int64))
    assert shape == (2, 2)


def test_pennylane_batch_execute_reuses_scaled_sparse_expval_readout(monkeypatch):
    pytest.importorskip("pennylane")
    sp = pytest.importorskip("scipy.sparse")
    _install_fake_binding(monkeypatch)
    monkeypatch.setattr(_FakeQuantumSimulator, "enable_sparse_moments", True)
    for name in list(sys.modules):
        if name.startswith("pennylane_rocq"):
            sys.modules.pop(name)

    import pennylane as qml

    hamiltonian_matrix = sp.csr_matrix(np.array([[1.0, 1.0], [1.0, 1.0]], dtype=np.complex128))
    dev = qml.device("lightning.rocq", wires=1)
    tapes = [
        qml.tape.QuantumScript(
            [qml.RY(0.1, wires=0)],
            [
                qml.expval(qml.SparseHamiltonian(hamiltonian_matrix, wires=[0])),
                qml.expval(2.0 * qml.SparseHamiltonian(hamiltonian_matrix.copy(), wires=[0])),
            ],
        ),
        qml.tape.QuantumScript(
            [qml.RY(0.2, wires=0)],
            [
                qml.expval(qml.SparseHamiltonian(hamiltonian_matrix, wires=[0])),
                qml.expval(2.0 * qml.SparseHamiltonian(hamiltonian_matrix.copy(), wires=[0])),
            ],
        ),
    ]

    results = dev.batch_execute(tapes)

    assert len(results) == 2
    for result in results:
        np.testing.assert_allclose(result, (1.0, 2.0))
    sim = _FakeQuantumSimulator.instances[-1]
    assert sim.batch_size() == 2
    assert sim.batch_ops == [("RY", (0,), (0.1, 0.2))]
    assert len(sim.sparse_batch_moments) == 1
    assert sim.sparse_moments == []
    assert sim.statevector_reads == 0
    data, indices, indptr, shape = sim.sparse_batch_moments[0]
    np.testing.assert_allclose(data, np.array([1.0, 1.0, 1.0, 1.0], dtype=np.complex128))
    np.testing.assert_array_equal(indices, np.array([0, 1, 0, 1], dtype=np.int64))
    np.testing.assert_array_equal(indptr, np.array([0, 2, 4], dtype=np.int64))
    assert shape == (2, 2)


def test_pennylane_mixed_sparse_sum_batch_uses_native_readouts(monkeypatch):
    pytest.importorskip("pennylane")
    sp = pytest.importorskip("scipy.sparse")
    _install_fake_binding(monkeypatch)
    monkeypatch.setattr(_FakeQuantumSimulator, "enable_sparse_moments", True)
    for name in list(sys.modules):
        if name.startswith("pennylane_rocq"):
            sys.modules.pop(name)

    import pennylane as qml

    hamiltonian_matrix = _single_qubit_sparse_x(sp)
    observable = qml.sum(
        qml.SparseHamiltonian(hamiltonian_matrix, wires=[0]),
        2.0 * qml.SparseHamiltonian(hamiltonian_matrix.copy(), wires=[0]),
        0.5 * qml.PauliZ(0),
    )
    dev = qml.device("lightning.rocq", wires=1)
    tapes = [
        qml.tape.QuantumScript([qml.RY(0.1, wires=0)], [qml.expval(observable)]),
        qml.tape.QuantumScript([qml.RY(0.2, wires=0)], [qml.expval(observable)]),
    ]

    results = dev.batch_execute(tapes)

    assert results == pytest.approx((0.25, 0.25))
    sim = _FakeQuantumSimulator.instances[-1]
    assert sim.batch_size() == 2
    assert sim.batch_ops == [("RY", (0,), (0.1, 0.2))]
    assert sim.batch_expectations == [("Z", (0,))]
    assert len(sim.sparse_batch_moments) == 1
    assert sim.sparse_moments == []
    assert sim.statevector_reads == 0
    data, indices, indptr, shape = sim.sparse_batch_moments[0]
    np.testing.assert_allclose(data, np.array([1.0, 1.0], dtype=np.complex128))
    np.testing.assert_array_equal(indices, np.array([1, 0], dtype=np.int64))
    np.testing.assert_array_equal(indptr, np.array([0, 1, 2], dtype=np.int64))
    assert shape == (2, 2)


def test_pennylane_batch_execute_reuses_sum_component_sparse_readouts(monkeypatch):
    pytest.importorskip("pennylane")
    sp = pytest.importorskip("scipy.sparse")
    _install_fake_binding(monkeypatch)
    monkeypatch.setattr(_FakeQuantumSimulator, "enable_sparse_moments", True)
    for name in list(sys.modules):
        if name.startswith("pennylane_rocq"):
            sys.modules.pop(name)

    import pennylane as qml

    hamiltonian_matrix = sp.csr_matrix(np.array([[1.0, 1.0], [1.0, 1.0]], dtype=np.complex128))
    observable = qml.sum(qml.PauliZ(0), qml.SparseHamiltonian(hamiltonian_matrix, wires=[0]))
    dev = qml.device("lightning.rocq", wires=1)
    tapes = [
        qml.tape.QuantumScript(
            [qml.RY(0.1, wires=0)],
            [qml.expval(qml.PauliZ(0)), qml.expval(observable)],
        ),
        qml.tape.QuantumScript(
            [qml.RY(0.2, wires=0)],
            [qml.expval(qml.PauliZ(0)), qml.expval(observable)],
        ),
    ]

    results = dev.batch_execute(tapes)

    assert len(results) == 2
    for result in results:
        np.testing.assert_allclose(result, (0.5, 1.5))
    sim = _FakeQuantumSimulator.instances[-1]
    assert sim.batch_size() == 2
    assert sim.batch_ops == [("RY", (0,), (0.1, 0.2))]
    assert sim.batch_expectations == [("Z", (0,))]
    assert len(sim.sparse_batch_moments) == 1
    assert sim.sparse_moments == []
    assert sim.statevector_reads == 0
    data, indices, indptr, shape = sim.sparse_batch_moments[0]
    np.testing.assert_allclose(data, np.array([1.0, 1.0, 1.0, 1.0], dtype=np.complex128))
    np.testing.assert_array_equal(indices, np.array([0, 1, 0, 1], dtype=np.int64))
    np.testing.assert_array_equal(indptr, np.array([0, 2, 4], dtype=np.int64))
    assert shape == (2, 2)


def test_pennylane_mixed_sparse_sum_batch_variance_uses_native_csr_moments(monkeypatch):
    pytest.importorskip("pennylane")
    sp = pytest.importorskip("scipy.sparse")
    _install_fake_binding(monkeypatch)
    monkeypatch.setattr(_FakeQuantumSimulator, "enable_sparse_moments", True)
    for name in list(sys.modules):
        if name.startswith("pennylane_rocq"):
            sys.modules.pop(name)

    import pennylane as qml

    hamiltonian_matrix = _single_qubit_sparse_x(sp)
    observable = qml.sum(qml.SparseHamiltonian(hamiltonian_matrix, wires=[0]), 0.5 * qml.PauliZ(0))
    dev = qml.device("lightning.rocq", wires=1)
    tapes = [
        qml.tape.QuantumScript([qml.RY(0.1, wires=0)], [qml.var(observable)]),
        qml.tape.QuantumScript([qml.RY(0.2, wires=0)], [qml.var(observable)]),
    ]

    results = dev.batch_execute(tapes)

    assert results == pytest.approx((1.0, 1.0))
    sim = _FakeQuantumSimulator.instances[-1]
    assert sim.batch_size() == 2
    assert sim.batch_ops == [("RY", (0,), (0.1, 0.2))]
    assert sim.batch_expectations == []
    assert len(sim.sparse_batch_moments) == 1
    assert sim.sparse_moments == []
    assert sim.statevector_reads == 0
    data, indices, indptr, shape = sim.sparse_batch_moments[0]
    np.testing.assert_allclose(data, np.array([1.0, 2.0, 2.0, -1.0], dtype=np.complex128))
    np.testing.assert_array_equal(indices, np.array([0, 1, 0, 1], dtype=np.int64))
    np.testing.assert_array_equal(indptr, np.array([0, 2, 4], dtype=np.int64))
    assert shape == (2, 2)


def test_pennylane_mixed_dense_sparse_sum_batch_variance_uses_native_csr_moments(monkeypatch):
    pytest.importorskip("pennylane")
    sp = pytest.importorskip("scipy.sparse")
    _install_fake_binding(monkeypatch)
    monkeypatch.setattr(_FakeQuantumSimulator, "enable_matrix_expectation", True)
    monkeypatch.setattr(_FakeQuantumSimulator, "enable_sparse_moments", True)
    for name in list(sys.modules):
        if name.startswith("pennylane_rocq"):
            sys.modules.pop(name)

    import pennylane as qml

    sparse_matrix = _single_qubit_sparse_x(sp)
    dense_matrix = np.array([[0.0, 2.0], [2.0, 0.0]], dtype=np.complex128)
    observable = qml.sum(
        qml.Hermitian(dense_matrix, wires=0),
        qml.SparseHamiltonian(sparse_matrix, wires=[0]),
        0.5 * qml.PauliZ(0),
    )
    dev = qml.device("lightning.rocq", wires=1)
    tapes = [
        qml.tape.QuantumScript([qml.RY(0.1, wires=0)], [qml.var(observable)]),
        qml.tape.QuantumScript([qml.RY(0.2, wires=0)], [qml.var(observable)]),
    ]

    results = dev.batch_execute(tapes)

    assert results == pytest.approx((9.0, 9.0))
    sim = _FakeQuantumSimulator.instances[-1]
    assert sim.batch_size() == 2
    assert sim.batch_ops == [("RY", (0,), (0.1, 0.2))]
    assert sim.batch_expectations == []
    assert sim.matrix_batch_expectations == []
    assert sim.sparse_moments == []
    assert sim.statevector_reads == 0
    assert len(sim.sparse_batch_moments) == 1
    data, indices, indptr, shape = sim.sparse_batch_moments[0]
    np.testing.assert_allclose(data, np.array([1.0, 6.0, 6.0, -1.0], dtype=np.complex128))
    np.testing.assert_array_equal(indices, np.array([0, 1, 0, 1], dtype=np.int64))
    np.testing.assert_array_equal(indptr, np.array([0, 2, 4], dtype=np.int64))
    assert shape == (2, 2)


def test_pennylane_sparse_hamiltonian_batch_variance_uses_native_csr_moments(monkeypatch):
    pytest.importorskip("pennylane")
    sp = pytest.importorskip("scipy.sparse")
    _install_fake_binding(monkeypatch)
    monkeypatch.setattr(_FakeQuantumSimulator, "enable_sparse_moments", True)
    for name in list(sys.modules):
        if name.startswith("pennylane_rocq"):
            sys.modules.pop(name)

    import pennylane as qml

    hamiltonian_matrix = _single_qubit_sparse_x(sp)
    observable = qml.SparseHamiltonian(hamiltonian_matrix, wires=[0])
    dev = qml.device("lightning.rocq", wires=1)
    tapes = [
        qml.tape.QuantumScript([qml.RY(0.1, wires=0)], [qml.var(observable)]),
        qml.tape.QuantumScript([qml.RY(0.2, wires=0)], [qml.var(observable)]),
    ]

    results = dev.batch_execute(tapes)

    assert results == pytest.approx((1.0, 1.0))
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


def test_pennylane_controlled_global_phase_wrapper_decomposes_natively(monkeypatch):
    pytest.importorskip("pennylane")
    _install_fake_binding(monkeypatch)
    for name in list(sys.modules):
        if name.startswith("pennylane_rocq"):
            sys.modules.pop(name)

    import pennylane as qml

    dev = qml.device("lightning.rocq", wires=3)

    @qml.qnode(dev)
    def circuit():
        qml.ctrl(qml.GlobalPhase(0.4, wires=[]), control=[0, 1], control_values=[True, False])
        return qml.expval(qml.PauliZ(0))

    assert circuit() == pytest.approx(0.5)
    sim = _FakeQuantumSimulator.instances[-1]

    assert sim.ops[0] == ("X", (1,), ())
    assert sim.ops[-1] == ("X", (1,), ())
    assert any(name == "RZ" for name, _, _ in sim.ops)
    assert sim.matrices == []
    assert sim.controlled_matrices == []
    assert sim.expectations == [("Z", (0,))]


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


def test_pennylane_controlled_sequence_adjoint_phase_roots_decompose_natively(monkeypatch):
    pytest.importorskip("pennylane")
    _install_fake_binding(monkeypatch)
    for name in list(sys.modules):
        if name.startswith("pennylane_rocq"):
            sys.modules.pop(name)

    import pennylane as qml

    dev = qml.device("lightning.rocq", wires=3)

    @qml.qnode(dev)
    def circuit():
        qml.ControlledSequence(qml.adjoint(qml.S(wires=2)), control=[0, 1])
        qml.ControlledSequence(qml.adjoint(qml.T(wires=2)), control=[0, 1])
        return qml.state()

    circuit()
    sim = _FakeQuantumSimulator.instances[-1]

    assert sim.ops == [
        ("CP", (0, 2), (-np.pi,)),
        ("CP", (1, 2), (-np.pi / 2,)),
        ("CP", (0, 2), (-np.pi / 2,)),
        ("CP", (1, 2), (-np.pi / 4,)),
    ]
    assert sim.matrices == []
    assert sim.controlled_matrices == []


def test_pennylane_multi_controlled_single_qubit_wrappers_decompose_natively(monkeypatch):
    pytest.importorskip("pennylane")
    _install_fake_binding(monkeypatch)
    for name in list(sys.modules):
        if name.startswith("pennylane_rocq"):
            sys.modules.pop(name)

    import pennylane as qml

    dev = qml.device("lightning.rocq", wires=3)

    @qml.qnode(dev)
    def circuit():
        qml.ctrl(qml.PauliY(wires=2), control=[0, 1])
        qml.ctrl(qml.PauliZ(wires=2), control=[0, 1], control_values=[True, False])
        qml.ctrl(qml.Hadamard(wires=2), control=[0, 1])
        qml.ctrl(qml.S(wires=2), control=[0, 1])
        qml.ctrl(qml.T(wires=2), control=[0, 1], control_values=[False, True])
        return qml.state()

    circuit()
    sim = _FakeQuantumSimulator.instances[-1]

    assert sim.ops[:11] == [
        ("SDG", (2,), ()),
        ("MCX", (0, 1, 2), ()),
        ("S", (2,), ()),
        ("X", (1,), ()),
        ("H", (2,), ()),
        ("MCX", (0, 1, 2), ()),
        ("H", (2,), ()),
        ("X", (1,), ()),
        ("RY", (2,), (np.pi / 4,)),
        ("MCX", (0, 1, 2), ()),
        ("RY", (2,), (-np.pi / 4,)),
    ]
    assert any(name == "RZ" and targets == (0,) and params == (np.pi / 8,) for name, targets, params in sim.ops)
    assert any(name == "RZ" and targets == (0,) and params == (np.pi / 16,) for name, targets, params in sim.ops)
    assert any(name in {"CNOT", "CX"} for name, _, _ in sim.ops[11:])
    assert sim.ops[-1] == ("X", (0,), ())
    assert len(sim.matrices) == 2
    for matrix, targets in sim.matrices:
        np.testing.assert_allclose(matrix, np.eye(2) * matrix[0, 0])
        assert targets == (0,)
    assert sim.controlled_matrices == []


def test_pennylane_multi_controlled_swap_wrapper_decomposes_natively(monkeypatch):
    pytest.importorskip("pennylane")
    _install_fake_binding(monkeypatch)
    for name in list(sys.modules):
        if name.startswith("pennylane_rocq"):
            sys.modules.pop(name)

    import pennylane as qml

    dev = qml.device("lightning.rocq", wires=4)

    @qml.qnode(dev)
    def circuit():
        qml.ctrl(qml.SWAP(wires=[2, 3]), control=[0, 1], control_values=[True, False])
        return qml.state()

    circuit()
    sim = _FakeQuantumSimulator.instances[-1]

    assert sim.ops == [
        ("X", (1,), ()),
        ("MCX", (0, 1, 2, 3), ()),
        ("MCX", (0, 1, 3, 2), ()),
        ("MCX", (0, 1, 2, 3), ()),
        ("X", (1,), ()),
    ]
    assert sim.matrices == []
    assert sim.controlled_matrices == []


def test_pennylane_multi_controlled_iswap_wrapper_decomposes_natively(monkeypatch):
    pytest.importorskip("pennylane")
    _install_fake_binding(monkeypatch)
    for name in list(sys.modules):
        if name.startswith("pennylane_rocq"):
            sys.modules.pop(name)

    import pennylane as qml

    dev = qml.device("lightning.rocq", wires=4)

    @qml.qnode(dev)
    def circuit():
        qml.ctrl(qml.ISWAP(wires=[2, 3]), control=[0, 1], control_values=[True, False])
        return qml.expval(qml.PauliZ(0))

    assert circuit() == pytest.approx(0.5)
    sim = _FakeQuantumSimulator.instances[-1]

    assert sim.ops[0] == ("X", (1,), ())
    assert sim.ops[-1] == ("X", (1,), ())
    assert ("MCX", (0, 1, 2, 3), ()) in sim.ops
    assert ("MCX", (0, 1, 3, 2), ()) in sim.ops
    assert any(op == ("RY", (2,), (np.pi / 4,)) for op in sim.ops)
    assert any(op == ("RY", (3,), (np.pi / 4,)) for op in sim.ops)
    assert any(name == "RZ" for name, _, _ in sim.ops)
    assert sim.matrices == []
    assert sim.controlled_matrices == []
    assert sim.expectations == [("Z", (0,))]


def test_pennylane_multi_controlled_pswap_wrapper_decomposes_natively(monkeypatch):
    pytest.importorskip("pennylane")
    _install_fake_binding(monkeypatch)
    for name in list(sys.modules):
        if name.startswith("pennylane_rocq"):
            sys.modules.pop(name)

    import pennylane as qml

    dev = qml.device("lightning.rocq", wires=4)

    @qml.qnode(dev)
    def circuit():
        qml.ctrl(qml.PSWAP(0.45, wires=[2, 3]), control=[0, 1], control_values=[True, False])
        return qml.expval(qml.PauliZ(0))

    assert circuit() == pytest.approx(0.5)
    sim = _FakeQuantumSimulator.instances[-1]

    assert sim.ops[0] == ("X", (1,), ())
    assert sim.ops[-1] == ("X", (1,), ())
    assert ("MCX", (0, 1, 2, 3), ()) in sim.ops
    assert ("MCX", (0, 1, 3, 2), ()) in sim.ops
    assert any(name == "RZ" and wires == (3,) for name, wires, _ in sim.ops)
    assert sim.matrices == []
    assert sim.controlled_matrices == []
    assert sim.expectations == [("Z", (0,))]


def test_pennylane_multi_controlled_siswap_wrappers_decompose_natively(monkeypatch):
    pytest.importorskip("pennylane")
    _install_fake_binding(monkeypatch)
    for name in list(sys.modules):
        if name.startswith("pennylane_rocq"):
            sys.modules.pop(name)

    import pennylane as qml

    dev = qml.device("lightning.rocq", wires=4)

    @qml.qnode(dev)
    def circuit():
        qml.ctrl(qml.SISWAP(wires=[2, 3]), control=[0, 1], control_values=[True, False])
        qml.ctrl(qml.SQISW(wires=[2, 3]), control=[0, 1], control_values=[True, False])
        return qml.expval(qml.PauliZ(0))

    assert circuit() == pytest.approx(0.5)
    sim = _FakeQuantumSimulator.instances[-1]

    assert sim.ops[0] == ("X", (1,), ())
    assert sim.ops[-1] == ("X", (1,), ())
    assert ("MCX", (0, 1, 2, 3), ()) in sim.ops
    assert any(name == "RZ" for name, _, _ in sim.ops)
    assert sim.matrices == []
    assert sim.controlled_matrices == []
    assert sim.expectations == [("Z", (0,))]


def test_pennylane_multi_controlled_ecr_wrapper_decomposes_natively(monkeypatch):
    pytest.importorskip("pennylane")
    _install_fake_binding(monkeypatch)
    for name in list(sys.modules):
        if name.startswith("pennylane_rocq"):
            sys.modules.pop(name)

    import pennylane as qml

    dev = qml.device("lightning.rocq", wires=4)

    @qml.qnode(dev)
    def circuit():
        qml.ctrl(qml.ECR(wires=[2, 3]), control=[0, 1], control_values=[True, False])
        return qml.expval(qml.PauliZ(0))

    assert circuit() == pytest.approx(0.5)
    sim = _FakeQuantumSimulator.instances[-1]

    assert sim.ops[0] == ("X", (1,), ())
    assert sim.ops[-1] == ("X", (1,), ())
    assert ("MCX", (0, 1, 2, 3), ()) in sim.ops
    assert any(name == "RZ" for name, _, _ in sim.ops)
    assert sim.matrices == []
    assert sim.controlled_matrices == []
    assert sim.expectations == [("Z", (0,))]


def test_pennylane_multi_controlled_parametric_wrappers_decompose_natively(monkeypatch):
    pytest.importorskip("pennylane")
    _install_fake_binding(monkeypatch)
    for name in list(sys.modules):
        if name.startswith("pennylane_rocq"):
            sys.modules.pop(name)

    import pennylane as qml

    dev = qml.device("lightning.rocq", wires=3)

    @qml.qnode(dev)
    def circuit():
        qml.ctrl(qml.RX(0.31, wires=2), control=[0, 1])
        qml.ctrl(qml.RY(0.32, wires=2), control=[0, 1], control_values=[True, False])
        qml.ctrl(qml.RZ(0.33, wires=2), control=[0, 1])
        qml.ctrl(qml.PhaseShift(0.34, wires=2), control=[0, 1])
        qml.ctrl(qml.Rot(0.1, 0.2, 0.3, wires=2), control=[0, 1])
        return qml.expval(qml.PauliZ(0))

    assert circuit() == pytest.approx(0.5)
    sim = _FakeQuantumSimulator.instances[-1]
    assert sim.matrices == []
    assert sim.controlled_matrices == []
    assert sim.expectations == [("Z", (0,))]
    assert any(op == ("X", (1,), ()) for op in sim.ops)
    assert any(name == "H" and wires == (2,) for name, wires, _ in sim.ops)
    assert any(name == "SDG" and wires == (2,) for name, wires, _ in sim.ops)
    assert any(name == "RZ" and wires == (2,) for name, wires, _ in sim.ops)


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


def test_pennylane_select_product_basis_native_decomposes_natively(monkeypatch):
    pytest.importorskip("pennylane")
    _install_fake_binding(monkeypatch)
    for name in list(sys.modules):
        if name.startswith("pennylane_rocq"):
            sys.modules.pop(name)

    import pennylane as qml

    selected_product = qml.prod(
        qml.BasisEmbedding(np.array([1, 0]), wires=[1, 2]),
        qml.PauliX(wires=2),
    )
    dev = qml.device("lightning.rocq", wires=3)

    @qml.qnode(dev)
    def circuit():
        qml.Select([selected_product, qml.Identity(wires=1)], control=[0])
        return qml.state()

    circuit()
    sim = _FakeQuantumSimulator.instances[-1]

    assert sim.ops == [
        ("X", (0,), ()),
        ("CNOT", (0, 1), ()),
        ("X", (0,), ()),
        ("X", (0,), ()),
        ("CNOT", (0, 2), ()),
        ("X", (0,), ()),
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


def test_pennylane_select_global_phase_decomposes_natively(monkeypatch):
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

    assert any(name == "RZ" for name, _, _ in sim.ops)
    assert any(name in {"CNOT", "CX"} for name, _, _ in sim.ops)
    assert len(sim.matrices) == 1
    matrix, targets = sim.matrices[0]
    np.testing.assert_allclose(matrix, np.eye(2) * matrix[0, 0])
    assert targets == (0,)
    assert sim.controlled_matrices == []


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


def test_pennylane_device_jacobian_batches_parameter_shift_tapes(monkeypatch):
    pytest.importorskip("pennylane")

    class _RYBatchSimulator(_FakeQuantumSimulator):
        def __init__(self, num_qubits, batch_size=1):
            super().__init__(num_qubits, batch_size=batch_size)
            self._states = np.zeros((self._batch_size, 1 << self._num_qubits), dtype=np.complex128)
            self._states[:, 0] = 1.0

        def apply_gate(self, name, targets, params=None):
            super().apply_gate(name, targets, params)
            if name == "RY" and tuple(targets) == (0,):
                self._apply_ry_to_batch([float((params or ())[0])])

        def apply_gate_batch(self, name, targets, params_by_batch):
            super().apply_gate_batch(name, targets, params_by_batch)
            if name == "RY" and tuple(targets) == (0,):
                self._apply_ry_to_batch([float(theta) for theta in params_by_batch])

        def _apply_ry_to_batch(self, angles):
            if len(angles) == 1:
                angles = angles * self._batch_size
            for batch_index, theta in enumerate(angles):
                c = math.cos(theta / 2.0)
                s = math.sin(theta / 2.0)
                old_state = self._states[batch_index].copy()
                self._states[batch_index, 0] = c * old_state[0] - s * old_state[1]
                self._states[batch_index, 1] = s * old_state[0] + c * old_state[1]

        def expectation_pauli_string(self, pauli_string, targets):
            self.expectations.append((pauli_string, tuple(targets)))
            if pauli_string == "Z" and tuple(targets) == (0,):
                state = self._states[0]
                return float(abs(state[0]) ** 2 - abs(state[1]) ** 2)
            return super().expectation_pauli_string(pauli_string, targets)

        def expectation_pauli_string_batch(self, pauli_string, targets):
            self.batch_expectations.append((pauli_string, tuple(targets)))
            if pauli_string == "Z" and tuple(targets) == (0,):
                return np.asarray(
                    [abs(state[0]) ** 2 - abs(state[1]) ** 2 for state in self._states],
                    dtype=float,
                )
            return super().expectation_pauli_string_batch(pauli_string, targets)

    fake = _install_fake_binding(monkeypatch)
    fake.QuantumSimulator = _RYBatchSimulator
    fake.QSim = _RYBatchSimulator
    for name in list(sys.modules):
        if name.startswith("pennylane_rocq"):
            sys.modules.pop(name)

    import pennylane as qml
    from pennylane import numpy as pnp
    from pennylane.devices import ExecutionConfig

    dev = qml.device("lightning.rocq", wires=1)
    assert dev.supports_derivatives(ExecutionConfig(gradient_method="device"))

    @qml.qnode(dev, diff_method="device")
    def circuit(theta):
        qml.RY(theta, wires=0)
        return qml.expval(qml.PauliZ(0))

    theta = pnp.array(0.321, requires_grad=True)

    assert circuit(theta) == pytest.approx(math.cos(float(theta)))
    assert qml.grad(circuit)(theta) == pytest.approx(-math.sin(float(theta)))

    batched_sims = [sim for sim in _RYBatchSimulator.instances if sim.batch_size() == 2]
    assert batched_sims
    sim = batched_sims[-1]
    assert len(sim.batch_ops) == 1
    assert sim.batch_ops[0][:2] == ("RY", (0,))
    np.testing.assert_allclose(sim.batch_ops[0][2], (float(theta) + np.pi / 2, float(theta) - np.pi / 2))
    assert sim.batch_expectations == [("Z", (0,))]
    assert sim.statevector_reads == 0

    def _ry_expval_tape(angle):
        tape = qml.tape.QuantumScript(
            [qml.RY(angle, wires=0)],
            [qml.expval(qml.PauliZ(0))],
        )
        tape.trainable_params = [0]
        return tape

    angles = (0.111, 0.456)
    tapes = tuple(_ry_expval_tape(angle) for angle in angles)

    before = len(_RYBatchSimulator.instances)
    jacs = dev.compute_derivatives(tapes, ExecutionConfig(gradient_method="device"))

    np.testing.assert_allclose(np.asarray(jacs, dtype=float), [-math.sin(angle) for angle in angles])
    derivative_batches = [
        sim for sim in _RYBatchSimulator.instances[before:] if sim.batch_size() == 4
    ]
    assert derivative_batches
    np.testing.assert_allclose(
        derivative_batches[-1].batch_ops[0][2],
        (
            angles[0] + np.pi / 2,
            angles[0] - np.pi / 2,
            angles[1] + np.pi / 2,
            angles[1] - np.pi / 2,
        ),
    )
    assert derivative_batches[-1].batch_expectations == [("Z", (0,))]
    assert derivative_batches[-1].statevector_reads == 0

    before = len(_RYBatchSimulator.instances)
    results, jacs = dev.execute_and_compute_derivatives(
        tapes,
        ExecutionConfig(gradient_method="device"),
    )

    np.testing.assert_allclose(np.asarray(results, dtype=float), [math.cos(angle) for angle in angles])
    np.testing.assert_allclose(np.asarray(jacs, dtype=float), [-math.sin(angle) for angle in angles])
    execute_batches = [
        sim for sim in _RYBatchSimulator.instances[before:] if sim.batch_size() == 2
    ]
    derivative_batches = [
        sim for sim in _RYBatchSimulator.instances[before:] if sim.batch_size() == 4
    ]
    assert execute_batches
    assert derivative_batches
    np.testing.assert_allclose(execute_batches[-1].batch_ops[0][2], angles)
    np.testing.assert_allclose(
        derivative_batches[-1].batch_ops[0][2],
        (
            angles[0] + np.pi / 2,
            angles[0] - np.pi / 2,
            angles[1] + np.pi / 2,
            angles[1] - np.pi / 2,
        ),
    )


def test_pennylane_device_jacobian_batches_probability_shift_tapes(monkeypatch):
    pytest.importorskip("pennylane")

    class _RYProbabilityBatchSimulator(_FakeQuantumSimulator):
        def __init__(self, num_qubits, batch_size=1):
            super().__init__(num_qubits, batch_size=batch_size)
            self._reset_states()

        def reset(self):
            super().reset()
            self._reset_states()

        def _reset_states(self):
            self._states = np.zeros((self._batch_size, 1 << self._num_qubits), dtype=np.complex128)
            self._states[:, 0] = 1.0

        def apply_gate(self, name, targets, params=None):
            super().apply_gate(name, targets, params)
            if name == "RY" and tuple(targets) == (0,):
                self._apply_ry_to_batch([float((params or ())[0])])

        def apply_gate_batch(self, name, targets, params_by_batch):
            super().apply_gate_batch(name, targets, params_by_batch)
            if name == "RY" and tuple(targets) == (0,):
                self._apply_ry_to_batch([float(theta) for theta in params_by_batch])

        def _apply_ry_to_batch(self, angles):
            if len(angles) == 1:
                angles = angles * self._batch_size
            for batch_index, theta in enumerate(angles):
                c = math.cos(theta / 2.0)
                s = math.sin(theta / 2.0)
                old_state = self._states[batch_index].copy()
                self._states[batch_index, 0] = c * old_state[0] - s * old_state[1]
                self._states[batch_index, 1] = s * old_state[0] + c * old_state[1]

        def probabilities(self, qubits):
            qubits = tuple(int(qubit) for qubit in qubits)
            self.probability_requests.append(qubits)
            return self._probabilities_for_state(self._states[0], qubits)

        def probabilities_batch(self, qubits):
            qubits = tuple(int(qubit) for qubit in qubits)
            self.probability_requests.append(qubits)
            return np.asarray(
                [self._probabilities_for_state(state, qubits) for state in self._states],
                dtype=float,
            )

        def _probabilities_for_state(self, state, qubits):
            qubits = tuple(int(qubit) for qubit in qubits)
            probabilities = np.zeros(1 << len(qubits), dtype=float)
            for basis_index, amplitude in enumerate(state):
                outcome = 0
                for output_bit, qubit in enumerate(qubits):
                    if (basis_index >> qubit) & 1:
                        outcome |= 1 << output_bit
                probabilities[outcome] += float(abs(amplitude) ** 2)
            return probabilities

    fake = _install_fake_binding(monkeypatch)
    fake.QuantumSimulator = _RYProbabilityBatchSimulator
    fake.QSim = _RYProbabilityBatchSimulator
    for name in list(sys.modules):
        if name.startswith("pennylane_rocq"):
            sys.modules.pop(name)

    import pennylane as qml
    from pennylane import numpy as pnp
    from pennylane.devices import ExecutionConfig

    dev = qml.device("lightning.rocq", wires=1)
    assert dev.supports_derivatives(ExecutionConfig(gradient_method="device"))

    @qml.qnode(dev, diff_method="device")
    def circuit(theta):
        qml.RY(theta, wires=0)
        return qml.probs(wires=[0])

    theta = pnp.array(0.321, requires_grad=True)
    expected_probabilities = np.asarray(
        [math.cos(float(theta) / 2.0) ** 2, math.sin(float(theta) / 2.0) ** 2],
        dtype=float,
    )

    np.testing.assert_allclose(np.asarray(circuit(theta), dtype=float), expected_probabilities)

    before = len(_RYProbabilityBatchSimulator.instances)
    jacobian = qml.jacobian(circuit)(theta)

    np.testing.assert_allclose(
        np.asarray(jacobian, dtype=float).reshape(-1),
        [-0.5 * math.sin(float(theta)), 0.5 * math.sin(float(theta))],
    )
    batched_sims = [
        sim
        for sim in _RYProbabilityBatchSimulator.instances[before:]
        if sim.batch_size() == 2
    ]
    assert batched_sims
    sim = batched_sims[-1]
    assert len(sim.batch_ops) == 1
    assert sim.batch_ops[0][:2] == ("RY", (0,))
    np.testing.assert_allclose(sim.batch_ops[0][2], (float(theta) + np.pi / 2, float(theta) - np.pi / 2))
    assert sim.probability_requests == [(0,)]
    assert sim.statevector_reads == 0


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
    assert dev.supports_derivatives(ExecutionConfig(gradient_method="device"))

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

    class _NativeAdjointSimulator(_RYPreservingSimulator):
        def __init__(self, num_qubits):
            super().__init__(num_qubits)
            self.native_adjoint_calls = []

        def adjoint_jacobian(self, operations, observables, trainable_params):
            self.native_adjoint_calls.append(
                {
                    "operations": operations,
                    "observables": observables,
                    "trainable_params": trainable_params,
                }
            )
            theta = float(operations[0]["params"][0])
            return np.asarray([[-math.sin(theta)]], dtype=float)

    fake = _install_fake_binding(monkeypatch)
    fake.QuantumSimulator = _NativeAdjointSimulator
    fake.QSim = _NativeAdjointSimulator
    for name in list(sys.modules):
        if name.startswith("pennylane_rocq"):
            sys.modules.pop(name)

    native_dev = qml.device("lightning.rocq", wires=1)

    @qml.qnode(native_dev, diff_method="adjoint")
    def native_circuit(theta):
        qml.RY(theta, wires=0)
        return qml.expval(qml.PauliZ(0))

    assert native_circuit(theta) == pytest.approx(math.cos(float(theta)))
    assert qml.grad(native_circuit)(theta) == pytest.approx(-math.sin(float(theta)))
    native_sim = _NativeAdjointSimulator.instances[-1]
    assert native_sim.statevector_reads == 0
    assert len(native_sim.native_adjoint_calls) == 1
    native_call = native_sim.native_adjoint_calls[0]
    assert native_call["operations"] == [
        {
            "name": "RY",
            "rocq_name": "RY",
            "wires": [0],
            "params": [float(theta)],
            "param_indices": [0],
            "trainable_param_indices": [0],
            "trainable_param_positions": [0],
        }
    ]
    assert native_call["observables"] == [
        [
            {
                "coefficient": (1.0, 0.0),
                "pauli_string": "Z",
                "targets": [0],
            }
        ]
    ]
    assert native_call["trainable_params"] == [0]

    class _RejectingNativeAdjointSimulator(_RYPreservingSimulator):
        def __init__(self, num_qubits):
            super().__init__(num_qubits)
            self.native_adjoint_calls = 0

        def adjoint_jacobian(self, operations, observables, trainable_params):
            self.native_adjoint_calls += 1
            raise NotImplementedError("unsupported native adjoint payload")

    fake = _install_fake_binding(monkeypatch)
    fake.QuantumSimulator = _RejectingNativeAdjointSimulator
    fake.QSim = _RejectingNativeAdjointSimulator
    for name in list(sys.modules):
        if name.startswith("pennylane_rocq"):
            sys.modules.pop(name)

    fallback_dev = qml.device("lightning.rocq", wires=1)

    @qml.qnode(fallback_dev, diff_method="adjoint")
    def fallback_circuit(theta):
        qml.RY(theta, wires=0)
        return qml.expval(qml.PauliZ(0))

    assert fallback_circuit(theta) == pytest.approx(math.cos(float(theta)))
    assert qml.grad(fallback_circuit)(theta) == pytest.approx(-math.sin(float(theta)))
    fallback_sims = [
        sim for sim in _RejectingNativeAdjointSimulator.instances
        if isinstance(sim, _RejectingNativeAdjointSimulator)
    ]
    assert sum(sim.native_adjoint_calls for sim in fallback_sims) >= 1
    assert any(sim.statevector_reads == 1 for sim in fallback_sims)

    class _StatusRejectingNativeAdjointSimulator(_RYPreservingSimulator):
        def __init__(self, num_qubits):
            super().__init__(num_qubits)
            self.native_adjoint_calls = 0

        def adjoint_jacobian(self, operations, observables, trainable_params):
            self.native_adjoint_calls += 1
            raise RuntimeError("rocQuantum status 5: adjoint payload unsupported")

    fake = _install_fake_binding(monkeypatch)
    fake.QuantumSimulator = _StatusRejectingNativeAdjointSimulator
    fake.QSim = _StatusRejectingNativeAdjointSimulator
    for name in list(sys.modules):
        if name.startswith("pennylane_rocq"):
            sys.modules.pop(name)

    status_fallback_dev = qml.device("lightning.rocq", wires=1)

    @qml.qnode(status_fallback_dev, diff_method="adjoint")
    def status_fallback_circuit(theta):
        qml.RY(theta, wires=0)
        return qml.expval(qml.PauliZ(0))

    assert status_fallback_circuit(theta) == pytest.approx(math.cos(float(theta)))
    assert qml.grad(status_fallback_circuit)(theta) == pytest.approx(-math.sin(float(theta)))
    status_fallback_sims = [
        sim for sim in _StatusRejectingNativeAdjointSimulator.instances
        if isinstance(sim, _StatusRejectingNativeAdjointSimulator)
    ]
    assert sum(sim.native_adjoint_calls for sim in status_fallback_sims) >= 1
    assert any(sim.statevector_reads == 1 for sim in status_fallback_sims)

    class _MatrixOpNativeAdjointSimulator(_RYPreservingSimulator):
        def __init__(self, num_qubits):
            super().__init__(num_qubits)
            self.native_adjoint_calls = []

        def adjoint_jacobian(self, operations, observables, trainable_params):
            self.native_adjoint_calls.append(
                {
                    "operations": operations,
                    "observables": observables,
                    "trainable_params": trainable_params,
                }
            )
            theta = float(operations[1]["params"][0])
            return np.asarray([[-math.sin(theta)]], dtype=float)

    fake = _install_fake_binding(monkeypatch)
    fake.QuantumSimulator = _MatrixOpNativeAdjointSimulator
    fake.QSim = _MatrixOpNativeAdjointSimulator
    for name in list(sys.modules):
        if name.startswith("pennylane_rocq"):
            sys.modules.pop(name)

    matrix_native_dev = qml.device("lightning.rocq", wires=1)

    @qml.qnode(matrix_native_dev, diff_method="adjoint")
    def matrix_native_circuit(theta):
        qml.QubitUnitary(np.eye(2, dtype=np.complex128), wires=0)
        qml.RY(theta, wires=0)
        return qml.expval(qml.PauliZ(0))

    assert matrix_native_circuit(theta) == pytest.approx(math.cos(float(theta)))
    assert qml.grad(matrix_native_circuit)(theta) == pytest.approx(-math.sin(float(theta)))
    matrix_native_sims = [
        sim for sim in _MatrixOpNativeAdjointSimulator.instances
        if isinstance(sim, _MatrixOpNativeAdjointSimulator)
    ]
    native_matrix_sim = matrix_native_sims[-1]
    assert native_matrix_sim.statevector_reads == 0
    assert len(native_matrix_sim.native_adjoint_calls) == 1
    native_matrix_call = native_matrix_sim.native_adjoint_calls[0]
    assert native_matrix_call["trainable_params"] == [1]
    assert native_matrix_call["operations"] == [
        {
            "name": "QubitUnitary",
            "rocq_name": "matrix",
            "wires": [0],
            "params": [],
            "param_indices": [0],
            "trainable_param_indices": [],
            "trainable_param_positions": [],
            "matrix": [[(1.0, 0.0), (0.0, 0.0)], [(0.0, 0.0), (1.0, 0.0)]],
        },
        {
            "name": "RY",
            "rocq_name": "RY",
            "wires": [0],
            "params": [float(theta)],
            "param_indices": [1],
            "trainable_param_indices": [1],
            "trainable_param_positions": [0],
        },
    ]


def test_pennylane_native_adjoint_rejects_trainable_qubit_unitary_matrix(monkeypatch):
    pytest.importorskip("pennylane")
    _install_fake_binding(monkeypatch)
    for name in list(sys.modules):
        if name.startswith("pennylane_rocq"):
            sys.modules.pop(name)

    import pennylane as qml

    dev = qml.device("lightning.rocq", wires=1)
    tape = qml.tape.QuantumScript(
        [
            qml.QubitUnitary(np.eye(2, dtype=np.complex128), wires=0),
            qml.RY(0.321, wires=0),
        ],
        [qml.expval(qml.PauliZ(0))],
    )
    tape.trainable_params = [0, 1]

    assert dev._native_adjoint_payload(tape) is None


def test_pennylane_native_adjoint_folds_identity_hermitian_observable(monkeypatch):
    pytest.importorskip("pennylane")
    _install_fake_binding(monkeypatch)
    for name in list(sys.modules):
        if name.startswith("pennylane_rocq"):
            sys.modules.pop(name)

    import pennylane as qml

    dev = qml.device("lightning.rocq", wires=1)
    observable = qml.s_prod(2.0, qml.Hermitian(np.eye(2, dtype=np.complex128), wires=0))
    tape = qml.tape.QuantumScript(
        [qml.RY(0.321, wires=0)],
        [qml.expval(observable)],
    )
    tape.trainable_params = [0]

    operations, observables, trainable_params = dev._native_adjoint_payload(tape)

    assert trainable_params == [0]
    assert observables == [[{"coefficient": (2.0, 0.0), "pauli_string": "", "targets": []}]]
    assert operations == [
        {
            "name": "RY",
            "rocq_name": "RY",
            "wires": [0],
            "params": [0.321],
            "param_indices": [0],
            "trainable_param_indices": [0],
            "trainable_param_positions": [0],
        }
    ]


def test_pennylane_native_adjoint_lowers_diagonal_hermitian_to_pauli_terms(monkeypatch):
    pytest.importorskip("pennylane")
    _install_fake_binding(monkeypatch)
    for name in list(sys.modules):
        if name.startswith("pennylane_rocq"):
            sys.modules.pop(name)

    import pennylane as qml

    dev = qml.device("lightning.rocq", wires=1)
    observable = 2.0 * qml.Hermitian(np.diag([1.0, -1.0]).astype(np.complex128), wires=0)
    tape = qml.tape.QuantumScript(
        [qml.RY(0.321, wires=0)],
        [qml.expval(observable)],
    )
    tape.trainable_params = [0]

    operations, observables, trainable_params = dev._native_adjoint_payload(tape)

    assert trainable_params == [0]
    assert observables == [[{"coefficient": (2.0, 0.0), "pauli_string": "Z", "targets": [0]}]]
    assert [op["rocq_name"] for op in operations] == ["RY"]


def test_pennylane_native_adjoint_folds_identity_sparse_hamiltonian_observable(monkeypatch):
    pytest.importorskip("pennylane")
    sp = pytest.importorskip("scipy.sparse")
    _install_fake_binding(monkeypatch)
    for name in list(sys.modules):
        if name.startswith("pennylane_rocq"):
            sys.modules.pop(name)

    import pennylane as qml

    dev = qml.device("lightning.rocq", wires=1)
    observable = 2.0 * qml.SparseHamiltonian(sp.eye(2, dtype=np.complex128, format="csr"), wires=[0])
    tape = qml.tape.QuantumScript(
        [qml.RY(0.321, wires=0)],
        [qml.expval(observable)],
    )
    tape.trainable_params = [0]

    operations, observables, trainable_params = dev._native_adjoint_payload(tape)

    assert trainable_params == [0]
    assert observables == [[{"coefficient": (2.0, 0.0), "pauli_string": "", "targets": []}]]
    assert operations == [
        {
            "name": "RY",
            "rocq_name": "RY",
            "wires": [0],
            "params": [0.321],
            "param_indices": [0],
            "trainable_param_indices": [0],
            "trainable_param_positions": [0],
        }
    ]


def test_pennylane_native_adjoint_lowers_diagonal_sparse_hamiltonian_to_pauli_terms(monkeypatch):
    pytest.importorskip("pennylane")
    sp = pytest.importorskip("scipy.sparse")
    _install_fake_binding(monkeypatch)
    for name in list(sys.modules):
        if name.startswith("pennylane_rocq"):
            sys.modules.pop(name)

    import pennylane as qml

    dev = qml.device("lightning.rocq", wires=1)
    hamiltonian_matrix = sp.csr_matrix(np.diag([1.0, -1.0]).astype(np.complex128))
    observable = -0.5 * qml.SparseHamiltonian(hamiltonian_matrix, wires=[0])
    tape = qml.tape.QuantumScript(
        [qml.RY(0.321, wires=0)],
        [qml.expval(observable)],
    )
    tape.trainable_params = [0]

    operations, observables, trainable_params = dev._native_adjoint_payload(tape)

    assert trainable_params == [0]
    assert observables == [[{"coefficient": (-0.5, 0.0), "pauli_string": "Z", "targets": [0]}]]
    assert [op["rocq_name"] for op in operations] == ["RY"]


def test_pennylane_native_adjoint_uses_controlled_qubit_unitary_matrix_payload(monkeypatch):
    pytest.importorskip("pennylane")
    _install_fake_binding(monkeypatch)
    for name in list(sys.modules):
        if name.startswith("pennylane_rocq"):
            sys.modules.pop(name)

    import pennylane as qml

    base = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
    dev = qml.device("lightning.rocq", wires=2)
    tape = qml.tape.QuantumScript(
        [
            qml.ControlledQubitUnitary(base, wires=[0, 1], control_values=[False]),
            qml.RY(0.321, wires=1),
        ],
        [qml.expval(qml.PauliZ(1))],
    )
    tape.trainable_params = [1]

    operations, observables, trainable_params = dev._native_adjoint_payload(tape)

    assert trainable_params == [1]
    assert observables == [[{"coefficient": (1.0, 0.0), "pauli_string": "Z", "targets": [1]}]]
    assert operations == [
        {
            "name": "ControlledQubitUnitary",
            "rocq_name": "matrix",
            "wires": [1],
            "params": [],
            "param_indices": [0],
            "trainable_param_indices": [],
            "trainable_param_positions": [],
            "matrix": [[(0.0, 0.0), (1.0, 0.0)], [(1.0, 0.0), (0.0, 0.0)]],
            "controls": [0],
            "control_values": [False],
        },
        {
            "name": "RY",
            "rocq_name": "RY",
            "wires": [1],
            "params": [0.321],
            "param_indices": [1],
            "trainable_param_indices": [1],
            "trainable_param_positions": [0],
        },
    ]


def test_pennylane_native_adjoint_rejects_trainable_controlled_qubit_unitary_matrix(monkeypatch):
    pytest.importorskip("pennylane")
    _install_fake_binding(monkeypatch)
    for name in list(sys.modules):
        if name.startswith("pennylane_rocq"):
            sys.modules.pop(name)

    import pennylane as qml

    base = np.eye(2, dtype=np.complex128)
    dev = qml.device("lightning.rocq", wires=2)
    tape = qml.tape.QuantumScript(
        [
            qml.ControlledQubitUnitary(base, wires=[0, 1]),
            qml.RY(0.321, wires=1),
        ],
        [qml.expval(qml.PauliZ(1))],
    )
    tape.trainable_params = [0, 1]

    assert dev._native_adjoint_payload(tape) is None


def test_pennylane_native_adjoint_uses_block_encode_matrix_payload(monkeypatch):
    pytest.importorskip("pennylane")
    _install_fake_binding(monkeypatch)
    for name in list(sys.modules):
        if name.startswith("pennylane_rocq"):
            sys.modules.pop(name)

    import pennylane as qml
    from rocquantum.framework_runtime import matrix_to_little_endian_wires

    block = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.complex128)
    dev = qml.device("lightning.rocq", wires=2)
    block_op = qml.BlockEncode(block, wires=[0, 1])
    tape = qml.tape.QuantumScript(
        [
            block_op,
            qml.RY(0.321, wires=0),
        ],
        [qml.expval(qml.PauliZ(0))],
    )
    tape.trainable_params = [1]

    operations, observables, trainable_params = dev._native_adjoint_payload(tape)

    expected_matrix = [
        [(float(np.real(value)), float(np.imag(value))) for value in row]
        for row in matrix_to_little_endian_wires(qml.matrix(block_op))
    ]
    assert trainable_params == [1]
    assert observables == [[{"coefficient": (1.0, 0.0), "pauli_string": "Z", "targets": [0]}]]
    assert operations == [
        {
            "name": "BlockEncode",
            "rocq_name": "matrix",
            "wires": [0, 1],
            "params": [],
            "param_indices": [0],
            "trainable_param_indices": [],
            "trainable_param_positions": [],
            "matrix": expected_matrix,
        },
        {
            "name": "RY",
            "rocq_name": "RY",
            "wires": [0],
            "params": [0.321],
            "param_indices": [1],
            "trainable_param_indices": [1],
            "trainable_param_positions": [0],
        },
    ]


def test_pennylane_native_adjoint_rejects_trainable_block_encode_matrix(monkeypatch):
    pytest.importorskip("pennylane")
    _install_fake_binding(monkeypatch)
    for name in list(sys.modules):
        if name.startswith("pennylane_rocq"):
            sys.modules.pop(name)

    import pennylane as qml

    block = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.complex128)
    dev = qml.device("lightning.rocq", wires=2)
    tape = qml.tape.QuantumScript(
        [
            qml.BlockEncode(block, wires=[0, 1]),
            qml.RY(0.321, wires=0),
        ],
        [qml.expval(qml.PauliZ(0))],
    )
    tape.trainable_params = [0, 1]

    assert dev._native_adjoint_payload(tape) is None


def test_pennylane_native_adjoint_uses_sparse_block_encode_payload(monkeypatch):
    pytest.importorskip("pennylane")
    sp = pytest.importorskip("scipy.sparse")
    _install_fake_binding(monkeypatch)
    for name in list(sys.modules):
        if name.startswith("pennylane_rocq"):
            sys.modules.pop(name)

    import pennylane as qml
    from rocquantum.framework_runtime import sparse_matrix_to_little_endian_wires

    block = sp.csr_matrix(np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.complex128))
    dev = qml.device("lightning.rocq", wires=2)
    block_op = qml.BlockEncode(block, wires=[0, 1])
    tape = qml.tape.QuantumScript(
        [
            block_op,
            qml.RY(0.321, wires=0),
        ],
        [qml.expval(qml.PauliZ(0))],
    )
    tape.trainable_params = [1]

    operations, observables, trainable_params = dev._native_adjoint_payload(tape)

    sparse_matrix = sparse_matrix_to_little_endian_wires(block_op.sparse_matrix(format="csr"))
    assert trainable_params == [1]
    assert observables == [[{"coefficient": (1.0, 0.0), "pauli_string": "Z", "targets": [0]}]]
    assert operations == [
        {
            "name": "BlockEncode",
            "rocq_name": "sparse_matrix",
            "wires": [0, 1],
            "params": [],
            "param_indices": [0],
            "trainable_param_indices": [],
            "trainable_param_positions": [],
            "sparse_data": [(float(np.real(value)), float(np.imag(value))) for value in sparse_matrix.data],
            "sparse_indices": [int(index) for index in sparse_matrix.indices],
            "sparse_indptr": [int(offset) for offset in sparse_matrix.indptr],
            "sparse_shape": [int(dim) for dim in sparse_matrix.shape],
        },
        {
            "name": "RY",
            "rocq_name": "RY",
            "wires": [0],
            "params": [0.321],
            "param_indices": [1],
            "trainable_param_indices": [1],
            "trainable_param_positions": [0],
        },
    ]


def test_pennylane_native_adjoint_rejects_trainable_sparse_block_encode_matrix(monkeypatch):
    pytest.importorskip("pennylane")
    sp = pytest.importorskip("scipy.sparse")
    _install_fake_binding(monkeypatch)
    for name in list(sys.modules):
        if name.startswith("pennylane_rocq"):
            sys.modules.pop(name)

    import pennylane as qml

    block = sp.csr_matrix(np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.complex128))
    dev = qml.device("lightning.rocq", wires=2)
    tape = qml.tape.QuantumScript(
        [
            qml.BlockEncode(block, wires=[0, 1]),
            qml.RY(0.321, wires=0),
        ],
        [qml.expval(qml.PauliZ(0))],
    )
    tape.trainable_params = [0, 1]

    assert dev._native_adjoint_payload(tape) is None


def test_pennylane_native_adjoint_lowers_diagonal_qubit_unitary_payload(monkeypatch):
    pytest.importorskip("pennylane")
    _install_fake_binding(monkeypatch)
    for name in list(sys.modules):
        if name.startswith("pennylane_rocq"):
            sys.modules.pop(name)

    import pennylane as qml

    diagonal = np.exp(1j * np.array([0.1, 0.3, 0.6, 1.2]))
    dev = qml.device("lightning.rocq", wires=2)
    tape = qml.tape.QuantumScript(
        [
            qml.DiagonalQubitUnitary(diagonal, wires=[0, 1]),
            qml.RY(0.321, wires=0),
        ],
        [qml.expval(qml.PauliZ(0))],
    )
    tape.trainable_params = [1]

    operations, observables, trainable_params = dev._native_adjoint_payload(tape)

    assert trainable_params == [1]
    assert observables == [[{"coefficient": (1.0, 0.0), "pauli_string": "Z", "targets": [0]}]]
    assert [
        (
            op["name"],
            op["rocq_name"],
            op["wires"],
            op["param_indices"],
            op["trainable_param_indices"],
            op["trainable_param_positions"],
        )
        for op in operations
    ] == [
        ("RZ", "RZ", [0], [], [], []),
        ("RZ", "RZ", [1], [], [], []),
        ("CNOT", "CNOT", [0, 1], [], [], []),
        ("RZ", "RZ", [1], [], [], []),
        ("CNOT", "CNOT", [0, 1], [], [], []),
        ("RY", "RY", [0], [1], [1], [0]),
    ]
    np.testing.assert_allclose(
        [operations[index]["params"][0] for index in (0, 1, 3, 5)],
        [0.7, 0.4, -0.2, 0.321],
    )
    assert operations[2]["params"] == []
    assert operations[4]["params"] == []


def test_pennylane_native_adjoint_rejects_trainable_diagonal_qubit_unitary(monkeypatch):
    pytest.importorskip("pennylane")
    _install_fake_binding(monkeypatch)
    for name in list(sys.modules):
        if name.startswith("pennylane_rocq"):
            sys.modules.pop(name)

    import pennylane as qml

    diagonal = np.exp(1j * np.array([0.1, 0.3, 0.6, 1.2]))
    dev = qml.device("lightning.rocq", wires=2)
    tape = qml.tape.QuantumScript(
        [
            qml.DiagonalQubitUnitary(diagonal, wires=[0, 1]),
            qml.RY(0.321, wires=0),
        ],
        [qml.expval(qml.PauliZ(0))],
    )
    tape.trainable_params = [0, 1]

    assert dev._native_adjoint_payload(tape) is None


def test_pennylane_native_adjoint_lowers_fixed_template_payloads(monkeypatch):
    pytest.importorskip("pennylane")
    _install_fake_binding(monkeypatch)
    for name in list(sys.modules):
        if name.startswith("pennylane_rocq"):
            sys.modules.pop(name)

    import pennylane as qml

    dev = qml.device("lightning.rocq", wires=4)
    tape = qml.tape.QuantumScript(
        [
            qml.QFT(wires=[0, 1]),
            qml.BasisEmbedding(np.array([1, 0, 1]), wires=[0, 1, 2]),
            qml.Permute([2, 0, 1], wires=[0, 1, 2]),
            qml.QubitSum(wires=[0, 1, 2]),
            qml.QubitCarry(wires=[0, 1, 2, 3]),
            qml.GroverOperator(wires=[0, 1, 2]),
            qml.RY(0.321, wires=0),
        ],
        [qml.expval(qml.PauliZ(0))],
    )
    tape.trainable_params = [1]

    operations, observables, trainable_params = dev._native_adjoint_payload(tape)

    assert trainable_params == [1]
    assert observables == [[{"coefficient": (1.0, 0.0), "pauli_string": "Z", "targets": [0]}]]
    assert all(
        op["rocq_name"]
        not in {"QFT", "BasisEmbedding", "Permute", "QubitSum", "QubitCarry", "GroverOperator"}
        for op in operations
    )
    assert [(op["rocq_name"], op["wires"]) for op in operations] == [
        ("H", [0]),
        ("CP", [1, 0]),
        ("H", [1]),
        ("SWAP", [0, 1]),
        ("X", [0]),
        ("X", [2]),
        ("SWAP", [0, 2]),
        ("SWAP", [1, 2]),
        ("CNOT", [1, 2]),
        ("CNOT", [0, 2]),
        ("MCX", [1, 2, 3]),
        ("CNOT", [1, 2]),
        ("MCX", [0, 2, 3]),
        ("H", [0]),
        ("H", [1]),
        ("Z", [2]),
        ("X", [0]),
        ("X", [1]),
        ("MCX", [0, 1, 2]),
        ("X", [1]),
        ("X", [0]),
        ("Z", [2]),
        ("H", [0]),
        ("H", [1]),
        ("RY", [0]),
    ]
    np.testing.assert_allclose(operations[1]["params"], [np.pi / 2])
    assert all(not op["params"] for index, op in enumerate(operations) if index not in {1, 24})
    assert operations[-1]["params"] == [0.321]
    assert operations[-1]["param_indices"] == [1]
    assert operations[-1]["trainable_param_indices"] == [1]


def test_pennylane_native_adjoint_lowers_fixed_select_pauli_rot_payloads(monkeypatch):
    pytest.importorskip("pennylane")
    _install_fake_binding(monkeypatch)
    for name in list(sys.modules):
        if name.startswith("pennylane_rocq"):
            sys.modules.pop(name)

    import pennylane as qml

    dev = qml.device("lightning.rocq", wires=2)
    tape = qml.tape.QuantumScript(
        [
            qml.SelectPauliRot(np.array([0.25, 1.25]), control_wires=[0], target_wire=1, rot_axis="Z"),
            qml.SelectPauliRot(np.array([0.5, 1.5]), control_wires=[0], target_wire=1, rot_axis="X"),
            qml.SelectPauliRot(np.array([0.75, 1.75]), control_wires=[0], target_wire=1, rot_axis="Y"),
            qml.RY(0.321, wires=0),
        ],
        [qml.expval(qml.PauliZ(0))],
    )
    tape.trainable_params = [3]

    operations, observables, trainable_params = dev._native_adjoint_payload(tape)

    assert trainable_params == [3]
    assert observables == [[{"coefficient": (1.0, 0.0), "pauli_string": "Z", "targets": [0]}]]
    assert all(op["rocq_name"] != "SelectPauliRot" for op in operations)
    assert [(op["rocq_name"], op["wires"]) for op in operations] == [
        ("RZ", [1]),
        ("CNOT", [0, 1]),
        ("RZ", [1]),
        ("CNOT", [0, 1]),
        ("H", [1]),
        ("RZ", [1]),
        ("CNOT", [0, 1]),
        ("RZ", [1]),
        ("CNOT", [0, 1]),
        ("H", [1]),
        ("SDG", [1]),
        ("H", [1]),
        ("RZ", [1]),
        ("CNOT", [0, 1]),
        ("RZ", [1]),
        ("CNOT", [0, 1]),
        ("H", [1]),
        ("S", [1]),
        ("RY", [0]),
    ]
    np.testing.assert_allclose(
        [operations[index]["params"][0] for index in (0, 2, 5, 7, 12, 14, 18)],
        [0.75, -0.5, 1.0, -0.5, 1.25, -0.5, 0.321],
    )
    assert operations[-1]["param_indices"] == [3]
    assert operations[-1]["trainable_param_indices"] == [3]


def test_pennylane_native_adjoint_lowers_trainable_select_pauli_rot_angles(monkeypatch):
    pytest.importorskip("pennylane")
    _install_fake_binding(monkeypatch)
    for name in list(sys.modules):
        if name.startswith("pennylane_rocq"):
            sys.modules.pop(name)

    import pennylane as qml

    dev = qml.device("lightning.rocq", wires=2)
    tape = qml.tape.QuantumScript(
        [
            qml.SelectPauliRot(np.array([0.25, 1.25]), control_wires=[0], target_wire=1, rot_axis="Z"),
            qml.RY(0.321, wires=0),
        ],
        [qml.expval(qml.PauliZ(0))],
    )
    tape.trainable_params = [0, 1]

    operations, observables, trainable_params = dev._native_adjoint_payload(tape)

    select_columns = trainable_params[:2]
    assert trainable_params[2] == 1
    assert all(column < 0 for column in select_columns)
    assert len(set(select_columns)) == 2
    assert observables == [[{"coefficient": (1.0, 0.0), "pauli_string": "Z", "targets": [0]}]]
    assert [(op["rocq_name"], op["wires"]) for op in operations] == [
        ("RZ", [1]),
        ("CNOT", [0, 1]),
        ("RZ", [1]),
        ("CNOT", [0, 1]),
        ("RY", [0]),
    ]
    np.testing.assert_allclose(
        [operations[index]["params"][0] for index in (0, 2, 4)],
        [0.75, -0.5, 0.321],
    )
    assert operations[0]["param_indices"] == select_columns
    assert operations[0]["trainable_param_indices"] == select_columns
    assert operations[0]["trainable_param_positions"] == [0, 1]
    np.testing.assert_allclose(operations[0]["param_derivative_scales"], [0.5, 0.5])
    assert operations[2]["param_indices"] == select_columns
    assert operations[2]["trainable_param_indices"] == select_columns
    assert operations[2]["trainable_param_positions"] == [0, 1]
    np.testing.assert_allclose(operations[2]["param_derivative_scales"], [0.5, -0.5])
    assert operations[-1]["param_indices"] == [1]
    assert operations[-1]["trainable_param_indices"] == [1]


def test_pennylane_native_adjoint_reshapes_trainable_select_pauli_rot_jacobian(monkeypatch):
    pytest.importorskip("pennylane")

    class _SelectPauliRotNativeAdjointSimulator(_FakeQuantumSimulator):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.native_adjoint_calls = []

        def adjoint_jacobian(self, operations, observables, trainable_params):
            self.native_adjoint_calls.append(
                {
                    "operations": operations,
                    "observables": observables,
                    "trainable_params": trainable_params,
                }
            )
            return np.array([[0.11, 0.22, 0.33]], dtype=float)

    fake = _install_fake_binding(monkeypatch)
    fake.QuantumSimulator = _SelectPauliRotNativeAdjointSimulator
    fake.QSim = _SelectPauliRotNativeAdjointSimulator
    for name in list(sys.modules):
        if name.startswith("pennylane_rocq"):
            sys.modules.pop(name)

    import pennylane as qml

    dev = qml.device("lightning.rocq", wires=2)
    tape = qml.tape.QuantumScript(
        [
            qml.SelectPauliRot(np.array([0.25, 1.25]), control_wires=[0], target_wire=1, rot_axis="Z"),
            qml.RY(0.321, wires=0),
        ],
        [qml.expval(qml.PauliZ(0))],
    )
    tape.trainable_params = [0, 1]

    jacobian = dev.adjoint_jacobian(tape)

    assert isinstance(jacobian, tuple)
    np.testing.assert_allclose(jacobian[0], [0.11, 0.22])
    assert jacobian[1] == pytest.approx(0.33)
    native_call = _FakeQuantumSimulator.instances[-1].native_adjoint_calls[0]
    assert native_call["trainable_params"][:2] == native_call["operations"][0]["param_indices"]
    assert native_call["trainable_params"][2] == 1


def test_pennylane_native_adjoint_lowers_fixed_controlled_sequence_payloads(monkeypatch):
    pytest.importorskip("pennylane")
    _install_fake_binding(monkeypatch)
    for name in list(sys.modules):
        if name.startswith("pennylane_rocq"):
            sys.modules.pop(name)

    import pennylane as qml

    dev = qml.device("lightning.rocq", wires=3)
    tape = qml.tape.QuantumScript(
        [
            qml.ControlledSequence(qml.RX(0.2, wires=2), control=[0, 1]),
            qml.ControlledSequence(qml.PhaseShift(0.3, wires=2), control=[0, 1]),
            qml.ControlledSequence(qml.Hadamard(wires=2), control=[0, 1]),
            qml.ControlledSequence(qml.adjoint(qml.S(wires=2)), control=[0, 1]),
            qml.RY(0.321, wires=0),
        ],
        [qml.expval(qml.PauliZ(0))],
    )
    tape.trainable_params = [2]

    operations, observables, trainable_params = dev._native_adjoint_payload(tape)

    assert trainable_params == [2]
    assert observables == [[{"coefficient": (1.0, 0.0), "pauli_string": "Z", "targets": [0]}]]
    assert all(op["rocq_name"] != "ControlledSequence" for op in operations)
    assert [(op["rocq_name"], op["wires"]) for op in operations] == [
        ("CRX", [0, 2]),
        ("CRX", [1, 2]),
        ("CP", [0, 2]),
        ("CP", [1, 2]),
        ("RY", [2]),
        ("CNOT", [1, 2]),
        ("RY", [2]),
        ("CP", [0, 2]),
        ("CP", [1, 2]),
        ("RY", [0]),
    ]
    np.testing.assert_allclose(
        [operations[index]["params"][0] for index in (0, 1, 2, 3, 4, 6, 7, 8, 9)],
        [0.4, 0.2, 0.6, 0.3, np.pi / 4, -np.pi / 4, -np.pi, -np.pi / 2, 0.321],
    )
    assert all(not op["param_indices"] for op in operations[:-1])
    assert operations[-1]["param_indices"] == [2]
    assert operations[-1]["trainable_param_indices"] == [2]


def test_pennylane_native_adjoint_lowers_trainable_controlled_sequence_base(monkeypatch):
    pytest.importorskip("pennylane")
    _install_fake_binding(monkeypatch)
    for name in list(sys.modules):
        if name.startswith("pennylane_rocq"):
            sys.modules.pop(name)

    import pennylane as qml

    dev = qml.device("lightning.rocq", wires=3)
    tape = qml.tape.QuantumScript(
        [
            qml.ControlledSequence(qml.RX(0.2, wires=2), control=[0, 1]),
            qml.ControlledSequence(qml.PhaseShift(0.3, wires=2), control=[0, 1]),
            qml.RY(0.321, wires=0),
        ],
        [qml.expval(qml.PauliZ(0))],
    )
    tape.trainable_params = [0, 1, 2]

    operations, observables, trainable_params = dev._native_adjoint_payload(tape)

    assert trainable_params == [0, 1, 2]
    assert observables == [[{"coefficient": (1.0, 0.0), "pauli_string": "Z", "targets": [0]}]]
    assert [(op["rocq_name"], op["wires"], op["params"]) for op in operations] == [
        ("CRX", [0, 2], [0.4]),
        ("CRX", [1, 2], [0.2]),
        ("CP", [0, 2], [0.6]),
        ("CP", [1, 2], [0.3]),
        ("RY", [0], [0.321]),
    ]
    assert [op["param_indices"] for op in operations] == [[0], [0], [1], [1], [2]]
    assert [op["trainable_param_indices"] for op in operations] == [[0], [0], [1], [1], [2]]
    assert operations[0]["param_derivative_scales"] == [2.0]
    assert "param_derivative_scales" not in operations[1]
    assert operations[2]["param_derivative_scales"] == [2.0]
    assert "param_derivative_scales" not in operations[3]
    assert "param_derivative_scales" not in operations[4]


def test_pennylane_native_adjoint_lowers_select_payloads(monkeypatch):
    pytest.importorskip("pennylane")
    _install_fake_binding(monkeypatch)
    for name in list(sys.modules):
        if name.startswith("pennylane_rocq"):
            sys.modules.pop(name)

    import pennylane as qml

    dev = qml.device("lightning.rocq", wires=3)
    tape = qml.tape.QuantumScript(
        [
            qml.Select([qml.RX(0.2, wires=1), qml.RY(0.3, wires=1)], control=[0]),
            qml.Select([qml.PauliX(wires=1), qml.PauliZ(wires=1)], control=[0]),
            qml.RY(0.321, wires=2),
        ],
        [qml.expval(qml.PauliZ(2))],
    )
    tape.trainable_params = [0, 1, 2]

    operations, observables, trainable_params = dev._native_adjoint_payload(tape)

    assert trainable_params == [0, 1, 2]
    assert observables == [[{"coefficient": (1.0, 0.0), "pauli_string": "Z", "targets": [2]}]]
    assert all(op["rocq_name"] != "Select" for op in operations)
    assert [(op["rocq_name"], op["wires"]) for op in operations] == [
        ("X", [0]),
        ("CRX", [0, 1]),
        ("X", [0]),
        ("CRY", [0, 1]),
        ("X", [0]),
        ("CNOT", [0, 1]),
        ("X", [0]),
        ("CZ", [0, 1]),
        ("RY", [2]),
    ]
    assert operations[1]["params"] == [0.2]
    assert operations[1]["param_indices"] == [0]
    assert operations[1]["trainable_param_indices"] == [0]
    assert operations[3]["params"] == [0.3]
    assert operations[3]["param_indices"] == [1]
    assert operations[3]["trainable_param_indices"] == [1]
    assert operations[-1]["params"] == [0.321]
    assert operations[-1]["param_indices"] == [2]
    assert operations[-1]["trainable_param_indices"] == [2]


def test_pennylane_native_adjoint_lowers_partial_select_payloads(monkeypatch):
    pytest.importorskip("pennylane")
    _install_fake_binding(monkeypatch)
    for name in list(sys.modules):
        if name.startswith("pennylane_rocq"):
            sys.modules.pop(name)

    import pennylane as qml

    dev = qml.device("lightning.rocq", wires=6)
    tape = qml.tape.QuantumScript(
        [
            qml.Select(
                [qml.PauliX(wires=3), qml.PauliX(wires=4), qml.PauliX(wires=5)],
                control=[0, 1, 2],
                partial=True,
            ),
            qml.RY(0.321, wires=0),
        ],
        [qml.expval(qml.PauliZ(0))],
    )
    tape.trainable_params = [0]

    operations, observables, trainable_params = dev._native_adjoint_payload(tape)

    assert trainable_params == [0]
    assert observables == [[{"coefficient": (1.0, 0.0), "pauli_string": "Z", "targets": [0]}]]
    assert [(op["rocq_name"], op["wires"]) for op in operations] == [
        ("X", [1]),
        ("X", [2]),
        ("MCX", [1, 2, 3]),
        ("X", [2]),
        ("X", [1]),
        ("CNOT", [2, 4]),
        ("CNOT", [1, 5]),
        ("RY", [0]),
    ]
    assert operations[-1]["params"] == [0.321]
    assert operations[-1]["param_indices"] == [0]
    assert operations[-1]["trainable_param_indices"] == [0]


def test_pennylane_native_adjoint_lowers_select_basis_embedding_payloads(monkeypatch):
    pytest.importorskip("pennylane")
    _install_fake_binding(monkeypatch)
    for name in list(sys.modules):
        if name.startswith("pennylane_rocq"):
            sys.modules.pop(name)

    import pennylane as qml

    dev = qml.device("lightning.rocq", wires=3)
    tape = qml.tape.QuantumScript(
        [
            qml.Select(
                [
                    qml.BasisEmbedding(np.array([1, 0]), wires=[1, 2]),
                    qml.BasisEmbedding(np.array([0, 1]), wires=[1, 2]),
                ],
                control=[0],
            ),
            qml.RY(0.321, wires=0),
        ],
        [qml.expval(qml.PauliZ(0))],
    )
    tape.trainable_params = [2]

    operations, observables, trainable_params = dev._native_adjoint_payload(tape)

    assert trainable_params == [2]
    assert observables == [[{"coefficient": (1.0, 0.0), "pauli_string": "Z", "targets": [0]}]]
    assert [(op["rocq_name"], op["wires"]) for op in operations] == [
        ("X", [0]),
        ("CNOT", [0, 1]),
        ("X", [0]),
        ("CNOT", [0, 2]),
        ("RY", [0]),
    ]
    assert all(not op["params"] for op in operations[:-1])
    assert operations[-1]["params"] == [0.321]
    assert operations[-1]["param_indices"] == [2]
    assert operations[-1]["trainable_param_indices"] == [2]


def test_pennylane_native_adjoint_lowers_select_product_basis_phase_payloads(monkeypatch):
    pytest.importorskip("pennylane")
    _install_fake_binding(monkeypatch)
    for name in list(sys.modules):
        if name.startswith("pennylane_rocq"):
            sys.modules.pop(name)

    import pennylane as qml

    selected_product = qml.prod(
        qml.BasisEmbedding(np.array([1, 0]), wires=[1, 2]),
        qml.GlobalPhase(0.4, wires=[]),
    )
    dev = qml.device("lightning.rocq", wires=3)
    tape = qml.tape.QuantumScript(
        [
            qml.Select([selected_product, qml.PauliX(wires=1)], control=[0]),
            qml.RY(0.321, wires=2),
        ],
        [qml.expval(qml.PauliZ(2))],
    )
    tape.trainable_params = [2]

    operations, observables, trainable_params = dev._native_adjoint_payload(tape)

    assert trainable_params == [2]
    assert observables == [[{"coefficient": (1.0, 0.0), "pauli_string": "Z", "targets": [2]}]]
    assert [(op["rocq_name"], op["wires"]) for op in operations] == [
        ("X", [0]),
        ("CNOT", [0, 1]),
        ("X", [0]),
        ("X", [0]),
        ("P", [0]),
        ("X", [0]),
        ("CNOT", [0, 1]),
        ("RY", [2]),
    ]
    assert operations[4]["params"] == [-0.4]
    assert operations[4]["param_indices"] == [1]
    assert operations[4]["trainable_param_indices"] == []
    assert operations[-1]["params"] == [0.321]
    assert operations[-1]["param_indices"] == [2]
    assert operations[-1]["trainable_param_indices"] == [2]


def test_pennylane_native_adjoint_lowers_select_product_basis_native_payloads(monkeypatch):
    pytest.importorskip("pennylane")
    _install_fake_binding(monkeypatch)
    for name in list(sys.modules):
        if name.startswith("pennylane_rocq"):
            sys.modules.pop(name)

    import pennylane as qml

    selected_product = qml.prod(
        qml.BasisEmbedding(np.array([1, 0]), wires=[1, 2]),
        qml.PauliX(wires=2),
    )
    dev = qml.device("lightning.rocq", wires=3)
    tape = qml.tape.QuantumScript(
        [
            qml.Select([selected_product, qml.Identity(wires=1)], control=[0]),
            qml.RY(0.321, wires=0),
        ],
        [qml.expval(qml.PauliZ(0))],
    )
    tape.trainable_params = [1]

    operations, observables, trainable_params = dev._native_adjoint_payload(tape)

    assert trainable_params == [1]
    assert observables == [[{"coefficient": (1.0, 0.0), "pauli_string": "Z", "targets": [0]}]]
    assert [(op["rocq_name"], op["wires"]) for op in operations] == [
        ("X", [0]),
        ("CNOT", [0, 1]),
        ("X", [0]),
        ("X", [0]),
        ("CNOT", [0, 2]),
        ("X", [0]),
        ("RY", [0]),
    ]
    assert all(op["rocq_name"] != "Select" for op in operations)
    assert all(op["rocq_name"] != "matrix" for op in operations)
    assert operations[-1]["params"] == [0.321]
    assert operations[-1]["param_indices"] == [1]
    assert operations[-1]["trainable_param_indices"] == [1]


def test_pennylane_native_adjoint_lowers_select_multi_native_product_payloads(monkeypatch):
    pytest.importorskip("pennylane")
    _install_fake_binding(monkeypatch)
    for name in list(sys.modules):
        if name.startswith("pennylane_rocq"):
            sys.modules.pop(name)

    import pennylane as qml

    selected_product = qml.prod(
        qml.PauliX(wires=1),
        qml.PauliZ(wires=2),
        qml.GlobalPhase(0.4, wires=[]),
    )
    dev = qml.device("lightning.rocq", wires=3)
    tape = qml.tape.QuantumScript(
        [
            qml.Select([selected_product, qml.Identity(wires=1)], control=[0]),
            qml.RY(0.321, wires=2),
        ],
        [qml.expval(qml.PauliZ(2))],
    )
    tape.trainable_params = [1]

    operations, observables, trainable_params = dev._native_adjoint_payload(tape)

    assert trainable_params == [1]
    assert observables == [[{"coefficient": (1.0, 0.0), "pauli_string": "Z", "targets": [2]}]]
    assert [(op["rocq_name"], op["wires"]) for op in operations] == [
        ("X", [0]),
        ("CNOT", [0, 1]),
        ("X", [0]),
        ("X", [0]),
        ("CZ", [0, 2]),
        ("X", [0]),
        ("X", [0]),
        ("P", [0]),
        ("X", [0]),
        ("RY", [2]),
    ]
    assert operations[7]["params"] == [-0.4]
    assert operations[7]["param_indices"] == [0]
    assert operations[7]["trainable_param_indices"] == []
    assert operations[-1]["params"] == [0.321]
    assert operations[-1]["param_indices"] == [1]
    assert operations[-1]["trainable_param_indices"] == [1]


def test_pennylane_native_adjoint_lowers_select_qubit_unitary_payloads(monkeypatch):
    pytest.importorskip("pennylane")
    _install_fake_binding(monkeypatch)
    for name in list(sys.modules):
        if name.startswith("pennylane_rocq"):
            sys.modules.pop(name)

    import pennylane as qml

    x_matrix = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
    z_matrix = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)
    dev = qml.device("lightning.rocq", wires=2)
    tape = qml.tape.QuantumScript(
        [
            qml.Select(
                [
                    qml.QubitUnitary(x_matrix, wires=1),
                    qml.QubitUnitary(z_matrix, wires=1),
                ],
                control=[0],
            ),
            qml.RY(0.321, wires=1),
        ],
        [qml.expval(qml.PauliZ(1))],
    )
    tape.trainable_params = [2]

    operations, observables, trainable_params = dev._native_adjoint_payload(tape)

    assert trainable_params == [2]
    assert observables == [[{"coefficient": (1.0, 0.0), "pauli_string": "Z", "targets": [1]}]]
    assert operations == [
        {
            "name": "QubitUnitary",
            "rocq_name": "matrix",
            "wires": [1],
            "params": [],
            "param_indices": [0],
            "trainable_param_indices": [],
            "trainable_param_positions": [],
            "matrix": [[(0.0, 0.0), (1.0, 0.0)], [(1.0, 0.0), (0.0, 0.0)]],
            "controls": [0],
            "control_values": [False],
        },
        {
            "name": "QubitUnitary",
            "rocq_name": "matrix",
            "wires": [1],
            "params": [],
            "param_indices": [1],
            "trainable_param_indices": [],
            "trainable_param_positions": [],
            "matrix": [[(1.0, 0.0), (0.0, 0.0)], [(0.0, 0.0), (-1.0, 0.0)]],
            "controls": [0],
            "control_values": [True],
        },
        {
            "name": "RY",
            "rocq_name": "RY",
            "wires": [1],
            "params": [0.321],
            "param_indices": [2],
            "trainable_param_indices": [2],
            "trainable_param_positions": [0],
        },
    ]


def test_pennylane_native_adjoint_rejects_trainable_select_qubit_unitary_payload(monkeypatch):
    pytest.importorskip("pennylane")
    _install_fake_binding(monkeypatch)
    for name in list(sys.modules):
        if name.startswith("pennylane_rocq"):
            sys.modules.pop(name)

    import pennylane as qml

    matrix = np.eye(2, dtype=np.complex128)
    dev = qml.device("lightning.rocq", wires=2)
    tape = qml.tape.QuantumScript(
        [
            qml.Select(
                [
                    qml.QubitUnitary(matrix, wires=1),
                    qml.QubitUnitary(matrix.copy(), wires=1),
                ],
                control=[0],
            ),
            qml.RY(0.321, wires=1),
        ],
        [qml.expval(qml.PauliZ(1))],
    )
    tape.trainable_params = [0, 2]

    assert dev._native_adjoint_payload(tape) is None


def test_pennylane_native_adjoint_rejects_trainable_select_basis_embedding_payload(monkeypatch):
    pytest.importorskip("pennylane")
    _install_fake_binding(monkeypatch)
    for name in list(sys.modules):
        if name.startswith("pennylane_rocq"):
            sys.modules.pop(name)

    import pennylane as qml

    dev = qml.device("lightning.rocq", wires=3)
    tape = qml.tape.QuantumScript(
        [
            qml.Select(
                [
                    qml.BasisEmbedding(np.array([1, 0]), wires=[1, 2]),
                    qml.BasisEmbedding(np.array([0, 1]), wires=[1, 2]),
                ],
                control=[0],
            ),
            qml.RY(0.321, wires=0),
        ],
        [qml.expval(qml.PauliZ(0))],
    )
    tape.trainable_params = [0, 2]

    assert dev._native_adjoint_payload(tape) is None


def test_pennylane_native_adjoint_marks_trainable_operation_parameters(monkeypatch):
    pytest.importorskip("pennylane")
    _install_fake_binding(monkeypatch)
    for name in list(sys.modules):
        if name.startswith("pennylane_rocq"):
            sys.modules.pop(name)

    import pennylane as qml

    dev = qml.device("lightning.rocq", wires=2)
    tape = qml.tape.QuantumScript(
        [
            qml.Rot(0.1, 0.2, 0.3, wires=0),
            qml.CRot(0.4, 0.5, 0.6, wires=[0, 1]),
            qml.RZ(0.7, wires=0),
        ],
        [qml.expval(qml.PauliZ(0))],
    )
    tape.trainable_params = [1, 4, 6]

    operations, observables, trainable_params = dev._native_adjoint_payload(tape)

    assert trainable_params == [1, 4, 6]
    assert observables == [[{"coefficient": (1.0, 0.0), "pauli_string": "Z", "targets": [0]}]]
    assert operations == [
        {
            "name": "RZ",
            "rocq_name": "RZ",
            "wires": [0],
            "params": [0.1],
            "param_indices": [0],
            "trainable_param_indices": [],
            "trainable_param_positions": [],
        },
        {
            "name": "RY",
            "rocq_name": "RY",
            "wires": [0],
            "params": [0.2],
            "param_indices": [1],
            "trainable_param_indices": [1],
            "trainable_param_positions": [0],
        },
        {
            "name": "RZ",
            "rocq_name": "RZ",
            "wires": [0],
            "params": [0.3],
            "param_indices": [2],
            "trainable_param_indices": [],
            "trainable_param_positions": [],
        },
        {
            "name": "CRZ",
            "rocq_name": "CRZ",
            "wires": [0, 1],
            "params": [0.4],
            "param_indices": [3],
            "trainable_param_indices": [],
            "trainable_param_positions": [],
        },
        {
            "name": "CRY",
            "rocq_name": "CRY",
            "wires": [0, 1],
            "params": [0.5],
            "param_indices": [4],
            "trainable_param_indices": [4],
            "trainable_param_positions": [0],
        },
        {
            "name": "CRZ",
            "rocq_name": "CRZ",
            "wires": [0, 1],
            "params": [0.6],
            "param_indices": [5],
            "trainable_param_indices": [],
            "trainable_param_positions": [],
        },
        {
            "name": "RZ",
            "rocq_name": "RZ",
            "wires": [0],
            "params": [0.7],
            "param_indices": [6],
            "trainable_param_indices": [6],
            "trainable_param_positions": [0],
        },
    ]


def test_pennylane_native_adjoint_lowers_phase_payloads(monkeypatch):
    pytest.importorskip("pennylane")
    _install_fake_binding(monkeypatch)
    for name in list(sys.modules):
        if name.startswith("pennylane_rocq"):
            sys.modules.pop(name)

    import pennylane as qml

    dev = qml.device("lightning.rocq", wires=2)
    tape = qml.tape.QuantumScript(
        [
            qml.PhaseShift(0.1, wires=0),
            qml.ControlledPhaseShift(0.2, wires=[0, 1]),
            qml.CPhaseShift01(0.3, wires=[0, 1]),
        ],
        [qml.expval(qml.PauliZ(0))],
    )
    tape.trainable_params = [0, 1, 2]

    operations, observables, trainable_params = dev._native_adjoint_payload(tape)

    assert trainable_params == [0, 1, 2]
    assert observables == [[{"coefficient": (1.0, 0.0), "pauli_string": "Z", "targets": [0]}]]
    assert operations == [
        {
            "name": "PhaseShift",
            "rocq_name": "P",
            "wires": [0],
            "params": [0.1],
            "param_indices": [0],
            "trainable_param_indices": [0],
            "trainable_param_positions": [0],
        },
        {
            "name": "ControlledPhaseShift",
            "rocq_name": "CP",
            "wires": [0, 1],
            "params": [0.2],
            "param_indices": [1],
            "trainable_param_indices": [1],
            "trainable_param_positions": [0],
        },
        {
            "name": "PauliX",
            "rocq_name": "X",
            "wires": [0],
            "params": [],
            "param_indices": [],
            "trainable_param_indices": [],
            "trainable_param_positions": [],
        },
        {
            "name": "ControlledPhaseShift",
            "rocq_name": "CP",
            "wires": [0, 1],
            "params": [0.3],
            "param_indices": [2],
            "trainable_param_indices": [2],
            "trainable_param_positions": [0],
        },
        {
            "name": "PauliX",
            "rocq_name": "X",
            "wires": [0],
            "params": [],
            "param_indices": [],
            "trainable_param_indices": [],
            "trainable_param_positions": [],
        },
    ]


def test_pennylane_native_adjoint_elides_trainable_global_phase_payload(monkeypatch):
    pytest.importorskip("pennylane")
    _install_fake_binding(monkeypatch)
    for name in list(sys.modules):
        if name.startswith("pennylane_rocq"):
            sys.modules.pop(name)

    import pennylane as qml

    dev = qml.device("lightning.rocq", wires=1)
    tape = qml.tape.QuantumScript(
        [
            qml.GlobalPhase(0.123),
            qml.RY(0.321, wires=0),
        ],
        [qml.expval(qml.PauliZ(0))],
    )
    tape.trainable_params = [0, 1]

    operations, observables, trainable_params = dev._native_adjoint_payload(tape)

    assert trainable_params == [0, 1]
    assert observables == [[{"coefficient": (1.0, 0.0), "pauli_string": "Z", "targets": [0]}]]
    assert operations == [
        {
            "name": "RY",
            "rocq_name": "RY",
            "wires": [0],
            "params": [0.321],
            "param_indices": [1],
            "trainable_param_indices": [1],
            "trainable_param_positions": [0],
        }
    ]


def test_pennylane_native_adjoint_lowers_controlled_global_phase_wrapper(monkeypatch):
    pytest.importorskip("pennylane")
    _install_fake_binding(monkeypatch)
    for name in list(sys.modules):
        if name.startswith("pennylane_rocq"):
            sys.modules.pop(name)

    import pennylane as qml

    dev = qml.device("lightning.rocq", wires=3)
    tape = qml.tape.QuantumScript(
        [
            qml.ctrl(qml.GlobalPhase(0.7, wires=[]), control=[0, 1], control_values=[True, False]),
        ],
        [qml.expval(qml.PauliZ(2))],
    )
    tape.trainable_params = [0]

    operations, observables, trainable_params = dev._native_adjoint_payload(tape)

    assert trainable_params == [0]
    assert observables == [[{"coefficient": (1.0, 0.0), "pauli_string": "Z", "targets": [2]}]]
    assert all(not op["name"].startswith("C(") for op in operations)
    assert all(not op["rocq_name"].startswith("C(") for op in operations)
    assert operations[0]["rocq_name"] == "X"
    assert operations[-1]["rocq_name"] == "X"
    assert any(
        op["rocq_name"] == "RZ"
        and op["param_indices"] == [0]
        and op.get("param_derivative_scales") is not None
        for op in operations
    )


def test_pennylane_native_adjoint_lowers_controlled_phase_variant_wrappers(monkeypatch):
    pytest.importorskip("pennylane")
    _install_fake_binding(monkeypatch)
    for name in list(sys.modules):
        if name.startswith("pennylane_rocq"):
            sys.modules.pop(name)

    import pennylane as qml

    dev = qml.device("lightning.rocq", wires=4)
    tape = qml.tape.QuantumScript(
        [
            qml.ctrl(
                qml.CPhaseShift10(0.7, wires=[2, 3]),
                control=[0, 1],
                control_values=[True, False],
            ),
        ],
        [qml.expval(qml.PauliZ(0))],
    )
    tape.trainable_params = [0]

    operations, observables, trainable_params = dev._native_adjoint_payload(tape)

    assert trainable_params == [0]
    assert observables == [[{"coefficient": (1.0, 0.0), "pauli_string": "Z", "targets": [0]}]]
    assert all(not op["name"].startswith("C(") for op in operations)
    assert all(not op["rocq_name"].startswith("C(") for op in operations)
    assert operations[0]["rocq_name"] == "X"
    assert operations[0]["wires"] == [1]
    assert operations[1]["rocq_name"] == "X"
    assert operations[1]["wires"] == [3]
    assert operations[-2]["rocq_name"] == "X"
    assert operations[-2]["wires"] == [3]
    assert operations[-1]["rocq_name"] == "X"
    assert operations[-1]["wires"] == [1]
    assert {
        index
        for op in operations
        for index in op["trainable_param_indices"]
    } == {0}
    assert any(
        op["rocq_name"] == "RZ"
        and op["param_indices"] == [0]
        and op.get("param_derivative_scales") is not None
        for op in operations
    )


def test_pennylane_native_adjoint_lowers_controlled_wrapper_payloads(monkeypatch):
    pytest.importorskip("pennylane")
    _install_fake_binding(monkeypatch)
    for name in list(sys.modules):
        if name.startswith("pennylane_rocq"):
            sys.modules.pop(name)

    import pennylane as qml

    dev = qml.device("lightning.rocq", wires=4)
    tape = qml.tape.QuantumScript(
        [
            qml.ctrl(qml.RX(0.2, wires=2), control=[0, 1]),
            qml.ctrl(qml.RY(0.4, wires=2), control=[0, 1], control_values=[True, False]),
            qml.ctrl(qml.RZ(0.6, wires=2), control=[0, 1]),
            qml.ctrl(qml.PhaseShift(0.8, wires=2), control=[0, 1]),
            qml.ctrl(qml.Rot(0.1, 0.3, 0.5, wires=2), control=[0, 1]),
            qml.ctrl(qml.Hadamard(wires=2), control=[0, 1]),
            qml.ctrl(qml.S(wires=2), control=[0, 1]),
            qml.ctrl(qml.T(wires=2), control=[0, 1]),
            qml.ctrl(qml.SWAP(wires=[2, 3]), control=[0, 1], control_values=[True, False]),
            qml.ctrl(qml.ISWAP(wires=[2, 3]), control=[0, 1]),
            qml.ctrl(qml.PSWAP(0.9, wires=[2, 3]), control=[0, 1]),
            qml.ctrl(qml.SISWAP(wires=[2, 3]), control=[0, 1], control_values=[True, False]),
            qml.ctrl(qml.SQISW(wires=[2, 3]), control=[0, 1]),
            qml.ctrl(qml.ECR(wires=[2, 3]), control=[0, 1]),
        ],
        [qml.expval(qml.PauliZ(0))],
    )
    tape.trainable_params = list(range(8))

    operations, observables, trainable_params = dev._native_adjoint_payload(tape)

    assert trainable_params == list(range(8))
    assert observables == [[{"coefficient": (1.0, 0.0), "pauli_string": "Z", "targets": [0]}]]
    assert all(not op["name"].startswith("C(") for op in operations)
    assert all(not op["rocq_name"].startswith("C(") for op in operations)
    assert {op["rocq_name"] for op in operations}.issubset(
        {"CNOT", "CP", "CRX", "CRY", "CRZ", "CZ", "H", "MCX", "P", "RY", "RZ", "S", "SDG", "X"}
    )
    assert any(op["rocq_name"] == "X" and op["wires"] == [1] for op in operations)
    assert any(op["rocq_name"] == "MCX" and op["wires"] == [0, 1, 2] for op in operations)
    assert any(op["rocq_name"] == "MCX" and op["wires"] == [0, 1, 2, 3] for op in operations)
    assert any(op["rocq_name"] == "MCX" and op["wires"] == [0, 1, 3, 2] for op in operations)

    trainable_indices_in_payload = {
        index
        for op in operations
        for index in op["trainable_param_indices"]
    }
    assert trainable_indices_in_payload == set(range(8))

    scaled_ops = [
        (
            op["rocq_name"],
            tuple(op["wires"]),
            tuple(op["param_indices"]),
            tuple(op.get("param_derivative_scales", ())),
        )
        for op in operations
        if op.get("param_derivative_scales") is not None
    ]
    assert ("RZ", (0,), (0,), (-0.25,)) in scaled_ops
    assert ("RZ", (2,), (3,), (0.25,)) in scaled_ops
    assert any(rocq_name == "RZ" and param_indices == (7,) for rocq_name, _, param_indices, _ in scaled_ops)
    assert any(op["rocq_name"] == "RY" and op["params"] == [np.pi / 4] for op in operations)


def test_pennylane_native_adjoint_lowers_direct_fixed_gate_payloads(monkeypatch):
    pytest.importorskip("pennylane")
    _install_fake_binding(monkeypatch)
    for name in list(sys.modules):
        if name.startswith("pennylane_rocq"):
            sys.modules.pop(name)

    import pennylane as qml

    dev = qml.device("lightning.rocq", wires=3)
    tape = qml.tape.QuantumScript(
        [
            qml.adjoint(qml.S(wires=0)),
            qml.adjoint(qml.T(wires=1)),
            qml.CH(wires=[0, 1]),
            qml.CY(wires=[1, 2]),
            qml.CCZ(wires=[0, 1, 2]),
            qml.MultiControlledX(wires=[0, 1, 2], control_values=[False, True]),
            qml.ISWAP(wires=[1, 2]),
            qml.SISWAP(wires=[1, 2]),
            qml.ECR(wires=[1, 2]),
            qml.RY(0.4, wires=2),
        ],
        [qml.expval(qml.PauliZ(2))],
    )
    tape.trainable_params = [0]

    operations, observables, trainable_params = dev._native_adjoint_payload(tape)

    assert trainable_params == [0]
    assert observables == [[{"coefficient": (1.0, 0.0), "pauli_string": "Z", "targets": [2]}]]
    assert all(
        op["rocq_name"]
        not in {"CH", "CY", "CCZ", "MultiControlledX", "ISWAP", "SISWAP", "SQISW", "ECR", "Adjoint(S)", "Adjoint(T)"}
        for op in operations
    )
    assert operations[0]["rocq_name"] == "SDG"
    assert operations[1]["rocq_name"] == "TDG"
    assert ("X", [0]) in [(op["rocq_name"], op["wires"]) for op in operations]
    assert any(op["rocq_name"] == "MCX" and op["wires"] == [0, 1, 2] for op in operations)
    assert any(op["rocq_name"] == "CNOT" and op["wires"] == [1, 2] for op in operations)
    assert any(op["rocq_name"] == "RX" and op["params"] == [np.pi / 2] for op in operations)
    assert operations[-1]["rocq_name"] == "RY"
    assert operations[-1]["trainable_param_indices"] == [0]


def test_pennylane_native_adjoint_uses_direct_fixed_gate_payload(monkeypatch):
    pytest.importorskip("pennylane")

    class _DirectFixedGateNativeAdjointSimulator(_FakeQuantumSimulator):
        def __init__(self, num_qubits):
            super().__init__(num_qubits)
            self.native_adjoint_calls = []

        def adjoint_jacobian(self, operations, observables, trainable_params):
            self.native_adjoint_calls.append(
                {
                    "operations": operations,
                    "observables": observables,
                    "trainable_params": trainable_params,
                }
            )
            return np.asarray([[2.5]], dtype=float)

    fake = _install_fake_binding(monkeypatch)
    fake.QuantumSimulator = _DirectFixedGateNativeAdjointSimulator
    fake.QSim = _DirectFixedGateNativeAdjointSimulator
    for name in list(sys.modules):
        if name.startswith("pennylane_rocq"):
            sys.modules.pop(name)

    import pennylane as qml
    from pennylane import numpy as pnp

    dev = qml.device("lightning.rocq", wires=3)

    @qml.qnode(dev, diff_method="adjoint")
    def circuit(theta):
        qml.adjoint(qml.S(wires=0))
        qml.CH(wires=[0, 1])
        qml.CY(wires=[1, 2])
        qml.CCZ(wires=[0, 1, 2])
        qml.MultiControlledX(wires=[0, 1, 2], control_values=[False, True])
        qml.ISWAP(wires=[1, 2])
        qml.SISWAP(wires=[1, 2])
        qml.ECR(wires=[1, 2])
        qml.RY(theta, wires=2)
        return qml.expval(qml.PauliZ(2))

    theta = pnp.array(0.123, requires_grad=True)

    assert qml.grad(circuit)(theta) == pytest.approx(2.5)
    sim = _DirectFixedGateNativeAdjointSimulator.instances[-1]
    assert sim.statevector_reads == 0
    assert len(sim.native_adjoint_calls) == 1
    native_call = sim.native_adjoint_calls[0]
    assert all(
        op["rocq_name"]
        not in {"CH", "CY", "CCZ", "MultiControlledX", "ISWAP", "SISWAP", "SQISW", "ECR", "Adjoint(S)", "Adjoint(T)"}
        for op in native_call["operations"]
    )
    trainable_ry_ops = [
        op for op in native_call["operations"]
        if op["rocq_name"] == "RY" and op["trainable_param_indices"]
    ]
    assert len(trainable_ry_ops) == 1
    assert native_call["trainable_params"] == trainable_ry_ops[0]["trainable_param_indices"]


def test_pennylane_native_adjoint_uses_controlled_wrapper_payload(monkeypatch):
    pytest.importorskip("pennylane")

    class _ControlledWrapperNativeAdjointSimulator(_FakeQuantumSimulator):
        def __init__(self, num_qubits):
            super().__init__(num_qubits)
            self.native_adjoint_calls = []

        def adjoint_jacobian(self, operations, observables, trainable_params):
            self.native_adjoint_calls.append(
                {
                    "operations": operations,
                    "observables": observables,
                    "trainable_params": trainable_params,
                }
            )
            return np.asarray([[1.25]], dtype=float)

    fake = _install_fake_binding(monkeypatch)
    fake.QuantumSimulator = _ControlledWrapperNativeAdjointSimulator
    fake.QSim = _ControlledWrapperNativeAdjointSimulator
    for name in list(sys.modules):
        if name.startswith("pennylane_rocq"):
            sys.modules.pop(name)

    import pennylane as qml
    from pennylane import numpy as pnp

    dev = qml.device("lightning.rocq", wires=3)

    @qml.qnode(dev, diff_method="adjoint")
    def circuit(theta):
        qml.ctrl(qml.RX(theta, wires=2), control=[0, 1])
        return qml.expval(qml.PauliZ(2))

    theta = pnp.array(0.123, requires_grad=True)

    assert qml.grad(circuit)(theta) == pytest.approx(1.25)
    sim = _ControlledWrapperNativeAdjointSimulator.instances[-1]
    assert sim.statevector_reads == 0
    assert len(sim.native_adjoint_calls) == 1
    native_call = sim.native_adjoint_calls[0]
    assert native_call["trainable_params"] == [0]
    assert all(not op["rocq_name"].startswith("C(") for op in native_call["operations"])
    assert any(op["rocq_name"] == "RZ" and op["trainable_param_indices"] == [0] for op in native_call["operations"])


def test_pennylane_native_adjoint_lowers_decomposition_payloads_with_scales(monkeypatch):
    pytest.importorskip("pennylane")
    _install_fake_binding(monkeypatch)
    for name in list(sys.modules):
        if name.startswith("pennylane_rocq"):
            sys.modules.pop(name)

    import pennylane as qml

    dev = qml.device("lightning.rocq", wires=4)
    tape = qml.tape.QuantumScript(
        [
            qml.MultiRZ(0.1, wires=[0, 1, 2]),
            qml.PauliRot(0.2, "XYZ", wires=[0, 1, 2]),
            qml.IsingXY(0.3, wires=[0, 1]),
            qml.SingleExcitation(0.4, wires=[0, 1]),
            qml.PSWAP(0.5, wires=[0, 1]),
        ],
        [qml.expval(qml.PauliZ(0))],
    )
    tape.trainable_params = [0, 1, 2, 3, 4]

    operations, observables, trainable_params = dev._native_adjoint_payload(tape)

    assert trainable_params == [0, 1, 2, 3, 4]
    assert observables == [[{"coefficient": (1.0, 0.0), "pauli_string": "Z", "targets": [0]}]]
    assert [op["rocq_name"] for op in operations[:5]] == ["CNOT", "CNOT", "RZ", "CNOT", "CNOT"]
    assert operations[2] == {
        "name": "RZ",
        "rocq_name": "RZ",
        "wires": [2],
        "params": [0.1],
        "param_indices": [0],
        "trainable_param_indices": [0],
        "trainable_param_positions": [0],
    }

    pauli_rot_rz = [
        op
        for op in operations
        if op["rocq_name"] == "RZ" and op["wires"] == [2] and op["param_indices"] == [1]
    ]
    assert len(pauli_rot_rz) == 1
    assert pauli_rot_rz[0]["params"] == [0.2]
    assert "param_derivative_scales" not in pauli_rot_rz[0]

    scaled_ops = [
        {
            "rocq_name": op["rocq_name"],
            "wires": op["wires"],
            "params": op["params"],
            "param_indices": op["param_indices"],
            "param_derivative_scales": op.get("param_derivative_scales"),
        }
        for op in operations
        if op.get("param_derivative_scales") is not None
    ]
    assert scaled_ops == [
        {
            "rocq_name": "RY",
            "wires": [0],
            "params": [0.15],
            "param_indices": [2],
            "param_derivative_scales": [0.5],
        },
        {
            "rocq_name": "RX",
            "wires": [1],
            "params": [-0.15],
            "param_indices": [2],
            "param_derivative_scales": [-0.5],
        },
        {
            "rocq_name": "RY",
            "wires": [0],
            "params": [-0.2],
            "param_indices": [3],
            "param_derivative_scales": [-0.5],
        },
        {
            "rocq_name": "RY",
            "wires": [1],
            "params": [-0.2],
            "param_indices": [3],
            "param_derivative_scales": [-0.5],
        },
    ]
    assert any(
        op["rocq_name"] == "P"
        and op["wires"] == [1]
        and op["params"] == [0.5]
        and op["param_indices"] == [4]
        for op in operations
    )


def test_pennylane_native_adjoint_lowers_excitation_phase_and_fermionic_swap_payloads(monkeypatch):
    pytest.importorskip("pennylane")
    _install_fake_binding(monkeypatch)
    for name in list(sys.modules):
        if name.startswith("pennylane_rocq"):
            sys.modules.pop(name)

    import pennylane as qml

    dev = qml.device("lightning.rocq", wires=2)
    tape = qml.tape.QuantumScript(
        [
            qml.SingleExcitationPlus(0.6, wires=[0, 1]),
            qml.SingleExcitationMinus(0.8, wires=[0, 1]),
            qml.FermionicSWAP(1.0, wires=[0, 1]),
        ],
        [qml.expval(qml.PauliZ(0))],
    )
    tape.trainable_params = [0, 1, 2]

    operations, observables, trainable_params = dev._native_adjoint_payload(tape)

    assert trainable_params == [0, 1, 2]
    assert observables == [[{"coefficient": (1.0, 0.0), "pauli_string": "Z", "targets": [0]}]]
    scaled_ops = [
        (
            op["rocq_name"],
            tuple(op["wires"]),
            tuple(op["params"]),
            tuple(op["param_indices"]),
            tuple(op.get("param_derivative_scales", ())),
        )
        for op in operations
        if op.get("param_derivative_scales") is not None
    ]
    assert scaled_ops[:6] == [
        ("RY", (0,), (0.3,), (0,), (0.5,)),
        ("RY", (1,), (0.3,), (0,), (0.5,)),
        ("RZ", (1,), (-0.3,), (0,), (-0.5,)),
        ("RY", (0,), (0.4,), (1,), (0.5,)),
        ("RY", (1,), (0.4,), (1,), (0.5,)),
        ("RZ", (1,), (0.4,), (1,), (0.5,)),
    ]
    assert scaled_ops[6:] == [
        ("RZ", (1,), (0.5,), (2,), (0.5,)),
        ("RZ", (1,), (0.5,), (2,), (0.5,)),
        ("RZ", (0,), (0.5,), (2,), (0.5,)),
        ("RZ", (1,), (0.5,), (2,), (0.5,)),
    ]


def test_pennylane_native_adjoint_lowers_double_excitation_and_orbital_rotation_payloads(monkeypatch):
    pytest.importorskip("pennylane")
    _install_fake_binding(monkeypatch)
    for name in list(sys.modules):
        if name.startswith("pennylane_rocq"):
            sys.modules.pop(name)

    import pennylane as qml

    dev = qml.device("lightning.rocq", wires=4)
    tape = qml.tape.QuantumScript(
        [
            qml.DoubleExcitation(0.8, wires=[0, 1, 2, 3]),
            qml.DoubleExcitationPlus(1.6, wires=[0, 1, 2, 3]),
            qml.DoubleExcitationMinus(0.6, wires=[0, 1, 2, 3]),
            qml.OrbitalRotation(0.4, wires=[0, 1, 2, 3]),
        ],
        [qml.expval(qml.PauliZ(0))],
    )
    tape.trainable_params = [0, 1, 2, 3]

    operations, observables, trainable_params = dev._native_adjoint_payload(tape)

    assert trainable_params == [0, 1, 2, 3]
    assert observables == [[{"coefficient": (1.0, 0.0), "pauli_string": "Z", "targets": [0]}]]

    def trainable_ops(param_index):
        return [op for op in operations if op["param_indices"] == [param_index]]

    double_ops = trainable_ops(0)
    assert [op["rocq_name"] for op in double_ops] == ["RY"] * 8
    assert [op["wires"] for op in double_ops] == [[1], [0], [1], [0], [1], [0], [1], [0]]
    assert [op["params"][0] for op in double_ops] == pytest.approx(
        [0.1, -0.1, 0.1, -0.1, -0.1, 0.1, -0.1, 0.1]
    )
    assert [op["param_derivative_scales"][0] for op in double_ops] == pytest.approx(
        [0.125, -0.125, 0.125, -0.125, -0.125, 0.125, -0.125, 0.125]
    )

    double_plus_ops = trainable_ops(1)
    assert [op["rocq_name"] for op in double_plus_ops[:7]] == ["RZ"] * 7
    assert [op["param_derivative_scales"][0] for op in double_plus_ops[:7]] == pytest.approx(
        [0.125, -0.125, -0.125, -0.125, -0.125, 0.125, 0.125]
    )
    assert [op["rocq_name"] for op in double_plus_ops[7:]] == ["RY"] * 8

    double_minus_ops = trainable_ops(2)
    assert [op["rocq_name"] for op in double_minus_ops[:7]] == ["RZ"] * 7
    assert [op["param_derivative_scales"][0] for op in double_minus_ops[:7]] == pytest.approx(
        [-0.125, 0.125, 0.125, 0.125, 0.125, -0.125, -0.125]
    )
    assert [op["rocq_name"] for op in double_minus_ops[7:]] == ["RY"] * 8

    orbital_ops = trainable_ops(3)
    assert [op["rocq_name"] for op in orbital_ops] == ["RY"] * 4
    assert [op["wires"] for op in orbital_ops] == [[0], [1], [2], [3]]
    assert [op["params"][0] for op in orbital_ops] == pytest.approx([-0.2, -0.2, -0.2, -0.2])
    assert [op["param_derivative_scales"][0] for op in orbital_ops] == pytest.approx([-0.5] * 4)


def test_pennylane_native_adjoint_accepts_hermitian_payload(monkeypatch):
    pytest.importorskip("pennylane")

    class _HermitianNativeAdjointSimulator(_FakeQuantumSimulator):
        def __init__(self, num_qubits):
            super().__init__(num_qubits)
            self.enable_matrix_expectation = True
            self.native_adjoint_calls = []
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

        def adjoint_jacobian(self, operations, observables, trainable_params):
            self.native_adjoint_calls.append(
                {
                    "operations": operations,
                    "observables": observables,
                    "trainable_params": trainable_params,
                }
            )
            theta = float(operations[0]["params"][0])
            return np.asarray([[math.cos(theta)]], dtype=float)

    fake = _install_fake_binding(monkeypatch)
    fake.QuantumSimulator = _HermitianNativeAdjointSimulator
    fake.QSim = _HermitianNativeAdjointSimulator
    for name in list(sys.modules):
        if name.startswith("pennylane_rocq"):
            sys.modules.pop(name)

    import pennylane as qml
    from pennylane import numpy as pnp

    dev = qml.device("lightning.rocq", wires=1)
    observable = qml.Hermitian(np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128), wires=0)

    @qml.qnode(dev, diff_method="adjoint")
    def circuit(theta):
        qml.RY(theta, wires=0)
        return qml.expval(observable)

    theta = pnp.array(0.321, requires_grad=True)

    assert circuit(theta) == pytest.approx(math.sin(float(theta)))
    assert qml.grad(circuit)(theta) == pytest.approx(math.cos(float(theta)))

    sim = [
        instance for instance in _HermitianNativeAdjointSimulator.instances
        if isinstance(instance, _HermitianNativeAdjointSimulator)
    ][-1]
    assert sim.statevector_reads == 0
    assert len(sim.matrix_expectations) >= 1
    assert len(sim.native_adjoint_calls) == 1
    native_call = sim.native_adjoint_calls[0]
    assert native_call["observables"] == [
        [
            {
                "kind": "matrix",
                "matrix": [
                    [(0.0, 0.0), (1.0, 0.0)],
                    [(1.0, 0.0), (0.0, 0.0)],
                ],
                "targets": [0],
            }
        ]
    ]
    assert native_call["trainable_params"] == [0]


def test_pennylane_native_adjoint_accepts_scaled_hermitian_payload(monkeypatch):
    pytest.importorskip("pennylane")
    _install_fake_binding(monkeypatch)
    for name in list(sys.modules):
        if name.startswith("pennylane_rocq"):
            sys.modules.pop(name)

    import pennylane as qml

    dev = qml.device("lightning.rocq", wires=1)
    observable = 2.0 * qml.Hermitian(
        np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128),
        wires=0,
    )
    tape = qml.tape.QuantumScript(
        [qml.RY(0.321, wires=0)],
        [qml.expval(observable)],
    )
    tape.trainable_params = [0]

    operations, observables, trainable_params = dev._native_adjoint_payload(tape)

    assert trainable_params == [0]
    assert [op["rocq_name"] for op in operations] == ["RY"]
    assert observables == [
        [
            {
                "kind": "matrix",
                "matrix": [
                    [(0.0, 0.0), (2.0, 0.0)],
                    [(2.0, 0.0), (0.0, 0.0)],
                ],
                "targets": [0],
            }
        ]
    ]


def test_pennylane_native_adjoint_accepts_sparse_hamiltonian_payload(monkeypatch):
    pytest.importorskip("pennylane")
    sp = pytest.importorskip("scipy.sparse")

    class _SparseNativeAdjointSimulator(_FakeQuantumSimulator):
        def __init__(self, num_qubits):
            super().__init__(num_qubits)
            self.native_adjoint_calls = []
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

        def sparse_hamiltonian_moments(self, data, indices, indptr, shape):
            data = np.asarray(data, dtype=np.complex128)
            indices = np.asarray(indices, dtype=np.int64)
            indptr = np.asarray(indptr, dtype=np.int64)
            shape = tuple(int(dim) for dim in shape)
            self.sparse_moments.append((data, indices, indptr, shape))
            h_state = np.zeros_like(self._state)
            for row in range(shape[0]):
                for offset in range(int(indptr[row]), int(indptr[row + 1])):
                    h_state[row] += data[offset] * self._state[int(indices[offset])]
            return np.vdot(self._state, h_state), np.vdot(h_state, h_state)

        def adjoint_jacobian(self, operations, observables, trainable_params):
            self.native_adjoint_calls.append(
                {
                    "operations": operations,
                    "observables": observables,
                    "trainable_params": trainable_params,
                }
            )
            theta = float(operations[0]["params"][0])
            return np.asarray([[math.cos(theta)]], dtype=float)

    fake = _install_fake_binding(monkeypatch)
    fake.QuantumSimulator = _SparseNativeAdjointSimulator
    fake.QSim = _SparseNativeAdjointSimulator
    for name in list(sys.modules):
        if name.startswith("pennylane_rocq"):
            sys.modules.pop(name)

    import pennylane as qml
    from pennylane import numpy as pnp

    dev = qml.device("lightning.rocq", wires=1)
    hamiltonian_matrix = _single_qubit_sparse_x(sp)
    observable = qml.SparseHamiltonian(hamiltonian_matrix, wires=[0])

    @qml.qnode(dev, diff_method="adjoint")
    def circuit(theta):
        qml.RY(theta, wires=0)
        return qml.expval(observable)

    theta = pnp.array(0.321, requires_grad=True)

    assert circuit(theta) == pytest.approx(math.sin(float(theta)))
    assert qml.grad(circuit)(theta) == pytest.approx(math.cos(float(theta)))

    sim = [
        instance for instance in _SparseNativeAdjointSimulator.instances
        if isinstance(instance, _SparseNativeAdjointSimulator)
    ][-1]
    assert sim.statevector_reads == 0
    assert len(sim.sparse_moments) >= 1
    assert len(sim.native_adjoint_calls) == 1
    native_call = sim.native_adjoint_calls[0]
    assert native_call["observables"] == [
        [
            {
                "kind": "sparse",
                "data": [(1.0, 0.0), (1.0, 0.0)],
                "indices": [1, 0],
                "indptr": [0, 1, 2],
                "shape": [2, 2],
            }
        ]
    ]
    assert native_call["trainable_params"] == [0]


def test_pennylane_native_adjoint_accepts_scaled_sparse_hamiltonian_payload(monkeypatch):
    pytest.importorskip("pennylane")
    sp = pytest.importorskip("scipy.sparse")
    _install_fake_binding(monkeypatch)
    for name in list(sys.modules):
        if name.startswith("pennylane_rocq"):
            sys.modules.pop(name)

    import pennylane as qml

    dev = qml.device("lightning.rocq", wires=1)
    hamiltonian_matrix = _single_qubit_sparse_x(sp)
    observable = qml.s_prod(-0.5, qml.SparseHamiltonian(hamiltonian_matrix, wires=[0]))
    tape = qml.tape.QuantumScript(
        [qml.RY(0.321, wires=0)],
        [qml.expval(observable)],
    )
    tape.trainable_params = [0]

    operations, observables, trainable_params = dev._native_adjoint_payload(tape)

    assert trainable_params == [0]
    assert [op["rocq_name"] for op in operations] == ["RY"]
    assert observables == [
        [
            {
                "kind": "sparse",
                "data": [(-0.5, 0.0), (-0.5, 0.0)],
                "indices": [1, 0],
                "indptr": [0, 1, 2],
                "shape": [2, 2],
            }
        ]
    ]


def test_pennylane_native_adjoint_accepts_mixed_observable_sum_payload(monkeypatch):
    pytest.importorskip("pennylane")
    sp = pytest.importorskip("scipy.sparse")
    _install_fake_binding(monkeypatch)
    for name in list(sys.modules):
        if name.startswith("pennylane_rocq"):
            sys.modules.pop(name)

    import pennylane as qml

    dev = qml.device("lightning.rocq", wires=1)
    matrix = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
    observable = qml.sum(
        qml.Hermitian(matrix, wires=0),
        -0.25 * qml.SparseHamiltonian(sp.csr_matrix(matrix), wires=[0]),
        0.5 * qml.PauliZ(0),
    )
    tape = qml.tape.QuantumScript(
        [qml.RY(0.321, wires=0)],
        [qml.expval(observable)],
    )
    tape.trainable_params = [0]

    operations, observables, trainable_params = dev._native_adjoint_payload(tape)

    assert trainable_params == [0]
    assert [op["rocq_name"] for op in operations] == ["RY"]
    assert observables == [
        [
            {
                "kind": "matrix",
                "matrix": [
                    [(0.0, 0.0), (1.0, 0.0)],
                    [(1.0, 0.0), (0.0, 0.0)],
                ],
                "targets": [0],
            },
            {
                "kind": "sparse",
                "data": [(-0.25, 0.0), (-0.25, 0.0)],
                "indices": [1, 0],
                "indptr": [0, 1, 2],
                "shape": [2, 2],
            },
            {
                "coefficient": (0.5, 0.0),
                "pauli_string": "Z",
                "targets": [0],
            },
        ]
    ]


def test_pennylane_native_adjoint_uses_partial_sparse_hamiltonian_payload(monkeypatch):
    pytest.importorskip("pennylane")
    sp = pytest.importorskip("scipy.sparse")
    _install_fake_binding(monkeypatch)
    for name in list(sys.modules):
        if name.startswith("pennylane_rocq"):
            sys.modules.pop(name)

    import pennylane as qml

    dev = qml.device("lightning.rocq", wires=2)
    hamiltonian_matrix = _single_qubit_sparse_x(sp)
    observable = qml.SparseHamiltonian(hamiltonian_matrix, wires=[1])
    tape = qml.tape.QuantumScript(
        [qml.RY(0.321, wires=1)],
        [qml.expval(observable)],
    )
    tape.trainable_params = [0]

    operations, observables, trainable_params = dev._native_adjoint_payload(tape)

    assert trainable_params == [0]
    assert operations == [
        {
            "name": "RY",
            "rocq_name": "RY",
            "wires": [1],
            "params": [0.321],
            "param_indices": [0],
            "trainable_param_indices": [0],
            "trainable_param_positions": [0],
        }
    ]
    assert observables == [
        [
            {
                "kind": "sparse",
                "data": [(1.0, 0.0), (1.0, 0.0)],
                "indices": [1, 0],
                "indptr": [0, 1, 2],
                "shape": [2, 2],
                "targets": [1],
            }
        ]
    ]


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


def test_runtime_apply_sparse_matrix_prefers_native_hook():
    from rocquantum.framework_runtime import RocQuantumRuntime

    class _SparseApplySimulator(_FakeQuantumSimulator):
        def __init__(self, num_qubits):
            super().__init__(num_qubits)
            self.sparse_applications = []

        def apply_sparse_matrix(self, data, indices, indptr, shape, targets):
            self.sparse_applications.append(
                (
                    np.asarray(data, dtype=np.complex128).copy(),
                    np.asarray(indices, dtype=np.int64).copy(),
                    np.asarray(indptr, dtype=np.int64).copy(),
                    tuple(shape),
                    tuple(targets),
                )
            )

    sim = _SparseApplySimulator(2)
    runtime = RocQuantumRuntime(sim)

    runtime.apply_sparse_matrix(
        np.array([1.0, -1.0], dtype=np.complex128),
        np.array([0, 1], dtype=np.int64),
        np.array([0, 1, 2], dtype=np.int64),
        (2, 2),
        [1],
    )

    assert len(sim.sparse_applications) == 1
    data, indices, indptr, shape, targets = sim.sparse_applications[0]
    np.testing.assert_allclose(data, np.array([1.0, -1.0], dtype=np.complex128))
    np.testing.assert_array_equal(indices, np.array([0, 1], dtype=np.int64))
    np.testing.assert_array_equal(indptr, np.array([0, 1, 2], dtype=np.int64))
    assert shape == (2, 2)
    assert targets == (1,)
    assert sim.statevector_reads == 0
    assert sim.statevectors == []


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


def test_runtime_dense_expectation_falls_back_to_statevector():
    from rocquantum.framework_runtime import RocQuantumRuntime

    class _StateOnlySimulator:
        def __init__(self):
            self.statevector_reads = 0

        def batch_size(self):
            return 1

        def expectation_matrix(self, matrix, targets):
            raise NotImplementedError("native dense expectation disabled")

        def get_statevector(self):
            self.statevector_reads += 1
            return np.array([1.0 / math.sqrt(2.0), 1.0 / math.sqrt(2.0)], dtype=np.complex128)

    simulator = _StateOnlySimulator()
    runtime = RocQuantumRuntime(simulator)

    assert runtime.expectation_matrix(np.array([[0.0, 1.0], [1.0, 0.0]]), [0]) == pytest.approx(1.0)
    assert simulator.statevector_reads == 1


def test_runtime_dense_expectation_moments_fallback_reads_state_once():
    from rocquantum.framework_runtime import RocQuantumRuntime

    class _StateOnlySimulator:
        def __init__(self):
            self.statevector_reads = 0

        def batch_size(self):
            return 1

        def expectation_matrix(self, matrix, targets):
            raise NotImplementedError("native dense expectation disabled")

        def get_statevector(self):
            self.statevector_reads += 1
            return np.array([1.0 / math.sqrt(2.0), 1.0 / math.sqrt(2.0)], dtype=np.complex128)

    simulator = _StateOnlySimulator()
    runtime = RocQuantumRuntime(simulator)

    mean, second_moment = runtime.expectation_matrix_moments(np.array([[0.0, 1.0], [1.0, 0.0]]), [0])

    assert mean == pytest.approx(1.0)
    assert second_moment == pytest.approx(1.0)
    assert simulator.statevector_reads == 1


def test_runtime_dense_expectation_moments_prefers_native_hook():
    from rocquantum.framework_runtime import RocQuantumRuntime

    class _NativeMomentsSimulator:
        def __init__(self):
            self.calls = []

        def batch_size(self):
            return 1

        def expectation_matrix_moments(self, matrix, targets):
            self.calls.append((np.asarray(matrix, dtype=np.complex128).copy(), tuple(targets)))
            return 0.25 + 0.0j, 1.25 + 0.0j

        def expectation_matrix(self, matrix, targets):
            raise AssertionError("single expectation should not be used when moments hook exists")

        def get_statevector(self):
            raise AssertionError("statevector fallback should not be used when moments hook exists")

    simulator = _NativeMomentsSimulator()
    runtime = RocQuantumRuntime(simulator)

    mean, second_moment = runtime.expectation_matrix_moments(np.array([[1.0, 0.0], [0.0, -1.0]]), [0])

    assert mean == pytest.approx(0.25)
    assert second_moment == pytest.approx(1.25)
    assert len(simulator.calls) == 1
    np.testing.assert_allclose(simulator.calls[0][0], np.diag([1.0, -1.0]))
    assert simulator.calls[0][1] == (0,)


def test_runtime_dense_expectation_moments_batch_prefers_native_hook():
    from rocquantum.framework_runtime import RocQuantumRuntime

    class _NativeBatchMomentsSimulator:
        def __init__(self):
            self.calls = []

        def batch_size(self):
            return 2

        def expectation_matrix_moments_batch(self, matrix, targets):
            self.calls.append((np.asarray(matrix, dtype=np.complex128).copy(), tuple(targets)))
            return (
                np.array([0.25, -0.5], dtype=np.complex128),
                np.array([1.25, 1.5], dtype=np.complex128),
            )

        def expectation_matrix_batch(self, matrix, targets):
            raise AssertionError("batched expectation should not be used when moments hook exists")

        def get_statevectors(self):
            raise AssertionError("statevector fallback should not be used when moments hook exists")

    simulator = _NativeBatchMomentsSimulator()
    runtime = RocQuantumRuntime(simulator)

    means, second_moments = runtime.expectation_matrix_moments_batch(
        np.array([[1.0, 0.0], [0.0, -1.0]]),
        [0],
    )

    np.testing.assert_allclose(means, np.array([0.25, -0.5]))
    np.testing.assert_allclose(second_moments, np.array([1.25, 1.5]))
    assert len(simulator.calls) == 1
    np.testing.assert_allclose(simulator.calls[0][0], np.diag([1.0, -1.0]))
    assert simulator.calls[0][1] == (0,)


def test_runtime_dense_expectation_moments_batch_fallback_reads_statevectors_once():
    from rocquantum.framework_runtime import RocQuantumRuntime

    class _StateBatchSimulator:
        num_qubits = 1

        def __init__(self):
            self.statevector_reads = 0

        def batch_size(self):
            return 2

        def get_statevectors(self):
            self.statevector_reads += 1
            return np.array(
                [
                    [1.0 / math.sqrt(2.0), 1.0 / math.sqrt(2.0)],
                    [1.0, 0.0],
                ],
                dtype=np.complex128,
            )

    simulator = _StateBatchSimulator()
    runtime = RocQuantumRuntime(simulator)

    means, second_moments = runtime.expectation_matrix_moments_batch(
        np.array([[0.0, 1.0], [1.0, 0.0]]),
        [0],
    )

    np.testing.assert_allclose(means, np.array([1.0, 0.0]))
    np.testing.assert_allclose(second_moments, np.array([1.0, 1.0]))
    assert simulator.statevector_reads == 1


def test_framework_runtime_revalidates_native_complex_expectation_results():
    from rocquantum.framework_runtime import (
        RocQuantumRuntime,
        normalize_complex_result_scalar,
        normalize_complex_result_vector,
    )

    invalid_scalars = (
        True,
        np.bool_(False),
        "0.5",
        b"0.5",
        np.nan,
        np.inf,
        1.0j * np.inf,
        [0.5, 0.25],
    )
    for value in invalid_scalars:
        with pytest.raises(ValueError, match="Dense expectation"):
            normalize_complex_result_scalar(value, "Dense expectation value")

    invalid_vectors = (
        [0.5, True],
        [0.5, "0.25"],
        [0.5, np.nan],
        [0.5],
        [0.5, 0.25, 0.0],
    )
    for values in invalid_vectors:
        with pytest.raises(ValueError, match="Batched dense"):
            normalize_complex_result_vector(values, "Batched dense expectation values", expected_count=2)

    matrix = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)
    sparse_data = np.array([1.0, -1.0], dtype=np.complex128)
    sparse_indices = np.array([0, 1], dtype=np.int64)
    sparse_indptr = np.array([0, 1, 2], dtype=np.int64)

    class _BadDenseExpectationSimulator:
        def __init__(self, value):
            self.value = value
            self.calls = []

        def batch_size(self):
            return 1

        def expectation_matrix(self, matrix_arg, targets):
            self.calls.append((np.asarray(matrix_arg, dtype=np.complex128).copy(), tuple(targets)))
            return self.value

    for value in invalid_scalars:
        sim = _BadDenseExpectationSimulator(value)
        with pytest.raises(ValueError, match="Dense expectation"):
            RocQuantumRuntime(sim).expectation_matrix(matrix, [0])
        assert len(sim.calls) == 1

    class _BadDenseMomentsSimulator:
        def __init__(self, mean, second_moment):
            self.mean = mean
            self.second_moment = second_moment
            self.calls = []

        def batch_size(self):
            return 1

        def expectation_matrix_moments(self, matrix_arg, targets):
            self.calls.append((np.asarray(matrix_arg, dtype=np.complex128).copy(), tuple(targets)))
            return self.mean, self.second_moment

    for mean, second_moment in ((np.nan, 1.0), (0.5, "1.0")):
        sim = _BadDenseMomentsSimulator(mean, second_moment)
        with pytest.raises(ValueError, match="Dense expectation"):
            RocQuantumRuntime(sim).expectation_matrix_moments(matrix, [0])
        assert len(sim.calls) == 1

    class _BadDenseBatchExpectationSimulator:
        def __init__(self, values):
            self.values = values
            self.calls = []

        def batch_size(self):
            return 2

        def expectation_matrix_batch(self, matrix_arg, targets):
            self.calls.append((np.asarray(matrix_arg, dtype=np.complex128).copy(), tuple(targets)))
            return self.values

    for values in invalid_vectors:
        sim = _BadDenseBatchExpectationSimulator(values)
        with pytest.raises(ValueError, match="Batched dense"):
            RocQuantumRuntime(sim).expectation_matrix_batch(matrix, [0])
        assert len(sim.calls) == 1

    class _BadDenseBatchMomentsSimulator:
        def __init__(self, means, second_moments):
            self.means = means
            self.second_moments = second_moments
            self.calls = []

        def batch_size(self):
            return 2

        def expectation_matrix_moments_batch(self, matrix_arg, targets):
            self.calls.append((np.asarray(matrix_arg, dtype=np.complex128).copy(), tuple(targets)))
            return self.means, self.second_moments

    for means, second_moments in (([0.5, np.nan], [1.0, 1.0]), ([0.5, -0.5], [1.0])):
        sim = _BadDenseBatchMomentsSimulator(means, second_moments)
        with pytest.raises(ValueError, match="Batched dense"):
            RocQuantumRuntime(sim).expectation_matrix_moments_batch(matrix, [0])
        assert len(sim.calls) == 1

    class _BadSparseMomentsSimulator:
        def __init__(self, mean, second_moment):
            self.mean = mean
            self.second_moment = second_moment
            self.calls = []

        def num_qubits(self):
            return 1

        def batch_size(self):
            return 1

        def sparse_hamiltonian_moments(self, data, indices, indptr, shape):
            self.calls.append((np.asarray(data, dtype=np.complex128).copy(), tuple(shape)))
            return self.mean, self.second_moment

    for mean, second_moment in ((np.inf, 1.0), (0.5, "1.0")):
        sim = _BadSparseMomentsSimulator(mean, second_moment)
        with pytest.raises(ValueError, match="Sparse Hamiltonian"):
            RocQuantumRuntime(sim).sparse_hamiltonian_moments(
                sparse_data,
                sparse_indices,
                sparse_indptr,
                (2, 2),
            )
        assert len(sim.calls) == 1

    class _BadSparseBatchMomentsSimulator:
        def __init__(self, means, second_moments):
            self.means = means
            self.second_moments = second_moments
            self.calls = []

        def num_qubits(self):
            return 1

        def batch_size(self):
            return 2

        def sparse_hamiltonian_moments_batch(self, data, indices, indptr, shape):
            self.calls.append((np.asarray(data, dtype=np.complex128).copy(), tuple(shape)))
            return self.means, self.second_moments

    for means, second_moments in (([0.5, np.nan], [1.0, 1.0]), ([0.5, -0.5], [1.0])):
        sim = _BadSparseBatchMomentsSimulator(means, second_moments)
        with pytest.raises(ValueError, match="Batched sparse"):
            RocQuantumRuntime(sim).sparse_hamiltonian_moments_batch(
                sparse_data,
                sparse_indices,
                sparse_indptr,
                (2, 2),
            )
        assert len(sim.calls) == 1


def test_runtime_sparse_moments_fall_back_to_statevector():
    from rocquantum.framework_runtime import RocQuantumRuntime

    class _StateOnlySimulator:
        def __init__(self):
            self.statevector_reads = 0

        def batch_size(self):
            return 1

        def sparse_hamiltonian_moments(self, data, indices, indptr, shape):
            raise NotImplementedError("native sparse moments disabled")

        def get_statevector(self):
            self.statevector_reads += 1
            return np.array([1.0 / math.sqrt(2.0), 1.0 / math.sqrt(2.0)], dtype=np.complex128)

    simulator = _StateOnlySimulator()
    runtime = RocQuantumRuntime(simulator)

    mean, second_moment = runtime.sparse_hamiltonian_moments(
        np.array([1.0, 1.0], dtype=np.complex128),
        np.array([1, 0], dtype=np.int64),
        np.array([0, 1, 2], dtype=np.int64),
        (2, 2),
    )

    assert mean == pytest.approx(1.0)
    assert second_moment == pytest.approx(1.0)
    assert simulator.statevector_reads == 1


def test_runtime_sparse_moments_batch_falls_back_to_statevectors():
    from rocquantum.framework_runtime import RocQuantumRuntime

    class _BatchStateOnlySimulator:
        def __init__(self):
            self.statevector_reads = 0

        def batch_size(self):
            return 2

        def num_qubits(self):
            return 1

        def sparse_hamiltonian_moments_batch(self, data, indices, indptr, shape):
            raise NotImplementedError("native sparse moments batch disabled")

        def get_statevectors(self):
            self.statevector_reads += 1
            return np.array(
                [
                    [1.0, 0.0],
                    [0.0, 1.0],
                ],
                dtype=np.complex128,
            )

    simulator = _BatchStateOnlySimulator()
    runtime = RocQuantumRuntime(simulator)

    means, second_moments = runtime.sparse_hamiltonian_moments_batch(
        np.array([1.0, -1.0], dtype=np.complex128),
        np.array([0, 1], dtype=np.int64),
        np.array([0, 1, 2], dtype=np.int64),
        (2, 2),
    )

    np.testing.assert_allclose(means, np.array([1.0, -1.0], dtype=np.complex128))
    np.testing.assert_allclose(second_moments, np.array([1.0, 1.0], dtype=np.complex128))
    assert simulator.statevector_reads == 1


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
