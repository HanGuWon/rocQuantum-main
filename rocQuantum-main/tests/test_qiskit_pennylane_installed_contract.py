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

    def __init__(self, num_qubits):
        self._num_qubits = int(num_qubits)
        self._measured = False
        self.ops = []
        self.matrices = []
        self.measurements = []
        self.expectations = []
        self.statevector_reads = 0
        _FakeQuantumSimulator.instances.append(self)

    def reset(self):
        self.ops.clear()
        self.matrices.clear()
        self.measurements.clear()
        self.expectations.clear()
        self.statevector_reads = 0
        self._measured = False

    def num_qubits(self):
        return self._num_qubits

    def apply_gate(self, name, targets, params=None):
        self.ops.append((name, tuple(targets), tuple(params or ())))

    def apply_matrix(self, matrix, targets):
        self.matrices.append((np.asarray(matrix, dtype=np.complex128), tuple(targets)))

    def measure(self, qubits, shots):
        self.measurements.append((tuple(qubits), int(shots)))
        self._measured = True
        if self._num_qubits == 1 and len(qubits) == 1:
            return [0 for _ in range(int(shots))]
        high = (1 << len(qubits)) - 1
        return [0 if shot % 2 == 0 else high for shot in range(int(shots))]

    def get_statevector(self):
        self.statevector_reads += 1
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

    def expectation_pauli_string(self, pauli_string, targets):
        self.expectations.append((pauli_string, tuple(targets)))
        if pauli_string == "Z" and tuple(targets) == (0,):
            return 0.5
        if pauli_string == "XZ" and tuple(targets) == (0, 1):
            return 0.25
        if pauli_string == "XX" and tuple(targets) == (0, 1):
            return 0.25
        return 0.0

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

    state = backend.run(circuit, shots=4).result().get_statevector()
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

    result = backend.run(circuit, sampling=False).result()
    state = result.get_statevector()

    expected = np.array([1.0 / math.sqrt(2.0), 0.0, 0.0, 1.0 / math.sqrt(2.0)])
    np.testing.assert_allclose(state, expected)
    assert result.results[0].shots == 0
    assert _FakeQuantumSimulator.instances[-1].measurements == []
    assert _FakeQuantumSimulator.instances[-1].statevector_reads == 1


def test_qiskit_backend_rejects_mid_circuit_measurement(monkeypatch):
    pytest.importorskip("qiskit")
    _install_fake_binding(monkeypatch)

    from qiskit import QuantumCircuit
    from qiskit_rocquantum_provider import RocQuantumProvider

    backend = RocQuantumProvider().get_backend("rocq_simulator")
    circuit = QuantumCircuit(1, 1)
    circuit.measure(0, 0)
    circuit.x(0)

    with pytest.raises(ValueError, match="terminal measurements"):
        backend.run(circuit).result()


def test_qiskit_backend_applies_global_phase_for_statevector(monkeypatch):
    pytest.importorskip("qiskit")
    _install_fake_binding(monkeypatch)

    from qiskit import QuantumCircuit
    from qiskit_rocquantum_provider import RocQuantumProvider

    backend = RocQuantumProvider().get_backend("rocq_simulator")
    circuit = QuantumCircuit(1)
    circuit.global_phase = math.pi / 3
    circuit.h(0)

    backend.run(circuit, sampling=False).result()

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


def test_qiskit_target_supports_transpile_and_matrix_fallback_gates(monkeypatch):
    pytest.importorskip("qiskit")
    _install_fake_binding(monkeypatch)

    from qiskit import QuantumCircuit, transpile
    from qiskit_rocquantum_provider import RocQuantumProvider

    backend = RocQuantumProvider().get_backend("rocq_simulator")
    circuit = QuantumCircuit(2)
    circuit.sx(0)
    circuit.p(0.2, 0)
    circuit.cp(0.3, 0, 1)
    circuit.rzz(0.4, 0, 1)

    assert backend.target.num_qubits >= 2
    assert {"sx", "p", "cp", "rzz", "u"}.issubset(set(backend.target.operation_names))

    transpiled = transpile(circuit, backend)
    assert transpiled.num_qubits == 2

    backend.run(circuit, shots=1).result()

    assert len(_FakeQuantumSimulator.instances[-1].matrices) == 4


def test_qiskit_matrix_fallback_covers_controlled_and_multi_qubit_gates(monkeypatch):
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

    assert len(_FakeQuantumSimulator.instances[-1].matrices) == 10


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
    assert [result.data.c[idx].get_counts() for idx in range(2)] == [{"0": 3}, {"0": 3}]
    assert _FakeQuantumSimulator.instances[-1].ops == [("RY", (0,), (0.2,))]


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
    assert _FakeQuantumSimulator.instances[-1].ops == [("RY", (0,), (0.2,))]


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
    assert _FakeQuantumSimulator.instances[-1].expectations == [("Z", (0,))]


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
    assert _FakeQuantumSimulator.instances[-1].expectations == [("Z", (0,))]


def test_pennylane_non_pauli_observables_use_statevector_fallback(monkeypatch):
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

    @qml.qnode(dev)
    def projector_circuit():
        return qml.expval(qml.Projector([0], wires=0))

    assert hermitian_circuit() == pytest.approx(1.0)
    assert projector_circuit() == pytest.approx(1.0)
    assert _FakeQuantumSimulator.instances[-1].expectations == []


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


def test_pennylane_common_gates_use_matrix_fallback_without_decomposition(monkeypatch):
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
        qml.CRot(0.6, 0.7, 0.8, wires=[0, 1])
        qml.Toffoli(wires=[0, 1, 2])
        return qml.state()

    circuit()
    sim = _FakeQuantumSimulator.instances[-1]

    assert sim.ops == []
    assert [targets for _, targets in sim.matrices] == [
        (0,),
        (0, 1),
        (0, 1),
        (0, 1),
        (0, 1),
        (0, 1),
        (0, 1, 2),
    ]
    assert [matrix.shape for matrix, _ in sim.matrices] == [
        (2, 2),
        (4, 4),
        (4, 4),
        (4, 4),
        (4, 4),
        (4, 4),
        (8, 8),
    ]


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


def test_pennylane_extended_gates_use_matrix_fallback(monkeypatch):
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

    assert sim.ops == []
    assert [targets for _, targets in sim.matrices] == [
        (0, 1),
        (0, 1),
        (0, 1, 2),
        (0, 1),
        (0, 1),
        (0, 1),
        (0, 1, 2),
        (0, 1, 2),
        (0, 1),
        (0, 1),
        (0, 1),
        (0, 1),
        (0, 1),
        (0, 1),
        (0, 1, 2, 3),
        (0, 1, 2, 3),
        (0, 1),
    ]


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
    assert _FakeQuantumSimulator.instances[-1].measurements == [((0, 1), 4)]

    with pytest.warns(PennyLaneDeprecationWarning):
        counts_dev = qml.device("lightning.rocq", wires=2, shots=4)

    @qml.qnode(counts_dev)
    def counts_circuit():
        qml.Hadamard(wires=0)
        qml.CNOT(wires=[0, 1])
        return qml.counts(wires=[0, 1])

    counts = {str(key): int(value) for key, value in counts_circuit().items()}

    assert counts == {"00": 2, "11": 2}
    assert _FakeQuantumSimulator.instances[-1].measurements == [((0, 1), 4)]


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
    assert _FakeQuantumSimulator.instances[-1].expectations == [
        ("Z", (0,)),
        ("XZ", (0, 1)),
    ]


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
    assert _FakeQuantumSimulator.instances[-1].expectations == [("Z", (0,))]


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
    assert _FakeQuantumSimulator.instances[-1].expectations == [
        ("Z", (0,)),
        ("XZ", (0, 1)),
        ("Z", (0,)),
        ("XZ", (0, 1)),
    ]


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
