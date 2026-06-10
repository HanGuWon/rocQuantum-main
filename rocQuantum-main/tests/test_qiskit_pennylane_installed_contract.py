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
        _FakeQuantumSimulator.instances.append(self)

    def reset(self):
        self.ops.clear()
        self.matrices.clear()
        self.measurements.clear()
        self.expectations.clear()
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


def test_qiskit_provider_exposes_backend_primitives(monkeypatch):
    pytest.importorskip("qiskit")
    _install_fake_binding(monkeypatch)

    from qiskit.primitives import BackendEstimatorV2, BackendSamplerV2
    from qiskit_rocquantum_provider import RocQuantumEstimator, RocQuantumProvider

    provider = RocQuantumProvider()

    assert isinstance(provider.get_sampler(), BackendSamplerV2)
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
        ("YZ", (0, 1)),
        ("Z", (0,)),
        ("YZ", (0, 1)),
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
