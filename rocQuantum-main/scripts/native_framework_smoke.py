#!/usr/bin/env python3
"""Run a tiny native ROCm framework-integration smoke suite."""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INTEGRATIONS = PROJECT_ROOT / "integrations"
for path in [
    PROJECT_ROOT,
    INTEGRATIONS / "pennylane-rocq",
    INTEGRATIONS / "qiskit-rocquantum-provider",
    INTEGRATIONS / "cirq-rocm",
]:
    sys.path.insert(0, str(path))


def _require_rocm_device() -> None:
    if os.environ.get("ROCQ_NATIVE_SMOKE_ASSUME_ROCM_DEVICE", "").strip().lower() in {"1", "true", "yes", "on"}:
        return
    if not Path("/dev/kfd").exists():
        raise RuntimeError("Native framework smoke requires a ROCm runtime device at /dev/kfd.")


def _assert_bell_state(statevector: object) -> None:
    state = np.asarray(statevector, dtype=np.complex128).reshape(-1)
    expected = np.array([1 / np.sqrt(2), 0, 0, 1 / np.sqrt(2)], dtype=np.complex128)
    if state.shape != expected.shape or not np.allclose(state, expected, atol=1e-5):
        raise AssertionError(f"Bell state mismatch: got {state!r}")


def smoke_native_binding() -> None:
    binding = importlib.import_module("rocquantum_bind")
    simulator = binding.QuantumSimulator(2)
    simulator.apply_gate("H", [0], [])
    simulator.apply_gate("CNOT", [0, 1], [])
    _assert_bell_state(simulator.get_statevector())
    print("native binding smoke: ok")


def smoke_pennylane() -> None:
    qml = importlib.import_module("pennylane")
    pennylane_rocq = importlib.import_module("pennylane_rocq")

    dev = pennylane_rocq.RocqDevice(wires=2, shots=None)

    @qml.qnode(dev)
    def bell_state():
        qml.Hadamard(wires=0)
        qml.CNOT(wires=[0, 1])
        return qml.state()

    _assert_bell_state(bell_state())
    print("PennyLane smoke: ok")


def smoke_qiskit() -> None:
    qiskit = importlib.import_module("qiskit")
    provider_module = importlib.import_module("qiskit_rocquantum_provider.provider")

    circuit = qiskit.QuantumCircuit(2, 2)
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.measure([0, 1], [0, 1])

    backend = provider_module.RocQuantumProvider().get_backend("rocq_simulator")
    result = backend.run(circuit, shots=32, sampling=True).result()
    counts = result.get_counts()
    if sum(int(value) for value in counts.values()) != 32:
        raise AssertionError(f"Qiskit smoke returned invalid shot count: {counts!r}")
    print("Qiskit smoke: ok")


def smoke_cirq() -> None:
    cirq = importlib.import_module("cirq")
    simulator_module = importlib.import_module("cirq_rocm")

    q0, q1 = cirq.LineQubit.range(2)
    circuit = cirq.Circuit(cirq.H(q0), cirq.CNOT(q0, q1))
    simulator = simulator_module.RocQuantumSimulator()
    result = simulator.simulate(circuit)
    _assert_bell_state(result.final_state_vector)
    print("Cirq smoke: ok")


def main() -> int:
    _require_rocm_device()
    smoke_native_binding()
    smoke_pennylane()
    smoke_qiskit()
    smoke_cirq()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
