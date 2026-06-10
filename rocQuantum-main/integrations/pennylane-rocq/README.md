# PennyLane-rocq: PennyLane device for the rocQuantum Simulator

## Overview

This package provides a PennyLane device that uses the [rocQuantum-1](https://github.com/ROCm/rocQuantum-1) C++/HIP library for high-performance quantum circuit simulation on AMD GPUs. It acts as a direct backend for PennyLane, allowing users to seamlessly offload quantum computations to a powerful, GPU-accelerated statevector simulator.

## Features

- **High-performance statevector simulation**: Leverages the rocQuantum C++ backend for fast and efficient calculations.
- **Full PennyLane Integration**: Supports core PennyLane measurement types, including `qml.state()`, `qml.probs()`, `qml.counts()`, Pauli `qml.expval()`, and Pauli `qml.var()`.
- **Hardware-like Sampling**: Simulates measurement shots to provide realistic count dictionaries.
- **Lightning-style Plugin Aliases**: Registers `lightning.rocq` and `lightning.rocm` in addition to `rocquantum.qpu` and the historical `rocq.pennylane` alias.
- **Initial Basis State Preparation**: Supports `qml.BasisState` as an initial state preparation by dispatching native `X` gates for one bits.
- **GPU Matrix Fallbacks**: Dispatches common PennyLane matrix-defined gates such as `PhaseShift`, controlled phase variants, `CH` / `CY` / `CCZ`, `MultiControlledX`, `MultiRZ`, Ising gates, swap-family gates, excitation gates, `OrbitalRotation`, and `FermionicSWAP` through rocQuantum matrix application instead of decomposing them into longer gate sequences.

## Installation

### Prerequisites

Before installing this package, you must have the core `rocquantum` Python package importable and the native Python binding (`rocquantum_bind`) built and installed. Build rocQuantum with `ROCQUANTUM_BUILD_BINDINGS=ON` on a ROCm host.

### Installation Steps

Once the prerequisites are met, you can install this package from the root directory using pip:

```bash
pip install .
```

PennyLane will automatically discover the `lightning.rocq`, `lightning.rocm`, `rocquantum.qpu`, and `rocq.pennylane` devices upon successful installation.

## Compatibility Notes

- Verified in adapter tests against PennyLane `0.45.0`.
- The package can be imported before `rocquantum_bind` is present; device creation raises a clear binding-install error.
- `lightning.rocq` / `lightning.rocm` are compatibility entry points intended to mirror the user experience of PennyLane Lightning devices on AMD GPUs. They currently share the rocQuantum `QubitDevice` adapter; full Lightning.GPU parity still requires native adjoint differentiation, batching, and ROCm E2E performance validation.
- The current device is a legacy `QubitDevice` adapter behind PennyLane's compatibility facade. It exposes `qml.state()`, full-wire and marginal `qml.probs()`, finite-shot `qml.sample()` / `qml.counts()` including shot vectors through `QuantumSimulator.measure()`, native analytic Pauli, Hadamard, and Pauli-Hamiltonian expectation / variance through `QuantumSimulator.expectation_pauli_string()` without full statevector readback, direct native dispatch for `RX` / `RY` / `RZ`, `CRX` / `CRY` / `CRZ`, `Rot` decomposition, initial `qml.BasisState`, and matrix fallback dispatch for common one-, two-, and three-qubit PennyLane gates, with statevector fallbacks for older bindings.
- General `qml.StatePrep` can run when PennyLane decomposes it into supported operations. A native arbitrary statevector setter is still not exposed by the rocQuantum binding, so full Lightning.GPU-style state-preparation parity remains a future runtime feature.
- PennyLane parameter-shift gradients can run through the plugin for supported parametric gates and Pauli expectations. This is not yet equivalent to Lightning.GPU's native adjoint differentiation path.
- On machines without an AMD GPU, use the fake-binding installed-contract test as the local validation baseline and skip ROCm hardware E2E/performance validation:

```bash
python -m pytest tests/test_qiskit_pennylane_installed_contract.py -q -p no:cacheprovider
```

## Usage Example

The following example demonstrates how to use the `rocquantum.qpu` device to simulate a Bell state circuit, first retrieving the final statevector and then getting measurement counts.

```python
import pennylane as qml
from pennylane import numpy as np

# 1. Load the rocQuantum device for statevector simulation
dev_state = qml.device("lightning.rocq", wires=2)

@qml.qnode(dev_state)
def bell_state_circuit_state():
    """A QNode that creates a Bell state and returns the final statevector."""
    qml.Hadamard(wires=0)
    qml.CNOT(wires=[0, 1])
    return qml.state()

# Run the QNode and print the statevector
final_state = bell_state_circuit_state()
print("Final State Vector:")
print(final_state)
print("-" * 20)

# 2. Load the rocQuantum device for measurement sampling
shots = 1000
dev_counts = qml.device("lightning.rocq", wires=2, shots=shots)

@qml.qnode(dev_counts)
def bell_state_circuit_counts():
    """A QNode that creates a Bell state and returns measurement counts."""
    qml.Hadamard(wires=0)
    qml.CNOT(wires=[0, 1])
    return qml.counts()

# Run the QNode and print the counts
counts = bell_state_circuit_counts()
print(f"Measurement Counts (for {shots} shots):")
print(counts)

# Expected output:
# Final State Vector:
# [0.70710678+0.j 0.        +0.j 0.        +0.j 0.70710678+0.j]
# --------------------
# Measurement Counts (for 1000 shots):
# {'00': 498, '11': 502}  (Note: Counts will vary slightly on each run)
```
