# PennyLane-rocq: PennyLane device for the rocQuantum Simulator

## Overview

This package provides a PennyLane device that uses the [rocQuantum-1](https://github.com/ROCm/rocQuantum-1) C++/HIP library for high-performance quantum circuit simulation on AMD GPUs. It acts as a direct backend for PennyLane, allowing users to seamlessly offload quantum computations to a powerful, GPU-accelerated statevector simulator.

## Features

- **High-performance statevector simulation**: Leverages the rocQuantum C++ backend for fast and efficient calculations.
- **Full PennyLane Integration**: Supports core PennyLane measurement types, including `qml.state()`, `qml.probs()`, `qml.counts()`, Pauli / Projector `qml.expval()`, and Pauli / Projector `qml.var()`.
- **Hardware-like Sampling**: Simulates measurement shots to provide realistic count dictionaries.
- **Lightning-style Plugin Aliases**: Registers `lightning.rocq` and `lightning.rocm` in addition to `rocquantum.qpu` and the historical `rocq.pennylane` alias.
- **Native Probability Hook**: Analytic `qml.probs()` uses `QuantumSimulator.probabilities()` when a binding exposes it, with the existing cached statevector path kept as the compatibility fallback.
- **Initial State Preparation**: Supports initial `qml.BasisState` through native `X` gates and full-wire initial `qml.StatePrep` through native statevector upload, with partial state preparation kept on the matrix fallback path.
- **Sparse Hamiltonian Moments**: Prefers native CSR sparse-Hamiltonian moments through `rocsvGetSparseMatrixMoments` when available, with Python statevector CSR evaluation as a compatibility fallback that avoids densifying the observable.
- **Native Multi-control Aliases**: Dispatches `qml.MultiControlledX`, including open-control `control_values`, plus `qml.Toffoli` and `qml.CSWAP` through rocQuantum native `MCX` / `CSWAP` gate aliases when the binding supports them.
- **Native Controlled-Qubit-Unitary Dispatch**: Dispatches `qml.ControlledQubitUnitary` through `QuantumSimulator.apply_controlled_matrix()` when the binding exposes it, including open-control `control_values` via exact `X`-flip wrapping.
- **Native Controlled-Pauli Decomposition**: Dispatches `qml.CH`, `qml.CY`, and `qml.CCZ` through exact native `RY` / `S` / `SDG` / `H` / `CNOT` / `MCX` decompositions.
- **Native MultiRZ Decomposition**: Dispatches `qml.MultiRZ` as a CNOT-chain plus native `RZ`, avoiding dense matrix upload for common Pauli-rotation decompositions.
- **Native Controlled Rotation Decomposition**: Dispatches `qml.CRot` through PennyLane's exact native `RZ` / `RY` / `CNOT` decomposition instead of dense two-qubit matrix upload.
- **Native Phase Decomposition**: Dispatches `qml.PhaseShift`, `qml.ControlledPhaseShift`, and open-control `qml.CPhaseShift00/01/10` through exact global-phase plus native `RZ` / `CNOT` decompositions, preserving global phase for state outputs and skipping unobservable global-phase matrices for non-state measurements.
- **Native Ising Interaction Decomposition**: Dispatches `qml.IsingXX`, `qml.IsingYY`, `qml.IsingXY`, and `qml.IsingZZ` through exact native `RX` / `RY` / `RZ` / `H` / `CNOT` sequences instead of dense two-qubit matrices.
- **Native Swap Decomposition**: Dispatches `qml.ISWAP`, `qml.PSWAP`, `qml.SISWAP`, and `qml.SQISW` through exact native `S` / `H` / `SWAP` / `SX` / `RZ` / `CNOT` / phase decompositions instead of dense two-qubit matrices.
- **Native ECR Decomposition**: Dispatches `qml.ECR` through exact native `Z` / `CNOT` / `RX` / `RY` plus global-phase-preserving `SX` decomposition instead of a dense two-qubit matrix.
- **Native Excitation Decomposition**: Dispatches `qml.SingleExcitation`, `qml.SingleExcitationPlus`, `qml.SingleExcitationMinus`, `qml.DoubleExcitation`, `qml.DoubleExcitationPlus`, and `qml.DoubleExcitationMinus` through exact native `H` / `S` / `CNOT` / `CY` / `RY` / `MultiRZ` plus global-phase decompositions instead of dense two- and four-qubit matrices.
- **Native Fermionic Swap Decomposition**: Dispatches `qml.FermionicSWAP` through exact native `H` / `RX` / `RZ` / `CNOT` plus global-phase decomposition instead of a dense two-qubit matrix.
- **Native Orbital Rotation Decomposition**: Dispatches `qml.OrbitalRotation` through exact native `FermionicSWAP` and `SingleExcitation` decompositions instead of a dense four-qubit matrix.
- **GPU Matrix Fallbacks**: Dispatches remaining PennyLane matrix-defined gates such as explicit unitary operations through rocQuantum matrix application instead of rejecting them.

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
- The current device is a legacy `QubitDevice` adapter behind PennyLane's compatibility facade. It exposes `qml.state()`, full-wire and marginal `qml.probs()` through a native probability hook when available or cached statevector fallback otherwise, finite-shot `qml.sample()` / `qml.counts()` including shot vectors through `QuantumSimulator.measure()`, native analytic Pauli, Hadamard, computational-basis Projector, and Pauli-Hamiltonian expectation / variance through `QuantumSimulator.expectation_pauli_string()` without full statevector readback, analytic `qml.SparseHamiltonian` expectation / variance through native CSR moments when available or a statevector CSR fallback otherwise, direct native dispatch for `RX` / `RY` / `RZ`, `CRX` / `CRY` / `CRZ`, `qml.CRot`, `qml.MultiControlledX` including open controls, `qml.ControlledQubitUnitary` via native controlled-matrix dispatch when exposed, `qml.Toffoli`, `qml.CSWAP`, `qml.CH`, `qml.CY`, `qml.CCZ`, `qml.ECR`, `qml.ISWAP`, `qml.PSWAP`, `qml.SISWAP`, `qml.SQISW`, `qml.SingleExcitation`, `qml.SingleExcitationPlus`, `qml.SingleExcitationMinus`, `qml.DoubleExcitation`, `qml.DoubleExcitationPlus`, `qml.DoubleExcitationMinus`, `qml.FermionicSWAP`, `qml.OrbitalRotation`, `qml.MultiRZ` and `qml.IsingXX/YY/XY/ZZ` via native CNOT/rotation decompositions, `qml.PhaseShift` / controlled-phase variants via exact phase decompositions with measurement-aware global-phase elision, `Rot` decomposition, initial `qml.BasisState`, full-wire initial `qml.StatePrep` via native statevector upload, and matrix fallback dispatch for remaining common one-, two-, and three-/four-qubit PennyLane gates, with statevector fallbacks for older bindings and arbitrary Hermitian observables.
- `QuantumSimulator.sparse_hamiltonian_moments()` uploads CSR buffers and calls `rocsvGetSparseMatrixMoments` for a native HIP row-wise sparse moments reduction on local single-state vectors. Distributed/batched sparse moments and AMD GPU performance validation are still pending.
- Full-wire initial `qml.StatePrep` uses `QuantumSimulator.set_statevector()` when available; partial state preparation still uses PennyLane's preparation matrix. Mid-circuit state overwrite is rejected until runtime-level non-unitary state reset semantics exist.
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
