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
- **Native Basis/Permutation Templates**: Dispatches `qml.BasisEmbedding` through native `X` gates and `qml.Permute` through native `SWAP` sequences instead of dense permutation/state-preparation matrices.
- **Native Controlled Sequence Decomposition**: Dispatches `qml.ControlledSequence` for native single-qubit bases through controlled powers using `CRX` / `CRY` / `CRZ` / controlled-phase / controlled-Pauli decompositions, including batched rotation-angle sweeps.
- **Native Select Decomposition**: Dispatches full and partial `qml.Select` blocks through native controlled single-qubit operations, selected `BasisEmbedding` products used by `qml.QROM`, or the small controlled-matrix hook for selected target operators, with batched selected rotation angles when compatible.
- **Block-Encoding Template Coverage**: Verified PennyLane-expanded `qml.QROM`, signed/complex-coefficient `qml.PrepSelPrep`, and `qml.FABLE` template paths avoid dense full-template matrix fallback for the covered shapes by lowering to native `CNOT` / `MCX` / `CSWAP` / controlled-phase / rotation / swap sequences.
- **BlockEncode Matrix Dispatch**: Accepts dense `qml.BlockEncode` operations through rocQuantum matrix application and keeps fixed `BlockEncode` blocks inside compatible PennyLane batch executions. Sparse `BlockEncode` inputs prefer the public `QuantumSimulator.apply_sparse_matrix()` / `ApplySparseMatrix()` CSR hook, which routes local single-state, batched state vectors, and local-domain distributed slices through `rocsvApplySparseMatrix`; non-local distributed sparse apply follows the same explicit slow/debug distributed host fallback policy as dense distributed matrix fallbacks.
- **Sparse Hamiltonian Moments**: Prefers native CSR sparse-Hamiltonian moments through `rocsvGetSparseMatrixMoments` / `rocsvGetSparseMatrixMomentsBatch` when available, with Python statevector CSR evaluation as a compatibility fallback that avoids densifying the observable. Distributed sparse moments have an explicit slow/debug host fallback for full-state CSR observables.
- **Batched Hermitian Readouts**: Simple parameter sweeps over `qml.Hermitian` expval/variance can use `QuantumSimulator.expectation_matrix_batch()` and `expectation_matrix_moments_batch()` instead of replaying each circuit when the binding exposes the dense batch hooks. Larger dense Hermitian readouts and distributed dense-matrix expectation have explicit slow/debug host fallbacks.
- **Native Multi-control Aliases**: Dispatches `qml.MultiControlledX`, including open-control `control_values`, plus `qml.Toffoli` and `qml.CSWAP` through rocQuantum native `MCX` / `CSWAP` gate aliases when the binding supports them.
- **Native Arithmetic Template Decomposition**: Dispatches `qml.QubitSum` and `qml.QubitCarry` through exact native `CNOT` / `MCX` sequences instead of dense 3- and 4-qubit matrices.
- **Native Grover Diffuser Decomposition**: Dispatches `qml.GroverOperator` through exact native `H` / `Z` / open-control `MCX` / global-phase handling instead of dense diffuser matrices.
- **Native Controlled-Qubit-Unitary Dispatch**: Dispatches `qml.ControlledQubitUnitary` through `QuantumSimulator.apply_controlled_matrix()` when the binding exposes it, including open-control `control_values` via exact `X`-flip wrapping.
- **Native General Controlled Wrapper Dispatch**: Accepts PennyLane `qml.ctrl(...)` wrappers for multi-controlled single-qubit `RX` / `RY` / `RZ` / `PhaseShift` / `Rot` / `PauliX` / `PauliY` / `PauliZ` / `Hadamard` / `S` / `T` / adjoint phase-root bases, open-control phase variants (`C(CPhaseShift00/01/10)`), plus `SWAP` / `ISWAP` / `PSWAP` / `SISWAP` / `SQISW` / `ECR`, lowering parametric rotation/phase, Pauli/Hadamard/phase-root, open-control phase-variant, and swap-family/ECR cases through native `RZ` / `CNOT` / `MCX`-backed decompositions instead of PennyLane-expanded local matrix fallback; compatible wrappers stay inside `batch_execute`, including varying parametric sweeps.
- **Native Controlled-Pauli Decomposition**: Dispatches `qml.CH`, `qml.CY`, and `qml.CCZ` through exact native `RY` / `S` / `SDG` / `H` / `CNOT` / `MCX` decompositions.
- **Native Pauli Rotation Decomposition**: Dispatches `qml.MultiRZ`, `qml.PauliRot`, and `qml.SelectPauliRot` as basis changes plus CNOT/rotation sequences, avoiding dense matrix upload for common Pauli-rotation ansatz and uniformly controlled rotation blocks.
- **Native Diagonal-Unitary Decomposition**: Dispatches `qml.DiagonalQubitUnitary` through recursive global-phase / `RZ` / uniformly controlled-`RZ` decompositions, and batches varying diagonal phases by binding only the generated `RZ` angles.
- **Native Controlled Rotation Decomposition**: Dispatches `qml.CRot` through PennyLane's exact native `RZ` / `RY` / `CNOT` decomposition instead of dense two-qubit matrix upload.
- **Native Global Phase Handling**: Dispatches `qml.GlobalPhase` as a single-wire phase identity only when state/amplitude outputs need it, and skips it for measurement-only circuits and compatible batch sweeps.
- **Native Phase Decomposition**: Dispatches `qml.PhaseShift`, `qml.ControlledPhaseShift`, and open-control `qml.CPhaseShift00/01/10` through exact global-phase plus native `RZ` / `CNOT` decompositions, preserving global phase for state outputs and skipping unobservable global-phase matrices for non-state measurements.
- **Native QFT Decomposition**: Dispatches `qml.QFT` through exact native `H` / controlled-phase / `SWAP` decompositions instead of relying on dense template matrices or per-tape framework expansion.
- **Native Ising Interaction Decomposition**: Dispatches `qml.IsingXX`, `qml.IsingYY`, `qml.IsingXY`, and `qml.IsingZZ` through exact native `RX` / `RY` / `RZ` / `H` / `CNOT` sequences instead of dense two-qubit matrices.
- **Native Swap Decomposition**: Dispatches `qml.ISWAP`, `qml.PSWAP`, `qml.SISWAP`, and `qml.SQISW` through exact native `S` / `H` / `SWAP` / `SX` / `RZ` / `CNOT` / phase decompositions instead of dense two-qubit matrices.
- **Native ECR Decomposition**: Dispatches `qml.ECR` through exact native `Z` / `CNOT` / `RX` / `RY` plus global-phase-preserving `SX` decomposition instead of a dense two-qubit matrix.
- **Native Fixed-Gate Adjoint Payloads**: Lowers direct fixed `Adjoint(S)` / `Adjoint(T)` / `CH` / `CY` / `CCZ` / open-control `MultiControlledX` / `ISWAP` / `SISWAP` / `ECR` operations into primitive root-binding adjoint payloads, avoiding Python adjoint fallback when those fixed gates appear around supported trainable rotations.
- **Native Excitation Decomposition**: Dispatches `qml.SingleExcitation`, `qml.SingleExcitationPlus`, `qml.SingleExcitationMinus`, `qml.DoubleExcitation`, `qml.DoubleExcitationPlus`, and `qml.DoubleExcitationMinus` through exact native `H` / `S` / `CNOT` / `CY` / `RY` / `MultiRZ` plus global-phase decompositions instead of dense two- and four-qubit matrices.
- **Native Fermionic Swap Decomposition**: Dispatches `qml.FermionicSWAP` through exact native `H` / `RX` / `RZ` / `CNOT` plus global-phase decomposition instead of a dense two-qubit matrix.
- **Native Orbital Rotation Decomposition**: Dispatches `qml.OrbitalRotation` through exact native `FermionicSWAP` and `SingleExcitation` decompositions instead of a dense four-qubit matrix.
- **GPU Matrix Fallbacks**: Dispatches remaining PennyLane matrix-defined gates such as explicit unitary operations and dense `qml.BlockEncode` through rocQuantum matrix application instead of rejecting them.

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
- `lightning.rocq` / `lightning.rocm` are compatibility entry points intended to mirror the user experience of PennyLane Lightning devices on AMD GPUs. They currently share the rocQuantum `QubitDevice` adapter; device parameter-shift gradients can batch compatible shift tapes through native batch state paths, while full Lightning.GPU parity still requires native HIP adjoint differentiation and ROCm E2E performance validation.
- The current device is a legacy `QubitDevice` adapter behind PennyLane's compatibility facade. It exposes `qml.state()`, full-wire and marginal `qml.probs()` through a native probability hook when available or cached statevector fallback otherwise, finite-shot `qml.sample()` / `qml.counts()` / `qml.probs()` including shot vectors and batched `all_outcomes=True` counts through `QuantumSimulator.measure()` / `measure_batch()`, native analytic Pauli, Hadamard, computational-basis Projector, and Pauli-Hamiltonian expectation / variance through `QuantumSimulator.expectation_pauli_string()` without full statevector readback, analytic `qml.Hermitian` expectation / variance through dense expectation and moment hooks when available, analytic `qml.SparseHamiltonian` expectation / variance through native CSR moments, including simple batch parameter sweeps, when available or shared-runtime statevector CSR fallback otherwise, direct native dispatch for `RX` / `RY` / `RZ`, `CRX` / `CRY` / `CRZ`, `qml.CRot`, `qml.MultiControlledX` including open controls, general multi-control `qml.ctrl(...)` wrappers for parametric rotation/phase, Pauli/Hadamard/phase-root, open-control phase-variant, SWAP, iSWAP, PSWAP, SISWAP, SQISW, and ECR bases, `qml.ControlledQubitUnitary` via native controlled-matrix dispatch when exposed, dense `qml.BlockEncode` via matrix dispatch, sparse `qml.BlockEncode` via the public sparse-apply CSR hook when exposed, `qml.ControlledSequence` for native single-qubit bases, full and partial `qml.Select` blocks for supported selected operations including PennyLane-expanded `qml.QROM` selected `BasisEmbedding` products and signed/complex-coefficient `qml.PrepSelPrep` selected Pauli/global-phase products, `qml.Toffoli`, `qml.CSWAP`, `qml.QubitSum`, `qml.QubitCarry`, `qml.GroverOperator`, `qml.BasisEmbedding`, `qml.Permute`, `qml.CH`, `qml.CY`, `qml.CCZ`, `qml.ECR`, `qml.ISWAP`, `qml.PSWAP`, `qml.SISWAP`, `qml.SQISW`, `qml.SingleExcitation`, `qml.SingleExcitationPlus`, `qml.SingleExcitationMinus`, `qml.DoubleExcitation`, `qml.DoubleExcitationPlus`, `qml.DoubleExcitationMinus`, `qml.FermionicSWAP`, `qml.OrbitalRotation`, `qml.MultiRZ`, `qml.PauliRot`, `qml.SelectPauliRot`, `qml.DiagonalQubitUnitary`, `qml.GlobalPhase`, `qml.QFT`, and `qml.IsingXX/YY/XY/ZZ` via native CNOT/rotation/controlled-phase decompositions, `qml.PhaseShift` / controlled-phase variants via exact phase decompositions with measurement-aware global-phase elision, `Rot` decomposition, initial `qml.BasisState`, full-wire initial `qml.StatePrep` via native statevector upload, framework-expanded `qml.FABLE` via native rotations and swaps for tested shapes, and matrix fallback dispatch for remaining common one-, two-, and three-/four-qubit PennyLane gates, with shared-runtime statevector fallbacks for older dense/sparse expectation bindings, including single-read Hermitian variance fallback, and arbitrary Hermitian observables.
- `QuantumSimulator.sparse_hamiltonian_moments()` / `sparse_hamiltonian_moments_batch()` upload CSR buffers and call `rocsvGetSparseMatrixMoments` / `rocsvGetSparseMatrixMomentsBatch` for native HIP row-wise sparse moments reductions on local single-state and batched state vectors. Older or unsupported local bindings fall back through the shared runtime's statevector CSR moments path; distributed sparse moments are available only through the explicit slow/debug distributed host fallback; AMD GPU performance validation is still pending.
- Full-wire initial `qml.StatePrep` uses `QuantumSimulator.set_statevector()` when available; partial state preparation still uses PennyLane's preparation matrix. Mid-circuit state overwrite is rejected until runtime-level non-unitary state reset semantics exist.
- PennyLane parameter-shift gradients can run through the plugin for supported parametric gates and Pauli expectations. `diff_method="device"` exposes a device jacobian that generates parameter-shift tapes and executes compatible shifts through the same batched runtime path as `batch_execute`, including native rotations, native batched Pauli/Hermitian/SparseHamiltonian expectation hooks, and probability-vector Jacobians through `probabilities_batch()`. Simple `batch_execute` parameter sweeps use rocQuantum batch state paths for native rotations, controlled rotations, `Rot` / `CRot` / `ControlledSequence` / `Select` decompositions, `DiagonalQubitUnitary`, `SelectPauliRot`, `GlobalPhase` elision, `PhaseShift`, `ControlledPhaseShift`, open-control `CPhaseShift00/01/10`, multi-control parametric `qml.ctrl(...)` wrappers, `MultiRZ`, `PauliRot`, `IsingXX`/`IsingYY`/`IsingZZ`/`IsingXY`, `PSWAP`, `SingleExcitation` plus/minus variants, `DoubleExcitation` plus/minus variants, `FermionicSWAP`, and `OrbitalRotation` decompositions, including analytic circuits that return multiple Pauli expectation values, Hermitian expectation values, variances, or probability readouts and compatible finite-shot `sample` / `counts` / `probs` readouts, including repeated analytic readouts that reuse cached expectation/moment/probability payloads in single and batched executions, multiple readouts, and `all_outcomes=True` counts. Initial full-wire `StatePrep`, identical initial `BasisState` preparation, fixed `QubitUnitary` / `ControlledQubitUnitary`, fixed general `qml.ctrl(...)` Pauli/Hadamard/phase-root/open-control phase-variant/SWAP/iSWAP/PSWAP/SISWAP/SQISW/ECR wrappers, and fixed native decompositions in the same tapes, such as `QFT`, `QubitSum`, `QubitCarry`, `GroverOperator`, `BasisEmbedding`, `Permute`, `CH`, `CY`, `CCZ`, `ECR`, `ISWAP`, `SISWAP`, and `SQISW`, remain in the batched execution path when compatible across the batch. Explicit `diff_method="adjoint"` calls the root binding's adjoint hook for trainable RX/RY/RZ/P/CRX/CRY/CRZ/CP operations with Pauli-term, dense Hermitian, full-state CSR SparseHamiltonian, targeted partial CSR SparseHamiltonian observables, or fixed `QubitUnitary` / `ControlledQubitUnitary` / dense `BlockEncode` matrix operation payloads when available; `Rot` and `CRot` are lowered into primitive rotation payloads, `PhaseShift` / `ControlledPhaseShift` map to native phase payloads, open-control `CPhaseShift00/01/10` and covered `qml.ctrl(...)` wrappers for `RX` / `RY` / `RZ` / `PhaseShift` / `Rot` / `PauliX` / `PauliY` / `PauliZ` / `Hadamard` / `S` / `T` / `Adjoint(S)` / `Adjoint(T)` / `CPhaseShift00/01/10` / `SWAP` / `ISWAP` / `PSWAP` / `SISWAP` / `SQISW` / `ECR` / targetless `GlobalPhase` lower through primitive `RZ` / `CNOT` / `MCX` / `CR*` / `CP` / `CSWAP` payloads with open-control `X` wrapping, and decomposition-backed `MultiRZ`, `PauliRot`, `IsingXX/YY/ZZ/XY`, `SingleExcitation` plus/minus, `DoubleExcitation` plus/minus, `PSWAP`, `FermionicSWAP`, and `OrbitalRotation` lower through primitive payloads with explicit parameter-derivative scales for half-/eighth-angle gates. Operation payloads include global `trainable_param_indices`, operation-local `trainable_param_positions`, and optional `param_derivative_scales` for native kernels. Trainable matrix-parameter, remaining trainable controlled-matrix or multi-target controlled-wrapper payloads, and native GPU-resident adjoint differentiation remain pending. This reduces repeated Python-level fallback for the supported analytic expectation-value subset, but is not yet equivalent to Lightning.GPU's native GPU-resident adjoint differentiation path.
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
