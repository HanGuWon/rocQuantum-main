# Qiskit-rocQuantum-Provider

## Overview

This package provides a Qiskit Provider that allows users to run quantum circuits on the [rocQuantum-1](https://github.com/ROCm/rocQuantum-1) high-performance simulator. It follows the modern `BackendV2` provider plugin architecture, ensuring stability and compatibility with the latest Qiskit features. This provider enables Qiskit users to seamlessly accelerate their simulations using AMD GPUs.

## Features

- **Fully Compliant `BackendV2` Interface**: A modern, robust, and maintainable provider implementation.
- **Statevector Simulation**: Returns the pre-sampling statevector in the Qiskit `Result` data when `statevector=True` or a `save_statevector` marker is requested.
- **Measurement Sampling**: Supports realistic measurement outcomes and provides a counts dictionary via `get_counts()`.
- **Native Controlled Rotations**: Exposes Qiskit `crx`, `cry`, `crz`, fixed multi-control `RX` / `RY` / `RZ` (`ccrx` / `c3rx` / `ccry` / `c3ry` / `ccrz` / `c3rz`), and controlled `R` (`cr` / `ccr` / `c3r`) target operations, dispatching axis rotations to rocQuantum native controlled gates and lowering controlled `R(theta, phi)` plus multi-control rotations through exact native decompositions.
- **Native Multi-control Gates**: Dispatches Qiskit `ccx`, `mcx`, and `cswap` through rocQuantum native `MCX` / `CSWAP` gate aliases, and lowers fixed multi-control `SWAP` (`ccswap` / `c3swap`) through exact native `MCX` swap decompositions, while preserving matrix fallback where Qiskit exposes a dense matrix.
- **Native Open-Control Decomposition**: Dispatches direct open-control Qiskit controlled-X/Y/Z, controlled rotations/phase, and single-control controlled-H operations through exact `x`-flip plus native `cx` / `mcx` / `cy` / `cz` / `ccz` / `crx` / `cry` / `crz` / phase `rz` / `cx` / `ch` decompositions instead of dense matrices.
- **Native Controlled-Matrix Dispatch**: Dispatches generic Qiskit controlled base unitaries through `QuantumSimulator.apply_controlled_matrix()` when the binding exposes it, avoiding full dense controlled-matrix upload for that path.
- **Native Controlled-Pauli Decomposition**: Dispatches Qiskit `ch`, `cy`, `ccz`, `dcx`, controlled `DCX` (`cdcx` / `ccdcx` / `c3dcx`), `ecr`, controlled `ECR` (`cecr` / `ccecr` / `c3ecr`), and fixed multi-control `H` / `Y` / `Z` gates through exact native `rx` / `ry` / `s` / `sdg` / `h` / `x` / `cx` / `mcx` decompositions.
- **Native Controlled-U Family Decomposition**: Dispatches Qiskit `cs`, `csdg`, fixed multi-control `S` / `Sdg` / `T` / `Tdg`, `csx`, fixed multi-control `SX` (`ccsx` / `c3sx`), `cu1`, legacy multi-control `u1` (`mcu1`), controlled `u2` / `u3` (`cu2` / `ccu2` / `c3u2` / `cu3` / `ccu3` / `c3u3`), and controlled `u` (`cu` / `ccu` / `c3u`) through exact native `p` / `cp` / `crx` / `cx` / `u` / multi-control-phase decompositions instead of dense controlled matrices; `cu1` / `mcu1` / `cu2` / `ccu2` / `c3u2` / `cu3` / `ccu3` / `c3u3` / `cu` / `ccu` / `c3u` parameter sweeps stay in the batched simulator path.
- **Native Phase Decomposition**: Dispatches Qiskit `p` and `cp` through exact global-phase plus native `rz` / `cx` decompositions, skipping physically irrelevant global phase work on sampling-only and estimator paths.
- **Native Multi-Control Phase Decomposition**: Dispatches Qiskit `MCPhaseGate` through exact native `MultiRZ` / `RZ` / `CX` projector decompositions instead of controlled-matrix upload; compatible phase sweeps stay in the batched simulator path.
- **Global Phase Handling**: Dispatches Qiskit `GlobalPhaseGate` without empty-target matrix upload, preserving it as a single-wire phase identity only for statevector-producing runs and eliding unobservable parameter sweeps on sampling / estimator paths; controlled global phase (`cglobal_phase` / `ccglobal_phase` / `c3global_phase`) lowers through native `P` / multi-control phase decompositions with compatible sweeps batched.
- **Native Pauli Rotation Decomposition**: Dispatches Qiskit `rxx`, `ryy`, `rzz`, `rzx`, and fixed controlled variants (`crxx` / `ccrxx` / `c3rxx`, `cryy` / `ccryy` / `c3ryy`, `crzz` / `ccrzz` / `c3rzz`, `crzx` / `ccrzx` / `c3rzx`) through exact native basis-change / `MultiRZ` decompositions instead of dense two-qubit matrices or dense controlled matrices.
- **Native Pauli Evolution Decomposition**: Dispatches direct `PauliEvolutionGate` operations for single Pauli strings and commuting Pauli sums through exact native rotation / `MultiRZ` decompositions instead of dense matrix fallback.
- **Native Swap Decomposition**: Dispatches Qiskit `iswap` and fixed controlled `iSWAP` (`ciswap` / `cciswap` / `c3iswap`) through exact native phase / Hadamard / `MCX` gate decompositions instead of dense matrix or controlled-matrix upload.
- **Native Relative-Phase Control Decomposition**: Dispatches Qiskit `rccx` and `rcccx` through exact native `h` / `rz` / `cx` decompositions instead of dense three- and four-qubit matrices.
- **Native Single-Qubit Decomposition**: Dispatches Qiskit `sx`, `sxdg`, `tdg`, `u`, `u1`, `u2`, and `u3` through exact native phase / rotation sequences, preserving global phase only for statevector-producing runs; `u1` / `u2` / `u3` parameter sweeps stay in the batched simulator path.
- **Direct Unitary Fallbacks**: For direct `backend.run()` execution, small Qiskit unitary instructions outside the advertised target can fall back through `to_matrix()` / `Operator` when Qiskit can produce a dense matrix.
- **State Preparation Boundary**: Uploads full-wire initial `QuantumCircuit.prepare_state()` / `initialize()` vectors through `QuantumSimulator.set_statevector()` and keeps partial state preparation on the matrix fallback path; later `initialize()` is rejected because state overwrite after prior operations still needs explicit runtime semantics.
- **Runtime Reset Sampling**: Executes Qiskit `reset` after prior operations through `QuantumSimulator.reset_qubit()` in a shot-by-shot sampling path, preserving stochastic reset semantics without pretending there is one final pure statevector.
- **Basic Dynamic Sampling**: Executes simple Qiskit `if_test` / `if_else` conditioned blocks, finite `for_loop` blocks including loop-parameter binding, bounded `while_loop` blocks, `break_loop` / `continue_loop` inside supported loops, and `switch_case` blocks in `backend.run(..., sampling=True)` by using a state-collapsing `QuantumSimulator.measure_qubit()` trajectory per shot. Statevector and estimator outputs still reject dynamic circuits.
- **Automatic Discovery**: Once installed, Qiskit can automatically discover and list this provider's backends.
- **Modern Job Contract**: `backend.run()` returns a synchronous Qiskit `Job` object whose `result()` method returns the `Result`.
- **Primitive Factories**: `RocQuantumProvider.get_sampler()` and `get_estimator()` return native rocQuantum `SamplerV2` / `EstimatorV2` implementations that avoid generic backend wrapper overhead on the default path.
- **Native Expectations**: `RocQuantumProvider.estimate_expectation()` evaluates Qiskit `SparsePauliOp`/`Pauli` observables through the rocQuantum Pauli-string expectation path and dense `Operator` observables through the native dense-matrix expectation path when the operator spans all circuit qubits, the default low-index subset implied by compact dimensions, or a dimension-tagged subset; single dense expectations fall back inside the shared runtime to statevector correctness when an older binding lacks the dense hook. The shared runtime also exposes batched dense-matrix expectations for adapter paths that can supply dense observable plans directly.
- **Estimator Observable Batching**: `RocQuantumEstimator` applies each bound circuit once per parameter point and caches canonical duplicate observables while evaluating broadcasted observables on that state; simple parameter sweeps over Pauli observables use rocQuantum's batch state and batch expectation path instead of replaying the circuit for each value, including full-wire initial `prepare_state()` / `initialize()` uploads, fixed native `PauliGate` plus fixed `unitary` / generic controlled-unitary operations, native `p` / `cp` phase sweeps, open-control controlled rotation/phase sweeps such as `crx_o0` / `cp_o0`, batched native-rotation `u` / `r`, controlled-`r`, multi-control `RX` / `RY` / `RZ` / `R`, and controlled two-qubit Pauli-rotation decompositions, decomposition-backed `rxx`, `ryy`, `rzz`, `rzx`, `xx_plus_yy`, and `xx_minus_yy` sweeps, `PauliEvolutionGate` time sweeps for supported Pauli strings / commuting sums, and fixed native decompositions such as `sx` / `ch` / multi-control `S` / `T` / `SX` that appear alongside varying parameters. State preparation after earlier operations stays on the per-parameter matrix fallback path and is not treated as an initial batch upload.

## Installation

### Prerequisites

Before installing this package, you must have the core `rocquantum` Python package importable and the native Python binding (`rocquantum_bind`) built and installed. Build rocQuantum with `ROCQUANTUM_BUILD_BINDINGS=ON` on a ROCm host.

### Installation Steps

Once the prerequisites are met, you can install this package from the root directory using pip:

```bash
pip install .
```

After installation, Qiskit will automatically discover the `rocq_simulator` backend.

## Compatibility Notes

- Verified in adapter tests against Qiskit `2.4.1`.
- The provider uses `BackendV2`, exposes `max_circuits`, and imports result model classes from both old and new Qiskit locations.
- The Qiskit target declares a finite compiler capacity (`max_target_qubits`, default `64`) so `transpile(circuit, backend)` can compile local simulator circuits.
- The target includes `crx`, `cry`, `crz`, fixed `ccrx` / `c3rx` / `ccry` / `c3ry` / `ccrz` / `c3rz`, and `cr` / `ccr` / `c3r`; older bindings can still fall back through matrix application when available.
- `ccx`, general all-one-control `mcx`, and `cswap` are routed to native multi-control public simulator gates when available. `ch`, `cy`, `ccz`, fixed `cch` / `ccy` / `c3h` / `c3y` / `c3z`, fixed `ccswap` / `c3swap`, fixed `cdcx` / `ccdcx` / `c3dcx`, fixed `cecr` / `ccecr` / `c3ecr`, fixed `ciswap` / `cciswap` / `c3iswap`, fixed `ccs` / `c3s` / `ccsdg` / `c3sdg` / `ct` / `cct` / `c3t` / `ctdg` / `cctdg` / `c3tdg` / `ccsx` / `c3sx`, `dcx`, `ecr`, `iswap`, `rccx`, `rcccx`, `MCPhaseGate`, and the controlled-U family `cs` / `csdg` / `csx` / `cu1` / `mcu1` / `cu2` / `ccu2` / `c3u2` / `cu3` / `ccu3` / `c3u3` / `cu` / `ccu` / `c3u` use exact native decompositions.
- Qiskit `sx`, `sxdg`, `tdg`, `u`, `u1`, `u2`, `u3`, `p`, `cp`, `global_phase`, `cglobal_phase`, `ccglobal_phase`, `c3global_phase`, `mcphase`, `cr`, `ccr`, `c3r`, `ccrx`, `c3rx`, `ccry`, `c3ry`, `ccrz`, `c3rz`, `crxx`, `ccrxx`, `c3rxx`, `cryy`, `ccryy`, `c3ryy`, `crzz`, `ccrzz`, `c3rzz`, `crzx`, `ccrzx`, `c3rzx`, `cs`, `ccs`, `c3s`, `csdg`, `ccsdg`, `c3sdg`, `ct`, `cct`, `c3t`, `ctdg`, `cctdg`, `c3tdg`, `csx`, `ccsx`, `c3sx`, `cu1`, `mcu1`, `cu2`, `ccu2`, `c3u2`, `cu3`, `ccu3`, `c3u3`, `cu`, `ccu`, `c3u`, `ccswap`, `c3swap`, `cdcx`, `ccdcx`, `c3dcx`, `cecr`, `ccecr`, `c3ecr`, `ciswap`, `cciswap`, `c3iswap`, `rxx`, `ryy`, `rzz`, and `rzx` are target-visible; `p` / `cp` prefer native `P` / `CP` simulator dispatch, while the others prefer exact native decompositions. Direct `PauliEvolutionGate` operations with single Pauli strings or commuting Pauli sums also prefer exact native rotation / `MultiRZ` decomposition. Statevector-producing fallback decompositions include required global phase, while sampling-only and estimator runs omit it because it cannot affect observed results.
- Direct execution can also matrix-dispatch non-target unitary instructions with up to four qubits when Qiskit can provide a dense matrix. Open-control controlled-X/Y/Z, controlled rotation/phase, and single-control controlled-H variants such as `ccx_o1` / `mcx_o5` / `cy_o0` / `cz_o0` / `ccz_o1` / `crx_o0` / `cp_o0` / `ch_o0` prefer exact native decomposition; generic controlled base unitaries prefer `apply_controlled_matrix()` when available. Larger, non-unitary, or unsupported controlled-unitary instructions remain unsupported unless added explicitly.
- Direct `unitary` and partial `state_preparation` operations execute through `apply_matrix()` without attempting to normalize matrix/vector parameters as scalar gate angles; full-wire initial state preparation prefers native statevector upload.
- `reset` is target-visible. Initial reset remains a no-op from the all-zero state; reset after prior operations runs through `QuantumSimulator.reset_qubit()` only on `backend.run(..., sampling=True)`, re-executing the circuit per shot so stochastic reset is not collapsed into one shared final state. Statevector output and estimator expectation for runtime-reset circuits are rejected explicitly.
- Qiskit `if_test` / `if_else` blocks with tuple-style classical conditions, finite `for_loop` blocks with numeric loop-parameter binding, bounded `while_loop` blocks with tuple-style classical conditions, `break_loop` / `continue_loop` inside supported loops, and `switch_case` blocks over classical bits/registers can run in the same shot-trajectory sampling mode after prior measurements. `while_loop` uses the `max_dynamic_loop_iterations` backend run option, defaulting to 1024, to reject non-terminating trajectories. Statevector output and estimator expectation for dynamic circuits are still rejected explicitly.
- Qiskit sampler support defaults to `RocQuantumSampler`, a native `SamplerV2` over `QuantumSimulator.measure()` and the same shot-trajectory runtime reset / dynamic-circuit path used by `backend.run()`; pass `max_dynamic_loop_iterations` to `get_sampler()` to bound native Sampler `while_loop` trajectories. Compatible parameter sweeps, including circuits with no-op initial `reset`, use the batched simulator plus `measure_batch()` instead of replaying each bound circuit. Estimator support defaults to `RocQuantumEstimator`, a native deterministic `EstimatorV2` over Qiskit-supported Pauli-style observables through `QuantumSimulator.expectation_pauli_string()`. Pass `native=False` to `get_sampler()` / `get_estimator()` to request Qiskit's generic backend wrappers.
- `RocQuantumEstimator` groups broadcast entries by parameter index, so a pub with one bound circuit and many observables reuses the prepared simulator state instead of reapplying the circuit for each observable. For simple RX/RY/RZ/P/U/U1/U2/U3/R/CRX/CRY/CRZ/CR/CCR/C3R/CCRX/C3RX/CCRY/C3RY/CCRZ/C3RZ/CRXX/CCRXX/C3RXX/CRYY/CCRYY/C3RYY/CRZZ/CCRZZ/C3RZZ/CRZX/CCRZX/C3RZX/CP/CU1/MCU1/CU2/CCU2/C3U2/CU3/CCU3/C3U3/CU/CCU/C3U, compatible `MCPhaseGate` and controlled-global-phase sweeps, and observation-invisible `GlobalPhaseGate` sweeps, open-control controlled rotation/phase variants such as `crx_o0` / `cp_o0`, fixed native `PauliGate` plus fixed `unitary` / generic controlled-unitary operations, no-op initial `reset`, full-wire initial `prepare_state()` / `initialize()` plus parameter sweeps, decomposition-backed `rxx`/`ryy`/`rzz`/`rzx`/`xx_plus_yy`/`xx_minus_yy`, and supported `PauliEvolutionGate` time sweeps with Pauli-style or supported dense `Operator` observables, it can also allocate a batched simulator state and evaluate each broadcasted observable through `expectation_pauli_string_batch()` or `expectation_matrix_batch()`. Fixed native decompositions in the same circuit, including `sx` / `sxdg` / `tdg` / `ch` / `cy` / `ccz` / fixed multi-control `H` / `Y` / `Z` / `S` / `T` / `SX` / `SWAP` / `DCX` / `ECR` / `ISWAP` / `dcx` / `ecr` / `iswap` / `rccx` / `rcccx` / `cs` / `csdg` / `csx`, stay inside the batched circuit application instead of forcing matrix fallback or per-parameter replay; later `prepare_state()` stays outside the batch upload path. Dense `Operator` pubs bypass Qiskit's Pauli-only observable coercion inside the native provider path.
- Direct expectation support accepts `SparsePauliOp`, `Pauli`, Pauli label strings, `(label, coeff)` term lists, and dense `qiskit.quantum_info.Operator` observables. Dense `Operator` observables must have square power-of-two dimension no larger than `2^num_qubits`; smaller compact operators default to low-index qubits, matching the provider's padded short Pauli-label convention. Non-low-index partial placement is supported when the Qiskit `Operator` carries matching `input_dims()` / `output_dims()` metadata with `1` axes for idle circuit qubits.
- `backend.run()` defaults to sampling without pre-measurement statevector readback, avoiding a full host transfer on larger GPU simulations. Pass `statevector=True` or include a `save_statevector` marker when state output is needed.
- `backend.run(..., sampling=False)` skips measurement and counts formatting for statevector-only workloads.
- Aer-style `save_statevector` marker instructions are treated as no-op result annotations; `QuantumCircuit.save_statevector()` is not part of base Qiskit `2.4.1`.
- `rocquantum_bind` is loaded when a circuit is executed, so importing the provider remains possible before the native extension is present.
- On machines without an AMD GPU, use the installed-contract test with the fake native binding as the local validation baseline:

```bash
python -m pytest tests/test_qiskit_pennylane_installed_contract.py -q -p no:cacheprovider
```

## Usage Example

The following example demonstrates how to use the `RocQuantumBackend` to get both a final statevector and measurement counts for a GHZ state circuit.

```python
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit_rocquantum_provider import RocQuantumProvider

# 1. Instantiate the provider and get the backend
# Although Qiskit can discover the backend, direct instantiation is also clear.
provider = RocQuantumProvider()
backend = provider.get_backend("rocq_simulator")

# 2. Create a circuit and run for the statevector
qc_state = QuantumCircuit(3)
qc_state.h(0)
qc_state.cx(0, 1)
qc_state.cx(0, 2)

print("Running statevector simulation...")
job_state = backend.run(qc_state, statevector=True)
result_state = job_state.result()
final_state = result_state.get_statevector()

print("Final State Vector:")
print(np.around(final_state, 3))
print("-" * 20)


# 3. Create a similar circuit and run for measurement counts
shots = 2000
qc_counts = QuantumCircuit(3, 3)
qc_counts.h(0)
qc_counts.cx(0, 1)
qc_counts.cx(0, 2)
qc_counts.measure_all()

print(f"Running measurement simulation ({shots} shots)...")
job_counts = backend.run(qc_counts, shots=shots)
result_counts = job_counts.result()
counts = result_counts.get_counts()

print("Measurement Counts:")
print(counts)

sampler = provider.get_sampler()
estimator = provider.get_estimator()
backend_sampler = provider.get_sampler(native=False)
backend_estimator = provider.get_estimator(native=False)

qc_expectation = QuantumCircuit(3)
qc_expectation.h(0)
qc_expectation.cx(0, 1)
qc_expectation.cx(0, 2)
observable = SparsePauliOp.from_list([("ZZI", 1.0), ("IXX", -0.25)])
expectation = provider.estimate_expectation(qc_expectation, observable)


# Expected output:
# Running statevector simulation...
# Final State Vector:
# [0.707+0.j 0.   +0.j 0.   +0.j 0.   +0.j 0.   +0.j 0.   +0.j 0.   +0.j 0.707+0.j]
# --------------------
# Running measurement simulation (2000 shots)...
# Measurement Counts:
# {'000': 1005, '111': 995} (Note: Counts will vary slightly on each run)
```
