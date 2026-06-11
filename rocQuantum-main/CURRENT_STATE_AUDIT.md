# Current State Audit

Audit date: 2026-04-05

Runtime update note (2026-04-06):

- Canonical `rocq.execute()`, `rocq.sample()`, and `rocq.observe()` now route through a unified backend contract.
- `rocq.operator.get_expectation_value()` now delegates to `rocq.observe()`.
- Native state-vector sampling and Pauli expectation paths are wired into the canonical `rocq` surface.
- Packaging has moved to a CMake-first `scikit-build-core` path.

Audit refresh note (2026-06-10):

- Official AMD production ROCm documentation now points to ROCm `7.2.4`, released 2026-05-29.
- Previous rows that described canonical `rocq.operator.get_expectation_value()` as unimplemented are stale; the current gap is split API behavior between canonical `rocq` and legacy `python/rocq`.
- GateFusion is wired opportunistically into canonical `rocq.backends.StateVectorBackend` and legacy `python/rocq.Circuit.flush()` for supported CNOT-adjacent spans; broader fusion patterns remain missing.
- `rocqCompiler::MLIRCompiler::compile_and_execute()` is no longer a hard stub for the current MVP subset: qalloc, H/X/Y/Z/S/Sdg/T/Tdg, CNOT/CZ/SWAP/CCX/MCX/CSWAP, RX/RY/RZ/P, CRX/CRY/CRZ/CP.
- Qiskit and PennyLane adapters now avoid several avoidable full statevector readbacks: Qiskit `backend.run()` defaults to sampling without state output, Qiskit Estimator reuses a bound circuit across Qiskit-supported Pauli observable batches and caches canonical duplicates, Qiskit `RocQuantumProvider.estimate_expectation()` can evaluate supported full-circuit and low-index partial dense `Operator` observables through native dense-matrix expectation, Qiskit full-wire initial state preparation, Qiskit/PennyLane phase sweeps, Qiskit `u` / `r`, fixed native Qiskit `PauliGate` plus fixed Qiskit `unitary` / generic controlled-unitary operations, Qiskit open-control controlled rotation/phase sweeps, Qiskit `rxx` / `ryy` / `rzz` / `rzx` / `xx_plus_yy` / `xx_minus_yy` sweeps, and supported `PauliEvolutionGate` time sweeps plus PennyLane full-wire `StatePrep`, fixed `QubitUnitary` / `ControlledQubitUnitary`, `Rot` / `CRot` / `MultiRZ` / `PauliRot` / Ising / `PSWAP` / open-control phase / excitation plus-minus / fermionic-orbital sweeps can use batched native state or rotation paths, identical PennyLane `BasisState` initializers and fixed native decompositions can stay inside Qiskit/PennyLane batch circuit execution instead of forcing matrix fallback or per-parameter replay, PennyLane finite-shot measurements use native `measure()` where available, PennyLane `batch_execute` can batch Pauli/probability/Hermitian analytic readouts for simple parameter sweeps, and PennyLane analytic Pauli/Hadamard/Projector/Hamiltonian expectations skip diagonalizing rotations and use native Pauli-string expectations where they can be represented as Pauli terms.
- PennyLane analytic `qml.probs()` now reaches a shared `QuantumSimulator.probabilities()` runtime hook backed by `rocsvProbabilities` when native bindings are available, while keeping the cached statevector fallback for unsupported targets and legacy builds. Local-domain distributed probability requests can share the RCCL outcome-probability reduction used by distributed sampling.
- Public simulator and framework paths now expose native `MCX` / `CSWAP` dispatch for Qiskit `ccx` / `mcx` / `cswap` and PennyLane `MultiControlledX` / `Toffoli` / `CSWAP`; non-default PennyLane `control_values` use exact `X`-flip plus native-`MCX` decomposition instead of dense matrix fallback.
- `QuantumSimulator.set_statevector()` is exposed through the Python binding, and Qiskit/PennyLane full-wire initial state preparation now uses native statevector upload instead of state-preparation matrix fallback.
- Qiskit `ch` / `cy` / `ccz` / `dcx` and PennyLane `qml.CH` / `qml.CY` / `qml.CCZ` now use exact native controlled-Pauli decompositions instead of dense matrix dispatch.
- Qiskit `ecr` and PennyLane `qml.ECR` now use exact native decompositions instead of dense two-qubit matrix dispatch; PennyLane preserves the `SX` global phase with a one-qubit global-phase matrix.
- Qiskit `iswap` and PennyLane `qml.ISWAP` / `qml.PSWAP` / `qml.SISWAP` / `qml.SQISW` now use exact native swap/phase decompositions instead of dense two-qubit matrix dispatch.
- Qiskit `rccx` / `rcccx` now use exact native relative-phase `H` / `RZ` / `CNOT` decompositions instead of dense three- and four-qubit matrix dispatch.
- Qiskit `PauliGate` now uses exact native `X` / `Y` / `Z` dispatch instead of dense Pauli-string matrix dispatch, including when it appears as a fixed operation inside Estimator parameter batches.
- Direct Qiskit open-control controlled-X/Y/Z, controlled rotations/phase, and single-control controlled-H operations now use exact `X`-flip plus native `CX` / `MCX` / `CY` / `CZ` / `CCZ` / `CRX` / `CRY` / `CRZ` / phase `RZ` / `CX` / `CH` decomposition instead of dense matrix dispatch.
- Generic Qiskit controlled base unitaries can now use `QuantumSimulator.apply_controlled_matrix()` through the shared runtime, avoiding full dense controlled-matrix upload when the binding exposes the public controlled-matrix surface.
- Qiskit `sx` / `sxdg` / `tdg` / `u` / `r` now use exact native single-qubit rotation decompositions, including global phase only for statevector-producing runs.
- Qiskit `p` / `cp` now prefer native `P` / `CP` simulator dispatch and fall back to exact `rz` / `cx` decompositions only when the binding lacks those gates.
- Qiskit `rxx` / `ryy` / `rzz` / `rzx` / `xx_plus_yy` / `xx_minus_yy` and PennyLane `qml.IsingXX` / `qml.IsingYY` / `qml.IsingXY` / `qml.IsingZZ` now use exact native CNOT/rotation decompositions instead of dense two-qubit matrix dispatch.
- Direct Qiskit `PauliEvolutionGate` operations for single Pauli strings and commuting Pauli sums now use exact native rotation / `MultiRZ` decompositions instead of dense matrix dispatch.
- Qiskit `reset` after prior operations now runs through `QuantumSimulator.reset_qubit()` in the sampling path by re-executing the circuit per shot; simple `if_test` / `if_else`, finite `for_loop` with loop-parameter binding, bounded `while_loop`, loop-local `break_loop` / `continue_loop`, and `switch_case` dynamic sampling uses `QuantumSimulator.measure_qubit()` to collapse and record mid-circuit measurement bits per shot. Statevector and estimator paths still reject shot-trajectory circuits because there is no single final pure state trajectory.
- PennyLane `qml.ControlledQubitUnitary` can now use `QuantumSimulator.apply_controlled_matrix()` through the shared runtime, including open controls via exact `X`-flip wrapping, avoiding full dense controlled-matrix upload when the binding exposes the controlled-matrix surface.
- PennyLane `qml.SingleExcitation` and `qml.DoubleExcitation` now use exact native `H` / `CNOT` / `RY` decompositions instead of dense two- and four-qubit matrix dispatch.
- PennyLane `qml.SingleExcitationPlus` / `qml.SingleExcitationMinus` now use exact native decompositions with one-qubit global-phase matrices instead of dense two-qubit matrix dispatch.
- PennyLane `qml.DoubleExcitationPlus` / `qml.DoubleExcitationMinus` now use exact global-phase plus Z-string `MultiRZ` phase corrections and native `DoubleExcitation` decomposition instead of dense four-qubit matrix dispatch.
- PennyLane `qml.CRot` now uses PennyLane's exact native `RZ` / `RY` / `CNOT` decomposition instead of dense two-qubit matrix dispatch.
- PennyLane `qml.FermionicSWAP` now uses an exact native `H` / `RX` / `RZ` / `CNOT` plus global-phase decomposition instead of dense two-qubit matrix dispatch.
- PennyLane `qml.OrbitalRotation` now uses exact native `FermionicSWAP` and `SingleExcitation` decompositions instead of dense four-qubit matrix dispatch.
- PennyLane `qml.PhaseShift`, `qml.ControlledPhaseShift`, and open-control `qml.CPhaseShift00/01/10` now prefer native `P` / `CP` simulator dispatch and fall back to exact global-phase plus native `RZ` / `CNOT` decompositions only when needed, avoiding dense controlled-phase matrix upload in QFT-style circuits.
- PennyLane `qml.GlobalPhase` now avoids full-register dense matrix dispatch by using a single-wire phase identity for state/amplitude outputs and by eliding observation-invisible phase sweeps in compatible batched measurements.
- PennyLane `qml.DiagonalQubitUnitary` now decomposes to recursive global-phase / `RZ` / uniformly controlled-`RZ` sequences instead of dense diagonal matrix upload, and varying diagonal phase sweeps batch the generated `RZ` angles.
- PennyLane `qml.SelectPauliRot` now uses native uniformly controlled `RZ` plus `X` / `Y` basis changes instead of dense template matrices, and compatible angle sweeps batch only the generated `RZ` rotations.
- PennyLane `qml.QFT` is now accepted directly by the device and decomposed into native `H` / controlled-phase / `SWAP` operations, so QFT blocks can stay inside the batched execution path when surrounding parameters vary.
- PennyLane arithmetic templates `qml.QubitSum` and `qml.QubitCarry` now decompose to native `CNOT` / `MCX` sequences instead of dense 3- and 4-qubit matrices, including when fixed arithmetic blocks accompany batched parameter sweeps.
- PennyLane `qml.GroverOperator` now decomposes to native `H` / `Z` / open-control `MCX` plus measurement-aware global phase handling instead of dense diffuser matrices, including fixed diffuser blocks inside batched sweeps.
- PennyLane `qml.BasisEmbedding` and `qml.Permute` now decompose to native `X` and `SWAP` sequences instead of dense basis/permutation matrices, including fixed template blocks inside batched sweeps.
- PennyLane `qml.ControlledSequence` for native single-qubit bases now decomposes phase-estimation-style controlled powers into native controlled rotations, controlled phase, and controlled Pauli/Hadamard sequences, including batched base-angle sweeps.
- PennyLane full and partial `qml.Select` blocks now dispatch supported selected operations through native controlled gates or the small controlled-matrix hook instead of full-register dense select matrices, including batched selected single-qubit rotation angles.
- PennyLane-expanded `qml.QROM` selected `Prod(BasisEmbedding, ...)` loaders now lower to grouped open-control `CNOT` / `MCX` target batches and native `CSWAP` swap networks instead of dense full-select matrices; signed/complex-coefficient `qml.PrepSelPrep` selected Pauli/global-phase products lower through partial `Select` native controlled-Pauli and controlled-phase decompositions, and `qml.FABLE` is verified to follow PennyLane's native rotation/swap decomposition for covered shapes.
- PennyLane dense `qml.BlockEncode` is now accepted by the device and dispatched through rocQuantum matrix application instead of being rejected by the gate set; fixed dense `BlockEncode` blocks can remain inside compatible `batch_execute` parameter sweeps. Sparse `BlockEncode` inputs now avoid dense unitary materialization through `QuantumSimulator.apply_sparse_matrix()` / `ApplySparseMatrix()` when the public binding exposes it, routing local single-state and batched state vectors through `rocsvApplySparseMatrix` with Python statevector CSR fallback for older or unsupported bindings; distributed sparse apply and AMD GPU performance validation remain open.
- PennyLane phase-style fallbacks preserve global phase for `qml.state()` / amplitude-style outputs and skip those unobservable one-qubit global-phase matrices for expval/probs/sample/counts measurements.
- PennyLane computational-basis `qml.Projector` expectations now expand to Pauli-Z product terms and use native Pauli-string expectations instead of mandatory statevector fallback; `qml.Hermitian` expectation and variance can use dense `QuantumSimulator.expectation_matrix()` / `expectation_matrix_batch()` paths for supported target sizes instead of mandatory statevector fallback or per-parameter replay.
- PennyLane `qml.SparseHamiltonian` analytic expectation / variance now prefers `QuantumSimulator.sparse_hamiltonian_moments()` / `sparse_hamiltonian_moments_batch()`, which upload CSR buffers and call `rocsvGetSparseMatrixMoments` / `rocsvGetSparseMatrixMomentsBatch` for native HIP row-wise sparse moments reductions; it falls back to Python statevector CSR evaluation only when the hook is unavailable.
- Top-level CMake now delegates `_rocq_hip_backend` to `python/rocq/CMakeLists.txt`, so the legacy native Python backend path is part of the default bindings build instead of a parallel dormant CMake fragment.

The detailed findings below still describe the original audit snapshot; use them together with the runtime update above.

## Scope

- Authoritative project root: `c:\Users\한구원\Desktop\rocQuantum-main\rocQuantum-main`
- Outer-root control files kept in scope only when they affect the inner repo, most notably `.github/workflows/rocm-linux-build.yml`
- Ground truth comes from code, build scripts, bindings, tests, and runtime-facing APIs. README and roadmap claims were treated as non-authoritative unless backed by code.

## Executive Verdict

`rocQuantum-main` contains real ROCm-native simulator components, but it is not yet a credible ROCm-first counterpart to CUDA-Q, cuQuantum, and CUDA-QX as a unified product.

What is real today:

- Native HIP state-vector simulation exists in `rocquantum/src/hipStateVec`.
- Native tensor-network contraction exists in `rocquantum/src/hipTensorNet`.
- Native density-matrix kernels and a limited noise stack exist in `rocquantum/src/hipDensityMat`.
- A direct simulator runtime exists through `rocquantum::QuantumSimulator` and the top-level `rocq` Python execution path.

What is not real today:

- End-to-end compiler-driven execution parity with CUDA-Q is not present.
- `rocqCompiler::MLIRCompiler::compile_and_execute()` is only an MVP subset executor, not a CUDA-Q-style full compiler runtime.
- Distributed multi-GPU is only partially implemented and was overclaimed in docs.
- High-level expectation-value workflows are split across canonical runtime APIs and legacy Python bindings.
- Packaging is CMake-first and now exports package config files, but the release artifact story remains split across canonical `rocq`, legacy `python/rocq`, framework plugins, and ROCm CI validation.

## Code Surface Inventory

| Surface | Reality |
| --- | --- |
| `rocquantum/src/hipStateVec` | Real HIP kernels for state-vector simulation, sampling, measurement, expectation values, some distributed scaffolding |
| `rocquantum/src/hipTensorNet` | Real contraction/SVD core, but pathfinder/slicing/dtype breadth is overstated |
| `rocquantum/src/hipDensityMat` | Real density-matrix core with limited gate/noise/observable support |
| `rocquantum/src/simulator.cpp` | Real public C++ simulator wrapper; now exposes `MCX` / `CSWAP`, `reset_qubit`, and generic all-one-control `apply_controlled_matrix`, but controlled-unitary breadth remains narrower than full backend capability |
| `rocqCompiler/*` | Partial MLIR/QIR codegen path; not integrated into a working execution loop |
| `rocq/*` | Top-level Python surface with direct backend execution and mock fallbacks |
| `python/rocq/*` | Separate legacy-ish Python surface with top-level CMake-built `_rocq_hip_backend` bindings; Pauli expectation paths now use native helpers, but broader runtime behavior remains split |
| `integrations/*` | Thin framework adapters with mixed native and host-side behavior |
| `rocquantum/backends/*` | Mixed remote-provider clients, local mocks, and skeleton placeholders |
| `rocquantum/qec`, `rocquantum/solvers` | Experimental VQE/QAOA/repetition-code helpers, not production-ready CUDA-QX analogs |

## What Works Today

### Native simulator core

- `hipStateVec` implements native single-qubit gates, `CNOT`, `CZ`, `SWAP`, `CRX`, `CRY`, `CRZ`, destructive measurement, sampling, state readback, single-Pauli expectations, Z-product expectations, and generic Pauli-string expectations.
- Backend-native `MCX` and `CSWAP` logic in `rocquantum/src/hipStateVec/hipStateVec.cpp` is now surfaced through `QuantumSimulator::apply_gate()` and framework adapter tests.
- `rocquantum::QuantumSimulator` exposes basic gate application, matrix application, statevector readback, selected-qubit probability vectors, single-state and batched dense matrix expectations, measurement/count generation, state-collapsing single-qubit measurement, and single-qubit reset through measurement collapse plus conditional X.

### Density matrix and noise

- `hipDensityMat` supports density-matrix allocation, reset, single-qubit gates, `CNOT`, controlled single-qubit gate application, and basic channels such as bit flip, phase flip, depolarizing, and amplitude damping.
- `hipDensityMatApplyChannel` now accepts single-qubit Kraus channel descriptors, and the named channels share that helper path.
- `hipDensityMatSample` plus canonical `rocq.backends.DensityMatrixBackend.sample()` provide density-matrix sampling through a host-side correctness path over diagonal probabilities.
- Density-matrix expectation helpers exist for single-qubit X/Y/Z and Pauli-Z products.

### Tensor networks

- `hipTensorNet` contains real permutation helpers, pairwise contraction, greedy multi-tensor contraction, and complex64 SVD.
- TensorNet exposes build-time capabilities for C64/C128 dtype availability, pathfinder algorithms, memory-limit planning, and runtime slicing status. GREEDY is the only always-available optimizer; unsupported METIS/KAHYPAR choices fail explicitly.
- CI contains one real GPU regression target for tensor-network contraction.

### Direct runtime execution

- The top-level `rocq` stack executes circuits directly against backend adapters.
- The `python/rocq` stack can execute circuits directly through `_rocq_hip_backend` without going through a compiler/runtime bridge.

## What Is Partial

### Compiler/runtime

- `rocqCompiler::MLIRCompiler::emit_qir()` exists and does partial lowering to LLVM IR/QIR.
- Lowering coverage is incomplete, but `compile_and_execute()` now parses the emitted MLIR subset and dispatches qalloc/H/X/Y/Z/S/Sdg/T/Tdg, CNOT/CZ/SWAP/CCX/MCX/CSWAP, RX/RY/RZ/P, and CRX/CRY/CRZ/CP to the configured backend.
- The repo carries two separate compiler/IR stories: `rocqCompiler/*` and `rocquantum/include/src/rocqCompiler/*`.

### Generic matrix and controlled-unitary support

- Generic matrix application is partially native; larger/general host-side logic still exists in `rocquantum/src/hipStateVec/hipStateVec.cpp` but is no longer the default path.
- Generic controlled unitary support is also partial. `QuantumSimulator::apply_controlled_matrix()` now exposes the backend controlled-matrix path to framework adapters, and Qiskit / PennyLane adapters use it for generic controlled base-unitary dispatch where the binding supports it. Unsupported breadth still follows the same explicit-fallback policy.
- As of the 2026-06-10 fast-path refresh, those host fallbacks are no longer a default performance path. Unsupported cases return `ROCQ_STATUS_NOT_IMPLEMENTED` unless `ROCQ_ALLOW_HOST_MATRIX_FALLBACK=1` is set for explicit slow/debug fallback.

### Multi-GPU/distributed

- Distributed handles, allocation, distributed metadata, and some local-domain distributed operations exist.
- Many distributed operations still return `ROCQ_STATUS_NOT_IMPLEMENTED`.
- Non-local single-qubit, controlled single-qubit, CNOT/CZ, and generic matrix/control-matrix operations can use an explicit correctness-first host fallback with `ROCQ_DISTRIBUTED_FALLBACK_MODE=host` or `ROCQ_ENABLE_DISTRIBUTED_HOST_FALLBACK=1`.
- When RCCL is found at build time, local-domain distributed Pauli expectation, sampling, and probability-vector reductions can use RCCL `AllReduce(sum)` instead of gathering the full state to host.
- `MULTI_GPU_GUIDE.md` overclaimed implemented RCCL behavior relative to the actual code path.

### High-level expectations

- Native expectation helpers exist in `hipStateVec` and are bound in `python/rocq/bindings.cpp`.
- The canonical top-level `rocq.operator.get_expectation_value()` delegates to `rocq.observe()` for supported Pauli operators, `HermitianOperator` dense matrices, and full-state `SparseHamiltonianOperator` CSR observables; the state-vector backend prefers the low-level `get_expectation_matrix` / `rocsvGetExpectationMatrix` and `get_sparse_matrix_moments` / `rocsvGetSparseMatrixMoments` hooks when available, and the legacy `_rocq_hip_backend` binding also exposes `get_expectation_pauli_string_batch` / `rocsvGetExpectationPauliStringBatch` plus `get_expectation_matrix_batch` / `rocsvGetExpectationMatrixBatch` for local batched readouts.
- `python/rocq/api.py::Circuit.expval()` now uses native backend Pauli expectation helpers, matching `get_expval()` for supported Pauli terms.

### Packaging and CI

- Native CMake build exists and now follows the ROCm HIP-language floor of CMake `3.21` plus official config-package targets such as `hip` / `hip::host`, `roc::rocblas`, and `roc::rocsolver`. The install package config uses the same ROCm package naming, and the root build activates `python/rocq` for `_rocq_hip_backend`; Python packaging is still split between canonical and legacy surfaces.
- CI covers Python import/package contracts, one GPU runtime regression, and a release benchmark artifact registry. The registry records statevec, distributed reduction, TensorNet, and DensityMat benchmark JSON when native ROCm binaries are available, or explicit skipped JSON when they are not.

## What Is Stubbed Or Absent

- `rocqCompiler::MLIRCompiler::compile_and_execute()` rejects unsupported ops with diagnostics outside the MVP subset.
- Compiler-driven execution parity with CUDA-Q is absent.
- Mid-circuit measurement plus classical control flow is limited to simple Qiskit `if_test` / `if_else`, finite `for_loop`, bounded `while_loop`, loop-local `break_loop` / `continue_loop`, and `switch_case` sampling trajectories; broader statevector/estimator dynamic semantics remain absent.
- Public `QuantumSimulator` now exposes Pauli, dense-matrix, batched dense-matrix, and sparse-CSR expectation helpers through `expectation_value()`, `expectation_pauli_string()`, `expectation_matrix()`, `expectation_matrix_batch()`, and `sparse_hamiltonian_moments()`, with root pybind coverage for framework adapters.
- Public `QuantumSimulator` named APIs now route `MCX` and `CSWAP`; `apply_controlled_matrix()` exposes a generic controlled-unitary surface, but breadth and native ROCm E2E validation remain partial.
- PennyLane `diff_method="device"` now exposes a device jacobian that builds parameter-shift tapes with PennyLane's standard transform and routes compatible shift batches through rocQuantum's batched runtime path, reducing isolated replay for supported parametric gates and Pauli/Hermitian/SparseHamiltonian expectation readouts.
- Density-matrix multi-qubit/gpu-resident generic channel planning is absent.
- Density-matrix sampling is not yet a GPU-fast path; it copies probability information to host before drawing shots.
- Stabilizer/tableau/Pauli-propagation backends were not found.
- CUDA-QX-style higher-level solver/QEC libraries are limited to experimental VQE objective/gradient over the supported canonical observable subset, MaxCut-style QAOA helper, and one 3-qubit repetition-code syndrome round.

## Highest-Risk Overclaims Before This Audit

- The repo read as a hardware-agnostic CUDA-Q-like framework even though the real working core is primarily a local HIP simulator plus mixed provider/client scaffolding.
- Multi-GPU docs implied implemented RCCL-based distribution while the same file also admitted the distributed wrappers are still single-GPU compatibility wrappers.
- Roadmap items still listed already-implemented backend-native gates as future work.
- Public high-level APIs implied expectation-value and compiler/runtime functionality that is either stubbed or host-side fallback.

## Native Vs Fallback Summary

| Area | Native | Fallback / mock / host-side |
| --- | --- | --- |
| State-vector gates | HIP kernels in `hipStateVec`; canonical `rocq` and legacy `python/rocq` can use GateFusion for supported CNOT-adjacent spans | None for core named gates; broader fusion patterns remain unfused |
| Generic matrix/control matrix | Partial HIP path | Broader cases return `ROCQ_STATUS_NOT_IMPLEMENTED` by default; `ROCQ_ALLOW_HOST_MATRIX_FALLBACK=1` enables explicit slow/debug host fallback |
| Expectations | Native `hipStateVec` helpers exist; canonical `rocq.observe()` and legacy `Circuit.expval()` are wired for supported Pauli operators; local batched Pauli-string expectation is exposed through public and legacy binding surfaces; canonical `HermitianOperator` and PennyLane `qml.Hermitian` can use dense matrix expectation for supported target sizes; local batched dense-matrix expectation is exposed through public and legacy binding surfaces; canonical `SparseHamiltonianOperator` and PennyLane `qml.SparseHamiltonian` can use native CSR sparse moments for local single-state and batched state vectors | Broader arbitrary operator coverage, large dense matrices, distributed dense expectations, and distributed sparse expectations remain limited |
| Tensor-network contraction | Native HIP/rocBLAS/rocSOLVER path | Pathfinder/slicing breadth not fully wired |
| Density matrix | Native limited kernel set | Generic single-qubit Kraus channels and sampling use correctness-first paths; multi-qubit channels and GPU-fast density sampling remain absent |
| Framework adapters | Use native simulator for core operations; PennyLane/Cirq/Qiskit sampling paths now prefer `QuantumSimulator.measure()`; PennyLane analytic `qml.probs()` can use `QuantumSimulator.probabilities()` / `rocsvProbabilities` without statevector readback; PennyLane `qml.Hermitian` can use `QuantumSimulator.expectation_matrix()` for expval/var and `expectation_matrix_batch()` for simple parameter sweeps; PennyLane `qml.SparseHamiltonian` can use `rocsvGetSparseMatrixMoments` / `rocsvGetSparseMatrixMomentsBatch` through `QuantumSimulator.sparse_hamiltonian_moments()` / `sparse_hamiltonian_moments_batch()` without statevector readback; Qiskit runtime reset sampling uses shot-by-shot `QuantumSimulator.reset_qubit()` trajectories; Qiskit simple `if_test` / `if_else`, finite `for_loop`, bounded `while_loop`, loop-local `break_loop` / `continue_loop`, and `switch_case` sampling uses shot-by-shot `QuantumSimulator.measure_qubit()` trajectories; Qiskit estimator batches reuse prepared circuits for multiple Pauli observables and cache canonical duplicates; Qiskit provider direct and native-estimator dense `Operator` expectations can use `QuantumSimulator.expectation_matrix()` and `expectation_matrix_batch()` for supported full-circuit and low-index partial dense observables; Qiskit/PennyLane multi-control defaults reach native `MCX` / `CSWAP`; PennyLane analytic Pauli/Hadamard/Projector/Hamiltonian expectations and Qiskit estimator expectations use native Pauli-string helpers without mandatory statevector readback | PennyLane/Cirq keep host sampling fallback for legacy bindings; PennyLane probability vectors, Hermitian observables, and SparseHamiltonian fall back to statevector readback for unsupported native targets; Qiskit dense `Operator` observables still cannot encode arbitrary non-low-index partial wire placement without explicit padding; distributed sparse moments remain unsupported; Qiskit reset/dynamic sampling rejects statevector/estimator output; broad statevector/estimator dynamic coverage remains partial; many tests are mock/source-contract only |
| Top-level `rocq` backend selection | Can hit native bindings | Falls back to mock state objects when compiled backend is missing |

## Comparison Baselines Used

- CUDA-Q official docs: `https://nvidia.github.io/cuda-quantum/latest/`
- cuQuantum official docs: `https://docs.nvidia.com/cuda/cuquantum/latest/`
- CUDA-QX official repo: `https://github.com/NVIDIA/cudaqx`

The current repo is closest to a partial ROCm analogue of cuStateVec plus parts of cuTensorNet plus a thin local simulator API. It is materially behind CUDA-Q in compiler/runtime integration and behind CUDA-QX in higher-level libraries.

## External ROCm Ground Truth Used

- ROCm release history: `https://rocm.docs.amd.com/en/latest/about/release-history.html`
- ROCm compatibility matrix: `https://rocm.docs.amd.com/en/latest/compatibility/compatibility-matrix.html`
- ROCm Linux install/system requirements: `https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html`

As of 2026-06-10, the official AMD ROCm release history and production documentation show ROCm `7.2.4` as the current production target. The earlier 2026-04-05 audit snapshot used `7.2.0`; compatibility guidance is now updated to `7.2.4` while retaining `6.2.2` as the conservative non-experimental CI baseline until runner/container availability is proven.

## Verification Limits

- This shell exposes Python, but it does not expose `hipcc`, `cmake`, `ninja`, or a local ROCm runtime.
- Native ROCm runtime validation therefore remains a CI/ROCm-Linux execution task.
