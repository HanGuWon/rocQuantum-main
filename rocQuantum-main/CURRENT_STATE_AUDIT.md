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
- GateFusion is wired opportunistically into canonical `rocq.backends.StateVectorBackend`, but the legacy `python/rocq` queue flush path still replays gates one by one.
- `rocqCompiler::MLIRCompiler::compile_and_execute()` is no longer a hard stub for the current MVP subset: qalloc, H/X/Y/Z, CNOT, RX/RY/RZ.
- Qiskit and PennyLane adapters now avoid several avoidable full statevector readbacks: Qiskit `backend.run()` defaults to sampling without state output, Qiskit Estimator reuses a bound circuit across observable batches, PennyLane finite-shot measurements use native `measure()` where available, and PennyLane analytic Pauli/Hadamard/Hamiltonian expectations skip diagonalizing rotations and use native Pauli-string expectations.
- Public simulator and framework paths now expose native `MCX` / `CSWAP` dispatch for Qiskit `ccx` / `mcx` / `cswap` and PennyLane `MultiControlledX` / `Toffoli` / `CSWAP`; non-default PennyLane `control_values` use exact `X`-flip plus native-`MCX` decomposition instead of dense matrix fallback.
- Qiskit `ch` / `cy` / `ccz` / `dcx` and PennyLane `qml.CH` / `qml.CY` / `qml.CCZ` now use exact native controlled-Pauli decompositions instead of dense matrix dispatch.
- Qiskit `ecr` and PennyLane `qml.ECR` now use exact native decompositions instead of dense two-qubit matrix dispatch; PennyLane preserves the `SX` global phase with a one-qubit global-phase matrix.
- Qiskit `iswap` and PennyLane `qml.ISWAP` / `qml.PSWAP` / `qml.SISWAP` / `qml.SQISW` now use exact native swap/phase decompositions instead of dense two-qubit matrix dispatch.
- Qiskit `rccx` / `rcccx` now use exact native relative-phase `H` / `RZ` / `CNOT` decompositions instead of dense three- and four-qubit matrix dispatch.
- Direct Qiskit open-control controlled-X and single-control controlled-H operations now use exact `X`-flip plus native `CX` / `MCX` / `CH` decomposition instead of dense matrix dispatch.
- Qiskit `sx` / `sxdg` / `tdg` / `u` now use exact native single-qubit rotation decompositions, including global phase only for statevector-producing runs.
- Qiskit `p` / `cp` now use exact native `rz` / `cx` decompositions, including global phase only for statevector-producing runs and skipping it on sampling / estimator paths.
- Qiskit `rxx` / `ryy` / `rzz` and PennyLane `qml.IsingXX` / `qml.IsingYY` / `qml.IsingXY` / `qml.IsingZZ` now use exact native CNOT/rotation decompositions instead of dense two-qubit matrix dispatch.
- PennyLane `qml.SingleExcitation` and `qml.DoubleExcitation` now use exact native `H` / `CNOT` / `RY` decompositions instead of dense two- and four-qubit matrix dispatch.
- PennyLane `qml.SingleExcitationPlus` / `qml.SingleExcitationMinus` now use exact native decompositions with one-qubit global-phase matrices instead of dense two-qubit matrix dispatch.
- PennyLane `qml.FermionicSWAP` now uses an exact native `H` / `RX` / `RZ` / `CNOT` plus global-phase decomposition instead of dense two-qubit matrix dispatch.
- PennyLane `qml.OrbitalRotation` now uses exact native `FermionicSWAP` and `SingleExcitation` decompositions instead of dense four-qubit matrix dispatch.
- PennyLane `qml.PhaseShift`, `qml.ControlledPhaseShift`, and open-control `qml.CPhaseShift00/01/10` now use exact global-phase plus native `RZ` / `CNOT` decompositions, avoiding dense controlled-phase matrix upload in QFT-style circuits.
- PennyLane `qml.SparseHamiltonian` analytic expectation / variance is now supported through a statevector CSR fallback, not a native sparse-observable GPU kernel.

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
- Packaging, install/export, and CI do not describe one coherent release artifact.

## Code Surface Inventory

| Surface | Reality |
| --- | --- |
| `rocquantum/src/hipStateVec` | Real HIP kernels for state-vector simulation, sampling, measurement, expectation values, some distributed scaffolding |
| `rocquantum/src/hipTensorNet` | Real contraction/SVD core, but pathfinder/slicing/dtype breadth is overstated |
| `rocquantum/src/hipDensityMat` | Real density-matrix core with limited gate/noise/observable support |
| `rocquantum/src/simulator.cpp` | Real public C++ simulator wrapper; now exposes `MCX` / `CSWAP`, but generic controlled-unitary breadth remains narrower than backend capability |
| `rocqCompiler/*` | Partial MLIR/QIR codegen path; not integrated into a working execution loop |
| `rocq/*` | Top-level Python surface with direct backend execution and mock fallbacks |
| `python/rocq/*` | Separate legacy-ish Python surface with `_rocq_hip_backend` bindings; Pauli expectation paths now use native helpers, but broader runtime behavior remains split |
| `integrations/*` | Thin framework adapters with mixed native and host-side behavior |
| `rocquantum/backends/*` | Mixed remote-provider clients, local mocks, and skeleton placeholders |
| `rocquantum/qec`, `rocquantum/solvers` | Experimental VQE/QAOA/repetition-code helpers, not production-ready CUDA-QX analogs |

## What Works Today

### Native simulator core

- `hipStateVec` implements native single-qubit gates, `CNOT`, `CZ`, `SWAP`, `CRX`, `CRY`, `CRZ`, destructive measurement, sampling, state readback, single-Pauli expectations, Z-product expectations, and generic Pauli-string expectations.
- Backend-native `MCX` and `CSWAP` logic in `rocquantum/src/hipStateVec/hipStateVec.cpp` is now surfaced through `QuantumSimulator::apply_gate()` and framework adapter tests.
- `rocquantum::QuantumSimulator` exposes basic gate application, matrix application, statevector readback, and measurement/count generation.

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
- Lowering coverage is incomplete, but `compile_and_execute()` now parses the emitted MLIR subset and dispatches qalloc/H/X/Y/Z/CNOT/RX/RY/RZ to the configured backend.
- The repo carries two separate compiler/IR stories: `rocqCompiler/*` and `rocquantum/include/src/rocqCompiler/*`.

### Generic matrix and controlled-unitary support

- Generic matrix application is partially native; larger/general host-side logic still exists in `rocquantum/src/hipStateVec/hipStateVec.cpp` but is no longer the default path.
- Generic controlled unitary support is also partial and follows the same explicit-fallback policy.
- As of the 2026-06-10 fast-path refresh, those host fallbacks are no longer a default performance path. Unsupported cases return `ROCQ_STATUS_NOT_IMPLEMENTED` unless `ROCQ_ALLOW_HOST_MATRIX_FALLBACK=1` is set for explicit slow/debug fallback.

### Multi-GPU/distributed

- Distributed handles, allocation, distributed metadata, and some local-domain distributed operations exist.
- Many distributed operations still return `ROCQ_STATUS_NOT_IMPLEMENTED`.
- Non-local single-qubit, controlled single-qubit, CNOT/CZ, and generic matrix/control-matrix operations can use an explicit correctness-first host fallback with `ROCQ_DISTRIBUTED_FALLBACK_MODE=host` or `ROCQ_ENABLE_DISTRIBUTED_HOST_FALLBACK=1`.
- When RCCL is found at build time, local-domain distributed Pauli expectation and sampling probability reductions can use RCCL `AllReduce(sum)` instead of gathering the full state to host.
- `MULTI_GPU_GUIDE.md` overclaimed implemented RCCL behavior relative to the actual code path.

### High-level expectations

- Native expectation helpers exist in `hipStateVec` and are bound in `python/rocq/bindings.cpp`.
- The canonical top-level `rocq.operator.get_expectation_value()` delegates to `rocq.observe()` for supported Pauli operators.
- `python/rocq/api.py::Circuit.expval()` now uses native backend Pauli expectation helpers, matching `get_expval()` for supported Pauli terms.

### Packaging and CI

- Native CMake build exists and now follows the ROCm HIP-language floor of CMake `3.21` plus official config-package targets such as `hip` / `hip::host`, `roc::rocblas`, and `roc::rocsolver`. Python packaging is still split between `pyproject.toml`, `setup.py`, `rocquantum_bind`, and `_rocq_hip_backend`.
- CI covers Python import/package contracts, one GPU runtime regression, and a release benchmark artifact registry. The registry records statevec, distributed reduction, TensorNet, and DensityMat benchmark JSON when native ROCm binaries are available, or explicit skipped JSON when they are not.

## What Is Stubbed Or Absent

- `rocqCompiler::MLIRCompiler::compile_and_execute()` rejects unsupported ops with diagnostics outside the MVP subset.
- Compiler-driven execution parity with CUDA-Q is absent.
- Mid-circuit measurement plus classical control flow is absent as a coherent supported feature.
- Public `QuantumSimulator` now exposes Pauli expectation helpers through `expectation_value()` and `expectation_pauli_string()`, with root pybind coverage for framework adapters.
- Public `QuantumSimulator` named APIs now route `MCX` and `CSWAP`; a broad generic controlled-unitary public surface is still absent.
- Density-matrix multi-qubit/gpu-resident generic channel planning is absent.
- Density-matrix sampling is not yet a GPU-fast path; it copies probability information to host before drawing shots.
- Stabilizer/tableau/Pauli-propagation backends were not found.
- CUDA-QX-style higher-level solver/QEC libraries are limited to experimental VQE objective/gradient, MaxCut-style QAOA helper, and one 3-qubit repetition-code syndrome round.

## Highest-Risk Overclaims Before This Audit

- The repo read as a hardware-agnostic CUDA-Q-like framework even though the real working core is primarily a local HIP simulator plus mixed provider/client scaffolding.
- Multi-GPU docs implied implemented RCCL-based distribution while the same file also admitted the distributed wrappers are still single-GPU compatibility wrappers.
- Roadmap items still listed already-implemented backend-native gates as future work.
- Public high-level APIs implied expectation-value and compiler/runtime functionality that is either stubbed or host-side fallback.

## Native Vs Fallback Summary

| Area | Native | Fallback / mock / host-side |
| --- | --- | --- |
| State-vector gates | HIP kernels in `hipStateVec` | None for core named gates |
| Generic matrix/control matrix | Partial HIP path | Broader cases return `ROCQ_STATUS_NOT_IMPLEMENTED` by default; `ROCQ_ALLOW_HOST_MATRIX_FALLBACK=1` enables explicit slow/debug host fallback |
| Expectations | Native `hipStateVec` helpers exist; canonical `rocq.observe()` and legacy `Circuit.expval()` are wired for supported Pauli operators | Hermitian/broader operator coverage remains limited |
| Tensor-network contraction | Native HIP/rocBLAS/rocSOLVER path | Pathfinder/slicing breadth not fully wired |
| Density matrix | Native limited kernel set | Generic single-qubit Kraus channels and sampling use correctness-first paths; multi-qubit channels and GPU-fast density sampling remain absent |
| Framework adapters | Use native simulator for core operations; PennyLane/Cirq/Qiskit sampling paths now prefer `QuantumSimulator.measure()`; Qiskit estimator batches reuse prepared circuits for multiple observables; Qiskit/PennyLane multi-control defaults reach native `MCX` / `CSWAP`; PennyLane analytic Pauli/Hadamard/Hamiltonian expectations and Qiskit estimator expectations use native Pauli-string helpers without mandatory statevector readback | PennyLane/Cirq keep host sampling fallback for legacy bindings; PennyLane `SparseHamiltonian` uses a statevector CSR fallback; Qiskit and PennyLane broad non-Pauli/dynamic/non-unitary coverage remains partial; many tests are mock/source-contract only |
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
