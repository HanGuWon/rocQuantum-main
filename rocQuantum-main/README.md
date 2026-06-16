# rocQuantum

`rocQuantum` is an experimental ROCm-first quantum computing repository centered on native HIP simulator components:

- `hipStateVec`
- `hipTensorNet`
- `hipDensityMat`

The repo also contains partially implemented compiler, Python, provider, and framework-integration surfaces inspired by CUDA-Q, cuQuantum, and CUDA-QX. It does not yet provide end-to-end parity with those NVIDIA stacks.

## Current Reality

Implemented today:

- Native HIP state-vector simulation for core named gates, sampling, measurement, and several expectation-value primitives
- Native tensor-network contraction core
- Native density-matrix core with named noise channels, generic single- and multi-qubit Kraus channel APIs, decomposed CCX/CSWAP helpers in the canonical backend, and density sampling with GPU-side marginal probability reduction
- Direct simulator execution through the active local runtime path
- CUDA-Q-style `get_state()` alias for canonical state readback plus host-side `Future` wrappers for `get_state_async()`, `execute_async()`, `sample_async()`, `observe_async()`, and `compile_and_execute_async()`

Only partial today:

- MLIR/QIR compiler flow
- `compile_and_execute()` for a narrow MLIR subset: qalloc, H/X/Y/Z/S/Sdg/T/Tdg, CNOT/CZ/SWAP/CCX/MCX/CSWAP, RX/RY/RZ/P, CRX/CRY/CRZ/CP
- Generic matrix and controlled-unitary coverage
- Multi-GPU / distributed execution
- Observable breadth and density-matrix GPU-fast sampling coverage
- Release wheel and native install-tree validation
- PennyLane, Cirq, and Qiskit adapter maturity

Not implemented today:

- End-to-end compiler-driven execution parity with CUDA-Q beyond the supported MVP subset
- Release-grade distributed multi-GPU support
- CUDA-QX-style higher-level libraries beyond the experimental VQE/QAOA/repetition-code subset with narrow readout-error mitigation; `rocquantum.solvers.solver_capabilities()` and `rocquantum.qec.qec_capabilities()` expose the current supported/unsupported contracts

## Audit Documents

This repo now includes an audit-first truth set in the inner repo root:

- `CURRENT_STATE_AUDIT.md`
- `FEATURE_TRUTH_MATRIX.md`
- `ROCM_INTEGRATION_AUDIT.md`
- `TOP_GAPS_AND_PRIORITIES.md`
- `FINAL_GAP_REPORT.md`
- `IMPLEMENT_NOW_PLAN.md`

Use those files as the authoritative capability summary for the current codebase.

## Repo Layout

- `rocquantum/src/hipStateVec`: native state-vector kernels and distributed scaffolding
- `rocquantum/src/hipTensorNet`: tensor-network contraction core
- `rocquantum/src/hipDensityMat`: density-matrix and limited noise support
- `rocquantum/src/simulator.cpp`: public C++ simulator wrapper
- `rocqCompiler/`: partial MLIR/QIR source pipeline; the default Python binding does not link it and fails fast for compiler execution
- `rocq/`: top-level Python surface with direct native execution plus `observe()` / `sample()`
- `python/rocq/`: separate legacy-style Python surface whose `_rocq_hip_backend` extension is built from the top-level CMake graph
- `integrations/`: PennyLane, Cirq, and Qiskit adapters

## Support Policy For This Audit Pass

- Primary release target: Linux x86_64
- Latest production ROCm target verified on 2026-06-10: `7.2.4`
- Native ROCm CMake minimum: CMake `3.21`
- Recommended Tier 1 GPU targets: `gfx950`, `gfx942`, `gfx90a`
- Recommended future minimum release-grade GPU target: `gfx90a`
- Recommended future minimum ROCm target: `6.4.0`
- Current non-experimental CI baseline: ROCm `6.2.2`
- Current experimental/latest CI lane: ROCm `7.2.4`
- ROCm CMake packages are consumed through official config targets such as `hip` / `hip::host`, `roc::rocblas`, `roc::rocsolver`, and optional RCCL target `rccl`
- Windows helper scripts are kept for development convenience but are not treated as release-grade support
- When no AMD GPU is available, the accepted local validation baseline is the Python mock/native-binding contract suite. ROCm hardware E2E and performance validation should be skipped explicitly, not inferred from mock results.

## Native Component Snapshot

| Component | Current State |
| --- | --- |
| `hipStateVec` | Real and useful, with local batched state allocation/readback, batch-specific RX/RY/RZ/P and CRX/CRY/CRZ/CP, batched probabilities, batched Pauli expectations, batched dense-matrix expectations, and active parallel measurement/probability kernels exposed through `QuantumSimulator`; stale single-thread measurement scaffolding is not built; not yet fully surfaced through every framework adapter |
| `hipTensorNet` | Real contraction core with explicit optimizer/dtype/slicing/permutation-rank capabilities; Python TensorNet contractions use the simulator stream and a reused rocBLAS handle, stale unwired Pathfinder scaffold is not shipped, and >16-mode tensor permutations fail fast before the fixed-local-array HIP kernel; still narrower than a full cuTensorNet analogue |
| `hipDensityMat` | Real but limited; `rocq.density_matrix_capabilities()` exposes generic channels, decomposed canonical CCX/CSWAP helpers, device-marginal sampling, dense-observable scope, and cuDensityMat descriptor/sampling boundaries |
| `rocqCompiler` | Partial source-level codegen path plus a narrow compile-and-execute MVP for qalloc/H/X/Y/Z/S/Sdg/T/Tdg/CNOT/CZ/SWAP/CCX/MCX/CSWAP/RX/RY/RZ/P/CRX/CRY/CRZ/CP; `rocq.compiler_capabilities()` exposes that partial subset plus `mlir_runtime_available` / `mlir_runtime_kind`, the dialect-definition boundary, and the transform-pipeline boundary, and the default `rocquantum_bind` build now exposes a fail-fast compiler guard instead of unresolved MLIR symbols |
| Top-level `rocq` | Canonical runtime path with native execute/sample/observe wiring, `rocq.runtime_capabilities()` metadata for the canonical/legacy runtime boundary, explicit bool-safe state-vector `enable_fusion=` execution option, canonical backend-name validation, top-level phase-gate exports (`tdg`/`tdag`, `p`/`phase`, `cp`/`cphase`), strict positive-integer `qvec` allocation, direct backend size, shot, selected-qubit, spin-factory target, finite square power-of-two dense observable matrix/target, square power-of-two sparse shape/CSR payload, finite observable-coefficient, quiet noise-channel target/probability validation, and direct density-noise target/probability validation, integer in-range, duplicate multi-qubit, and arity-invalid gate-target validation plus finite real gate-parameter validation during kernel recording and direct backend gate dispatch, duplicate-combined Pauli expectations including inside mixed sums, coefficient-aware duplicate dense-Hermitian/CSR sum readout reuse, zero-coefficient matrix/sparse sum-term elision, coefficient-preserving composite observable sums, numeric identity constants, CUDA-Q-style `rocq.spin.x/y/z/i` Pauli factories, Pauli sum/product terms and scalar division in operator arithmetic, density-matrix correctness fallback for dense Hermitian / full-state CSR observables, an explicit CPU mock state-vector fallback for local named-gate statevector contract tests, and an experimental Clifford-only `stabilizer` / `tableau` / `clifford` backend for Pauli propagation |
| Higher-level helpers | Experimental VQE public energy evaluation, objective/gradient, VQE-compatible MaxCut-style QAOA kernel/cost/solve helper that accepts edge lists or edge-weight mappings and maximizes cut value via a negated-cost VQE objective, and 3-qubit repetition-code single/repeated-round helpers with syndrome/correction analysis plus independent syndrome readout-error mitigation; solver/QEC capability metadata exposes the current CUDA-QX comparison boundary, host-loop execution scope, sequential sampled feedback scope, and hardware-evidence boundary |
| `python/rocq` | Top-level CMake-built legacy compatibility surface; Pauli expectations, batched state allocation/readback, same-target single-qubit plus CNOT-adjacent GateFusion spans, and bool-safe finite input validation for circuit sizes, gate targets/angles, samples, and Pauli coefficients now use explicit Python contracts; legacy `build()` records whether execution is conceptual MLIR or Python circuit replay and warns when replay is used, while broader fusion and runtime unification still need consolidation |

## Important Limitations

- `rocqCompiler::MLIRCompiler::compile_and_execute()` has a source-level MVP subset for qalloc, H/X/Y/Z/S/Sdg/T/Tdg, CNOT/CZ/SWAP/CCX/MCX/CSWAP, RX/RY/RZ/P, and CRX/CRY/CRZ/CP. The same partial scope is available through `rocq.compiler_capabilities()` without invoking MLIR; `binding_available` is separated from `mlir_runtime_available` / `mlir_runtime_kind`, `dialect_definition` marks release-wired TableGen op generation as absent, and `transform_pipeline` marks the old adjoint-generation pass as a legacy scaffold, so a default fail-fast `disabled_runtime_guard` binding or the older `rocquantum/Dialect` / adjoint scaffold is not confused with a release-linked compiler runtime. The default `rocquantum_bind` build does not link the experimental MLIR compiler stack; `ROCQUANTUM_ENABLE_MLIR_COMPILER=ON` currently fails at CMake configure time until that target is release-wired, and `rocq.compile_and_execute()` / `QuantumKernel.compile_and_execute()` validate boolean `strict` options and fail fast with an actionable diagnostic. Canonical MLIR emission and raw MLIR execution reject duplicate gate operands, non-positive or out-of-range `qalloc` sizes, and non-finite parametric gate angles before backend dispatch.
- Canonical `rocq.QuantumKernel.qir()` requires `rocquantum_bind`; if the binding is missing, `emit_qir()` returns a compiler error sentinel, or the default binding was built without MLIR compiler support, the Python API raises an actionable `RuntimeError` instead of returning an `"Error:"` string as if it were QIR.
- `rocq.get_state()` aliases the canonical state readback path, and `rocq.get_state_async()`, `rocq.execute_async()`, `rocq.sample_async()`, `rocq.observe_async()`, and `rocq.compile_and_execute_async()` are host-side `concurrent.futures.Future` wrappers around the canonical synchronous paths. They improve CUDA-Q-style Python ergonomics and preserve the same validation/backend contracts, but they are not yet native HIP-stream, multi-QPU, or distributed scheduler futures.
- `rocq.runtime_capabilities()` exposes the canonical runtime entry points, supported backends, host-threadpool async execution scope, runtime options, environment switches, legacy `python/rocq` compatibility note, and unsupported native-HIP-stream / multi-QPU / unified-compiler boundaries without running a kernel.
- Canonical `rocq.execute()`, `get_state()`, `sample()`, `observe()`, their host-side async wrappers, and direct `StateVectorBackend` construction accept only boolean `enable_fusion=` for the `state_vector` backend, giving users an explicit performance/debug switch in addition to `ROCQ_DISABLE_GATE_FUSION`; passing that option to non-state-vector backends raises `ValueError` instead of being ignored.
- Legacy `python/rocq.build()` emits conceptual MLIR for inspection, but simulator-backed execution replays the Python circuit API rather than calling `MLIRCompiler.compile_and_execute()`; `_rocq_hip_backend.MLIRCompiler` is a conceptual MLIR holder in the default build, and `QuantumProgram.execution_mode`, `compiler_execution_supported`, and `execution_notes` expose that contract.
- `multi_gpu=True` should be treated as experimental partial support, not full distributed execution; legacy `Circuit(..., multi_gpu=True)` emits an `ExperimentalMultiGpuWarning` and stores the same note on `Circuit.execution_notes`. Canonical `rocq.distributed_capabilities()` exposes the supported/unsupported distributed runtime contract, runtime switches, execution scope, and hardware-evidence requirements without performing a hardware probe. See `rocquantum/src/hipStateVec/MULTI_GPU_GUIDE.md` for the behavior matrix and runtime switches.
- Multi-node execution is not implemented: `rocsvAllocateMultiNodeDistributedState` returns `ROCQ_STATUS_NOT_IMPLEMENTED` for `nodeCount > 1`, and legacy `Circuit(..., multi_node=True)` / `node_count > 1` raises `NotImplementedError`.
- Distributed non-local single-qubit, controlled single-qubit, CNOT/CZ, and generic matrix/control-matrix correctness fallback is explicit slow/debug mode: set `ROCQ_DISTRIBUTED_FALLBACK_MODE=host` or `ROCQ_ENABLE_DISTRIBUTED_HOST_FALLBACK=1`.
- RCCL-backed distributed expectation and sampling reductions are limited to local-domain qubits; set `ROCQ_DISTRIBUTED_COMM=rccl` or `ROCQ_REQUIRE_RCCL=1` to require RCCL on a ROCm runner. The installed CMake package probes RCCL with optional `find_package(rccl QUIET)` so exported targets can resolve RCCL when the build linked it without making non-RCCL consumers fail.
- Generic matrix/control-matrix cases outside HIP fast paths return `NOT_IMPLEMENTED` by default; current controlled-matrix fast paths cover one or more all-one controls over a single 2x2 target matrix, while broader controlled dense matrices still require `ROCQ_ALLOW_HOST_MATRIX_FALLBACK=1` for explicit slow/debug host fallback.
- Dense matrix moments for supported local single-state and batched paths use fused HIP reductions through `rocsvGetExpectationMatrixMoments` / `rocsvGetExpectationMatrixMomentsBatch`; unsupported fused paths fall back to the existing dense expectation hooks or explicit slow/debug host fallback rules.
- TensorNet supports the build's compiled complex dtype (`C64` by default, `C128` in `ROCQ_PRECISION_DOUBLE` builds); METIS is optional behind `ROCQUANTUM_TENSORNET_ENABLE_METIS`, `ROCQUANTUM_TENSORNET_ENABLE_KAHYPAR=ON` fails fast because KaHyPar is not release-wired, unavailable pathfinders fall back to greedy with warnings, and `memory_limit_bytes` / `num_slices` now drive deterministic runtime K-sliced GEMM accumulation for pair contractions. `get_tensornet_capabilities()` reports that runtime slicing kind as `limited_pair_contraction_k_sliced_gemm`, reports the hard 16-mode tensor permutation limit, and marks open-index slicing, mixed precision, and simultaneous runtime C64/C128 support as unsupported. Python TensorNet contract calls retrieve the active simulator stream and reuse a rocBLAS handle owned by the TensorNet wrapper instead of passing placeholder handles. This is still narrower than cuTensorNet-style open-index slicing and high-rank permutation coverage, and mixed precision remains a documented future lane rather than simultaneous runtime C64/C128 execution.
- `rocq.density_matrix_capabilities()` exposes the canonical density-matrix boundary without running a kernel: `hipDensityMatApplyChannel` accepts generic single- and multi-qubit Kraus channels, but uses correctness-first per-Kraus kernels rather than GPU-resident cuDensityMat-style channel descriptors; canonical `DensityMatrixBackend` decomposes `CCX` / two-control `MCX` / `CSWAP` through supported density-matrix primitive gates while larger `MCX` still needs an explicit ancilla policy; density-matrix sampling reduces measured-qubit marginal probabilities on the GPU before drawing shots on host, and dense Hermitian density-matrix expectations now use a native HIP reduction for up to four target qubits, while larger dense observables and full-state CSR density-matrix expectations still use host correctness fallback rather than cuDensityMat-style descriptor reductions.
- Higher-level CUDA-QX-style helpers are explicitly experimental: VQE supports public one-shot energy evaluation, canonical `QuantumOperator` objective validation, ansatz-kernel validation, Pauli-observable objectives, coefficient-preserving composite sums, numeric identity constants, CUDA-Q-style `rocq.spin.x/y/z/i` Pauli factories, Pauli sum/product terms and scalar division in operator arithmetic, `TypeError` diagnostics for unsupported observable arithmetic operand types, scalar single-parameter gradient/optimizer inputs, bool-safe finite real parameter and finite-difference step validation, supported gradient-method validation, canonical runtime backend-name validation, bool-safe verbose-option validation, ansatz positional parameter-count validation, optimizer-trace-preserving gradient probes, finite-real observed energy and optimizer `fun`/`x` result value/count validation, custom optimizer `minimize()` validation, string-keyed `SciPyOptimizer` option validation/copying, and dense Hermitian / full-state CSR observables through the state-vector native/fallback path or density-matrix correctness fallback; vector-parameter QAOA and one-element vector ansatz evaluation goes through `rocq.observe()`, QAOA is a MaxCut-style kernel/cost/solve helper with edge-list or edge-weight mapping normalization, duplicate/reversed undirected edges aggregated into weighted `0.5 * w * (I - Zi Zj)` edge terms, validated edge containers/shapes, integer endpoints, and bool-safe finite real weights/parameters, and `solve_maxcut_qaoa()` now minimizes the negated cost operator while reporting `optimal_cut_value` for the maximized cut objective; QEC covers generic sampled stabilizer-fragment orchestration plus a 3-qubit repetition-code syndrome subset with canonical backend-name, bool-safe verbose-option, code/decoder callable-interface, non-empty non-mapping stabilizer-fragment sequence, callable-or-None initial-state kernel, logical-operator result, and decoder-correction result validation, positive-integer shot/round/num_qubits and ancilla-index validation, minimum-5-qubit circuit generation, bool-safe one- or two-bit count keys and syndrome-bit validation, single-round sampling, sequential repeated-round histogram/correction aggregation, and independent syndrome readout-error mitigation. `rocquantum.solvers.solver_capabilities()` and `rocquantum.qec.qec_capabilities()` expose those supported and unsupported subsets plus the current canonical `supported_backends`, execution scopes, and hardware-evidence boundaries for programmatic CUDA-QX comparison. This is not a native-adjoint solver stack, distributed hybrid workflow scheduler, fault-tolerant decoder stack, or general noise-aware QEC library.
- `pyproject.toml` declares `numpy>=1.21` as a base runtime dependency for the canonical Python package and a `solvers` extra with `scipy>=1.10` for the experimental VQE optimizer path. Adapter-local `setup.py` files are compatibility installers only; they read the root project version and keep their adapter dependency floor aligned with the matching root optional extra, while the supported project install path remains the repository root.
- The canonical `stabilizer` backend is experimental and Clifford-only: H/X/Y/Z/S/Sdg/CNOT/CZ/SWAP circuits can use tableau Pauli expectation membership, while non-Clifford gates, noise, and GPU-accelerated stabilizer execution remain unsupported.
- `rocquantum_bind.QuantumSimulator` can allocate `batch_size > 1` local state batches, read one state slice or the full `(batch_size, 2**num_qubits)` host array, apply batch-specific RX/RY/RZ/P and CRX/CRY/CRZ/CP angles, return native batch-major probability matrices, Pauli-string expectation vectors, dense-matrix expectation vectors, and dense-matrix moment vectors, with shared-runtime statevector correctness fallback for single dense expectations and single-read batched dense moments when bindings lack native dense hooks, and expose `measure_batch()` as a batch-major sampling hook over the existing `rocsvSample()` primitive. Qiskit backend sampling-only circuit lists and global-phase-corrected statevector-only circuit lists, including fixed Pauli/unitary operations, two-qubit Pauli-rotation, and `PauliEvolutionGate` sweeps with identity terms, Qiskit Estimator, and PennyLane `batch_execute` can route simple Pauli-observable parameter batches through this batch surface, including Qiskit and PennyLane full-wire initial state preparation, Qiskit `u` / `r`, fixed native Qiskit `PauliGate` plus fixed Qiskit `unitary` / generic controlled-unitary operations, Qiskit open-control controlled rotation/phase sweeps, Qiskit `rxx` / `ryy` / `rzz` / `rzx` / `xx_plus_yy` / `xx_minus_yy` sweeps, supported Qiskit `PauliEvolutionGate` time sweeps, identical PennyLane `BasisState` initializers, fixed PennyLane `QubitUnitary` / `ControlledQubitUnitary` / dense `BlockEncode` plus sparse-`BlockEncode` public sparse-apply dispatch, PennyLane `Rot` / `CRot` / `ControlledSequence` / `Select` / `MultiRZ` / `PauliRot` / `SelectPauliRot` / `DiagonalQubitUnitary` / Ising / `PSWAP` / open-control phase / excitation plus-minus / fermionic-orbital sweeps, measurement-only PennyLane `GlobalPhase` sweeps, and fixed native PennyLane decompositions such as `QFT`, `QubitSum`, `QubitCarry`, `GroverOperator`, `BasisEmbedding`, `Permute`, and PennyLane-expanded `QROM` selected basis-loader blocks that appear alongside the swept parameters; Qiskit Sampler and PennyLane also route simple batched probability readout through native batch probabilities, Qiskit dense scalar or identity `Operator` readouts fold to constants and small non-identity diagonal dense `Operator` readouts lower to Pauli-Z payloads before dense hooks, PennyLane Hermitian and scalar-scaled Hermitian readouts can use dense-matrix expectation and moment hooks, small diagonal Hermitian and SparseHamiltonian readouts lower to Pauli-Z payloads before dense/CSR hooks, PennyLane SparseHamiltonian and scalar-scaled SparseHamiltonian readouts can use native CSR moments, PennyLane-expanded signed/complex-coefficient `PrepSelPrep` and `FABLE` lower through native controlled-Pauli, controlled-phase, rotation, and swap decompositions for tested shapes, and PennyLane finite-shot `sample`/`counts` parameter batches use the shared batched measurement hook. Broader broadcasted framework workloads still need more adapter coverage. `python/rocq/api.py::Circuit` exposes batched state readback.
- Qiskit native Estimator folds dense scalar `Operator([[c]])` and dense identity `Operator(c*I)` observables into constant expectation values for single and batched pubs, lowers small diagonal dense `Operator` observables to Pauli-Z payloads, and reuses normalized Pauli and dense readouts for scalar-multiple `SparsePauliOp` and dense `Operator` observables within single and batched pubs, avoiding unnecessary expectation hooks or statevector readout work.
- PennyLane single analytic execution and compatible `batch_execute` normalize scalar-multiple Pauli, dense Hermitian, and CSR `SparseHamiltonian` readout payloads inside the prepared-state cache, including reordered Pauli sums in batch mode, so duplicate base observables are evaluated once and coefficients are reapplied to means/moments without extra native hooks or statevector fallback.
- PennyLane computational-basis `qml.Projector` variance now uses `P^2=P` after lowering the projector to Pauli-Z terms, including scalar wrappers and compatible `batch_execute`, avoiding a separate Pauli-square readout plan.
- PennyLane expectation sums that mix Pauli-representable terms with `qml.Hermitian` or `qml.SparseHamiltonian` now split into native component readouts for analytic expval and compatible `batch_execute`, with sum components sharing the prepared batch readout cache with top-level measurements, same-target dense Hermitian components coalesced before readout and reclassified to constant or Pauli-Z payloads when the merged matrix becomes identity or diagonal; same-structure CSR sparse components are coalesced before sparse readout and zero merged CSR components are elided; scalar identity Hermitian, small diagonal Hermitian, and small diagonal CSR sparse matrices fold into constant or Pauli-Z payloads without dense/CSR readout; dense-only mixed variance can use dense matrix moment hooks for small target sets, while unsupported larger dense mixed variance falls back cleanly through PennyLane's upper device path; mixed variance containing `SparseHamiltonian` uses PennyLane's CSR observable representation with native sparse moments; and the same heterogeneous Pauli/dense/sparse term lists can enter native adjoint payloads.
- PennyLane `ControlledSequence` now keeps native single-qubit-base controlled powers on native controlled-rotation / controlled-phase / controlled-Pauli paths for execution and compatible `batch_execute`; fixed blocks and trainable scalar `RX` / `RY` / `RZ` / `PhaseShift` base angles lower into primitive adjoint payloads with power-aware derivative scales.
- PennyLane `qml.ctrl(...)` wrappers around open-control phase variants (`C(CPhaseShift00/01/10)`) now decompose through native multi-control phase projectors for execution, compatible `batch_execute`, and adjoint payloads instead of generic controlled-matrix fallback.
- PennyLane direct fixed gates in explicit adjoint payloads now lower `Adjoint(S)` / `Adjoint(T)` / `CH` / `CY` / `CCZ` / open-control `MultiControlledX` / `ISWAP` / `SISWAP` / `ECR` through primitive native payloads instead of forcing Python adjoint fallback.
- PennyLane fixed template operations `QFT` / `BasisEmbedding` / `Permute` / `QubitSum` / `QubitCarry` / `GroverOperator` now lower through primitive native adjoint payloads instead of forcing Python adjoint fallback around supported trainable rotations.
- PennyLane fixed and trainable-angle-array `SelectPauliRot` operations now lower through primitive native adjoint payloads instead of forcing Python adjoint fallback around supported trainable rotations.
- PennyLane targetless controlled `qml.GlobalPhase` wrappers now stay on native phase-projector paths for execution, compatible `batch_execute` sweeps, and adjoint payload lowering, avoiding the previous small controlled-matrix fallback for multi-control cases.
- `QuantumSimulator.apply_sparse_matrix()` / `ApplySparseMatrix()` expose a CSR sparse-operation hook through the public binding and route local single-state, batched state vectors, and local-domain distributed slices through `rocsvApplySparseMatrix`, avoiding dense sparse-operator materialization and Python statevector readback. Non-local distributed sparse apply is available only through the explicit slow/debug distributed host fallback; AMD GPU performance validation remains pending.
- PennyLane `diff_method="device"` now uses `qml.gradients.param_shift()` and routes generated shift tapes through the device `batch_execute()` fast path, so supported parameter-shift gradients can reuse native batched rotations and batched Pauli/Hermitian/SparseHamiltonian expectation hooks, including scalar-scaled dense/sparse observable wrappers, mixed Pauli+dense/sparse expectation sums, dense-only mixed variance moments, and SparseHamiltonian-containing mixed variance CSR moments, instead of executing each shift tape as an isolated simulator run; repeated analytic measurements in single and batched executions reuse cached native expectation/moment/probability results within the same prepared state; older bindings fall back through shared runtime dense/sparse statevector correctness paths, including single-read Hermitian variance fallback. `diff_method="adjoint"` can probe a binding-level adjoint hook; the root binding now supplies an exact RX/RY/RZ/P/CRX/CRY/CRZ/CP path for Pauli-term, dense Hermitian, scalar-scaled dense Hermitian, full-state or targeted CSR sparse observable payloads, scalar-scaled CSR sparse observable payloads, and mixed Pauli+dense/sparse observable-sum payloads, fixed `QubitUnitary` / `ControlledQubitUnitary` / dense `BlockEncode` matrix operation payloads, sparse `BlockEncode` CSR operation payloads, fixed `DiagonalQubitUnitary` phase-decomposition payloads, fixed and trainable-angle-array `SelectPauliRot` payloads, fixed and scalar-parametric `ControlledSequence` controlled-power payloads, covered `Select` controlled-operation payloads including fixed selected `BasisEmbedding`, selected products that mix fixed BasisEmbedding with native operations, simple selected basis/phase or multi-native products, and fixed selected matrix operations, and fixed `QFT` / `BasisEmbedding` / `Permute` / `QubitSum` / `QubitCarry` / `GroverOperator` template payloads, with plain `GlobalPhase` adjoint payloads elided as zero-gradient globals, PennyLane `Rot` / `CRot` lowered into primitive rotation payloads, `PhaseShift` / controlled-phase payloads mapped to native phase gates, and decomposition-backed `MultiRZ` / `PauliRot` / `IsingXX/YY/ZZ/XY` / `SingleExcitation` plus/minus / `DoubleExcitation` plus/minus / `PSWAP` / `FermionicSWAP` / `OrbitalRotation` payloads using explicit parameter-derivative scales, plus explicit unsupported-payload fallback; trainable SelectPauliRot angle arrays expand into element-wise synthetic adjoint columns with Walsh-Hadamard derivative scales, while trainable matrix- or diagonal-unitary payloads, remaining trainable controlled-matrix or multi-target controlled-wrapper payloads, trainable selected BasisEmbedding arrays or trainable selected matrix payloads, and native GPU-resident adjoint differentiation remain pending.
- `python/rocq/api.py::Circuit.expval()` now uses native Pauli expectation helpers, legacy `Circuit` now rejects bool/non-integral qubit indices, non-finite gate angles, ambiguous sample shots, and non-finite Pauli coefficients before backend dispatch, legacy `build()` now warns/marks Python replay execution instead of implying compiler execution, and `multi_gpu=True` construction warns/records its partial distributed contract, but the legacy surface remains separate from canonical `rocq`.
- Canonical `rocq.backends` mock state-vector and density-matrix fallbacks now emit `MockBackendWarning` when `ROCQ_ENABLE_MOCK_BACKENDS=1` is used without native ROCm bindings; the state-vector mock performs CPU statevector semantics for canonical named-gate contract tests, but local smoke tests are still not native ROCm/cuQuantum-style execution or performance validation.
- Qiskit, PennyLane, and Cirq adapters now prefer `QuantumSimulator.measure()` for sampling, shared framework dispatch rejects bool, string, complex, and non-finite gate parameters plus bool/string/non-integral qubit targets and non-positive/non-integral shot counts before native single or batched gate calls, framework statevector upload plus matrix/control-matrix/dense-expectation/sparse-CSR payload paths reject non-finite values before native binding dispatch or statevector fallback, framework probability readout/sampling fallback rejects non-finite probability vectors, and public native `QuantumSimulator` dispatch rejects non-finite single/batched angle parameters, statevector payloads, dense matrix/control-matrix/expectation payloads, and sparse CSR data values before HIP kernels, host fallback, or device uploads are invoked; host-side fallback paths still remain where needed for older bindings that do not expose `measure`.
- The self-hosted ROCm runtime workflow builds native Python bindings and runs `scripts/native_framework_smoke.py`, a Bell-state smoke check through `rocquantum_bind`, PennyLane, Qiskit, and Cirq. It passes `--require-native-rocm-evidence`, uploads `native-framework-smoke.log`, machine-readable `native-framework-smoke.json`, and `native-framework-smoke.md`, appends the Markdown to the GitHub step summary, and records `native_rocm_evidence` / `evidence_kind` from the actual `/dev/kfd` device probe, so local source/mock tests, assumed-device runs, or locally skipped smoke JSON do not prove AMD GPU execution.
- Qiskit direct `prepare_state()` and untouched-qubit `initialize()` are mapped to matrix state-preparation fallback; `reset` after prior operations plus simple `if_test` / `if_else`, finite `for_loop`, bounded `while_loop`, loop-local `break_loop` / `continue_loop`, and `switch_case` are supported only in `backend.run(..., sampling=True)` through shot-by-shot `QuantumSimulator` trajectories. Later `initialize()`, statevector/estimator output for runtime-reset or dynamic-control circuits, and broader Qiskit control-flow semantics remain explicit unsupported boundaries.
- `rocquantum.core.list_backends()` hides unsupported skeleton providers by default, while `list_backends(include_experimental=True)` reports their `unsupported_stub` status, disabled job-submission flag, unsupported reason, and missing authentication/payload/submission/status/result capabilities. `rocq list-backends` prints the same status metadata before target selection, and `set_target()` blocks those skeleton providers unless `allow_experimental=True` or `ROCQ_ENABLE_EXPERIMENTAL_PROVIDERS=1` is set for contract tests or integration development.
- The Qristal bridge now checks for the real local Qristal SDK CLI (`qristal`, or `ROCQ_QRISTAL_CLI`) and invokes it through `subprocess.run()` instead of returning a mocked local histogram; missing SDK/CLI and failed executions raise explicit backend errors.
- Several non-skeleton provider backends remain thin clients and require provider credentials for real validation.

## Build

Use an out-of-tree build directory:

```bash
cmake -S . -B build-ci -G Ninja \
  -DBUILD_TESTING=ON \
  -DROCQUANTUM_BUILD_BINDINGS=ON \
  -DCMAKE_HIP_COMPILER=/opt/rocm/bin/hipcc
cmake --build build-ci --parallel
```

To validate the installed CMake package shape on a ROCm build host, install the build tree and
configure a downstream consumer smoke project:

```bash
bash scripts/validate_cmake_install_consumer.sh build-ci
```

The release benchmark registry is defined in `benchmarks/benchmark_manifest.json`. It covers
state-vector fast path/fallback and fusion timing, distributed RCCL vs host fallback reductions
including dense expectation, sparse moments, and generic matrix paths, TensorNet contraction planning, and
DensityMat channel/observable/sampling timing. The runner
emits one JSON file per benchmark plus `benchmark-summary.json` and `benchmark-summary.md`;
distributed RCCL-vs-host artifacts include host-fallback-over-RCCL speedup ratios when both cases
run, and configured minimum speedups or missing configured speedup metrics fail the benchmark job
through `--fail-on-error`. Each result and summary also records whether it is native performance
evidence, including the actual `/dev/kfd` device probe, so skipped, CPU-only, mock, fake, or
assumed-device benchmark runs cannot be mistaken for ROCm timing proof. If a native binary or ROCm device is unavailable, it writes an explicit skipped artifact
instead of pretending a result exists. If a benchmark executable runs but fails to write valid JSON,
the runner marks that benchmark failed so `--fail-on-error` cannot publish an empty performance
proof. Passing `--require-native-performance-evidence` makes self-hosted ROCm jobs fail unless at
least one benchmark produces a passed native ROCm timing result, while
`--require-all-native-benchmark-evidence` fails unless every ROCm-required benchmark declared in the
manifest produces passed native evidence. Passing `--history-path` also updates a bounded
`benchmark-history.json` so CI artifacts can retain recent speedup/status trends.

```bash
python3 benchmarks/run_release_benchmarks.py \
  --build-dir build-ci \
  --output-dir benchmark-artifacts
```

To gate against a previous run, pass its summary artifact:

```bash
python3 benchmarks/run_release_benchmarks.py \
  --build-dir build-ci \
  --output-dir benchmark-artifacts \
  --baseline-summary previous-benchmark-artifacts/benchmark-summary.json \
  --history-path previous-benchmark-artifacts/benchmark-history.json \
  --max-speedup-regression 0.20 \
  --require-native-performance-evidence \
  --require-all-native-benchmark-evidence \
  --fail-on-error
```

The self-hosted ROCm workflows restore the previous benchmark summary and bounded history from the
GitHub Actions cache, pass the summary as a baseline automatically when one is available, require
native performance evidence for every declared native benchmark, and save the current
summary/history back to the cache for the next run only when the summary contains successful native
performance evidence and no native-evidence gate failure.

`benchmarks/run_benchmark.py` provides a smaller QFT comparison through the PennyLane and Qiskit
adapters. It uses the current `lightning.rocq` PennyLane entry point, falls back from `qiskit-aer`
to Qiskit's `BasicSimulator` for the CPU baseline when Aer is not installed, writes
`framework-benchmark-results.json`, and records explicit skipped framework results when
`rocquantum_bind` is unavailable instead of emitting a stack trace or pretending GPU timings exist.

On a ROCm multi-GPU runner, the distributed reduction benchmark can also be run directly:

```bash
./build-ci/rocquantum/src/hipStateVec/benchmark_hipStateVec_distributed_reductions \
  --output distributed-reductions.json
```

For a release-grade Linux ROCm build, set explicit GPU targets:

```bash
-DCMAKE_HIP_ARCHITECTURES="gfx950;gfx942;gfx90a"
```

## Comparison Baselines

- CUDA-Q: `https://nvidia.github.io/cuda-quantum/latest/`
- cuQuantum: `https://docs.nvidia.com/cuda/cuquantum/latest/`
- CUDA-QX: `https://nvidia.github.io/cudaqx`

This repo is currently closest to a ROCm-native simulator project with partial higher-level surfaces, not to a finished CUDA-Q/CUDA-QX equivalent.
