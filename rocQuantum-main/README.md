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
- Native density-matrix core with named noise channels, generic single- and multi-qubit Kraus channel APIs, and density sampling correctness path
- Direct simulator execution through the active local runtime path

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
- CUDA-QX-style higher-level libraries beyond the experimental VQE/QAOA/repetition-code subset

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
- `rocqCompiler/`: partial MLIR/QIR pipeline
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
| `hipStateVec` | Real and useful, with local batched state allocation/readback, batch-specific RX/RY/RZ/P and CRX/CRY/CRZ/CP, batched probabilities, batched Pauli expectations, and batched dense-matrix expectations exposed through `QuantumSimulator`; not yet fully surfaced through every framework adapter |
| `hipTensorNet` | Real contraction core with explicit optimizer/dtype/slicing capabilities, narrower than a full cuTensorNet analogue |
| `hipDensityMat` | Real but limited; generic channels and sampling are correctness-first paths |
| `rocqCompiler` | Partial codegen path plus a narrow compile-and-execute MVP for qalloc/H/X/Y/Z/S/Sdg/T/Tdg/CNOT/CZ/SWAP/CCX/MCX/CSWAP/RX/RY/RZ/P/CRX/CRY/CRZ/CP |
| Top-level `rocq` | Canonical runtime path with native execute/sample/observe wiring, duplicate-combined Pauli expectations including inside mixed sums, coefficient-aware duplicate dense-Hermitian/CSR sum readout reuse, zero-coefficient matrix/sparse sum-term elision, coefficient-preserving composite observable sums, numeric identity constants, CUDA-Q-style `rocq.spin.x/y/z/i` Pauli factories, Pauli sum/product terms and scalar division in operator arithmetic, and density-matrix correctness fallback for dense Hermitian / full-state CSR observables |
| Higher-level helpers | Experimental VQE objective/gradient, VQE-compatible MaxCut-style QAOA helper, and 3-qubit repetition-code single-round helper |
| `python/rocq` | Top-level CMake-built legacy compatibility surface; Pauli expectations, batched state allocation/readback, and CNOT-adjacent GateFusion now use native helpers; legacy `build()` records whether execution is conceptual MLIR or Python circuit replay and warns when replay is used, while broader fusion and runtime unification still need consolidation |

## Important Limitations

- `rocqCompiler::MLIRCompiler::compile_and_execute()` executes only the current MVP subset: qalloc, H/X/Y/Z/S/Sdg/T/Tdg, CNOT/CZ/SWAP/CCX/MCX/CSWAP, RX/RY/RZ/P, CRX/CRY/CRZ/CP. The canonical Python API exposes this MVP through `rocq.compile_and_execute()` / `QuantumKernel.compile_and_execute()`, and unsupported compiler ops raise diagnostics.
- Canonical `rocq.QuantumKernel.qir()` requires `rocquantum_bind`; if the binding is missing or `emit_qir()` returns a compiler error sentinel, the Python API raises an actionable `RuntimeError` instead of returning an `"Error:"` string as if it were QIR.
- Legacy `python/rocq.build()` emits conceptual MLIR for inspection, but simulator-backed execution replays the Python circuit API rather than calling `MLIRCompiler.compile_and_execute()`; `QuantumProgram.execution_mode`, `compiler_execution_supported`, and `execution_notes` expose that contract.
- `multi_gpu=True` should be treated as experimental partial support, not full distributed execution; legacy `Circuit(..., multi_gpu=True)` emits an `ExperimentalMultiGpuWarning` and stores the same note on `Circuit.execution_notes`. See `rocquantum/src/hipStateVec/MULTI_GPU_GUIDE.md` for the behavior matrix and runtime switches.
- Distributed non-local single-qubit, controlled single-qubit, CNOT/CZ, and generic matrix/control-matrix correctness fallback is explicit slow/debug mode: set `ROCQ_DISTRIBUTED_FALLBACK_MODE=host` or `ROCQ_ENABLE_DISTRIBUTED_HOST_FALLBACK=1`.
- RCCL-backed distributed expectation and sampling reductions are limited to local-domain qubits; set `ROCQ_DISTRIBUTED_COMM=rccl` or `ROCQ_REQUIRE_RCCL=1` to require RCCL on a ROCm runner. The installed CMake package probes RCCL with optional `find_package(rccl QUIET)` so exported targets can resolve RCCL when the build linked it without making non-RCCL consumers fail.
- Generic matrix/control-matrix cases outside HIP fast paths return `NOT_IMPLEMENTED` by default; set `ROCQ_ALLOW_HOST_MATRIX_FALLBACK=1` only for explicit slow/debug host fallback.
- TensorNet supports the build's compiled complex dtype (`C64` by default, `C128` in `ROCQ_PRECISION_DOUBLE` builds); METIS is optional behind `ROCQUANTUM_TENSORNET_ENABLE_METIS`, unavailable METIS/KAHYPAR pathfinders fall back to greedy with warnings, and `memory_limit_bytes` / `num_slices` now drive deterministic runtime K-sliced GEMM accumulation for pair contractions. This is still narrower than cuTensorNet-style open-index slicing, and mixed precision remains a documented future lane rather than simultaneous runtime C64/C128 execution.
- `hipDensityMatApplyChannel` accepts generic single- and multi-qubit Kraus channels, but uses correctness-first per-Kraus kernels rather than GPU-resident cuDensityMat-style channel descriptors; density-matrix sampling currently copies diagonal probability information to host before drawing shots, and dense Hermitian / full-state CSR density-matrix expectations use host correctness fallback rather than GPU-native cuDensityMat-style reductions.
- Higher-level CUDA-QX-style helpers are explicitly experimental: VQE supports Pauli-observable objectives, coefficient-preserving composite sums, numeric identity constants, CUDA-Q-style `rocq.spin.x/y/z/i` Pauli factories, Pauli sum/product terms and scalar division in operator arithmetic, scalar single-parameter gradient/optimizer inputs, and dense Hermitian / full-state CSR observables through the state-vector native/fallback path or density-matrix correctness fallback; vector-parameter QAOA and one-element vector ansatz evaluation goes through `rocq.observe()`, QAOA is a MaxCut-style ansatz/cost helper with duplicate/reversed undirected edges aggregated into weighted `0.5 * w * (I - Zi Zj)` edge terms, and QEC is limited to a single 3-qubit repetition-code syndrome round.
- `pyproject.toml` declares `numpy>=1.21` as a base runtime dependency for the canonical Python package and a `solvers` extra with `scipy>=1.10` for the experimental VQE optimizer path.
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
- `python/rocq/api.py::Circuit.expval()` now uses native Pauli expectation helpers, legacy `build()` now warns/marks Python replay execution instead of implying compiler execution, and `multi_gpu=True` construction warns/records its partial distributed contract, but the legacy surface remains separate from canonical `rocq`.
- Canonical `rocq.backends` mock state-vector and density-matrix fallbacks now emit `MockBackendWarning` when `ROCQ_ENABLE_MOCK_BACKENDS=1` is used without native ROCm bindings, so local smoke tests are not mistaken for native ROCm/cuQuantum-style execution or performance validation.
- Qiskit, PennyLane, and Cirq adapters now prefer `QuantumSimulator.measure()` for sampling, but still keep host-side fallback paths where needed for older bindings that do not expose `measure`.
- Qiskit direct `prepare_state()` and untouched-qubit `initialize()` are mapped to matrix state-preparation fallback; `reset` after prior operations is supported only in `backend.run(..., sampling=True)` through shot-by-shot `QuantumSimulator.reset_qubit()` trajectories. Later `initialize()`, statevector/estimator output for runtime-reset circuits, classically conditioned operations, and Qiskit control-flow ops remain explicit unsupported boundaries.
- `rocquantum.core.list_backends()` hides unsupported skeleton providers by default, while `list_backends(include_experimental=True)` reports their `unsupported_stub` status. `rocq list-backends` prints the same status metadata before target selection, and `set_target()` blocks those skeleton providers unless `allow_experimental=True` or `ROCQ_ENABLE_EXPERIMENTAL_PROVIDERS=1` is set for contract tests or integration development.
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

The release benchmark registry is defined in `benchmarks/benchmark_manifest.json`. It covers
state-vector fast path/fallback and fusion timing, distributed RCCL vs host fallback reductions
including dense expectation and generic matrix paths, TensorNet contraction planning, and
DensityMat channel/observable/sampling timing. The runner
emits one JSON file per benchmark plus `benchmark-summary.json`; if a native binary or ROCm
device is unavailable, it writes an explicit skipped artifact instead of pretending a result exists.

```bash
python3 benchmarks/run_release_benchmarks.py \
  --build-dir build-ci \
  --output-dir benchmark-artifacts
```

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
