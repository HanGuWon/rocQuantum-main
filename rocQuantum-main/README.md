# rocQuantum

`rocQuantum` is an experimental ROCm-first quantum computing repository centered on native HIP simulator components:

- `hipStateVec`
- `hipTensorNet`
- `hipDensityMat`

The repo also contains partially implemented compiler, Python, provider, and framework-integration surfaces inspired by CUDA-Q, cuQuantum, and CUDA-QX. Its current product position is **a ROCm-native simulation SDK with a CUDA-Q-inspired Python API**. It is not a drop-in replacement for, or an end-to-end equivalent of, those NVIDIA stacks.

## Comparison Baseline And Evidence Policy

This capability summary was refreshed on 2026-07-16 against the released baselines below:

- cuQuantum SDK `26.06.0`: cuStateVec `1.14.0`, cuTensorNet `2.13.0`, cuDensityMat `0.6.0`, cuPauliProp `0.4.0`, and cuStabilizer `0.4.0`
- CUDA-Q `0.15.0`
- CUDA-QX `0.6.0` (whose 0.6 release line targets CUDA-Q 0.14 rather than CUDA-Q 0.15)

Capability claims use ordered evidence stages: `source-present`, `host-contract-tested`, `native-single-GPU-verified`, `native-multi-GPU-verified`, and `performance-verified`. This workstation has no AMD GPU or ROCm runtime, so this refresh makes **no** claim at any of the three native/performance stages. Source inspection and passing CPU/mock tests must not be interpreted as HIP correctness or performance evidence.

## Current Reality

Source-present today (not locally GPU-verified):

- Native HIP state-vector simulation for core named gates, sampling, measurement, and several expectation-value primitives
- Native tensor-network contraction core
- Native density-matrix core with named noise channels, generic single- and multi-qubit Kraus channel APIs, decomposed CCX/CSWAP helpers in the canonical backend, and density sampling with GPU-side marginal probability reduction
- Direct simulator execution through the active local runtime path
- A CPU-buildable native TableGen/MLIR/LLVM compiler path with generated `!quantum.qubit` / `!quantum.result` types, terminal `quantum.mz`, explicit `rocq-qir-static-pipeline` / `rocq-qir-base-pipeline` registrations, `rocq-opt`, a strict `rocq-translate` CLI, QIR 2 static-custom and labeled terminal-MZ Base Profile output, LLVM verification, LLVM IR/bitcode/generic-host relocatable-object emission, and an opt-in content-addressed artifact cache
- CUDA-Q-style local target/result APIs, `get_state()` alias, and context-preserving host-side `AsyncResult` wrappers for `get_state_async()`, `execute_async()`, `sample_async()`, `observe_async()`, `evolve_async()`, and `compile_and_execute_async()`

Only partial today:

- Compiler breadth beyond host-specialized Python composition and straight-line terminal-measurement QIR: native typed SSA arguments/results/helper functions, mid-circuit measurement/reset/`read_result`, branches/loops, adaptive or dynamic-resource profiles, a runnable ORC-to-QIS runtime bridge, arbitrary multi-control synthesis, and a reusable installed compiler SDK/pass-plugin surface remain absent
- `compile_and_execute()` for a narrow MLIR subset: qalloc, H/X/Y/Z/S/Sdg/T/Tdg, CNOT/CZ/SWAP/CCX/MCX/CSWAP, RX/RY/RZ/P, CRX/CRY/CRZ/CP
- Generic matrix and controlled-unitary coverage
- Multi-GPU / distributed execution
- Observable breadth and density-matrix GPU-fast sampling coverage
- Native wheel and native install-tree validation; the clean host-only wheel path is validated separately
- PennyLane, Cirq, and Qiskit adapter maturity

Not implemented as NVIDIA-equivalent components today:

- End-to-end compiler-driven execution parity with CUDA-Q beyond the supported MVP subset
- Release-grade distributed multi-GPU support
- Production CUDA-QX parity beyond the expanded host reference layer: GPU-native gradients/optimizers, GQE/state preparation, broad chemistry drivers, surface-code/DEM/tensor-network/realtime QEC, and distributed workflows remain absent; `rocquantum.solvers.solver_capabilities()` and `rocquantum.qec.qec_capabilities()` expose the exact boundary
- A dedicated cuPauliProp analogue; Pauli expectations and the CPU Clifford helper do not implement cuPauliProp's propagation, truncation, view, noise, and reverse-AD contract
- A dedicated GPU cuStabilizer analogue; the Python `stabilizer` / `tableau` / `clifford` aliases are a CPU-only, Clifford-only behavioral prototype without the noisy many-shot, detector-error-model, GPU GF(2), or JAX surface of cuStabilizer

## Audit Documents

This repo now includes an audit-first truth set in the inner repo root:

- `CURRENT_STATE_AUDIT.md`
- `FEATURE_TRUTH_MATRIX.md`
- `ROCM_INTEGRATION_AUDIT.md`
- `TOP_GAPS_AND_PRIORITIES.md`
- `FINAL_GAP_REPORT.md`
- `IMPLEMENT_NOW_PLAN.md`
- `CUDA_Q_CUDAQX_IMPLEMENTATION_REPORT.md`
- `NATIVE_COMPILER_STATUS.md`

Use those files as a review snapshot. Code, clean build results, executed tests, and native runner artifacts take precedence; `FEATURE_TRUTH_MATRIX.md` records the conservative evidence stage for each area.

## Repo Layout

- `rocquantum/src/hipStateVec`: native state-vector kernels and distributed scaffolding
- `rocquantum/src/hipTensorNet`: tensor-network contraction core
- `rocquantum/src/hipDensityMat`: density-matrix and limited noise support
- `rocquantum/src/simulator.cpp`: public C++ simulator wrapper
- `rocqCompiler/`: optional LLVM/MLIR 22.1.x native compiler, generated qubit/result dialect, direct static-custom and terminal-MZ Base Profile QIR 2 lowering, verified IR/bitcode/host-object artifacts, content-addressed CLI cache, CPU-only tools/tests, and optional HIP execution bridge
- `rocq/`: canonical Python runtime with direct execution, target/result/noise APIs, typed `make_kernel`, inspection/translation tools, and small-system CPU reference dynamics
- `python/rocq/`: separate legacy-style Python surface whose `_rocq_hip_backend` extension is built from the top-level CMake graph
- `integrations/`: PennyLane, Cirq, and Qiskit adapters

## Support Policy For This Audit Pass

- Primary release target: Linux x86_64
- Latest production ROCm target verified on 2026-06-10: `7.2.4`
- Native ROCm/packaging CMake minimum: CMake `3.21`; optional MLIR compiler minimum: CMake `3.24`
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
| `rocqCompiler` | Release-wired optional compiler graph: TableGen `!quantum.qubit` / `!quantum.result` and operations including terminal `quantum.mz`; direct Quantum-to-LLVM/QIR conversion; explicit static/base pipelines; `rocq-opt`; strict `rocq-translate`; QIR 2 static-custom and labeled terminal-MZ Base Profile shaping; LLVM verification; deterministic LLVM IR, bitcode, and generic-host PIC relocatable objects; and a self-validating content-addressed CLI cache. Base Profile LLVM IR/bitcode is restricted to `-O0`, while host objects permit `-O0` through `-O3` and retain unresolved QIS/runtime symbols. HIP `compile_and_execute()` remains the narrow qalloc/core-gate subset, and variadic MCX fails closed. LLVM/MLIR 22.1.x is pinned; only the tools are installed, while compiler libraries/headers remain build-tree-only. |
| Top-level `rocq` | Canonical runtime path with native execute/sample/observe wiring, `rocq.runtime_capabilities()` metadata for the canonical/legacy runtime boundary, explicit bool-safe state-vector `enable_fusion=` execution option, canonical backend-name validation, top-level phase-gate exports (`tdg`/`tdag`, `p`/`phase`, `cp`/`cphase`), strict positive-integer `qvec` allocation, direct backend size, shot, selected-qubit, spin-factory target, finite square power-of-two dense observable matrix/target, square power-of-two sparse shape/CSR payload, finite observable-coefficient, quiet noise-channel target/probability validation, and direct density-noise target/probability validation, integer in-range, duplicate multi-qubit, and arity-invalid gate-target validation plus finite real gate-parameter validation during kernel recording and direct backend gate dispatch, duplicate-combined Pauli expectations including inside mixed sums, coefficient-aware duplicate dense-Hermitian/CSR sum readout reuse, zero-coefficient matrix/sparse sum-term elision, coefficient-preserving composite observable sums, numeric identity constants, CUDA-Q-style `rocq.spin.x/y/z/i` Pauli factories, Pauli sum/product terms and scalar division in operator arithmetic, density-matrix correctness fallback for dense Hermitian / full-state CSR observables, an explicit CPU mock state-vector fallback for local named-gate statevector contract tests, and an experimental Clifford-only `stabilizer` / `tableau` / `clifford` backend for Pauli propagation |
| Higher-level helpers | Experimental host reference layer with legacy VQE/MaxCut helpers plus functional VQE trace/results, generic real-Pauli QAOA and pools, ADAPT-VQE, limited precomputed-integral/Jordan-Wigner chemistry, QEC Code/Decoder registries, odd-distance repetition and Steane metadata, LUT/BP decoders, and seeded code-capacity sampling; capability metadata exposes unsupported GPU/distributed/production boundaries |
| `python/rocq` | Top-level CMake-built legacy compatibility surface; Pauli expectations, batched state allocation/readback, same-target single-qubit plus CNOT-adjacent GateFusion spans, and bool-safe finite input validation for circuit sizes, gate targets/angles, samples, and Pauli coefficients now use explicit Python contracts; legacy `build()` records whether execution is conceptual MLIR or Python circuit replay and warns when replay is used, while broader fusion and runtime unification still need consolidation |

## Important Limitations

- `rocq.make_kernel()` is a typed host-specialized dynamic builder. In addition to scalar/sequence expressions, fixed qalloc, canonical gates, and CUDA-Q-sign-compatible `exp_pauli`, it supports device-style `QuakeValue` arguments, `call` / `apply_call` through static inlining, exact gate-level adjoint synthesis for the supported canonical operations, a fail-closed subset of canonical controlled synthesis, and terminal `mz` / `mx` / `my` declarations. Terminal measurements select sampling qubits and emit labeled `quantum.mz` / `!quantum.result` MLIR; X/Y measurements are represented by exact basis changes followed by Z measurement. This remains host gate-IR specialization, not native typed SSA or a native MLIR JIT: typed returns, measurement-bearing callee composition, `read_result`, mid-circuit reset/feedback, branches, and loops are unsupported, and arbitrary multi-control synthesis fails closed.
- `rocq.set_target()` / `rocq.target()` select one context-local backend, `SampleResult` / `ObserveResult` retain dict/float compatibility, and `AsyncResult.get()` preserves the submission context. `sample()` / `sample_async()` accept CUDA-Q-style `shots_count` and default to 1000 shots when no legacy shot count is supplied. Async APIs accept only `qpu_id=0`; this is an explicit single-logical-QPU contract, not a multi-QPU scheduler.
- `rocq.operator_to_matrix()`, `Schedule`, `evolve()`, and `evolve_async()` are small-system CPU correctness utilities. Closed-system unitary and Lindblad RK4 paths are host-contract-tested, but they are not distributed/GPU dynamics or performance implementations.
- `rocqCompiler::MLIRCompiler::compile_and_execute()` has a source-level MVP subset for qalloc, H/X/Y/Z/S/Sdg/T/Tdg, CNOT/CZ/SWAP/CCX/MCX/CSWAP, RX/RY/RZ/P, and CRX/CRY/CRZ/CP; it does not execute measurement results. Independently, the one-argument compiler constructor performs offline artifact emission without creating a HIP backend. Both profiles accept one straight-line, no-argument/no-result source function with one static qalloc. `qir-v2-static` preserves the measurement-free `custom` profile with zero results. `qir-v2-base` requires labeled terminal `quantum.mz` operations, lowers them to static result handles, shapes initialize/body/measurements/output blocks, records each output, and returns `i64 0`. The project structural verifier plus LLVM `llvm-as` / `opt -passes=verify` checks this terminal-measurement subset. Native typed SSA/functions, adaptive control, dynamic resources, extra source functions, bad arity/types, duplicate operands, non-finite angles, and variadic MCX remain fail-closed boundaries. `rocq.compiler_capabilities()` reports offline artifacts and GPU execution separately.
- Canonical `rocq.QuantumKernel.qir(qir_profile=...)` and `emit_artifact(kind=..., optimization_level=..., cache_dir=...)` do not require an AMD GPU. They prefer the offline compiler-enabled binding where applicable and otherwise invoke an installed `rocq-translate` by argument vector and standard input without a shell. LLVM IR and bitcode support `-O0` through `-O3` for `qir-v2-static`; Base Profile IR/bitcode requires `-O0` so generic LLVM optimization cannot erase the required four-block shape. Generic-CPU PIC objects for the build host support `-O0` through `-O3`, are not executables, and intentionally leave QIS/runtime symbols unresolved. Missing tools, unsupported combinations, corrupt cache entries, and compiler failures raise actionable exceptions rather than returning an `"Error:"` artifact.
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
- StateVec and TensorNet support a build-selected complex dtype (`C64` by default, `C128` with the real CMake option `-DROCQ_PRECISION_DOUBLE=ON`). The precision definition is PUBLIC on their installed targets so downstream headers use the same ABI; the legacy `_rocq_hip_backend` exposes `COMPILED_COMPLEX_DTYPE` and `COMPILED_COMPLEX_ITEMSIZE`, and its Python readback/matrix paths preserve the selected precision through explicit `std::complex<float>` / `std::complex<double>` conversion rather than relying on a HIP complex NumPy ABI. DensityMat remains a separate C64/`hipComplex` implementation and is not covered by this option. The ROCm 6.2.2 workflow contains a C128 compile plus C128/METIS install-consumer gate and builds separate C64/C128 native wheels; fresh external environments assert each wheel's dtype/itemsize and run a device-free complex round trip before native import is accepted. Those gates are source-defined but have not been run on this host. METIS is optional behind `ROCQUANTUM_TENSORNET_ENABLE_METIS`; enabled exports reference the relocatable `METIS::METIS` target and the installed package rediscovers headers/library before loading its targets. The same stable workflow combines METIS with its C128 install-consumer lane. `ROCQUANTUM_TENSORNET_ENABLE_KAHYPAR=ON` fails fast because KaHyPar is not release-wired, unavailable pathfinders fall back to greedy with warnings, and `memory_limit_bytes` / `num_slices` now drive deterministic runtime K-sliced GEMM accumulation for pair contractions. `get_tensornet_capabilities()` reports that runtime slicing kind as `limited_pair_contraction_k_sliced_gemm`, reports the hard 16-mode tensor permutation limit, and marks open-index slicing, mixed precision, and simultaneous runtime C64/C128 support as unsupported. Python TensorNet contract calls retrieve the active simulator stream and reuse a rocBLAS handle owned by the TensorNet wrapper instead of passing placeholder handles. This is still narrower than cuTensorNet-style open-index slicing and high-rank permutation coverage, and mixed precision remains a documented future lane rather than simultaneous runtime C64/C128 execution.
- The native Python boundary now fails closed on host-auditable lifetime and size contracts. State `DeviceBuffer` views carry their owning handle plus allocation generation and are rejected after free/reallocation, across handles, or for the wrong qubit count; device matrices must have the exact overflow-checked `2^targets × 2^targets × sizeof(rocComplex)` byte size; and `GateFusion.process_queue()` revalidates the retained state before invoking a native object that stores a raw pointer. TensorNet keeps added tensors alive and snapshots pointer/shape/label/stride metadata, tensor storage uses RAII, permutation maps/strides/capacity/aliasing are validated before kernels, and DensityMat Python holders and temporary device matrices use single-owner exception-safe cleanup. These are source and host-contract checks; they are not a substitute for HIP sanitizer or device execution.
- `rocq.density_matrix_capabilities()` exposes the canonical density-matrix boundary without running a kernel: `hipDensityMatApplyChannel` accepts generic single- and multi-qubit Kraus channels, but uses correctness-first per-Kraus kernels rather than GPU-resident cuDensityMat-style channel descriptors; canonical `DensityMatrixBackend` decomposes `CCX` / two-control `MCX` / `CSWAP` through supported density-matrix primitive gates while larger `MCX` still needs an explicit ancilla policy; density-matrix sampling reduces measured-qubit marginal probabilities on the GPU before drawing shots on host, and dense Hermitian density-matrix expectations now use a native HIP reduction for up to four target qubits, while larger dense observables and full-state CSR density-matrix expectations still use host correctness fallback rather than cuDensityMat-style descriptor reductions.
- Higher-level CUDA-QX-style helpers are explicitly experimental: VQE supports public one-shot energy evaluation, canonical `QuantumOperator` objective validation, ansatz-kernel validation, Pauli-observable objectives, coefficient-preserving composite sums, numeric identity constants, CUDA-Q-style `rocq.spin.x/y/z/i` Pauli factories, Pauli sum/product terms and scalar division in operator arithmetic, `TypeError` diagnostics for unsupported observable arithmetic operand types, scalar single-parameter gradient/optimizer inputs, bool-safe finite real parameter and finite-difference step validation, supported gradient-method validation, canonical runtime backend-name validation, bool-safe verbose-option validation, ansatz positional parameter-count validation, optimizer-trace-preserving gradient probes, finite-real observed energy and optimizer `fun`/`x` result value/count validation, custom optimizer `minimize()` validation, string-keyed `SciPyOptimizer` option validation/copying, and dense Hermitian / full-state CSR observables through the state-vector native/fallback path or density-matrix correctness fallback; vector-parameter QAOA and one-element vector ansatz evaluation goes through `rocq.observe()`, QAOA is a MaxCut-style kernel/cost/solve helper with edge-list or edge-weight mapping normalization, duplicate/reversed undirected edges aggregated into weighted `0.5 * w * (I - Zi Zj)` edge terms, validated edge containers/shapes, integer endpoints, and bool-safe finite real weights/parameters, and `solve_maxcut_qaoa()` now minimizes the negated cost operator while reporting `optimal_cut_value` for the maximized cut objective; QEC covers generic sampled stabilizer-fragment orchestration plus a 3-qubit repetition-code syndrome subset with canonical backend-name, bool-safe verbose-option, code/decoder callable-interface, non-empty non-mapping stabilizer-fragment sequence, callable-or-None initial-state kernel, logical-operator result, and decoder-correction result validation, positive-integer shot/round/num_qubits and ancilla-index validation, minimum-5-qubit circuit generation, bool-safe one- or two-bit count keys and syndrome-bit validation, single-round sampling, sequential repeated-round histogram/correction aggregation, and independent syndrome readout-error mitigation. `rocquantum.solvers.solver_capabilities()` and `rocquantum.qec.qec_capabilities()` expose those supported and unsupported subsets plus the current canonical `supported_backends`, execution scopes, and hardware-evidence boundaries for programmatic CUDA-QX comparison. This is not a native-adjoint solver stack, distributed hybrid workflow scheduler, fault-tolerant decoder stack, or general noise-aware QEC library.
- `pyproject.toml` declares `numpy>=1.21` as a base runtime dependency for the canonical Python package and a `solvers` extra with `scipy>=1.10` for the experimental VQE optimizer path. The tested adapter ranges are Cirq 1.x on Python 3.9+, Qiskit `>=2.4,<3` on Python 3.10+, and PennyLane `>=0.45,<0.46` on Python 3.11+. Older PennyLane versions are deliberately not advertised: import-only checks were insufficient, and their device/template behavior did not satisfy the current adapter contract; untested future major/minor APIs are capped until their contract suite passes. Adapter-local `setup.py` files are compatibility installers only and mirror these ranges, while the supported project install path remains the repository root.
- Clean PEP 517 host-only builds now produce a source-only `py3-none-any` wheel with no native sources or binaries. The 2026-07-16 current-host Python 3.11 regression reports `979 passed, 15 skipped, 512 subtests passed`; the skips are AMD-device, native-binding, or external-credential dependent and do not count as hardware evidence. The CI definitions continue to cover Python 3.9 through 3.13, while isolated installation, `rocq` / `rocquantum` / `rocq_cli` import, installed `rocq --help`, and copied-example execution are separate package gates rather than inferred from that one host count. Native scikit-build wheels must opt in with `ROCQ_BUILD_NATIVE=1`. Their three extension modules use a wheel-relative `$ORIGIN/<libdir>` install RPATH, and the stable ROCm build workflow builds C64 and C128 wheels, checks `readelf`/`ldd`, installs each in a fresh environment, and imports it outside the source/build tree. This is a source-defined CI gate; native compile, link, install, and import remain unverified on this non-ROCm host until a retained green artifact exists.
- The supported examples use the canonical `rocq` API. `tests/test_examples_contract.py` executes all 19 example scripts in isolated subprocesses with explicit mock/capability handling; hardware-only features report their boundary rather than presenting CPU execution as ROCm proof.
- Native CTest definitions now register labeled single-GPU StateVec, DensityMat, TensorNet contraction/SVD, and RCCL-required multi-GPU inter-rank regressions. Topology-optional single-GPU tests return skip code 77 when no device is visible; `StateVecSingleGpuSmoke` is isolated with `HIP_VISIBLE_DEVICES=0`. The required multi-GPU evidence test does not treat code 77 as a skip, so missing two-GPU/RCCL topology fails the nightly evidence gate. These definitions are `source-present`; they have not been executed on this host.
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
- The self-hosted ROCm runtime workflow is configured to build native Python bindings and run `scripts/native_framework_smoke.py`, a Bell-state smoke check through `rocquantum_bind`, PennyLane, Qiskit, and Cirq. It requires `--require-native-rocm-evidence` and records the `/dev/kfd` probe in uploaded artifacts. The workflow definition is not itself native evidence; only a retained successful actual-device artifact qualifies.
- Qiskit direct `prepare_state()` and untouched-qubit `initialize()` are mapped to matrix state-preparation fallback; `reset` after prior operations plus simple `if_test` / `if_else`, finite `for_loop`, bounded `while_loop`, loop-local `break_loop` / `continue_loop`, and `switch_case` are supported only in `backend.run(..., sampling=True)` through shot-by-shot `QuantumSimulator` trajectories. Later `initialize()`, statevector/estimator output for runtime-reset or dynamic-control circuits, and broader Qiskit control-flow semantics remain explicit unsupported boundaries.
- `rocquantum.core.list_backends()` hides unsupported skeleton providers by default, while `list_backends(include_experimental=True)` reports their `unsupported_stub` status, disabled job-submission flag, unsupported reason, and missing authentication/payload/submission/status/result capabilities. `rocq list-backends` prints the same status metadata before target selection, and `set_target()` blocks those skeleton providers unless `allow_experimental=True` or `ROCQ_ENABLE_EXPERIMENTAL_PROVIDERS=1` is set for contract tests or integration development.
- The Qristal bridge now checks for the real local Qristal SDK CLI (`qristal`, or `ROCQ_QRISTAL_CLI`) and invokes it through `subprocess.run()` instead of returning a mocked local histogram; missing SDK/CLI and failed executions raise explicit backend errors.
- Several non-skeleton provider backends remain thin clients and require provider credentials for real validation.

## Build

The default Python artifact is the host-only universal wheel. The native wheel selector also switches its compatibility tags and must only be used on a Linux ROCm build host:

```bash
python -m build --wheel
ROCQ_BUILD_NATIVE=1 python -m build --wheel  # native path; not validated on this host
```

For a direct native CMake build, use an out-of-tree build directory:

```bash
cmake -S . -B build-ci -G Ninja \
  -DBUILD_TESTING=ON \
  -DROCQUANTUM_BUILD_BINDINGS=ON \
  -DROCQUANTUM_BUILD_NATIVE=ON \
  -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ \
  -DCMAKE_PREFIX_PATH=/opt/rocm
cmake --build build-ci --parallel
```

The compiler can be built and tested without ROCm or an AMD GPU. It intentionally pins the
same LLVM/MLIR 22.1 API line used by the current CUDA-Q compiler; arbitrary MLIR versions fail
at configure time:

```bash
cmake -S . -B build-compiler -G Ninja \
  -DROCQUANTUM_BUILD_NATIVE=OFF \
  -DROCQUANTUM_ENABLE_MLIR_COMPILER=ON \
  -DMLIR_DIR=/opt/llvm-22.1/lib/cmake/mlir \
  -DLLVM_DIR=/opt/llvm-22.1/lib/cmake/llvm \
  -DBUILD_TESTING=ON
cmake --build build-compiler --parallel
ctest --test-dir build-compiler --output-on-failure
build-compiler/rocqCompiler/rocq-translate rocqCompiler/tests/bell.mlir
build-compiler/rocqCompiler/rocq-translate --profile=qir-v2-base \
  --emit=llvm-bc -O0 -o bell-base.bc rocqCompiler/tests/base_profile.mlir
build-compiler/rocqCompiler/rocq-translate --emit=object -O2 \
  --cache-dir .rocq-cache -o bell.o rocqCompiler/tests/bell.mlir
cmake --install build-compiler --prefix "$HOME/.local/rocq-compiler"
```

`ROCQUANTUM_ALLOW_UNSUPPORTED_MLIR=ON` exists only for development smoke tests; it is not a
release compatibility claim. `rocq-translate` accepts exactly one file or `-` input, strict profile,
format, `-O0`-`-O3`, output, and cache options, writes file outputs atomically, uses exit status 2
for usage errors and 1 for compilation/cache/I/O failures, and never writes binary bitcode/object
payloads to stdout. Its cache key includes the source, profile, qubit constraint, artifact kind,
optimization level, target contract, LLVM version, and compiler-source fingerprint; entries use a
self-validating envelope and fail closed when corrupt. This cache is an integrity/determinism
mechanism for an explicitly trusted directory, not an authenticity boundary. When a
compiler-enabled Python binding is unavailable,
`QuantumKernel.qir()` automatically uses an installed `rocq-translate` from `PATH`; set
`ROCQ_TRANSLATE_EXECUTABLE` to an explicit executable path to override discovery. This fallback
passes MLIR over stdin without invoking a shell and still does not require ROCm or an AMD GPU.

Select the StateVec/TensorNet C128 ABI with `-DROCQ_PRECISION_DOUBLE=ON`; consumers of those
exported targets inherit the same compile definition. C64 and C128 are separate build artifacts,
not simultaneously selectable runtime dtypes. DensityMat remains C64-only.

To validate the installed CMake package on a ROCm build host, install the build tree and configure
a downstream consumer that resolves core, StateVec, TensorNet, and DensityMat symbols and runs
device-free destroy/capability calls. The optional compiler tools install only when their build
option is enabled. The compiler's static libraries, headers, aliases, cache API, and pass pipeline
API are intentionally build-tree-only and are not a supported installed/exported C++ SDK; only
`rocq-opt` and `rocq-translate` are installed. Compiler internals remain private to avoid forcing
LLVM/MLIR on core SDK consumers:

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
proof. Every manifest entry declares an exact nonempty required case set, per-case required metrics,
and a subprocess timeout. Duplicate, missing, or unexpected cases; boolean, negative, or non-finite
metrics; missing required metrics; stale output; and timeout all fail closed even when the process
exits zero. Every declared `cases[].status` must also be integer zero. The distributed executable
independently queries its distributed backend, requires at least two GPUs, and emits the exact RCCL
or host-fallback backend, so visibility variables or a one-GPU `/dev/kfd` probe cannot become
distributed evidence.
Passing `--require-native-performance-evidence` makes self-hosted ROCm jobs fail unless at
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

The self-hosted ROCm runtime workflow restores the previous benchmark summary and bounded history,
passes the summary as a baseline when available, and requires at least one passed native benchmark;
its one-GPU topology may truthfully skip the distributed entry. The two-or-more-GPU nightly adds
`--require-all-native-benchmark-evidence`, so every declared native benchmark must pass there.
Each workflow saves a new baseline only when its applicable native-evidence gates succeed.

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

- CUDA-Q `0.15.0`: `https://github.com/NVIDIA/cuda-quantum/releases/tag/0.15.0`
- cuQuantum SDK `26.06.0`: `https://github.com/NVIDIA/cuQuantum/releases/tag/v26.06.0`
- cuQuantum component documentation: `https://docs.nvidia.com/cuda/cuquantum/latest/`
- CUDA-QX `0.6.0`: `https://github.com/NVIDIA/cudaqx/releases/tag/0.6.0`

This repo is currently a ROCm-native simulation SDK with a CUDA-Q-inspired Python API, not a finished cuQuantum/CUDA-Q/CUDA-QX equivalent.
`rocq.runtime_capabilities()["limits"]` exposes the current state-vector and density-matrix size boundaries, including the 60-qubit state-vector size-arithmetic ceiling used by canonical, framework, and legacy Python runtime validation.
`rocq.density_matrix_capabilities()["limits"]` exposes the current native density-matrix hard bounds, including the 30-qubit dense-size arithmetic ceiling and four-target native dense-observable/Kraus-channel paths.
