# Final Gap Report

Audit date: 2026-04-05

Runtime update note (2026-04-06):

- Canonical `rocq` now exposes `execute()`, `sample()`, and `observe()` on a unified backend contract.
- Native state-vector Pauli expectation helpers are now reachable through `rocq.observe()` and `rocq.operator.get_expectation_value()`.
- Packaging has moved to a CMake-first `scikit-build-core` path and root CMake now builds `_rocq_hip_backend`, `rocq_hip`, and `rocquantum_bind`.

Audit refresh note (2026-06-10):

- AMD production ROCm documentation now identifies ROCm `7.2.4` as the production release; `7.2.0` remains historical audit context.
- Canonical `rocq` expectation and narrow GateFusion wiring have moved ahead of the original 2026-04-05 text.

Compiler/runtime execution parity, distributed execution maturity, and higher-level CUDA-QX-style libraries remain outstanding.

## Executive Summary

`rocQuantum-main` has real ROCm-native simulator value today, especially in `hipStateVec`, `hipTensorNet`, and `hipDensityMat`. It now has a release benchmark artifact registry for those areas, but it still does not have a coherent product story that matches CUDA-Q, cuQuantum, or CUDA-QX.

The repo is currently strongest as:

- a partial ROCm analogue to cuStateVec
- a narrower partial ROCm analogue to cuTensorNet
- a limited partial ROCm analogue to cuDensityMat
- a direct local simulator runtime with mixed Python surfaces

It is currently weakest where CUDA-Q and CUDA-QX depend on unified compiler/runtime integration, robust high-level observe/sample APIs, and higher-level libraries.

## Truth Matrix Snapshot

Full row-by-row matrix: `FEATURE_TRUTH_MATRIX.md`

| Area | Status | Summary |
| --- | --- | --- |
| Native HIP simulator core | IMPLEMENTED | Core statevector, tensor-network, and limited density-matrix primitives are real |
| Compiler-driven execution | PARTIAL | `compile_and_execute()` has a narrow source-level qalloc/H/X/Y/Z/S/Sdg/T/Tdg/CNOT/CZ/SWAP/CCX/MCX/CSWAP/RX/RY/RZ/P/CRX/CRY/CRZ/CP MLIR subset; default Python bindings and the unwired `ROCQUANTUM_ENABLE_MLIR_COMPILER` CMake option now fail fast instead of linking unresolved MLIR compiler symbols, and broad compiler/runtime parity is still absent |
| High-level expectation APIs | PARTIAL | Native kernels exist, and canonical `rocq` now has an experimental Clifford-only stabilizer/tableau path for Pauli propagation, but public API coverage is split and inconsistent |
| Multi-GPU / distributed | PARTIAL | Real single-node scaffolding exists, multi-node requests now fail fast as explicit unsupported stubs, and many distributed paths remain `NOT_IMPLEMENTED` |
| Packaging / install / export | PARTIAL | Native build exists, but releasable packaging is not coherent |
| Python surfaces | PARTIAL | Two divergent stacks exist, one with mock fallbacks |
| Integrations | PARTIAL | Thin adapters exist, but PennyLane/Cirq/Qiskit sampling now prefers native `measure()` where available; Qiskit simple `if_test` / `if_else`, finite `for_loop`, bounded `while_loop`, loop-local `break_loop` / `continue_loop`, and `switch_case` sampling can use state-collapsing `measure_qubit()` trajectories; Qiskit/PennyLane Pauli-observable paths avoid mandatory statevector readback; PennyLane `qml.Hermitian` and supported Qiskit dense `Operator` observables can use native dense-matrix single/batch expectation paths for supported targets; default multi-control gates route to native `MCX` / `CSWAP`; PennyLane `SparseHamiltonian` can use native CSR moments, including local batch readouts, with a correctness-first CSR statevector fallback; self-hosted ROCm CI now has a native binding/PennyLane/Qiskit/Cirq Bell-state smoke path, but artifacts are still required for hardware proof |
| Higher-level libraries | PARTIAL | Experimental VQE/QAOA/repetition-code helpers exist, including single-round repetition-code syndrome/correction analysis, but they are not robust CUDA-QX analogues |

## ROCm Integration Maturity

Overall ROCm maturity: `PARTIAL`

- Strongest point: native HIP simulator components
- Weakest point: productization, packaging, support policy, and runtime validation breadth

Current truth:

- Non-experimental CI ROCm lane: `6.2.2`
- Experimental CI ROCm lane: `7.2.4`
- Latest production ROCm verified from official AMD docs during this refresh: `7.2.4`
- Newest AMD GPU target verified during this audit: `MI355X` / `gfx950`
- Distributed non-local single-qubit, controlled single-qubit, CNOT/CZ, and generic matrix/control-matrix correctness fallback exists only as explicit slow/debug mode via `ROCQ_DISTRIBUTED_FALLBACK_MODE=host` or `ROCQ_ENABLE_DISTRIBUTED_HOST_FALLBACK=1`.
- RCCL is now wired for local-domain distributed expectation and sampling probability reductions when `ROCQ_HAVE_RCCL` is available, but this is still not general distributed execution.

Recommended compatibility plan:

- Tier 1: ROCm `7.2.4`, `gfx950`, `gfx942`, `gfx90a`, Linux x86_64, Python `3.10`-`3.12`
- Tier 2 best-effort: ROCm `6.4.0`, `gfx908`, selected workstation targets such as `gfx1100`, `gfx1101`, `gfx1030`
- Recommended minimum release-grade GPU target: `gfx90a`

## CUDA-Q Gap Summary

Compared with the official CUDA-Q baseline (`https://nvidia.github.io/cuda-quantum/latest/`), the largest gaps are:

- no release-wired compile-and-execute loop by default
- no fully unified compiler/runtime/kernel story; the default bindings now separate the canonical runtime compiler guard from the legacy conceptual MLIR holder
- only a narrow mid-circuit measurement and classical-control story: Qiskit simple `if_test` / `if_else`, finite `for_loop`, bounded `while_loop`, loop-local `break_loop` / `continue_loop`, and `switch_case` sampling trajectories work, but estimator/statevector dynamic semantics remain open
- no broad arbitrary-operator expectation coverage beyond the supported Pauli, dense Hermitian / Qiskit dense Operator, and full-state CSR sparse paths
- no bounded, tested `mqpu`-style distributed story; multi-node requests are explicit unsupported stubs

What the repo does have:

- a direct local simulator path
- some native observable kernels in the backend
- multiple beginnings of a compiler stack, now with clearer default-build guards

What it lacks is the integration layer that makes those pieces act like CUDA-Q rather than a collection of subsystems.

## cuQuantum Gap Summary

Compared with the official cuQuantum baseline (`https://docs.nvidia.com/cuda/cuquantum/latest/`), the repo is closest in scope but still incomplete:

- `hipStateVec` is the strongest analogue, with multi-control/single-target controlled-matrix fast paths now covered, but broader controlled-matrix breadth and distributed completeness still lag behind
- `hipTensorNet` has a real core and now reports optimizer/dtype/slicing capabilities, METIS is explicitly optional, and KaHyPar configure attempts fail fast until release-wired, but the optimizer stack and broad runtime sliced execution still lag behind cuTensorNet expectations
- `hipDensityMat` exists, and now has single-qubit Kraus channels plus density sampling that reduces measured-qubit marginals on GPU before host-side shot drawing, but GPU-resident RNG/CDF sampling, optimized channel scheduling, and broader observable coverage lag behind cuDensityMat expectations

For state-vector matrix application, unsupported cases now fail clearly unless `ROCQ_ALLOW_HOST_MATRIX_FALLBACK=1` is set for explicit slow/debug fallback.
Dense matrix moments now have local single-state and batched fused HIP reductions, so supported Hermitian variance paths no longer need to scan the state once for `<M>` and again for `<M^2>`.

The main difference is not just feature count; it is product completeness and test-backed breadth.

## CUDA-QX Gap Summary

Compared with the official CUDA-QX baseline (`https://github.com/NVIDIA/cudaqx`), the repo now has a small experimental supported subset but is not yet competitive at the higher-level library layer.

Current state:

- QEC is limited to a 3-qubit repetition-code single-round helper with sampled syndrome/correction analysis
- VQE is limited to Pauli-observable objectives through `rocq.observe()` plus numerical gradient helpers
- QAOA is limited to a MaxCut-style ansatz/cost helper
- no repeated-round/noise-aware decoder, robust solver, or hybrid-library stack was found

This is a P2 area. It should not be used to market parity while P0 and P1 remain open.

## Top 10 Missing Or Misleading Areas

1. `compile_and_execute()` has an MVP source implementation, but the default Python binding now fails fast unless the experimental MLIR compiler stack is actually linked; it is still not a full CUDA-Q-style compiler runtime.
2. Multi-GPU support is partial and previously overclaimed.
3. Native expectations exist but the public API story is split and misleading.
4. Two divergent Python stacks exist without one canonical answer.
5. Packaging/install/export does not describe one releasable artifact.
6. Gate fusion exists and is wired for narrow canonical `rocq` spans; unsupported fusion queue entries fail instead of being silently dropped, but legacy `python/rocq` and broader patterns remain unfused.
7. `hipTensorNet` breadth is overstated relative to what is built and tested.
8. `hipDensityMat` is real but still too narrow for broad noisy-simulation claims: generic Kraus channels and GPU-side marginal probability reduction for density sampling exist, while GPU-resident shot workflows, optimized channel scheduling, and broad observable coverage remain incomplete. A small Clifford-only stabilizer/tableau backend now reduces the broader simulator-portfolio gap for Pauli propagation, but it is not GPU-accelerated and does not cover noise or non-Clifford circuits.
9. Integrations are still thin adapters, though PennyLane/Cirq/Qiskit sampling now prefers the native simulator `measure()` path where available, Qiskit simple `if_test` / `if_else`, finite `for_loop`, bounded `while_loop`, loop-local `break_loop` / `continue_loop`, and `switch_case` sampling can use state-collapsing `measure_qubit()` trajectories, PennyLane/Qiskit Pauli-observable paths use native expectation helpers where possible, PennyLane `qml.Hermitian` and supported Qiskit dense `Operator` observables can use native dense-matrix single/batch expectation paths for supported targets, Qiskit/PennyLane default multi-control gates reach native `MCX` / `CSWAP`, PennyLane `SparseHamiltonian` can use native CSR moments, including local batch readouts, with a CSR statevector fallback, and self-hosted ROCm CI now has a native framework Bell-state smoke path whose artifacts are needed for hardware proof.
10. CUDA-QX-style libraries are still shells.

## P0 / P1 / P2 Roadmap

### P0

- Truth-fix docs, bindings, and tests
- Gate compiler/runtime parity claims
- Make multi-GPU partial support explicit
- Make expectation-value limitations explicit
- Publish a Linux-first ROCm compatibility statement

### P1

- Unify Python surfaces
- Expose native expectations in the canonical API
- Wire gate fusion into the active runtime
- Repair install/export and packaging
- Expand ROCm runtime CI

### P2

- Complete compiler-driven runtime
- Expand distributed execution
- Expand higher-level solver and QEC libraries beyond the experimental VQE/QAOA/repetition-code subset

## Compatibility Plan For Latest And Older AMD GPUs

| Tier | ROCm | Architectures | Status |
| --- | --- | --- | --- |
| Tier 1 | `7.2.4` | `gfx950`, `gfx942`, `gfx90a` | Recommended target |
| Tier 2 | `6.4.0` | `gfx908`, `gfx1100`, `gfx1101`, `gfx1030` | Best-effort |
| Legacy not recommended | older pre-`gfx90a` targets | including `gfx906` | Do not advertise as supported in this pass |

## Evidence Appendix

Primary code evidence:

- `rocqCompiler/MLIRCompiler.cpp`
- `bindings.cpp`
- `python/rocq/api.py`
- `rocq/kernel.py`
- `rocq/backends.py`
- `rocquantum/src/hipStateVec/hipStateVec.cpp`
- `rocquantum/src/hipTensorNet/hipTensorNet.cpp`
- `rocquantum/src/hipDensityMat/hipDensityMat.cpp`
- `rocquantum/src/simulator.cpp`
- `.github/workflows/rocm-linux-build.yml`

Primary truth docs produced in this pass:

- `CURRENT_STATE_AUDIT.md`
- `FEATURE_TRUTH_MATRIX.md`
- `ROCM_INTEGRATION_AUDIT.md`
- `TOP_GAPS_AND_PRIORITIES.md`
- `IMPLEMENT_NOW_PLAN.md`

External baselines:

- CUDA-Q: `https://nvidia.github.io/cuda-quantum/latest/`
- cuQuantum: `https://docs.nvidia.com/cuda/cuquantum/latest/`
- CUDA-QX: `https://nvidia.github.io/cudaqx`
- ROCm release history: `https://rocm.docs.amd.com/en/latest/release/versions.html`
- ROCm compatibility matrix: `https://rocm.docs.amd.com/en/latest/compatibility/compatibility-matrix.html`
- ROCm Linux requirements: `https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html`

Fork note:

- The local repo audit was sufficient to classify current behavior. This pass did not find any current-code dependency that required a deeper fork-divergence override from the user’s `cuQuantum`, `cuda-quantum`, or `cudaqx` forks.

Verification limit:

- No local ROCm runtime, CMake, Ninja, or HIP compiler was available in this shell, so native runtime validation remains a ROCm-Linux CI or container task.
