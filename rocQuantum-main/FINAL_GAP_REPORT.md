# Final Gap Report

Audit date: 2026-04-05

Runtime update note (2026-04-06):

- Canonical `rocq` now exposes `execute()`, `sample()`, and `observe()` on a unified backend contract.
- Native state-vector Pauli expectation helpers are now reachable through `rocq.observe()` and `rocq.operator.get_expectation_value()`.
- Packaging has moved to a CMake-first `scikit-build-core` path and root CMake now builds `_rocq_hip_backend`, `rocq_hip`, and `rocquantum_bind`.

Compiler/runtime execution parity, distributed execution maturity, and higher-level CUDA-QX-style libraries remain outstanding.

## Executive Summary

`rocQuantum-main` has real ROCm-native simulator value today, especially in `hipStateVec`, `hipTensorNet`, and `hipDensityMat`. What it does not yet have is a coherent product story that matches CUDA-Q, cuQuantum, or CUDA-QX.

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
| Compiler-driven execution | STUB / ABSENT | `compile_and_execute()` is stubbed and no real MLIR/QIR execution bridge exists |
| High-level expectation APIs | PARTIAL | Native kernels exist, but public API coverage is split and inconsistent |
| Multi-GPU / distributed | PARTIAL | Real scaffolding exists, but many paths remain `NOT_IMPLEMENTED` |
| Packaging / install / export | PARTIAL | Native build exists, but releasable packaging is not coherent |
| Python surfaces | PARTIAL | Two divergent stacks exist, one with mock fallbacks |
| Integrations | PARTIAL | Thin adapters exist, but PennyLane/Cirq still sample on host |
| Higher-level libraries | STUB / ABSENT | QEC and VQE are framework shells, not robust CUDA-QX analogues |

## ROCm Integration Maturity

Overall ROCm maturity: `PARTIAL`

- Strongest point: native HIP simulator components
- Weakest point: productization, packaging, support policy, and runtime validation breadth

Current truth:

- Non-experimental CI ROCm lane: `6.2.2`
- Experimental CI ROCm lane: `7.2.0`
- Latest stable ROCm verified from official AMD docs during this audit: `7.2.0`
- Newest AMD GPU target verified during this audit: `MI355X` / `gfx950`

Recommended compatibility plan:

- Tier 1: ROCm `7.2.0`, `gfx950`, `gfx942`, `gfx90a`, Linux x86_64, Python `3.10`-`3.12`
- Tier 2 best-effort: ROCm `6.4.0`, `gfx908`, selected workstation targets such as `gfx1100`, `gfx1101`, `gfx1030`
- Recommended minimum release-grade GPU target: `gfx90a`

## CUDA-Q Gap Summary

Compared with the official CUDA-Q baseline (`https://nvidia.github.io/cuda-quantum/latest/`), the largest gaps are:

- no real compile-and-execute loop
- no unified compiler/runtime/kernel story
- no credible mid-circuit measurement and classical-control story
- no clean `observe`-style native expectation API at the canonical public surface
- no bounded, tested `mqpu`-style distributed story

What the repo does have:

- a direct local simulator path
- some native observable kernels in the backend
- multiple beginnings of a compiler stack

What it lacks is the integration layer that makes those pieces act like CUDA-Q rather than a collection of subsystems.

## cuQuantum Gap Summary

Compared with the official cuQuantum baseline (`https://docs.nvidia.com/cuda/cuquantum/latest/`), the repo is closest in scope but still incomplete:

- `hipStateVec` is the strongest analogue, but public API exposure and distributed completeness lag behind
- `hipTensorNet` has a real core, but optimizer/slicing/dtype breadth and validation lag behind cuTensorNet expectations
- `hipDensityMat` exists, but generic channels, sampling, and broader observable coverage lag behind cuDensityMat expectations

The main difference is not just feature count; it is product completeness and test-backed breadth.

## CUDA-QX Gap Summary

Compared with the official CUDA-QX baseline (`https://github.com/NVIDIA/cudaqx`), the repo is not yet competitive at the higher-level library layer.

Current state:

- QEC framework exists only as a partial scaffold
- VQE solver exists only as a scaffold
- no robust decoder, solver, or hybrid-library stack was found

This is a P2 area. It should not be used to market parity while P0 and P1 remain open.

## Top 10 Missing Or Misleading Areas

1. `compile_and_execute()` is exposed but stubbed.
2. Multi-GPU support is partial and previously overclaimed.
3. Native expectations exist but the public API story is split and misleading.
4. Two divergent Python stacks exist without one canonical answer.
5. Packaging/install/export does not describe one releasable artifact.
6. Gate fusion exists but is not wired into the active Python execution path.
7. `hipTensorNet` breadth is overstated relative to what is built and tested.
8. `hipDensityMat` is real but too narrow for broad noisy-simulation claims.
9. Integrations are mostly thin adapters, not proof of a strong platform story.
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
- Build real higher-level solver and QEC libraries

## Compatibility Plan For Latest And Older AMD GPUs

| Tier | ROCm | Architectures | Status |
| --- | --- | --- | --- |
| Tier 1 | `7.2.0` | `gfx950`, `gfx942`, `gfx90a` | Recommended target |
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
- CUDA-QX: `https://github.com/NVIDIA/cudaqx`
- ROCm release history: `https://rocm.docs.amd.com/en/latest/about/release-history.html`
- ROCm compatibility matrix: `https://rocm.docs.amd.com/en/latest/compatibility/compatibility-matrix.html`
- ROCm Linux requirements: `https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html`

Fork note:

- The local repo audit was sufficient to classify current behavior. This pass did not find any current-code dependency that required a deeper fork-divergence override from the user’s `cuQuantum`, `cuda-quantum`, or `cudaqx` forks.

Verification limit:

- No local ROCm runtime or Python interpreter was available in this shell, so runtime validation remains a ROCm-Linux CI or container task.
