# Current State Audit

Audit date: 2026-04-05

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
- `rocqCompiler::MLIRCompiler::compile_and_execute()` is a hard stub.
- Distributed multi-GPU is only partially implemented and was overclaimed in docs.
- High-level expectation-value workflows are split across native helpers, legacy Python bindings, and host-side NumPy fallbacks.
- Packaging, install/export, and CI do not describe one coherent release artifact.

## Code Surface Inventory

| Surface | Reality |
| --- | --- |
| `rocquantum/src/hipStateVec` | Real HIP kernels for state-vector simulation, sampling, measurement, expectation values, some distributed scaffolding |
| `rocquantum/src/hipTensorNet` | Real contraction/SVD core, but pathfinder/slicing/dtype breadth is overstated |
| `rocquantum/src/hipDensityMat` | Real density-matrix core with limited gate/noise/observable support |
| `rocquantum/src/simulator.cpp` | Real public C++ simulator wrapper, but API surface is narrower than backend capability |
| `rocqCompiler/*` | Partial MLIR/QIR codegen path; not integrated into a working execution loop |
| `rocq/*` | Top-level Python surface with direct backend execution and mock fallbacks |
| `python/rocq/*` | Separate legacy-ish Python surface with `_rocq_hip_backend` bindings and host-side fallbacks |
| `integrations/*` | Thin framework adapters with mixed native and host-side behavior |
| `rocquantum/backends/*` | Mixed remote-provider clients, local mocks, and skeleton placeholders |
| `rocquantum/qec`, `rocquantum/solvers` | Framework shells, not production-ready CUDA-QX analogs |

## What Works Today

### Native simulator core

- `hipStateVec` implements native single-qubit gates, `CNOT`, `CZ`, `SWAP`, `CRX`, `CRY`, `CRZ`, destructive measurement, sampling, state readback, single-Pauli expectations, Z-product expectations, and generic Pauli-string expectations.
- Backend-native `MCX` and `CSWAP` logic exists in `rocquantum/src/hipStateVec/hipStateVec.cpp`, even though public simulator APIs and tests do not surface it cleanly.
- `rocquantum::QuantumSimulator` exposes basic gate application, matrix application, statevector readback, and measurement/count generation.

### Density matrix and noise

- `hipDensityMat` supports density-matrix allocation, reset, single-qubit gates, `CNOT`, controlled single-qubit gate application, and basic channels such as bit flip, phase flip, depolarizing, and amplitude damping.
- Density-matrix expectation helpers exist for single-qubit X/Y/Z and Pauli-Z products.

### Tensor networks

- `hipTensorNet` contains real permutation helpers, pairwise contraction, greedy multi-tensor contraction, and complex64 SVD.
- CI contains one real GPU regression target for tensor-network contraction.

### Direct runtime execution

- The top-level `rocq` stack executes circuits directly against backend adapters.
- The `python/rocq` stack can execute circuits directly through `_rocq_hip_backend` without going through a compiler/runtime bridge.

## What Is Partial

### Compiler/runtime

- `rocqCompiler::MLIRCompiler::emit_qir()` exists and does partial lowering to LLVM IR/QIR.
- Lowering coverage is incomplete and the compiler code path is not integrated into the shipped build in a way that yields a credible end-to-end execution loop.
- The repo carries two separate compiler/IR stories: `rocqCompiler/*` and `rocquantum/include/src/rocqCompiler/*`.

### Generic matrix and controlled-unitary support

- Generic matrix application is partially native, but larger/general cases still fall back to host-side logic in `rocquantum/src/hipStateVec/hipStateVec.cpp`.
- Generic controlled unitary support is also partial for the same reason.

### Multi-GPU/distributed

- Distributed handles, allocation, distributed metadata, and some local-domain distributed operations exist.
- Many distributed operations still return `ROCQ_STATUS_NOT_IMPLEMENTED`.
- `MULTI_GPU_GUIDE.md` overclaimed implemented RCCL behavior relative to the actual code path.

### High-level expectations

- Native expectation helpers exist in `hipStateVec` and are bound in `python/rocq/bindings.cpp`.
- The canonical top-level `rocq.operator.get_expectation_value()` is still intentionally unimplemented.
- `python/rocq/api.py::Circuit.expval()` computes host-side NumPy expectations after copying the statevector back to the host.

### Packaging and CI

- Native CMake build exists, but Python packaging is split between `pyproject.toml`, `setup.py`, `rocquantum_bind`, and `_rocq_hip_backend`.
- CI covers Python import/package contracts and one GPU runtime regression, but not a release-grade runtime matrix.

## What Is Stubbed Or Absent

- `rocqCompiler::MLIRCompiler::compile_and_execute()` always throws.
- Compiler-driven execution parity with CUDA-Q is absent.
- Mid-circuit measurement plus classical control flow is absent as a coherent supported feature.
- Public `QuantumSimulator` expectation APIs are absent.
- Public `QuantumSimulator` named APIs for `MCX`, `CSWAP`, and generic controlled unitary are absent.
- Density-matrix generic channel application returns `HIPDENSITYMAT_STATUS_NOT_IMPLEMENTED`.
- Density-matrix sampling was not found.
- Stabilizer/tableau/Pauli-propagation backends were not found.
- CUDA-QX-style higher-level solver/QEC libraries are not implemented beyond framework shells.

## Highest-Risk Overclaims Before This Audit

- The repo read as a hardware-agnostic CUDA-Q-like framework even though the real working core is primarily a local HIP simulator plus mixed provider/client scaffolding.
- Multi-GPU docs implied implemented RCCL-based distribution while the same file also admitted the distributed wrappers are still single-GPU compatibility wrappers.
- Roadmap items still listed already-implemented backend-native gates as future work.
- Public high-level APIs implied expectation-value and compiler/runtime functionality that is either stubbed or host-side fallback.

## Native Vs Fallback Summary

| Area | Native | Fallback / mock / host-side |
| --- | --- | --- |
| State-vector gates | HIP kernels in `hipStateVec` | None for core named gates |
| Generic matrix/control matrix | Partial HIP path | Host-side fallback in `hipStateVec.cpp` for broader cases |
| Expectations | Native `hipStateVec` helpers exist | Canonical `rocq` operator API still not wired; `python/rocq` has NumPy fallback path |
| Tensor-network contraction | Native HIP/rocBLAS/rocSOLVER path | Pathfinder/slicing breadth not fully wired |
| Density matrix | Native limited kernel set | Generic channels absent |
| Framework adapters | Use native simulator for some operations | PennyLane/Cirq sample on host with NumPy; many tests are mock-only |
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

As of 2026-04-05, the official AMD docs inspected during this audit showed ROCm `7.2.0` as the latest stable release. I did not find official AMD evidence for a `7.2.1` release in this pass, so earlier `7.2.1` planning assumptions were downgraded to `7.2.0` for the compatibility plan.

## Verification Limits

- This shell does not currently expose `python`, `py`, `hipcc`, `ninja`, or a local ROCm install.
- Runtime validation therefore remains a source audit plus CI/ROCm-Linux execution task.
