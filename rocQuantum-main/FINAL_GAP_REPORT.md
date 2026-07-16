# Final Gap Report

Audit date: 2026-04-05

Evidence refresh: 2026-07-15

- Current positioning: **ROCm-native simulation SDK with a CUDA-Q-inspired Python API**.
- Released baselines: cuQuantum SDK `26.06.0` (including cuPauliProp and cuStabilizer), CUDA-Q `0.15.0`, and CUDA-QX `0.6.0`.
- Evidence is reported as `source-present`, `host-contract-tested`, `native-single-GPU-verified`, `native-multi-GPU-verified`, or `performance-verified`. No AMD GPU is available on this host, so no native/performance evidence is claimed in this refresh.

Runtime update note (2026-04-06):

- Canonical `rocq` now exposes `execute()`, `sample()`, and `observe()` on a unified backend contract.
- Native state-vector Pauli expectation helpers are now reachable through `rocq.observe()` and `rocq.operator.get_expectation_value()`.
- Packaging has moved to a CMake-first `scikit-build-core` path and root CMake now builds `_rocq_hip_backend`, `rocq_hip`, and `rocquantum_bind`.

Audit refresh note (2026-06-10):

- AMD production ROCm documentation now identifies ROCm `7.2.4` as the production release; `7.2.0` remains historical audit context.
- Canonical `rocq` expectation and narrow GateFusion wiring have moved ahead of the original 2026-04-05 text.

CUDA-Q/CUDA-QX implementation update (2026-07-15):

- The canonical host runtime now includes context-local targets, CUDA-Q-style result/async wrappers, `shots_count`, single-QPU `qpu_id=0` validation, resource/draw/translation tools, validated channel objects, a typed host-specialized `make_kernel` builder, Pauli exponentials, dense operator conversion, and small-system CPU dynamics.
- The experimental CUDA-QX reference layer now includes functional VQE, generic real-Pauli QAOA and pools, host ADAPT-VQE, limited precomputed-integral Jordan-Wigner chemistry, Code/Decoder registries, repetition/Steane metadata, LUT/BP decoders, and code-capacity sampling.
- Detailed source-backed comparison and implementation evidence is recorded in `CUDA_Q_CUDAQX_IMPLEMENTATION_REPORT.md`.

Goal checkpoint (2026-06-16; GPU-independent evidence revalidated 2026-07-15):

- GitHub API triage showed no open pull requests in `HanGuWon/rocQuantum-main`; the first page of pull requests contained 17 closed PRs.
- The local `main` branch was aligned with `origin/main` after the latest Qiskit provider option-contract commit.
- Clean, version-scoped host suites passed across the full Python range from the final source tree: Python 3.9 reported `480 passed, 401 skipped`, Python 3.10 reported `659 passed, 225 skipped, 457 subtests passed`, and each of Python 3.11, 3.12, and 3.13 reported `872 passed, 15 skipped, 457 subtests passed`. The larger skip counts below 3.11 are the intentionally unavailable Qiskit/PennyLane lanes; modern-lane skips are credential-, native compiler-, or ROCm-hardware-bound. The final universal `py3-none-any` wheel was inspected to exclude native/source artifacts, then passed fresh external imports, CLI help, and all 19 copied examples on Python 3.9, 3.12, and 3.13.
- The native Python boundary now has host-auditable ownership and ABI contracts: C64/C128 NumPy conversion is explicit, state buffers are tied to authoritative handle metadata and allocation generations, matrix buffers are dimension/byte checked before native dispatch, and GateFusion revalidates its retained state on every queue execution. Tensor objects use RAII and validate dimensions, strides, labels, permutations, result capacity, and aliasing; the DensityMat holder and temporary device matrices are exception-safe.
- Native CI now builds and externally imports separate C64 and C128 wheels, validates their exported dtype/itemsize with a device-free round trip, and checks the combined C128/METIS installed consumer. These are source-defined gates, not locally executed ROCm evidence.
- The release benchmark runner now rejects stale output, timeouts, duplicate/missing/unexpected cases, non-finite or boolean metrics, missing required metrics, and wrong topology/backend evidence. The distributed benchmark independently requires at least two GPUs and reports the exact RCCL or host-fallback backend.
- Because this workstation has no AMD GPU/ROCm runtime, the remaining completion evidence cannot be produced locally. A full 100% claim still requires native ROCm artifacts from the self-hosted runner or equivalent AMD GPU host, plus benchmark JSON proving actual-device execution for the state-vector, density-matrix, tensor-network, distributed, and framework smoke paths.
- The remaining feature gaps are large-scope parity items rather than small local contract fixes: full CUDA-Q compiler/runtime parity, release-grade distributed/multi-node execution, broad cuTensorNet/cuDensityMat-style planning breadth, GPU-resident adjoint/solver workflows, production CUDA-QX breadth, and production provider lanes.

The static native compiler foundation is now implemented, while full CUDA-Q compiler/runtime parity, distributed execution maturity, and higher-level CUDA-QX-style libraries remain outstanding.

## Executive Summary

`rocQuantum-main` has substantial ROCm-oriented simulator source, especially in `hipStateVec`, `hipTensorNet`, and `hipDensityMat`. The local host contracts are useful evidence, but without an AMD GPU they do not verify native numerical correctness or performance. The repository does not match CUDA-Q, cuQuantum, or CUDA-QX as a complete product.

The repo is currently strongest as:

- a partial ROCm analogue to cuStateVec
- a narrower partial ROCm analogue to cuTensorNet
- a limited partial ROCm analogue to cuDensityMat
- a direct local simulator runtime with mixed Python surfaces

It is currently weakest where CUDA-Q and CUDA-QX depend on native compiler/dynamic-control integration, hardware scheduling, GPU-resident solver/QEC acceleration, and production algorithm breadth.

## Truth Matrix Snapshot

Full row-by-row matrix: `FEATURE_TRUTH_MATRIX.md`

| Area | Highest local evidence | Summary |
| --- | --- | --- |
| HIP simulator source | `source-present` | State-vector, tensor-network, and limited density-matrix implementations exist; actual-device verification remains pending |
| Python runtime contracts | `host-contract-tested` | CPU/mock tests exercise the canonical API, validation, fallbacks, and selected integrations; this is not HIP evidence |
| Native MLIR/LLVM/QIR compiler | `host-contract-tested` | The official LLVM/MLIR 22.1.8 release-gated configure/build/CTest/clean-install path and the installed-tool Python fallback pass locally; generated dialect/pass, CPU-only tools, static QIR 2 metadata, LLVM verification, and RecordingBackend tests exist. The first retained hosted CI artifact and HIP device execution remain pending, and this is not CUDA-Q parity |
| Multi-GPU / distributed | `source-present` | Single-node scaffolding exists and multi-node requests fail explicitly; no local multi-GPU verification is available |
| Host packaging / imports | `host-contract-tested` | Clean host-only wheel build, isolated install, installed imports, and CLI help pass without HIP discovery |
| Native packaging / install / export | `source-present` | Relocatable extension RPATHs, fresh-environment native-wheel `readelf`/`ldd`/import checks, symbol-resolving install consumers, release-header filtering, and C64/C128 ABI gates are defined in ROCm CI, but no retained green ROCm artifact was available locally |
| Integrations | `host-contract-tested` | Adapter behavior can be tested with fake/mock bindings; native framework execution still requires actual-device artifacts |
| CUDA-QX-style helpers | `host-contract-tested` | Functional VQE, generic QAOA, ADAPT-VQE, limited Jordan-Wigner chemistry, Code/Decoder registries, repetition/Steane metadata, LUT/BP decoding, and code-capacity sampling are CPU-reference tested; they are not CUDA-QX-equivalent GPU libraries |
| cuPauliProp / cuStabilizer equivalents | `source-present` (absence/guard evidence) | No dedicated cuPauliProp component exists; the stabilizer helper is CPU-only and Clifford-only, not a cuStabilizer-equivalent GPU component |

## ROCm Integration Maturity

Overall local evidence ceiling: `host-contract-tested`; native ROCm maturity remains `unverified` on this host.

- Strongest point: native HIP simulator components
- Weakest point: productization, packaging, support policy, and runtime validation breadth

Current truth:

- Non-experimental CI ROCm lane: `6.2.2`
- Experimental CI ROCm lane: `7.2.4`
- Latest production ROCm verified from official AMD docs during this refresh: `7.2.4`
- Newest AMD GPU target recorded from AMD documentation during this audit: `MI355X` / `gfx950`; it was not device-verified locally
- Distributed non-local single-qubit, controlled single-qubit, CNOT/CZ, generic matrix/control-matrix, and covered sampling/probability paths use RCCL-backed swap-localization where implemented; remaining unsupported distributed paths have explicit slow/debug fallback only via `ROCQ_DISTRIBUTED_FALLBACK_MODE=host` or `ROCQ_ENABLE_DISTRIBUTED_HOST_FALLBACK=1`.
- RCCL is now wired for local-domain distributed expectation and sampling probability reductions when `ROCQ_HAVE_RCCL` is available, but this is still not general distributed execution.

Recommended compatibility plan:

- Tier 1 native target proposal: ROCm `7.2.4`, `gfx950`, `gfx942`, `gfx90a`, Linux x86_64. The host package/CI matrix covers Python `3.9`-`3.13`; tested Qiskit coverage is Python 3.10+ with `qiskit>=2.4,<3`, tested PennyLane coverage is Python 3.11+ with `pennylane>=0.45,<0.46`, and native binding compatibility across the range is not yet actual-device-verified.
- Tier 2 best-effort: ROCm `6.4.0`, `gfx908`, selected workstation targets such as `gfx1100`, `gfx1101`, `gfx1030`
- Recommended minimum release-grade GPU target: `gfx90a`

## CUDA-Q Gap Summary

Compared with the official CUDA-Q baseline (`https://nvidia.github.io/cuda-quantum/latest/`), the largest gaps are:

- no release-wired GPU-backed compile-and-execute loop by default; offline static QIR emission is now release-wired
- no fully unified compiler/runtime/kernel story; the default bindings now separate the canonical runtime compiler guard from the legacy conceptual MLIR holder
- only a narrow mid-circuit measurement and classical-control story: Qiskit simple `if_test` / `if_else`, finite `for_loop`, bounded `while_loop`, loop-local `break_loop` / `continue_loop`, and `switch_case` sampling trajectories work, but estimator/statevector dynamic semantics remain open
- no broad arbitrary-operator expectation coverage beyond the supported Pauli, dense Hermitian / Qiskit dense Operator, and full-state CSR sparse paths
- no bounded, tested `mqpu`-style distributed story; multi-node requests are explicit unsupported stubs

What the repo does have:

- a direct local simulator path
- context-local target/result/async APIs and fail-closed single-QPU semantics
- a typed host-specialized `make_kernel` builder, Pauli exponentials, inspection/translation tools, and CPU dynamics
- some native observable kernels in the backend
- one canonical optional generated-dialect/direct-QIR compiler path, with the incompatible legacy scaffold and simulator-intermediate experiment excluded from the release lowering

What it lacks is the integration layer that makes those pieces act like CUDA-Q rather than a collection of subsystems.

## cuQuantum Gap Summary

Compared with the official cuQuantum baseline (`https://docs.nvidia.com/cuda/cuquantum/latest/`), the repo is closest in scope but still incomplete:

- `hipStateVec` is the strongest analogue, with multi-control/single-target controlled-matrix fast paths now covered, but broader controlled-matrix breadth and distributed completeness still lag behind
- `hipTensorNet` has a real core and now reports optimizer/dtype/slicing capabilities, METIS is explicitly optional, and KaHyPar configure attempts fail fast until release-wired, but the optimizer stack and broad runtime sliced execution still lag behind cuTensorNet expectations
- `hipDensityMat` exists, and now has single-qubit Kraus channels, canonical CCX/CSWAP decomposition over density primitives, density sampling that reduces measured-qubit marginals on GPU before host-side shot drawing, and native dense Hermitian expectation for up to four target qubits, but GPU-resident RNG/CDF sampling, optimized channel scheduling, CSR density observables, native broad multi-control density kernels, and broader descriptor coverage lag behind cuDensityMat expectations
- there is no dedicated cuPauliProp analogue
- the CPU-only Clifford helper is not a cuStabilizer analogue: it lacks noisy GPU many-shot simulation, detector-error-model processing, GPU GF(2) primitives, and JAX support

For state-vector matrix application, unsupported cases now fail clearly unless `ROCQ_ALLOW_HOST_MATRIX_FALLBACK=1` is set for explicit slow/debug fallback.
Dense matrix moments now have local single-state and batched fused HIP reductions, so supported Hermitian variance paths no longer need to scan the state once for `<M>` and again for `<M^2>`.

The main difference is not just feature count; it is product completeness and test-backed breadth.

## CUDA-QX Gap Summary

Compared with the official CUDA-QX baseline (`https://github.com/NVIDIA/cudaqx`), the repo now has a materially broader experimental CPU-reference layer but is not yet a production GPU-accelerated solver/QEC stack.

Current state:

- QEC retains the 3-qubit sampled helper and adds validated Code/Decoder registries, odd-distance repetition and Steane metadata, single-error LUT and dense NumPy belief-propagation decoders, and seeded code-capacity sampling
- VQE includes both the legacy class API and a functional result/immutable trace API with strict optimizer normalization; shot VQE remains fail-closed
- QAOA retains MaxCut compatibility and adds arbitrary real Pauli-sum problems, default/custom mixers, shared/full/counter-diabatic parameters, final sampling, and the QAOA operator pool
- host finite-difference ADAPT-VQE and a limited precomputed-integral Jordan-Wigner chemistry core exist
- GQE/state preparation, geometry chemistry drivers, GPU gradients, MQPU/MPI, surface-code/DEM/tensor-network/realtime QEC, and production workflow scheduling remain absent

This is a P2 area. It should not be used to market parity while P0 and P1 remain open.

## Top 10 Missing Or Misleading Areas

1. The optional compiler graph now builds generated MLIR dialects, a direct static QIR 2 lowering, tools, and CPU-only validation against official LLVM/MLIR 22.1.8. The installed `rocq-translate` also gives the canonical Python API a GPU-independent QIR fallback when the native binding omits the compiler. `compile_and_execute()` still has a narrow HIP subset, the default binding remains compiler-disabled unless explicitly enabled, and the stack is not a full CUDA-Q-style compiler runtime.
2. Multi-GPU support is partial and previously overclaimed.
3. Native expectations exist but the public API story is split and misleading.
4. Two divergent Python stacks exist without one canonical answer.
5. Host-only packaging is coherent and tested; native wheel/install/export now has fail-closed loader, symbol, header, and precision gates, but remains an unverified artifact boundary until the ROCm CI lane produces a retained green result.
6. Gate fusion exists and is wired for narrow canonical `rocq` spans; unsupported fusion queue entries fail instead of being silently dropped, but legacy `python/rocq` and broader patterns remain unfused.
7. `hipTensorNet` breadth is overstated relative to what is built and tested.
8. `hipDensityMat` is real but still too narrow for broad noisy-simulation claims: generic Kraus channels, canonical CCX/CSWAP decomposition over density primitives, GPU-side marginal probability reduction for density sampling, and small dense Hermitian observable reductions exist, while GPU-resident shot workflows, optimized channel scheduling, CSR density observables, native broad multi-control density kernels, and broad descriptor coverage remain incomplete. A small Clifford-only stabilizer/tableau backend now reduces the broader simulator-portfolio gap for Pauli propagation, but it is not GPU-accelerated and does not cover noise or non-Clifford circuits.
9. Integrations are still thin adapters, though PennyLane/Cirq/Qiskit sampling is wired to prefer the native simulator `measure()` path where available, Qiskit simple control flow has a shot-trajectory source path, and supported observable/gate adapters can dispatch to native hooks. These behaviors are host-contract-tested with fake/mock bindings only in this refresh. The self-hosted workflow defines a native framework Bell-state smoke path, but retained actual-device artifacts are still required for hardware proof.
10. CUDA-QX-style libraries are now substantial host reference implementations, but remain experimental and lack production GPU/distributed solver and fault-tolerant QEC stacks.

## Completed Contract Repairs And Remaining Roadmap

### Completed in the GPU-independent P0/P1 pass

- Truth-fix docs, bindings, and tests
- Make multi-GPU partial support explicit
- Make expectation-value limitations explicit
- Publish a Linux-first ROCm compatibility statement
- Expose the covered native expectation hooks through the canonical API contract
- Wire narrow GateFusion spans into the active canonical runtime
- Produce and test a universal host-only wheel plus installed examples
- Close binding ownership/ABI hazards for state buffers, device matrices, GateFusion, TensorNet tensors, and DensityMat holders
- Make benchmark evidence fail closed on topology, backend, exact case sets, required metrics, stale files, and timeouts
- Define external C64/C128 native-wheel ABI/import gates and C128/METIS installed-consumer validation in ROCm CI
- Add host-tested target/result/noise/tooling, typed dynamic builder, Pauli exponential, operator conversion, and CPU dynamics contracts
- Expand CUDA-QX host references with functional VQE, generic QAOA, ADAPT-VQE, limited chemistry, QEC registries/decoders, and code-capacity sampling

### Remaining P0 / P1

- Close the compiler/runtime gap rather than claim CUDA-Q parity
- Unify Python surfaces
- Preserve the clean host-only wheel/install gate and execute the source-defined native wheel/install/export gates on a ROCm builder
- Execute and retain green native ROCm runtime/benchmark artifacts from the configured CI

### P2

- Complete compiler-driven runtime
- Expand distributed execution
- Accelerate and broaden the experimental solver/QEC reference layer with GPU gradients, chemistry/state-preparation/GQE, surface/DEM/TN/realtime QEC, and distributed workflows

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

- CUDA-Q `0.15.0`: `https://github.com/NVIDIA/cuda-quantum/releases/tag/0.15.0`
- cuQuantum SDK `26.06.0`: `https://github.com/NVIDIA/cuQuantum/releases/tag/v26.06.0`
- cuQuantum component docs: `https://docs.nvidia.com/cuda/cuquantum/latest/`
- CUDA-QX `0.6.0`: `https://github.com/NVIDIA/cudaqx/releases/tag/0.6.0`
- ROCm release history: `https://rocm.docs.amd.com/en/latest/release/versions.html`
- ROCm compatibility matrix: `https://rocm.docs.amd.com/en/latest/compatibility/compatibility-matrix.html`
- ROCm Linux requirements: `https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html`

Fork note:

- The local repo audit was sufficient to classify current behavior. This pass did not find a current-code dependency requiring a fork-divergence override from the user's `cuQuantum`, `cuda-quantum`, or `cudaqx` snapshots.

Verification limit:

- No AMD GPU, HIP compiler, or local ROCm runtime is available on this host. Source, host contract, packaging, and CPU-reference evidence can be produced locally; HIP correctness, multi-GPU behavior, and performance require retained actual-device artifacts.
