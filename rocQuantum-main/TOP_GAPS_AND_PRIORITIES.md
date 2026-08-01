# Top Gaps And Priorities

Audit date: 2026-04-05

Evidence refresh: 2026-07-31

The current target is a **ROCm-native simulation SDK with a CUDA-Q-inspired Python API**. Priorities below deliberately separate work that can be completed without an AMD GPU from evidence that must wait for an actual ROCm device. Official comparison baselines are cuQuantum SDK `26.06.0`, CUDA-Q `0.15.0`, and CUDA-QX `0.6.0`.

Completed in the GPU-independent pass through 2026-07-31: clean host-only wheel/install/import/CLI checks; CPU and ROCm CI job separation; canonicalization and subprocess execution of all 19 examples; conversion of capability documents to the evidence ladder; host-specialized Python builder call/adjoint/limited-control/terminal-measurement contracts; and a CPU-buildable TableGen/MLIR/LLVM/QIR compiler path with static-custom plus labeled terminal-MZ Base Profile output, an in-process static-QIR LLVM ORC JIT/QIS-to-backend bridge, deterministic CPU reference state-vector execution, deterministic IR/bitcode/generic-host object artifacts, explicit pipelines, and a content-addressed self-validating cache. These results do not promote any native HIP claim, provide Base/adaptive runtime semantics or a general QIR runtime, or turn the build-tree compiler libraries into an installed C++ SDK.

## Top 10 Gaps

1. Obtain retained single-GPU correctness artifacts; this cannot be completed on the current host.
2. Consolidate or clearly demote duplicate Python and binding surfaces so one public API owns execution, sampling, observation, and errors.
3. Build broader CPU-reference and property/differential tests for state-vector layout, controls, adjoints, collapse, sampling distributions, and observables.
4. Expand the now-working static-custom JIT and terminal-MZ emission path into native typed SSA/helper functions/returns, mid-circuit reset/`read_result`, branch/loop/adaptive semantics, Base/adaptive target runtime support, and arbitrary multi-control without treating offline objects, the cache, or the bounded static QIS bridge as full execution parity.
5. Define honest product scope for TensorNet using explicit descriptor, workspace, dtype, gradient, slicing, and size-limit contracts.
6. Define the cuDensityMat comparison boundary for operators, actions, callbacks, dynamics, batching, gradients, and distributed execution.
7. Prove true inter-rank behavior on multiple AMD GPUs; source scaffolding and zero-device skips do not qualify.
8. Declare cuPauliProp and GPU cuStabilizer out of scope or design them as dedicated components; the current helpers are not equivalents.
9. Keep CUDA-QX expansion behind the stable simulator/runtime contract; current VQE/QAOA/repetition-code code is a host-tested experimental subset.
10. Produce retained, reproducible performance evidence only after the preceding correctness gates pass.

## Evidence Gates

| Stage | Can be completed without an AMD GPU? | Required evidence |
| --- | --- | --- |
| `source-present` | Yes | Reviewed source, build graph, public signatures, and explicit limitations |
| `host-contract-tested` | Yes | Reproducible CPU/mock/reference tests and clean package/example checks |
| `native-single-GPU-verified` | No | Actual-device artifact tied to commit, ROCm version, GPU model, and command |
| `native-multi-GPU-verified` | No | Actual multi-GPU test that crosses device/rank boundaries |
| `performance-verified` | No | Reproducible native benchmarks with workload, baseline, and retained raw results |

## Priority Framework

### P0 — completed host gates, now regression-protected

Scope: establish reproducible, GPU-independent release gates and stop overclaiming.

- Preserve clean host-only wheel/install/import/CLI checks
- Run the pure-Python, mock, CPU-reference, example, and adapter contracts independently of ROCm
- Keep all canonical examples executable in isolated subprocesses
- Keep compiler, multi-GPU, and native-performance claims below the corresponding evidence gate
- Preserve the dual QIR profile, artifact/cache, and host-builder synthesis contracts without
  relabeling terminal measurement as adaptive execution
- Pin comparison versions and publish explicit cuPauliProp/cuStabilizer scope

### P1

Scope: strengthen correctness contracts that can be completed on a CPU host.

- Unify the two Python surfaces or demote one to legacy status
- Add CPU-reference, property, and differential tests for gate/control/adjoint/layout/measurement semantics
- Define explicit TensorNet and DensityMat descriptors, limits, fallback behavior, and error contracts
- Validate the CMake graph, install/export shape, test registration, and benchmark artifact schemas without claiming device execution
- Keep the canonical public example path aligned while unifying the remaining runtime surfaces

### P2

Scope: perform architecture expansion and actual-device verification after P0/P1 are stable.

- Complete compiler/runtime integration beyond the static-custom JIT and terminal-MZ Base emission:
  native typed SSA/functions, adaptive control, Base/result runtime support, and a general
  target/backend runtime
- Expand distributed execution beyond the current partial single-node scaffolding
- Run retained single-GPU, multi-GPU, and performance evidence gates on suitable AMD hardware
- Build dedicated cuPauliProp/cuStabilizer analogues only if brought into scope
- Add robust higher-level solver and QEC libraries

## Completed P0 Host Gates

| Item | Result | Regression gate |
| --- | --- | --- |
| Clean host package build | Wheel, isolated install, installed imports, and CLI help pass without HIP discovery | `tests/test_p2_packaging.py` plus clean wheel/install smoke |
| Split host and ROCm CI | CPU fast checks and self-hosted ROCm evidence jobs are separate | Standard-runner host suite must pass before native jobs |
| Canonical examples | All 19 examples execute through canonical `rocq` with explicit capability boundaries | `tests/test_examples_contract.py` subprocess suite |
| Evidence-stage documentation | Single status words were replaced with the five-stage ladder | Documentation consistency checks and review |
| Explicit product scope | README states the ROCm-native simulation SDK / CUDA-Q-inspired API position and excluded components | Keep product wording and version-pinned baselines consistent |
| Host builder composition/synthesis | `call` / `apply_call` static inlining, supported-gate adjoint, limited canonical control, and terminal `mz` / `mx` / `my` are CPU-contract-tested | `tests/test_cudaq_builder_composition.py` plus fail-closed negative cases |
| Native terminal-MZ QIR | Generated `!quantum.result` / `quantum.mz` and `qir-v2-base` output are checked independently from the default static-custom profile | Project structural verifier, explicit pipeline tests, `llvm-as`, and `opt -passes=verify` |
| Static QIR ORC execution | Measurement-free static QIR is JIT-compiled in process and reaches `QuantumBackend` only through registered QIS callbacks | LLVM/MLIR 22.1 compiler smoke covers callback order, QIR decomposition, Bell/bounded-MCX CPU numerics, lifecycle failures, and concurrent engine isolation |
| Offline compiler artifacts/cache | Deterministic LLVM IR/bitcode/generic-host PIC objects, strict CLI behavior, and content-addressed self-validating cache are CPU-contract-tested | Compiler artifact/cache/CLI CTests; Base IR/bitcode remains `-O0` |

## P1 Backlog

| Item | Why It Is P1 | Acceptance |
| --- | --- | --- |
| Unify Python runtime surfaces | Current duplication causes product confusion | One primary surface owns execution and expectation APIs |
| CPU-reference conformance | HIP source cannot be executed locally | Reference tests cover endian/order, controls, adjoints, collapse, probabilities, and observables independently of native code |
| Define TensorNet/DensityMat scope | Source breadth is much narrower than NVIDIA descriptors | Public capabilities and errors describe supported dtype, rank, target, slicing, channel, observable, and fallback limits |
| Repair package/export/install tree | Release engineering is not yet credible | Static configure/schema checks, compiler tool-only install, and host package checks pass; native simulator install verification remains explicitly pending |
| Preserve native test registration | Future AMD runner time must not silently execute zero useful cases | State-vector, density, TensorNet/SVD, and true inter-rank tests stay discoverable and fail when required topology is absent |

## P2 Backlog

| Item | Why It Is P2 | Acceptance |
| --- | --- | --- |
| Broaden compiler-driven runtime | Static-custom ORC/QIS execution, terminal-MZ Base emission, and offline artifacts/cache are real, but far narrower than CUDA-Q | Add native typed SSA arguments/results/helper functions, mid-circuit reset/`read_result`, branches/loops, Base/adaptive/dynamic runtime profiles, MCX above two controls, a general target/runtime plugin surface, and retained 22.1 CI evidence |
| Complete distributed multi-GPU | Requires deeper runtime design and test infrastructure | Distributed gates, measurement, and sampling are proven on multi-GPU runners |
| Add CUDA-QX-style solver/QEC libraries | Higher-level scope should not mask base gaps | Any expanded solver/QEC scope has CPU-reference tests and later actual-device evidence |
| Broaden provider/integration maturity | Secondary to local ROCm credibility | Native and remote adapter guarantees are explicit and tested |
| Add cuPauliProp/cuStabilizer components | They are absent from the current product scope | Dedicated APIs and conformance plans exist before any parity claim is made |

## Verification Commands

```bash
# GPU-independent gates
python -m pip install '.[all]'
python -m pytest -q tests test_bindings.py integrations/cirq-rocm/cirq_rocm/tests integrations/pennylane-rocq/tests integrations/qiskit-rocquantum-provider/tests
python -m pip wheel --no-deps --no-cache-dir --wheel-dir dist .
rg -n "rocq\.Simulator|import rocq\.api|rocq\.grad" example.py examples

# GPU-independent native compiler gate (requires LLVM/MLIR 22.1.x)
cmake -S . -B build-compiler -G Ninja -DROCQUANTUM_BUILD_NATIVE=OFF -DROCQUANTUM_ENABLE_MLIR_COMPILER=ON -DMLIR_DIR=/opt/llvm-22.1/lib/cmake/mlir -DLLVM_DIR=/opt/llvm-22.1/lib/cmake/llvm -DBUILD_TESTING=ON
cmake --build build-compiler --parallel
ctest --test-dir build-compiler --output-on-failure
build-compiler/rocqCompiler/rocq-translate --profile=qir-v2-base --emit=llvm-ir -O0 -o base.ll rocqCompiler/tests/base_profile.mlir
llvm-as base.ll -o base.bc
opt -passes=verify -disable-output base.bc
build-compiler/rocqCompiler/rocq-translate --emit=llvm-bc -O3 -o static.bc rocqCompiler/tests/artifact.mlir
build-compiler/rocqCompiler/rocq-translate --emit=object -O2 --cache-dir .rocq-cache -o static.o rocqCompiler/tests/artifact.mlir

# Actual-device gates; run only on a Linux ROCm host
cmake -S . -B build-ci -G Ninja -DBUILD_TESTING=ON -DROCQUANTUM_BUILD_BINDINGS=ON -DROCQUANTUM_BUILD_NATIVE=ON -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ -DCMAKE_PREFIX_PATH=/opt/rocm
cmake --build build-ci --parallel
ctest --test-dir build-ci --output-on-failure
python3 benchmarks/run_release_benchmarks.py --build-dir build-ci --output-dir benchmark-artifacts
./build-ci/rocquantum/src/hipStateVec/benchmark_hipStateVec_distributed_reductions --output distributed-reductions.json
```

The compiler-only CMake/CTest, CPU reference JIT, and artifact commands require Linux plus
LLVM/MLIR 22.1 but not ROCm or an AMD GPU. Base Profile LLVM IR/bitcode must remain at `-O0`;
generic-host relocatable objects may use `-O0` through `-O3` but retain unresolved QIS/runtime
symbols and are not runnable programs—the static in-process JIT registers QIS separately. Only
the compiler command-line tools install; compiler libraries/headers/cache and pipeline APIs are
build-tree-only. The native simulator and benchmark commands require a Linux ROCm environment that
is not available in this shell.
