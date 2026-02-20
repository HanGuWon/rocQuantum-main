# Evidence Index (baseline `dbfd6816d4307b2f869487d0bf36f1c2ad324b3a`)

## CI / Build / Validation
- `.github/workflows/rocm-linux-build.yml:54`-`69`: ROCm compile uses container matrix (`6.2.2`, `7.2.0`).
- `.github/workflows/rocm-linux-build.yml:106`-`110`: runtime tests are skipped when `/dev/kfd` is absent.
- `.github/workflows/rocm-linux-build.yml:11`-`52`: CPU Python test lane exists.
- `rocQuantum-main/build_rocq.bat:13`-`14`: default GPU arch target is `gfx906`.

## hipStateVec Distributed Completeness
- `rocQuantum-main/rocquantum/src/hipStateVec/hipStateVec.cpp:1740`: `rocsvAllocateDistributedState` implementation.
- `rocQuantum-main/rocquantum/src/hipStateVec/hipStateVec.cpp:1874`: `rocsvInitializeDistributedState` implementation.
- `rocQuantum-main/rocquantum/src/hipStateVec/hipStateVec.cpp:744`, `833`, `2139`, `2213`: non-local distributed 1Q/2Q gate `NOT_IMPLEMENTED` paths.
- `rocQuantum-main/rocquantum/src/hipStateVec/hipStateVec.cpp:2612`, `2715`: distributed generic matrix limitations.
- `rocQuantum-main/rocquantum/src/hipStateVec/hipStateVec.cpp:3128`, `3181`, `3234`, `3292`, `3377`: distributed expectation APIs `NOT_IMPLEMENTED`.
- `rocQuantum-main/rocquantum/src/hipStateVec/hipStateVec.cpp:3498`: distributed sampling `NOT_IMPLEMENTED`.
- `rocQuantum-main/rocquantum/src/hipStateVec/hipStateVec.cpp:3688`, `3725`: distributed controlled matrix limits.

## Multi-GPU / RCCL Doc-Code Alignment
- `rocQuantum-main/rocquantum/src/hipStateVec/MULTI_GPU_GUIDE.md:7`: says distributed APIs return `NOT_IMPLEMENTED` on multi-GPU.
- `rocQuantum-main/rocquantum/src/hipStateVec/MULTI_GPU_GUIDE.md:31`, `53`: claims RCCL alltoallv implementation.
- `rocQuantum-main/rocquantum/src/hipStateVec/MULTI_GPU_GUIDE.md:95`: says multi-GPU state distribution not available.
- `rocQuantum-main/rocquantum/src/hipStateVec/hipStateVec.cpp:483`, `527`, `584`: host gather/scatter/remap helpers exist.
- `rocQuantum-main/rocquantum/src/hipStateVec/hipStateVec.cpp:1953`: swap path currently routes to host remap fallback.
- `rocQuantum-main/rocquantum/src/hipStateVec/CMakeLists.txt:41`-`45`: RCCL is optional link dependency.

## hipTensorNet Optimizer / dtype
- `rocQuantum-main/rocquantum/src/hipTensorNet/CMakeLists.txt:10`-`15`: tensornet sources omit `Pathfinder.cpp`.
- `rocQuantum-main/rocquantum/include/rocquantum/hipTensorNet_api.h:7`-`16`: optimizer enum includes GREEDY/KAHYPAR/METIS.
- `rocQuantum-main/rocquantum/src/hipTensorNet/hipTensorNet.cpp:169`-`207`: contract path uses in-file pair selection, not optimizer enum dispatch.
- `rocQuantum-main/rocquantum/src/Pathfinder.cpp:326`-`339`: METIS TODO and runtime throw.
- `rocQuantum-main/rocquantum/src/hipTensorNet/hipTensorNet.cpp:308`-`310`: only C64 accepted in `rocTensorNetworkCreate`.
- `rocQuantum-main/rocquantum/src/hipTensorNet/hipTensorNet.cpp:373`-`374`: SVD path rejects non-C64.
- `rocQuantum-main/python/rocq/bindings.cpp:13`-`16`: Python dtype mapping is broader than backend support.

## Compiler Pipeline
- `rocQuantum-main/rocqCompiler/MLIRCompiler.cpp:48`-`79`: `emit_qir()` implemented.
- `rocQuantum-main/rocqCompiler/MLIRCompiler.cpp:83`-`89`: `compile_and_execute()` throws `not yet implemented`.
- `rocQuantum-main/rocqCompiler/MLIRCompiler.h:18`-`23`: API declares `compile_and_execute` and `emit_qir`.
- `rocQuantum-main/bindings.cpp:22`-`26`: binding exposes `compile_and_execute` but drops dict args.

## Python API Exposure / Skeletons
- `rocQuantum-main/rocquantum/core.py:13`-`30`: backend registry includes skeleton targets.
- `rocQuantum-main/rocquantum/backends/iqm.py:1`-`7`: placeholder backend (`pass`).
- `rocQuantum-main/rocquantum/backends/alice_bob.py:1`-`10`: placeholder backend (`pass`).
- `rocQuantum-main/tests/test_iqm_backend.py:1`, `rocQuantum-main/tests/test_alice_bob_backend.py:1`: TODO-only tests.
- `rocQuantum-main/rocq/backends.py:199`-`204`: mock fallback active when HIP backend missing.

## Agent 6 E2E Compiler + Python Consolidation
- `rocQuantum-main/rocq/kernel.py:167`-`172`: Python `qir()` path emits MLIR then calls `MLIRCompiler.emit_qir()`.
- `rocQuantum-main/rocqCompiler/MLIRCompiler.cpp:48`-`79`: `emit_qir()` lowering path to LLVM IR/QIR text.
- `rocQuantum-main/rocqCompiler/MLIRCompiler.cpp:83`-`89`: `compile_and_execute()` still hard-throws not-implemented diagnostic.
- `rocQuantum-main/rocqCompiler/HipStateVecBackend.cpp:226`-`233`: backend result path returns statevector.
- `rocQuantum-main/rocquantum/core.py:34`-`50`: Python provider flow via `set_target()` backend import/auth.
- `rocQuantum-main/rocquantum/circuit.py:68`-`95`: Python circuit flow to OpenQASM via `to_qasm()`.
- `rocQuantum-main/tests/test_e2e_compiler_python_flows.py:43`-`145`: Agent 6 minimal E2E suite (3 circuits + compile/runtime diagnostic + Python API bell flow).
- `.github/workflows/rocm-linux-build.yml:49`: Python CI lane now includes `tests.test_e2e_compiler_python_flows`.
- `.github/workflows/rocm-linux-build.yml:61`-`64`: ROCm container matrix (`6.2.2`, `7.2.0`).
- `.github/workflows/rocm-linux-build.yml:107`-`111`: runtime gate depends on `/dev/kfd`; runs `HipTensorNetContractionRegression`.
- `docs/updates/support_policy.md:7`, `docs/updates/support_policy.md:9`, `docs/updates/support_policy.md:17`: policy target for ROCm baseline and `gfx90a` runtime lane.
- `docs/validation/agent6_local_env.log:5`-`9`, `19`-`20`: local Python launcher missing and unittest invocation blocked.

## ROCm Handoff Integration Phase (This Session)
- `PR_PLAN.md:1`-`80`: integration branch strategy, commit plan, merge plan, risks, and CI gates.
- `ROCM_CI_SETUP.md:73`-`162`: runner hardware profile, setup checklist, labels, permissions, and artifact policy.
- `.github/workflows/rocm-ci.yml:79`-`219`: self-hosted runtime workflow with artifact capture, `/dev/kfd` gate, and conditional `MultiGPUTests`.
- `.github/workflows/rocm-nightly.yml:74`-`157`: nightly multi-GPU + perf smoke, optional `rocprof`, telemetry snapshots.
- `docker/rocm/Dockerfile:1`-`26`: reproducible ROCm container lane (`6.2.2` / `7.2.0`) with pinned Python tooling.
- `VALIDATION_MATRIX.md:1`-`14`: final area-by-area local vs ROCm CI validation matrix.
- `VALIDATION_RESULTS.md:1`-`49`: dated runner summary, pass/fail counts, blocker excerpts, and triage notes.
- `agent1_integration.md:1`-`68`: upstream drift analysis and pack consistency checks.
- `agent2_dev_environment.md:1`-`49`: reproducible toolchain plan and support-policy linkage.
- `agent3_rocm_ci.md:1`-`46`: CI workflow implementation details and security defaults.
- `agent4_runtime_statevec_rccl.md:1`-`57`: distributed statevec/RCCL runtime coverage and blockers.
- `agent5_tensornet_validation_perf.md:1`-`70`: tensornet optimizer/dtype/perf validation findings.
- `agent6_e2e_compiler_python_consolidation.md:1`-`61`: compiler/python E2E coverage and consolidation updates.
- `docs/validation/agent6_local_env.log:1`-`21`: local environment execution blocker evidence.
