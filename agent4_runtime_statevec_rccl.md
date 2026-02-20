# Agent 4 — ROCm Runtime Correctness: hipStateVec Distributed & RCCL
## What I inspected (with evidence)
- `rocQuantum-main/rocquantum/src/hipStateVec/CMakeLists.txt:56-60` defines the only distributed hipStateVec CTest entry point: `MultiGPUTests` (`test_hipStateVec_multi_gpu`).
- `rocQuantum-main/rocquantum/src/hipStateVec/test_hipStateVec_multi_gpu.cpp:92`, `rocQuantum-main/rocquantum/src/hipStateVec/test_hipStateVec_multi_gpu.cpp:107`, and `rocQuantum-main/rocquantum/src/hipStateVec/test_hipStateVec_multi_gpu.cpp:146` exercise local distributed `X`, `CNOT(0,1)`, and fused 1Q matrix on qubit `0`; this test does not invoke non-local distributed paths.
- `rocQuantum-main/rocquantum/src/hipStateVec/hipStateVec.cpp:744-746`, `rocQuantum-main/rocquantum/src/hipStateVec/hipStateVec.cpp:831-833`, `rocQuantum-main/rocquantum/src/hipStateVec/hipStateVec.cpp:2137-2139`, `rocQuantum-main/rocquantum/src/hipStateVec/hipStateVec.cpp:2211-2213`, `rocQuantum-main/rocquantum/src/hipStateVec/hipStateVec.cpp:2611-2612`, `rocQuantum-main/rocquantum/src/hipStateVec/hipStateVec.cpp:3497-3498`, and `rocQuantum-main/rocquantum/src/hipStateVec/hipStateVec.cpp:3127-3234` return `ROCQ_STATUS_NOT_IMPLEMENTED` for key distributed non-local paths.
- `rocQuantum-main/rocquantum/src/hipStateVec/hipStateVec.cpp:483-618` and `rocQuantum-main/rocquantum/src/hipStateVec/hipStateVec.cpp:1910-1953` show non-local distributed swap uses host gather/remap/scatter (`distributed_swap_bits_host_remap`) instead of an RCCL collective.
- RCCL is link-time optional only in `rocQuantum-main/rocquantum/src/hipStateVec/CMakeLists.txt:41-45`; source scan result for `rocQuantum-main/rocquantum/src/hipStateVec/hipStateVec.cpp` was `NO_RCCL_TOKENS_IN_HIPSTATEVEC_CPP`.
- CI runtime coverage differs by workflow: `.github/workflows/rocm-ci.yml:151` excludes `MultiGPUTests`, `.github/workflows/rocm-nightly.yml:77` includes `MultiGPUTests|HipTensorNetContractionRegression`, and `.github/workflows/rocm-linux-build.yml:104-110` runs only `HipTensorNetContractionRegression`.
- Local execution feasibility checks returned `HIP_COMPILER_NOT_FOUND`, `ROCM_SMI_NOT_FOUND`, Ninja missing for `-G Ninja`, HIP unsupported for `Visual Studio 17 2022` generator, and `No tests were found!!!` for `ctest --test-dir build-local-check`.
## What changed since baseline (if relevant)
- No baseline commit hash or baseline CI run artifact was provided in assignment context; assessment is against current workspace state/workflows only.
## Findings (ordered by impact)
1) Primary ROCm runtime CI (`rocm-ci.yml`) does not run `MultiGPUTests`, so distributed hipStateVec smoke is not exercised in regular ROCm CI.
2) Distributed non-local operations are still largely `ROCQ_STATUS_NOT_IMPLEMENTED` across core APIs.
3) No executable RCCL collective path is present in `hipStateVec.cpp`; RCCL appears only as optional linkage/docs references.
4) Existing distributed runtime coverage is local-domain-focused and does not validate non-local distributed behavior.
5) Local runtime validation is blocked on this host by missing ROCm toolchain/runtime prerequisites.
## Actions taken
- Files changed:
  - `agent4_runtime_statevec_rccl.md` : Added the required Agent 4 runtime correctness report with evidence and triage.
  - `VALIDATION_MATRIX.md` : Appended Agent 4 statevec/RCCL validation matrix section and minimal ROCm smoke suite commands.
  - `VALIDATION_RESULTS.md` : Appended Agent 4 statevec/RCCL validation results section with local execution evidence and CI blocked status.
## Validation
- Local: ran environment/toolchain checks and configure/ctest attempts; runtime execution blocked by missing ROCm prerequisites on this machine.
- ROCm CI: no run link/artifacts available from this local workspace; workflow definitions inspected for `rocm-runtime-self-hosted` and `rocm-multigpu-perf-smoke` coverage status.
## Risks & follow-ups
- Add a `MultiGPUTests` smoke step to `rocm-runtime-self-hosted` so distributed statevec runs in regular ROCm CI.
- Gate a 2-GPU smoke step explicitly on detected GPU count and emit a clear skip marker when `<2` GPUs are present.
- Add a dedicated non-local distributed test target (or extend `test_hipStateVec_multi_gpu`) for global-qubit behavior.
- If RCCL collectives are intended, add explicit runtime path selection/logging and deterministic collective tests; otherwise document host-remap mode as current behavior.
## Evidence index
- `rocQuantum-main/rocquantum/src/hipStateVec/CMakeLists.txt:41`
- `rocQuantum-main/rocquantum/src/hipStateVec/CMakeLists.txt:56`
- `rocQuantum-main/rocquantum/src/hipStateVec/test_hipStateVec_multi_gpu.cpp:92`
- `rocQuantum-main/rocquantum/src/hipStateVec/test_hipStateVec_multi_gpu.cpp:107`
- `rocQuantum-main/rocquantum/src/hipStateVec/test_hipStateVec_multi_gpu.cpp:146`
- `rocQuantum-main/rocquantum/src/hipStateVec/hipStateVec.cpp:483`
- `rocQuantum-main/rocquantum/src/hipStateVec/hipStateVec.cpp:584`
- `rocQuantum-main/rocquantum/src/hipStateVec/hipStateVec.cpp:744`
- `rocQuantum-main/rocquantum/src/hipStateVec/hipStateVec.cpp:831`
- `rocQuantum-main/rocquantum/src/hipStateVec/hipStateVec.cpp:1910`
- `rocQuantum-main/rocquantum/src/hipStateVec/hipStateVec.cpp:2137`
- `rocQuantum-main/rocquantum/src/hipStateVec/hipStateVec.cpp:2211`
- `rocQuantum-main/rocquantum/src/hipStateVec/hipStateVec.cpp:2611`
- `rocQuantum-main/rocquantum/src/hipStateVec/hipStateVec.cpp:3127`
- `rocQuantum-main/rocquantum/src/hipStateVec/hipStateVec.cpp:3180`
- `rocQuantum-main/rocquantum/src/hipStateVec/hipStateVec.cpp:3233`
- `rocQuantum-main/rocquantum/src/hipStateVec/hipStateVec.cpp:3497`
- `.github/workflows/rocm-ci.yml:151`
- `.github/workflows/rocm-nightly.yml:77`
- `.github/workflows/rocm-linux-build.yml:104`
- Local command output: `NO_RCCL_TOKENS_IN_HIPSTATEVEC_CPP`
- Local command output: `HIP_COMPILER_NOT_FOUND`
- Local command output: `ROCM_SMI_NOT_FOUND`
- Local command output: `cmake -S rocQuantum-main -B build-agent4-check -G Ninja -DBUILD_TESTING=ON`
- Local command output: `cmake -S rocQuantum-main -B build-agent4-vs -G "Visual Studio 17 2022" -A x64 -DBUILD_TESTING=ON`
- Local command output: `ctest --test-dir build-local-check --output-on-failure`
