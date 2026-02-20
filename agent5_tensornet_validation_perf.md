# Agent 5 — hipTensorNet Runtime + Optimizer/dtype Validation + Perf Tooling Hooks
## What I inspected (with evidence)
- hipTensorNet build and test registration: `rocQuantum-main/rocquantum/src/hipTensorNet/CMakeLists.txt:10-15`, `rocQuantum-main/rocquantum/src/hipTensorNet/CMakeLists.txt:41-57`.
- Runtime contraction path and dtype gates: `rocQuantum-main/rocquantum/src/hipTensorNet/hipTensorNet.cpp:135-289`, `rocQuantum-main/rocquantum/src/hipTensorNet/hipTensorNet.cpp:297-315`, `rocQuantum-main/rocquantum/src/hipTensorNet/hipTensorNet.cpp:373-374`.
- Optimizer API and pathfinder implementation status: `rocQuantum-main/rocquantum/include/rocquantum/hipTensorNet_api.h:7-23`, `rocQuantum-main/rocquantum/src/Pathfinder.cpp:154-170`, `rocQuantum-main/rocquantum/src/Pathfinder.cpp:321-340`.
- Existing runtime tests and optimizer usage in tests: `rocQuantum-main/rocquantum/src/hipTensorNet/test_hipTensorNet_contraction_regression.cpp:126-129`, `rocQuantum-main/rocquantum/src/hipTensorNet/test_hipTensorNet_slicing.cpp:159-183`, `rocQuantum-main/rocquantum/tests/hipTensorNet/test_PermutationKernels.cpp:188-205`.
- CI workflows for runtime/perf: `.github/workflows/rocm-linux-build.yml:104-111`, `.github/workflows/rocm-ci.yml:148-166`, `.github/workflows/rocm-nightly.yml:94-105`.
- Local execution feasibility checks: `hipcc/rocprof/rocprofv2/amdsmi/rocm-smi` command discovery, CMake configure attempts, and GitHub Actions run pages.

## What changed since baseline (if relevant)
- Local workspace now includes additional ROCm CI workflow files (`.github/workflows/rocm-ci.yml`, `.github/workflows/rocm-nightly.yml`) that add runtime/perf lanes beyond baseline `rocm-linux-build.yml`.
- No agent-5 runtime source changes were applied in this pass; changes are validation artifacts and an optional patch file only.

## Findings (ordered by impact)
1) Optimizer selection is exposed in API but currently not wired into hipTensorNet runtime contraction.
- Evidence: `hipTensorNet.cpp` contraction loop uses local pair-cost heuristic; `rg -n "findOptimalPath|Pathfinder|pathfinder_algorithm|num_slices|algo_config" rocQuantum-main/rocquantum/src/hipTensorNet/hipTensorNet.cpp` returns no matches.
2) METIS fallback behavior is not implemented in active runtime path.
- Evidence: `Pathfinder.cpp` has `findMetisPath` TODO + throw (`rocQuantum-main/rocquantum/src/Pathfinder.cpp:321-340`), and `Pathfinder.cpp` is not part of `rocqsim_tensornet` sources (`rocQuantum-main/rocquantum/src/hipTensorNet/CMakeLists.txt:10-15`).
- Inference: Selecting METIS today does not exercise a true METIS (or fallback) path in runtime contraction because pathfinder dispatch is not connected.
3) dtype support is effectively C64-only for tensor-network create/SVD paths.
- Evidence: non-C64 returns `ROCQ_STATUS_NOT_IMPLEMENTED` in `rocTensorNetworkCreate` and `rocTensorSVD` (`rocQuantum-main/rocquantum/src/hipTensorNet/hipTensorNet.cpp:308-310`, `rocQuantum-main/rocquantum/src/hipTensorNet/hipTensorNet.cpp:373-374`).
4) hipTensorNet tests exist, but there is no dedicated smoke for optimizer variants (KAHYPAR/METIS selection) or dtype rejection behavior.
- Evidence: registered tests are RocTensorUtil/Slicing/ContractionRegression/Permutation (`rocQuantum-main/rocquantum/src/hipTensorNet/CMakeLists.txt:54-57`), and contraction regression hardcodes GREEDY (`rocQuantum-main/rocquantum/src/hipTensorNet/test_hipTensorNet_contraction_regression.cpp:127`).
5) Perf smoke exists as elapsed-time measurement, but explicit profiling telemetry hooks (`rocprof`, `amdsmi`) are missing.
- Evidence: nightly workflow records duration only (`.github/workflows/rocm-nightly.yml:94-105`); no `rocprof`/`amdsmi` references in tracked docs/workflows scanned in this pass.

## Actions taken
- Files changed:
  - `agent5_tensornet_validation_perf.md` : Documented validation scope, findings, blocked execution evidence, and CI commands.
  - `VALIDATION_MATRIX.md` : Added hipTensorNet/perf validation matrix section with current status and required CI commands.
  - `VALIDATION_RESULTS.md` : Added hipTensorNet/perf execution results and blocked evidence.
  - `patches/agent5_tensornet_fixes_perf.patch` : Added optional patch proposal for lightweight `rocprof`/`amdsmi` perf hooks in nightly workflow.

## Validation
- Local: attempted toolchain and build execution; blocked on missing runtime prerequisites.
  - `hipcc -> NOT_FOUND`, `rocprof -> NOT_FOUND`, `rocprofv2 -> NOT_FOUND`, `amdsmi -> NOT_FOUND`, `rocm-smi -> NOT_FOUND`.
  - `python`/`py` command not found.
  - `cmake -S rocQuantum-main -B build-agent5 -DBUILD_TESTING=ON` failed: HIP not supported by Visual Studio generator.
  - `cmake -S rocQuantum-main -B build-agent5-ninja -G Ninja -DBUILD_TESTING=ON` failed: Ninja/CXX/HIP compiler not set.
- ROCm CI: latest public run evidence available for current default workflow; no successful hipTensorNet runtime execution observed.
  - `ROCm Linux Build` run #7: https://github.com/HanGuWon/rocQuantum-main/actions/runs/22094202085 -> **Failure** (2026-02-17).
  - `Build ROCm 6.2.2` job: https://github.com/HanGuWon/rocQuantum-main/actions/runs/22094202085/job/63846856733 -> **Failure**, Configure step exit code 1.
  - `Build ROCm 7.2.0` in run #7 -> **Failure**, Docker pull failed (from run annotations).
  - `ROCm CI` / `ROCm Nightly` workflows are present locally but (in this pass) have no public run evidence tied to this commit.

## Risks & follow-ups
- Runtime behavior mismatch: users can set optimizer enums that currently do not alter contraction execution.
- METIS/KAHYPAR expectations may diverge from observed behavior until pathfinder wiring/fallback is implemented and tested.
- Perf visibility remains coarse without profiler/telemetry artifacts on ROCm runners.
- Follow-up CI commands (ROCm runner):
  - `cmake -S . -B build-rocm-ci -G Ninja -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON -DROCQUANTUM_BUILD_BINDINGS=OFF -DCMAKE_HIP_COMPILER="$(command -v hipcc)"`
  - `cmake --build build-rocm-ci --parallel`
  - `ctest --test-dir build-rocm-ci --output-on-failure -R "HipTensorNetContractionRegression|RocTensorUtilTest|SlicingLogicTest|PermutationKernelTest"`
  - `ctest --test-dir build-rocm-nightly --output-on-failure -R HipTensorNetContractionRegression`

## Evidence index
- `rocQuantum-main/rocquantum/src/hipTensorNet/CMakeLists.txt:10-15`
- `rocQuantum-main/rocquantum/src/hipTensorNet/CMakeLists.txt:41-57`
- `rocQuantum-main/rocquantum/src/hipTensorNet/hipTensorNet.cpp:135-289`
- `rocQuantum-main/rocquantum/src/hipTensorNet/hipTensorNet.cpp:308-310`
- `rocQuantum-main/rocquantum/src/hipTensorNet/hipTensorNet.cpp:373-374`
- `rocQuantum-main/rocquantum/src/Pathfinder.cpp:154-170`
- `rocQuantum-main/rocquantum/src/Pathfinder.cpp:321-340`
- `rocQuantum-main/rocquantum/src/hipTensorNet/test_hipTensorNet_contraction_regression.cpp:126-129`
- `.github/workflows/rocm-linux-build.yml:104-111`
- `.github/workflows/rocm-ci.yml:148-166`
- `.github/workflows/rocm-nightly.yml:94-105`
- GitHub Actions run: `https://github.com/HanGuWon/rocQuantum-main/actions/runs/22094202085`
- GitHub Actions job: `https://github.com/HanGuWon/rocQuantum-main/actions/runs/22094202085/job/63846856733`
- Local command evidence: tool discovery and configure failures captured in this session (commit context `895b5daeab35a8b199575b84c4bb040753978187`).
