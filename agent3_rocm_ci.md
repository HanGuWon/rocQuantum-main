# Agent 3 - ROCm CI Workflows
## What I inspected (with evidence)
- Existing workflow baseline in `.github/workflows/rocm-linux-build.yml` (CPU tests + container ROCm build, no dedicated self-hosted runtime gate).
- Available CTest targets from CMake:
  - `HipTensorNetContractionRegression`, `RocTensorUtilTest`, `SlicingLogicTest`, `PermutationKernelTest` in `rocQuantum-main/rocquantum/src/hipTensorNet/CMakeLists.txt`.
  - `MultiGPUTests` in `rocQuantum-main/rocquantum/src/hipStateVec/CMakeLists.txt`.
- Repository layout confirms nested source root (`rocQuantum-main/`) used by existing workflow commands.

## What changed since baseline (if relevant)
- Added a consolidated CI workflow with:
  - fast non-GPU checks (`fast-checks`)
  - self-hosted ROCm runtime gate (`rocm-runtime-self-hosted`)
- Added a separate optional nightly workflow for multi-GPU/perf smoke.
- Added explicit runner label conventions, required packages, security defaults, and artifact policy documentation.

## Findings (ordered by impact)
1) Baseline CI did not enforce ROCm runtime testing on self-hosted GPU hardware for PR/push.
2) Baseline ROCm test step could skip runtime coverage when `/dev/kfd` is absent, reducing signal quality.
3) Runner label conventions and required host packages were undocumented, increasing setup drift risk.
4) Artifact capture (logs/JUnit/summary) was incomplete for fast triage after failures.
5) Fork-PR safety policy for self-hosted jobs was not explicitly encoded.

## Actions taken
- Files changed:
  - `.github/workflows/rocm-ci.yml` : added fast CPU checks, self-hosted ROCm runtime job, artifact/JUnit collection, job summaries, and fork-PR safety gate.
  - `.github/workflows/rocm-nightly.yml` : added optional scheduled/manual multi-GPU and perf smoke workflow with artifacts and summaries.
  - `ROCM_CI_SETUP.md` : documented runner setup requirements, label conventions, secure defaults, secrets policy, and artifact handling.
  - `agent3_rocm_ci.md` : recorded inspection evidence, findings, actions, validation status, and risks.
  - `patches/agent3_rocm_ci.patch` : exported patch for Agent 3 owned changes.

## Validation
- Local: Ran `git diff --check` and targeted file diff inspection for owned files. Python-based checks were not run because `python`/`py` are unavailable in this shell, and ROCm runtime execution is not possible here.
- ROCm CI: Unknown. Exact run path once pushed: `GitHub -> Actions -> ROCm CI -> job "ROCm Runtime (Self-hosted GPU)"` and `GitHub -> Actions -> ROCm Nightly -> job "ROCm Multi-GPU / Perf Smoke"`.

## Risks & follow-ups
- Self-hosted runner label drift (`rocm-gpu`, `gfx90a`, `rocm-multigpu`) can cause queueing or skipped capacity.
- `ctest --output-junit` availability depends on runner CTest version; workflow includes fallback without JUnit.
- Nightly perf smoke currently records runtime only; no regression threshold gate is enforced yet.

## Evidence index
- `.github/workflows/rocm-linux-build.yml`
- `.github/workflows/rocm-ci.yml`
- `.github/workflows/rocm-nightly.yml`
- `rocQuantum-main/rocquantum/src/hipTensorNet/CMakeLists.txt`
- `rocQuantum-main/rocquantum/src/hipStateVec/CMakeLists.txt`
- `ROCM_CI_SETUP.md`

