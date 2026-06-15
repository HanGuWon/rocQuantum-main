# Validation Results

- Date/time: 2026-06-10T12:00:00+09:00
- Runner details:
  - Local runner: Windows PowerShell host; no ROCm runtime; Python `3.11.15`; `cmake`, `ninja`, and `hipcc` not available.
  - ROCm CI target runners:
    - `.github/workflows/rocm-ci.yml` self-hosted labels `self-hosted,linux,x64,rocm,rocm-gpu,gfx90a`.
    - `.github/workflows/rocm-nightly.yml` self-hosted labels `self-hosted,linux,x64,rocm,rocm-multigpu`.
    - `.github/workflows/rocm-linux-build.yml` container lanes `6.2.2` and experimental/latest `7.2.4`.
  - ROCm policy target: minimum `6.2.2`, latest production lane `7.2.4`, Tier 1 archs `gfx950`, `gfx942`, `gfx90a`.
- Summary: pass/fail/blocked
  - Local Python unit/adapter tests executed: 53 passing checks across unittest/pytest invocations.
  - Local native execution blocked: ROCm runtime/toolchain is not installed on this host.
  - ROCm CI jobs executed in this session: 0
  - Historical external ROCm CI evidence retained for context only.

## Local Validation (This Session)
- `BACKLOG.json` schema parse: pass.
- Required handoff files exist: pass.
- Workflow syntax review and command-path review: pass.
- Compiler/Python/hipStateVec/hipTensorNet behavior mapping from source: pass.
- `python -m unittest tests.test_p0_fixes tests.test_p1_compiler tests.test_p2_packaging tests.test_runtime_api_contract tests.test_cpp_expectation -v`: pass, 46 tests.
- `python -m pytest -q tests/test_cirq_integration.py tests/test_pennylane_integration.py`: pass, 7 tests.
- Runtime execution on GPU hardware: blocked (no ROCm runtime host in this session).

## ROCm CI Validation
- New/updated workflows were prepared but not executed in this session:
  - `.github/workflows/rocm-ci.yml`
  - `.github/workflows/rocm-nightly.yml`
  - `.github/workflows/rocm-linux-build.yml`
- Branch push completed:
  - `integration/rocm-validation-handoff`
  - PR creation URL: https://github.com/HanGuWon/rocQuantum-main/pull/new/integration/rocm-validation-handoff
- Workflow run poll for branch returned no runs yet (`total_count=0`) because push triggers target `main/master` and PR has not been opened.
- Known public run reference (pre-handoff, used for triage context):
  - https://github.com/HanGuWon/rocQuantum-main/actions/runs/22094202085
  - job: https://github.com/HanGuWon/rocQuantum-main/actions/runs/22094202085/job/63846856733
  - Outcome: failed configure/build in ROCm matrix lane.

## Failure/Blocker Excerpts
- Local ROCm toolchain probe:
  - Excerpt: `cmake`, `ninja`, and `hipcc` were not available in the Windows shell.
- `rocQuantum-main/rocqCompiler/MLIRCompiler.cpp:87`
  - Excerpt: `compile_and_execute() is not yet implemented.`
- `.github/workflows/rocm-linux-build.yml:107`
  - Excerpt: `if [ ! -e /dev/kfd ]; then`
- `.github/workflows/rocm-ci.yml:182`
  - Excerpt: `Runner labels: self-hosted,linux,x64,rocm,rocm-gpu,gfx90a`
- `.github/workflows/rocm-nightly.yml:117`
  - Excerpt: `Runner labels: self-hosted,linux,x64,rocm,rocm-multigpu`
- `docs/validation/ci_poll_2026-02-21.log:12`
  - Excerpt: `Result: total_count=0`
- `docs/validation/ci_poll_2026-02-21.log:17`
  - Excerpt: `Error: Resource not accessible by personal access token`

## Triage Notes
1. Compile/runtime E2E remains blocked by intentional stub in `compile_and_execute`; this is product code, not CI infra.
2. Distributed and performance validation now has CI entry points, but requires GPU runner capacity to generate first-run evidence.
3. Local Windows host remains documentation/integration-only for this phase; runtime authority is ROCm Linux CI.
