# PR Plan - ROCm Validation Handoff Integration

## Input Discovery And Assumptions
- Local repo path: `C:/Users/<redacted>/Desktop/rocQuantum-main` (discovered from workspace).
- Baseline commit: `dbfd6816d4307b2f869487d0bf36f1c2ad324b3a` (provided).
- GitHub remote: `origin=https://github.com/HanGuWon/rocQuantum-main.git` (discovered).
- Target branch: `main` (assumed; user placeholder `<TARGET_BRANCH>` was not provided).

## Current State Snapshot
- `origin/main` = `dbfd6816d4307b2f869487d0bf36f1c2ad324b3a`.
- Implementation-pack detached commit = `895b5daeab35a8b199575b84c4bb040753978187`.
- Divergence from baseline: `origin/main...895b5dae` = `0 1`.
- Pack consistency checks:
  - `BACKLOG.json` parses.
  - `NEXT_PLAN.md` / `EVIDENCE_INDEX.md` references resolve.
  - Legacy `patches/agent*_*.patch` files are blueprint docs (not `git apply` unified diffs).

## Integration Strategy
- Branch model: one integration branch with atomic commits.
- Working branch name: `integration/rocm-validation-handoff`.
- Rationale:
  - upstream has no drift vs baseline,
  - changes are tightly coupled (CI + validation docs + handoff artifacts),
  - keeps review linear while preserving atomic commit boundaries.

## Atomic Commit Plan
1. `ci(rocm): add self-hosted runtime and nightly workflows`
   - `.github/workflows/rocm-ci.yml`
   - `.github/workflows/rocm-nightly.yml`
2. `docs(rocm): add runner setup and reproducible environment artifacts`
   - `ROCM_CI_SETUP.md`
   - `docker/rocm/Dockerfile`
   - `docs/updates/support_policy.md`
   - `agent2_dev_environment.md`
   - `agent3_rocm_ci.md`
   - `patches/agent2_dev_environment.patch`
   - `patches/agent3_rocm_ci.patch`
3. `test(e2e): add compiler/python flow suite and CI hook`
   - `rocQuantum-main/tests/test_e2e_compiler_python_flows.py`
   - `.github/workflows/rocm-linux-build.yml`
4. `docs(validation): add runtime/tensornet/compiler validation matrix and reports`
   - `VALIDATION_MATRIX.md`
   - `VALIDATION_RESULTS.md`
   - `agent4_runtime_statevec_rccl.md`
   - `agent5_tensornet_validation_perf.md`
   - `agent6_e2e_compiler_python_consolidation.md`
   - `docs/validation/agent6_local_env.log`
   - `patches/agent5_tensornet_fixes_perf.patch`
   - `patches/agent6_e2e_fixes.patch`
5. `docs(handoff): finalize plan/backlog/evidence and integration report`
   - `NEXT_PLAN.md`
   - `BACKLOG.json`
   - `EVIDENCE_INDEX.md`
   - `agent1_integration.md`
   - `PR_PLAN.md`

## Local Integration Commits (Completed)
1. `1ae04e5` `ci(rocm): add self-hosted runtime and nightly workflows`
2. `0ad3137` `docs(rocm): add runner setup and reproducible environment`
3. `719c212` `test(e2e): add compiler/python flow suite and CI hook`
4. `d87d568` `docs(validation): add runtime and E2E validation artifacts`
5. `e81f7ce` `docs(handoff): finalize plan, backlog, and evidence index`

## Merge/PR Plan
- Preferred: one PR from `integration/rocm-validation-handoff` -> `main`.
- Optional split if reviewers request:
  1. PR-A (CI infra): commit 1 + 2.
  2. PR-B (tests/runtime validation docs): commit 3 + 4 (depends on PR-A).
  3. PR-C (final planning artifacts): commit 5 (depends on PR-B).

## Required CI Gates For Merge
- `ROCm CI / Fast Checks (CPU)` must pass.
- `ROCm CI / ROCm Runtime (Self-hosted GPU)` must pass on labeled `gfx90a` runner.
- `ROCm Linux Build / Build ROCm 6.2.2` must pass.
- `ROCm Linux Build / Build ROCm 7.2.0` should pass (currently experimental lane in workflow).

## Risks
1. ROCm runtime outcomes remain unverified in this local Windows session.
2. `MultiGPUTests` depends on runner GPU count and label hygiene; mislabeling can hide distributed regressions.
3. `compile_and_execute` is still not implemented; E2E runtime row remains blocked by design.

## Verification Commands
```powershell
git fetch origin --prune
git switch -c integration/rocm-validation-handoff 895b5daeab35a8b199575b84c4bb040753978187
git status --short
```

