# NEXT PLAN - rocQuantum-main

Baseline: `dbfd6816d4307b2f869487d0bf36f1c2ad324b3a`

## Executive Summary (Top 10 Actions)
1. Add mandatory self-hosted ROCm runtime CI (`gfx90a`) in PR gating.
2. Add fail-fast ROCm probe (`hipcc`, `rocminfo`, `rocm-smi`, `/dev/kfd`) for all runtime lanes.
3. Publish support policy (min ROCm `6.2.2`, primary `gfx90a`, legacy `gfx906/gfx908`).
4. Eliminate top distributed `NOT_IMPLEMENTED` paths for non-local 1Q/2Q gates via correctness fallback.
5. Implement distributed sampling/expectation correctness fallback (host path) and mark as slow.
6. Replace contradictory multi-GPU docs with a single behavior-accurate capability matrix.
7. Introduce comm abstraction (`host_fallback` now, `rccl` backend staged).
8. Wire hipTensorNet optimizer config to runtime behavior and fallback rules.
9. Implement `compile_and_execute()` MVP from MLIR gate replay to backend execution.
10. Enforce Python API exposure policy so skeleton providers are opt-in only.

## Dependency Graph / Critical Path
`CI foundation` -> `Distributed correctness` -> `TensorNet optimizer/dtype maturity` -> `Compiler execute path` -> `Python exposure hardening`

- CI foundation must land first so subsequent distributed/TN/compiler work has ROCm runtime validation.
- Distributed correctness should precede compiler runtime promotion because backend execution correctness is prerequisite.
- TensorNet optimizer/dtype work is partially parallel, but runtime validation still depends on CI foundation.
- Python exposure hardening can start early, but final policy should reflect real backend maturity from earlier milestones.

## Milestones

### P0 (0-4 weeks)

#### P0-1 ROCm Runtime CI Lane (Owner: Build/Infra)
- Scope: Self-hosted runtime workflow + ROCm probe script + policy doc.
- Acceptance criteria:
  - PR gate includes runtime workflow on self-hosted ROCm `gfx90a`.
  - Probe script fails fast when ROCm prerequisites are missing.
  - Compile matrix remains at ROCm `6.2.2` and `7.2.x`.
- Test plan:
  - Verify workflow dispatch and runner labels.
  - Run runtime ctest subset on self-hosted ROCm.
- Risk:
  - Runner availability and label drift.

#### P0-2 Distributed Top-N `NOT_IMPLEMENTED` Reduction (Owner: hipStateVec)
- Scope: Non-local 1Q/2Q fallback (swap/localize), distributed sampling/expectation host fallback.
- Acceptance criteria:
  - Non-local distributed H/CNOT/CZ no longer return `NOT_IMPLEMENTED` in covered paths.
  - Distributed sample and expectation return valid outputs for smoke circuits.
- Test plan:
  - 2-rank distributed vs single-device parity tests.
  - CI runtime lane execution on self-hosted ROCm.
- Risk:
  - Host fallback is slow; correctness only.

#### P0-3 Doc Truth Fix + Comms Abstraction Skeleton (Owner: hipStateVec + Docs)
- Scope: Update multi-GPU docs, add explicit host fallback mode, add comm abstraction API.
- Acceptance criteria:
  - Doc claims match code behavior for swap and measurement paths.
  - Runtime can report selected comm backend.
- Test plan:
  - Doc review against evidence index.
  - Runtime unit check for backend mode selection.
- Risk:
  - Documentation drift if code changes without matrix updates.

#### P0-4 Python API Exposure Guardrails (Owner: Python API)
- Scope: Status metadata and default blocking of skeleton providers.
- Acceptance criteria:
  - `set_target()` blocks skeleton providers unless explicit experimental opt-in is set.
  - `list_backends()` reports statuses.
- Test plan:
  - New unit tests for policy behavior.
- Risk:
  - Breaking implicit usage of skeleton providers.

### P1 (1-3 months)

#### P1-1 Non-local Distributed Completeness (Owner: hipStateVec)
- Scope: Controlled matrix multi-control/multi-target distributed path completion.
- Acceptance criteria:
  - Key distributed controlled matrix paths avoid `NOT_IMPLEMENTED` for covered arities.
- Test plan:
  - Expanded 2-rank gate matrix parity suite.
- Risk:
  - Correctness complexity for qubit localization/remap.

#### P1-2 RCCL Comms Backend (Owner: hipStateVec Comms)
- Scope: Implement RCCL backend behind comm abstraction.
- Acceptance criteria:
  - Remap path supports RCCL backend with parity to host fallback.
- Test plan:
  - A/B host fallback vs RCCL parity tests.
  - ROCm multi-GPU runtime CI.
- Risk:
  - RCCL/runtime compatibility differences across ROCm versions.

#### P1-3 hipTensorNet Optimizer Baseline (Owner: hipTensorNet)
- Scope: Runtime optimizer dispatch + fallback; optional METIS integration gating.
- Acceptance criteria:
  - `pathfinder_algorithm` impacts runtime behavior or falls back with explicit warning.
  - METIS/KAHYPAR toggles are build-guarded.
- Test plan:
  - Regression contraction tests for each algorithm mode.
- Risk:
  - Path changes can alter numerical/perf characteristics.

#### P1-4 compile_and_execute MVP (Owner: Compiler)
- Scope: Implement backend execution path and binding arg forwarding.
- Acceptance criteria:
  - End-to-end compile/execute works for representative circuits.
  - Errors are actionable for malformed MLIR/unsupported gates.
- Test plan:
  - 3-circuit runtime validation in ROCm CI.
  - Source-contract tests in CPU lane.
- Risk:
  - Textual parser brittleness to MLIR formatting drift.

#### P1-5 TensorNet dtype expansion phase-1 (Owner: hipTensorNet)
- Scope: C64 stable + C128 lane where supported.
- Acceptance criteria:
  - Dtype support matrix explicit and tested.
- Test plan:
  - dtype smoke tests and contraction parity checks.
- Risk:
  - Template/kernel coverage gaps for non-C64 paths.

### P2 (3-6+ months)

#### P2-1 Distributed Performance Parity (Owner: hipStateVec Perf)
- Scope: Replace host fallback on hot paths with device collectives.
- Acceptance criteria:
  - Significant speedup over host fallback baseline.
- Test plan:
  - Nightly perf benchmark trend tracking.
- Risk:
  - Complexity/performance tradeoffs across topologies.

#### P2-2 Advanced TensorNet Optimization (Owner: hipTensorNet Perf)
- Scope: slicing policy maturity, advanced partition heuristics, mixed precision roadmap.
- Acceptance criteria:
  - Stable and documented optimizer behavior under memory limits.
- Test plan:
  - Perf + numerical stability suites.
- Risk:
  - Reproducibility across versions and hardware.

#### P2-3 Ecosystem Polish (Owner: Python + Integrations)
- Scope: provider maturity lift, test modernization, stable capability reporting.
- Acceptance criteria:
  - No TODO-only provider tests for exposed backends.
- Test plan:
  - Provider-level contract tests with explicit skip conditions.
- Risk:
  - External API/provider drift.

## Verified Here vs Requires ROCm GPU CI

### Verified Here (this session)
- Baseline commit checkout and clean detached HEAD at `dbfd6816`.
- Evidence extraction and cross-check across CI/build/source/docs paths.
- Generation of implementation-pack artifacts:
  - `agent1_ci_build.md`
  - `agent2_hipstatevec_distributed.md`
  - `agent3_rccl_docs_alignment.md`
  - `agent4_hiptensornet_optimizer_dtype.md`
  - `agent5_compiler_compile_and_execute.md`
  - `agent6_python_api_and_consolidation.md`
  - `EVIDENCE_INDEX.md`
  - `BACKLOG.json`
  - `docs/updates/*`
  - `patches/*`

### Requires ROCm GPU CI
- Any runtime validation for hipStateVec distributed execution.
- Any runtime validation for hipTensorNet contraction, optimizer choices, dtype expansion.
- Any end-to-end runtime validation of `compile_and_execute` against backend execution.

## Unknowns and Verification Hooks
- Unknown: exact RCCL collective integration points currently active in runtime code.
  - Verify by implementing comm-mode telemetry and asserting backend mode in tests.
- Unknown: full non-C64 template compatibility in tensornet kernels/utilities.
  - Verify with CMake build matrix + dtype smoke tests on ROCm runners.
- Unknown: parser robustness of `compile_and_execute` against MLIR evolution.
  - Verify with parser contract tests fed by emitted MLIR fixtures.
