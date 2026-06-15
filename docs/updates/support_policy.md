# ROCm Support Policy (Draft)

## Scope
This policy is for `rocQuantum-main` GPU validation and release gating.

## Baseline
- Minimum ROCm runtime/toolchain: **6.2.2**
- Recommended latest production lane: **7.2.4**
- Primary architecture lane: **gfx90a**
- Tier 1 architecture targets: **gfx950**, **gfx942**, **gfx90a**
- Legacy lane (best effort, reduced SLA): **gfx906**, **gfx908**
- Reproducible environment source: `ROCM_CI_SETUP.md` + `docker/rocm/Dockerfile`

## CI Expectations
- PR gate must pass:
  - CPU/unit/source-contract checks (no GPU)
  - ROCm compile checks for 6.2.2 and 7.2.4 latest-production lane when a matching container is available
  - ROCm runtime checks on self-hosted Linux ROCm runner (gfx90a)
- Nightly optional lane:
  - Multi-GPU distributed checks
  - Performance smoke benchmarks

## Deprecation Process
- Legacy lane failures do not block patch releases by default.
- Primary lane failures block release candidates.
- Any architecture support removal requires:
  - one minor release notice
  - migration note in `NEXT_PLAN.md`/release notes.
