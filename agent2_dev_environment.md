# Agent 2 - Dev Environment & Reproducible Toolchain
## What I inspected (with evidence)
- `.github/workflows/rocm-linux-build.yml:60` and `.github/workflows/rocm-linux-build.yml:63` confirm active ROCm CI lanes are `6.2.2` and `7.2.0`.
- `.github/workflows/rocm-linux-build.yml:96` and `.github/workflows/rocm-linux-build.yml:110` define the current Linux configure/build/test flow used by ROCm CI.
- `rocQuantum-main/build_rocq.bat:13` and `rocQuantum-main/build_rocq.bat:14` show architecture targets include `gfx906/gfx908/gfx90a`, with `gfx906` currently defaulted in Windows helper flow.
- `docs/updates/support_policy.md:7` through `docs/updates/support_policy.md:10` capture policy baseline (`6.2.2`, `7.2.x`, primary `gfx90a`, legacy `gfx906/gfx908`).

## What changed since baseline (if relevant)
- Added a reproducible ROCm Docker toolchain artifact that can be built for `6.2.2` and `7.2.0`.
- Added `ROCM_CI_SETUP.md` as the environment/toolchain source of truth with explicit commands for primary and optional legacy architecture lanes.
- Updated support policy to point to the reproducible setup artifact.

## Findings (ordered by impact)
1) The repository lacked an executable, reproducible Linux ROCm environment artifact that developers can run locally for both CI ROCm lanes.
2) Existing ROCm CI commands were present but not collected in a single environment setup guide with explicit primary vs optional architecture lane command lines.
3) Support policy stated version/arch lanes but did not previously reference a concrete setup implementation.

## Actions taken
- Files changed:
  - `docker/rocm/Dockerfile` : Added reproducible ROCm build environment with pinned ROCm image lanes (`ARG ROCM_VERSION`), explicit build dependencies, and pinned Python tooling versions.
  - `ROCM_CI_SETUP.md` : Added toolchain/environment matrix for ROCm `6.2.2` + `7.2.0`, primary `gfx90a`, optional `gfx906/gfx908`, plus minimum configure and full build/test commands.
  - `docs/updates/support_policy.md` : Added pointer to `ROCM_CI_SETUP.md` and `docker/rocm/Dockerfile` as reproducible environment sources.

## Validation
- Local: Verified file creation, command presence, and policy alignment via `rg` checks on updated docs and Dockerfile; `docker --version` is unavailable in this host shell and ROCm device access (`/dev/kfd`) is absent, so container build/run and ROCm runtime tests were not executable locally.
- ROCm CI: Required job is `ROCm Linux Build` (`.github/workflows/rocm-linux-build.yml`) with matrix jobs `Build ROCm 6.2.2` and `Build ROCm 7.2.0`; outcome in this session is `not run` (needs GitHub Actions run log after push).

## Risks & follow-ups
- Docker reproducibility currently depends on upstream container tag stability (`rocm/dev-ubuntu-22.04:*`); pinning image digests in CI would further reduce drift.
- Optional legacy `gfx906/gfx908` lane is documented but not yet wired as a dedicated CI job; enabling it should remain non-blocking per support policy.

## Evidence index
- `.github/workflows/rocm-linux-build.yml:60`
- `.github/workflows/rocm-linux-build.yml:63`
- `.github/workflows/rocm-linux-build.yml:96`
- `.github/workflows/rocm-linux-build.yml:110`
- `rocQuantum-main/build_rocq.bat:13`
- `rocQuantum-main/build_rocq.bat:14`
- `docs/updates/support_policy.md:7`
- `docs/updates/support_policy.md:10`
- `docker/rocm/Dockerfile:1`
- `docker/rocm/Dockerfile:2`
- `docker/rocm/Dockerfile:19`
- `ROCM_CI_SETUP.md:3`
- `ROCM_CI_SETUP.md:6`
- `ROCM_CI_SETUP.md:8`
- `ROCM_CI_SETUP.md:14`
- `ROCM_CI_SETUP.md:35`
- `ROCM_CI_SETUP.md:67`

