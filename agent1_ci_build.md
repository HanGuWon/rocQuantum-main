# Agent 1 - ROCm CI / Build / Validation Infrastructure

## What Exists Now (Evidence)
| Area | Evidence | Observation |
|---|---|---|
| Python CI lane | `.github/workflows/rocm-linux-build.yml:11`-`52` | CPU-only Python unit/source checks exist. |
| ROCm compile lane | `.github/workflows/rocm-linux-build.yml:54`-`103` | Containerized ROCm build matrix for `6.2.2` and `7.2.0` exists. |
| Runtime gating | `.github/workflows/rocm-linux-build.yml:106`-`110` | GPU tests only run when `/dev/kfd` exists; otherwise skipped. |
| hipStateVec distributed async contract gate | `.github/workflows/rocm-linux-build.yml:89`-`90`, `rocQuantum-main/scripts/check_async_contract.sh:18`-`21` | Contract script exists. |
| Architecture defaults (Windows script) | `rocQuantum-main/build_rocq.bat:13`-`14` | Default target is `gfx906`, no explicit primary/legacy lane policy. |

## Top 5 Findings
- No dedicated self-hosted ROCm runtime job is required in PR gating.
- Current runtime test lane may silently skip on non-GPU runners.
- No repo-level support policy document defines ROCm/arch compatibility lanes.
- No standalone ROCm environment probe script for developer or CI diagnostics.
- Multi-GPU/nightly validation is absent from CI topology.

## Top 5 Actions
- Add a self-hosted ROCm runtime workflow (`gfx90a`) with mandatory runtime tests.
- Add `scripts/probe_rocm_env.sh` and call it before configure/test in ROCm jobs.
- Split CI into: CPU fast lane, ROCm compile lane, ROCm runtime lane, optional nightly multi-GPU lane.
- Add `docs/updates/support_policy.md` and wire summary into README/CI docs.
- Add explicit skip/fail policy: runtime tests fail on self-hosted lane when probes fail.

## What Is Missing (Gaps)
- Required PR gate for real ROCm runtime execution.
- Explicit minimum/support matrix in docs.
- Machine-readable CI labels/rules for multi-GPU nightly jobs.

## Proposed CI Matrix
| Lane | Runner | Trigger | Purpose | Blocking |
|---|---|---|---|---|
| `cpu-unit` | `ubuntu-latest` | PR + push | Python/source contract checks | Yes |
| `rocm-compile` | container `rocm/dev-ubuntu-22.04:{6.2.2,7.2.0}` | PR + push | Compile/link + ctest discovery | Yes |
| `rocm-runtime` | self-hosted Linux ROCm (`gfx90a`) | PR + push | Runtime correctness (statevec/tensornet/distributed smoke) | Yes |
| `rocm-multigpu-nightly` | self-hosted multi-GPU ROCm | schedule + manual | Distributed regression + perf smoke | No (nightly) |

## Concrete Edits (File List + Rationale)
- `.github/workflows/rocm-linux-build.yml`: keep compile lanes; remove runtime ambiguity from container-only job.
- `.github/workflows/rocm-runtime-selfhosted.yml` (new): enforce runtime execution on ROCm hardware.
- `.github/workflows/rocm-nightly-multigpu.yml` (new): optional distributed scale/perf lane.
- `rocQuantum-main/scripts/probe_rocm_env.sh` (new): deterministic environment diagnostics.
- `docs/updates/support_policy.md` (new): compatibility and lane policy.

## Acceptance Criteria
- PRs fail if self-hosted ROCm runtime checks fail.
- ROCm compile lane validates both ROCm `6.2.2` and latest lane.
- CI artifacts contain probe results (`hipcc`, `rocminfo`, `rocm-smi`, `/dev/kfd`).
- Support policy explicitly states min ROCm and architecture lanes.

## Test Plan
- Verified here (non-ROCm): workflow syntax diff inspection and probe script static review.
- Requires ROCm GPU CI:
  - run new self-hosted runtime workflow on `gfx90a`.
  - run nightly multi-GPU workflow with distributed tests.

## Risks
- Self-hosted runner maintenance and label drift can cause false negatives.
- ROCm container tags may shift package availability over time.
- Legacy architecture lane can increase CI cost; keep non-blocking initially.
