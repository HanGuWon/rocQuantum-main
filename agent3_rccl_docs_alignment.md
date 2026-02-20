# Agent 3 - Multi-GPU / RCCL Path: Doc-Code Alignment + Communication Layer

## Evidence Table
| Claim Area | Evidence | Observation |
|---|---|---|
| Docs claim APIs return not implemented on >1 GPU | `rocQuantum-main/rocquantum/src/hipStateVec/MULTI_GPU_GUIDE.md:7` | Conflicts with implemented allocation/init routines in code. |
| Docs claim RCCL Alltoallv swap | `rocQuantum-main/rocquantum/src/hipStateVec/MULTI_GPU_GUIDE.md:31`, `53` | No `rcclAlltoallv` usage in code path. |
| Code swap path | `rocQuantum-main/rocquantum/src/hipStateVec/hipStateVec.cpp:483`, `527`, `584`, `1953` | Host gather/remap/scatter fallback path is active. |
| Build links RCCL if found | `rocQuantum-main/rocquantum/src/hipStateVec/CMakeLists.txt:41`-`45` | RCCL dependency is optional link-time, not proof of runtime collective usage. |
| Docs contradict themselves | `rocQuantum-main/rocquantum/src/hipStateVec/MULTI_GPU_GUIDE.md:11`, `55`, `95` | Same file says both implemented RCCL communication and not yet available. |

## Truth Table
| Topic | Docs claim | Code does | Should happen |
|---|---|---|---|
| Distributed allocation/init | Not implemented wrappers in parts of guide | Implemented across visible devices | Document as implemented with constraints |
| Non-local swap communication | RCCL alltoallv implemented | Host gather/remap/scatter fallback | Document fallback today; add RCCL comm layer roadmap |
| Measurement global reduction | RCCL allreduce | Host-side accumulation across per-rank reductions | Document current behavior and perf limits |
| Current status summary | Inconsistent | Partially distributed with major limitations | Single source of truth with capability matrix |

## Top 5 Findings
- Multi-GPU guide is internally inconsistent and can mislead users.
- RCCL is linked optionally but no operational collective path is visible in `hipStateVec.cpp`.
- Host remap fallback is correctness-oriented but performance-critical behavior is undocumented.
- Non-local distributed gate support remains partial and should be explicitly scoped.
- Documentation needs a stable capability matrix tied to tested behavior.

## Top 5 Actions
- Introduce a communication abstraction (`distributed_comm`) with explicit backends: `host_fallback`, `rccl`.
- Keep host fallback available as `slow/debug` path with clear logging.
- Update `MULTI_GPU_GUIDE.md` to a behavior-first capability matrix.
- Add runtime mode selection via env/config (`ROCQ_DISTRIBUTED_COMM_MODE`).
- Add CI tests that assert selected comm backend and fail on accidental silent fallback.

## Concrete Edits (File List + Rationale)
- `rocQuantum-main/rocquantum/src/hipStateVec/hipStateVec.cpp`: route swap/remap through comm abstraction.
- `rocQuantum-main/rocquantum/src/hipStateVec/DistributedComm.h` (new): backend interface.
- `rocQuantum-main/rocquantum/src/hipStateVec/DistributedComm.cpp` (new): host fallback implementation + RCCL stub integration points.
- `rocQuantum-main/rocquantum/src/hipStateVec/MULTI_GPU_GUIDE.md`: replace contradictory text with validated matrix.
- `docs/updates/multi_gpu.md`: short external-facing behavior summary.

## Acceptance Criteria
- Docs and code capability matrix are consistent for each distributed API.
- Runtime logs/state expose active comm backend (`host_fallback` vs `rccl`).
- Fallback mode is explicit and documented as non-perf.

## Test Plan
- Verified here: evidence extraction only.
- Requires ROCm GPU CI:
  - single-node multi-GPU swap correctness test with forced host fallback.
  - same test with RCCL mode once implemented.

## Risks
- Introducing comm abstraction touches hot paths and may regress local performance if not isolated.
- RCCL API/version mismatches can destabilize runtime if not lane-tested.
