# Agent 2 - hipStateVec Distributed Completeness (Non-local Ops)

## What Exists Now (Evidence)
| Missing op/API | Code location | Current behavior | Priority |
|---|---|---|---|
| Non-local distributed 1Q gate | `rocQuantum-main/rocquantum/src/hipStateVec/hipStateVec.cpp:744`-`746` | `ROCQ_STATUS_NOT_IMPLEMENTED` when target is not local | P0 |
| Non-local distributed controlled 1Q gate | `rocQuantum-main/rocquantum/src/hipStateVec/hipStateVec.cpp:831`-`833` | `ROCQ_STATUS_NOT_IMPLEMENTED` when control/target not local | P0 |
| Non-local distributed CNOT/CZ | `rocQuantum-main/rocquantum/src/hipStateVec/hipStateVec.cpp:2137`-`2139`, `2211`-`2213` | `ROCQ_STATUS_NOT_IMPLEMENTED` | P0 |
| Distributed multi-controlled X | `rocQuantum-main/rocquantum/src/hipStateVec/hipStateVec.cpp:2384`-`2385` | Fully not implemented in distributed path | P1 |
| Distributed CSWAP | `rocQuantum-main/rocquantum/src/hipStateVec/hipStateVec.cpp:2439`-`2440` | Fully not implemented in distributed path | P1 |
| Distributed generic matrix (non-local targets) | `rocQuantum-main/rocquantum/src/hipStateVec/hipStateVec.cpp:2611`-`2612`, `2715` | Non-local/large target path returns not implemented | P0 |
| Distributed expectations (Z/X/Y/product/string) | `rocQuantum-main/rocquantum/src/hipStateVec/hipStateVec.cpp:3127`-`3128`, `3180`-`3181`, `3233`-`3234`, `3291`-`3292`, `3376`-`3377` | Not implemented in distributed mode | P0 |
| Distributed sampling | `rocQuantum-main/rocquantum/src/hipStateVec/hipStateVec.cpp:3497`-`3498` | Not implemented in distributed mode | P0 |
| Distributed controlled matrix (non-local) | `rocQuantum-main/rocquantum/src/hipStateVec/hipStateVec.cpp:3686`-`3688`, `3725` | Not implemented for non-local controls/targets and general cases | P1 |

## Top 5 Findings
- Distributed infra exists (allocation/init/metadata) but non-local gate completeness is incomplete.
- Many high-value APIs fail in distributed mode with explicit `NOT_IMPLEMENTED` returns.
- Measurement already has distributed handling, but sampling/expectation do not.
- Host gather/scatter helpers already exist and can be used as correctness fallback.
- Existing tests (`test_hipStateVec_multi_gpu.cpp`) validate only local distributed paths.

## Top 5 Actions
- Implement non-local 1Q/2Q distributed gate fallback using swap orchestration.
- Implement distributed sampling/expectation correctness fallback via host gather.
- Add explicit path markers (`slow fallback`) for host gather-based implementations.
- Add 2-rank correctness tests comparing distributed and single-device reference outcomes.
- Promote remaining not-implemented areas into backlog with priority labels.

## Proposed Algorithms
- **Non-local 1Q/2Q gates**: swap target/control qubits into local domain via `rocsvSwapIndexBits`, apply local kernel, swap back.
- **Sampling**: gather distributed state (`gather_distributed_state_to_host`) and run CPU probability/cdf sampling fallback.
- **Expectation values**: gather distributed state and compute expectation on host for correctness-first implementation.
- **Controlled matrix**: localize control/target sets where possible; fallback to host apply for unsupported shapes.

## Concrete Edits (File List + Rationale)
- `rocQuantum-main/rocquantum/src/hipStateVec/hipStateVec.cpp`: add distributed non-local fallback helpers and wire key APIs.
- `rocQuantum-main/rocquantum/src/hipStateVec/CMakeLists.txt`: add distributed non-local correctness test target.
- `rocQuantum-main/rocquantum/src/hipStateVec/test_hipStateVec_distributed_nonlocal.cpp` (new): 2-rank correctness harness.

## Acceptance Criteria
- Non-local distributed H/CNOT/CZ and generic matrix (1-2 targets) no longer return `NOT_IMPLEMENTED` in core paths.
- Distributed `rocsvSample` and expectation APIs return valid results for small circuits.
- New distributed test compares against single-device reference for deterministic cases.

## Test Plan
- Verified here: source evidence and patch design only.
- Requires ROCm GPU CI:
  - execute `test_hipStateVec_multi_gpu` and new `test_hipStateVec_distributed_nonlocal` on >=2 GPUs.
  - run measurement/sampling/expectation parity checks.

## Risks
- Host gather fallback may be too slow for large states; must be marked non-perf path.
- Swap orchestration correctness depends on index remap assumptions.
- Numerical drift in host fallback comparisons requires tolerance strategy.

## Unknowns
- Whether current rank-to-slice mapping remains valid across all target GPU topologies.
- Whether batch-mode distributed semantics should be enabled or remain unsupported.
