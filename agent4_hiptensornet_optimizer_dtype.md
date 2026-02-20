# Agent 4 - hipTensorNet Optimizer Maturity + dtype Roadmap

## Evidence Table
| Area | Evidence | Observation |
|---|---|---|
| Tensornet build sources | `rocQuantum-main/rocquantum/src/hipTensorNet/CMakeLists.txt:10`-`15` | `Pathfinder.cpp` is not built into `rocqsim_tensornet`. |
| Optimizer config API | `rocQuantum-main/rocquantum/include/rocquantum/hipTensorNet_api.h:7`-`35` | API exposes `GREEDY/KAHYPAR/METIS` and memory slicing knobs. |
| Runtime path selection | `rocQuantum-main/rocquantum/src/hipTensorNet/hipTensorNet.cpp:169`-`207` | Current contraction chooses pair by local cost; ignores algorithm enum. |
| METIS implementation | `rocQuantum-main/rocquantum/src/Pathfinder.cpp:326`-`339` | Explicit TODO and runtime throw for METIS. |
| dtype support in C API | `rocQuantum-main/rocquantum/src/hipTensorNet/hipTensorNet.cpp:308`-`310` | Non-`ROC_DATATYPE_C64` returns `NOT_IMPLEMENTED`. |
| dtype mapping in Python | `rocQuantum-main/python/rocq/bindings.cpp:13`-`16` | Python accepts broader dtype inputs than backend actually supports. |

## Top 5 Findings
- Optimizer config surface is richer than actual runtime behavior.
- `Pathfinder.cpp` exists but is effectively disconnected from build/runtime.
- METIS path is explicitly unimplemented.
- Dtype support is effectively C64-only in current tensornet C API.
- User-facing examples imply broader dtype experimentation without backend parity.

## Top 5 Actions
- Wire `Pathfinder.cpp` into build and select algorithm from config.
- Add robust default greedy heuristic as stable baseline.
- Add optional METIS integration behind explicit CMake option; fallback to greedy when unavailable.
- Add explicit dtype capability reporting and expand to C128 lane first.
- Add dtype correctness tests and small perf smoke harness in ROCm CI.

## Proposed Implementation Direction
- `ROCQUANTUM_TN_ENABLE_METIS` and `ROCQUANTUM_TN_ENABLE_KAHYPAR` CMake options.
- Contract path should branch on `config.pathfinder_algorithm` and fallback to greedy with warning when unavailable.
- Dtype phase-1:
  - maintain C64 stable lane,
  - add C128 creation/contract support where util kernels/BLAS paths permit,
  - reject unsupported dtypes with deterministic messages.

## Concrete Edits (File List + Rationale)
- `rocQuantum-main/rocquantum/src/hipTensorNet/CMakeLists.txt`: include `../Pathfinder.cpp`, add options/defs.
- `rocQuantum-main/rocquantum/src/hipTensorNet/hipTensorNet.cpp`: optimizer dispatch + fallback behavior + dtype gate updates.
- `rocQuantum-main/python/rocq/bindings.cpp`: expose `pathfinder_algorithm`, warn on unsupported dtypes.
- `rocQuantum-main/rocquantum/src/hipTensorNet/test_hipTensorNet_dtype_smoke.cpp` (new): C64/C128 correctness smoke.

## Acceptance Criteria
- Algorithm enum affects contraction order path selection.
- Missing METIS/KAHYPAR no longer hard-fails when selected; falls back with warning.
- Dtype support matrix is explicit in docs/tests and reflected in runtime errors.

## Test Plan
- Verified here: source-level evidence and plan only.
- Requires ROCm GPU CI:
  - regression contraction test for greedy baseline,
  - dtype smoke for supported dtypes,
  - optional perf smoke benchmark.

## Risks
- Path selection changes can alter numerical/perf characteristics.
- C128 support may require additional kernel/template coverage not present today.

## Unknowns
- Full template coverage in `rocTensorUtil` for non-C64 dtypes needs compile validation.
- METIS/KAHYPAR package availability on target ROCm runner images.
