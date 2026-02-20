# Multi-GPU Behavior: Doc/Code Truth (Draft)

## Truth Table
| Topic | Docs currently claim | Code at baseline `dbfd6816` | Required doc position |
|---|---|---|---|
| Distributed state allocation/init | Some sections say APIs return NOT_IMPLEMENTED for >1 GPU | `rocsvAllocateDistributedState` and `rocsvInitializeDistributedState` allocate/init per rank (`rocquantum/src/hipStateVec/hipStateVec.cpp:1740`, `rocquantum/src/hipStateVec/hipStateVec.cpp:1874`) | Mark as implemented with constraints |
| RCCL Alltoallv in swap | Guide claims `rcclAlltoallv` swap path is implemented | Current swap non-local path is host gather/remap/scatter (`rocquantum/src/hipStateVec/hipStateVec.cpp:483`, `rocquantum/src/hipStateVec/hipStateVec.cpp:527`, `rocquantum/src/hipStateVec/hipStateVec.cpp:584`, `rocquantum/src/hipStateVec/hipStateVec.cpp:1953`) | Document host fallback as current behavior, RCCL as roadmap |
| RCCL AllReduce in measurement | Guide claims per-rank allreduce | Code does host-aggregated loop over ranks and local reductions (`rocquantum/src/hipStateVec/hipStateVec.cpp:2848`-`2897`) | Document host aggregation current state |
| Global gate coverage | Guide notes non-local direct gate calls can return NOT_IMPLEMENTED | Many distributed gate paths still return NOT_IMPLEMENTED for non-local qubits (`rocquantum/src/hipStateVec/hipStateVec.cpp:745`, `833`, `2139`, `2213`, `2612`, `3688`) | Keep explicit limitation table |
| Status summary | Guide has internally conflicting statements | Same file has contradictory status text (`rocquantum/src/hipStateVec/MULTI_GPU_GUIDE.md:7`, `11`, `55`, `95`) | Replace with single authoritative section |

## Current Limitations (must be explicit)
- Non-local distributed gate operations are incomplete and can return `ROCQ_STATUS_NOT_IMPLEMENTED`.
- Sampling/expectation in distributed mode are incomplete for several APIs.
- Host remap fallback is correctness-oriented and slow.

## Performance Note
Host remap (`D2H gather -> CPU remap -> H2D scatter`) is expected to scale poorly with large state vectors and should be tagged as `slow/debug fallback`.
