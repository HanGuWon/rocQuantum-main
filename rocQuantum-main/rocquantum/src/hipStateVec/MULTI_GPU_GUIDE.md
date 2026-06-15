# hipStateVec Multi-GPU Status

This document describes the current implementation status, not the eventual design target.

## Current Truth

- Multi-GPU support is experimental and single-node only.
- Distributed handles, state-allocation helpers, local-domain operations, RCCL-backed non-local swap/remap, explicit slow/debug host fallback paths, and RCCL reduction fast paths exist.
- `rocsvGetDistributedBackend` reports `rccl`, `host_fallback`, or `none` for the current distributed handle state.
- Unsupported distributed code paths return `ROCQ_STATUS_NOT_IMPLEMENTED` unless an explicit fallback mode covers the operation.
- There is no release-grade multi-GPU CI coverage in this repo today.

## Behavior Matrix

| Capability | Current behavior | Native fast path | Explicit fallback | User contract |
| --- | --- | --- | --- | --- |
| Distributed state lifecycle | `rocsvAllocateDistributedState`, `rocsvInitializeDistributedState`, and communicator teardown paths exist for one host with multiple visible GPUs | HIP allocation/copy streams per local slice | None | Experimental infrastructure only |
| Local-domain named gates | Supported only when all touched qubits are local to each slice | HIP local-slice kernels | None | Non-local variants may return `ROCQ_STATUS_NOT_IMPLEMENTED` |
| Non-local single/control/CNOT/CZ/generic matrix paths | Named single-qubit gates, controlled single-qubit paths, `MCX`, `CSWAP`, and 1-4 target generic dense matrix apply localize touched qubits with `rocsvSwapIndexBits`, apply the local kernel, then restore layout; broader controlled/generic cases remain limited | `rocsvSwapIndexBits` uses RCCL send/recv for rank/rank and local/rank remap when communicators are ready | `ROCQ_DISTRIBUTED_FALLBACK_MODE=host` or `ROCQ_ENABLE_DISTRIBUTED_HOST_FALLBACK=1` gathers to host and reapplies correctness fallback for unsupported generic matrix paths or non-RCCL debug runs | Correctness path, not CUDA-Q/cuStateVec-scale distributed execution |
| Dense matrix expectation | Local small-target path is native; distributed local-domain small-target cases can reduce per-rank complex sums through RCCL; non-local distributed or large-target cases are limited | HIP reduction for supported local cases plus `distributed_expectation_matrix_rccl` for local-domain distributed targets | Explicit host fallback for large/non-local distributed cases when fallback env vars are set | Correctness fallback exists; broad performance parity is not claimed |
| Sparse matrix moments | Local single/batched CSR paths are native; distributed CSR is native only when each rank-owned row slice references columns inside the same rank slice | HIP CSR row reductions for supported local cases plus `distributed_sparse_matrix_moments_rccl` for rank-local distributed CSR after host-staged CSR slicing | Explicit distributed host fallback when fallback env vars are set for non-local/arbitrary distributed CSR | Rank-local sparse moments avoid full-state host gather; fully GPU-resident sparse sharding and broad sparse reduction parity are still future work |
| Sparse matrix apply | Local-domain distributed slices can use the CSR kernel | HIP CSR apply for local-domain slices | Explicit distributed host fallback for non-local distributed sparse apply | Avoids dense materialization only on supported local-domain paths |
| Expectation reductions over local-domain qubits | RCCL can sum per-rank expectation scalars when communicators are ready | `distributed_expectation_rccl`, `distributed_expectation_matrix_rccl`, and `distributed_sparse_matrix_moments_rccl` with `ncclAllReduce` | Host fallback after `ROCQ_DISTRIBUTED_FALLBACK_MODE=host` | RCCL path covers local-domain Pauli/dense and rank-local CSR sparse reductions; broader observables remain limited |
| Sampling probabilities over local-domain measured qubits | RCCL can sum per-rank probabilities when measured qubits are local-domain | `distributed_sample_rccl` / `accumulate_distributed_sample_probabilities_rccl` | Host fallback after `ROCQ_DISTRIBUTED_FALLBACK_MODE=host` | Measured slice-domain bits remain unsupported without host fallback |
| Multi-node execution | Not implemented; `rocsvAllocateMultiNodeDistributedState` returns `ROCQ_STATUS_NOT_IMPLEMENTED` for `nodeCount > 1`, and legacy `Circuit(..., multi_node=True)` / `node_count > 1` raises `NotImplementedError` | None | None | Explicit unsupported boundary; use one-host `multi_gpu=True` only |
| Public Python multi-GPU contract | Legacy `Circuit(..., multi_gpu=True)` exposes only experimental partial behavior | Binding-dependent | Warning and execution notes describe the boundary | Do not treat as a stable CUDA-Q-style distributed target |

## Runtime Switches

| Switch | Values | Effect |
| --- | --- | --- |
| `ROCQ_DISTRIBUTED_FALLBACK_MODE` | `host` or `host_fallback` | Enables explicit slow/debug host fallback for covered non-local distributed operations, dense/sparse expectations, and sampling paths |
| `ROCQ_ENABLE_DISTRIBUTED_HOST_FALLBACK` | truthy flag | Legacy alias for enabling the same explicit host fallback mode |
| `ROCQ_DISTRIBUTED_COMM` | `rccl` | Requires RCCL communicator initialization and returns an RCCL error when RCCL cannot initialize |
| `ROCQ_DISTRIBUTED_COMM` | `host` or `none` | Disables RCCL so host-fallback behavior can be tested explicitly |
| `ROCQ_REQUIRE_RCCL` | truthy flag | Requires RCCL even when `ROCQ_DISTRIBUTED_COMM` is unset |
| `ROCQ_DISABLE_RCCL` | truthy flag | Forces non-RCCL behavior for A/B testing |

## Backend Introspection

`rocsvGetDistributedBackend` is an observation API. It returns:

| Value | Meaning |
| --- | --- |
| `rccl` | The handle is distributed and RCCL communicators are initialized for reduction-capable paths |
| `host_fallback` | The handle is distributed, RCCL is not active, and explicit slow/debug host fallback mode is enabled |
| `none` | The handle is not distributed, or distributed execution has no active RCCL or host-fallback backend |

## Known Limitations

| Area | Limitation | Expected result |
| --- | --- | --- |
| Non-local gates | Swap-localized correctness paths exist for named single-qubit gates, controlled single-qubit/CNOT/CZ, `MCX`, `CSWAP`, and 1-4 target generic dense matrix apply, but high-arity generic/controlled matrix remap remains incomplete | Covered paths use RCCL-backed swap-localization when available; unsupported generic paths return `ROCQ_STATUS_NOT_IMPLEMENTED` unless an explicit host fallback applies |
| Controlled/multi-control matrices | Common named distributed multi-control paths are covered, but broader controlled-matrix arities remain incomplete | Covered named paths use swap-localization; unsupported arities or layouts return `ROCQ_STATUS_NOT_IMPLEMENTED` unless an explicit host fallback applies |
| Sampling | Slice-domain measured qubits do not have a native distributed sampler today | RCCL path returns `ROCQ_STATUS_NOT_IMPLEMENTED`; explicit host fallback can provide slow correctness when enabled |
| Expectations | Slice-domain X/Y/global Pauli cases, non-local dense expectation, and non-local/arbitrary sparse distributed observables are not fully native; dense matrix RCCL support is limited to local-domain small targets, and sparse RCCL support is limited to rank-local CSR row slices | RCCL path returns `ROCQ_STATUS_NOT_IMPLEMENTED` outside supported layouts; explicit host fallback can provide slow correctness when enabled |
| Performance | Host fallback gathers distributed state to CPU memory | Correctness/debug only; no performance parity claim |
| CI evidence | This repository cannot prove multi-GPU runtime behavior without a ROCm host exposing `/dev/kfd` and multiple GPUs | Mark runtime evidence as pending until self-hosted ROCm CI artifacts exist |

## User-Facing Guidance

- Treat `multi_gpu=True` as experimental partial support.
- Treat multi-node execution as unsupported; requests through `rocsvAllocateMultiNodeDistributedState`, `Circuit(..., multi_node=True)`, or `node_count > 1` fail fast.
- Unsupported operations should be expected to raise `ROCQ_STATUS_NOT_IMPLEMENTED` or a higher-level `NotImplementedError` unless an explicit fallback mode covers that path.
- Set `ROCQ_DISTRIBUTED_FALLBACK_MODE=host` or `ROCQ_ENABLE_DISTRIBUTED_HOST_FALLBACK=1` only for slow/debug correctness checks.
- Set `ROCQ_DISTRIBUTED_COMM=rccl` or `ROCQ_REQUIRE_RCCL=1` when a ROCm runner should fail fast if RCCL cannot initialize.
- Set `ROCQ_DISABLE_RCCL=1` or `ROCQ_DISTRIBUTED_COMM=host` to force non-RCCL behavior for A/B testing.
- Run `benchmark_hipStateVec_distributed_reductions --output distributed-reductions.json` on a ROCm multi-GPU runner to capture expectation/sampling reduction timings.
- If you need reliable runtime behavior today, use the single-GPU path.

## Design Direction

The codebase contains the beginnings of an RCCL-based distributed architecture, but that architecture is not yet complete enough to describe as finished functionality. Future work should continue to center on:

- explicit bit-sliced distributed state ownership
- RCCL-backed redistribution for more non-local operations and broader runtime validation
- capability-gated Python/runtime APIs
- dedicated multi-GPU runtime tests on ROCm Linux

Until that work is complete and validated, the correct product statement is: experimental single-node multi-GPU scaffolding, not full distributed simulator support.
