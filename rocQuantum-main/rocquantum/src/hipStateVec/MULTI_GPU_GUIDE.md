# hipStateVec Multi-GPU Status

This document describes the current implementation status, not the eventual design target.

## Current Truth

- Multi-GPU support is experimental and single-node only.
- Distributed handles, state-allocation helpers, and some local-domain distributed operations exist.
- Many distributed code paths still return `ROCQ_STATUS_NOT_IMPLEMENTED`.
- There is no release-grade multi-GPU CI coverage in this repo today.

## What Exists In Code Today

- `rocsvGetDistributedInfo`
- `rocsvAllocateDistributedState`
- `rocsvInitializeDistributedState`
- Partial local-domain distributed gate support in `hipStateVec.cpp`
- Partial local-domain distributed measurement support
- Optional RCCL discovery, linkage, communicator initialization, and communicator teardown
- RCCL `AllReduce(sum)` fast paths for local-domain distributed expectation reductions and sampling probability reductions
- `benchmark_hipStateVec_distributed_reductions`, a ROCm-only A/B benchmark for RCCL reduction mode versus explicit host fallback mode

## What Is Not Ready To Claim

- Arbitrary distributed gate application across slice-domain qubits
- A complete distributed sampling story for measured qubits that include slice-domain bits
- A complete distributed expectation-value story for observables that include slice-domain X/Y/global terms
- Multi-node execution
- A stable public Python API contract for general distributed execution

## User-Facing Guidance

- Treat `multi_gpu=True` as experimental partial support.
- Unsupported operations should be expected to raise `ROCQ_STATUS_NOT_IMPLEMENTED` or a higher-level `NotImplementedError`.
- Set `ROCQ_DISTRIBUTED_COMM=rccl` or `ROCQ_REQUIRE_RCCL=1` when a ROCm runner should fail fast if RCCL cannot initialize.
- Set `ROCQ_DISABLE_RCCL=1` or `ROCQ_DISTRIBUTED_COMM=host` to force non-RCCL behavior for A/B testing.
- Run `benchmark_hipStateVec_distributed_reductions --output distributed-reductions.json` on a ROCm multi-GPU runner to capture expectation/sampling reduction timings.
- If you need reliable runtime behavior today, use the single-GPU path.

## Design Direction

The codebase contains the beginnings of an RCCL-based distributed architecture, but that architecture is not yet complete enough to describe as finished functionality. Future work should continue to center on:

- explicit bit-sliced distributed state ownership
- RCCL-backed redistribution for non-local operations
- capability-gated Python/runtime APIs
- dedicated multi-GPU runtime tests on ROCm Linux

Until that work is complete and validated, the correct product statement is: experimental single-node multi-GPU scaffolding, not full distributed simulator support.
