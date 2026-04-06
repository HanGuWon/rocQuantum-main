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
- Optional RCCL discovery and linkage in `rocquantum/src/hipStateVec/CMakeLists.txt`

## What Is Not Ready To Claim

- Arbitrary distributed gate application across slice-domain qubits
- A complete distributed sampling story
- A complete distributed expectation-value story
- Multi-node execution
- A stable public Python API contract for general distributed execution

## User-Facing Guidance

- Treat `multi_gpu=True` as experimental partial support.
- Unsupported operations should be expected to raise `ROCQ_STATUS_NOT_IMPLEMENTED` or a higher-level `NotImplementedError`.
- If you need reliable runtime behavior today, use the single-GPU path.

## Design Direction

The codebase contains the beginnings of an RCCL-based distributed architecture, but that architecture is not yet complete enough to describe as finished functionality. Future work should continue to center on:

- explicit bit-sliced distributed state ownership
- RCCL-backed redistribution for non-local operations
- capability-gated Python/runtime APIs
- dedicated multi-GPU runtime tests on ROCm Linux

Until that work is complete and validated, the correct product statement is: experimental single-node multi-GPU scaffolding, not full distributed simulator support.
