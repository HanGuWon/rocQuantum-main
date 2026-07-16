# rocQuantum Roadmap

This roadmap starts from audited truth, not from legacy aspirational claims.

## P0: Credibility Recovery

Focus: remove false positives and make the current support boundary explicit.

- Keep CUDA-Q compiler/runtime parity claims gated beyond the implemented static-custom and
  terminal-MZ Base Profile QIR bridge plus offline artifact/cache layer
- Make `multi_gpu=True` explicitly experimental partial support
- Align expectation-value APIs and docs with actual native vs host-side behavior
- Normalize the Linux-first ROCm compatibility statement
- Clean up README, roadmap, guides, and placeholder tests so they match the code

## P1: Connect Existing Native Capability

Focus: improve the product story without large speculative rewrites.

- Unify or clearly separate the two Python surfaces, `rocq` and `python/rocq`
- Expose native expectation helpers through the canonical public API
- Wire `GateFusion.cpp` into the active execution path
- Repair packaging/install/export so one build/install path is defensible; the compiler currently
  installs only `rocq-opt` / `rocq-translate` and keeps its C++ libraries/headers build-tree-only
- Expand ROCm CI beyond the current tensor-network regression

## P2: Broader ROCm Platform Scope

Focus: only after P0 and P1 are stable.

- Extend the implemented native TableGen/MLIR/QIR path with native typed SSA kernels/helper
  functions/returns, mid-circuit reset/`read_result`, branch/loop/adaptive runtime semantics,
  arbitrary multi-control, a runnable ORC-QIS bridge, and target packaging
- Expand distributed multi-GPU beyond the current partial single-node scaffolding
- Add higher-level solver, QEC, and hybrid-library support that can credibly compete with CUDA-QX-style libraries
- Broaden ecosystem integrations after the base runtime contract is stable

## Already Implemented At The Native Backend Level

These are no longer roadmap items and should not be listed as future work:

- Controlled rotations: `CRX`, `CRY`, `CRZ`
- Backend-native `MCX` support used for `CCX`
- Backend-native `CSWAP`
- Basic density-matrix noise channels
- Native single-Pauli and Pauli-string expectation helpers in `hipStateVec`
- Host-specialized `make_kernel` call/`apply_call` inlining, supported-gate adjoint synthesis,
  limited canonical control synthesis, and terminal `mz` / `mx` / `my` with measurement MLIR
- Generated `!quantum.result` / `quantum.mz` plus separate `qir-v2-static` and labeled
  terminal-MZ `qir-v2-base` lowering, checked by the project structural verifier and LLVM tools
- Deterministic LLVM IR/bitcode/generic-host PIC relocatable objects, explicit static/base
  pipelines, strict `rocq-translate`, and a content-addressed self-validating artifact cache

## Still Missing Or Partial

- End-to-end CUDA-Q-style compiler breadth beyond straight-line static-custom/terminal-MZ Base QIR,
  offline artifacts/cache, and the narrow HIP gate execution subset
- Native typed SSA/helper functions/returns, adaptive measurement feedback/loops/`read_result`,
  arbitrary multi-control, and a runnable QIS-linked compiler runtime
- Installed/exported compiler C++ SDK and pass-plugin surface; current compiler libraries, headers,
  cache API, and pipeline API are build-tree-only
- Release-grade distributed multi-GPU
- Canonical high-level native expectation API
- Release-grade packaging and install/export
- Robust higher-level solver and QEC libraries
