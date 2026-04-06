# rocQuantum Roadmap

This roadmap starts from audited truth, not from legacy aspirational claims.

## P0: Credibility Recovery

Focus: remove false positives and make the current support boundary explicit.

- Keep compiler/runtime parity clearly gated until a real execution bridge exists
- Make `multi_gpu=True` explicitly experimental partial support
- Align expectation-value APIs and docs with actual native vs host-side behavior
- Normalize the Linux-first ROCm compatibility statement
- Clean up README, roadmap, guides, and placeholder tests so they match the code

## P1: Connect Existing Native Capability

Focus: improve the product story without large speculative rewrites.

- Unify or clearly separate the two Python surfaces, `rocq` and `python/rocq`
- Expose native expectation helpers through the canonical public API
- Wire `GateFusion.cpp` into the active execution path
- Repair packaging/install/export so one build/install path is defensible
- Expand ROCm CI beyond the current tensor-network regression

## P2: Broader ROCm Platform Scope

Focus: only after P0 and P1 are stable.

- Complete a real compiler-driven runtime path
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

## Still Missing Or Partial

- End-to-end compiler-driven execution
- Release-grade distributed multi-GPU
- Canonical high-level native expectation API
- Release-grade packaging and install/export
- Robust higher-level solver and QEC libraries
