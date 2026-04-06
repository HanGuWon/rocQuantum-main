# Optional Code Changes

This directory documents the safe, low-blast-radius credibility fixes applied during the audit-first pass.

## Changes Applied

### Documentation truth fixes

- Rewrote `README.md` to describe the repo as an experimental ROCm-first simulator stack rather than a finished CUDA-Q-style framework
- Rewrote `ROADMAP.md` so it starts from audited truth instead of stale future claims
- Rewrote `rocquantum/src/hipStateVec/MULTI_GPU_GUIDE.md` to match the actual partial multi-GPU state

### Compiler/runtime truth fixes

- Updated `bindings.cpp` so `compile_and_execute` is explicitly documented as a stub that currently raises

### Python API truth fixes

- Updated `rocq/operator.py` to make it explicit that the canonical top-level expectation API is not yet wired to native backend helpers
- Updated `python/rocq/api.py` to:
  - make gate-fusion status explicit in `flush()`
  - convert backend `NOT_IMPLEMENTED` statuses into clearer `NotImplementedError` messages
  - clarify that `Circuit.expval()` is a host-side NumPy fallback

### Packaging and CI truth fixes

- Added `pybind11>=2.10` to `pyproject.toml` build-system requirements
- Added explicit `CMAKE_HIP_ARCHITECTURES` values to `.github/workflows/rocm-linux-build.yml`
- Corrected the stale target name in `rocquantum/tests/hipDensityMat/CMakeLists.txt`

### Test truth fixes

- Converted `tests/test_advanced_gates.py` from placeholder passing tests into an explicit skipped blueprint module

## Why These Changes Were Safe

- They do not claim new functionality that the backend does not already have.
- They mainly reduce false positives, tighten user-facing messaging, and repair obvious packaging/test metadata issues.
- They avoid speculative large-scale refactors of compiler, runtime, or distributed code.

## Remaining High-Value Follow-Up Work

- Unify Python binding names and install paths
- Expose native expectation values through the canonical public API
- Wire gate fusion into the active execution path
- Add real ROCm runtime tests for statevector, density matrix, and multi-GPU smoke
- Complete install/export package configuration
