# Implement Now Plan

Audit date: 2026-04-05

Historical-plan notice (2026-07-15): this is the original low-blast-radius implementation plan, not the current backlog. `TOP_GAPS_AND_PRIORITIES.md` is the current priority list. The host-only packaging gate, example canonicalization, and evidence-stage documentation have since been completed; native ROCm validation remains pending because no AMD GPU is available.

## 1. Compiler/runtime MVP execution path

- Change: keep compiler/runtime parity claims narrow while wiring `rocqCompiler::MLIRCompiler::compile_and_execute()` for the supported qalloc/H/X/Y/Z/CNOT/RX/RY/RZ MLIR subset
- Files:
  - `bindings.cpp`
  - `rocqCompiler/MLIRCompiler.cpp`
  - `README.md`
  - `CURRENT_STATE_AUDIT.md`
- Tests to add or update:
  - `tests/test_p1_compiler.py`
- Validation:
  - `rg -n "compile_and_execute|emit_qir" bindings.cpp rocqCompiler/MLIRCompiler.cpp README.md`
  - `python -m unittest tests.test_p1_compiler -v`
- Risk: Medium; this is an MVP execution bridge, not a full CUDA-Q-style compiler runtime

## 2. Resolve the multi-GPU truth story

- Change: rewrite docs and improve Python-side error messages so `multi_gpu=True` is explicitly experimental partial support
- Files:
  - `rocquantum/src/hipStateVec/MULTI_GPU_GUIDE.md`
  - `python/rocq/api.py`
  - `README.md`
- Tests to add or update:
  - Add a Python contract test for clear `NOT_IMPLEMENTED` messaging in multi-GPU mode
- Validation:
  - `rg -n "multi_gpu|NOT_IMPLEMENTED|distributed" python/rocq/api.py rocquantum/src/hipStateVec/MULTI_GPU_GUIDE.md README.md`
  - multi-GPU smoke on ROCm Linux
- Risk: Low to Medium

## 3. Make the expectation-value story honest

- Change: keep the canonical top-level operator API explicitly gated and document that only the legacy `_rocq_hip_backend` path currently exposes native expectation helpers
- Files:
  - `rocq/operator.py`
  - `python/rocq/api.py`
  - `README.md`
- Tests to add or update:
  - `tests/test_p0_fixes.py`
  - `tests/test_cpp_expectation.py`
- Validation:
  - `python -m unittest tests.test_p0_fixes tests.test_cpp_expectation -v`
- Risk: Low

## 4. Collapse the packaging truth gap — completed for the host-only boundary

- Completed change: scikit-build-core uses `cmake.version`; the default PEP 517 path produces a pure `py3-none-any` host wheel without native sources/binaries, while `ROCQ_BUILD_NATIVE=1` atomically selects native CMake mode and platform wheel tags.
- Files:
  - `pyproject.toml`
  - `setup.py`
  - `CMakeLists.txt`
  - `.github/workflows/rocm-linux-build.yml`
- Tests to add or update:
  - `tests/test_p2_packaging.py`
- Validation:
  - clean `python -m pip wheel --no-deps --no-cache-dir --wheel-dir dist .`
  - install into a fresh environment and import `rocq`, `rocquantum`, and `rocq_cli` outside the source tree
  - run installed `rocq --help`
  - copy and run all 19 examples outside the repository with `PYTHONPATH` removed
- Result: host-only wheel/install/import/CLI/example checks pass on Python 3.9, 3.12, and 3.13. Native wheel/install gates now include relative RPATH, `readelf`/`ldd`, external fresh-environment imports for separate C64/C128 wheels, dtype/itemsize/device-free round-trip assertions, installed symbol execution, and C64 plus C128/METIS consumer ABI/capability checks in ROCm CI; they remain unverified locally without a retained ROCm artifact.

## 5. Replace placeholder advanced-gate tests with honest status

- Change: stop passing blueprint tests as if they verified runtime correctness; canonical mock state-vector tests now compare actual statevectors for advanced named gates when native ROCm hardware is unavailable
- Files:
  - `tests/test_advanced_gates.py`
  - follow-up runtime tests should target `python/rocq/api.py` and native bindings
- Tests to add or update:
  - `tests/test_advanced_gates.py` replaces the skip-only blueprint with explicit CPU mock statevector comparisons for phase, controlled phase/rotation, Toffoli, and CSWAP semantics
- Validation:
  - `python -m pytest tests/test_advanced_gates.py -q`
  - ROCm runtime regression on CRX, CCX, and CSWAP
- Risk: Low

## Immediate Edit Targets

- `README.md`
- `ROADMAP.md`
- `rocquantum/src/hipStateVec/MULTI_GPU_GUIDE.md`
- `bindings.cpp`
- `python/rocq/api.py`
- `rocq/operator.py`
- `pyproject.toml`
- `.github/workflows/rocm-linux-build.yml`
- `tests/test_advanced_gates.py`

## Notes

- Remaining follow-up: validate native install/export on ROCm hardware and unify the remaining Python binding/runtime names.
