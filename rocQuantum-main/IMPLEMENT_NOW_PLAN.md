# Implement Now Plan

Audit date: 2026-04-05

## 1. Truth-fix compiler/runtime parity

- Change: document and surface `rocqCompiler::MLIRCompiler::compile_and_execute()` as a stub everywhere the shipped API exposes it
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
- Risk: Low

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

## 4. Collapse the packaging truth gap

- Change: make the build-system requirements honest now, then follow with a larger unification pass later
- Files:
  - `pyproject.toml`
  - `setup.py`
  - `CMakeLists.txt`
  - `.github/workflows/rocm-linux-build.yml`
- Tests to add or update:
  - `tests/test_p2_packaging.py`
- Validation:
  - `pip install -e .`
  - `python -c "import rocq; import rocquantum"`
  - `python -m rocq_cli --help`
- Risk: Low for the immediate truth fix, Medium for the later full unification

## 5. Replace placeholder advanced-gate tests with honest status

- Change: stop passing blueprint tests as if they verified runtime correctness
- Files:
  - `tests/test_advanced_gates.py`
  - follow-up runtime tests should target `python/rocq/api.py` and native bindings
- Tests to add or update:
  - replace skip-only blueprint with actual statevector comparisons when ROCm runtime is available
- Validation:
  - `python -m unittest tests.test_advanced_gates -v`
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

- A later phase should also repair install/export completeness, unify Python binding names, and add install-tree consumption tests.
- Those follow-up items are intentionally not bundled into this first pass because the current goal is a safe audit plus low-blast-radius credibility recovery.
