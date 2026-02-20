# Agent 5 - Compiler Pipeline: `compile_and_execute` (QIR -> backend)

## Evidence Table
| Area | Evidence | Observation |
|---|---|---|
| QIR emission exists | `rocQuantum-main/rocqCompiler/MLIRCompiler.cpp:48`-`79` | `emit_qir()` lowers and prints LLVM IR string. |
| Runtime execution missing | `rocQuantum-main/rocqCompiler/MLIRCompiler.cpp:83`-`89` | `compile_and_execute()` unconditionally throws. |
| Public API surface includes method | `rocQuantum-main/rocqCompiler/MLIRCompiler.h:18`-`23` | API contract exists but implementation is absent. |
| Python binding exposes method | `rocQuantum-main/bindings.cpp:22`-`26` | Binding currently ignores `args` and forwards empty map. |
| Compiler test contract still placeholder-oriented | `rocQuantum-main/tests/test_p1_compiler.py:61`-`65` | Only checks no `return {}` and that throw exists. |

## Top 5 Findings
- Execution path is API-visible but functionally unavailable.
- Binding-level argument plumbing is incomplete.
- Existing tests encourage placeholder behavior rather than end-to-end MVP.
- Backend factory currently only supports `hip_statevec`; this is sufficient for MVP.
- Missing toolchain/runtime diagnostics reduce operability in non-ROCm environments.

## Top 5 Actions
- Implement minimal `compile_and_execute()` by parsing emitted MLIR ops and replaying on backend.
- Pass argument map through bindings instead of discarding input.
- Add source-contract tests for parser/executor path presence.
- Add runtime integration tests gated to ROCm self-hosted CI.
- Document dependencies and failure messages in `docs/updates/compiler_runtime.md`.

## Proposed MVP Behavior
- Accept MLIR emitted by `rocq.kernel.QuantumKernel.mlir()`.
- Extract gate ops (`h,x,y,z,cnot,rx,ry,rz,...`) and qubit operands.
- Call backend `initialize -> apply_* -> get_state_vector -> destroy`.
- Throw actionable diagnostics for unknown op or malformed angle attributes.

## Concrete Edits (File List + Rationale)
- `rocQuantum-main/rocqCompiler/MLIRCompiler.cpp`: add parser/executor path.
- `rocQuantum-main/bindings.cpp`: pass `py::dict` to `std::map<std::string,bool>` conversion and forward.
- `rocQuantum-main/tests/test_p1_compiler.py`: replace placeholder throw expectation with MVP contract checks.
- `docs/updates/compiler_runtime.md`: user/dependency guidance.

## Acceptance Criteria
- `compile_and_execute()` returns a non-empty statevector for supported MLIR programs.
- Binding forwards args map and preserves diagnostics.
- CPU/source-level tests validate contract; ROCm lane validates runtime behavior.

## Test Plan
- Verified here: source evidence only.
- Requires ROCm GPU CI:
  - run 3 representative circuits (`H`, `Bell`, `parametric`) through binding.
  - assert statevector parity against known references.

## Risks
- Regex/parser approach can be brittle against MLIR format drift.
- Backend gate coverage mismatches may surface as runtime errors.

## Unknowns
- Whether direct MLIR IR walking (instead of textual parsing) is immediately available without broader pass refactor.
