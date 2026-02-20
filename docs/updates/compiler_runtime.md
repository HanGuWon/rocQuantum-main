# Compiler Runtime Path (`compile_and_execute`) Draft

## Current State
- `emit_qir()` exists and lowers MLIR to LLVM/QIR text.
- `compile_and_execute()` currently throws at runtime (`rocqCompiler/MLIRCompiler.cpp:83`-`89`).
- Python binding currently discards `args` and passes `{}` (`bindings.cpp:22`-`26`).

## MVP Design
1. Parse MLIR program and extract gate operations.
2. Initialize selected backend (`hip_statevec` now; extensible later).
3. Replay extracted operations via backend API:
   - non-parametric: `h,x,y,z,s,sdg,t,cnot,cz,swap,mcx,cswap`
   - parameterized: `rx,ry,rz,crx,cry,crz`
4. Return final statevector.
5. Emit actionable diagnostics when runtime/toolchain/backend components are missing.

## Diagnostics Contract
- Missing backend name or unsupported backend -> `std::invalid_argument` with supported set.
- Parse failure -> `std::runtime_error` including first unparsed op line.
- Backend call failure -> propagate backend status and op context.

## Test Contract
- Source-contract checks (CPU only): `compile_and_execute` no longer hard-throws placeholder message.
- Runtime execution checks: ROCm self-hosted CI only.
