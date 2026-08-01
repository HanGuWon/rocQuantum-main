# Native MLIR/LLVM/QIR Compiler Status

Status date: 2026-07-31

The optional `rocqCompiler/` graph is now a real CPU-buildable compiler path rather than an
unreachable source fragment. It is deliberately narrower than CUDA-Q: it compiles rocQuantum's
recorded straight-line quantum MLIR, including a constrained terminal-measurement form, not
arbitrary C++ or Python source.

## Implemented Pipeline

```text
rocq QuantumKernel textual MLIR
  -> TableGen-generated qubit/result dialect types and operations
  -> validation of the static-custom or terminal-MZ Base Profile contract
  -> explicit rocq-qir-static-pipeline or rocq-qir-base-pipeline
  -> direct Quantum-to-LLVM-dialect conversion
  -> MLIR LLVM-IR translation
  -> QIR 2 profile shaping, entry attributes, and module flags
  -> project structural verifier and LLVM verifier
  -> offline: LLVM IR, binary bitcode, or a generic-host PIC relocatable object
  -> qir-v2-static execution: MLIR ExecutionEngine / LLVM ORC JIT
       -> registered QIS callbacks -> QuantumBackend -> final state vector
```

The supported compiler build is independent of HIP:

- `ROCQUANTUM_BUILD_NATIVE=OFF`
- `ROCQUANTUM_ENABLE_MLIR_COMPILER=ON`
- C++ only; no `hip`, rocBLAS, rocSOLVER, `/dev/kfd`, or AMD GPU is required
- LLVM/MLIR `22.1.x` is required for a release build
- `ROCQUANTUM_ALLOW_UNSUPPORTED_MLIR=ON` is a development-only compatibility escape hatch

The canonical targets are:

- `RocqQuantumIncGen`: dialect/type/op TableGen output
- `RocqCompilerPassIncGen`: declarative pass API output
- `QuantumDialect`: the single owner of generated definitions
- `RocqQuantumToQIR`: direct conversion library
- `rocqCompiler`: offline compiler and optional execution API
- `rocqCompilerTooling`: content-addressed cache implementation used by the CLI
- `rocq-opt`: parse/round-trip/pass driver
- `rocq-translate`: strict MLIR-to-QIR/artifact driver
- `rocq-run`: installed measurement-free static-QIR runner using `cpu_statevec`

The incompatible legacy `rocquantum/include/rocquantum/Dialect` tree is not linked. The simulator
intermediate dialect is retained only behind `ROCQ_COMPILER_BUILD_EXPERIMENTAL_SIMULATOR_DIALECT`;
it is not on the QIR release path.

## QIR Contract

Both profiles support one defined, no-argument/no-result, single-block source `func.func` with one
positive static `quantum.qalloc`. They intentionally have different result contracts:

- `qir-v2-static` is the original measurement-free `custom` profile and requires zero results
- `qir-v2-base` requires at least one terminal `quantum.mz` with a unique, non-empty, NUL-free
  `registerName` and does not accept a unitary operation or a second use of a qubit after measurement

Implemented behavior:

- qalloc results become opaque QIR qubit pointers via static `i64 -> ptr` handles
- `!quantum.result` values become static opaque result handles
- H/X/Y/Z/S/S-adjoint/T/T-adjoint/CNOT and RX/RY/RZ/R1 use arity-correct QIS declarations
- CZ and SWAP are decomposed to H/CNOT
- CCX uses a no-ancilla Clifford+T decomposition
- one-control MCX lowers to CNOT and two-control MCX uses the CCX decomposition
- CSWAP uses CNOT plus the CCX decomposition
- CRX/CRY/CRZ/CP are decomposed to single-qubit rotations/phases and CNOT
- `qir-v2-static` receives `entry_point`, `qir_profiles="custom"`,
  `required_num_qubits`, and `required_num_results="0"`
- `qir-v2-base` receives `qir_profiles="base_profile"`, exact static result counts,
  `output_labeling_schema="schema_id"`, and QIR v2 module flags; its entry is shaped into
  initialize/body/measurements/output blocks, calls `__quantum__qis__mz__body` with static result
  handles, calls `__quantum__rt__result_record_output` with the declared labels, and returns `i64 0`
- the module receives QIR major/minor and static resource-management flags
- both the lowered MLIR module and final LLVM module are verified, and the Base terminal-MZ subset
  is checked by the project's structural verifier plus LLVM `llvm-as` and `opt -passes=verify`
- parse, validation, pass, translation, and verification failures throw exceptions; error strings
  are never returned as if they were QIR

This is a deliberately bounded Base Profile implementation, not a claim about adaptive or full QIR.
The default remains `qir-v2-static` so existing measurement-free callers do not silently change
profile contracts.

Fail-closed boundaries:

- mid-circuit measurement/reset, `read_result`, and measurement-driven feedback
- classical branches/loops and adaptive execution
- native typed SSA function arguments/returns, helper functions, and multiple source functions
- dynamic qubit/result management
- MCX with three or more controls, pending a QIR control-array ABI/runtime
- arbitrary multi-control synthesis beyond the documented one- and two-control MCX subset
- unknown operations, wrong arities, duplicate qubits, invalid qalloc, and non-finite angles

## Artifact, Optimization, Pipeline, And Cache Contract

`MLIRCompiler::emit_artifact` and `rocq-translate --emit` support:

- textual `llvm-ir` and binary `llvm-bc` at `-O0` through `-O3` for `qir-v2-static`
- `qir-v2-base` LLVM IR/bitcode only at `-O0`, because generic LLVM optimization can destroy the
  required four-block profile shape
- a generic-CPU, position-independent relocatable `object` for the compiler build host at `-O0`
  through `-O3` for either profile

Objects are linker inputs, not runnable programs. They intentionally retain unresolved QIS/runtime
symbols. Separately, `compile_and_execute()` translates verified `qir-v2-static` LLVM-dialect IR
in process, creates an MLIR `ExecutionEngine` backed by LLVM ORC, registers the supported QIS
symbols, and forwards those callbacks to a `QuantumBackend`. That bounded JIT does not turn the
offline object format into a linked executable or provide a general QIR runtime. The
profile-preserving QIR interchange contract applies to IR/bitcode, not to an optimized native
object after lowering.

`rocq-opt` registers `rocq-qir-static-pipeline` and `rocq-qir-base-pipeline` explicitly. Their
generic canonicalization/CSE cleanup respects quantum operation effects but is not described as
quantum circuit optimization, cancellation, or commutation.

The opt-in `--cache-dir` cache is content-addressed by a length-delimited SHA-256 input containing
the MLIR source, profile, qubit constraint, artifact kind, optimization level, normalized target
contract, exact LLVM version, schema, and configured compiler-source fingerprint. Entries have a
self-validating envelope and same-directory atomic immutable commit; missing entries are misses,
while corrupt entries, I/O failures, and different bytes for an existing key fail closed. The
cache directory is an explicitly trusted input, so this is not an authenticity or sandbox boundary.

`rocq-translate` accepts exactly one file or stdin input, rejects duplicate/unknown/malformed
options before compilation, requires a file for binary output, writes file output atomically,
reports usage errors with status 2 and compile/cache/I/O failures with status 1, and supports
`--help`, `--version`, `--verbose`, explicit profiles, output formats, optimization levels, and
qubit constraints.

## Offline And GPU Execution Boundaries

`MLIRCompiler(num_qubits)` is an offline constructor. It does not construct a HIP backend and can
emit QIR/compiler artifacts on a machine with no AMD GPU. `QuantumKernel.qir()` and
`QuantumKernel.emit_artifact()` use the compiler-enabled binding where supported or discover
installed `rocq-translate` through `PATH` / `ROCQ_TRANSLATE_EXECUTABLE` and send MLIR over stdin
through a non-shell subprocess. Supplying `cache_dir` intentionally uses the CLI cache contract.

`MLIRCompiler(num_qubits, backend)` and `compile_and_execute()` are the execution boundary.
Measurement-free `qir-v2-static` source is structurally validated, lowered to LLVM/QIR, verified,
JIT-compiled in process, and executed through registered H/X/Y/Z/S/S-adjoint/T/T-adjoint/CNOT and
RX/RY/RZ/R1 QIS callbacks. Higher supported gates reach the backend through their verified QIR
decompositions rather than source-operation replay. `cpu_statevec` is a small deterministic host
reference backend and needs no AMD GPU; `hip_statevec` still requires a native ROCm build and AMD
device. Terminal `quantum.mz` / `qir-v2-base` remains emission-only and is rejected by
`compile_and_execute()`. Capability reporting exposes offline artifacts, static QIR JIT, CPU
reference execution, and HIP execution separately.

Only `rocq-opt`, `rocq-translate`, and `rocq-run` are installed. `QuantumDialect`, `RocqQuantumToQIR`,
`rocqCompiler`, `rocqCompilerTooling`, their headers, and the `rocquantum::compiler` alias remain
build-tree-only; this work does not expose a supported installed/exported compiler C++ SDK.

## Validation Performed Without An AMD GPU

The release version gate rejects non-22.1 LLVM/MLIR by default. Official apt.llvm.org
LLVM/MLIR `22.1.8` packages were extracted into a user-local WSL prefix, and the root project was
configured with the release gate intact (no `ROCQUANTUM_ALLOW_UNSUPPORTED_MLIR` override):

- top-level compiler-only configure succeeded
- all dialect/type/op/pass TableGen outputs were generated
- dialect, direct conversion, compiler/tooling libraries, both tools, and all compiler test
  executables compiled and linked
- the final release-pinned build passed all 12 registered compiler CTests, covering static and
  terminal-MZ Base Profile QIR, explicit pipeline selection/rejection, LLVM ORC execution,
  QIS-to-backend dispatch, QIR decomposition rather than source replay, deterministic CPU
  state-vector Bell and bounded-MCX results, callback/initialization failure cleanup, concurrent
  per-thread execution contexts, deterministic IR/bitcode/generic-host object emission, parallel
  object emission, strict CLI parsing/I/O, cache miss/hit, corrupt-entry rejection, and LLVM
  artifact inspection
- the final compiler smoke, which includes two concurrent independent JIT engines, passed 10
  consecutive repeat-until-fail runs
- generated static and Base IR/bitcode was accepted by in-process `verifyModule`, `llvm-as`,
  `llvm-dis`, and `opt -passes=verify`; `llvm-readobj` / `llvm-nm` verified relocatable host objects
  and their unresolved QIS symbols
- a clean install prefix contained only the three compiler tools plus the existing Python CLI;
  installed `rocq-run` inferred the Bell fixture's two-qubit allocation and emitted the verified
  `rocq-state-vector-v1` JSON state, while installed `rocq-translate` emitted static/Base QIR and
  offline artifacts accepted by the applicable LLVM 22.1 tools
- canonical Python `QuantumKernel.qir()` successfully used that installed translator with no
  `rocquantum_bind`; builder terminal-measurement MLIR reached the Base Profile path, and capability
  reporting identified the fallback as `rocq_translate_cli`

This is LLVM/MLIR 22.1 host compiler and CPU-reference runtime evidence, not ROCm execution
evidence. The CPU GitHub
workflow now defines the same 22.1 compiler-only build and retains configure/build/CTest logs;
the first green hosted artifact is still pending. HIP execution still needs an AMD runner.

## Remaining CUDA-Q Compiler Gap

This work closes the previously largest local defect: there is now a native generated dialect,
conversion pass, LLVM translation, verified QIR, tools, CMake graph, and GPU-independent test path.
It does not establish CUDA-Q parity. Major remaining items include:

- C++/Python AST frontends comparable to `nvq++` and CUDA-Q's Python compiler
- native typed SSA arguments/returns, helper functions, lambdas, and value semantics; Python
  builder calls are host static inlining rather than native function calls
- mid-circuit measurement/reset, `read_result`, branches, loops, feedback, and adaptive/full-profile
  runtime semantics; terminal MZ Base emission does not provide these
- quantum-aware canonicalization/optimization/decomposition pass libraries beyond the explicit
  effect-safe lowering pipelines
- Base/adaptive result runtime semantics, general executable linking, pass plugins, and installed
  compiler SDK/target/backend packaging; the static ORC/QIS bridge does not close these gaps
- arbitrary multi-control lowering beyond one- and two-control MCX
- dynamic/full-profile runtime ABI and multi-QPU scheduling
- a retained hosted LLVM/MLIR 22.1 CI artifact and native ROCm binding/device validation

## Primary Reference Baselines

- [CUDA-Q LLVM/MLIR 22.1 build graph](https://github.com/NVIDIA/cuda-quantum/blob/e8cf932e6b2769dce5763ccf6acc2aa4cf891b7d/CMakeLists.txt)
- [CUDA-Q in-process MLIR/LLVM JIT](https://github.com/NVIDIA/cuda-quantum/blob/e8cf932e6b2769dce5763ccf6acc2aa4cf891b7d/runtime/internal/compiler/JIT.cpp)
- [CUDA-Q QIS/runtime bridge](https://github.com/NVIDIA/cuda-quantum/blob/e8cf932e6b2769dce5763ccf6acc2aa4cf891b7d/runtime/nvqir/NVQIR.cpp)
- [CUDA-Q direct QIR conversion pipeline](https://github.com/NVIDIA/cuda-quantum/blob/e8cf932e6b2769dce5763ccf6acc2aa4cf891b7d/cudaq/lib/Optimizer/CodeGen/ConvertToQIRAPI.cpp)
- [LLVM standalone dialect example](https://github.com/llvm/llvm-project/tree/def143a6c624dc9b991ebfdfec5c36a7084171eb/mlir/examples/standalone)
- [MLIR dialect conversion](https://mlir.llvm.org/docs/DialectConversion/)
- [Official LLVM Debian/Ubuntu packages](https://apt.llvm.org/)
- [QIR 2 version compatibility](https://github.com/qir-alliance/qir-spec/blob/f5647346542d5a65225c3eb349847fe4df01d1b2/specification/README.md#version-compatibility)
- [QIR Base Profile](https://github.com/qir-alliance/qir-spec/blob/f5647346542d5a65225c3eb349847fe4df01d1b2/specification/profiles/Base_Profile.md)
- [QIR Adaptive Profile](https://github.com/qir-alliance/qir-spec/blob/f5647346542d5a65225c3eb349847fe4df01d1b2/specification/profiles/Adaptive_Profile.md)
