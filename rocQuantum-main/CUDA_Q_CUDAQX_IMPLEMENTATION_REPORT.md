# CUDA-Q / CUDA-QX 비교 및 GPU 독립 구현 보고서

기준일: 2026-07-31

## 결론

이번 작업 전의 rocQuantum은 CUDA-Q와 비슷한 Python 문법으로 게이트를 기록하고
시뮬레이터에서 재생하는 계층은 있었지만, CUDA-Q의 typed compiler/runtime 생태계와
CUDA-QX의 범용 solver/QEC 라이브러리라고 부르기에는 기능 격차가 컸다. 특히
`@rocq.kernel` 함수는 호출 시 Python에서 eager 실행되어 게이트 목록을 만들었고,
네이티브 MLIR 컴파일러는 release CMake 그래프에서 의도적으로 비활성화되어 있었다.
CUDA-QX 계층도 MaxCut 전용 QAOA, 제한된 VQE, 3-qubit repetition helper가 중심이었다.

이번 패스에서는 AMD GPU 없이도 결과를 판정할 수 있는 기능을 우선 구현했다.

- typed `make_kernel` builder와 인자 표현식, device-style qubit 인자, `call` / `apply_call`
  정적 인라이닝, 지원 게이트의 정확한 adjoint 역변환, 제한된 canonical control 합성
- terminal `mz` / `mx` / `my`, measurement handle/샘플 대상 선택,
  `quantum.mz` / `!quantum.result` MLIR
- local target context, CUDA-Q형 result/async result, `shots_count`, 단일-QPU `qpu_id=0`
- resource estimation, circuit drawing, MLIR/OpenQASM 2 변환, static-custom QIR과 labeled
  terminal-MZ Base Profile QIR
- LLVM IR/bitcode/generic-host PIC relocatable object 출력, 명시적 static/base pipeline,
  strict `rocq-translate`, content-addressed self-validating artifact cache
- measurement-free static QIR을 실행하는 LLVM ORC JIT/QIS bridge, deterministic CPU
  reference state-vector backend, 설치형 `rocq-run`, ROCm 독립 `qpp-cpu` Python target
- 검증된 Kraus 및 built-in noise channel 객체
- CUDA-Q 규약 `exp(+i theta P)` Pauli exponential과 dense operator 변환
- 소규모 CPU Schrodinger/Lindblad dynamics 및 host async wrapper
- functional VQE, 범용 real-Pauli QAOA, ADAPT-VQE, registry operator pool, PySCF 기반
  restricted chemistry/Jordan-Wigner, CUDA-QX 순서의 UCCSD pool/state preparation
- QEC Code/Decoder registry, repetition/Steane metadata, LUT/BP decoder, code-capacity sampling

이 결과는 큰 GPU 독립 기능 격차를 줄이지만 CUDA-Q 또는 CUDA-QX 전체 parity는 아니다.
네이티브 lowering, offline artifact 출력, measurement-free static JIT은 이제 존재하지만,
native typed SSA/helper
function/return, mid-circuit measurement/reset/`read_result`, branch/loop/adaptive control,
3-control 이상 MCX, Base/adaptive result runtime, 일반 target/plugin runtime, HIP-stream future,
multi-QPU scheduler, GPU 가속 solver/QEC,
surface-code/DEM/tensor-network/realtime QEC는 여전히 남아 있다.
host object는 미해결 QIS/runtime symbol을 가진 linker 입력일 뿐 실행 파일이 아니다.

## 조사 기준

공식 문서뿐 아니라 해당 시점의 NVIDIA 저장소 main도 함께 고정해 비교했다.

| 기준 | 조사 snapshot | 주요 근거 |
| --- | --- | --- |
| CUDA-Q | `cbc4798e9f0593e17b56ebbec765da60a588554e` | [공식 저장소 snapshot](https://github.com/NVIDIA/cuda-quantum/tree/cbc4798e9f0593e17b56ebbec765da60a588554e), [언어 명세](https://nvidia.github.io/cuda-quantum/latest/specification/cudaq/kernels.html), [Python API](https://nvidia.github.io/cuda-quantum/latest/api/languages/python_api.html) |
| CUDA-QX | `890e0d797a0e7a85814a8380557ef3543463bbd4` | [공식 저장소 snapshot](https://github.com/NVIDIA/cudaqx/tree/890e0d797a0e7a85814a8380557ef3543463bbd4), [solver 소개](https://nvidia.github.io/cudaqx/components/solvers/introduction.html), [QEC 소개](https://nvidia.github.io/cudaqx/components/qec/introduction.html) |
| CUDA-QX release | `0.6.0` | [release](https://github.com/NVIDIA/cudaqx/releases/tag/0.6.0), [설치/호환성](https://nvidia.github.io/cudaqx/quickstart/installation.html) |

증거 상태는 다음과 같이 구분한다.

- `host-contract-tested`: CPU/mock 실행과 수치 oracle로 검증됨
- `source-present`: 소스는 있으나 필요한 native toolchain/hardware에서 실행되지 않음
- `unsupported`: 구현하지 않았고 호출 시 명시적으로 실패하거나 capability에 표시됨
- `not-hardware-tested`: AMD GPU/ROCm 실행 및 성능 증거가 없음

## 상세 격차와 이번 구현

### Compiler / frontend

공식 CUDA-Q는 typed kernel argument/return, 정적·동적 control flow, measurement,
kernel composition, adjoint/control synthesis, dynamic builder를 compiler IR과 runtime으로
연결한다. 관련 기준은 [kernel specification](https://nvidia.github.io/cuda-quantum/latest/specification/cudaq/kernels.html),
[control flow](https://nvidia.github.io/cuda-quantum/latest/specification/cudaq/control_flow.html),
[synthesis](https://nvidia.github.io/cuda-quantum/latest/specification/cudaq/synthesis.html),
[dynamic kernels](https://nvidia.github.io/cuda-quantum/latest/specification/cudaq/dynamic_kernels.html)이다.

| 항목 | 작업 전 | 이번 구현 | 현재 상태 |
| --- | --- | --- | --- |
| Decorated kernel | Python 함수를 실행해 concrete gate list 기록 | 기존 strict validation과 specialization 유지, 기록 중 decorated-kernel 합성 지원 | `host-contract-tested`, 여전히 eager recorder |
| Typed dynamic builder | 없음 | `rocq.make_kernel`, typed scalar/sequence placeholder, device-style `QuakeValue`, index/arithmetic expression, builder qalloc/gates | `host-contract-tested`, host gate-IR specialization |
| Kernel composition | 없음 | `call` / `apply_call` signature 검증, classical expression 치환, local allocation remap을 포함한 정적 인라이닝 | `host-contract-tested`; native SSA call/helper function 아님, measurement-bearing callee 불가 |
| Pauli exponential | 없음 | 두 CUDA-Q형 호출 형태와 `exp(+i theta P)` decomposition | `host-contract-tested` |
| Terminal measurement | 없음 | `mz` / `mx` / `my` handle, basis change, sampling 대상 선택, named `quantum.mz` / `!quantum.result` MLIR | `host-contract-tested`; terminal sampling only, typed return/`read_result` 아님 |
| Measurement-based branch/loop | 없음 | 구현하지 않음 | `unsupported`: mid-circuit/reset/feedback/branch/loop |
| Kernel adjoint/control synthesis | legacy scaffold만 존재 | 지원 canonical gate sequence의 역순·정확한 inverse와 제한된 controlled subset을 host에서 합성 | `host-contract-tested`; native pass 및 arbitrary multi-control은 `unsupported` |

Pauli exponential의 부호는 추측하지 않고 NVIDIA lowering source를 확인했다. 공식
decomposition은 angle을 `-2`배 한 뒤 RZ에 전달하므로 로컬 구현도 같은 규약을 사용한다.
근거: [ExpPauli lowering test](https://github.com/NVIDIA/cuda-quantum/blob/cbc4798e9f0593e17b56ebbec765da60a588554e/cudaq/test/Transforms/DecompositionPatterns/ExpPauliToHRyRzCX.qke).

### Native compiler / IR / QIR

CUDA-Q는 Quake/CC dialect, 변환 pass, LLVM/QIR lowering과 target runtime을 하나의
실행 경로로 연결한다. 공식 확장 경계는 [CUDA-Q IR](https://nvidia.github.io/cuda-quantum/latest/using/extending/cudaq_ir.html)과
[backend extension](https://nvidia.github.io/cuda-quantum/latest/using/extending/backend.html)을 기준으로 했다.

후속 네이티브 컴파일러 구현을 반영한 현재 상태는 다음과 같다.

- top-level CMake는 `ROCQUANTUM_ENABLE_MLIR_COMPILER=ON`을 독립 CPU 전용 모드로 구성하고,
  ordinary host/native 패키징과 분리해 `rocqCompiler`를 release graph에 연결한다.
- `rocqCompiler/CMakeLists.txt`는 LLVM/MLIR 22.1.x를 기본 고정하고 TableGen
  `!quantum.qubit` / `!quantum.result` / `quantum.mz`, direct Quantum-to-LLVM/QIR conversion,
  explicit `rocq-qir-static-pipeline` / `rocq-qir-base-pipeline`, `rocq-opt`,
  `rocq-translate`, `rocq-run`과 compiler artifact/cache/JIT/CLI CTest를 소유한다.
- `compile_and_execute()`는 accepted measurement-free source를 static QIR로 낮추고 LLVM
  검증 후 MLIR `ExecutionEngine`/LLVM ORC JIT에서 실행한다. 등록된 QIS callback만을 통해
  `cpu_statevec` 또는 `hip_statevec` backend에 도달하며, composite gate와 1/2-control MCX도
  source replay가 아니라 QIR decomposition을 실행한다.
- offline `MLIRCompiler(num_qubits)`는 HIP backend를 만들지 않는다. `qir-v2-static`은
  measurement-free `custom` profile을 유지하고, `qir-v2-base`는 unique non-empty label을
  가진 terminal MZ를 initialize/body/measurements/output과 result-record calls로 형성한다.
  이 subset은 project structural verifier와 LLVM verifier, `llvm-as`,
  `opt -passes=verify`로 검증한다.
- `emit_artifact` / strict `rocq-translate --emit`은 LLVM IR, raw bitcode, build-host
  generic-CPU PIC relocatable object를 출력한다. static IR/bitcode와 object는 `-O0`-`-O3`,
  Base IR/bitcode는 `-O0`만, Base object는 `-O0`-`-O3`을 허용한다. object의 QIS/runtime
  symbol은 의도적으로 미해결 상태다.
- opt-in cache는 source/profile/options/target/LLVM/compiler fingerprint를 포함한 SHA-256
  key와 self-validating envelope/atomic immutable commit을 사용하며 corruption과
  determinism violation을 fail-closed한다. 신뢰된 cache directory의 authenticity
  boundary는 아니다.
- canonical Python `QuantumKernel.qir()` / `emit_artifact()`는 compiler binding 또는
  설치된 `rocq-translate`를 사용한다. 설치되는 compiler surface는 `rocq-opt`,
  `rocq-translate`, CPU 실행용 `rocq-run`이며 C++ compiler
  library/header/cache/pipeline API는 build-tree-only다.
- 공식 apt.llvm.org LLVM/MLIR 22.1.8로 configure/build/CTest/clean-install 및 설치 도구의
  Python fallback까지 로컬에서 통과했다. 전체 12개 compiler CTest와 설치된
  `rocq-run`의 Bell-state 파일/stdin 수치 검증도 통과했다. 이는
  `host-contract-tested` 근거이며 AMD GPU 실행 근거는 아니다.

따라서 과거의 build-graph, terminal-MZ QIR, static ORC JIT/QIS bridge,
offline object/cache P0는 닫혔다. 남은 큰
격차는 native typed SSA argument/return/helper function, mid-circuit reset/`read_result`와
adaptive branch/loop, quantum-aware 최적화 pass군, arbitrary multi-control,
Base/adaptive result runtime, general target/plugin ABI, multi-QPU scheduling 및 HIP device
검증이다. `compile_and_execute()`는 실제 static-QIR JIT이지만 여전히 좁은
measurement-free gate subset이며 measurement result를 실행하지 않는다.

### Runtime / target / execution result

공식 비교 기준은 [sample vs run](https://nvidia.github.io/cuda-quantum/latest/using/examples/sample_vs_run.html),
[simulator targets](https://nvidia.github.io/cuda-quantum/latest/using/backends/simulators.html),
[MQPU](https://nvidia.github.io/cuda-quantum/latest/using/backends/sims/mqpusims.html)다.

| 항목 | 작업 전 | 이번 구현 | 제한 |
| --- | --- | --- | --- |
| Target selection | 매 호출 `backend=` 문자열 | ContextVar 기반 registry, set/reset/context manager, explicit override | local backend만 포함 |
| QPU metadata | 없음 | `num_qpus()==1`, async `qpu_id=0` | 0 이외는 fail-closed, scheduler 없음 |
| Sampling result | plain dict | dict-compatible `SampleResult`, probability/shot helpers; builder measurement label은 handle/draw/MLIR/Base-QIR output에 보존 | `SampleResult`는 flat bitstring count이며 named-register partition/`read_result` 없음 |
| Observe result | plain float | float-compatible `ObserveResult` | term-level/shot metadata 없음 |
| Async result | raw Future | Future-compatible `AsyncResult.get()`, target context 보존 | host thread pool, HIP stream 아님 |
| Sampling call shape | mandatory positional shots | 기존 형식 유지 + `shots_count=`, default 1000 | broadcast launch 없음 |
| Tooling | MLIR/QIR 일부 | deterministic Resources, draw, MLIR/OpenQASM 2, QIR fail-closed | 지원하지 않는 op는 명시 실패 |

### Noise / dynamics / operators

공식 기준은 [noise simulation](https://nvidia.github.io/cuda-quantum/latest/examples/python/noisy_simulations.html)과
[dynamics](https://nvidia.github.io/cuda-quantum/latest/using/dynamics.html)다.

- `KrausChannel`은 shape, finite value, power-of-two dimension, CPTP completeness를 검증한다.
- bit flip, phase flip, depolarization, amplitude damping, phase damping 객체를 기존
  `NoiseModel` backend spec과 호환되게 연결했다.
- 기존 문자열 noise API는 그대로 보존했다.
- `operator_to_matrix`는 Pauli, nested/scaled Sum, local dense Hermitian embedding,
  full-register CSR을 qubit-0 LSB 규약으로 변환한다.
- `Schedule`, `evolve`, `evolve_async`는 작은 시스템의 closed-system unitary와
  Lindblad RK4 correctness oracle를 제공한다. CUDA-Q의 `NONE`, 단수형
  `EXPECTATION_VALUE`, `ALL` 저장 모드와 각 결과 shape도 맞췄다.

Dynamics는 CPU reference 구현이다. CUDA-Q의 distributed/GPU dynamics, operator
expression parameter binding, large-system integrator parity를 주장하지 않는다.

## CUDA-QX solver 상세

공식 Python API 기준은 [CUDA-QX solvers API](https://nvidia.github.io/cudaqx/api/solvers/python_api.html)다.

| 기능 | 작업 전 | 이번 구현 | 잔여 격차 |
| --- | --- | --- | --- |
| VQE | class 중심 제한적 objective/gradient | functional `vqe`, 인자 이름과 무관한 parameter-vector callable adapter, immutable iteration trace, execution type, COBYLA/L-BFGS/callable optimizer normalization, 공식 SciPy `method/jac/callback/options` 전달, 증명 가능한 단일 회전만 exact parameter-shift 사용 | GPU adjoint, distributed execution, shot VQE |
| QAOA | MaxCut 전용 | arbitrary real Pauli-sum QAOA, default/custom mixer, shared/full/counter-diabatic parameters, tuple-unpackable result와 `SampleResult` final sample | production batching, GPU gradient |
| Operator pool | 없음 | QAOA pool | broader chemistry/spin pools |
| ADAPT-VQE | 없음 | deterministic host finite-difference ADAPT loop, warm/cold modes, convergence criteria | GPU gradient/commutator acceleration |
| Chemistry | 없음 | precomputed integral container와 Jordan-Wigner tensor construction (`tol` alias 포함), optional PySCF fail-closed | full driver stack, active-space/CCSD breadth |
| GQE/state preparation | 없음 | 구현하지 않음 | `unsupported` |

Shot-based VQE처럼 현재 runtime이 정확히 제공하지 못하는 모드는 근사 성공으로 반환하지
않고 명시적으로 거부한다.

`gradient="parameter_shift"`도 무조건 2점 공식을 적용하지 않는다. eager 회로 기록을
여러 지점에서 비교해 한 host parameter가 단일 `RX/RY/RZ/P` 각도에 계수 ±1로 직접
연결됨을 증명할 수 있을 때만 정확한 shift rule을 사용한다. `2θ`, 공유·비선형·controlled
parameterization은 complex64 취소오차를 고려한 4점 중앙차분으로 전환하고 경고한다.

## CUDA-QX QEC 상세

공식 기준은 [CUDA-QX QEC API](https://nvidia.github.io/cudaqx/api/qec/python_api.html)와
[code-capacity example](https://nvidia.github.io/cudaqx/examples_rst/qec/code_capacity_noise.html)다.

| 기능 | 작업 전 | 이번 구현 | 잔여 격차 |
| --- | --- | --- | --- |
| Code abstraction | 3-qubit helper | validated CSS Code/metadata registry, odd repetition, Steane; 공식 repetition의 비대칭 logical shape, `PauliOperator` stabilizer 및 dense `get_pauli_word()` 반환 | surface/color code 및 circuit generation breadth |
| Decoder abstraction | repetition lookup helper | Decoder registry/result/batch/async, single-error LUT, dense NumPy BP | GPU/TN/DEM/OSD 및 calibration-aware models |
| Code-capacity | 없음 | seeded Bernoulli error 및 exact GF(2) syndrome sampling | native accelerated sampling |
| Runtime QEC | sequential helper | 기존 API 보존, registry/capability 연결 | mid-circuit feedback, realtime/device-resident QEC |

## 검증 결과

AMD GPU 없이 가능한 통합 검증을 완료했다.

- 전체 Python 회귀와 builder composition/adjoint/control/terminal-measurement focused suite가
  통과했다. 숫자는 최종 실행 artifact를 기준으로 하며 문서에 오래된 고정 count를 복제하지 않는다.
- skip은 ROCm/선택적 외부 환경 의존 항목이며, 명시적으로 활성화한
  `ROCQ_ENABLE_MOCK_BACKENDS=1` 경로의 경고는 예상된 mock-backend 경계다.
- Pauli exponential, Rabi evolution, density-unitary, amplitude damping, generic QAOA,
  Jordan-Wigner, GF(2) syndrome, VQE `ry(2θ)` gradient analytic oracle를 포함한다.
- 변경 Python 파일 Ruff 검사, `compileall`, `git diff --check`를 통과했다.
- PEP 517로 `rocquantum-0.1.0-py3-none-any.whl`을 실제 빌드했고, 새 builder/dynamics,
  solver, QEC 모듈이 wheel에 포함됨과 소스 트리 밖 isolated import smoke를 확인했다.
- LLVM/MLIR 22.1 compiler CTest 12/12는 static ORC JIT/QIS dispatch, CPU Bell/MCX 수치,
  lifecycle/concurrency, static/Base QIR, explicit pipeline, deterministic IR/bitcode/object,
  strict CLI, cache hit/miss/corruption을 검증했다. `llvm-as`/`opt`와
  `llvm-dis`/`llvm-readobj`/`llvm-nm`이 해당 artifact를 독립 검사했다.
- 실제 PySCF 2.14/Linux H₂/STO-3G에서 RHF/FCI, 2전자 sector 고유값, UCCSD VQE=FCI를
  검증했고 chemistry vertical 8/8이 통과했다.
- mock backend 결과는 native ROCm correctness/performance 증거가 아니라 CPU 계약·수치
  oracle로만 사용했다.

## 남은 작업 우선순위

1. **P0 — native typed/value semantics:** native SSA kernel argument/return, helper function/call,
   value semantics과 Python/C++ source frontend 연결.
2. **P0 — dynamic quantum semantics:** mid-circuit measurement/reset, `read_result`,
   classical branch/loop trajectory, adaptive/full runtime profile, `run` result contract.
3. **P0/P1 — broader compiler runtime:** Base/adaptive measurement/result ABI, typed helper
   functions/calls, general target/plugin lifecycle, reusable compiler/pass SDK. 현재 static ORC
   JIT/QIS bridge와 `rocq-run`은 measurement-free bounded subset이고 object/cache는 offline
   artifact 기능이다.
4. **P1 — runtime scheduling:** true QPU discovery, `qpu_id` routing, HIP-stream async,
   broadcast/batch execution, MPI/multi-node scheduler.
5. **P1 — finite-shot observe/noise:** term grouping, shot allocation, seed/reproducibility,
   trajectory-aware noise and result metadata.
6. **P1 — solver acceleration:** adjoint gradient, batched observe, GPU optimizer loop,
   chemistry driver breadth, GQE/state-preparation modules.
7. **P1/P2 — QEC breadth:** surface code, detector error model, correlated noise,
   tensor-network/GPU decoder, realtime feedback.
8. **필수 외부 검증:** AMD GPU에서 state-vector/density/noise/dynamics/solver/QEC native
   correctness 및 성능 artifact를 CI에 보존.

따라서 현재의 올바른 제품 설명은 **CUDA-Q-inspired ROCm simulation SDK with a
substantially expanded host-tested runtime and CUDA-QX reference layer**이다. CUDA-Q 또는
CUDA-QX의 drop-in replacement라는 표현은 아직 부정확하다.
