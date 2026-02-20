# Evidence Index (baseline `dbfd6816d4307b2f869487d0bf36f1c2ad324b3a`)

## CI / Build / Validation
- `.github/workflows/rocm-linux-build.yml:54`-`69`: ROCm compile uses container matrix (`6.2.2`, `7.2.0`).
- `.github/workflows/rocm-linux-build.yml:106`-`110`: runtime tests are skipped when `/dev/kfd` is absent.
- `.github/workflows/rocm-linux-build.yml:11`-`52`: CPU Python test lane exists.
- `rocQuantum-main/build_rocq.bat:13`-`14`: default GPU arch target is `gfx906`.

## hipStateVec Distributed Completeness
- `rocQuantum-main/rocquantum/src/hipStateVec/hipStateVec.cpp:1740`: `rocsvAllocateDistributedState` implementation.
- `rocQuantum-main/rocquantum/src/hipStateVec/hipStateVec.cpp:1874`: `rocsvInitializeDistributedState` implementation.
- `rocQuantum-main/rocquantum/src/hipStateVec/hipStateVec.cpp:744`, `833`, `2139`, `2213`: non-local distributed 1Q/2Q gate `NOT_IMPLEMENTED` paths.
- `rocQuantum-main/rocquantum/src/hipStateVec/hipStateVec.cpp:2612`, `2715`: distributed generic matrix limitations.
- `rocQuantum-main/rocquantum/src/hipStateVec/hipStateVec.cpp:3128`, `3181`, `3234`, `3292`, `3377`: distributed expectation APIs `NOT_IMPLEMENTED`.
- `rocQuantum-main/rocquantum/src/hipStateVec/hipStateVec.cpp:3498`: distributed sampling `NOT_IMPLEMENTED`.
- `rocQuantum-main/rocquantum/src/hipStateVec/hipStateVec.cpp:3688`, `3725`: distributed controlled matrix limits.

## Multi-GPU / RCCL Doc-Code Alignment
- `rocQuantum-main/rocquantum/src/hipStateVec/MULTI_GPU_GUIDE.md:7`: says distributed APIs return `NOT_IMPLEMENTED` on multi-GPU.
- `rocQuantum-main/rocquantum/src/hipStateVec/MULTI_GPU_GUIDE.md:31`, `53`: claims RCCL alltoallv implementation.
- `rocQuantum-main/rocquantum/src/hipStateVec/MULTI_GPU_GUIDE.md:95`: says multi-GPU state distribution not available.
- `rocQuantum-main/rocquantum/src/hipStateVec/hipStateVec.cpp:483`, `527`, `584`: host gather/scatter/remap helpers exist.
- `rocQuantum-main/rocquantum/src/hipStateVec/hipStateVec.cpp:1953`: swap path currently routes to host remap fallback.
- `rocQuantum-main/rocquantum/src/hipStateVec/CMakeLists.txt:41`-`45`: RCCL is optional link dependency.

## hipTensorNet Optimizer / dtype
- `rocQuantum-main/rocquantum/src/hipTensorNet/CMakeLists.txt:10`-`15`: tensornet sources omit `Pathfinder.cpp`.
- `rocQuantum-main/rocquantum/include/rocquantum/hipTensorNet_api.h:7`-`16`: optimizer enum includes GREEDY/KAHYPAR/METIS.
- `rocQuantum-main/rocquantum/src/hipTensorNet/hipTensorNet.cpp:169`-`207`: contract path uses in-file pair selection, not optimizer enum dispatch.
- `rocQuantum-main/rocquantum/src/Pathfinder.cpp:326`-`339`: METIS TODO and runtime throw.
- `rocQuantum-main/rocquantum/src/hipTensorNet/hipTensorNet.cpp:308`-`310`: only C64 accepted in `rocTensorNetworkCreate`.
- `rocQuantum-main/rocquantum/src/hipTensorNet/hipTensorNet.cpp:373`-`374`: SVD path rejects non-C64.
- `rocQuantum-main/python/rocq/bindings.cpp:13`-`16`: Python dtype mapping is broader than backend support.

## Compiler Pipeline
- `rocQuantum-main/rocqCompiler/MLIRCompiler.cpp:48`-`79`: `emit_qir()` implemented.
- `rocQuantum-main/rocqCompiler/MLIRCompiler.cpp:83`-`89`: `compile_and_execute()` throws `not yet implemented`.
- `rocQuantum-main/rocqCompiler/MLIRCompiler.h:18`-`23`: API declares `compile_and_execute` and `emit_qir`.
- `rocQuantum-main/bindings.cpp:22`-`26`: binding exposes `compile_and_execute` but drops dict args.

## Python API Exposure / Skeletons
- `rocQuantum-main/rocquantum/core.py:13`-`30`: backend registry includes skeleton targets.
- `rocQuantum-main/rocquantum/backends/iqm.py:1`-`7`: placeholder backend (`pass`).
- `rocQuantum-main/rocquantum/backends/alice_bob.py:1`-`10`: placeholder backend (`pass`).
- `rocQuantum-main/tests/test_iqm_backend.py:1`, `rocQuantum-main/tests/test_alice_bob_backend.py:1`: TODO-only tests.
- `rocQuantum-main/rocq/backends.py:199`-`204`: mock fallback active when HIP backend missing.
