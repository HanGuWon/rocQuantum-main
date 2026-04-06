# rocQuantum

`rocQuantum` is an experimental ROCm-first quantum computing repository centered on native HIP simulator components:

- `hipStateVec`
- `hipTensorNet`
- `hipDensityMat`

The repo also contains partially implemented compiler, Python, provider, and framework-integration surfaces inspired by CUDA-Q, cuQuantum, and CUDA-QX. It does not yet provide end-to-end parity with those NVIDIA stacks.

## Current Reality

Implemented today:

- Native HIP state-vector simulation for core named gates, sampling, measurement, and several expectation-value primitives
- Native tensor-network contraction core
- Native density-matrix core with a limited noise model set
- Direct simulator execution through the active local runtime path

Only partial today:

- MLIR/QIR compiler flow
- Generic matrix and controlled-unitary coverage
- Multi-GPU / distributed execution
- High-level expectation-value APIs
- Packaging and install/export
- PennyLane, Cirq, and Qiskit adapter maturity

Not implemented today:

- End-to-end compiler-driven `compile_and_execute` parity with CUDA-Q
- Release-grade distributed multi-GPU support
- CUDA-QX-style higher-level libraries with robust QEC/solver coverage

## Audit Documents

This repo now includes an audit-first truth set in the inner repo root:

- `CURRENT_STATE_AUDIT.md`
- `FEATURE_TRUTH_MATRIX.md`
- `ROCM_INTEGRATION_AUDIT.md`
- `TOP_GAPS_AND_PRIORITIES.md`
- `FINAL_GAP_REPORT.md`
- `IMPLEMENT_NOW_PLAN.md`

Use those files as the authoritative capability summary for the current codebase.

## Repo Layout

- `rocquantum/src/hipStateVec`: native state-vector kernels and distributed scaffolding
- `rocquantum/src/hipTensorNet`: tensor-network contraction core
- `rocquantum/src/hipDensityMat`: density-matrix and limited noise support
- `rocquantum/src/simulator.cpp`: public C++ simulator wrapper
- `rocqCompiler/`: partial MLIR/QIR pipeline
- `rocq/`: top-level Python surface with direct execution and mock fallbacks
- `python/rocq/`: separate legacy-style Python surface with `_rocq_hip_backend`
- `integrations/`: PennyLane, Cirq, and Qiskit adapters

## Support Policy For This Audit Pass

- Primary release target: Linux x86_64
- Latest stable ROCm target at audit time: `7.2.0`
- Recommended Tier 1 GPU targets: `gfx950`, `gfx942`, `gfx90a`
- Recommended future minimum release-grade GPU target: `gfx90a`
- Recommended future minimum ROCm target: `6.4.0`
- Current non-experimental CI baseline: ROCm `6.2.2`
- Windows helper scripts are kept for development convenience but are not treated as release-grade support

## Native Component Snapshot

| Component | Current State |
| --- | --- |
| `hipStateVec` | Real and useful, but not fully surfaced through every public API |
| `hipTensorNet` | Real contraction core, narrower than a full cuTensorNet analogue |
| `hipDensityMat` | Real but limited |
| `rocqCompiler` | Partial codegen path, no real compile-and-execute loop |
| Top-level `rocq` | Working direct runtime path with some mock fallbacks |
| `python/rocq` | Separate partial runtime/compiler surface with host-side fallbacks |

## Important Limitations

- `rocqCompiler::MLIRCompiler::compile_and_execute()` is a stub and currently raises.
- `multi_gpu=True` should be treated as experimental partial support, not full distributed execution.
- The canonical top-level `rocq` operator expectation API is not wired to native backend expectations.
- `python/rocq/api.py::Circuit.expval()` is still a host-side NumPy fallback after statevector readback.
- PennyLane and Cirq adapters use host-side sampling paths.
- Several provider backends remain skeletons or thin clients.

## Build

Use an out-of-tree build directory:

```bash
cmake -S . -B build-ci -G Ninja \
  -DBUILD_TESTING=ON \
  -DROCQUANTUM_BUILD_BINDINGS=ON \
  -DCMAKE_HIP_COMPILER=/opt/rocm/bin/hipcc
cmake --build build-ci --parallel
```

For a release-grade Linux ROCm build, set explicit GPU targets:

```bash
-DCMAKE_HIP_ARCHITECTURES="gfx950;gfx942;gfx90a"
```

## Comparison Baselines

- CUDA-Q: `https://nvidia.github.io/cuda-quantum/latest/`
- cuQuantum: `https://docs.nvidia.com/cuda/cuquantum/latest/`
- CUDA-QX: `https://github.com/NVIDIA/cudaqx`

This repo is currently closest to a ROCm-native simulator project with partial higher-level surfaces, not to a finished CUDA-Q/CUDA-QX equivalent.
