# rocQuantum Python examples

These examples use the canonical `rocq` Python API: `@rocq.kernel`, `qvec`,
`get_state`, `sample`, and `observe`. Install the project before running them.

On a machine without an AMD GPU or native ROCm bindings, explicitly enable the
CPU correctness fallback:

```powershell
$env:ROCQ_ENABLE_MOCK_BACKENDS = "1"
python examples/run_bell_state.py
```

```bash
ROCQ_ENABLE_MOCK_BACKENDS=1 python examples/run_bell_state.py
```

The fallback emits `MockBackendWarning`. A successful fallback run checks the
Python/runtime contract only; it is not evidence of native ROCm correctness or
performance.

Executable runtime examples include Bell/GHZ state preparation, sampling,
expectation values, density-matrix noise, controlled gates, parameter-shift
gradients, reduced educational VQE workloads, and the supported repetition-code
QEC subset.

`adjoint_example.py`, `dynamic_circuit_example.py`,
`multi_gpu_swap_example.py`, `slicing_example.py`, and
`tensornet_example.py` also print the relevant capability boundary. They do not
claim that release-wired adjoint generation, mid-circuit feedback, multi-GPU
execution, or a stable TensorNet Python API is currently available.
