# Top Gaps And Priorities

Audit date: 2026-04-05

## Top 10 Gaps

1. `compile_and_execute()` now has a narrow qalloc/H/X/Y/Z/CNOT/RX/RY/RZ execution MVP and `rocq.compiler_capabilities()` exposes that partial subset, but bindings and docs must keep it from reading as full compiler/runtime parity.
2. Multi-GPU support is partial and less ambiguous: real scaffolding exists, `rocq.distributed_capabilities()` exposes the supported/unsupported runtime contract, non-local single/control/CNOT/CZ/generic matrix paths now have explicit slow/debug host fallback, and local-domain expectation/sampling/probability reductions have an optional RCCL fast path, but many distributed paths still return `ROCQ_STATUS_NOT_IMPLEMENTED`.
3. Native expectation kernels exist in `hipStateVec`, canonical `rocq.observe()` / `rocq.operator.get_expectation_value()` plus legacy `python/rocq` Pauli expectation paths now reach native helpers, and an experimental Clifford-only `stabilizer` backend can evaluate Pauli propagation for small Clifford circuits, but the user-facing API story is still split across two Python surfaces.
4. The repo still contains two Python stacks, but the canonical story is less ambiguous: `rocq.runtime_capabilities()` identifies `rocq` as the primary runtime and documents `python/rocq` as a compatibility API; full compiler/runtime unification is still open.
5. Packaging and build surfaces are closer but still not one releasable product: root CMake now activates the `_rocq_hip_backend` owner in `python/rocq`, adapter-local compatibility `setup.py` files now follow the root version and dependency floors, while `pyproject.toml` and the canonical/legacy Python surfaces still need consolidation.
6. Gate fusion exists in C++ and is used by the canonical `rocq` backend and legacy `python/rocq` flush path for same-target single-qubit spans plus narrow CNOT-adjacent spans; canonical runtime calls now expose state-vector-only `enable_fusion=` for explicit performance/debug control, and unsupported fusion inputs fail rather than being silently dropped, but broader fusion patterns are still unfused.
7. `hipTensorNet` has real core functionality and now exposes optimizer/dtype/slicing capabilities, but METIS/KAHYPAR pathfinders and runtime slicing remain unsupported unless compiled in.
8. `hipDensityMat` is real but narrow; single- and multi-qubit Kraus channels, canonical CCX/CSWAP decomposition, GPU-side measured-marginal reduction for density sampling, and small dense-observable reductions now exist, while GPU-resident shot sampling, GPU-resident channel planning, native broad multi-control density kernels, and richer observable support are still missing.
9. Framework integrations now cover native sampling, native selected-qubit probability vectors, native Pauli-observable paths, native sparse moments for PennyLane SparseHamiltonian, default multi-control gate dispatch, and selected sparse-observable correctness fallbacks for PennyLane/Qiskit more directly, and self-hosted ROCm CI now has a native binding/PennyLane/Qiskit/Cirq Bell-state smoke path; native ROCm proof still depends on uploaded runner artifacts.
10. Higher-level CUDA-QX-style libraries now have an experimental VQE/QAOA/repetition-code subset, including public VQE energy evaluation, repeated-round repetition-code aggregation, narrow syndrome readout-error mitigation, and solver/QEC capability metadata that exposes the supported/unsupported boundary, but this is still far from a serious supported CUDA-QX analogue.

## Priority Framework

### P0

Scope: stop overclaiming, make unsupported features fail clearly, and define an honest ROCm support boundary.

- Truth-fix docs and API messaging around compiler/runtime, multi-GPU, expectations, and integrations
- Keep compiler/runtime parity explicitly gated until there is a real end-to-end bridge
- Make `multi_gpu=True` read as experimental partial support, not full distributed execution
- Align expectation-value claims with actual native vs host-side behavior
- Normalize the Linux-first ROCm support statement and GPU architecture policy

### P1

Scope: connect existing native capabilities into the real user path and improve release engineering.

- Unify the two Python surfaces or demote one to legacy status
- Expose native expectation capabilities through the canonical public API
- Wire gate fusion into the active Python execution path
- Clean up packaging/build/export so one install path is defensible
- Add real runtime CI coverage across statevector, density matrix, expectations, and basic multi-GPU smoke

### P2

Scope: broaden scope only after truth and core execution are stable.

- Complete compiler/runtime integration beyond QIR emission
- Expand distributed execution beyond the current partial single-node scaffolding
- Add robust higher-level solver and QEC libraries
- Expand ecosystem integrations once the base runtime contract is stable

## P0 Backlog

| Item | Why It Is P0 | Acceptance |
| --- | --- | --- |
| Truth-fix compiler/runtime docs and bindings | Largest false positive versus CUDA-Q | `compile_and_execute` is visibly documented as a narrow MVP subset everywhere it appears |
| Truth-fix multi-GPU docs and Python errors | Largest ambiguity in advertised capability | `multi_gpu=True` unsupported paths raise explicit partial-support errors |
| Truth-fix expectation-value story | VQE/hybrid workflows depend on it | High-level docs distinguish native Pauli expectation helpers from broader operator gaps |
| Truth-fix packaging/install story | Users cannot infer one supported install path today | README and audit docs explain the active build path and current limitations |
| Normalize ROCm support policy | Repo needs a current AMD-targeted compatibility statement | Linux-first, ROCm `7.2.4` target, `gfx950/gfx942/gfx90a` plan, and `gfx90a` minimum are documented |

## P1 Backlog

| Item | Why It Is P1 | Acceptance |
| --- | --- | --- |
| Unify Python runtime surfaces | Current duplication causes product confusion | One primary surface owns execution and expectation APIs |
| Wire gate fusion into active execution path | Native code already exists | Queue flush path can enable fusion and is tested |
| Expand canonical expectation breadth | Pauli, supported canonical/PennyLane Hermitian expectations, and local full-state sparse moments are wired, but broader observables are not | Top-level public API keeps Pauli/dense-matrix/CSR sparse native paths and clearly gates remaining arbitrary-operator gaps |
| Repair package/export/install tree | Release engineering is not yet credible | Install tree has config/version files and working headers |
| Expand runtime CI | Current runtime proof is too narrow | Statevector, density matrix, and expectation tests run on ROCm CI |

## P2 Backlog

| Item | Why It Is P2 | Acceptance |
| --- | --- | --- |
| Complete compiler-driven runtime | Bigger architectural work | MLIR/QIR execution path is real, tested, and documented |
| Complete distributed multi-GPU | Requires deeper runtime design and test infrastructure | Distributed gates, measurement, and sampling are proven on multi-GPU runners |
| Add CUDA-QX-style solver/QEC libraries | Higher-level scope should not mask base gaps | Experimental VQE/QAOA/QEC examples run on supported backends with real tests |
| Broaden provider/integration maturity | Secondary to local ROCm credibility | Native and remote adapter guarantees are explicit and tested |

## Verification Commands

```bash
rg -n "compile_and_execute|multi_gpu|expval|GateFusion|rocquantum_bind|_rocq_hip_backend" .
cmake -S . -B build-ci -G Ninja -DBUILD_TESTING=ON -DROCQUANTUM_BUILD_BINDINGS=ON -DCMAKE_HIP_COMPILER=/opt/rocm/bin/hipcc
cmake --build build-ci --parallel
ctest --test-dir build-ci --output-on-failure
python -m unittest tests.test_p0_fixes tests.test_p1_compiler tests.test_p2_packaging tests.test_statevec_fastpath_contract tests.test_rccl_distributed_contract tests.test_tensornet_contract tests.test_cpp_expectation
python3 benchmarks/run_release_benchmarks.py --build-dir build-ci --output-dir benchmark-artifacts
./build-ci/rocquantum/src/hipStateVec/benchmark_hipStateVec_distributed_reductions --output distributed-reductions.json
```

The CMake, CTest, and benchmark commands require a Linux ROCm environment that is not available in this shell.
