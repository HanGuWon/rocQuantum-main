# Top Gaps And Priorities

Audit date: 2026-04-05

## Top 10 Gaps

1. `compile_and_execute()` is a stub while bindings and surrounding docs still made compiler/runtime parity easy to overread.
2. Multi-GPU support is partial and ambiguous: real scaffolding exists, but many distributed paths still return `ROCQ_STATUS_NOT_IMPLEMENTED`.
3. Native expectation kernels exist in `hipStateVec`, but the high-level user-facing API story is split between unimplemented top-level APIs and host-side NumPy fallbacks.
4. The repo contains two divergent Python stacks, `rocq` and `python/rocq`, without one canonical runtime/compiler story.
5. Packaging and build surfaces do not describe one releasable product: `pyproject.toml`, `setup.py`, root CMake, and dormant `_rocq_hip_backend` CMake do not agree.
6. Gate fusion exists in C++ but is not wired into the main Python execution path that claims to flush fused queues.
7. `hipTensorNet` has real core functionality, but optimizer, slicing, and dtype breadth are overstated relative to what is built and tested.
8. `hipDensityMat` is real but narrow; generic channel application, density-matrix sampling, and richer observable support are still missing.
9. Framework integrations are thin adapters with host-side sampling or mock-heavy tests, not strong native ROCm end-to-end proof.
10. Higher-level CUDA-QX-style libraries are still shells; VQE and QEC are not serious supported workflows yet.

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
| Truth-fix compiler/runtime docs and bindings | Largest false positive versus CUDA-Q | `compile_and_execute` is visibly documented as stub everywhere it appears |
| Truth-fix multi-GPU docs and Python errors | Largest ambiguity in advertised capability | `multi_gpu=True` unsupported paths raise explicit partial-support errors |
| Truth-fix expectation-value story | VQE/hybrid workflows depend on it | High-level docs distinguish native expectation helpers from host-side fallback |
| Truth-fix packaging/install story | Users cannot infer one supported install path today | README and audit docs explain the active build path and current limitations |
| Normalize ROCm support policy | Repo needs a current AMD-targeted compatibility statement | Linux-first, ROCm `7.2.0` target, `gfx950/gfx942/gfx90a` plan, and `gfx90a` minimum are documented |

## P1 Backlog

| Item | Why It Is P1 | Acceptance |
| --- | --- | --- |
| Unify Python runtime surfaces | Current duplication causes product confusion | One primary surface owns execution and expectation APIs |
| Wire gate fusion into active execution path | Native code already exists | Queue flush path can enable fusion and is tested |
| Expose native expectations in canonical API | Backend capability is currently hidden | Top-level public API can call native helpers for supported cases |
| Repair package/export/install tree | Release engineering is not yet credible | Install tree has config/version files and working headers |
| Expand runtime CI | Current runtime proof is too narrow | Statevector, density matrix, and expectation tests run on ROCm CI |

## P2 Backlog

| Item | Why It Is P2 | Acceptance |
| --- | --- | --- |
| Complete compiler-driven runtime | Bigger architectural work | MLIR/QIR execution path is real, tested, and documented |
| Complete distributed multi-GPU | Requires deeper runtime design and test infrastructure | Distributed gates, measurement, and sampling are proven on multi-GPU runners |
| Add CUDA-QX-style solver/QEC libraries | Higher-level scope should not mask base gaps | QEC/VQE examples run on supported backends with real tests |
| Broaden provider/integration maturity | Secondary to local ROCm credibility | Native and remote adapter guarantees are explicit and tested |

## Verification Commands

```bash
rg -n "compile_and_execute|multi_gpu|expval|GateFusion|rocquantum_bind|_rocq_hip_backend" .
cmake -S . -B build-ci -G Ninja -DBUILD_TESTING=ON -DROCQUANTUM_BUILD_BINDINGS=ON -DCMAKE_HIP_COMPILER=/opt/rocm/bin/hipcc
cmake --build build-ci --parallel
ctest --test-dir build-ci --output-on-failure
python -m unittest tests.test_p0_fixes tests.test_p1_compiler tests.test_p2_packaging tests.test_cpp_expectation
```

The last four commands require a Linux ROCm environment that is not available in this shell.
