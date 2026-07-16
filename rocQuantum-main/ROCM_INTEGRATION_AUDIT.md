# ROCm Integration Audit

Audit date: 2026-04-05
Refresh date: 2026-06-10

Historical-snapshot notice (2026-07-15): this document preserves the original ROCm policy analysis. For current capability evidence, use `CURRENT_STATE_AUDIT.md` and `FEATURE_TRUTH_MATRIX.md`. Since the snapshot, the project added a clean host-only wheel/install lane, canonical example tests, and clearer native/host build separation. No AMD GPU was available for the refresh, so none of those changes establishes native ROCm execution or performance.

## External ROCm Ground Truth

Official sources checked:

- ROCm release history: `https://rocm.docs.amd.com/en/latest/release/versions.html`
- ROCm compatibility matrix: `https://rocm.docs.amd.com/en/latest/compatibility/compatibility-matrix.html`
- ROCm Linux install and system requirements: `https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html`

### Latest Production ROCm At Refresh Time

The official AMD docs inspected on 2026-06-10 showed ROCm `7.2.4` as the production release, with release date `2026-05-29`. The earlier 2026-04-05 snapshot used ROCm `7.2.0`; this refresh updates policy and CI recommendations to `7.2.4`.

### Latest AMD GPU Target At Audit Time

The official AMD docs inspected in this pass showed AMD Instinct `MI355X` / `gfx950` as the newest datacenter GPU target relevant to ROCm planning.

## Current ROCm Integration Depth

Evidence assessment: native source is present, host packaging/contracts are tested, and actual-device ROCm maturity is unverified in this refresh.

What is wired:

- Root `CMakeLists.txt` requires CMake `3.21`; `ROCQUANTUM_BUILD_NATIVE=ON` enables the CMake HIP language and consumes official ROCm config-package targets, while the default host-only packaging path does not discover HIP.
- Native component libraries are built from `rocquantum/src/hipStateVec`, `rocquantum/src/hipTensorNet`, and `rocquantum/src/hipDensityMat`.
- `hipStateVec` optionally looks for RCCL via `rocquantum/src/hipStateVec/CMakeLists.txt`, preferring the official `rccl` CMake target and retaining `rccl::rccl` / library-variable fallbacks for older layouts.
- CI definitions include separate host fast checks and self-hosted ROCm runtime/evidence jobs. Their presence is source evidence; a retained green native artifact is still required for execution proof.

What is missing or weak:

- The default architecture policy covers only Tier 1 datacenter targets (`gfx950`, `gfx942`, `gfx90a`); Radeon/workstation targets require explicit user override.
- No release-grade ROCm/GPU matrix is fully proven across all target families.
- Host-only wheel metadata, installation, imports, and CLI checks are coherent; canonical and legacy runtime/binding names remain split.
- Install/export is improved but still needs downstream consumption tests across ROCm installs.
- CI proves more than the original audit did, but full statevector/density/distributed runtime coverage still depends on ROCm self-hosted runners.

## Repo-Verified Compatibility Story

This section is about what the repo can defend today from code and CI, not what it should target next.

| Dimension | Current Truth |
| --- | --- |
| Primary OS | Linux x86_64 |
| Windows | Development helper scripts exist, but Windows is not a release-grade path |
| Non-experimental ROCm in CI | `6.2.2` |
| Experimental ROCm in CI | `7.2.4` |
| Native CMake floor | CMake `3.21` for HIP language support |
| ROCm CMake package style | Official config packages and imported targets: `hip` / `hip::host`, `roc::rocblas`, `roc::rocsolver`, optional `rccl` |
| GPU architecture policy | `gfx950;gfx942;gfx90a` by default; older ROCm lane uses `gfx942;gfx90a` |
| Runtime GPU proof | Evidence harnesses exist, but no retained green native artifact was reviewed locally |
| Multi-GPU proof | MultiGPUTests and RCCL benchmark hooks exist as source; actual cross-device proof is pending |

## Recommended Compatibility Policy

This is the proposed release policy after the audit, not a claim that the repo already verifies all of it.

### Tier 1 Target

- ROCm: `7.2.4`
- CMake: `3.21` or newer
- GPU architectures: `gfx950`, `gfx942`, `gfx90a`
- OS: Linux x86_64
- Host package: Python `3.9` through `3.13`; native binding support across that range remains a target, not actual-device evidence

Rationale:

- `gfx950` covers the newest AMD target at audit time.
- `gfx942` covers MI300 generation.
- `gfx90a` covers MI210/MI250/MI250X and is a practical floor for serious datacenter ROCm support.

### Tier 2 Best-Effort

- ROCm: `6.4.0`
- GPU architectures: `gfx908`, selected Radeon workstation targets already supported by current ROCm docs such as `gfx1100`, `gfx1101`, and `gfx1030`

Rationale:

- `6.4.0` is a better future minimum than `6.2.2` if the project wants a cleaner support floor while retaining modern ROCm feature coverage.
- `gfx908` can remain best-effort only if it stays build-clean.
- `gfx906` should not be advertised as supported going forward because current ROCm documentation no longer treats it as a practical forward-looking target.

## Current Gaps In ROCm Productization

### Build system

- The compiler stack is not coherently built as part of the root product.
- Downstream install-tree consumption is not yet tested against multiple ROCm package layouts.
- RCCL remains optional; builds without RCCL must still make the missing distributed fast path obvious.

### Python packaging

- A clean host-only PEP 517 wheel, isolated install, installed imports, and CLI help now pass with `ROCQUANTUM_BUILD_NATIVE=OFF`.
- Native packaging remains a separate, unverified path: scikit-build wheels opt in with `ROCQ_BUILD_NATIVE=1`, while direct CMake builds use `ROCQUANTUM_BUILD_NATIVE=ON` on a ROCm host.
- `rocq` is the canonical Python surface, but `rocquantum_bind`, `_rocq_hip_backend`, `rocq_hip`, and legacy `python/rocq` ownership still need consolidation.

### CI and validation

- First-party labeled CTest definitions now cover StateVec, DensityMat, TensorNet contraction/SVD, and an RCCL-required inter-rank multi-GPU smoke, but no retained successful actual-device run was available for this audit.
- There is no GPU-generation matrix proving behavior across more than one architectural family.
- Host wheel build/install/import tests now exist. Native wheel publishing and actual-ROCm install-tree consumption remain unverified.

### Multi-GPU

- RCCL linkage is optional and partial.
- Code contains real distributed scaffolding, but many distributed code paths remain `ROCQ_STATUS_NOT_IMPLEMENTED`.
- The repo must be described as single-node, experimental multi-GPU only until real CI/runtime proof exists.

## Recommended CI Matrix

### Source and packaging lane

- Ubuntu latest
- Python `3.9` through `3.13` for core/Cirq 1.x; tested Qiskit coverage starts at Python 3.10 with `qiskit>=2.4,<3`, and fully contract-tested PennyLane coverage starts at Python 3.11 with `pennylane>=0.45,<0.46`
- clean `python -m build --wheel` producing `py3-none-any`
- inspect/install the wheel and verify imports/CLI outside the source tree
- copy and run all 19 examples outside the repository with `PYTHONPATH` removed

### Native ROCm lane

- ROCm `7.2.4`
- CMake `3.21+`
- `CMAKE_HIP_ARCHITECTURES="gfx950;gfx942;gfx90a"`
- build native libraries and bindings
- run statevector, density-matrix, tensor-network, and expectation tests

### Best-effort ROCm lane

- ROCm `6.4.x`
- `CMAKE_HIP_ARCHITECTURES="gfx942;gfx90a"`
- build plus reduced runtime suite

### Multi-GPU lane

- ROCm `7.2.4`
- at least 2 GPUs
- distributed allocation, local-domain gates, local-domain measurement smoke
- explicit skip/fail for currently unsupported distributed operations

## Recommended Build Flags

For serious Linux ROCm builds:

```bash
cmake -S . -B build-ci -G Ninja \
  -DBUILD_TESTING=ON \
  -DROCQUANTUM_BUILD_BINDINGS=ON \
  -DROCQUANTUM_BUILD_NATIVE=ON \
  -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ \
  -DCMAKE_HIP_ARCHITECTURES="gfx950;gfx942;gfx90a"
```

For best-effort older lane:

```bash
cmake -S . -B build-ci -G Ninja \
  -DBUILD_TESTING=ON \
  -DROCQUANTUM_BUILD_BINDINGS=ON \
  -DROCQUANTUM_BUILD_NATIVE=ON \
  -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ \
  -DCMAKE_HIP_ARCHITECTURES="gfx942;gfx90a"
```

## Minimum Support Recommendation

Separate current truth from forward policy:

- Conservative configured CI baseline: ROCm `6.2.2` (not locally device-verified)
- Recommended future minimum policy: ROCm `6.4.0`
- Recommended minimum release-grade GPU architecture: `gfx90a`

## Release Readiness Checklist

- Preserve the clean host-only package gate; unify remaining runtime/binding identities
- Add explicit HIP architecture matrix to CI and build docs
- Keep CMake at `3.21+` and use ROCm config-package targets (`hip::host`, `roc::rocblas`, `roc::rocsolver`, optional `rccl`)
- Validate the install/export package config from an installed ROCm tree
- Execute the registered StateVec, DensityMat, TensorNet, expectation, and multi-GPU regressions on real ROCm runners and retain the artifacts
- Publish a single Linux-first support statement
- Stop implying Windows release support until a real Windows ROCm path is validated

## Conclusion

The repo has meaningful ROCm-native code, but the ROCm integration story is still a prototype story, not a release story. The fastest path to credibility is not a giant refactor; it is to tighten support boundaries, unify packaging/build surfaces, and prove a smaller but honest ROCm target matrix.
