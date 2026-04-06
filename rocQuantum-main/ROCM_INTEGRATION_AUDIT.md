# ROCm Integration Audit

Audit date: 2026-04-05

## External ROCm Ground Truth

Official sources checked:

- ROCm release history: `https://rocm.docs.amd.com/en/latest/about/release-history.html`
- ROCm compatibility matrix: `https://rocm.docs.amd.com/en/latest/compatibility/compatibility-matrix.html`
- ROCm Linux install and system requirements: `https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html`

### Latest Stable ROCm At Audit Time

The official AMD docs inspected in this pass showed ROCm `7.2.0` as the latest stable release, with release date `2026-01-21`.

I did not find official AMD evidence for ROCm `7.2.1` on 2026-04-05, so the earlier planning assumption of `7.2.1` was downgraded. All compatibility recommendations in this audit therefore use `7.2.0` as the latest stable target.

### Latest AMD GPU Target At Audit Time

The official AMD docs inspected in this pass showed AMD Instinct `MI355X` / `gfx950` as the newest datacenter GPU target relevant to ROCm planning.

## Current ROCm Integration Depth

Maturity assessment: `PARTIAL`, early productization stage

What is wired:

- Root `CMakeLists.txt` requires `HIP`, `hiprand`, `rocblas`, and `rocsolver`.
- Native component libraries are built from `rocquantum/src/hipStateVec`, `rocquantum/src/hipTensorNet`, and `rocquantum/src/hipDensityMat`.
- `hipStateVec` optionally looks for RCCL via `rocquantum/src/hipStateVec/CMakeLists.txt`.
- CI builds inside ROCm containers and can execute one tensor-network GPU regression if `/dev/kfd` is present.

What is missing or weak:

- No repository-wide `CMAKE_HIP_ARCHITECTURES` policy existed before this audit pass.
- No release-grade ROCm/GPU matrix is documented in-repo.
- Python packaging is split across inconsistent module names and version metadata.
- The `_rocq_hip_backend` binding path exists on disk but is not built by the root CMake flow.
- Install/export is incomplete for downstream consumption.
- CI proves only a narrow subset of runtime behavior.

## Repo-Verified Compatibility Story

This section is about what the repo can defend today from code and CI, not what it should target next.

| Dimension | Current Truth |
| --- | --- |
| Primary OS | Linux x86_64 |
| Windows | Development helper scripts exist, but Windows is not a release-grade path |
| Non-experimental ROCm in CI | `6.2.2` |
| Experimental ROCm in CI | `7.2.0` |
| GPU architecture policy | Not explicitly encoded in the repo before this audit |
| Runtime GPU proof | Only `HipTensorNetContractionRegression` in CI |
| Multi-GPU proof | No release-grade multi-GPU CI proof |

## Recommended Compatibility Policy

This is the proposed release policy after the audit, not a claim that the repo already verifies all of it.

### Tier 1 Target

- ROCm: `7.2.0`
- GPU architectures: `gfx950`, `gfx942`, `gfx90a`
- OS: Linux x86_64
- Python: `3.10`, `3.11`, `3.12`

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

- Root CMake does not encode a default HIP architecture list.
- The compiler stack is not coherently built as part of the root product.
- The install tree does not provide a complete package config and version story.
- Exported targets still publish source-tree include paths instead of install interfaces.

### Python packaging

- `pyproject.toml` and `setup.py` disagree on package identity and version.
- `setup.py` assumes library outputs under `build/`, while CI builds under `build-ci/`.
- One active binding name is `rocquantum_bind`; another is `_rocq_hip_backend`; the repo does not present one supported answer.
- `rocq_hip` is imported by top-level `rocq/backends.py`, but no root build hook was found for it.

### CI and validation

- There is no first-party ROCm runtime suite covering statevector, density matrix, expectations, and multi-GPU smoke.
- There is no GPU-generation matrix proving behavior across more than one architectural family.
- There is no wheel build, package publish, or install-tree consumption test.

### Multi-GPU

- RCCL linkage is optional and partial.
- Code contains real distributed scaffolding, but many distributed code paths remain `ROCQ_STATUS_NOT_IMPLEMENTED`.
- The repo must be described as single-node, experimental multi-GPU only until real CI/runtime proof exists.

## Recommended CI Matrix

### Source and packaging lane

- Ubuntu latest
- Python `3.10`, `3.11`, `3.12`
- `pip install -e .`
- import smoke
- source contract tests

### Native ROCm lane

- ROCm `7.2.0`
- `CMAKE_HIP_ARCHITECTURES="gfx950;gfx942;gfx90a"`
- build native libraries and bindings
- run statevector, density-matrix, tensor-network, and expectation tests

### Best-effort ROCm lane

- ROCm `6.4.x`
- `CMAKE_HIP_ARCHITECTURES="gfx942;gfx90a"`
- build plus reduced runtime suite

### Multi-GPU lane

- ROCm `7.2.0`
- at least 2 GPUs
- distributed allocation, local-domain gates, local-domain measurement smoke
- explicit skip/fail for currently unsupported distributed operations

## Recommended Build Flags

For serious Linux ROCm builds:

```bash
cmake -S . -B build-ci -G Ninja \
  -DBUILD_TESTING=ON \
  -DROCQUANTUM_BUILD_BINDINGS=ON \
  -DCMAKE_HIP_COMPILER=/opt/rocm/bin/hipcc \
  -DCMAKE_HIP_ARCHITECTURES="gfx950;gfx942;gfx90a"
```

For best-effort older lane:

```bash
cmake -S . -B build-ci -G Ninja \
  -DBUILD_TESTING=ON \
  -DROCQUANTUM_BUILD_BINDINGS=ON \
  -DCMAKE_HIP_COMPILER=/opt/rocm/bin/hipcc \
  -DCMAKE_HIP_ARCHITECTURES="gfx942;gfx90a"
```

## Minimum Support Recommendation

Separate current truth from forward policy:

- Current repo-verified minimum from CI: ROCm `6.2.2`
- Recommended future minimum policy: ROCm `6.4.0`
- Recommended minimum release-grade GPU architecture: `gfx90a`

## Release Readiness Checklist

- Unify Python package identity, versioning, and binding name
- Add explicit HIP architecture matrix to CI and build docs
- Add complete install/export package config
- Add runtime tests for statevector, density matrix, expectations, and multi-GPU smoke
- Publish a single Linux-first support statement
- Stop implying Windows release support until a real Windows ROCm path is validated

## Conclusion

The repo has meaningful ROCm-native code, but the ROCm integration story is still a prototype story, not a release story. The fastest path to credibility is not a giant refactor; it is to tighten support boundaries, unify packaging/build surfaces, and prove a smaller but honest ROCm target matrix.
