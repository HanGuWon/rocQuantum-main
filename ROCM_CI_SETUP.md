# ROCm CI Setup

## Toolchain And Environment Matrix
| Lane | ROCm Toolchain | AMD GPU Targets | Scope |
|---|---|---|---|
| Primary | `rocm/dev-ubuntu-22.04:6.2.2` | `gfx90a` | Minimum supported ROCm lane and release gate baseline. |
| Latest | `rocm/dev-ubuntu-22.04:7.2.0` | `gfx90a` | Forward-compatibility lane. |
| Optional Legacy | `rocm/dev-ubuntu-22.04:{6.2.2,7.2.0}` | `gfx906;gfx908` | Best-effort compatibility lane (non-blocking by default). |

## Reproducible Container Environment
Use `docker/rocm/Dockerfile` to build deterministic ROCm toolchain images:

```bash
docker build --build-arg ROCM_VERSION=6.2.2 -t rocq-dev:rocm-6.2.2 -f docker/rocm/Dockerfile .
docker build --build-arg ROCM_VERSION=7.2.0 -t rocq-dev:rocm-7.2.0 -f docker/rocm/Dockerfile .
```

ROCm runtime shell (GPU-enabled host):

```bash
docker run --rm -it \
  --device=/dev/kfd \
  --device=/dev/dri \
  --group-add video \
  --ipc=host \
  -v "$PWD/rocQuantum-main:/workspace/rocQuantum-main" \
  rocq-dev:rocm-6.2.2 \
  bash
```

## Minimum Configure Commands
From `rocQuantum-main/`:

```bash
cmake -S . -B build-ci -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_TESTING=ON \
  -DCMAKE_HIP_COMPILER=/opt/rocm/bin/hipcc \
  -DAMDGPU_TARGETS=gfx90a
```

Optional legacy lane configure:

```bash
cmake -S . -B build-legacy -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_TESTING=ON \
  -DCMAKE_HIP_COMPILER=/opt/rocm/bin/hipcc \
  -DAMDGPU_TARGETS=gfx906\;gfx908
```

## Full Build/Test Commands
From `rocQuantum-main/`:

```bash
bash scripts/check_async_contract.sh

cmake -S . -B build-ci -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_TESTING=ON \
  -DCMAKE_HIP_COMPILER=/opt/rocm/bin/hipcc \
  -DAMDGPU_TARGETS=gfx90a

cmake --build build-ci --parallel

# Runtime subset currently used by ROCm CI.
ctest --test-dir build-ci --output-on-failure -R HipTensorNetContractionRegression

# Full suite for dedicated ROCm runtime runners.
ctest --test-dir build-ci --output-on-failure
```

## Runner Setup (Self-hosted)

Required baseline on Linux x86_64 runners:
- ROCm stack installed and on `PATH`: `hipcc`, `rocminfo`, `rocm-smi`.
- ROCm development libraries available for CMake discovery: HIP runtime, `rocblas`, `rocsolver`, `hiprand`.
- Build tools: `cmake` (>= 3.18), `ninja`, `gcc/g++`, `make`, `git`.
- Python tools: `python3`, `python3-pip`.
- Device access: `/dev/kfd` available to the runner service user.

Example Ubuntu packages (names may vary by distro/ROCm channel):
- `cmake`, `ninja-build`, `build-essential`, `git`, `python3`, `python3-pip`
- ROCm packages providing `hipcc`, `rocblas`, `rocsolver`, `hiprand`, `rocminfo`, `rocm-smi`

## Runner Hardware Profile

Required profiles for this repo's workflows:
- PR/runtime lane (`rocm-gpu`, `gfx90a`): at least 1 ROCm-capable GPU, `gfx90a` policy target.
- Nightly distributed/perf lane (`rocm-multigpu`): at least 2 ROCm-capable GPUs on one host for `MultiGPUTests`.
- Host baseline: 16+ CPU threads, 64+ GB RAM, 100+ GB free disk for repeated ROCm builds/artifacts.

## ROCm Installation + Validation Checklist

Install ROCm from the official AMD package channel for your chosen lane (`6.2.2` or `7.2.0`), then validate:

```bash
hipcc --version
rocminfo
rocm-smi
ls -l /dev/kfd
```

Expected:
- `hipcc`, `rocminfo`, and `rocm-smi` resolve on `PATH`.
- `/dev/kfd` exists and is readable by the runner service user.

## GitHub Runner Service + Permissions

1. Create runner in GitHub repository settings and apply labels:
   - Runtime lane: `self-hosted,linux,x64,rocm,rocm-gpu,gfx90a`
   - Nightly lane: `self-hosted,linux,x64,rocm,rocm-multigpu`
2. Ensure the runner user can access GPU devices:

```bash
sudo usermod -aG video,render <runner-user>
sudo systemctl restart actions.runner.<org>-<repo>.<runner-name>.service
```

3. Validate under the runner user:

```bash
id
ls -l /dev/kfd /dev/dri
rocminfo | head -n 40
```

## Runner Labels

Label conventions used by workflows:
- Runtime PR/push lane (`.github/workflows/rocm-ci.yml`):
  - `self-hosted`, `linux`, `x64`, `rocm`, `rocm-gpu`, `gfx90a`
- Optional nightly multi-GPU/perf lane (`.github/workflows/rocm-nightly.yml`):
  - `self-hosted`, `linux`, `x64`, `rocm`, `rocm-multigpu`

Guidelines:
- Keep generic capability labels (`rocm`, `rocm-gpu`, `rocm-multigpu`) stable.
- Use architecture label (`gfx90a`) only where job affinity is required.
- Do not reuse `rocm-multigpu` for single-GPU hosts.

## Security And Secrets

Secure-by-default controls in workflows:
- Minimal token scope: `permissions: contents: read`.
- No `pull_request_target`.
- `actions/checkout` uses `persist-credentials: false`.
- Self-hosted runtime job is skipped for forked PRs by default.

Secrets requirements:
- No repository/org secret is required for the current ROCm CI workflows.
- If private package mirrors are added later, use environment-scoped secrets and avoid exposing them to fork-triggered jobs.

## Artifacts

Artifact collection is enabled with `if: always()` so failures still produce logs.

Uploaded contents:
- `fast-checks-*`: pytest log, JUnit XML, summary markdown.
- `rocm-runtime-*`: runner probe log, configure/build/test logs, optional CTest JUnit XML, summary markdown.
- `rocm-nightly-*`: runner probe log, configure/build/test/perf logs, optional CTest JUnit XML, summary markdown.

Retention:
- Default retention set to `14` days in workflows; adjust based on storage budget.
