#!/usr/bin/env bash

set -Eeuo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/run_rocm_hardware_validation.sh [options]

Run fail-closed ROCm hardware validation and preserve a compressed evidence bundle.

Options:
  --profile smoke|full   smoke: C64 runtime + RCCL + compiler/GPU E2E (default)
                         full: smoke plus C128, integrations, install, benchmarks
  --artifact-dir PATH    Evidence directory (default: hardware-validation-artifacts/<UTC>)
  --build-root PATH      Reusable build root (default: build-hardware-validation)
  --jobs N               Parallel build jobs (default: host CPU count)
  --skip-compiler        Skip the combined MLIR/LLVM/HIP build and strict GPU smoke
  --require-multi-gpu    Fail unless at least two ROCm GPUs are visible
  -h, --help             Show this help

Required for compiler validation:
  MLIR_DIR and LLVM_DIR, defaulting to /usr/lib/llvm-22/lib/cmake/{mlir,llvm}.

This script does not create or terminate cloud resources. Copy the evidence bundle
off-host before terminating any ephemeral validation machine.
EOF
}

profile="smoke"
artifact_dir=""
build_root=""
jobs=""
run_compiler=1
require_multi_gpu=0

while (($# > 0)); do
  case "$1" in
    --profile)
      [[ $# -ge 2 ]] || { echo "--profile requires a value" >&2; exit 2; }
      profile="$2"
      shift 2
      ;;
    --artifact-dir)
      [[ $# -ge 2 ]] || { echo "--artifact-dir requires a value" >&2; exit 2; }
      artifact_dir="$2"
      shift 2
      ;;
    --build-root)
      [[ $# -ge 2 ]] || { echo "--build-root requires a value" >&2; exit 2; }
      build_root="$2"
      shift 2
      ;;
    --jobs)
      [[ $# -ge 2 ]] || { echo "--jobs requires a value" >&2; exit 2; }
      jobs="$2"
      shift 2
      ;;
    --skip-compiler)
      run_compiler=0
      shift
      ;;
    --require-multi-gpu)
      require_multi_gpu=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ "${profile}" != "smoke" && "${profile}" != "full" ]]; then
  echo "--profile must be smoke or full" >&2
  exit 2
fi
if [[ -n "${jobs}" && ! "${jobs}" =~ ^[1-9][0-9]*$ ]]; then
  echo "--jobs must be a positive integer" >&2
  exit 2
fi

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
timestamp_utc="$(date -u +'%Y%m%dT%H%M%SZ')"
artifact_dir="${artifact_dir:-${project_root}/hardware-validation-artifacts/${timestamp_utc}}"
build_root="${build_root:-${project_root}/build-hardware-validation}"
jobs="${jobs:-$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 2)}"
rocm_root="${ROCM_PATH:-/opt/rocm}"
cmake_hip_compiler="${CMAKE_HIP_COMPILER:-${rocm_root}/llvm/bin/clang++}"
mlir_dir="${MLIR_DIR:-/usr/lib/llvm-22/lib/cmake/mlir}"
llvm_dir="${LLVM_DIR:-/usr/lib/llvm-22/lib/cmake/llvm}"

mkdir -p "${artifact_dir}" "${build_root}"
artifact_dir="$(cd "${artifact_dir}" && pwd)"
build_root="$(cd "${build_root}" && pwd)"
bundle_path="${artifact_dir}.tar.gz"

finalize() {
  local exit_code=$?
  trap - EXIT
  set +e
  write_summary() {
    {
      echo "schema_version=1"
      echo "profile=${profile}"
      echo "status=$([[ ${exit_code} -eq 0 ]] && echo passed || echo failed)"
      echo "exit_code=${exit_code}"
      echo "timestamp_utc=$(date -u +'%Y-%m-%dT%H:%M:%SZ')"
      echo "artifact_dir=${artifact_dir}"
      echo "bundle_path=${bundle_path}"
    } > "${artifact_dir}/summary.env"
  }
  write_summary
  if ! tar -czf "${bundle_path}" -C "$(dirname "${artifact_dir}")" "$(basename "${artifact_dir}")"; then
    echo "Failed to create the ROCm hardware validation evidence bundle." >&2
    exit_code=1
    write_summary
  else
    echo "ROCm hardware validation evidence: ${bundle_path}"
  fi
  exit "${exit_code}"
}
trap finalize EXIT

require_command() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Required command is unavailable: $1" >&2
    exit 1
  fi
}

run_logged() {
  local name="$1"
  shift
  echo "==> ${name}"
  "$@" 2>&1 | tee "${artifact_dir}/${name}.log"
}

for command_name in awk cmake find git ninja paste python3 rocminfo sort tar tee; do
  require_command "${command_name}"
done

if [[ "${run_compiler}" -eq 1 ]]; then
  [[ -d "${mlir_dir}" ]] || {
    echo "MLIR_DIR does not exist: ${mlir_dir}" >&2
    exit 1
  }
  [[ -d "${llvm_dir}" ]] || {
    echo "LLVM_DIR does not exist: ${llvm_dir}" >&2
    exit 1
  }
fi

cd "${project_root}"

{
  echo "timestamp_utc=$(date -u +'%Y-%m-%dT%H:%M:%SZ')"
  echo "profile=${profile}"
  echo "git_commit=$(git rev-parse HEAD)"
  echo "git_branch=$(git branch --show-current)"
  echo "git_dirty=$([[ -n "$(git status --short)" ]] && echo yes || echo no)"
  echo "rocm_root=${rocm_root}"
  echo "cmake_hip_compiler=${cmake_hip_compiler}"
  echo "mlir_dir=${mlir_dir}"
  echo "llvm_dir=${llvm_dir}"
  echo "jobs=${jobs}"
  uname -a
  cmake --version
  ninja --version
  python3 --version
} | tee "${artifact_dir}/environment.log"

run_logged \
  rocm-runtime-probe \
  bash scripts/probe_rocm_runtime.sh

gpu_count="$(
  awk -F= '/^visible_gpu_count=/ { value=$2 } END { print value }' \
    "${artifact_dir}/rocm-runtime-probe.log"
)"
gpu_architectures="$(
  awk -F= '/^gpu_architectures=/ { value=$2 } END { print value }' \
    "${artifact_dir}/rocm-runtime-probe.log"
)"
if [[ ! "${gpu_count}" =~ ^[1-9][0-9]*$ || -z "${gpu_architectures}" ]]; then
  echo "Unable to recover GPU topology from the ROCm probe." >&2
  exit 1
fi
if [[ "${require_multi_gpu}" -eq 1 && "${gpu_count}" -lt 2 ]]; then
  echo "Multi-GPU evidence was required, but only ${gpu_count} GPU was detected." >&2
  exit 1
fi
{
  echo "visible_gpu_count=${gpu_count}"
  echo "gpu_architectures=${gpu_architectures}"
} | tee "${artifact_dir}/gpu-topology.env"

run_logged async-contract bash scripts/check_async_contract.sh

c64_build="${build_root}/c64"
run_logged c64-configure \
  cmake -S . -B "${c64_build}" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_TESTING=ON \
    -DROCQUANTUM_BUILD_BINDINGS=OFF \
    -DROCQUANTUM_BUILD_NATIVE=ON \
    -DROCQUANTUM_REQUIRE_RCCL=ON \
    -DCMAKE_HIP_COMPILER="${cmake_hip_compiler}" \
    -DCMAKE_HIP_ARCHITECTURES="${gpu_architectures}" \
    -DCMAKE_PREFIX_PATH="${rocm_root}"
run_logged c64-build cmake --build "${c64_build}" --parallel "${jobs}"
run_logged c64-single-gpu-ctest \
  ctest --test-dir "${c64_build}" --output-on-failure --no-tests=error \
    --output-junit "${artifact_dir}/c64-single-gpu.junit.xml" \
    -L native -LE multi-gpu
if [[ "${gpu_count}" -ge 2 ]]; then
  run_logged c64-multi-gpu-ctest \
    ctest --test-dir "${c64_build}" --output-on-failure --no-tests=error \
      --output-junit "${artifact_dir}/c64-multi-gpu.junit.xml" \
      -R MultiGPUTests
else
  printf '%s\n' \
    "status=skipped" \
    "reason=requires at least two visible GPUs; detected ${gpu_count}" \
    | tee "${artifact_dir}/c64-multi-gpu-ctest.log"
fi

compiler_build="${build_root}/compiler-gpu"
python_binding_build="${compiler_build}"
if [[ "${run_compiler}" -eq 1 || "${profile}" == "full" ]]; then
  venv_dir="${build_root}/venv"
  run_logged python-venv python3 -m venv --clear "${venv_dir}"
  venv_python="${venv_dir}/bin/python"
  run_logged python-base-dependencies \
    "${venv_python}" -m pip install --upgrade pip numpy pybind11
  pybind11_cmake_dir="$("${venv_python}" -m pybind11 --cmakedir)"
fi

if [[ "${run_compiler}" -eq 1 ]]; then
  run_logged compiler-gpu-configure \
    cmake -S . -B "${compiler_build}" -G Ninja \
      -DCMAKE_BUILD_TYPE=Release \
      -DBUILD_TESTING=OFF \
      -DROCQUANTUM_BUILD_BINDINGS=ON \
      -DROCQUANTUM_BUILD_NATIVE=ON \
      -DROCQUANTUM_ENABLE_MLIR_COMPILER=ON \
      -DROCQUANTUM_REQUIRE_RCCL=ON \
      -DPython3_EXECUTABLE="${venv_python}" \
      -DCMAKE_HIP_COMPILER="${cmake_hip_compiler}" \
      -DCMAKE_HIP_ARCHITECTURES="${gpu_architectures}" \
      -DCMAKE_PREFIX_PATH="${pybind11_cmake_dir};${rocm_root}" \
      -DMLIR_DIR="${mlir_dir}" \
      -DLLVM_DIR="${llvm_dir}"
  run_logged compiler-gpu-build \
    cmake --build "${compiler_build}" --parallel "${jobs}"
  compiler_pythonpath="${project_root}:${compiler_build}:${compiler_build}/python/rocq"
  run_logged compiler-gpu-smoke \
    env PYTHONPATH="${compiler_pythonpath}" \
      "${venv_python}" scripts/native_compiler_gpu_smoke.py \
        --json-output "${artifact_dir}/compiler-gpu-smoke.json" \
        --markdown-output "${artifact_dir}/compiler-gpu-smoke.md"
elif [[ "${profile}" == "full" ]]; then
  python_binding_build="${build_root}/python-native"
  run_logged python-native-configure \
    cmake -S . -B "${python_binding_build}" -G Ninja \
      -DCMAKE_BUILD_TYPE=Release \
      -DBUILD_TESTING=OFF \
      -DROCQUANTUM_BUILD_BINDINGS=ON \
      -DROCQUANTUM_BUILD_NATIVE=ON \
      -DROCQUANTUM_REQUIRE_RCCL=ON \
      -DPython3_EXECUTABLE="${venv_python}" \
      -DCMAKE_HIP_COMPILER="${cmake_hip_compiler}" \
      -DCMAKE_HIP_ARCHITECTURES="${gpu_architectures}" \
      -DCMAKE_PREFIX_PATH="${pybind11_cmake_dir};${rocm_root}"
  run_logged python-native-build \
    cmake --build "${python_binding_build}" --parallel "${jobs}"
fi

if [[ "${profile}" == "full" ]]; then
  c128_build="${build_root}/c128"
  run_logged c128-configure \
    cmake -S . -B "${c128_build}" -G Ninja \
      -DCMAKE_BUILD_TYPE=Release \
      -DBUILD_TESTING=ON \
      -DROCQUANTUM_BUILD_BINDINGS=OFF \
      -DROCQUANTUM_BUILD_NATIVE=ON \
      -DROCQUANTUM_REQUIRE_RCCL=ON \
      -DROCQ_PRECISION_DOUBLE=ON \
      -DCMAKE_HIP_COMPILER="${cmake_hip_compiler}" \
      -DCMAKE_HIP_ARCHITECTURES="${gpu_architectures}" \
      -DCMAKE_PREFIX_PATH="${rocm_root}"
  run_logged c128-build cmake --build "${c128_build}" --parallel "${jobs}"
  run_logged c128-single-gpu-ctest \
    ctest --test-dir "${c128_build}" --output-on-failure --no-tests=error \
      --output-junit "${artifact_dir}/c128-single-gpu.junit.xml" \
      -L native -LE multi-gpu
  if [[ "${gpu_count}" -ge 2 ]]; then
    run_logged c128-multi-gpu-ctest \
      ctest --test-dir "${c128_build}" --output-on-failure --no-tests=error \
        --output-junit "${artifact_dir}/c128-multi-gpu.junit.xml" \
        -R MultiGPUTests
  else
    printf '%s\n' \
      "status=skipped" \
      "reason=requires at least two visible GPUs; detected ${gpu_count}" \
      | tee "${artifact_dir}/c128-multi-gpu-ctest.log"
  fi

  run_logged framework-dependencies \
    "${venv_python}" -m pip install \
      scipy pytest "qiskit>=2.4,<3" "pennylane>=0.45,<0.46" "cirq-core>=1.0,<2"
  framework_pythonpath="${project_root}:${python_binding_build}:${python_binding_build}/python/rocq:${project_root}/integrations/pennylane-rocq:${project_root}/integrations/qiskit-rocquantum-provider:${project_root}/integrations/cirq-rocm"
  run_logged native-framework-smoke \
    env PYTHONPATH="${framework_pythonpath}" \
      "${venv_python}" scripts/native_framework_smoke.py \
        --json-output "${artifact_dir}/native-framework-smoke.json" \
        --markdown-output "${artifact_dir}/native-framework-smoke.md" \
        --require-native-rocm-evidence

  install_prefix="${build_root}/install-c64"
  consumer_build="${build_root}/install-consumer"
  run_logged install-consumer-smoke \
    env ROCQUANTUM_INSTALL_PREFIX="${install_prefix}" \
      ROCQUANTUM_INSTALL_CONSUMER_BUILD_DIR="${consumer_build}" \
      CMAKE_HIP_COMPILER="${cmake_hip_compiler}" \
      CMAKE_HIP_ARCHITECTURES="${gpu_architectures}" \
      CMAKE_PREFIX_PATH="${rocm_root}" \
      bash scripts/validate_cmake_install_consumer.sh "${c64_build}"

  run_logged release-benchmarks \
    "${venv_python}" benchmarks/run_release_benchmarks.py \
      --build-dir "${c64_build}" \
      --output-dir "${artifact_dir}/benchmarks" \
      --fail-on-error \
      --require-native-performance-evidence \
      --require-all-native-benchmark-evidence \
      --history-path "${artifact_dir}/benchmark-history.json"
fi
