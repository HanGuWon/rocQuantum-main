#!/usr/bin/env bash

set -euo pipefail

missing=()

require_command() {
  local name="$1"
  if ! command -v "${name}" >/dev/null 2>&1; then
    missing+=("${name}")
  fi
}

require_command hipcc
require_command rocminfo

if command -v amd-smi >/dev/null 2>&1; then
  smi_command="amd-smi"
  smi_args=(list)
elif command -v rocm-smi >/dev/null 2>&1; then
  smi_command="rocm-smi"
  smi_args=()
else
  smi_command=""
  smi_args=()
  missing+=("amd-smi or rocm-smi")
fi

rocm_root="${ROCM_PATH:-/opt/rocm}"
cmake_hip_compiler="${CMAKE_HIP_COMPILER:-${rocm_root}/llvm/bin/clang++}"
if [[ ! -x "${cmake_hip_compiler}" ]]; then
  missing+=("CMake HIP compiler (${cmake_hip_compiler})")
fi

if [[ ! -e /dev/kfd ]]; then
  missing+=("/dev/kfd")
fi
if [[ ! -d /dev/dri ]]; then
  missing+=("/dev/dri")
fi

rccl_config=""
if [[ -d "${rocm_root}" ]]; then
  rccl_config="$(
    find -L "${rocm_root}" -maxdepth 5 -type f \
      \( -iname "rccl-config.cmake" -o -iname "rcclConfig.cmake" \) \
      -print -quit 2>/dev/null || true
  )"
fi

if (( ${#missing[@]} > 0 )); then
  echo "ROCm runtime prerequisites are missing:"
  for item in "${missing[@]}"; do
    echo "- ${item}"
  done
  exit 1
fi

echo "ROCm runtime prerequisites detected."
echo "hipcc_path=$(command -v hipcc)"
hipcc --version
echo "cmake_hip_compiler=${cmake_hip_compiler}"
"${cmake_hip_compiler}" --version
echo "rocminfo_path=$(command -v rocminfo)"
rocminfo_output="$(mktemp)"
trap 'rm -f "${rocminfo_output}"' EXIT
rocminfo | tee "${rocminfo_output}"
gpu_count="$(awk '/^[[:space:]]*Name:[[:space:]]*gfx[0-9A-Za-z]+/ { count++ } END { print count + 0 }' "${rocminfo_output}")"
gpu_architectures="$(
  awk '/^[[:space:]]*Name:[[:space:]]*gfx[0-9A-Za-z]+/ { print $2 }' \
    "${rocminfo_output}" \
    | sort -u \
    | paste -sd ';' -
)"
if [[ "${gpu_count}" -lt 1 || -z "${gpu_architectures}" ]]; then
  echo "No ROCm GPU agents were detected in rocminfo." >&2
  exit 1
fi
echo "visible_gpu_count=${gpu_count}"
echo "gpu_architectures=${gpu_architectures}"
echo "smi_path=$(command -v "${smi_command}")"
"${smi_command}" "${smi_args[@]}"
if [[ -n "${rccl_config}" ]]; then
  echo "rccl_cmake_config=${rccl_config}"
else
  echo "rccl_cmake_config=unavailable"
fi
echo "kfd_device=/dev/kfd"
ls -l /dev/kfd
echo "dri_devices=/dev/dri"
ls -l /dev/dri
