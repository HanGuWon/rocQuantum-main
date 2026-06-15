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
require_command rocm-smi

if [[ ! -e /dev/kfd ]]; then
  missing+=("/dev/kfd")
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
echo "rocminfo_path=$(command -v rocminfo)"
rocminfo
echo "rocm_smi_path=$(command -v rocm-smi)"
rocm-smi
echo "kfd_device=/dev/kfd"
ls -l /dev/kfd
