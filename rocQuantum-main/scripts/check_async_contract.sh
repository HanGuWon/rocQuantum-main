#!/usr/bin/env bash

set -euo pipefail

statevec_file="rocquantum/src/hipStateVec/hipStateVec.cpp"
tensornet_files=(
  "rocquantum/src/hipTensorNet/rocTensorUtil.cpp"
  "rocquantum/src/hipTensorNet/hipTensorNet.cpp"
)

statevec_violations="$(rg -n "hipStreamSynchronize\(" "${statevec_file}" | rg -v "ROCQ_ASYNC_ALLOWED_SYNC" || true)"
if [[ -n "${statevec_violations}" ]]; then
  echo "Found disallowed hipStreamSynchronize calls in ${statevec_file}:"
  echo "${statevec_violations}"
  exit 1
fi

distributed_sync_violations="$(rg -n "return\\s+sync_distributed_streams\\(" "${statevec_file}" | rg -v "ROCQ_ASYNC_ALLOWED_SYNC" || true)"
if [[ -n "${distributed_sync_violations}" ]]; then
  echo "Found disallowed sync_distributed_streams returns in ${statevec_file}:"
  echo "${distributed_sync_violations}"
  exit 1
fi

for file in "${tensornet_files[@]}"; do
  sync_calls="$(rg -n "hipStreamSynchronize\(" "${file}" || true)"
  if [[ -n "${sync_calls}" ]]; then
    echo "Found disallowed hipStreamSynchronize calls in ${file}:"
    echo "${sync_calls}"
    exit 1
  fi
done

echo "Async contract check passed."

