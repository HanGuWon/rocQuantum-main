#!/usr/bin/env bash
set -euo pipefail

SOURCE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${1:-${SOURCE_DIR}/build-ci}"
INSTALL_PREFIX="${ROCQUANTUM_INSTALL_PREFIX:-${SOURCE_DIR}/install-rocquantum-smoke}"
CONSUMER_BUILD_DIR="${ROCQUANTUM_INSTALL_CONSUMER_BUILD_DIR:-${SOURCE_DIR}/build-rocquantum-install-consumer}"
CONSUMER_SOURCE_DIR="${SOURCE_DIR}/cmake/install_consumer_smoke"

cmake --install "${BUILD_DIR}" --prefix "${INSTALL_PREFIX}"

cmake_args=(
    "-DCMAKE_PREFIX_PATH=${INSTALL_PREFIX}"
)

if [[ -n "${CMAKE_HIP_COMPILER:-}" ]]; then
    cmake_args+=("-DCMAKE_HIP_COMPILER=${CMAKE_HIP_COMPILER}")
fi

if [[ -n "${CMAKE_HIP_ARCHITECTURES:-}" ]]; then
    cmake_args+=("-DCMAKE_HIP_ARCHITECTURES=${CMAKE_HIP_ARCHITECTURES}")
fi

cmake -S "${CONSUMER_SOURCE_DIR}" -B "${CONSUMER_BUILD_DIR}" -G Ninja "${cmake_args[@]}"
cmake --build "${CONSUMER_BUILD_DIR}" --parallel
