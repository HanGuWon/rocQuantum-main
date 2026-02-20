ARG ROCM_VERSION=6.2.2
FROM rocm/dev-ubuntu-22.04:${ROCM_VERSION}

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      ca-certificates \
      cmake \
      git \
      ninja-build \
      pybind11-dev \
      python3-dev \
      python3-pip \
      ripgrep && \
    rm -rf /var/lib/apt/lists/*

# Keep Python tooling deterministic across lanes.
RUN python3 -m pip install --no-cache-dir --upgrade \
      pip==24.0 \
      numpy==1.26.4 \
      scipy==1.13.1

ENV CMAKE_PREFIX_PATH=/opt/rocm \
    CMAKE_HIP_COMPILER=/opt/rocm/bin/hipcc

WORKDIR /workspace/rocQuantum-main
