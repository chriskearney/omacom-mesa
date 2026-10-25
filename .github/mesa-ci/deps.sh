#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
set -euo pipefail
# Ubuntu 26.04 supplies a matching LLVM/Clang/translator 22 toolchain on both CPUs.
sudo apt-get update
sudo apt-get install --no-install-recommends -y \
  build-essential ccache cmake ninja-build pkg-config python3-venv \
  python3-mako python3-yaml python3-ply python3-packaging \
  clang-22 llvm-22-dev libclang-22-dev libclang-cpp22-dev \
  libllvmspirvlib-22-dev libclc-22-dev spirv-tools glslang-tools \
  libdrm-dev libexpat1-dev libglvnd-dev libvulkan-dev \
  libx11-dev libx11-xcb-dev libxcb-dri2-0-dev libxcb-dri3-dev \
  libxcb-present-dev libxcb-randr0-dev libxcb-shm0-dev libxcb-sync-dev \
  libxcb-xfixes0-dev libxcb-glx0-dev libxdamage-dev libxext-dev \
  libxfixes-dev libxrandr-dev libxshmfence-dev libxxf86vm-dev \
  libwayland-dev wayland-protocols libzstd-dev zlib1g-dev \
  libdisplay-info-dev libunwind-dev vulkan-validationlayers
python3 -m venv --system-site-packages .ci-venv
.ci-venv/bin/pip install --require-hashes -r .github/mesa-ci/requirements.txt
