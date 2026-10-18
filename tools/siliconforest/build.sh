#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
set -euo pipefail
sf_mesa_source=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
sf_mesa_build=${1:-"$sf_mesa_source/../siliconforested-mesa-build"}
sf_mesa_meson=${MESON:-meson}
"$sf_mesa_meson" setup "$sf_mesa_build" "$sf_mesa_source" \
  -Dgallium-drivers=asahi,zink,softpipe -Dvulkan-drivers=asahi \
  -Dplatforms=x11,wayland -Dglx=dri -Degl=enabled -Dgbm=enabled \
  -Dglvnd=disabled -Dllvm=enabled -Dbuild-tests=true \
  -Dtools=asahi,drm-shim -Dmesa-clc=enabled -Dprecomp-compiler=enabled \
  -Dvideo-codecs= -Dgallium-va=disabled
python3 "$sf_mesa_source/tools/siliconforest/build-targets.py" "$sf_mesa_build"
"$sf_mesa_meson" test -C "$sf_mesa_build" --no-rebuild --suite asahi --print-errorlogs
