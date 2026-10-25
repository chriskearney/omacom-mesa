#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
set -euo pipefail
export PATH="$PWD/.ci-venv/bin:/usr/lib/llvm-22/bin:$PATH"
export CC='ccache gcc' CXX='ccache g++'
export CCACHE_BASEDIR="$PWD" CCACHE_COMPILERCHECK=content
export CCACHE_MAXSIZE=400M
if [[ ${GITHUB_EVENT_NAME:-local} != pull_request ]]; then
  export CCACHE_DISABLE=true
else
  unset CCACHE_DISABLE
fi
# Keep hosted 16 GiB runners below OOM even on large compiler translation units.
ci_jobs=${CI_JOBS:-4}
ci_meson=(--buildtype=debugoptimized -Db_ndebug=false --wrap-mode=nodownload
  --prefix=/usr --libdir=lib -Dplatforms=x11,wayland
  -Dgallium-drivers=asahi,zink,softpipe -Dvulkan-drivers=asahi
  -Degl=enabled -Dgbm=enabled -Dglx=dri -Dgles1=enabled -Dgles2=enabled
  -Dglvnd=enabled -Dvideo-codecs= -Dgallium-va=disabled -Dvulkan-layers=
  -Dvalgrind=disabled -Dlibunwind=disabled -Dtools=asahi,drm-shim)
meson setup .ci-tools "${ci_meson[@]}" -Dllvm=enabled -Dbuild-tests=false
ninja -C .ci-tools -j "$ci_jobs" \
  src/compiler/clc/mesa_clc src/compiler/spirv/vtn_bindgen2 src/asahi/clc/asahi_clc
printf '%s\n' '[binaries]' \
  "mesa_clc = '$PWD/.ci-tools/src/compiler/clc/mesa_clc'" \
  "vtn_bindgen2 = '$PWD/.ci-tools/src/compiler/spirv/vtn_bindgen2'" \
  "asahi_clc = '$PWD/.ci-tools/src/asahi/clc/asahi_clc'" > .ci-tools.ini
meson setup .ci-build "${ci_meson[@]}" --native-file .ci-tools.ini \
  -Dllvm=disabled -Dspirv-tools=disabled -Dmesa-clc=system \
  -Dprecomp-compiler=system -Dbuild-tests=true
meson compile -C .ci-build -j "$ci_jobs"
python3 .github/mesa-ci/build-tests.py .ci-build
# Run ALL configured suites: NIR, SPIR-V and Vulkan support tests as well as Asahi.
meson test -C .ci-build --no-rebuild --print-errorlogs --num-processes "$ci_jobs"
VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation \
  python3 tools/siliconforest/drm-smoke.py .ci-build
python3 .github/mesa-ci/pipelines.py
DESTDIR="$PWD/.ci-stage" meson install -C .ci-build --no-rebuild
python3 .github/mesa-ci/manifest.py
ccache --show-stats
