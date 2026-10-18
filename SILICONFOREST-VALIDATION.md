# SiliconForest Mesa validation — 2026-10-18

Validated source: **`3cca5b305cda31558540dbba86f0074568a4cb27`**.
The following documentation/evidence commit does not change driver source.
Machine-readable results and exact driver subtree IDs are in
[SILICONFOREST-VALIDATION.json](SILICONFOREST-VALIDATION.json).

| Check | Result |
| --- | --- |
| Asahi/Zink shared Gallium driver, Honeykrisp shared Vulkan driver, GLX/EGL/GBM | Built successfully on x86_64 Linux |
| Tree-matched shader compiler tools and Asahi drm-shim | Built successfully |
| Complete registered Asahi host suite | **36 passed, 0 failed** |
| Quick production CPU test programs | **4 passed**, including **3,035** compression-policy checks |
| Public provenance / compatibility profile validation | **209** exact tags and **6** public source profiles checked |
| Native experimental kernel wire ABI | **18** selected ioctl/layout/feature facts matched |
| Maintained legacy kernel negative control | Rejected with **10** differing facts |
| Host Vulkan device/pipeline/mapping/XFB probes | **4 passed**: G13G, G15S, G15C and G17P drm-shim identities |
| Same Vulkan probes with Khronos validation enabled | **4 passed**, **0 validation errors**, **0 GPU submissions** |
| Build helper shell syntax / ShellCheck, Python syntax, Git whitespace | Passed |

The host Vulkan probe primes the BO cache with an oversized allocation, maps a
smaller allocation at a reserved address, checks the following guard page,
reserves/remaps it, and frees it while mapped. It also records NULL counter
arrays inside rendering with a bound XFB-capable pipeline and feedback buffer.
It creates and compiles a graphics pipeline but never submits a command buffer.
The embedded shader header is generated from the adjacent GLSL source.

The build used GCC 16.2.1, LLVM/Clang 22.1.8, Meson 1.10.2, Mako 1.3.10 and
PLY 3.11. Shader compiler dependencies were provided from a private build prefix,
without installing Mesa or changing system packages. Configuration:

```sh
meson setup BUILD SOURCE \
  -Dgallium-drivers=asahi,zink,softpipe -Dvulkan-drivers=asahi \
  -Dplatforms=x11,wayland -Dglx=dri -Degl=enabled -Dgbm=enabled \
  -Dglvnd=disabled -Dllvm=enabled -Dbuild-tests=true \
  -Dtools=asahi,drm-shim -Dmesa-clc=enabled -Dprecomp-compiler=enabled \
  -Dvideo-codecs= -Dgallium-va=disabled
python3 tools/siliconforest/build-targets.py BUILD
meson test -C BUILD --no-rebuild --suite asahi --print-errorlogs
python3 tools/siliconforest/check.py --check-git
python3 tools/siliconforest/check-kernel-abi.py PATH_TO_SILICONFOREST_LINUX
VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation \
  python3 tools/siliconforest/drm-smoke.py BUILD
```

`build-targets.py` includes all registered Asahi test dependencies, including
executables launched through the Python drm-shim wrapper. It does not build or
run unrelated Mesa-wide tests. The shader compiler tools are built from this
source tree, rather than reused from an older Mesa build.

The stale 88-byte compute assertion was corrected against the actual 136-byte
kernel payload, retaining explicit field-offset checks. The ray-traversal
fixture now reads output lengths after encoding, eliminating an unsequenced C
read/write; all **611** checks and their expected instruction bytes are retained.
The generic drm-shim mmap wrapper now forwards the requested address, enabling
real `MAP_FIXED` mapping coverage instead of accidentally requesting address zero.

No physical Apple GPU, full OpenGL/Vulkan CTS, compositor, games, firmware
recovery, AArch64 deployment or virtio execution was qualified. The six other
source profiles were preserved, not built or tested by this run. No system Mesa
installation, kernel-option change, m1n1 replacement or reboot was performed.
Private-source work remains local as instructed.
