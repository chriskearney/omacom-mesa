# SiliconForest Mesa validation

## October 24 integration candidate

Tested source: **`aefe5c2dd1525e37e0d329fdf39595dc25abbbcd`**.
Later validation-only documentation does not change the tested driver subtrees.

| Check | Result |
| --- | --- |
| Asahi/Zink OpenGL, Honeykrisp Vulkan, GLX/EGL/GBM, matched shader tools | Built on x86_64 with GCC 16.2.1, LLVM 22.1.8 and Meson 1.10.2 |
| Complete registered Asahi host suite | **91 passed, 0 failed**; existing optional FMA oracle subtest remains skipped |
| Standalone CPU programs and provenance | **5 passed**, 3,359 compression checks, 211 exact public tags and six preserved compatibility refs |
| Vulkan device/pipeline/placed-map/XFB probes | **7 passed**: G13G, G15S, G15C, G17P and G15G with its obsolete opt-in unset, `0` and `1` |
| Vulkan refusal controls | **1 passed**: held G16G with the obsolete Air opt-in set |
| Khronos validation | **0 errors**, **0 GPU submissions** across those eight probes |
| Unified Aurora kernel contract | **18 wire facts match** `7b969982306e1498a8cbe6e5da9bde743acd8fc5`; old SiliconForest native header rejected with ten differences |
| Deliberate negative mutations | Earlier removed Air address-cap and float-as-integer image-write mutations compile and are caught; unchanged code, original receipts retained |
| Syntax and Git whitespace | Passed |

The build uses the configuration below with `--buildtype=debugoptimized` and
`-Db_ndebug=false`. It uses private shader-tool dependencies and performs no
system installation. Build warnings are retained in the log; this is not a
warning-free build claim.

G15G is now admitted without a Mesa environment opt-in for its exact supported
identity and firmware/kernel contract. Kernel and package switch-on is separate.
The G15G drm-shim identity is synthetic. Its baseline features match the Air
runtime's VM_STATUS/SCHEDULED_QUEUES contract; it refuses binds into the four
reserved ranges. It exercises host allocation and pipeline setup without
executing a shader. The revised Air layout, physical M1/M2/M3 regressions,
AArch64 package, full CTS, desktop stability and any FPS change remain untested.
No kernel, firmware, m1n1, system package or boot setting was changed.

The host run exposed stale expectations for G15G wide barriers and held M4
admission, plus a compiler fixture comparing unrelated chip schedules. The
corrected compiler fixture retains all type/format cases, compares the single
image-write packet, and checks its data-type bit independently; a deliberate
wrong-type compiler mutation fails it. No production compiler change was made.

Exact tree IDs, pinned sources, evidence hashes and the earlier baseline are in
[SILICONFOREST-VALIDATION.json](SILICONFOREST-VALIDATION.json).

## October 18 baseline

Validated source: **`3cca5b305cda31558540dbba86f0074568a4cb27`**.
Later documentation and provenance-helper commits do not change driver source.
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
