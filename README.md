# SiliconForest Mesa

Personally maintained Mesa integration for Apple Silicon, by
[iconidentify](https://github.com/iconidentify). Default branch:
**`siliconforest/next`**, Mesa **26.3.0-devel**.

This combines DJ's Neo/G17P work, Eryk Wieliczko's current M3 Pro native
OpenGL/Vulkan work, Naeem's M3 Max additions, and selected correctness fixes.
Original commits, author identities, copyrights and licenses are retained.
The upstream Mesa README is [README.rst](README.rst).

## Source profiles

| Hardware / contract | Branch | Status in this integration |
| --- | --- | --- |
| M1/M2 and AGX2 | `siliconforest/next` | Existing Asahi/Honeykrisp paths retained; no physical-device qualification here |
| M3 Pro, T6030 / G15S, legacy firmware interface with native AGX3 shaders | `siliconforest/next` | Pinned Aurora WIP native GL/Vulkan stack; this update needs hardware requalification |
| M3 Max, T6031 / G15C, single die / four clusters | `siliconforest/next` | Naeem's identity, native admission and USC helper-table fixes merged with Eryk's follow-ups |
| Neo / A18 Pro, T8140 / G17P, USC3 / HAL200 | `siliconforest/next` | DJ's current AGX3b stack retained, including its feature restrictions |
| M3 Air, T8122 / G15G, legacy 14.x firmware | `siliconforest/next` | Scott Jones PR71 integrated; enabled without a Mesa environment opt-in for its exact supported contract; revised address layout is not hardware-qualified |
| M3 Air, T8122, HAL200 / macOS 26.6.2 boot stack | `compat/m3-air-hal200` | Older OpenGL-only compatibility source; not part of the native main build |
| Older M3 boot/firmware stacks | `compat/m3-legacy`, `compat/m3-pro-legacy` | Previous personal source profiles preserved with exact parent pins |
| M4 | `experimental/m4` | Eryk's separate development/evidence branch; unqualified here |
| M5 recognition research | `experimental/m5-recognition` | Public DJ recognition/encoder source; hardware admission stays fail-closed. Native private HAL300 work stays local. |
| Earlier M3 Max experiments | `experimental/m3-max-legacy` | Historical source profile; use the combined native branch for new development |

A chip name or shader encoder alone does not establish support. Main deliberately
refuses M3 Ultra, M4 and M5 admission. Native M3 Air is experimental and
requires its exact identity, supported firmware and a compatible kernel. No Mesa
environment opt-in is required.
Compatibility and experimental profiles are separate source trees, not additional
drivers silently enabled in one binary. They have **not** undergone the main
branch's build/test run.

## Kernel and firmware pairing

Native M3 Pro/Max, experimental M3 Air and Neo use the unified Aurora ABI
represented by `include/drm-uapi/asahi_drm.h`. The checked kernel source is
[omacom/linux-aurora at 7b969982](https://github.com/omacom/linux-aurora/tree/7b969982306e1498a8cbe6e5da9bde743acd8fc5),
`include/uapi/drm/asahi_drm.h`. The update uses the high feature-bit assignments
and timestamp-copy command fields from that contract. It rejects the older
low-bit native contract before VM creation; this is a check for the known
collision, not universal ABI negotiation. The older SiliconForest experimental
header checked on October 18 is no longer the matching contract.

Do not substitute headers, force a chip identity or use an old kernel simply
because its version string resembles the checked one. Kernel configuration,
firmware and m1n1 handoffs still need qualification on the actual board.
M3 Air needs no `ASAHI_M3_EXPERIMENTAL` variable; its kernel startup
configuration and package defaults are separate. This source update does not change any installed boot stack.

## Build and check

A source build does not install files, replace system Mesa, enable a kernel
option, or reboot anything. The integration was configured on x86_64 with
LLVM/Clang 22.1.8, a matching SPIR-V LLVM translator, Meson 1.10.2, Mako, PLY,
PyYAML, libdrm, SPIR-V Tools, and the development dependencies for X11/Wayland,
EGL, GBM and Vulkan. See [validation](SILICONFOREST-VALIDATION.md) for the exact
configuration and outcomes.

Quick checks require a C compiler and Python 3. They compile and run production
chip admission, compression, plane-layout and scratch-layout tests:

```sh
python3 tools/siliconforest/check.py
# In a full clone with all source tags:
python3 tools/siliconforest/check.py --check-git
```

With the build dependencies available:

```sh
python3 -m venv ../mesa-venv
../mesa-venv/bin/pip install 'meson==1.10.2' 'Mako==1.3.10' 'ply==3.11' PyYAML pytest
MESON="$PWD/../mesa-venv/bin/meson" JOBS=8 tools/siliconforest/build.sh ../mesa-build
```

The October 18 baseline passed **36 Asahi host suites** and Vulkan regression
probes for four drm-shim identities, with Khronos validation enabled. Results
for the October 24 integration are recorded separately in the validation files. To rerun
those probes after the build:

```sh
python3 tools/siliconforest/check-kernel-abi.py PATH_TO_SILICONFOREST_LINUX
VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation \
  python3 tools/siliconforest/drm-smoke.py ../mesa-build
```

The script builds Asahi and Zink OpenGL, Honeykrisp Vulkan, tree-matched shader
compiler tools and drm-shim, then runs the Asahi host suite. Do not reuse shader
compiler binaries from an older Mesa tree. There are no binary releases or
system-install commands for this integration yet.

## What was reviewed and retained

[The review ledger](SILICONFOREST-REVIEW.md) explains merged changes, duplicate
fixes, ABI findings, held regressions and experimental branches.
[SILICONFOREST-SOURCES.json](SILICONFOREST-SOURCES.json) retains the 209 original public branch,
PR and MR snapshots under `sources/2026-10-18/*`, adds two pinned October 24
inputs, and retains the dated inventory of 37 GitHub PR records and
25 open upstream Asahi MR records. A snapshot is evidence, not approval to ship
its contents. This is a dated public-source investigation; it cannot establish
that no private, deleted, unindexed or newer work exists.

GitHub Actions runs the quick CPU checks on main and pull requests. Full host
compilation and hardware qualification are distinct checks.

<!-- Personal-fork CI documentation-only control; not a production change. -->
