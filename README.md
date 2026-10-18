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
| M3 Pro, T6030 / G15S, legacy firmware interface with native AGX3 shaders | `siliconforest/next` | Eryk's latest native GL/Vulkan stack plus all nine follow-ups missing from Naeem's branch |
| M3 Max, T6031 / G15C, single die / four clusters | `siliconforest/next` | Naeem's identity, native admission and USC helper-table fixes merged with Eryk's follow-ups |
| Neo / A18 Pro, T8140 / G17P, USC3 / HAL200 | `siliconforest/next` | DJ's current AGX3b stack retained, including its feature restrictions |
| M3 Air, T8122, HAL200 / macOS 26.6.2 boot stack | `compat/m3-air-hal200` | Older OpenGL-only compatibility source; not part of the native main build |
| Older M3 boot/firmware stacks | `compat/m3-legacy`, `compat/m3-pro-legacy` | Previous personal source profiles preserved with exact parent pins |
| M4 | `experimental/m4` | Eryk's separate development/evidence branch; unqualified here |
| M5 recognition research | `experimental/m5-recognition` | Public DJ recognition/encoder source; hardware admission stays fail-closed. Native private HAL300 work stays local. |
| Earlier M3 Max experiments | `experimental/m3-max-legacy` | Historical source profile; use the combined native branch for new development |

A chip name or shader encoder alone does not establish support. Main deliberately
refuses M3 Air/Ultra, M4 and M5 identities that lack its complete native path.
Compatibility and experimental profiles are separate source trees, not additional
drivers silently enabled in one binary. They have **not** undergone the main
branch's build/test run.

## Kernel and firmware pairing

Native M3 Pro/Max and Neo require the ABI represented by
`include/drm-uapi/asahi_drm.h`. In
[SiliconForest Linux](https://github.com/iconidentify/siliconforest-linux), the
matched header is `include/uapi/drm/asahi_m3_drm.h` in the experimental
`CONFIG_SILICONFOREST_M3` stack, imported at
`6a421d4e76ff282b414138182fd28597ebf20a18`.
That kernel option remains off by default. Matching firmware and m1n1 handoffs
must also be qualified on the target board; the current personal m1n1 pin alone
is not proof of those handoffs.

The maintained legacy kernel has different feature bit assignments and command
payload layouts despite using the same ioctl family. Main now rejects its known
incompatible native contract before VM creation. Do not substitute headers or
force a chip identity to bypass that check. This is a check for the **known**
legacy/native collision, not a universal ABI negotiation protocol.

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

The script builds Asahi and Zink OpenGL, Honeykrisp Vulkan, tree-matched shader
compiler tools and drm-shim, then runs the Asahi host suite. Do not reuse shader
compiler binaries from an older Mesa tree. There are no binary releases or
system-install commands for this integration yet.

## What was reviewed and retained

[The review ledger](SILICONFOREST-REVIEW.md) explains merged changes, duplicate
fixes, ABI findings, held regressions and experimental branches.
[SILICONFOREST-SOURCES.json](SILICONFOREST-SOURCES.json) pins 209 public branch,
PR and MR snapshots under `sources/2026-10-18/*`, with 37 GitHub PR records and
25 open upstream Asahi MR records. A snapshot is evidence, not approval to ship
its contents. This is a dated public-source investigation; it cannot establish
that no private, deleted, unindexed or newer work exists.

GitHub Actions runs the quick CPU checks on main and pull requests. Full host
compilation and hardware qualification are distinct checks. The existing
archived `iconidentify/aurora-linux` repository remains untouched.
