# Offline AGX compiler failures

Set `AGX_COMPILER_CAPTURE_DIR` to an existing private writable directory during a
bounded experiment. On a returned compiler failure, the compiler saves its
**entry-point** NIR and shader key in `input.bin`, final backend IR when available,
and a versioned manifest. Successful compilations publish nothing by default. Set
`AGX_COMPILER_CAPTURE_SUCCESS=fragment` to retain successful fragment inputs;
optionally set `AGX_COMPILER_CAPTURE_BINARY_HASH=12345678,abcdef01` to select
executables by the eight-digit lowercase hash reported by HK_M4_SHADER_PROFILE.
To retain the same source before and after a transform, use
`AGX_COMPILER_CAPTURE_SOURCE_HASH` with the 64-digit source BLAKE3 printed by
NIR shader bisection. Unlike the executable hash, this survives code-generation
changes; both filters may be combined. This only filters successful captures.
This filters diagnostic publication only; failures remain eligible and the
32-bundle quota still applies. Normal rendering leaves these options unset. Capturing does not change rejection behavior.

Build and replay on the development host (no Vulkan device needed):

```
ninja -C build/mesa-m4-asahi-host src/asahi/compiler/agx_shader_replay
build/mesa-m4-asahi-host/src/asahi/compiler/agx_shader_replay BUNDLE/input.bin
```

The workspace `tools/compiler-replay.py` wrapper saves compiler stdout/stderr,
input and executable SHA256 hashes, Git revision, dirty patch and hashes of
untracked source files. Preserve the capture build's source patch and installed
library checksum alongside the bundle; the serialized input alone is not a
build identity. Do not copy complete process environments or credentials.

The input uses Mesa's internal serialization and a pointer-free shader key. It
requires the same serialization ABI hash (derived from NIR/key/options sources),
standard Asahi NIR options, and compatible C layout/endian ABI. ARM64 Linux to
x86-64 Linux replay was verified. This is a trusted development format, not an
untrusted-file parser or promise of compatibility across Mesa updates. The tool
refuses mismatched ABI hashes/sizes. Capture declines nonstandard NIR options
rather than persisting callback addresses. Compiler debug overrides must be
recorded by the enclosing experiment; replay does not infer them.

Publication uses a directory lock, a private temporary directory and atomic
rename. The input BLAKE3 identifies/deduplicates the full serialized input, not a
normalized failure family. The directory admits at most 32 bundles, each with at
most 8 MiB of input and 8 MiB of published backend IR. Interrupted temporary
directories count toward that quota. Writing failure must not change compilation
success/failure. This captures returned failures, not an assertion/signal inside
the compiler or a later GPU execution fault.

## Firefox saturated FMA, 2026-09-12

WebRender startup input `8094828236c4e7b3732c5f30af8a12d3936b159e4aeac60a73307a009b0f484f`
(1788 bytes) rejected `ffma.sat -r0, r1, #0.5`. The saturated native form
requires three F32 registers. Materializing minifloats and widening F16 inputs
before register allocation preserves the fused operation, modifiers and final
saturation without selecting an ungrounded immediate encoding. The original
input now compiles to 516 bytes; all 384 existing modifier encoding tests pass.
Workspace evidence: `build/firefox-fma-before`, `build/firefox-fma-after`, and
`_provenance/experiments/20260912T025755127471Z-firefox-startup-shader-capture/`.
This is offline compiler evidence; Firefox hardware validation follows separately.

## First real regression, 2026-09-12

J713 SDDM/Hyprland fragment input:
`dea256bf3f8da51bb336633d29fb6640ab33d33ff16f4a9384ea901b7b8144d1` (BLAKE3).
The workspace retains the original bundle under
`_provenance/compiler-capture-20260912/agx-failures/`, plus before/after replay
logs. The 9356-byte input first rejected a signed half-width widening IADD with
shift 4. Its old diagnostic misleadingly printed the generic immediate as zero.
After legalization, the same input exposed F32-to-I16 conversion and mixed-width
FCMPSEL with a selected immediate. All three were fixed and the full input now
compiles to 6026 bytes offline. Focused generated tests cover shift/order and
integer narrowing. This establishes compiler acceptance, not rendering
conformance: the subsequent normal desktop/Foot test hit a fragment-stage timeout.

The captured GL stream subsequently isolated that timeout to a render pass
mixing textured draws with the border's two coverage updates. The early
discard-only update emitted a class-1 release without acquiring class 1; the
later discard/test update then acquired and released it again. Releasing only
after the testing update's matching wait fixes the full captured frame with
normal batching. All 4,259,840 pixels match llvmpipe within one RGB level, with
identical alpha. A real SDDM password login then passed with Quickshell, wallpaper
and a bordered Foot window, verified using native scanout screenshots.

Evidence in the development workspace:
`_provenance/experiments/20260912T005331460114Z-hypr-gl-final-release-ready/`
and `_provenance/normal-login-20260912/m4-normal-final-coverage-20260912/`.
Both used the 60-second watchdog and six-channel temperature guard (peaks
55.75 C and 58 C). No per-draw synchronization or disabled borders are required.
This is one reproduced hang and normal-login qualification, not full conformance
or long-duration stability qualification.

## SuperTuxKart kart preview, 2026-09-12

Ordinary GUI startup (real user graphics configuration) reached the profile and
main menus. Opening Singleplayer then aborted in Zink after pipeline creation
failed. Captured vertex input `0a4657c3a65992ef5212ca378961f573f4bf3ff002a98ac1dc90bcbe62d1a04f`
(5056 bytes, SHA256 `706b3d6a45d86d27ee191107a4c9ce5e6d8a6507bf25c4d25f53acba6d743921`)
rejected `if_fcmp register, 1.0`. Materializing nonzero floating branch constants
before register allocation compiles the complete input to 2070 bytes offline.
Both operand orders and signed minifloats have focused legalization coverage.
This fixes compiler acceptance only; the same workload also exposes an F16 EOT
store and packed HDR tile accesses that remain separate work.

The 804-byte fragment epilog input
`4eb59fe32f720c35b2c31e2b61a6c1ca39cadd3cf5bbe5d004a91adf90f52cfa`
adds into two R11G11B10_FLOAT targets at tile offsets 0 and 4. Those pixels
occupy one raw 32-bit word each, not three F32 words. Lowering their full RGB
loads/stores through Mesa's NIR format unpack/pack helpers compiles the captured
input to 682 bytes. A two-attachment HDR blend regression also exercises exact
F16-to-F32 widening after copy propagation folds a half source into F32
arithmetic. Run `agx3_graphics_compile_test --packed-hdr-only` on the host.
This is offline compiler coverage; native EOT support and pixel readback remain
necessary before qualifying the game. Partial write masks and explicit tile
store coordinates are not enabled by this lowering.
