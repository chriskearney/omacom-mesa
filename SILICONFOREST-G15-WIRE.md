# Omacom retained G15 wire compatibility

The J516S boot at kernel `7.2.11-lab7b969982-2e59`, built from
`omacom/linux-aurora@7b969982306e1498a8cbe6e5da9bde743acd8fc5`, reported
GET_PARAMS features `0x3e`, USC 3 and HAL 0. The initial integration refused it
before VM creation. No GPU benchmark was run on that failed admission.

The source explains the distinction:

- [g15/wire.rs](https://github.com/omacom/linux-aurora/blob/7b969982306e1498a8cbe6e5da9bde743acd8fc5/drivers/gpu/drm/asahi/g15/wire.rs)
  explicitly retains the Mesa-751 private wire. Feature words are `0x3e`, or
  `0xbe` with early tiling. Render is 384 bytes with priors at 256/264; compute
  is 136 bytes with scratch at 64 and priors at 72/80.
- The canonical header contains eight additional timestamp-copy records in
  each payload. Canonical render is 448 bytes with priors at 320/328; compute
  is 200 bytes with scratch at 128 and priors at 136/144. Its per-command sync,
  mirror and exact-prior feature assignments also differ.
- [g15/command.rs](https://github.com/omacom/linux-aurora/blob/7b969982306e1498a8cbe6e5da9bde743acd8fc5/drivers/gpu/drm/asahi/g15/command.rs)
  consumes the private structs, not those in the canonical header. Comparing
  the latter alone does not qualify this backend.

The compatibility path recognizes only already-admitted G15S/G15C identities
with one of those two exact feature words, on direct DRM. It normalizes the
features once before VM creation and converts packets at submission. It copies
the shared prefixes, verifies every removed timestamp-copy byte is zero, moves
the complete remaining tail without discarding nonzero fields, and preserves
command headers, dependency barriers, attachments, queue IDs and sync objects.
Kernel validation still checks all unsupported fields and GPU references.

The caller's command buffer is unchanged. Temporary packet storage lives until
the synchronous submit ioctl returns. Unknown signatures, mixed feature words,
truncated/unrecognized packets, output exhaustion and nonzero copy slots are
rejected by the retained-wire selector/converter. The existing canonical
feature gate is unchanged. No kernel flag, GPU identity gate, VA reservation or firmware check
is bypassed. No feature bits are changed in the kernel. This is compatibility
with these known experimental contracts, not general ABI negotiation.

`asahi-g15-wire` checks all 65,536 low feature combinations, identity refusals,
both feature normalizations, every current render/compute packet extent,
retained tail/barrier bytes, a mixed attachment/render/compute stream, all input
truncations, output bounds and every removed byte's nonzero refusal. The
existing canonical ABI/admission controls remain in place.

Both x86_64 and native ARM64 builds passed all 92 Asahi tests. The follow-up
passed reboot, compositor/display and independent GL ES/Vulkan pixel checks
on J516S, J313 and J293. On J516S it worked on both the newer pinned kernel
and the previously working default kernel; the native GET_PARAMS word remained
`0x3e`, with no kernel feature changes. Matched Quake medians were 185.8 and
184.7 FPS respectively. The complete normal lab job remains TESTS_FAILED due
to the recorded OMT checks, despite the specific Mesa subtests passing.
See [the selected hardware evidence](SILICONFOREST-HARDWARE-20261024.json).
J613/J615 revised-layout hardware qualification, full CTS and sustained behavior
remain unqualified. No M3 Air hardware result is inferred from the M1 Air.
