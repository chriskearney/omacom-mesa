/*
 * Copyright 2026 Satya Benson
 * SPDX-License-Identifier: MIT
 *
 * G15G (M3 base, T8122) scheduling stalls.
 *
 * G15G runs the same AGX3b instruction stream as G15S/G16x/G17P, plus an
 * 8-byte stall -- a `nop` whose wait field (bits 10..15) is W, followed by
 * three plain `06 00` nops -- immediately before every instruction whose
 * scoreboard wait mask covers a slot that may still have an outstanding
 * producer writing the general register file; W is exactly those slots.
 * `end` counts as waiting on every slot. Branch displacements are then
 * recomputed over the padded stream. agx3_g15g_stalls.c states the rule in
 * full.
 */
#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define AGX3_G15G_STALL_SIZE 8u

/*
 * Rewrite an AGX3b instruction stream with G15G stalls inserted.
 *
 * On success returns the number of stalls inserted (>= 0) and sets *out to a
 * malloc'd buffer of *out_size bytes that the caller frees. If offset_map is
 * non-NULL it must hold in_size / 2 + 1 entries; for every instruction start
 * (and for in_size) it receives the rewritten offset of what now begins
 * there, i.e. the stall if one was inserted in front of the instruction.
 *
 * Returns -1 and leaves *out NULL if any instruction cannot be decoded, a
 * branch does not land on an instruction boundary, or a displacement cannot
 * be re-encoded. Callers must fail the shader rather than run it unpadded.
 */
int agx3_g15g_insert_stalls(const uint8_t *in, size_t in_size, uint8_t **out,
                            size_t *out_size, uint32_t *offset_map);

#ifdef __cplusplus
}
#endif
