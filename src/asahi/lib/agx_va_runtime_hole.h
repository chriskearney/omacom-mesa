/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include "agx_chip.h"
#include "util/vma.h"

#include <stdbool.h>
#include <stdint.h>

/* T8140's per-VM kernel runtime-graph range, expressed as [start, end). */
#define AGX_G17_RUNTIME_GRAPH_VA_START UINT64_C(0x7000000000)
#define AGX_G17_RUNTIME_GRAPH_VA_END   UINT64_C(0x7100000000)

/* G17P's compact VDM pointer is a 39-bit offset from 0x1000000000.
 * Keep the ordinary allocation and sparse-shadow windows below this exclusive
 * limit so every VDM control stream and payload is directly representable. */
#define AGX_G17P_VDM_USER_BASE  UINT64_C(0x1000000000)
#define AGX_G17P_VDM_USER_SIZE  (UINT64_C(1) << 39)
#define AGX_G17P_VDM_USER_LIMIT                                            \
   (AGX_G17P_VDM_USER_BASE + AGX_G17P_VDM_USER_SIZE)

/* Native TVB blocks are sparse 128-KiB mappings with 32-KiB page IDs
 * 0x11..0xc7. Their last range ends at compact 0x658000; the fixed scene,
 * discard and status mappings also live below this bound. Start command BOs
 * at +8 MiB, above the complete retained render pool. Bind-tail and draw-state
 * records carry unsigned 32-bit context offsets; PPP/stream pointers are wider.
 * Use that narrowest encoding's entire range, rather than limiting live command
 * BOs to the original bring-up window's 24 MiB. This reserves virtual addresses,
 * not backing memory: BO storage and page tables are allocated on demand. */
#define AGX_G17P_VDM_ALIAS_OFFSET_START UINT64_C(0x00800000)
#define AGX_G17P_VDM_ALIAS_OFFSET_END   (UINT64_C(1) << 32)
#define AGX_G17P_VDM_ALIAS_BASE                                         \
   (AGX_G17P_VDM_USER_BASE + AGX_G17P_VDM_ALIAS_OFFSET_START)
#define AGX_G17P_VDM_ALIAS_SIZE                                         \
   (AGX_G17P_VDM_ALIAS_OFFSET_END - AGX_G17P_VDM_ALIAS_OFFSET_START)

/*
 * Reduce the kernel-reported user VA window to the power-of-two window the
 * driver actually allocates from, and derive the address bit that tells a
 * sparse buffer's read-only alias apart from its primary mapping.
 *
 * A sparse buffer is bound a second time at `addr + sparse_ro_offset`
 * (hk_buffer.c:95, d12_resource.c:1118).  For that to be a pure bit-set with
 * no carry -- and for the alias window to stay disjoint from the primary one
 * -- every address in [*out_start, *out_start + *out_size) must have the
 * sparse bit clear.  Since both are powers of two once rounded, that reduces
 * to *out_start > *out_size.  On success `*out_size` is both the window size
 * and the sparse read-only offset.
 *
 * Returns false if no usable split exists.
 */
bool agx_va_split_user_window(uint64_t user_start, uint64_t user_size,
                              uint64_t *out_start, uint64_t *out_size);

bool agx_va_overlaps_runtime_graph_hole(enum agx_chip chip, uint64_t addr,
                                        uint64_t size);

bool agx_g17p_vdm_alias_heap_init(struct util_vma_heap *heap);
bool agx_g17p_vdm_alias_range_valid(uint64_t addr, uint64_t size);
bool agx_g17p_vdm_alias_compact(uint64_t addr, uint32_t *offset);

bool agx_va_reserve_runtime_graph_hole(
   enum agx_chip chip, struct util_vma_heap *main_heap, uint64_t main_start,
   uint64_t main_size, uint64_t sparse_ro_offset,
   struct util_vma_heap *usc_heap, uint64_t usc_start, uint64_t usc_size);

/* A GPU VA range [start, end). */
struct agx_va_range {
   uint64_t start;
   uint64_t end;
   const char *name;
};

/* Names the kernel layout the G15C table in agx_va_runtime_hole.c copies,
 * so a run's stderr records which one this build carries. Update it with
 * the table. */
#define AGX_G15C_VA_LAYOUT "m3-runtime-6c982dda3eba"

/* The ranges the kernel M3 runtime owns in every client VM of a chip that
 * reserves them through agx_va_reserve_m3_runtime_ranges(), or NULL with
 * *count = 0. Only G15C has a table: G15S and G15G keep their two fixed
 * reservations in agx_open_device(). The table is in agx_va_runtime_hole.c. */
const struct agx_va_range *agx_va_m3_runtime_ranges(enum agx_chip chip,
                                                    unsigned *count);

/* One VA heap of a device: the heap, the window it was initialised with,
 * and, when nonzero, the offset at which every allocation of the heap is
 * bound a second time (the sparse read-only alias of the main heap). */
struct agx_va_heap_view {
   struct util_vma_heap *heap;
   uint64_t start;
   uint64_t size;
   uint64_t alias_offset;
};

/* Reserve, in each heap, every address that falls in one of the chip's
 * runtime-owned ranges, directly or through the heap's alias, so no client
 * allocation is ever bound there. Returns true without reserving anything
 * for a chip without a table, and false if a range cannot be reserved (it
 * overlaps an allocation already made); the caller must then refuse the
 * device. */
bool agx_va_reserve_m3_runtime_ranges(enum agx_chip chip,
                                      const struct agx_va_heap_view *heaps,
                                      unsigned heap_count);
