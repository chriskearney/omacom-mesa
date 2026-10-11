/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include "agx_va_runtime_hole.h"

#include "util/macros.h"
#include "util/u_math.h"

#include <stdlib.h>

struct agx_va_interval {
   uint64_t start;
   uint64_t end;
};

bool
agx_g17p_vdm_alias_heap_init(struct util_vma_heap *heap)
{
   if (!heap)
      return false;

   util_vma_heap_init(heap, AGX_G17P_VDM_ALIAS_BASE,
                      AGX_G17P_VDM_ALIAS_SIZE);
   heap->alloc_high = false;
   return heap->free_size == AGX_G17P_VDM_ALIAS_SIZE;
}

bool
agx_g17p_vdm_alias_range_valid(uint64_t addr, uint64_t size)
{
   if (size == 0 || addr < AGX_G17P_VDM_ALIAS_BASE ||
       size > UINT64_MAX - addr)
      return false;

   return addr + size <=
          AGX_G17P_VDM_USER_BASE + AGX_G17P_VDM_ALIAS_OFFSET_END;
}

bool
agx_g17p_vdm_alias_compact(uint64_t addr, uint32_t *offset)
{
   if (!offset || !agx_g17p_vdm_alias_range_valid(addr, 1))
      return false;

   const uint64_t compact = addr - AGX_G17P_VDM_USER_BASE;
   if (compact > UINT32_MAX)
      return false;

   *offset = (uint32_t)compact;
   return true;
}

bool
agx_va_split_user_window(uint64_t user_start, uint64_t user_size,
                         uint64_t *out_start, uint64_t *out_size)
{
   if (!out_start || !out_size || user_start == 0 || user_size == 0)
      return false;

   /* Round the user VA window to powers-of-two... */
   user_start = util_next_power_of_two64(user_start);
   user_size = util_next_power_of_two64(user_size + 1) >> 1;

   /* ...so when we cut user size in half to emulate sparse buffers... */
   user_size /= 2;

   /*
    * ...and again as many times as it takes for the whole window to sit below
    * the bit we are about to steal.  Upstream halves at most once more, which
    * is exactly enough for AGX2: a 39-bit user top leaves a window equal to
    * the rounded start, so one extra halving lands it below.  AGX3 needs the
    * loop -- T8140's 42-bit root leaves a window eight times the start, so a
    * single fixup would still straddle the sparse bit and the read-only alias
    * of a high allocation would carry out of its own half.
    */
   while (user_size >= user_start)
      user_size /= 2;

   if (user_size == 0)
      return false;

   /* ...we can distinguish the top/bottom half by an address bit */
   if ((user_start & user_size) != 0 ||
       ((user_start + (user_size - 1)) & user_size) != 0)
      return false;

   *out_start = user_start;
   *out_size = user_size;
   return true;
}

bool
agx_va_overlaps_runtime_graph_hole(enum agx_chip chip, uint64_t addr,
                                   uint64_t size)
{
   if (chip != AGX_CHIP_G17P || size == 0 ||
       addr >= AGX_G17_RUNTIME_GRAPH_VA_END)
      return false;

   return addr >= AGX_G17_RUNTIME_GRAPH_VA_START ||
          size > AGX_G17_RUNTIME_GRAPH_VA_START - addr;
}

static bool
range_end(uint64_t start, uint64_t size, uint64_t *end)
{
   if (size > UINT64_MAX - start)
      return false;

   *end = start + size;
   return true;
}

static void
add_intersection(struct agx_va_interval *intervals, unsigned *count,
                 uint64_t heap_start, uint64_t heap_end,
                 uint64_t reserved_start, uint64_t reserved_end)
{
   const uint64_t start = MAX2(heap_start, reserved_start);
   const uint64_t end = MIN2(heap_end, reserved_end);

   if (start < end)
      intervals[(*count)++] = (struct agx_va_interval){start, end};
}

static bool
reserve_intervals(struct util_vma_heap *heap,
                  struct agx_va_interval *intervals, unsigned count)
{
   if (count == 2 && intervals[1].start < intervals[0].start) {
      const struct agx_va_interval tmp = intervals[0];
      intervals[0] = intervals[1];
      intervals[1] = tmp;
   }

   if (count == 2 && intervals[1].start <= intervals[0].end) {
      intervals[0].end = MAX2(intervals[0].end, intervals[1].end);
      count = 1;
   }

   for (unsigned i = 0; i < count; ++i) {
      if (!util_vma_heap_alloc_addr(heap, intervals[i].start,
                                    intervals[i].end - intervals[i].start))
         return false;
   }

   return true;
}

bool
agx_va_reserve_runtime_graph_hole(
   enum agx_chip chip, struct util_vma_heap *main_heap, uint64_t main_start,
   uint64_t main_size, uint64_t sparse_ro_offset,
   struct util_vma_heap *usc_heap, uint64_t usc_start, uint64_t usc_size)
{
   if (chip != AGX_CHIP_G17P)
      return true;

   uint64_t main_end, usc_end;
   if (!range_end(main_start, main_size, &main_end) ||
       !range_end(usc_start, usc_size, &usc_end))
      return false;

   struct agx_va_interval main_intervals[2];
   unsigned main_count = 0;

   add_intersection(main_intervals, &main_count, main_start, main_end,
                    AGX_G17_RUNTIME_GRAPH_VA_START,
                    AGX_G17_RUNTIME_GRAPH_VA_END);

   /* Sparse buffers bind the main allocation again at sparse_ro_offset. */
   if (sparse_ro_offset < AGX_G17_RUNTIME_GRAPH_VA_END) {
      const uint64_t shadow_start =
         sparse_ro_offset < AGX_G17_RUNTIME_GRAPH_VA_START
            ? AGX_G17_RUNTIME_GRAPH_VA_START - sparse_ro_offset
            : 0;
      const uint64_t shadow_end =
         AGX_G17_RUNTIME_GRAPH_VA_END - sparse_ro_offset;

      add_intersection(main_intervals, &main_count, main_start, main_end,
                       shadow_start, shadow_end);
   }

   if (!reserve_intervals(main_heap, main_intervals, main_count))
      return false;

   struct agx_va_interval usc_interval[1];
   unsigned usc_count = 0;
   add_intersection(usc_interval, &usc_count, usc_start, usc_end,
                    AGX_G17_RUNTIME_GRAPH_VA_START,
                    AGX_G17_RUNTIME_GRAPH_VA_END);

   return reserve_intervals(usc_heap, usc_interval, usc_count);
}

/*
 * ==========================================================================
 * G15C (T6031, M3 Max): the GPU VA ranges the kernel M3 runtime owns in
 * every client VM. UPDATE THIS TABLE (and AGX_G15C_VA_LAYOUT) when the
 * kernel's T6031 layout changes; nothing else in Mesa encodes it.
 *
 * Source: the driver-reserved ranges that m3_runtime::new_vm
 * (drivers/gpu/drm/asahi/m3_runtime.rs) installs in each client VM. The
 * kernel refuses a client bind that touches one. The client-GPU allocations
 * of m3_pass_layout (the preemption buffers, TPC, tilemap, heap metadata,
 * scene and scratch arenas, the auxiliary and legacy per-cluster arenas and
 * the command aliases) lie inside them.
 *
 * Current values: integration/aurora 6c982dda3eba, where new_vm installs the
 * same four ranges for every M3 SoC and m3_pass_layout is qualified for two
 * clusters (QUALIFIED_CLUSTERS = 2; only the preemption buffers scale with
 * the cluster count, and four clusters still fit their slots). The T6031
 * four-cluster layout is written on the kernel branch
 * max/t6031-gpu-20261011; if it moves or grows a range, copy the new list
 * here.
 *
 * G15S and G15G reserve, in agx_open_device(), the two ranges their heaps
 * reach. This table also covers the top half of the USC window, which
 * their USC heap avoids only by allocating from the bottom.
 * ==========================================================================
 */
static const struct agx_va_range agx_g15c_runtime_ranges[] = {
   {UINT64_C(0x7000000000), UINT64_C(0x8000000000),
    "runtime graph and command aliases"},
   {UINT64_C(0x10080000000), UINT64_C(0x10100000000),
    "auxiliary and legacy per-cluster arenas"},
   {UINT64_C(0x1000000000), UINT64_C(0x1008000000),
    "parameter pool and render graph"},
   {UINT64_C(0x1074000000), UINT64_C(0x1074028000),
    "parameter page list"},
};

#define AGX_VA_MAX_RUNTIME_RANGES 8
static_assert(ARRAY_SIZE(agx_g15c_runtime_ranges) <= AGX_VA_MAX_RUNTIME_RANGES,
              "interval storage below holds two per range");

const struct agx_va_range *
agx_va_m3_runtime_ranges(enum agx_chip chip, unsigned *count)
{
   if (chip == AGX_CHIP_G15C) {
      if (count)
         *count = ARRAY_SIZE(agx_g15c_runtime_ranges);
      return agx_g15c_runtime_ranges;
   }

   if (count)
      *count = 0;
   return NULL;
}

static int
compare_intervals(const void *a, const void *b)
{
   const struct agx_va_interval *x = a, *y = b;
   return x->start < y->start ? -1 : x->start > y->start ? 1 : 0;
}

bool
agx_va_reserve_m3_runtime_ranges(enum agx_chip chip,
                                 const struct agx_va_heap_view *heaps,
                                 unsigned heap_count)
{
   unsigned count = 0;
   const struct agx_va_range *ranges = agx_va_m3_runtime_ranges(chip, &count);
   if (!ranges)
      return true;
   if (count > AGX_VA_MAX_RUNTIME_RANGES || (heap_count && !heaps))
      return false;

   for (unsigned h = 0; h < heap_count; ++h) {
      const struct agx_va_heap_view *view = &heaps[h];
      uint64_t heap_end;
      if (!view->heap || view->size == 0 ||
          !range_end(view->start, view->size, &heap_end))
         return false;

      struct agx_va_interval intervals[2 * AGX_VA_MAX_RUNTIME_RANGES];
      unsigned n = 0;
      for (unsigned r = 0; r < count; ++r) {
         if (ranges[r].start >= ranges[r].end)
            return false;

         add_intersection(intervals, &n, view->start, heap_end,
                          ranges[r].start, ranges[r].end);

         /* An allocation at a is also bound at a + alias_offset, so the
          * preimage of the range must stay free too. */
         if (view->alias_offset && ranges[r].end > view->alias_offset) {
            const uint64_t preimage_start =
               ranges[r].start > view->alias_offset
                  ? ranges[r].start - view->alias_offset
                  : 0;
            add_intersection(intervals, &n, view->start, heap_end,
                             preimage_start,
                             ranges[r].end - view->alias_offset);
         }
      }

      /* Ranges may touch or overlap once intersected; reserve each address
       * once. */
      qsort(intervals, n, sizeof(intervals[0]), compare_intervals);
      unsigned merged = 0;
      for (unsigned i = 0; i < n; ++i) {
         if (merged && intervals[i].start <= intervals[merged - 1].end) {
            intervals[merged - 1].end =
               MAX2(intervals[merged - 1].end, intervals[i].end);
         } else {
            intervals[merged++] = intervals[i];
         }
      }

      for (unsigned i = 0; i < merged; ++i) {
         if (!util_vma_heap_alloc_addr(view->heap, intervals[i].start,
                                       intervals[i].end - intervals[i].start))
            return false;
      }
   }

   return true;
}
