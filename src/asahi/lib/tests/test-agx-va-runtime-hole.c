/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include "agx_va_runtime_hole.h"
#include "util/macros.h"

#include <assert.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>

#define PAGE UINT64_C(0x4000)
#define RUNTIME_GRAPH_HOLE_SIZE                                             \
   (AGX_G17_RUNTIME_GRAPH_VA_END - AGX_G17_RUNTIME_GRAPH_VA_START)

static bool
overlaps_runtime_graph_hole(uint64_t addr, uint64_t size)
{
   return addr < AGX_G17_RUNTIME_GRAPH_VA_END &&
          addr + size > AGX_G17_RUNTIME_GRAPH_VA_START;
}

static void
test_exact_contract(void)
{
   static_assert(AGX_G17_RUNTIME_GRAPH_VA_START == UINT64_C(0x7000000000),
                 "kernel contract start");
   static_assert(AGX_G17_RUNTIME_GRAPH_VA_END == UINT64_C(0x7100000000),
                 "kernel contract end");
   static_assert(AGX_G17_RUNTIME_GRAPH_VA_END -
                       AGX_G17_RUNTIME_GRAPH_VA_START ==
                    UINT64_C(0x100000000),
                 "kernel contract size");

   assert(!agx_va_overlaps_runtime_graph_hole(
      AGX_CHIP_G17P, AGX_G17_RUNTIME_GRAPH_VA_START - PAGE, PAGE));
   assert(agx_va_overlaps_runtime_graph_hole(
      AGX_CHIP_G17P, AGX_G17_RUNTIME_GRAPH_VA_START - PAGE, PAGE + 1));
   assert(agx_va_overlaps_runtime_graph_hole(
      AGX_CHIP_G17P, AGX_G17_RUNTIME_GRAPH_VA_START, PAGE));
   assert(!agx_va_overlaps_runtime_graph_hole(
      AGX_CHIP_G17P, AGX_G17_RUNTIME_GRAPH_VA_END, PAGE));
   assert(!agx_va_overlaps_runtime_graph_hole(
      AGX_CHIP_G17P, AGX_G17_RUNTIME_GRAPH_VA_START, 0));
   assert(!agx_va_overlaps_runtime_graph_hole(
      AGX_CHIP_G17A, AGX_G17_RUNTIME_GRAPH_VA_START, PAGE));
}

static void
test_main_and_timestamp_allocations_skip_hole(void)
{
   struct util_vma_heap main, usc;
   const uint64_t main_start = AGX_G17_RUNTIME_GRAPH_VA_START - (2 * PAGE);
   const uint64_t main_size =
      (AGX_G17_RUNTIME_GRAPH_VA_END - AGX_G17_RUNTIME_GRAPH_VA_START) +
      (4 * PAGE);
   util_vma_heap_init(&main, main_start, main_size);
   util_vma_heap_init(&usc, UINT64_C(0x1000000000), UINT64_C(0x100000000));

   assert(agx_va_reserve_runtime_graph_hole(
      AGX_CHIP_G17P, &main, main_start, main_size, UINT64_C(0x8000000000),
      &usc, UINT64_C(0x1000000000), UINT64_C(0x100000000)));
   assert(main.free_size == main_size - RUNTIME_GRAPH_HOLE_SIZE);

   assert(!util_vma_heap_alloc_addr(&main,
                                    AGX_G17_RUNTIME_GRAPH_VA_START, PAGE));
   assert(!util_vma_heap_alloc_addr(
      &main, AGX_G17_RUNTIME_GRAPH_VA_END - PAGE, PAGE));
   assert(!util_vma_heap_alloc_addr(
      &main, AGX_G17_RUNTIME_GRAPH_VA_START - PAGE, 2 * PAGE));

   const uint64_t ordinary = util_vma_heap_alloc(&main, PAGE, PAGE);
   const uint64_t timestamp = util_vma_heap_alloc(&main, PAGE, PAGE);
   assert(ordinary != 0 && !overlaps_runtime_graph_hole(ordinary, PAGE));
   assert(timestamp != 0 && !overlaps_runtime_graph_hole(timestamp, PAGE));

   util_vma_heap_finish(&usc);
   util_vma_heap_finish(&main);
}

static void
test_sparse_shadow_preimage_is_reserved(void)
{
   struct util_vma_heap main, usc;
   const uint64_t main_start = UINT64_C(0x2000000000);
   const uint64_t main_size = UINT64_C(0x4000000000);
   const uint64_t sparse_ro_offset = UINT64_C(0x4000000000);
   const uint64_t preimage =
      AGX_G17_RUNTIME_GRAPH_VA_START - sparse_ro_offset;
   util_vma_heap_init(&main, main_start, main_size);
   util_vma_heap_init(&usc, UINT64_C(0x1000000000), UINT64_C(0x100000000));

   assert(agx_va_reserve_runtime_graph_hole(
      AGX_CHIP_G17P, &main, main_start, main_size, sparse_ro_offset, &usc,
      UINT64_C(0x1000000000), UINT64_C(0x100000000)));
   assert(main.free_size == main_size - RUNTIME_GRAPH_HOLE_SIZE);

   assert(!util_vma_heap_alloc_addr(&main, preimage, PAGE));
   assert(!util_vma_heap_alloc_addr(
      &main, preimage + RUNTIME_GRAPH_HOLE_SIZE - PAGE, PAGE));
   assert(!util_vma_heap_alloc_addr(&main, preimage - PAGE, 2 * PAGE));

   const uint64_t sparse = util_vma_heap_alloc(&main, PAGE, PAGE);
   assert(sparse != 0);
   assert(!overlaps_runtime_graph_hole(sparse + sparse_ro_offset, PAGE));

   util_vma_heap_finish(&usc);
   util_vma_heap_finish(&main);
}

static void
test_fixed_usc_allocation_skips_hole(void)
{
   struct util_vma_heap main, usc;
   const uint64_t usc_start = AGX_G17_RUNTIME_GRAPH_VA_START - PAGE;
   const uint64_t usc_size =
      (AGX_G17_RUNTIME_GRAPH_VA_END - AGX_G17_RUNTIME_GRAPH_VA_START) +
      (2 * PAGE);
   util_vma_heap_init(&main, UINT64_C(0x2000000000), UINT64_C(0x100000000));
   util_vma_heap_init(&usc, usc_start, usc_size);

   assert(agx_va_reserve_runtime_graph_hole(
      AGX_CHIP_G17P, &main, UINT64_C(0x2000000000), UINT64_C(0x100000000),
      UINT64_C(0x4000000000), &usc, usc_start, usc_size));
   assert(usc.free_size == usc_size - RUNTIME_GRAPH_HOLE_SIZE);

   assert(!util_vma_heap_alloc_addr(&usc,
                                    AGX_G17_RUNTIME_GRAPH_VA_START, PAGE));
   assert(!util_vma_heap_alloc_addr(
      &usc, AGX_G17_RUNTIME_GRAPH_VA_END - PAGE, PAGE));
   const uint64_t automatic = util_vma_heap_alloc(&usc, PAGE, PAGE);
   assert(automatic != 0 && !overlaps_runtime_graph_hole(automatic, PAGE));

   util_vma_heap_finish(&usc);
   util_vma_heap_finish(&main);
}

static void
test_other_chips_keep_existing_layout(void)
{
   struct util_vma_heap main, usc;
   const uint64_t start = AGX_G17_RUNTIME_GRAPH_VA_START - PAGE;
   const uint64_t size = 3 * PAGE;
   util_vma_heap_init(&main, start, size);
   util_vma_heap_init(&usc, start, size);

   assert(agx_va_reserve_runtime_graph_hole(
      AGX_CHIP_G16C, &main, start, size, UINT64_C(0x8000000000), &usc,
      start, size));
   assert(util_vma_heap_alloc_addr(&main,
                                   AGX_G17_RUNTIME_GRAPH_VA_START, PAGE));
   assert(util_vma_heap_alloc_addr(&usc,
                                   AGX_G17_RUNTIME_GRAPH_VA_START, PAGE));

   util_vma_heap_finish(&usc);
   util_vma_heap_finish(&main);
}

/*
 * The user-window split, driven by the two kernel geometries that exist.
 *
 * Inputs are what agx_open_device() computes before the reduction:
 *   shader_base  = ALIGN_POT(MAX2(vm_start, 2^36 + 16384), 4 GiB) = 0x1100000000
 *   user_start   = shader_base + 4 GiB                            = 0x1200000000
 *   kernel_start = vm_end - MAX2(vm_kernel_min_size, 32 GiB)
 *   user_size    = kernel_start - user_start
 * with vm_start = UAT_PGSZ = 0x4000 on both geometries (kernel mmu.rs:145
 * IOVA_USER_BASE, and g17_drm.rs:24 T8140_USER_START), and vm_end two pages
 * below the root's translation top: 1 << 39 for AGX2 (mmu.rs:149), 1 << 42 for
 * AGX3 (g17_drm.rs:25).
 */
static void
check_split(const char *what, uint64_t user_start, uint64_t user_size,
            uint64_t expect_start, uint64_t expect_size)
{
   uint64_t start = 0, size = 0;
   const bool ok = agx_va_split_user_window(user_start, user_size, &start, &size);

   if (!ok || start != expect_start || size != expect_size) {
      fprintf(stderr,
              "%s: split(%#" PRIx64 ", %#" PRIx64 ") -> ok=%d start=%#" PRIx64
              " size=%#" PRIx64 ", expected start=%#" PRIx64 " size=%#" PRIx64
              "\n",
              what, user_start, user_size, ok, start, size, expect_start,
              expect_size);
      abort();
   }

   /* The invariant the whole scheme rests on: a sparse buffer's read-only
    * alias is `addr + size`, so every address in the window must have that
    * bit clear or the alias carries out of its own half. */
   assert((start & size) == 0);
   assert(((start + (size - 1)) & size) == 0);
   assert(start > size);
}

static void
test_user_window_split(void)
{
   const uint64_t vm_start = PAGE;
   const uint64_t shader_base = UINT64_C(0x1100000000);
   const uint64_t user_start = shader_base + UINT64_C(0x100000000);
   const uint64_t kernel_reserve = UINT64_C(32) << 30;
   assert(user_start > vm_start);

   /* AGX2 / G13-G14: 39-bit root. Upstream's single "quarters" fixup is
    * exactly enough here, and this case pins that behaviour unchanged. */
   const uint64_t agx2_vm_end = (UINT64_C(1) << 39) - 2 * PAGE;
   check_split("AGX2 39-bit", user_start,
               (agx2_vm_end - kernel_reserve) - user_start,
               UINT64_C(0x2000000000), UINT64_C(0x1000000000));

   /* AGX3 / T8140 G17P: 42-bit root. The window is eight times the rounded
    * start, so one fixup leaves 2^40 -- which straddles the sparse bit and
    * used to trip the assert in agx_open_device. */
   const uint64_t agx3_vm_end = (UINT64_C(1) << 42) - 2 * PAGE;
   check_split("AGX3 42-bit", user_start,
               (agx3_vm_end - kernel_reserve) - user_start,
               UINT64_C(0x2000000000), UINT64_C(0x1000000000));

   /* The AGX3 window and its read-only alias must both stay below the kernel
    * heap the same open() reserves at the top of the address space. */
   const uint64_t split_start = UINT64_C(0x2000000000);
   const uint64_t split_size = UINT64_C(0x1000000000);
   assert(split_start + 2 * split_size <= agx3_vm_end - kernel_reserve);
   /* ...and above the USC/shader heap, which the split does not move. */
   assert(split_start >= shader_base + UINT64_C(0x100000000));

   /* Degenerate inputs are refused, not asserted on. */
   uint64_t o1, o2;
   assert(!agx_va_split_user_window(0, 1 << 20, &o1, &o2));
   assert(!agx_va_split_user_window(1 << 20, 0, &o1, &o2));
   assert(!agx_va_split_user_window(1 << 20, 1 << 20, NULL, &o2));

   /* A window smaller than one page of slack still splits or is refused --
    * never returns a bit that the window straddles. */
   for (unsigned k = 1; k < 48; ++k) {
      uint64_t start = UINT64_C(1) << k, size = 0, out = 0;
      if (agx_va_split_user_window(start, start, &size, &out)) {
         assert((size & out) == 0);
         assert(((size + (out - 1)) & out) == 0);
      }
   }
}


/*
 * G17P's user heap and its sparse read-only alias must both fit inside the
 * compact VDM aperture.
 *
 * The VDM PPP State Update carries its payload pointer in a 40-bit field
 * (agx_vdm_ppp_state_core.h: word 0 bits 0..7 are pointer bits 32..39, read
 * off Apple's own G15/HAL200/HAL300 emitters). G17P's USC window sits at
 * exactly 2^40, so a user heap placed AFTER the window -- which is what every
 * other chip does, and what G17P used to do -- puts every command-stream
 * allocation out of reach of that field. agx_ppp_fini() then refused every
 * payload and vkEndCommandBuffer() failed before a render command buffer
 * could be recorded.
 *
 * The VDM control-stream register carries a 39-bit offset from 0x1000000000,
 * so merely remaining below 2^40 is insufficient: 0xbfff... is a valid
 * 40-bit absolute address but is not representable by that register.
 * agx_device.c bounds the two allocation windows by 0x9000000000.
 */
static void
test_g17p_user_window_fits_ppp_pointer_field(void)
{
   const uint64_t ppp_pointer_limit = UINT64_C(1) << 40;
   const uint64_t shader_base = UINT64_C(0x10000000000); /* AGX_G17P_USC_WINDOW_BASE */
   const uint64_t vdm_pointer_limit = AGX_G17P_VDM_USER_LIMIT;
   const uint64_t reservation = UINT64_C(1) << 36;

   assert(shader_base == ppp_pointer_limit &&
          "the USC window sits exactly at the PPP pointer limit");

   /* agx_device.c starts the low heap at shader_base/2 -- as high as it can
    * go -- because the split needs size < start and the alias needs a second
    * window of the same size. */
   uint64_t start = shader_base / 2;
   assert(start >= reservation);
   assert(start < vdm_pointer_limit && vdm_pointer_limit < shader_base);
   uint64_t size = vdm_pointer_limit - start;
   uint64_t out_start = 0, out_size = 0;
   assert(agx_va_split_user_window(start, size, &out_start, &out_size));

   /* Whatever the split picks, halving until the alias fits must terminate
    * with a non-empty window -- the same loop agx_device.c runs. */
   while (out_size > 0 && out_start + 2 * out_size > vdm_pointer_limit)
      out_size /= 2;
   assert(out_size > 0);

   /* The heap itself, and the read-only alias above it, are both addressable
    * by the 39-bit base-relative VDM pointer and the wider PPP pointer. */
   assert(out_start >= reservation);
   assert(out_size == UINT64_C(0x800000000)); /* 32 GiB */
   assert(out_start + out_size <= vdm_pointer_limit);
   assert(out_start + 2 * out_size <= vdm_pointer_limit);
   assert(out_start + out_size <= ppp_pointer_limit);
   assert(out_start + 2 * out_size <= ppp_pointer_limit);

   /* And it does not run into the USC window. */
   assert(out_start + 2 * out_size <= shader_base);
}

static void
test_g17p_vdm_command_alias_heap(void)
{
   struct util_vma_heap heap;
   assert(agx_g17p_vdm_alias_heap_init(&heap));

   const uint64_t command_bo_size = UINT64_C(0x20000);
   const uint64_t first =
      util_vma_heap_alloc(&heap, command_bo_size, PAGE);
   assert(first == AGX_G17P_VDM_ALIAS_BASE);
   assert(agx_g17p_vdm_alias_range_valid(first, command_bo_size));
   assert(first - AGX_G17P_VDM_USER_BASE <
          AGX_G17P_VDM_ALIAS_OFFSET_END);

   /* Kernel g17_resources.rs, native 092506 page-list capture. Each sparse
    * four-page block must remain disjoint from the entire command aperture. */
   const uint32_t native_tvb_pages[] = {
      0x11, 0x16, 0x1b, 0x20, 0x25, 0x2a, 0x4a, 0x4f,
      0x54, 0x59, 0x5e, 0x63, 0x68, 0x6d, 0x72, 0x77,
      0x7c, 0x81, 0x86, 0x8b, 0x90, 0x95, 0x9a, 0x9f,
      0xa4, 0xa9, 0xae, 0xb3, 0xb8, 0xbd, 0xc2, 0xc7,
   };
   for (unsigned i = 0; i < sizeof(native_tvb_pages) / sizeof(native_tvb_pages[0]); ++i) {
      const uint64_t end = ((uint64_t)native_tvb_pages[i] + 4) * 0x8000;
      assert(end <= AGX_G17P_VDM_ALIAS_OFFSET_START);
   }
   /* The old third command BO collided with block six. */
   assert(0x200000 + 3 * command_bo_size > (uint64_t)native_tvb_pages[6] * 0x8000);
   assert(0x200000 + 2 * command_bo_size < ((uint64_t)native_tvb_pages[6] + 4) * 0x8000);
   assert(AGX_G17P_VDM_ALIAS_OFFSET_START == 0x800000);

   const unsigned capacity = AGX_G17P_VDM_ALIAS_SIZE / command_bo_size;
   uint64_t last = first;
   for (unsigned i = 1; i < capacity; ++i) {
      last = util_vma_heap_alloc(&heap, command_bo_size, PAGE);
      assert(last != 0);
      assert(agx_g17p_vdm_alias_range_valid(last, command_bo_size));
   }
   assert(last + command_bo_size ==
          AGX_G17P_VDM_USER_BASE + AGX_G17P_VDM_ALIAS_OFFSET_END);
   assert(util_vma_heap_alloc(&heap, command_bo_size, PAGE) == 0);

   util_vma_heap_free(&heap, first, command_bo_size);
   assert(util_vma_heap_alloc(&heap, command_bo_size, PAGE) == first);

   assert(!agx_g17p_vdm_alias_range_valid(
      AGX_G17P_VDM_ALIAS_BASE - PAGE, PAGE));
   assert(!agx_g17p_vdm_alias_range_valid(
      AGX_G17P_VDM_USER_BASE + AGX_G17P_VDM_ALIAS_OFFSET_END, PAGE));
   assert(!agx_g17p_vdm_alias_range_valid(AGX_G17P_VDM_ALIAS_BASE, 0));

   uint32_t compact = 0;
   assert(agx_g17p_vdm_alias_compact(first, &compact));
   assert(compact == AGX_G17P_VDM_ALIAS_OFFSET_START);
   /* SuperTuxKart creates more than the old 192 live 128-KiB command BOs
    * during pipeline warmup. Offsets must survive the former +32-MiB bound. */
   assert(agx_g17p_vdm_alias_compact(
      AGX_G17P_VDM_USER_BASE + UINT64_C(0x02000000), &compact));
   assert(compact == UINT32_C(0x02000000));
   assert(agx_g17p_vdm_alias_compact(last + command_bo_size - 1, &compact));
   assert(compact == UINT32_MAX);
   assert(!agx_g17p_vdm_alias_compact(
      AGX_G17P_VDM_USER_BASE + (UINT64_C(1) << 32), &compact));
   assert(!agx_g17p_vdm_alias_compact(AGX_G17P_VDM_ALIAS_BASE - 1,
                                      &compact));
   assert(!agx_g17p_vdm_alias_compact(first, NULL));
   util_vma_heap_finish(&heap);
}

/* The G15C table: well formed, and a superset of what G15S and G15G
 * reserve in agx_open_device(). */
static void
test_g15c_runtime_table(void)
{
   unsigned count = 0;
   const struct agx_va_range *ranges =
      agx_va_m3_runtime_ranges(AGX_CHIP_G15C, &count);
   assert(ranges && count > 0);

   for (unsigned i = 0; i < count; ++i) {
      assert(ranges[i].start < ranges[i].end);
      assert(ranges[i].start % PAGE == 0 && ranges[i].end % PAGE == 0);
      assert(ranges[i].name && ranges[i].name[0]);
      for (unsigned j = 0; j < i; ++j)
         assert(ranges[i].end <= ranges[j].start ||
                ranges[j].end <= ranges[i].start);
   }

   const struct agx_va_range g15s_fixed[] = {
      {AGX_G17P_VDM_ALIAS_BASE, UINT64_C(0x1008000000), "render graph"},
      {UINT64_C(0x1074000000), UINT64_C(0x1074028000), "page list"},
   };
   for (unsigned k = 0; k < ARRAY_SIZE(g15s_fixed); ++k) {
      bool covered = false;
      for (unsigned i = 0; i < count; ++i)
         covered |= ranges[i].start <= g15s_fixed[k].start &&
                    g15s_fixed[k].end <= ranges[i].end;
      assert(covered);
   }

   const enum agx_chip others[] = {
      AGX_CHIP_G13G, AGX_CHIP_G15G, AGX_CHIP_G15S, AGX_CHIP_G15D,
      AGX_CHIP_G16G, AGX_CHIP_G17P, AGX_CHIP_UNKNOWN,
   };
   for (unsigned i = 0; i < ARRAY_SIZE(others); ++i) {
      count = 1;
      assert(!agx_va_m3_runtime_ranges(others[i], &count) && count == 0);
   }
}

/* The heaps agx_open_device() builds for the M3 runtime's user range
 * [0x4000, 0x2ffffff8000): the USC window at 2^40, the main heap below the
 * compact VDM limit with its sparse alias, and the low VDM alias heap. */
struct m3_heaps {
   struct util_vma_heap main, usc, vdm;
   struct agx_va_heap_view views[3];
};

static void
m3_heaps_init(struct m3_heaps *h)
{
   const uint64_t shader_base = UINT64_C(0x10000000000);
   uint64_t user_start = shader_base / 2;
   uint64_t user_size = AGX_G17P_VDM_USER_LIMIT - user_start;
   assert(agx_va_split_user_window(user_start, user_size, &user_start,
                                   &user_size));
   assert(user_start == UINT64_C(0x8000000000));
   assert(user_size == UINT64_C(0x800000000));

   util_vma_heap_init(&h->main, user_start, user_size);
   util_vma_heap_init(&h->usc, shader_base, UINT64_C(0x100000000));
   assert(agx_g17p_vdm_alias_heap_init(&h->vdm));
   h->usc.alloc_high = false;

   h->views[0] = (struct agx_va_heap_view){&h->main, user_start, user_size,
                                           user_size};
   h->views[1] = (struct agx_va_heap_view){&h->usc, shader_base,
                                           UINT64_C(0x100000000), 0};
   h->views[2] = (struct agx_va_heap_view){&h->vdm, AGX_G17P_VDM_ALIAS_BASE,
                                           AGX_G17P_VDM_ALIAS_SIZE, 0};
}

static void
m3_heaps_finish(struct m3_heaps *h)
{
   util_vma_heap_finish(&h->main);
   util_vma_heap_finish(&h->usc);
   util_vma_heap_finish(&h->vdm);
}

static void
test_g15c_device_heaps(void)
{
   struct m3_heaps h;
   m3_heaps_init(&h);
   const uint64_t main_free = h.main.free_size;
   const uint64_t usc_free = h.usc.free_size;
   const uint64_t vdm_free = h.vdm.free_size;

   assert(agx_va_reserve_m3_runtime_ranges(AGX_CHIP_G15C, h.views, 3));

   /* Main heap and its alias: no range reaches [0x8000000000, 0x9000000000). */
   assert(h.main.free_size == main_free);

   /* USC window: the top half belongs to the runtime. */
   assert(h.usc.free_size == usc_free - UINT64_C(0x80000000));
   assert(!util_vma_heap_alloc_addr(&h.usc, UINT64_C(0x10080000000), PAGE));
   assert(!util_vma_heap_alloc_addr(&h.usc, UINT64_C(0x100fffffc000), PAGE));
   assert(util_vma_heap_alloc(&h.usc, PAGE, PAGE) == UINT64_C(0x10000000000));
   assert(util_vma_heap_alloc_addr(&h.usc, UINT64_C(0x1007fffc000), PAGE));

   /* Low VDM heap: the same two reservations G15S and G15G make. */
   assert(h.vdm.free_size ==
          vdm_free - (UINT64_C(0x1008000000) - AGX_G17P_VDM_ALIAS_BASE) -
             UINT64_C(0x28000));
   assert(!util_vma_heap_alloc_addr(&h.vdm, AGX_G17P_VDM_ALIAS_BASE, PAGE));
   assert(!util_vma_heap_alloc_addr(&h.vdm, UINT64_C(0x1007ffc000), PAGE));
   assert(!util_vma_heap_alloc_addr(&h.vdm, UINT64_C(0x1074000000), PAGE));
   assert(!util_vma_heap_alloc_addr(&h.vdm, UINT64_C(0x1074024000), PAGE));
   assert(util_vma_heap_alloc(&h.vdm, PAGE, PAGE) == UINT64_C(0x1008000000));
   assert(util_vma_heap_alloc_addr(&h.vdm, UINT64_C(0x1073ffc000), PAGE));
   assert(util_vma_heap_alloc_addr(&h.vdm, UINT64_C(0x1074028000), PAGE));

   m3_heaps_finish(&h);
}

static void
test_m3_reservations_other_chips_and_failures(void)
{
   /* No table: nothing reserved, for G15S and G15G as before. */
   const enum agx_chip unchanged[] = {AGX_CHIP_G15S, AGX_CHIP_G15G,
                                      AGX_CHIP_G17P, AGX_CHIP_G13G};
   for (unsigned i = 0; i < ARRAY_SIZE(unchanged); ++i) {
      struct m3_heaps h;
      m3_heaps_init(&h);
      const uint64_t free_sizes[3] = {h.main.free_size, h.usc.free_size,
                                      h.vdm.free_size};
      assert(agx_va_reserve_m3_runtime_ranges(unchanged[i], h.views, 3));
      assert(h.main.free_size == free_sizes[0] &&
             h.usc.free_size == free_sizes[1] &&
             h.vdm.free_size == free_sizes[2]);
      m3_heaps_finish(&h);
   }

   /* A client allocation already inside a range: refuse the device. */
   struct m3_heaps h;
   m3_heaps_init(&h);
   assert(util_vma_heap_alloc_addr(&h.vdm, UINT64_C(0x1074010000), PAGE));
   assert(!agx_va_reserve_m3_runtime_ranges(AGX_CHIP_G15C, h.views, 3));
   m3_heaps_finish(&h);

   /* Malformed views. */
   m3_heaps_init(&h);
   struct agx_va_heap_view bad = h.views[1];
   bad.size = 0;
   assert(!agx_va_reserve_m3_runtime_ranges(AGX_CHIP_G15C, &bad, 1));
   assert(!agx_va_reserve_m3_runtime_ranges(AGX_CHIP_G15C, NULL, 1));
   assert(agx_va_reserve_m3_runtime_ranges(AGX_CHIP_G15C, NULL, 0));
   m3_heaps_finish(&h);
}

/* A heap whose alias lands on a range reserves the range's preimage, and
 * direct and aliased intersections that overlap are reserved once. */
static void
test_m3_reservation_alias_preimage(void)
{
   struct util_vma_heap heap;
   /* Allocations in [0x5800000000, 0x7800000000) are also bound 0x1800000000
    * higher, i.e. in [0x7000000000, 0x9000000000). */
   const uint64_t start = UINT64_C(0x5800000000);
   const uint64_t size = UINT64_C(0x2000000000);
   const uint64_t alias = UINT64_C(0x1800000000);
   util_vma_heap_init(&heap, start, size);
   const struct agx_va_heap_view view = {&heap, start, size, alias};

   assert(agx_va_reserve_m3_runtime_ranges(AGX_CHIP_G15C, &view, 1));

   /* Direct: [0x7000000000, 0x7800000000) (heap end). Preimage of
    * [0x7000000000, 0x8000000000): [0x5800000000, 0x6800000000). */
   assert(!util_vma_heap_alloc_addr(&heap, UINT64_C(0x7000000000), PAGE));
   assert(!util_vma_heap_alloc_addr(&heap, UINT64_C(0x77ffffc000), PAGE));
   assert(!util_vma_heap_alloc_addr(&heap, start, PAGE));
   assert(!util_vma_heap_alloc_addr(&heap, UINT64_C(0x67ffffc000), PAGE));
   assert(heap.free_size == size - UINT64_C(0x800000000) -
                               UINT64_C(0x1000000000));
   assert(util_vma_heap_alloc_addr(&heap, UINT64_C(0x6800000000), PAGE));
   assert(util_vma_heap_alloc_addr(&heap, UINT64_C(0x6fffffc000), PAGE));
   util_vma_heap_finish(&heap);

   /* [0x6000000000, 0x8000000000) aliased 0x800000000 higher: the direct
    * part [0x7000000000, 0x8000000000) and the preimage [0x6800000000,
    * 0x7800000000) overlap and are reserved as one interval. */
   util_vma_heap_init(&heap, UINT64_C(0x6000000000), size);
   const struct agx_va_heap_view overlapping = {
      &heap, UINT64_C(0x6000000000), size, UINT64_C(0x800000000)};
   assert(agx_va_reserve_m3_runtime_ranges(AGX_CHIP_G15C, &overlapping, 1));
   assert(heap.free_size == size - UINT64_C(0x1800000000));
   assert(!util_vma_heap_alloc_addr(&heap, UINT64_C(0x6800000000), PAGE));
   assert(!util_vma_heap_alloc_addr(&heap, UINT64_C(0x7fffffc000), PAGE));
   assert(util_vma_heap_alloc_addr(&heap, UINT64_C(0x67ffffc000), PAGE));
   util_vma_heap_finish(&heap);
}

int
main(void)
{
   test_exact_contract();
   test_user_window_split();
   test_main_and_timestamp_allocations_skip_hole();
   test_sparse_shadow_preimage_is_reserved();
   test_fixed_usc_allocation_skips_hole();
   test_other_chips_keep_existing_layout();
   test_g17p_user_window_fits_ppp_pointer_field();
   test_g17p_vdm_command_alias_heap();
   test_g15c_runtime_table();
   test_g15c_device_heaps();
   test_m3_reservations_other_chips_and_failures();
   test_m3_reservation_alias_preimage();
   puts("AGX VA layout: user-window split, G17 runtime-graph hole and G15C runtime range checks passed");
   return 0;
}
