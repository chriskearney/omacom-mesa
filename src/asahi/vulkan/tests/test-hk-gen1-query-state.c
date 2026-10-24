/* Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * Exercise the production query uploader with CPU-backed allocation and BO
 * reference fixtures. No DRM device or GPU execution is required.
 */
#include "hk_acceleration_structure.h"
#include "hk_cmd_buffer.h"
#include "asahi/lib/agx_scs_layout.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#undef NDEBUG
#include <assert.h>

static const uint64_t packed = UINT64_C(0x0130000060010070);
static const uint64_t table = UINT64_C(0x4321000000);
static void *allocations[16];
static unsigned allocation_count, retained_count;
static uint32_t last_size, last_alignment;
static bool fail_alloc;

struct agx_ptr
hk_pool_alloc_from_slow(struct hk_cmd_buffer *cmd, struct hk_uploader *uploader,
                        uint32_t size, uint32_t alignment,
                        enum hk_cmd_bo_type type)
{
   assert(type == HK_CMD_BO_MAIN);
   last_size = size;
   last_alignment = alignment;
   if (fail_alloc)
      return (struct agx_ptr){0};

   assert(allocation_count < ARRAY_SIZE(allocations));
   void *map = malloc(size);
   assert(map);
   /* Uninitialized pool memory must never leak into the legacy image/tail. */
   memset(map, 0xa5, size);
   allocations[allocation_count++] = map;
   return (struct agx_ptr){
      .cpu = map,
      .gpu = UINT64_C(0x3000000000) + allocation_count * UINT64_C(0x1000000),
   };
}

void
hk_cmd_buffer_retain_bo(struct hk_cmd_buffer *cmd, struct agx_bo *bo)
{
   assert(bo == hk_cmd_buffer_device(cmd)->gen1_rq_state ||
          bo == hk_cmd_buffer_device(cmd)->gen1_rq_code ||
          bo == hk_cmd_buffer_device(cmd)->gen1_rq_ift);
   retained_count++;
}

uint64_t
hk_gen1_ray_query_scs_layout(const struct hk_device *dev)
{
   return packed;
}

uint64_t
hk_gen1_upload_as_table(struct hk_cmd_buffer *cmd)
{
   return table;
}

VkResult
__vk_errorf(const void *obj, VkResult error, const char *file, int line,
            const char *format, ...)
{
   return error;
}

static void
reset_allocations(void)
{
   for (unsigned i = 0; i < allocation_count; ++i)
      free(allocations[i]);
   allocation_count = retained_count = 0;
}

static uint64_t
read64(const void *ptr)
{
   uint64_t value;
   memcpy(&value, ptr, sizeof(value));
   return value;
}

int
main(void)
{
   struct hk_device *dev = calloc(1, sizeof(*dev));
   struct hk_cmd_buffer *cmd = calloc(1, sizeof(*cmd));
   struct hk_cs *cs = calloc(1, sizeof(*cs));
   assert(dev && cmd && cs);
   cmd->vk.base.device = &dev->vk;
   cmd->current_cs.gfx = cs;
   dev->gen1_native_rq = dev->gen1_native_fragment_rq = true;
   dev->gen1_native_graphics_rq = true;
   dev->gen1_rq_instances = dev->gen1_rq_candidates = true;
   dev->gen1_rq_user_image = malloc(0x60000);
   assert(dev->gen1_rq_user_image);
   for (unsigned i = 0; i < 0x60000; ++i)
      dev->gen1_rq_user_image[i] = i * 7 + 3;

   struct agx_scs_layout_gen2 scs;
   assert(agx_scs_layout_gen1_init(packed, &scs));
   const uint32_t size = ALIGN_POT(scs.end[12], 0x4000);
   assert(size > 0x60000);

   /* Diagnostics keep independent full images, with current parameters and
    * zeroed padding even when the pool recycles dirty memory. */
   const uint64_t first = hk_gen1_query_state(cmd, &scs);
   assert(first && allocation_count == 1);
   assert(last_size == size && last_alignment == 0x4000);
   uint8_t parameters[48];
   agx_scs_layout_gen1_parameters(&scs, parameters);
   const uint8_t *image = allocations[0];
   assert(!memcmp(image, dev->gen1_rq_user_image, 0x4000));
   assert(!memcmp(image + 0x4000, parameters, sizeof(parameters)));
   assert(!memcmp(image + 0x4030, dev->gen1_rq_user_image + 0x4030,
                  0x60000 - 0x4030));
   for (unsigned i = 0x60000; i < size; ++i)
      assert(image[i] == 0);
   assert(hk_gen1_query_state(cmd, &scs) != first);
   assert(allocation_count == 2 && retained_count == 0);
   reset_allocations();

   fail_alloc = true;
   assert(hk_gen1_query_state(cmd, &scs) == 0);
   fail_alloc = false;

   /* Shared storage is addressed without allocating or mapping it. A second
    * command buffer sees the same immutable template. */
   struct agx_va va = {.addr = UINT64_C(0x4567000000)};
   struct agx_bo bo = {.va = &va};
   struct agx_bo code = {0}, ift = {0};
   dev->gen1_rq_state = &bo;
   dev->gen1_rq_code = &code;
   dev->gen1_rq_ift = &ift;
   struct hk_cmd_buffer *other = calloc(1, sizeof(*other));
   assert(other);
   other->vk.base.device = &dev->vk;
   fail_alloc = true;
   assert(hk_gen1_query_state(cmd, &scs) == va.addr);
   assert(hk_gen1_query_state(other, &scs) == va.addr);
   assert(allocation_count == 0 && retained_count == 2);
   fail_alloc = false;

   /* The real fragment and vertex upload paths allocate only uniforms and
    * the 8-byte profile, while retaining and rebasing the device image. */
   const uint64_t old = UINT64_C(0x20d0004000);
   memcpy(dev->gen1_rq_uniforms + 144, &old, sizeof(old));
   struct agx3_ioto_uniform_range range;
   for (unsigned stage = 0; stage < 2; ++stage) {
      reset_allocations();
      assert(hk_gen1_upload_fragment_query_state(cmd, stage, &range));
      assert(allocation_count == 2);
      assert(last_size == 8 && last_alignment == 8);
      const uint8_t *u = allocations[0];
      assert(range.start_halfs == 0 && range.size_halfs == 168);
      assert(read64(u + 144) == va.addr + 0x4000);
      assert(read64(u + 328) == table && read64(u + 352 + 48) == table);
      assert(read64(allocations[1]) == packed);
      assert(retained_count == 3); /* image, code, IFT */
   }
   assert(cs->gen1_native_fragment_rq && cs->gen1_native_vertex_rq);
   reset_allocations();

   fail_alloc = true;
   assert(!hk_gen1_upload_fragment_query_state(cmd, false, &range));
   assert(allocation_count == 0);
   fail_alloc = false;
   dev->gen1_native_fragment_rq = false;
   assert(!hk_gen1_upload_fragment_query_state(cmd, false, &range));
   assert(cmd->vk.record_result == VK_ERROR_FEATURE_NOT_PRESENT);
   assert(allocation_count == 0);

   printf("legacy state bytes per upload: %u (393216 copied, %u zeroed)\n",
          size, size - 0x60000);
   puts("shared fragment/vertex upload: 512 uniform bytes + 8 profile bytes");
   free(other);
   free(dev->gen1_rq_user_image);
   free(cs);
   free(cmd);
   free(dev);
   return 0;
}
