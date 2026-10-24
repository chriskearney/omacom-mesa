/* Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */
#include "hk_host_as.h"
#include "vk_acceleration_structure.h"
#undef NDEBUG
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* A non-coherent fixture: device-visible bytes change only on publication. */
static uint8_t *device_map;
static uint64_t published_size;
void __wrap_util_flush_range(void *map, size_t size);
void __wrap_util_flush_range(void *map, size_t size)
{
   published_size = size;
   if (device_map)
      memcpy(device_map, map, size);
}

static const uint64_t va = UINT64_C(0x1800000000);

static void normalized(const VkAccelerationStructureBuildRangeInfoKHR *r)
{
   assert(r->primitiveOffset == 0 && r->firstVertex == 0 && r->transformOffset == 0);
}

static void instances(unsigned count, bool pointers)
{
   const unsigned prefix = 128;
   size_t stride = pointers ? sizeof(void *) : sizeof(VkAccelerationStructureInstanceKHR);
   uint8_t *input = calloc(1, prefix + (size_t)count * stride);
   VkAccelerationStructureInstanceKHR *items = calloc(count, sizeof(*items));
   struct vk_acceleration_structure *as = calloc(count, sizeof(*as));
   assert(input && items && as);
   for (unsigned i = 0; i < count; ++i) {
      vk_object_base_init(NULL, &as[i].base, VK_OBJECT_TYPE_ACCELERATION_STRUCTURE_KHR);
      as[i].addr = va + (uint64_t)i * 32768 + 123;
      items[i].accelerationStructureReference = i % 17
         ? (uint64_t)(uintptr_t)vk_acceleration_structure_to_handle(&as[i]) : 0;
      items[i].instanceCustomIndex = i;
      items[i].instanceShaderBindingTableRecordOffset = i % 2;
      items[i].mask = 0xff;
      items[i].transform.matrix[0][0] = 1;
      items[i].transform.matrix[1][1] = 1;
      items[i].transform.matrix[2][2] = 1;
      items[i].transform.matrix[0][3] = (float)i;
      if (pointers)
         ((const void **)(input + prefix))[i] = &items[i];
      else
         memcpy(input + prefix + i * stride, &items[i], sizeof(items[i]));
   }
   VkAccelerationStructureGeometryKHR g = {
      .geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR,
      .geometry.instances = {.arrayOfPointers = pointers, .data.hostAddress = input},
   };
   VkAccelerationStructureBuildRangeInfoKHR r = {.primitiveCount = count, .primitiveOffset = prefix};
   struct hk_host_as_staging layout = {0};
   uint64_t bytes = 256;
   assert(hk_host_as_stage_layout(&g, &r, &layout, &bytes));
   assert(bytes == ((256 + (uint64_t)count * 64 + 255) & ~UINT64_C(255)));
   uint8_t *map = calloc(1, bytes + 16);
   assert(map);
   memset(map + bytes, 0xa5, 16);
   device_map = calloc(1, bytes);
   assert(device_map);
   published_size = 0;
   hk_host_as_stage_geometries(1, &g, &r, &layout, map, va, bytes);
   assert(published_size == bytes);
   assert(!memcmp(map, device_map, bytes));
   free(device_map);
   device_map = NULL;
   normalized(&r);
   assert(!g.geometry.instances.arrayOfPointers);
   assert(g.geometry.instances.data.deviceAddress == va + layout.data);
   for (unsigned i = 0; i < count; ++i) {
      VkAccelerationStructureInstanceKHR expected = items[i];
      expected.accelerationStructureReference = i % 17 ? as[i].addr : 0;
      assert(!memcmp(map + layout.data + (uint64_t)i * 64, &expected, sizeof(expected)));
      vk_object_base_finish(&as[i].base);
   }
   for (unsigned i = 0; i < 16; ++i) assert(map[bytes + i] == 0xa5);
   free(map); free(input); free(items); free(as);
}

static void geometry_offsets(void)
{
   uint32_t input[256], transforms[64];
   for (unsigned i = 0; i < 256; ++i) input[i] = 1000 + i;
   for (unsigned i = 0; i < 64; ++i) transforms[i] = 2000 + i;
   uint8_t map[4096];
   for (unsigned indexed = 0; indexed < 2; ++indexed) {
      uint16_t indices[] = {99, 99, 99, 0, 5, 2};
      VkAccelerationStructureGeometryKHR g = {
         .geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR,
         .geometry.triangles = {
            .vertexFormat = VK_FORMAT_R32G32B32_SFLOAT, .vertexStride = 16,
            .vertexData.hostAddress = input, .maxVertex = 7,
            .indexType = indexed ? VK_INDEX_TYPE_UINT16 : VK_INDEX_TYPE_NONE_KHR,
            .indexData.hostAddress = indices, .transformData.hostAddress = transforms,
         },
      };
      VkAccelerationStructureBuildRangeInfoKHR r = {
         .primitiveCount = 1, .firstVertex = 2, .primitiveOffset = indexed ? 6 : 16,
         .transformOffset = 32,
      };
      struct hk_host_as_staging l = {0}; uint64_t bytes = 0;
      assert(hk_host_as_stage_layout(&g, &r, &l, &bytes));
      memset(map, 0, sizeof(map));
      hk_host_as_stage_geometries(1, &g, &r, &l, map, va, bytes);
      normalized(&r);
      assert(!memcmp(map + l.vertices, (uint8_t *)input + 32 + (indexed ? 0 : 16),
                     (l.vertex_count - 1) * 16 + 12));
      assert(!memcmp(map + l.transform, (uint8_t *)transforms + 32, 48));
      if (indexed) {
         uint32_t expected[] = {0, 5, 2};
         assert(!memcmp(map + l.indices, expected, sizeof(expected)));
      }
   }
   VkAccelerationStructureGeometryKHR g = {
      .geometryType = VK_GEOMETRY_TYPE_AABBS_KHR,
      .geometry.aabbs = {.stride = 32, .data.hostAddress = input},
   };
   VkAccelerationStructureBuildRangeInfoKHR r = {.primitiveCount = 4, .primitiveOffset = 64};
   struct hk_host_as_staging l = {0}; uint64_t bytes = 256;
   assert(hk_host_as_stage_layout(&g, &r, &l, &bytes));
   hk_host_as_stage_geometries(1, &g, &r, &l, map, va, bytes);
   normalized(&r);
   assert(!memcmp(map + l.data, (uint8_t *)input + 64, 3 * 32 + 24));
}

int main(void)
{
   geometry_offsets();
   const unsigned counts[] = {65535, 65536, 65537};
   for (unsigned i = 0; i < 3; ++i)
      for (unsigned p = 0; p < 2; ++p) instances(counts[i], p);
   puts("PASS: published host geometry offsets and 65,535/65,536/65,537 packed/pointer instances");
   return 0;
}
