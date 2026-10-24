/* Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */
#include "hk_host_as.h"
#include "util/cache_ops.h"
#include "vk_acceleration_structure.h"
#include "vk_format.h"
#include "asahi/libagx/gen1_as.h"
#include <string.h>

bool
hk_host_as_stage_layout(const VkAccelerationStructureGeometryKHR *g,
                        const VkAccelerationStructureBuildRangeInfoKHR *r,
                        struct hk_host_as_staging *layout, uint64_t *offset)
{
   const uint64_t count = r->primitiveCount;
   if (!count)
      return true;
   switch (g->geometryType) {
   case VK_GEOMETRY_TYPE_TRIANGLES_KHR: {
      const VkAccelerationStructureGeometryTrianglesDataKHR *t = &g->geometry.triangles;
      const uint32_t fmt = vk_format_get_blocksize(t->vertexFormat);
      if (!fmt || !t->vertexStride)
         return false;
      const bool indexed = t->indexType != VK_INDEX_TYPE_NONE_KHR;
      uint64_t vertex_count;
      if (indexed) {
         vertex_count = r->firstVertex <= t->maxVertex ?
            (uint64_t)t->maxVertex + 1 - r->firstVertex : 0;
      } else {
         vertex_count = count * 3;
      }
      if (vertex_count > UINT32_MAX)
         return false;
      layout->vertex_count = vertex_count;
      layout->vertices = *offset;
      *offset = ALIGN_POT(*offset + (vertex_count ?
         (vertex_count - 1) * t->vertexStride + fmt : 0), UINT64_C(256));
      layout->indices = *offset;
      if (indexed)
         *offset = ALIGN_POT(*offset + count * 12, UINT64_C(256));
      layout->transform = *offset;
      if (t->transformData.hostAddress)
         *offset = ALIGN_POT(*offset + 48, UINT64_C(256));
      return true;
   }
   case VK_GEOMETRY_TYPE_AABBS_KHR:
      if (g->geometry.aabbs.stride < 24)
         return false;
      layout->data = *offset;
      *offset = ALIGN_POT(*offset + (count ?
         (count - 1) * g->geometry.aabbs.stride + 24 : 0), UINT64_C(256));
      return true;
   case VK_GEOMETRY_TYPE_INSTANCES_KHR:
      layout->data = *offset;
      *offset = ALIGN_POT(*offset + count * sizeof(VkAccelerationStructureInstanceKHR),
                          UINT64_C(256));
      return true;
   default:
      return false;
   }
}

static uint32_t
hk_host_as_read_index(const uint8_t *indices, VkIndexType type, uint64_t i)
{
   switch (type) {
   case VK_INDEX_TYPE_UINT8_EXT:
      return indices[i];
   case VK_INDEX_TYPE_UINT16: {
      uint16_t v;
      memcpy(&v, indices + 2 * i, 2);
      return v;
   }
   default: {
      uint32_t v;
      memcpy(&v, indices + 4 * i, 4);
      return v;
   }
   }
}

static void
hk_host_as_stage_geometry(VkAccelerationStructureGeometryKHR *g,
                          VkAccelerationStructureBuildRangeInfoKHR *r,
                          const struct hk_host_as_staging *layout,
                          uint8_t *map, uint64_t va)
{
   const VkAccelerationStructureBuildRangeInfoKHR input_range = *r;
   const uint64_t count = input_range.primitiveCount;
   r->primitiveOffset = 0;
   r->firstVertex = 0;
   r->transformOffset = 0;
   if (!count) {
      switch (g->geometryType) {
      case VK_GEOMETRY_TYPE_TRIANGLES_KHR:
         g->geometry.triangles.vertexData.deviceAddress = va;
         g->geometry.triangles.indexData.deviceAddress = va;
         g->geometry.triangles.transformData.deviceAddress = 0;
         break;
      case VK_GEOMETRY_TYPE_AABBS_KHR:
         g->geometry.aabbs.data.deviceAddress = va;
         break;
      case VK_GEOMETRY_TYPE_INSTANCES_KHR:
         g->geometry.instances.arrayOfPointers = VK_FALSE;
         g->geometry.instances.data.deviceAddress = va;
         break;
      default:
         break;
      }
      return;
   }
   switch (g->geometryType) {
   case VK_GEOMETRY_TYPE_TRIANGLES_KHR: {
      VkAccelerationStructureGeometryTrianglesDataKHR *t = &g->geometry.triangles;
      const uint32_t fmt = vk_format_get_blocksize(t->vertexFormat);
      const bool indexed = t->indexType != VK_INDEX_TYPE_NONE_KHR;
      const uint8_t *src = (const uint8_t *)t->vertexData.hostAddress +
                           (uint64_t)input_range.firstVertex * t->vertexStride +
                           (indexed ? 0 : input_range.primitiveOffset);
      if (layout->vertex_count)
         memcpy(map + layout->vertices, src,
                (layout->vertex_count - 1) * t->vertexStride + fmt);
      if (indexed) {
         const uint8_t *index_src = (const uint8_t *)t->indexData.hostAddress +
                                    input_range.primitiveOffset;
         uint32_t *dst = (uint32_t *)(map + layout->indices);
         const uint32_t last = layout->vertex_count ? layout->vertex_count - 1 : 0;
         for (uint64_t i = 0; i < count * 3; ++i)
            dst[i] = MIN2(hk_host_as_read_index(index_src, t->indexType, i), last);
         t->indexType = VK_INDEX_TYPE_UINT32;
         t->indexData.deviceAddress = va + layout->indices;
      } else {
         t->indexData.deviceAddress = 0;
      }
      if (t->transformData.hostAddress) {
         memcpy(map + layout->transform,
                (const uint8_t *)t->transformData.hostAddress + input_range.transformOffset, 48);
         t->transformData.deviceAddress = va + layout->transform;
      }
      t->vertexData.deviceAddress = va + layout->vertices;
      t->maxVertex = layout->vertex_count ? layout->vertex_count - 1 : 0;
      break;
   }
   case VK_GEOMETRY_TYPE_AABBS_KHR:
      if (count)
         memcpy(map + layout->data,
                (const uint8_t *)g->geometry.aabbs.data.hostAddress + input_range.primitiveOffset,
                (count - 1) * g->geometry.aabbs.stride + 24);
      g->geometry.aabbs.data.deviceAddress = va + layout->data;
      break;
   case VK_GEOMETRY_TYPE_INSTANCES_KHR: {
      const uint8_t *src = (const uint8_t *)g->geometry.instances.data.hostAddress +
                           input_range.primitiveOffset;
      VkAccelerationStructureInstanceKHR *dst =
         (VkAccelerationStructureInstanceKHR *)(map + layout->data);
      for (uint64_t i = 0; i < count; ++i) {
         const void *instance = g->geometry.instances.arrayOfPointers ?
            ((const void *const *)src)[i] : src + i * sizeof(*dst);
         memcpy(&dst[i], instance, sizeof(*dst));
         VK_FROM_HANDLE(vk_acceleration_structure, blas,
                        (VkAccelerationStructureKHR)dst[i].accelerationStructureReference);
         dst[i].accelerationStructureReference = blas ? vk_acceleration_structure_get_va(blas) : 0;
      }
      g->geometry.instances.arrayOfPointers = VK_FALSE;
      g->geometry.instances.data.deviceAddress = va + layout->data;
      break;
   }
   default:
      break;
   }
}

void
hk_host_as_stage_geometries(uint32_t count,
                            VkAccelerationStructureGeometryKHR *geometries,
                            VkAccelerationStructureBuildRangeInfoKHR *ranges,
                            const struct hk_host_as_staging *layouts,
                            uint8_t *map, uint64_t va, uint64_t size)
{
   for (uint32_t i = 0; i < count; ++i)
      hk_host_as_stage_geometry(&geometries[i], &ranges[i], &layouts[i], map, va);

   /* The CPU wrote a write-back BO, which is outside the command buffer's
    * upload pools. Publish all staged vertices, AABBs and instance records
    * before the internal queue can launch the device builder. */
   if (size && util_has_cache_ops())
      util_flush_range(map, size);
}
