/* Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */
#include "hk_ray_tracing_pipeline.h"
#include "hk_shader.h"
#include <stdio.h>
#include <string.h>

#define CHECK(x) do { if (!(x)) { \
   fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #x); return false; \
} } while (0)

static unsigned stats_calls, ir_calls;

static VkResult
get_stats(struct vk_device *dev, const struct vk_shader *shader,
          uint32_t index, uint32_t *count,
          VkPipelineExecutableStatisticKHR *stats)
{
   ++stats_calls;
   if (index != 0)
      return VK_ERROR_UNKNOWN;
   VK_OUTARRAY_MAKE_TYPED(VkPipelineExecutableStatisticKHR, out, stats, count);
   vk_outarray_append_typed(VkPipelineExecutableStatisticKHR, &out, stat) {
      VK_COPY_STR(stat->name, "test statistic");
      stat->format = VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_UINT64_KHR;
      stat->value.u64 = 42;
   }
   return vk_outarray_status(&out);
}

static VkResult
get_irs(struct vk_device *dev, const struct vk_shader *shader,
        uint32_t index, uint32_t *count,
        VkPipelineExecutableInternalRepresentationKHR *irs)
{
   ++ir_calls;
   if (index != 0)
      return VK_ERROR_UNKNOWN;
   /* Honeykrisp currently exposes no shader IRs. */
   *count = 0;
   return VK_SUCCESS;
}

static bool
check_pipeline(void)
{
   const struct vk_shader_ops shader_ops = {
      .get_executable_statistics = get_stats,
      .get_executable_internal_representations = get_irs,
   };
   struct hk_api_shader shader = {.vk.ops = &shader_ops};
   struct hk_ray_tracing_pipeline pipeline = {
      .base = {
         .ops = &hk_ray_tracing_pipeline_ops,
         .bind_point = VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR,
         .stages = VK_SHADER_STAGE_RAYGEN_BIT_KHR |
                   VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR |
                   VK_SHADER_STAGE_MISS_BIT_KHR,
      },
      .shader = &shader,
   };
   const struct vk_pipeline_ops *ops = pipeline.base.ops;
   CHECK(ops->get_executable_properties);
   CHECK(ops->get_executable_statistics);
   CHECK(ops->get_internal_representations);
   uint32_t count = 99;
   CHECK(ops->get_executable_properties(NULL, &pipeline.base, &count, NULL) ==
         VK_SUCCESS);
   CHECK(count == 1);
   VkPipelineExecutablePropertiesKHR props = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_PROPERTIES_KHR,
      .pNext = &shader,
   };
   count = 0;
   CHECK(ops->get_executable_properties(NULL, &pipeline.base, &count, &props) ==
         VK_INCOMPLETE);
   CHECK(count == 0);
   count = 1;
   CHECK(ops->get_executable_properties(NULL, &pipeline.base, &count, &props) ==
         VK_SUCCESS);
   CHECK(count == 1 && props.stages == pipeline.base.stages);
   CHECK(props.subgroupSize == 32 && strlen(props.name) && strlen(props.description));
   CHECK(props.sType == VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_PROPERTIES_KHR);
   CHECK(props.pNext == &shader);

   count = 0;
   CHECK(ops->get_executable_statistics(NULL, &pipeline.base, 0, &count, NULL) ==
         VK_SUCCESS);
   CHECK(count == 1);
   VkPipelineExecutableStatisticKHR stat = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_STATISTIC_KHR,
   };
   count = 0;
   CHECK(ops->get_executable_statistics(NULL, &pipeline.base, 0, &count, &stat) ==
         VK_INCOMPLETE);
   count = 1;
   CHECK(ops->get_executable_statistics(NULL, &pipeline.base, 0, &count, &stat) ==
         VK_SUCCESS);
   CHECK(count == 1 && stat.value.u64 == 42 && stats_calls == 3);
   count = 99;
   CHECK(ops->get_internal_representations(NULL, &pipeline.base, 0, &count, NULL) ==
         VK_SUCCESS);
   CHECK(count == 0);
   VkPipelineExecutableInternalRepresentationKHR ir = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_INTERNAL_REPRESENTATION_KHR,
   };
   count = 1;
   CHECK(ops->get_internal_representations(NULL, &pipeline.base, 0, &count, &ir) ==
         VK_SUCCESS);
   CHECK(count == 0 && ir_calls == 2);

   /* A library and a linked pipeline with imported stages both use this ops
    * table. Libraries have no megakernel, regardless of their stage count. */
   pipeline.shader = NULL;
   pipeline.stage_count = 6;
   count = 99;
   CHECK(ops->get_executable_properties(NULL, &pipeline.base, &count, NULL) ==
         VK_SUCCESS);
   CHECK(count == 0);
   count = 1;
   CHECK(ops->get_executable_properties(NULL, &pipeline.base, &count, &props) ==
         VK_SUCCESS);
   CHECK(count == 0);
   return true;
}

int main(void) { return check_pipeline() ? 0 : 1; }
