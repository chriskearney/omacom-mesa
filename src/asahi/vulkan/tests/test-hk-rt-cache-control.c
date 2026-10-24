/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

/* CPU control of the production RT creation entry point.  Translation and
 * megakernel construction are real; only GPU shader compilation/upload is
 * replaced.  These checks do not execute shaders or establish conformance.
 */
#include "spirv/spirv.h"
#include "asahi/compiler/agx3_compute_program.h"
#include "util/set.h"
#include "hk_device.h"
#include "hk_entrypoints.h"
#include "hk_physical_device.h"
#include "hk_ray_tracing_pipeline.h"
#include "hk_shader.h"
#include "nir.h"
#include "vk_alloc.h"
#include "vk_common_entrypoints.h"
#include "vk_instance.h"
#include "vk_nir.h"
#include "vk_pipeline_cache.h"
#include "vk_pipeline_layout.h"
#include "vk_shader_module.h"

#include <stdio.h>

#define CHECK(x)                                                               \
   do {                                                                        \
      if (!(x)) {                                                              \
         fprintf(stderr, "%s:%d: %s\n", __func__, __LINE__, #x);               \
         abort();                                                              \
      }                                                                        \
   } while (0)
#define FAIL  VK_PIPELINE_CREATE_FAIL_ON_PIPELINE_COMPILE_REQUIRED_BIT
#define LIB   VK_PIPELINE_CREATE_LIBRARY_BIT_KHR
#define EARLY VK_PIPELINE_CREATE_EARLY_RETURN_ON_FAILURE_BIT

static unsigned translations, backends, shader_destroys, live_nir;
static nir_shader *tracked_nir[64];
static VkResult backend_result = VK_SUCCESS;
static unsigned fail_translation;

VkResult __real_vk_pipeline_shader_stage_to_nir(
   struct vk_device *, VkPipelineCreateFlags2KHR,
   const VkPipelineShaderStageCreateInfo *, const struct spirv_to_nir_options *,
   const nir_shader_compiler_options *, void *, nir_shader **);

VkResult
__wrap_vk_pipeline_shader_stage_to_nir(
   struct vk_device *dev, VkPipelineCreateFlags2KHR flags,
   const VkPipelineShaderStageCreateInfo *stage,
   const struct spirv_to_nir_options *spirv_options,
   const nir_shader_compiler_options *nir_options, void *ctx, nir_shader **out);

VkResult
__wrap_vk_pipeline_shader_stage_to_nir(
   struct vk_device *dev, VkPipelineCreateFlags2KHR flags,
   const VkPipelineShaderStageCreateInfo *stage,
   const struct spirv_to_nir_options *spirv_options,
   const nir_shader_compiler_options *nir_options, void *ctx, nir_shader **out)
{
   translations++;
   if (fail_translation == translations)
      return VK_ERROR_OUT_OF_HOST_MEMORY;
   VkResult result = __real_vk_pipeline_shader_stage_to_nir(
      dev, flags, stage, spirv_options, nir_options, ctx, out);
   if (result == VK_SUCCESS) {
      unsigned i;
      for (i = 0; i < ARRAY_SIZE(tracked_nir); i++) {
         if (!tracked_nir[i]) {
            tracked_nir[i] = *out;
            live_nir++;
            break;
         }
      }
      CHECK(i < ARRAY_SIZE(tracked_nir));
   }
   return result;
}

void __real_ralloc_free(void *);
void
__wrap_ralloc_free(void *ptr);

void
__wrap_ralloc_free(void *ptr)
{
   if (ptr) {
      for (unsigned i = 0; i < ARRAY_SIZE(tracked_nir); i++) {
         if (tracked_nir[i] == ptr) {
            tracked_nir[i] = NULL;
            live_nir--;
         }
      }
   }
   __real_ralloc_free(ptr);
}

VkResult
__wrap_hk_compile_shader(struct hk_device *dev,
                         struct vk_shader_compile_info *info,
                         const struct vk_graphics_pipeline_state *state,
                         const struct vk_features *features,
                         const VkAllocationCallbacks *alloc,
                         struct hk_api_shader **out);

VkResult
__wrap_hk_compile_shader(struct hk_device *dev,
                         struct vk_shader_compile_info *info,
                         const struct vk_graphics_pipeline_state *state,
                         const struct vk_features *features,
                         const VkAllocationCallbacks *alloc,
                         struct hk_api_shader **out)
{
   backends++;
   ralloc_free(info->nir);
   if (backend_result != VK_SUCCESS)
      return backend_result;
   *out = calloc(1, sizeof(**out));
   CHECK(*out);
   return VK_SUCCESS;
}

void
__wrap_hk_api_shader_destroy(struct vk_device *dev, struct vk_shader *shader,
                             const VkAllocationCallbacks *alloc);

void
__wrap_hk_api_shader_destroy(struct vk_device *dev, struct vk_shader *shader,
                             const VkAllocationCallbacks *alloc)
{
   shader_destroys++;
   free(container_of(shader, struct hk_api_shader, vk));
}

/* Empty raygen SPIR-V, assembled from the public SPIR-V opcodes. */
static const uint32_t raygen[] = {
   SpvMagicNumber,
   0x00010400,
   0,
   5,
   0,
   (2 << 16) | SpvOpCapability,
   SpvCapabilityShader,
   (2 << 16) | SpvOpCapability,
   SpvCapabilityRayTracingKHR,
   (6 << 16) | SpvOpExtension,
   0x5f565053,
   0x5f52484b,
   0x5f796172,
   0x63617274,
   0x00676e69,
   (3 << 16) | SpvOpMemoryModel,
   SpvAddressingModelLogical,
   SpvMemoryModelGLSL450,
   (5 << 16) | SpvOpEntryPoint,
   SpvExecutionModelRayGenerationKHR,
   3,
   0x6e69616d,
   0,
   (2 << 16) | SpvOpTypeVoid,
   1,
   (3 << 16) | SpvOpTypeFunction,
   2,
   1,
   (5 << 16) | SpvOpFunction,
   1,
   3,
   SpvFunctionControlMaskNone,
   2,
   (2 << 16) | SpvOpLabel,
   4,
   (1 << 16) | SpvOpReturn,
   (1 << 16) | SpvOpFunctionEnd,
};

static struct vk_instance instance;
static struct hk_device dev;
static struct hk_physical_device pdev;
static VkDevice device;
static struct vk_pipeline_layout *layout;
static VkShaderModule module;
static VkPipelineShaderStageCreateInfo stage;
static VkRayTracingShaderGroupCreateInfoKHR group = {
   .sType = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR,
   .type = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR,
   .generalShader = 0,
   .closestHitShader = VK_SHADER_UNUSED_KHR,
   .anyHitShader = VK_SHADER_UNUSED_KHR,
   .intersectionShader = VK_SHADER_UNUSED_KHR,
};
static unsigned pipeline_allocs, pipeline_frees;

static void *VKAPI_CALL
alloc_pipeline(void *data, size_t size, size_t alignment,
               VkSystemAllocationScope scope)
{
   void *ptr = NULL;
   if (posix_memalign(&ptr, MAX2(alignment, sizeof(void *)), size) != 0)
      return NULL;
   pipeline_allocs++;
   return ptr;
}
static void VKAPI_CALL
free_pipeline(void *data, void *ptr)
{
   if (ptr)
      pipeline_frees++;
   free(ptr);
}
static const VkAllocationCallbacks alloc = {
   .pfnAllocation = alloc_pipeline,
   .pfnFree = free_pipeline,
};

static void VKAPI_CALL
get_properties(VkPhysicalDevice physical, VkPhysicalDeviceProperties *props)
{
   memset(props, 0, sizeof(*props));
}

static VkRayTracingPipelineCreateInfoKHR
info(VkPipelineCreateFlags flags)
{
   return (VkRayTracingPipelineCreateInfoKHR){
      .sType = VK_STRUCTURE_TYPE_RAY_TRACING_PIPELINE_CREATE_INFO_KHR,
      .flags = flags,
      .stageCount = 1,
      .pStages = &stage,
      .groupCount = 1,
      .pGroups = &group,
      .maxPipelineRayRecursionDepth = 1,
      .layout = vk_pipeline_layout_to_handle(layout),
   };
}

static void
destroy(VkPipeline pipeline)
{
   if (pipeline) {
      struct hk_ray_tracing_pipeline *p =
         hk_ray_tracing_pipeline_from_handle(pipeline);
      p->base.ops->destroy(&dev.vk, &p->base, &alloc);
   }
}

static VkResult
create(VkPipelineCache cache, const VkRayTracingPipelineCreateInfoKHR *ci,
       VkPipeline *pipeline)
{
   *pipeline = (VkPipeline)(uintptr_t)0xbad;
   return hk_CreateRayTracingPipelinesKHR(device, VK_NULL_HANDLE, cache, 1, ci,
                                          &alloc, pipeline);
}

static void
check_refusal(VkPipelineCache cache, VkRayTracingPipelineCreateInfoKHR ci)
{
   VkPipeline p;
   unsigned t = translations, b = backends, n = live_nir;
   unsigned a = pipeline_allocs, f = pipeline_frees, refs = layout->ref_cnt;
   VkResult result = create(cache, &ci, &p);
   if (result != VK_PIPELINE_COMPILE_REQUIRED || translations != t ||
       backends != b)
      fprintf(stderr, "flag oracle: result=%d, translations=%u, backends=%u\n",
              result, translations - t, backends - b);
   CHECK(result == VK_PIPELINE_COMPILE_REQUIRED);
   CHECK(p == VK_NULL_HANDLE);
   CHECK(translations == t && backends == b && live_nir == n);
   CHECK(layout->ref_cnt == refs);
   CHECK(pipeline_allocs - a == pipeline_frees - f);
}

int
main(void)
{
   glsl_type_singleton_init_or_ref();
   vk_object_base_init(NULL, &instance.base, VK_OBJECT_TYPE_INSTANCE);
   instance.base.client_visible = true;
   list_inithead(&instance.debug_utils.callbacks);
   list_inithead(&instance.debug_report.callbacks);
   vk_object_base_init(&dev.vk, &dev.vk.base, VK_OBJECT_TYPE_DEVICE);
   vk_object_base_init(NULL, &pdev.vk.base, VK_OBJECT_TYPE_PHYSICAL_DEVICE);
   instance.app_info.api_version = VK_API_VERSION_1_4;
   pdev.vk.instance = &instance;
   dev.vk.physical = &pdev.vk;
   dev.vk.alloc = *vk_default_allocator();
   dev.dev.chip = AGX_CHIP_G17P;
   dev.vk.robustness_state = (struct vk_pipeline_robustness_state){
      .storage_buffers = VK_PIPELINE_ROBUSTNESS_BUFFER_BEHAVIOR_DISABLED_EXT,
      .uniform_buffers = VK_PIPELINE_ROBUSTNESS_BUFFER_BEHAVIOR_DISABLED_EXT,
      .vertex_inputs = VK_PIPELINE_ROBUSTNESS_BUFFER_BEHAVIOR_DISABLED_EXT,
      .images = VK_PIPELINE_ROBUSTNESS_IMAGE_BEHAVIOR_DISABLED_EXT,
   };
   dev.vk.enabled_features.rayTracingPipeline = true;
   dev.vk.enabled_features.shaderModuleIdentifier = true;
   dev.vk.enabled_features.shaderInt64 = true;
   pdev.vk.supported_features = dev.vk.enabled_features;
   pdev.vk.dispatch_table.GetPhysicalDeviceProperties = get_properties;
   pdev.vk.properties.subgroupSize = 32;
   pdev.vk.properties.minSubgroupSize = 32;
   pdev.vk.properties.maxSubgroupSize = 32;
   device = hk_device_to_handle(&dev);
   layout = vk_pipeline_layout_zalloc(
      &dev.vk, sizeof(*layout),
      &(VkPipelineLayoutCreateInfo){
         .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
      });
   CHECK(layout);
   VkShaderModuleCreateInfo module_info = {
      .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
      .codeSize = sizeof(raygen),
      .pCode = raygen,
   };
   CHECK(vk_common_CreateShaderModule(device, &module_info, NULL, &module) ==
         VK_SUCCESS);
   stage = (VkPipelineShaderStageCreateInfo){
      .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
      .stage = VK_SHADER_STAGE_RAYGEN_BIT_KHR,
      .module = module,
      .pName = "main",
   };
   struct vk_pipeline_cache *empty =
      vk_pipeline_cache_create(&dev.vk,
                               &(struct vk_pipeline_cache_create_info){
                                  .force_enable = true,
                                  .skip_disk_cache = true,
                               },
                               NULL);
   struct vk_pipeline_cache *populated =
      vk_pipeline_cache_create(&dev.vk,
                               &(struct vk_pipeline_cache_create_info){
                                  .force_enable = true,
                                  .skip_disk_cache = true,
                               },
                               NULL);
   CHECK(empty && populated);
   const char key[] = "unrelated raw cache entry";
   struct vk_raw_data_cache_object *object = vk_raw_data_cache_object_create(
      &dev.vk, key, sizeof(key), raygen, sizeof(raygen));
   CHECK(object);
   vk_pipeline_cache_object_unref(
      &dev.vk, vk_pipeline_cache_add_object(populated, &object->base));
   CHECK(empty->object_cache->entries == 0);
   CHECK(populated->object_cache->entries == 1);
   VkPipelineCache caches[] = {VK_NULL_HANDLE,
                               vk_pipeline_cache_to_handle(empty),
                               vk_pipeline_cache_to_handle(populated)};
   VkPipelineShaderStageModuleIdentifierCreateInfoEXT identifier = {
      .sType =
         VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_MODULE_IDENTIFIER_CREATE_INFO_EXT,
      .identifierSize = 1,
      .pIdentifier = (const uint8_t *)"x",
   };
   VkPipeline p;
   for (unsigned c = 0; c < ARRAY_SIZE(caches); c++) {
      for (unsigned inline_code = 0; inline_code < 2; inline_code++) {
         stage.module = inline_code ? VK_NULL_HANDLE : module;
         stage.pNext = inline_code ? &module_info : NULL;
         check_refusal(caches[c], info(FAIL));
         check_refusal(caches[c], info(FAIL | LIB));
         VkRayTracingPipelineCreateInfoKHR ci = info(0);
         unsigned t = translations, b = backends;
         CHECK(create(caches[c], &ci, &p) == VK_SUCCESS);
         CHECK(translations == t + 1 && backends == b + 1);
         destroy(p);
         CHECK(live_nir == 0);
         /* Warming a cache through this API cannot create an RT cache hit. */
         check_refusal(caches[c], info(FAIL));
         VkPipelineCreateFlags2CreateInfoKHR flags2 = {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_CREATE_FLAGS_2_CREATE_INFO_KHR,
            .flags = FAIL,
         };
         ci.pNext = &flags2;
         check_refusal(caches[c], ci);
      }
      stage.module = VK_NULL_HANDLE;
      stage.pNext = &identifier;
      check_refusal(caches[c], info(FAIL));
      check_refusal(caches[c], info(FAIL | LIB));
   }
   /* The nonempty-ID cases without FAIL here are defensive function controls,
    * not valid application calls.  A module or inline code takes precedence
    * over an identifier in the translator.  A zero
    * identifier size with code must retain the same ordinary creation path.
    */
   for (unsigned size = 0; size < 2; size++) {
      identifier.identifierSize = size;
      for (unsigned inline_code = 0; inline_code < 2; inline_code++) {
         module_info.pNext = &identifier;
         stage.module = inline_code ? VK_NULL_HANDLE : module;
         stage.pNext = inline_code ? (const void *)&module_info : &identifier;
         check_refusal(VK_NULL_HANDLE, info(FAIL));
         VkRayTracingPipelineCreateInfoKHR code_ci = info(LIB);
         unsigned t = translations;
         CHECK(create(VK_NULL_HANDLE, &code_ci, &p) == VK_SUCCESS);
         CHECK(translations == t + 1);
         destroy(p);
         CHECK(live_nir == 0);
      }
   }
   identifier.identifierSize = 1;
   module_info.pNext = NULL;
   puts(
      "PASS: module/inline/identifier; null/empty/populated cache; warm call; flags2");

   stage.module = module;
   stage.pNext = NULL;
   VkRayTracingPipelineCreateInfoKHR ci = info(LIB);
   VkPipeline library;
   CHECK(create(VK_NULL_HANDLE, &ci, &library) == VK_SUCCESS);
   struct hk_ray_tracing_pipeline *lib =
      hk_ray_tracing_pipeline_from_handle(library);
   CHECK(lib->stages[0]->ref_cnt == 1 && live_nir == 1);
   VkPipelineLibraryCreateInfoKHR imports = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_LIBRARY_CREATE_INFO_KHR,
      .libraryCount = 1,
      .pLibraries = &library,
   };
   ci = info(FAIL | LIB);
   ci.stageCount = ci.groupCount = 0;
   ci.pLibraryInfo = &imports;
   unsigned t = translations, b = backends;
   CHECK(create(VK_NULL_HANDLE, &ci, &p) == VK_SUCCESS);
   CHECK(translations == t && backends == b && lib->stages[0]->ref_cnt == 2);
   destroy(p);
   CHECK(lib->stages[0]->ref_cnt == 1);
   ci.flags = FAIL;
   check_refusal(VK_NULL_HANDLE, ci);
   CHECK(lib->stages[0]->ref_cnt == 1); /* failure releases imported NIR */
   ci.flags = 0;
   CHECK(create(VK_NULL_HANDLE, &ci, &p) == VK_SUCCESS);
   CHECK(translations == t && backends == b + 1);
   destroy(p);
   CHECK(lib->stages[0]->ref_cnt == 1);
   ci.flags = FAIL | LIB;
   ci.stageCount = ci.groupCount = 1;
   check_refusal(VK_NULL_HANDLE, ci); /* new local code plus imported NIR */
   puts("PASS: library reuse and executable link compilation boundary");

   /* A later stage failure must release earlier translated stages. */
   VkPipelineShaderStageCreateInfo stages[2] = {stage, stage};
   stages[1].module = VK_NULL_HANDLE;
   stages[1].pNext = &identifier;
   ci = info(LIB);
   ci.stageCount = 2;
   ci.pStages = stages;
   unsigned n = live_nir, refs = layout->ref_cnt;
   CHECK(create(VK_NULL_HANDLE, &ci, &p) == VK_PIPELINE_COMPILE_REQUIRED);
   CHECK(p == VK_NULL_HANDLE && live_nir == n && layout->ref_cnt == refs);
   stages[1] = stage;
   fail_translation = translations + 2;
   CHECK(create(VK_NULL_HANDLE, &ci, &p) == VK_ERROR_OUT_OF_HOST_MEMORY);
   CHECK(p == VK_NULL_HANDLE && live_nir == n && layout->ref_cnt == refs);
   fail_translation = 0;
   backend_result = VK_ERROR_OUT_OF_HOST_MEMORY;
   ci = info(0);
   CHECK(create(VK_NULL_HANDLE, &ci, &p) == VK_ERROR_OUT_OF_HOST_MEMORY);
   CHECK(p == VK_NULL_HANDLE && live_nir == n && layout->ref_cnt == refs);
   backend_result = VK_SUCCESS;
   puts("PASS: partial stage, translation failure and backend failure cleanup");

   for (unsigned early = 0; early < 2; early++) {
      VkRayTracingPipelineCreateInfoKHR batch[] = {
         info(LIB),
         info(FAIL | (early ? EARLY : 0)),
         info(LIB),
      };
      VkPipeline pipelines[3] = {(VkPipeline)1, (VkPipeline)1, (VkPipeline)1};
      t = translations;
      CHECK(hk_CreateRayTracingPipelinesKHR(device, VK_NULL_HANDLE, caches[2],
                                            3, batch, &alloc, pipelines) ==
            VK_PIPELINE_COMPILE_REQUIRED);
      CHECK(pipelines[0] != VK_NULL_HANDLE && pipelines[1] == VK_NULL_HANDLE);
      CHECK((pipelines[2] == VK_NULL_HANDLE) == (early != 0));
      CHECK(translations == t + (early ? 1 : 2));
      for (unsigned i = 0; i < 3; i++)
         destroy(pipelines[i]);
   }
   puts(
      "PASS: mixed batches, success preservation, EARLY_RETURN and skipped null handles");
   /* CTS misc.shaders_from_lib creates stage-only libraries and supplies
    * groups when linking, with recursion depth 10 in all three creates. */
   for (unsigned depth = 10; depth <= HK_G17P_MAX_REQUESTED_RAY_RECURSION_DEPTH;
        depth += HK_G17P_MAX_REQUESTED_RAY_RECURSION_DEPTH - 10) {
      VkPipeline libs[2];
      ci = info(LIB);
      ci.maxPipelineRayRecursionDepth = depth;
      ci.groupCount = 0;
      ci.pGroups = NULL;
      CHECK(create(VK_NULL_HANDLE, &ci, &libs[0]) == VK_SUCCESS);
      CHECK(create(VK_NULL_HANDLE, &ci, &libs[1]) == VK_SUCCESS);
      VkPipelineLibraryCreateInfoKHR imports = {
         .sType = VK_STRUCTURE_TYPE_PIPELINE_LIBRARY_CREATE_INFO_KHR,
         .libraryCount = 2, .pLibraries = libs,
      };
      ci = info(0);
      ci.maxPipelineRayRecursionDepth = depth;
      ci.stageCount = 0;
      ci.pStages = NULL;
      ci.pLibraryInfo = &imports;
      CHECK(create(VK_NULL_HANDLE, &ci, &p) == VK_SUCCESS);
      struct hk_ray_tracing_pipeline *linked =
         hk_ray_tracing_pipeline_from_handle(p);
      CHECK(linked->stage_count == 2 && linked->group_count == 1);
      destroy(libs[0]);
      destroy(libs[1]);
      destroy(p);
   }
   ci = info(0);
   ci.maxPipelineRayRecursionDepth = HK_G17P_MAX_REQUESTED_RAY_RECURSION_DEPTH + 1;
   CHECK(create(VK_NULL_HANDLE, &ci, &p) == VK_ERROR_FEATURE_NOT_PRESENT);
   CHECK(p == VK_NULL_HANDLE);
   puts("PASS: stage-only libraries, link-time groups, legacy-client depth 10/31 and overflow");
   /* M1-M3 retain the legacy limit even for the same library request. */
   dev.dev.chip = AGX_CHIP_G13G;
   ci = info(LIB);
   ci.maxPipelineRayRecursionDepth = HK_MAX_RAY_RECURSION_DEPTH + 1;
   t = translations;
   CHECK(create(VK_NULL_HANDLE, &ci, &p) == VK_ERROR_FEATURE_NOT_PRESENT);
   CHECK(p == VK_NULL_HANDLE && translations == t);
   dev.dev.chip = AGX_CHIP_G17P;

   /* Exercise the real G17P loader and its SPECTPR spill frame. The nested
    * callable CTS has a 256-element local array and reserves 31 frames. */
   struct agx3_g17p_compute_program_desc desc = {
      .program_va = 0x100000, .shader_va = 0x200000,
      .shader_binary_size = 64, .nr_gprs = 32,
      .compiler_d_valid = true, .compiler_d = 64,
      .ldshdr_f1 = AGX_IOTO_G17P_COMPUTE_LDSHDR_F1_DIRECT_ADD3,
   };
   CHECK(agx3_cdm_direct_template_sources_compute(false, false, false,
                                                  &desc.launch_sources));
   const uint32_t frames[] = {0, 8192, 31 * 1088, UINT16_MAX * 4u};
   for (unsigned i = 0; i < ARRAY_SIZE(frames); ++i) {
      desc.scratch_size = frames[i];
      struct agx3_g17p_compute_program program;
      CHECK(agx3_g17p_compute_program_generate(AGX_CHIP_G17P, &desc, &program));
      uint8_t spectpr[AGX_IOTO_SPECTPR_SIZE];
      CHECK(agx_ioto_g17p_compiler_spectpr_pack(frames[i], desc.compiler_d,
                                               spectpr, sizeof(spectpr)));
      bool found = false;
      for (unsigned j = 0; j + sizeof(spectpr) <= program.length; ++j)
         found |= !memcmp(program.bytes + j, spectpr, sizeof(spectpr));
      CHECK(found);
   }
   desc.scratch_size = UINT16_MAX * 4u + 4u;
   struct agx3_g17p_compute_program program;
   CHECK(!agx3_g17p_compute_program_generate(AGX_CHIP_G17P, &desc, &program));
   desc.scratch_size = 8193;
   CHECK(!agx3_g17p_compute_program_generate(AGX_CHIP_G17P, &desc, &program));
   puts("PASS: G17P callable-size spill frames and encoding bounds");

   destroy(library);
   CHECK(live_nir == 0 && pipeline_allocs == pipeline_frees);
   CHECK(layout->ref_cnt == 1);
   CHECK(shader_destroys == backends - 1); /* injected backend failure */
   vk_pipeline_layout_unref(&dev.vk, layout);
   vk_common_DestroyShaderModule(device, module, NULL);
   vk_pipeline_cache_destroy(empty, NULL);
   vk_pipeline_cache_destroy(populated, NULL);
   vk_object_base_finish(&dev.vk.base);
   vk_object_base_finish(&pdev.vk.base);
   vk_object_base_finish(&instance.base);
   glsl_type_singleton_decref();
   puts(
      "PASS: all pipeline allocations, stage NIR, layout and shader references released");
   return 0;
}
