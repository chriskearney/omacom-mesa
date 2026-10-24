/* Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 * Offline regression probe: records real Vulkan transfers and compiles their
 * meta pipelines under drm-shim. Never submits work or checks GPU pixels.
 * Build with Honeykrisp's include flags and link to the Vulkan loader.
 */
#undef NDEBUG
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <vulkan/vulkan.h>
#include "hk_device.h"
#include "hk_image.h"
#include "hk_image_view.h"
#include "compiler/nir/nir.h"
#include "vulkan/util/vk_internal_exts.h"

#define CHECK(call)                                                            \
   do {                                                                        \
      VkResult r = (call);                                                     \
      if (r != VK_SUCCESS) {                                                   \
         fprintf(stderr, "%s: %d\n", #call, r);                                \
         exit(2);                                                              \
      }                                                                        \
   } while (0)

static PFN_vkCreateImageView create_view;
static PFN_vkCreateComputePipelines create_compute;
static unsigned views, depth_views, stencil_views, packed_views;
static unsigned msaa_stores;
static bool require_fixed;

static VKAPI_ATTR VkResult VKAPI_CALL
observe_compute(VkDevice device, VkPipelineCache cache, uint32_t count,
                const VkComputePipelineCreateInfo *infos,
                const VkAllocationCallbacks *alloc, VkPipeline *out)
{
   for (unsigned i = 0; i < count; ++i) {
      const VkPipelineShaderStageNirCreateInfoMESA *ni =
         vk_find_struct_const(infos[i].stage.pNext,
                              PIPELINE_SHADER_STAGE_NIR_CREATE_INFO_MESA);
      if (!ni)
         continue;
      nir_foreach_function_impl(impl, ni->nir) {
         nir_foreach_block(block, impl) {
            nir_foreach_instr(instr, block) {
               if (instr->type != nir_instr_type_intrinsic)
                  continue;
               nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
               if (intr->intrinsic != nir_intrinsic_image_deref_store ||
                   nir_intrinsic_image_dim(intr) != GLSL_SAMPLER_DIM_MS)
                  continue;
               msaa_stores++;
               /* Deref stores inherit their format from the image variable
                * when the driver lowers descriptors. */
               enum pipe_format format = nir_intrinsic_format(intr);
               if (format == PIPE_FORMAT_NONE)
                  format = nir_intrinsic_get_var(intr, 0)->data.image.format;
               printf("MSAA store format=%u array=%u\n",
                      format, nir_intrinsic_image_array(intr));
               if (require_fixed)
                  assert(format != PIPE_FORMAT_NONE);
            }
         }
      }
   }
   return create_compute(device, cache, count, infos, alloc, out);
}

static VKAPI_ATTR VkResult VKAPI_CALL
observe_view(VkDevice device, const VkImageViewCreateInfo *info,
             const VkAllocationCallbacks *alloc, VkImageView *out)
{
   VkResult result = create_view(device, info, alloc, out);
   CHECK(result);
   const struct hk_image *image = hk_image_from_handle(info->image);
   const struct hk_image_view *view = hk_image_view_from_handle(*out);
   const bool packed = image->vk.format == VK_FORMAT_B10G11R11_UFLOAT_PACK32 ||
                       image->vk.format == VK_FORMAT_E5B9G9R9_UFLOAT_PACK32;
   views++;
   if (packed) {
      packed_views++;
      if (require_fixed) {
         assert(info->format == VK_FORMAT_R32_UINT);
         assert(!image->planes[0].layout.compressed);
      }
   }
   if (image->vk.format == VK_FORMAT_D32_SFLOAT_S8_UINT) {
      if (info->subresourceRange.aspectMask == VK_IMAGE_ASPECT_DEPTH_BIT) {
         depth_views++;
         if (require_fixed) {
            assert(info->format == VK_FORMAT_R32_UINT);
            assert(view->planes[0].image_plane == 0);
         }
      } else if (info->subresourceRange.aspectMask ==
                 VK_IMAGE_ASPECT_STENCIL_BIT) {
         stencil_views++;
         if (require_fixed) {
            assert(info->format == VK_FORMAT_R8_UINT);
            assert(view->planes[0].image_plane == 1);
         }
      } else if (require_fixed) {
         assert(!"split depth/stencil copy lost its original aspect");
      }
   }
   printf("view image=%u format=%u aspect=%u plane=%u compressed=%u\n",
          image->vk.format, info->format, info->subresourceRange.aspectMask,
          view->planes[0].image_plane,
          image->planes[view->planes[0].image_plane].layout.compressed);
   return result;
}

struct bound_image {
   VkImage image;
   VkDeviceMemory memory;
};
static struct bound_image
make_image(VkDevice device, VkFormat format, VkImageType type,
           VkImageTiling tiling, unsigned samples)
{
   struct bound_image out;
   VkImageCreateInfo info = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
      .imageType = type,
      .format = format,
      .extent = {64, type == VK_IMAGE_TYPE_1D ? 1 : 64,
                 type == VK_IMAGE_TYPE_3D ? 4 : 1},
      .mipLevels = tiling == VK_IMAGE_TILING_LINEAR || samples > 1 ? 1 : 2,
      .arrayLayers = type == VK_IMAGE_TYPE_3D ? 1 : 2,
      .samples = samples,
      .tiling = tiling,
      .usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
               VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
   };
   CHECK(vkCreateImage(device, &info, NULL, &out.image));
   VkMemoryRequirements req;
   vkGetImageMemoryRequirements(device, out.image, &req);
   VkMemoryAllocateInfo alloc = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .allocationSize = req.size,
      .memoryTypeIndex = __builtin_ctz(req.memoryTypeBits)};
   CHECK(vkAllocateMemory(device, &alloc, NULL, &out.memory));
   CHECK(vkBindImageMemory(device, out.image, out.memory, 0));
   return out;
}

int
main(void)
{
   setbuf(stdout, NULL);
   require_fixed = getenv("EXPECT_FIXED") != NULL;
   VkApplicationInfo app = {.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
                            .apiVersion = VK_API_VERSION_1_3};
   VkInstanceCreateInfo ici = {.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
                               .pApplicationInfo = &app};
   VkInstance instance;
   CHECK(vkCreateInstance(&ici, NULL, &instance));
   uint32_t count = 1;
   VkPhysicalDevice physical;
   CHECK(vkEnumeratePhysicalDevices(instance, &count, &physical));
   assert(count == 1);
   float priority = 1;
   VkDeviceQueueCreateInfo queue = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
      .queueFamilyIndex = 0,
      .queueCount = 1,
      .pQueuePriorities = &priority};
   VkPhysicalDeviceVulkan13Features f13 = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES,
      .dynamicRendering = true};
   VkDeviceCreateInfo dci = {.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
                             .pNext = &f13,
                             .queueCreateInfoCount = 1,
                             .pQueueCreateInfos = &queue};
   VkDevice device;
   CHECK(vkCreateDevice(physical, &dci, NULL, &device));
   struct hk_device *dev = (void *)vk_device_from_handle(device);
   create_view = dev->vk.dispatch_table.CreateImageView;
   dev->vk.dispatch_table.CreateImageView = observe_view;
   create_compute = dev->vk.dispatch_table.CreateComputePipelines;
   dev->vk.dispatch_table.CreateComputePipelines = observe_compute;
   VkCommandPoolCreateInfo pci = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
      .queueFamilyIndex = 0,
      .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT};
   VkCommandPool pool;
   CHECK(vkCreateCommandPool(device, &pci, NULL, &pool));
   VkCommandBufferAllocateInfo cai = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
      .commandPool = pool,
      .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
      .commandBufferCount = 1};
   VkCommandBuffer cmd;
   CHECK(vkAllocateCommandBuffers(device, &cai, &cmd));
   VkBufferCreateInfo bci = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                             .size = 1 << 20,
                             .usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                                      VK_BUFFER_USAGE_TRANSFER_DST_BIT};
   VkBuffer buffer;
   CHECK(vkCreateBuffer(device, &bci, NULL, &buffer));
   VkMemoryRequirements req;
   vkGetBufferMemoryRequirements(device, buffer, &req);
   VkMemoryAllocateInfo alloc = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .allocationSize = req.size,
      .memoryTypeIndex = __builtin_ctz(req.memoryTypeBits)};
   VkDeviceMemory memory;
   CHECK(vkAllocateMemory(device, &alloc, NULL, &memory));
   CHECK(vkBindBufferMemory(device, buffer, memory, 0));

   const VkFormat formats[] = {VK_FORMAT_B10G11R11_UFLOAT_PACK32,
                               VK_FORMAT_E5B9G9R9_UFLOAT_PACK32,
                               VK_FORMAT_R32_UINT, VK_FORMAT_R8G8B8A8_UNORM};
   unsigned cases = 0;
   /* The whole-image CTS form uses one two-layer region, while _regions_
    * uses a separate one-layer view for each layer. Exercise both, plus a
    * single region starting at layer 1, at each supported MSAA count. */
   for (unsigned f = 0; f < ARRAY_SIZE(formats); ++f) {
      for (unsigned samples = 2; samples <= 4; samples *= 2) {
         struct bound_image a = make_image(device, formats[f], VK_IMAGE_TYPE_2D,
                                           VK_IMAGE_TILING_OPTIMAL, samples);
         struct bound_image b = make_image(device, formats[f], VK_IMAGE_TYPE_2D,
                                           VK_IMAGE_TILING_OPTIMAL, samples);
         for (unsigned shape = 0; shape < 3; ++shape) {
            VkImageCopy2 regions[2] = {{
               .sType = VK_STRUCTURE_TYPE_IMAGE_COPY_2,
               .srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, shape == 2,
                                  shape == 0 ? 2 : 1},
               .dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, shape == 2,
                                  shape == 0 ? 2 : 1},
               .extent = {64, 64, 1},
            }};
            regions[1] = regions[0];
            regions[1].srcSubresource.baseArrayLayer = 1;
            regions[1].dstSubresource.baseArrayLayer = 1;
            CHECK(vkResetCommandBuffer(cmd, 0));
            VkCommandBufferBeginInfo begin = {
               .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
            CHECK(vkBeginCommandBuffer(cmd, &begin));
            VkCopyImageInfo2 copy = {
               .sType = VK_STRUCTURE_TYPE_COPY_IMAGE_INFO_2,
               .srcImage = a.image,
               .srcImageLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
               .dstImage = b.image,
               .dstImageLayout = VK_IMAGE_LAYOUT_GENERAL,
               .regionCount = shape == 1 ? 2 : 1,
               .pRegions = regions,
            };
            vkCmdCopyImage2(cmd, &copy);
            CHECK(vkEndCommandBuffer(cmd));
            CHECK(vkResetCommandBuffer(cmd, 0));
            cases++;
         }
         vkDestroyImage(device, a.image, NULL);
         vkFreeMemory(device, a.memory, NULL);
         vkDestroyImage(device, b.image, NULL);
         vkFreeMemory(device, b.memory, NULL);
      }
   }
   if (getenv("ONLY_MSAA"))
      goto done;
   for (unsigned src = 0; src < 4; ++src) {
      for (unsigned dst = 0; dst < 4; ++dst) {
         for (unsigned shape = 0; shape < 5; ++shape) {
            VkImageType type = shape == 0   ? VK_IMAGE_TYPE_1D
                               : shape == 1 ? VK_IMAGE_TYPE_3D
                                            : VK_IMAGE_TYPE_2D;
            VkImageTiling tiling =
               shape == 2 ? VK_IMAGE_TILING_LINEAR : VK_IMAGE_TILING_OPTIMAL;
            struct bound_image a =
               make_image(device, formats[src], VK_IMAGE_TYPE_2D, tiling, 1);
            struct bound_image b =
               make_image(device, formats[dst], type, tiling, 1);
            CHECK(vkResetCommandBuffer(cmd, 0));
            VkCommandBufferBeginInfo begin = {
               .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
            CHECK(vkBeginCommandBuffer(cmd, &begin));
            const unsigned level = shape == 4;
            VkImageCopy2 region = {
               .sType = VK_STRUCTURE_TYPE_IMAGE_COPY_2,
               .srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, level, 0, 1},
               .dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, level, 0, 1},
               .srcOffset = {2, 0, 0},
               .dstOffset = {3, 0, 0},
               .extent = {17, type == VK_IMAGE_TYPE_1D ? 1 : 13, 1}};
            if (shape == 1)
               region.dstOffset.z = 1;
            if (shape == 3)
               region.srcSubresource.layerCount =
                  region.dstSubresource.layerCount = 2;
            VkCopyImageInfo2 copy = {
               .sType = VK_STRUCTURE_TYPE_COPY_IMAGE_INFO_2,
               .srcImage = a.image,
               .srcImageLayout = VK_IMAGE_LAYOUT_GENERAL,
               .dstImage = b.image,
               .dstImageLayout = VK_IMAGE_LAYOUT_GENERAL,
               .regionCount = 1,
               .pRegions = &region};
            vkCmdCopyImage2(cmd, &copy);
            VkBufferImageCopy2 br = {
               .sType = VK_STRUCTURE_TYPE_BUFFER_IMAGE_COPY_2,
               .bufferOffset = 256,
               .bufferRowLength = 40,
               .bufferImageHeight = 24,
               .imageSubresource = region.dstSubresource,
               .imageOffset = region.dstOffset,
               .imageExtent = region.extent};
            VkCopyBufferToImageInfo2 upload = {
               .sType = VK_STRUCTURE_TYPE_COPY_BUFFER_TO_IMAGE_INFO_2,
               .srcBuffer = buffer,
               .dstImage = b.image,
               .dstImageLayout = VK_IMAGE_LAYOUT_GENERAL,
               .regionCount = 1,
               .pRegions = &br};
            vkCmdCopyBufferToImage2(cmd, &upload);
            VkCopyImageToBufferInfo2 readback = {
               .sType = VK_STRUCTURE_TYPE_COPY_IMAGE_TO_BUFFER_INFO_2,
               .srcImage = b.image,
               .srcImageLayout = VK_IMAGE_LAYOUT_GENERAL,
               .dstBuffer = buffer,
               .regionCount = 1,
               .pRegions = &br};
            vkCmdCopyImageToBuffer2(cmd, &readback);
            CHECK(vkEndCommandBuffer(cmd));
            CHECK(vkResetCommandBuffer(cmd, 0));
            vkDestroyImage(device, a.image, NULL);
            vkFreeMemory(device, a.memory, NULL);
            vkDestroyImage(device, b.image, NULL);
            vkFreeMemory(device, b.memory, NULL);
            cases++;
         }
      }
   }
   for (unsigned samples = 2; samples <= 4; samples *= 2) {
      struct bound_image a =
         make_image(device, VK_FORMAT_D32_SFLOAT_S8_UINT, VK_IMAGE_TYPE_2D,
                    VK_IMAGE_TILING_OPTIMAL, samples);
      struct bound_image b =
         make_image(device, VK_FORMAT_D32_SFLOAT_S8_UINT, VK_IMAGE_TYPE_2D,
                    VK_IMAGE_TILING_OPTIMAL, samples);
      VkImageCopy2 regions[2] = {
         {.sType = VK_STRUCTURE_TYPE_IMAGE_COPY_2,
          .srcSubresource = {VK_IMAGE_ASPECT_DEPTH_BIT |
                                VK_IMAGE_ASPECT_STENCIL_BIT,
                             0, 0, 2},
          .dstSubresource = {VK_IMAGE_ASPECT_DEPTH_BIT |
                                VK_IMAGE_ASPECT_STENCIL_BIT,
                             0, 0, 2},
          .srcOffset = {1, 2, 0},
          .dstOffset = {3, 4, 0},
          .extent = {9, 11, 1}},
      };
      regions[1] = regions[0];
      regions[1].srcOffset.x += 16;
      regions[1].dstOffset.x += 16;
      CHECK(vkResetCommandBuffer(cmd, 0));
      VkCommandBufferBeginInfo begin = {
         .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
      CHECK(vkBeginCommandBuffer(cmd, &begin));
      VkCopyImageInfo2 copy = {.sType = VK_STRUCTURE_TYPE_COPY_IMAGE_INFO_2,
                               .srcImage = a.image,
                               .srcImageLayout = VK_IMAGE_LAYOUT_GENERAL,
                               .dstImage = b.image,
                               .dstImageLayout = VK_IMAGE_LAYOUT_GENERAL,
                               .regionCount = 2,
                               .pRegions = regions};
      vkCmdCopyImage2(cmd, &copy);
      CHECK(vkEndCommandBuffer(cmd));
      CHECK(vkResetCommandBuffer(cmd, 0));
      vkDestroyImage(device, a.image, NULL);
      vkFreeMemory(device, a.memory, NULL);
      vkDestroyImage(device, b.image, NULL);
      vkFreeMemory(device, b.memory, NULL);
      cases++;
   }
done:
   if (require_fixed) {
      assert(packed_views && msaa_stores);
      if (!getenv("ONLY_MSAA"))
         assert(depth_views && stencil_views);
   }
   dev->vk.dispatch_table.CreateImageView = create_view;
   dev->vk.dispatch_table.CreateComputePipelines = create_compute;
   vkDestroyBuffer(device, buffer, NULL);
   vkFreeMemory(device, memory, NULL);
   vkDestroyCommandPool(device, pool, NULL);
   vkDestroyDevice(device, NULL);
   vkDestroyInstance(instance, NULL);
   printf(
      "PASS %u recording cases; views=%u packed=%u depth=%u stencil=%u; no GPU execution\n",
      cases, views, packed_views, depth_views, stencil_views);
   return 0;
}
