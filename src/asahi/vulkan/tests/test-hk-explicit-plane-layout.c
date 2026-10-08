/* SPDX-License-Identifier: MIT */
#include "../hk_image.c"
#include "vk_instance.h"

static struct hk_device dev;
static struct hk_physical_device pdev;
static struct vk_instance instance;
static unsigned char pixels[131072];

static VkResult
make_image(struct hk_image *image, VkFormat format, bool disjoint,
           unsigned plane_count, VkSubresourceLayout *planes)
{
   *image = (struct hk_image){0};
   VkImageDrmFormatModifierExplicitCreateInfoEXT modifier = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_DRM_FORMAT_MODIFIER_EXPLICIT_CREATE_INFO_EXT,
      .drmFormatModifier = DRM_FORMAT_MOD_LINEAR,
      .drmFormatModifierPlaneCount = plane_count,
      .pPlaneLayouts = planes,
   };
   VkImageCreateInfo info = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
      .pNext = planes ? &modifier : NULL,
      .flags = disjoint ? VK_IMAGE_CREATE_DISJOINT_BIT : 0,
      .imageType = VK_IMAGE_TYPE_2D,
      .format = format,
      .extent = {64, 32, 1},
      .mipLevels = 1,
      .arrayLayers = 1,
      .samples = VK_SAMPLE_COUNT_1_BIT,
      .tiling = planes ? VK_IMAGE_TILING_DRM_FORMAT_MODIFIER_EXT
                       : VK_IMAGE_TILING_LINEAR,
      .usage = VK_IMAGE_USAGE_SAMPLED_BIT,
   };
   return hk_image_init(&dev, image, &info);
}

static void
finish_image(struct hk_image *image)
{
   vk_object_base_finish(&image->vk.base);
}

static void
check_plane(struct hk_image *image, unsigned p, uint64_t bind_base,
            uint64_t expected_offset, unsigned char expected_byte)
{
   struct agx_va va = {.addr = 0x100000};
   struct agx_bo bo = {
      .va = &va, .size = sizeof(pixels), ._map = pixels,
      .flags = AGX_BO_SHARED,
   };
   struct hk_device_memory memory = {.bo = &bo};
   struct hk_image_plane *plane = &image->planes[p];
   uint64_t offset = bind_base;
   hk_image_plane_bind(&dev, plane, &memory, &offset);
   uint64_t bound = bind_base +
      (image->vk.tiling == VK_IMAGE_TILING_DRM_FORMAT_MODIFIER_EXT
         ? expected_offset : 0);
   assert(plane->addr == va.addr + bound);
   assert(plane->map == pixels + bound);
   assert(plane->rem == sizeof(pixels) - bound);
   assert(!plane->private_binding);
   assert(*(unsigned char *)plane->map == expected_byte);
   if (plane->explicit_layout)
      assert(offset == bind_base);

   VkImageSubresource2KHR subresource = {
      .imageSubresource = {
         .aspectMask = VK_IMAGE_ASPECT_MEMORY_PLANE_0_BIT_EXT << p,
      },
   };
   VkSubresourceLayout2KHR layout = {0};
   hk_get_image_subresource_layout(&dev, image, &subresource, &layout);
   assert(layout.subresourceLayout.offset == expected_offset);
   assert(layout.subresourceLayout.rowPitch == plane->layout.linear_stride_B);
   assert(layout.subresourceLayout.size == plane->layout.size_B);
}

static void
check_explicit(VkFormat format, bool disjoint, unsigned count,
               uint64_t offset0, uint64_t offset1)
{
   VkSubresourceLayout layouts[2] = {
      {.offset = offset0, .rowPitch = 128},
      {.offset = offset1, .rowPitch = 128},
   };
   struct hk_image image;
   assert(make_image(&image, format, disjoint, count, layouts) == VK_SUCCESS);
   memset(pixels, 0x40, sizeof(pixels));
   uint64_t max_end = 0;
   for (unsigned p = 0; p < count; p++) {
      uint64_t off = layouts[p].offset;
      memset(pixels + 4096 + off, 0xc0 + p, image.planes[p].layout.size_B);
      check_plane(&image, p, 4096, off, 0xc0 + p);
      max_end = MAX2(max_end, off + image.planes[p].layout.size_B);
   }
   VkMemoryRequirements2 reqs = {0};
   if (!disjoint) {
      hk_get_image_memory_requirements(&dev, &image, image.vk.aspects, &reqs);
      assert(reqs.memoryRequirements.size == max_end);
      assert(reqs.memoryRequirements.alignment == HK_PLANE_ALIGN_B);
   } else {
      for (unsigned p = 0; p < count; p++) {
         hk_get_image_memory_requirements(&dev, &image,
            VK_IMAGE_ASPECT_MEMORY_PLANE_0_BIT_EXT << p, &reqs);
         assert(reqs.memoryRequirements.size ==
                layouts[p].offset + image.planes[p].layout.size_B);
      }
   }
   finish_image(&image);
}

static void
check_invalid(unsigned count, uint64_t offset, uint64_t stride)
{
   VkSubresourceLayout layouts[2] = {
      {.offset = offset, .rowPitch = stride},
      {.offset = 32768, .rowPitch = 128},
   };
   struct hk_image image;
   assert(make_image(&image, VK_FORMAT_R8_UNORM, false, count, layouts) ==
          VK_ERROR_INVALID_DRM_FORMAT_MODIFIER_PLANE_LAYOUT_EXT);
   finish_image(&image);
}

static void
check_ordinary(void)
{
   struct hk_image image;
   assert(make_image(&image, VK_FORMAT_G8_B8R8_2PLANE_420_UNORM,
                     false, 2, NULL) == VK_SUCCESS);
   uint64_t size = 0;
   uint32_t align = 0;
   hk_image_plane_add_req(&image.planes[0], false, &size, &align);
   uint64_t plane1 = align64(size, HK_PLANE_ALIGN_B);
   hk_image_plane_add_req(&image.planes[1], false, &size, &align);
   assert(size == plane1 + image.planes[1].layout.size_B);
   memset(pixels, 0x40, sizeof(pixels));
   check_plane(&image, 0, 0, 0, 0x40);
   check_plane(&image, 1, plane1, plane1, 0x40);
   finish_image(&image);
}

int main(void)
{
   dev.dev.chip = AGX_CHIP_G15S;
   dev.vk.physical = &pdev.vk;
   dev.vk.base.type = VK_OBJECT_TYPE_DEVICE;
   dev.vk.base.device = &dev.vk;
   pdev.vk.instance = &instance;
   list_inithead(&instance.debug_utils.callbacks);
   list_inithead(&instance.debug_report.callbacks);
   pdev.mem_type_count = 1;
   check_explicit(VK_FORMAT_R8_UNORM, false, 1, 32768, 0);
   check_explicit(VK_FORMAT_R8G8_UNORM, false, 1, 32768, 0);
   check_explicit(VK_FORMAT_G8_B8R8_2PLANE_420_UNORM, false, 2, 0, 32768);
   check_explicit(VK_FORMAT_G8_B8R8_2PLANE_420_UNORM, false, 2, 32768, 0);
   check_explicit(VK_FORMAT_G8_B8R8_2PLANE_420_UNORM, true, 2, 0, 32768);
   check_invalid(2, 0, 128);
   check_invalid(1, 1, 128);
   check_invalid(1, UINT64_MAX - 127, 128);
   check_invalid(1, 0, (uint64_t)UINT32_MAX + 1);
   check_invalid(1, 0, 65);
   check_invalid(1, 0, 16);
   check_ordinary();
   puts("explicit plane layout controls passed");
   return 0;
}
