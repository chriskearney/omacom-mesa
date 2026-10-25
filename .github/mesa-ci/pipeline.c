/* SPDX-License-Identifier: MIT
 * Compile real SPIR-V through the complete driver pipeline without submitting work.
 * Cover float->int64 and perspective offset interpolation introduced after NIR prep.
 */
#include <vulkan/vulkan.h>
#include <dlfcn.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define REQUIRE(x) do { if (!(x)) { fprintf(stderr, "line %u: %s\n", __LINE__, #x); exit(1); } } while (0)
#define OK(x) REQUIRE((x) == VK_SUCCESS)

static VkShaderModule module(VkDevice dev, const char *path)
{
   FILE *f = fopen(path, "rb");
   REQUIRE(f);
   REQUIRE(fseek(f, 0, SEEK_END) == 0);
   long size = ftell(f);
   REQUIRE(size > 0 && size % 4 == 0);
   rewind(f);
   uint32_t *code = malloc(size);
   REQUIRE(code && fread(code, size, 1, f) == 1);
   REQUIRE(fclose(f) == 0);
   VkShaderModuleCreateInfo ci = {
      .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
      .codeSize = size, .pCode = code,
   };
   VkShaderModule result;
   OK(vkCreateShaderModule(dev, &ci, NULL, &result));
   free(code);
   return result;
}

int main(int argc, char **argv)
{
   REQUIRE(argc == 4);
   REQUIRE(dlsym(RTLD_DEFAULT, "asahi_noop_is_registered_fd") != NULL);
   VkInstance instance;
   VkApplicationInfo app = {.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
                           .pApplicationName = "mesa-ci-compiler-regression",
                           .apiVersion = VK_API_VERSION_1_3};
   VkInstanceCreateInfo ici = {.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
                               .pApplicationInfo = &app};
   OK(vkCreateInstance(&ici, NULL, &instance));
   uint32_t count = 1;
   VkPhysicalDevice physical;
   OK(vkEnumeratePhysicalDevices(instance, &count, &physical));
   REQUIRE(count == 1);
   VkPhysicalDeviceFeatures features;
   vkGetPhysicalDeviceFeatures(physical, &features);
   REQUIRE(features.shaderInt64 && features.sampleRateShading);
   count = 0;
   vkGetPhysicalDeviceQueueFamilyProperties(physical, &count, NULL);
   VkQueueFamilyProperties *queues = calloc(count, sizeof(*queues));
   REQUIRE(queues);
   vkGetPhysicalDeviceQueueFamilyProperties(physical, &count, queues);
   uint32_t family;
   for (family = 0; family < count; family++)
      if ((queues[family].queueFlags & (VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT)) ==
          (VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT)) break;
   REQUIRE(family < count);
   free(queues);
   float priority = 1.0f;
   VkDeviceQueueCreateInfo q = {.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
      .queueFamilyIndex = family, .queueCount = 1, .pQueuePriorities = &priority};
   VkPhysicalDeviceVulkan13Features f13 = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES,
      .dynamicRendering = VK_TRUE};
   features = (VkPhysicalDeviceFeatures){.shaderInt64 = VK_TRUE, .sampleRateShading = VK_TRUE};
   VkDeviceCreateInfo dci = {.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
      .pNext = &f13, .queueCreateInfoCount = 1, .pQueueCreateInfos = &q,
      .pEnabledFeatures = &features};
   VkDevice dev;
   OK(vkCreateDevice(physical, &dci, NULL, &dev));
   VkDescriptorSetLayoutBinding binding = {.binding = 0,
      .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 1,
      .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT};
   VkDescriptorSetLayoutCreateInfo sci = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
      .bindingCount = 1, .pBindings = &binding};
   VkDescriptorSetLayout set;
   OK(vkCreateDescriptorSetLayout(dev, &sci, NULL, &set));
   VkPipelineLayoutCreateInfo lci = {.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
      .setLayoutCount = 1, .pSetLayouts = &set};
   VkPipelineLayout layout;
   OK(vkCreatePipelineLayout(dev, &lci, NULL, &layout));
   VkShaderModule comp = module(dev, argv[1]);
   VkPipelineShaderStageCreateInfo cs = {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
      .stage = VK_SHADER_STAGE_COMPUTE_BIT, .module = comp, .pName = "main"};
   VkComputePipelineCreateInfo cci = {.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
      .stage = cs, .layout = layout};
   VkPipeline compute;
   OK(vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &cci, NULL, &compute));
   VkShaderModule vert = module(dev, argv[2]), frag = module(dev, argv[3]);
   VkPipelineShaderStageCreateInfo stages[] = {
      {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
       .stage = VK_SHADER_STAGE_VERTEX_BIT, .module = vert, .pName = "main"},
      {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
       .stage = VK_SHADER_STAGE_FRAGMENT_BIT, .module = frag, .pName = "main"}};
   VkPipelineVertexInputStateCreateInfo vi = {.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
   VkPipelineInputAssemblyStateCreateInfo ia = {.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
      .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST};
   VkViewport viewport = {.width = 1, .height = 1, .maxDepth = 1};
   VkRect2D scissor = {.extent = {1, 1}};
   VkPipelineViewportStateCreateInfo vp = {.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
      .viewportCount = 1, .pViewports = &viewport, .scissorCount = 1, .pScissors = &scissor};
   VkPipelineRasterizationStateCreateInfo rs = {.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
      .polygonMode = VK_POLYGON_MODE_FILL, .cullMode = VK_CULL_MODE_NONE, .lineWidth = 1};
   VkPipelineMultisampleStateCreateInfo ms = {.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
      .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT};
   VkPipelineColorBlendAttachmentState ba = {.colorWriteMask = 0xf};
   VkPipelineColorBlendStateCreateInfo bs = {.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
      .attachmentCount = 1, .pAttachments = &ba};
   VkFormat format = VK_FORMAT_R8G8B8A8_UNORM;
   VkPipelineRenderingCreateInfo render = {.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO,
      .colorAttachmentCount = 1, .pColorAttachmentFormats = &format};
   VkGraphicsPipelineCreateInfo gci = {.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
      .pNext = &render, .stageCount = 2, .pStages = stages, .pVertexInputState = &vi,
      .pInputAssemblyState = &ia, .pViewportState = &vp, .pRasterizationState = &rs,
      .pMultisampleState = &ms, .pColorBlendState = &bs, .layout = layout};
   VkPipeline graphics;
   OK(vkCreateGraphicsPipelines(dev, VK_NULL_HANDLE, 1, &gci, NULL, &graphics));
   vkDestroyPipeline(dev, graphics, NULL);
   vkDestroyPipeline(dev, compute, NULL);
   vkDestroyShaderModule(dev, frag, NULL);
   vkDestroyShaderModule(dev, vert, NULL);
   vkDestroyShaderModule(dev, comp, NULL);
   vkDestroyPipelineLayout(dev, layout, NULL);
   vkDestroyDescriptorSetLayout(dev, set, NULL);
   vkDestroyDevice(dev, NULL);
   vkDestroyInstance(instance, NULL);
   puts("PASS: float-to-int64 and perspective-offset pipelines compiled; zero submissions");
   return 0;
}
