/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

/* Every check in this test is an assert(); keep them in NDEBUG builds. */
#undef NDEBUG

#include <assert.h>
#include <stdio.h>

#include "agx_chip.h"

static struct drm_asahi_params_global
g15_params(char variant, uint32_t chip_id, uint32_t clusters)
{
   struct drm_asahi_params_global params = {
      .gpu_generation = 15,
      .gpu_variant = variant,
      .chip_id = chip_id,
      .num_dies = 1,
      .num_clusters_total = clusters,
      .num_cores_per_cluster = 10,
      .usc_generation = 3,
      .gpu_hal_generation = DRM_ASAHI_GPU_HAL_LEGACY,
   };

   for (unsigned i = 0; i < clusters; ++i)
      params.core_masks[i] = 0x3ff;
   return params;
}

static void
check_native_g15(enum agx_chip chip)
{
   assert(agx_chip_native_agx3(chip));
   assert(agx_chip_native_g15(chip));
   assert(agx_chip_native_imageblock(chip));
   assert(agx_chip_native_indirect_compute(chip));
   assert(agx_chip_driver_supported(chip));
   assert(agx_chip_usc_helper_table_slots(chip) == 10);
}

int
main(void)
{
   check_native_g15(AGX_CHIP_G15S);
   check_native_g15(AGX_CHIP_G15C);

   /* The other M3 parts stay off the native path. */
   assert(!agx_chip_native_agx3(AGX_CHIP_G15G));
   assert(!agx_chip_native_agx3(AGX_CHIP_G15D));
   assert(!agx_chip_driver_supported(AGX_CHIP_G15G));
   assert(!agx_chip_driver_supported(AGX_CHIP_G15D));

   /* T6030 and T6031 are each admitted with their own identity only. */
   struct drm_asahi_params_global g15s = g15_params('S', 0x6030, 2);
   struct drm_asahi_params_global g15c = g15_params('C', 0x6031, 4);
   assert(agx_chip_from_params(&g15s) == AGX_CHIP_G15S);
   assert(agx_chip_from_params(&g15c) == AGX_CHIP_G15C);
   assert(agx_params_driver_supported(&g15s));
   assert(agx_params_driver_supported(&g15c));

   /* A binned part reports fewer cores in its masks and is still admitted. */
   struct drm_asahi_params_global binned = g15c;
   binned.core_masks[0] = binned.core_masks[3] = 0x1ff;
   assert(agx_params_driver_supported(&binned));

   struct drm_asahi_params_global bad = g15c;
   bad.chip_id = 0x6030;
   assert(!agx_params_driver_supported(&bad));

   bad = g15c;
   bad.num_clusters_total = 2;
   assert(!agx_params_driver_supported(&bad));

   bad = g15c;
   bad.num_dies = 2;
   assert(!agx_params_driver_supported(&bad));

   bad = g15c;
   bad.gpu_hal_generation = DRM_ASAHI_GPU_HAL_200;
   assert(!agx_params_driver_supported(&bad));

   bad = g15c;
   bad.usc_generation = 2;
   assert(!agx_params_driver_supported(&bad));

   bad = g15s;
   bad.chip_id = 0x6031;
   assert(!agx_params_driver_supported(&bad));

   bad = g15s;
   bad.num_clusters_total = 4;
   assert(!agx_params_driver_supported(&bad));

   /* The same GPU identity must not admit the incompatible legacy payloads. */
   assert(!agx_params_native_kernel_abi_supported(NULL));
   assert(!agx_params_native_kernel_abi_supported(&g15s));
   const uint64_t native = DRM_ASAHI_FEATURE_PER_COMMAND_INPUT_SYNCS |
                           DRM_ASAHI_FEATURE_EXACT_PRIOR_BARRIERS;
   g15s.features = native;
   g15c.features = native;
   assert(agx_params_native_kernel_abi_supported(&g15s));
   assert(agx_params_native_kernel_abi_supported(&g15c));
   for (unsigned bit = 10; bit <= 15; ++bit) {
      bad = g15c;
      bad.features |= UINT64_C(1) << bit;
      assert(!agx_params_native_kernel_abi_supported(&bad));
   }
   bad = g15c;
   bad.features &= ~DRM_ASAHI_FEATURE_EXACT_PRIOR_BARRIERS;
   assert(!agx_params_native_kernel_abi_supported(&bad));
   bad.features = DRM_ASAHI_FEATURE_EXACT_PRIOR_BARRIERS;
   assert(!agx_params_native_kernel_abi_supported(&bad));
   bad = (struct drm_asahi_params_global){
      .gpu_generation = 13, .gpu_variant = 'G', .num_clusters_total = 1,
   };
   assert(agx_params_native_kernel_abi_supported(&bad));

   printf("chip admission: ok\n");
   return 0;
}
