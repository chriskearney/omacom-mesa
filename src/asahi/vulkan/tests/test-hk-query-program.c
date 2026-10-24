/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include "asahi/compiler/agx_compile.h"
#include "libagx.h"
#include "libagx_shaders.h"

#undef NDEBUG
#include <assert.h>
#include <stdio.h>

int
main(void)
{
   const struct agx_precompiled_kernel_info *info =
      (const void *)libagx_g17p[LIBAGX_COPY_QUERY_G17P];
   assert(info);
   assert(info->main_size && info->nr_gprs);
   assert(info->main_offset <= info->binary_size);
   assert(info->main_size <= info->binary_size - info->main_offset);
   assert(info->agx3_staging_known);
   assert(info->nr_preamble_gprs == 0 && info->preamble_size == 0);
   assert(info->workgroup_size[0] == 32);
   assert(info->workgroup_size[1] == 1 && info->workgroup_size[2] == 1);
   assert(info->printf_uniform == 0);
   assert(libagx_g13g[LIBAGX_COPY_QUERY_G17P] == NULL);
   assert(libagx_g13x[LIBAGX_COPY_QUERY_G17P] == NULL);
   assert(libagx_g13g[LIBAGX_COPY_QUERY] != NULL);
   assert(libagx_g13x[LIBAGX_COPY_QUERY] != NULL);
   puts("query range program: native launch metadata and G13 exclusion passed");
   return 0;
}
