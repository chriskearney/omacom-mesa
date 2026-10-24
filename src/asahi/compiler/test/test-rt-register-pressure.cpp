/* Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */
#include <gtest/gtest.h>
#include "agx_test.h"
#include "agx_debug.h"

static void
check_texture_bank(bool g17p)
{
   void *mem = ralloc_context(NULL);
   agx_builder *b = agx_test_builder(mem);
   agx_shader_key key = {};
   key.dev.encoding = g17p ? AGX_ISA_ENCODING_AGX3B : AGX_ISA_ENCODING_AGX2;
   key.dev.chip = g17p ? AGX_CHIP_G17P : AGX_CHIP_G13G;
   key.has_scratch = true;
   b->shader->key = &key;
   b->shader->stage = MESA_SHADER_COMPUTE;
   b->shader->nir = rzalloc(mem, nir_shader);
   b->shader->nir->info.stage = MESA_SHADER_COMPUTE;
   b->shader->nir->info.workgroup_size[0] = 32;
   b->shader->nir->info.workgroup_size[1] = 1;
   b->shader->nir->info.workgroup_size[2] = 1;
   b->shader->any_cf = true;

   /* Match the live register layout at the failing callable's image store:
    * an unmovable prefix, scalar live-ins, then a texture coordinate in the
    * final partially free eight-half region of a 112-half file.
    */
   agx_index values[49];
   for (unsigned i = 0; i < 49; ++i)
      values[i] = agx_preload(b, agx_register(6 + 2 * i, AGX_SIZE_32));
   agx_index physical = agx_register(104, AGX_SIZE_32);
   physical.channels_m1 = 1;
   agx_index coords = agx_vec_temp(b->shader, AGX_SIZE_32, 2);
   agx_preload_to(b, coords, physical);
   agx_index data = agx_vec_temp(b->shader, AGX_SIZE_32, 4);
   agx_instr *collect = agx_collect_to(b, data, 4);
   collect->src[0] = values[18];
   for (unsigned c = 1; c < 4; ++c)
      collect->src[c] = agx_undef(AGX_SIZE_32);
   agx_image_write(b, data, coords, agx_zero(), agx_uniform(8, AGX_SIZE_64),
                   values[4], AGX_DIM_2D, false);
   for (unsigned i = 0; i < 49; ++i) {
      if (i != 18 && i != 4)
         agx_unit_test(b, values[i]);
   }
   int previous = agx_compiler_debug;
   agx_compiler_debug |= AGX_DBG_DEMAND;
   agx_ra(b->shader);
   agx_compiler_debug = previous;
   ASSERT_FALSE(b->shader->ra_failed);
   EXPECT_LT(b->shader->max_reg, 112u);
   agx_foreach_instr_global(b->shader, I) {
      if (I->op == AGX_OPCODE_IMAGE_WRITE) {
         EXPECT_LT(I->src[0].value + 7, 128u);
         EXPECT_LT(I->src[1].value + 3, 128u);
      }
   }
   ralloc_free(mem);
}

TEST(AGX3RegisterPressure, TextureBankContainsTheWholeFile)
{
   check_texture_bank(true);
}

TEST(AGX3RegisterPressure, ClassicControl)
{
   check_texture_bank(false);
}
