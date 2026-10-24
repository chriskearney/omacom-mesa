/* Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */
#include "hk_nir_root_size.h"
#include "nir_builder.h"
#undef NDEBUG
#include <assert.h>
#include <stdio.h>

static nir_builder
builder(void)
{
   return nir_builder_init_simple_shader(MESA_SHADER_COMPUTE, NULL, "root-size");
}

static void
load(nir_builder *b, nir_def *addr, unsigned components, unsigned bits)
{
   nir_load_global_constant(b, components, bits, addr, .align_mul = bits / 8);
}

int main(void)
{
   const unsigned full = 8192;
   nir_builder b = builder();
   assert(hk_nir_root_size(b.shader, full) == 0);
   load(&b, nir_iadd_imm(&b, nir_load_root_agx(&b), 832), 4, 32);
   assert(hk_nir_root_size(b.shader, full) == 848);
   load(&b, nir_iadd_imm(&b, nir_load_root_agx(&b), 40), 1, 64);
   assert(hk_nir_root_size(b.shader, full) == 848);
   ralloc_free(b.shader);

   b = builder();
   nir_def *addr = nir_iadd_imm(&b, nir_load_root_agx(&b), 32);
   load(&b, nir_iadd_imm(&b, addr, 64), 2, 32);
   assert(hk_nir_root_size(b.shader, full) == 104);
   ralloc_free(b.shader);

   b = builder();
   addr = nir_iadd(&b, nir_load_root_agx(&b), nir_u2u64(&b, nir_load_local_invocation_index(&b)));
   load(&b, addr, 1, 32);
   assert(hk_nir_root_size(b.shader, full) == full);
   ralloc_free(b.shader);

   b = builder();
   nir_store_global(&b, nir_imm_int(&b, 0), nir_load_root_agx(&b), .align_mul = 4, .write_mask = 1);
   assert(hk_nir_root_size(b.shader, full) == full);
   ralloc_free(b.shader);

   for (int offset = -1; offset <= (int)full; offset += full + 1) {
      b = builder();
      load(&b, nir_iadd_imm(&b, nir_load_root_agx(&b), offset), 4, 32);
      assert(hk_nir_root_size(b.shader, full) == full);
      ralloc_free(b.shader);
   }
   b = builder();
   load(&b, nir_iadd_imm(&b, nir_load_root_agx(&b), full - 4), 4, 32);
   assert(hk_nir_root_size(b.shader, full) == full);
   ralloc_free(b.shader);

   b = builder();
   addr = nir_load_root_agx(&b);
   for (unsigned i = 0; i < 10; ++i)
      addr = nir_iadd_imm(&b, addr, 8);
   load(&b, addr, 1, 32);
   assert(hk_nir_root_size(b.shader, full) == full);
   ralloc_free(b.shader);
   b = builder();
   nir_def *reflected = nir_load_global_constant(&b, 1, 64,
      nir_load_root_agx(&b), .align_mul = 8);
   load(&b, nir_iadd_imm(&b, reflected, 4096), 1, 32);
   assert(hk_nir_root_size(b.shader, full) == full);
   ralloc_free(b.shader);
   puts("root footprint: constant, dynamic, escape, overflow, depth and reflection controls passed");
   return 0;
}
