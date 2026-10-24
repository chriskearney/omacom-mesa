/* Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include "compiler/nir/nir_builder.h"
#include "vulkan/runtime/bvh/vk_bvh_defines.h"
#include "agx_compile.h"
#include "agx_nir.h"
#include "agx_nir_gen1_ray_query.h"
#define CHECK(x)                                                               \
   do {                                                                        \
      if (!(x)) {                                                              \
         fprintf(stderr, "%d: %s\n", __LINE__, #x);                            \
         abort();                                                              \
      }                                                                        \
   } while (0)
static void
check(uint32_t id, uint32_t geometry, uint32_t primitive)
{
   const uint64_t instance = 0x10008000000ull, blas = 0x10009000000ull,
                  table = 0x1000a000000ull;
   const uint32_t bases[] = {0, 3, 4, 9};
   const uint32_t vertices = 32;
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, &agx_nir_options, "vertex table address");
   nir_def *addr = agx_nir_gen1_vertex_position_address(
      &b, nir_imm_int64(&b, instance), nir_imm_int(&b, geometry),
      nir_imm_int(&b, primitive));
   nir_store_global(&b, addr, nir_imm_int64(&b, 0), .align_mul = 8);
   unsigned reads = 0;
   bool progress;
   do {
      progress = nir_opt_constant_folding(b.shader);
      nir_foreach_block(block, b.impl) {
         nir_foreach_instr_safe(instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;
            nir_intrinsic_instr *i = nir_instr_as_intrinsic(instr);
            if (i->intrinsic != nir_intrinsic_load_global ||
                !nir_src_is_const(i->src[0]))
               continue;
            uint64_t at = nir_src_as_uint(i->src[0]), value;
            if (at == instance + 88)
               value = blas;
            else if (at == blas + 176)
               value = id;
            else if (at == AGX_GEN1_VERTEX_DIRECTORY_VA + (uint64_t)id * 16)
               value = table;
            else if (at == table + 8)
               value = vertices;
            else if (at == table + 16 + geometry * 4)
               value = bases[geometry];
            else {
               fprintf(stderr, "unexpected read at %llx for id %x\n",
                       (unsigned long long)at, id);
               abort();
            }
            nir_builder f = nir_builder_at(nir_before_instr(instr));
            nir_def_rewrite_uses(&i->def,
                                 nir_imm_intN_t(&f, value, i->def.bit_size));
            nir_instr_remove(instr);
            ++reads;
            progress = true;
         }
      }
   } while (progress);
   nir_intrinsic_instr *store =
      nir_instr_as_intrinsic(nir_block_last_instr(nir_start_block(b.impl)));
   CHECK(reads == 5 && nir_src_is_const(store->src[0]));
   CHECK(nir_src_as_uint(store->src[0]) ==
         table + vertices + 36ull * (bases[geometry] + primitive));
   /* The sidecar producer copies all nine scalar coordinates, with no
    * dependence on input vertex/index stride after common IR conversion. */
   CHECK(offsetof(struct vk_ir_triangle_node, coords) == 24);
   CHECK(offsetof(struct vk_ir_triangle_node, triangle_id) == 60);
   CHECK(offsetof(struct vk_ir_triangle_node, geometry_id_and_flags) == 64);
   CHECK(sizeof(((struct vk_ir_triangle_node *)0)->coords) == 36);
   ralloc_free(b.shader);
}
int
main(void)
{
   glsl_type_singleton_init_or_ref();
   const uint32_t ids[] = {1, 2047, 2048, 2049, 65535, 1048575};
   for (unsigned i = 0; i < 6; i++)
      for (unsigned g = 0; g < 4; g++)
         for (unsigned p = 0; p < 3; p++)
            check(ids[i], g, p);
   glsl_type_singleton_decref();
   puts("PASS: 72 production vertex-table address cases and IR coordinate ABI");
}
