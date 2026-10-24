/* Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */
#include "agx_compile.h"
#include "agx_nir.h"
#include "agx_nir_gen1_ray_query.h"
#include "compiler/nir/nir_builder.h"
#include <stdio.h>
#include <stdlib.h>

#define CHECK(x) do { if (!(x)) { \
   fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #x); abort(); \
} } while (0)

static void
initialize(nir_builder *b, nir_def *query)
{
   nir_rq_initialize(b, query, nir_imm_int64(b, 0x10003000000ull),
      nir_imm_int(b, 0), nir_imm_int(b, 255), nir_imm_vec3(b, 0, 0, 1),
      nir_imm_float(b, 0.0001), nir_imm_vec3(b, 0, 0, -1),
      nir_imm_float(b, 9));
}

static void
check_terminate(mesa_shader_stage stage, bool graphics, bool array,
                bool proceed_first, bool reinitialize)
{
   nir_builder b = nir_builder_init_simple_shader(stage, &agx_nir_options,
      "Gen1 termination state");
   b.shader->info.workgroup_size[0] = 32;
   b.shader->info.workgroup_size[1] = b.shader->info.workgroup_size[2] = 1;
   const struct glsl_type *type = array ?
      glsl_array_type(glsl_uint64_t_type(), 8, 0) : glsl_uint64_t_type();
   nir_variable *query = nir_local_variable_create(b.impl, type, "queries");
   query->data.ray_query = true;
   nir_deref_instr *deref = nir_build_deref_var(&b, query);
   if (array)
      deref = nir_build_deref_array_imm(&b, deref, 4);
   nir_def *rq = &deref->def;
   initialize(&b, rq);
   if (proceed_first)
      nir_store_global(&b, nir_b2i32(&b, nir_rq_proceed(&b, 1, rq)),
         nir_imm_int64(&b, 0x10004000000ull), .align_mul = 4, .write_mask = 1);
   nir_rq_terminate(&b, rq);
   nir_store_global(&b, nir_b2i32(&b, nir_rq_proceed(&b, 1, rq)),
      nir_imm_int64(&b, 0x10004000004ull), .align_mul = 4, .write_mask = 1);
   if (reinitialize) {
      initialize(&b, rq);
      nir_store_global(&b, nir_b2i32(&b, nir_rq_proceed(&b, 1, rq)),
         nir_imm_int64(&b, 0x10004000008ull), .align_mul = 4, .write_mask = 1);
   }
   nir_shader_gather_info(b.shader, b.impl);
   struct agx_gen1_ray_query_layout layout = {
      .payload_uniform = 118, .bvh_uniform = 96, .status_uniform = 128,
      .context_uniform = 160, .as_table_uniform = 164, .as_table_first = 1,
      .launch_header = 0x210000, .instance_traversal = true,
      .candidates = true, .graphics_stage = graphics, .compute_kernel = true,
   };
   CHECK(agx_nir_lower_gen1_ray_query(b.shader, &layout));
   bool progress;
   do {
      progress = false;
      NIR_PASS(progress, b.shader, nir_lower_vars_to_ssa);
      NIR_PASS(progress, b.shader, nir_opt_copy_prop);
      NIR_PASS(progress, b.shader, nir_opt_constant_folding);
      NIR_PASS(progress, b.shader, nir_opt_dead_cf);
      NIR_PASS(progress, b.shader, nir_opt_dce);
   } while (progress);
   unsigned launches = 0, terminated_stores = 0;
   nir_foreach_block(block, b.impl) {
      nir_foreach_instr(instr, block) {
         if (instr->type != nir_instr_type_intrinsic)
            continue;
         nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
         CHECK(!nir_intrinsic_is_ray_query(intr->intrinsic));
         launches += intr->intrinsic == nir_intrinsic_intersect_ray_gen1_agx;
         if (intr->intrinsic == nir_intrinsic_store_global &&
             nir_src_is_const(intr->src[1]) &&
             nir_src_as_uint(intr->src[1]) == 0x10004000004ull) {
            CHECK(nir_src_is_const(intr->src[0]));
            CHECK(nir_src_as_uint(intr->src[0]) == 0);
            ++terminated_stores;
         }
      }
   }
   CHECK(terminated_stores == 1);
   CHECK(launches == (unsigned)proceed_first + reinitialize);
   CHECK(b.shader->info.ray_queries == 0);
   nir_validate_shader(b.shader, "termination controls");
   ralloc_free(b.shader);
}

int main(void)
{
   glsl_type_singleton_init_or_ref();
   const mesa_shader_stage stages[] = {MESA_SHADER_COMPUTE,
      MESA_SHADER_TESS_CTRL, MESA_SHADER_TESS_EVAL, MESA_SHADER_VERTEX};
   unsigned cases = 0;
   for (unsigned s = 0; s < ARRAY_SIZE(stages); ++s)
      for (unsigned graphics = 0; graphics < 2; ++graphics)
         for (unsigned array = 0; array < 2; ++array)
            for (unsigned first = 0; first < 2; ++first)
               for (unsigned reinit = 0; reinit < 2; ++reinit) {
                  check_terminate(stages[s], graphics, array, first, reinit);
                  ++cases;
               }
   glsl_type_singleton_decref();
   printf("Gen1 termination controls: %u passed\n", cases);
   return 0;
}
