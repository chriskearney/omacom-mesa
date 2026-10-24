/* Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */
#include <stdio.h>
#include <stdlib.h>
#include "compiler/nir/nir_builder.h"
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
check(bool fragment, bool null)
{
   nir_builder b = nir_builder_init_simple_shader(
      fragment ? MESA_SHADER_FRAGMENT : MESA_SHADER_COMPUTE, &agx_nir_options,
      "null AS must not touch native traversal context");
   b.shader->info.workgroup_size[0] = 32;
   b.shader->info.workgroup_size[1] = b.shader->info.workgroup_size[2] = 1;
   b.shader->info.ray_queries = 1;
   nir_variable *q =
      nir_local_variable_create(b.impl, glsl_uint64_t_type(), "query");
   q->data.ray_query = true;
   nir_def *deref = &nir_build_deref_var(&b, q)->def;
   nir_def *as = nir_load_preamble(&b, 1, 64, .base = 200);
   nir_rq_initialize(&b, deref, as, nir_imm_int(&b, 0), nir_imm_int(&b, 255),
                     nir_imm_vec3(&b, 0, 0, 0), nir_imm_float(&b, 0),
                     nir_imm_vec3(&b, 0, 0, -1), nir_imm_float(&b, 9));
   nir_def *more = nir_rq_proceed(&b, 1, deref);
   nir_push_if(&b, more);
   nir_rq_terminate(&b, deref);
   nir_pop_if(&b, NULL);
   nir_store_global(&b, nir_iadd_imm(&b, nir_b2i32(&b, more), 1),
                    nir_imm_int64(&b, 0x10008000000ull), .align_mul = 4);
   const struct agx_gen1_ray_query_layout layout = {
      .payload_uniform = 118,
      .bvh_uniform = 96,
      .status_uniform = 128,
      .context_uniform = 160,
      .as_table_uniform = 164,
      .instance_traversal = true,
      .candidates = true,
      .fragment_shader = fragment,
      .launch_header = 0x210000,
   };
   CHECK(agx_nir_lower_gen1_ray_query(b.shader, &layout));
   if (null) {
      nir_builder fix = nir_builder_at(nir_before_instr(nir_def_instr(as)));
      nir_def_rewrite_uses(as, nir_imm_int64(&fix, 0));
   }
   bool progress;
   do {
      progress = nir_lower_vars_to_ssa(b.shader);
      progress |= nir_opt_copy_prop(b.shader);
      progress |= nir_opt_constant_folding(b.shader);
      progress |= nir_opt_dead_cf(b.shader);
      progress |= nir_opt_dce(b.shader);
   } while (progress);
   unsigned native = 0, scs = 0, stores = 0;
   nir_foreach_block(block, b.impl) {
      nir_foreach_instr(instr, block) {
         if (instr->type != nir_instr_type_intrinsic)
            continue;
         nir_intrinsic_instr *i = nir_instr_as_intrinsic(instr);
         native += i->intrinsic == nir_intrinsic_intersect_ray_gen1_agx;
         scs += i->intrinsic == nir_intrinsic_store_scs_agx;
         if (i->intrinsic == nir_intrinsic_store_global &&
             nir_src_is_const(i->src[1]) &&
             nir_src_as_uint(i->src[1]) == 0x10008000000ull) {
            ++stores;
            if (null)
               CHECK(nir_src_is_const(i->src[0]) &&
                     nir_src_as_uint(i->src[0]) == 1);
         }
      }
   }
   CHECK(stores == 1);
   CHECK(null ? native == 0 && scs == 0 : native > 0 && scs > 0);
   nir_validate_shader(b.shader, "null AS lowering");
   ralloc_free(b.shader);
}
int
main(void)
{
   glsl_type_singleton_init_or_ref();
   check(false, false);
   check(false, true);
   check(true, false);
   check(true, true);
   glsl_type_singleton_decref();
   puts(
      "PASS: null descriptors skip native context and traversal; application store remains 1 (4 controls)");
}
