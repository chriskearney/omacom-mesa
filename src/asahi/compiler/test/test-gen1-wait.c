/* Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */
#include "agx_nir_ray_query.h"
#include "compiler/nir/nir_builder.h"
#include "compiler/glsl_types.h"
#undef NDEBUG
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

/* Execute the actual wait helper's scalar NIR control flow. Only the native
 * status/continue operations are modeled: completion can be delayed past the
 * former 65,536-poll cutoff. This is not a GPU scheduling simulator. */
struct execution {
   uint64_t *ssa;
   uint64_t vars[16];
   unsigned resumes, delay;
   bool pending;
};

static uint64_t src(struct execution *e, nir_src s) { return e->ssa[s.ssa->index]; }

static bool
run_list(struct execution *e, struct exec_list *list)
{
   foreach_list_typed(nir_cf_node, node, node, list) {
      if (node->type == nir_cf_node_if) {
         nir_if *nif = nir_cf_node_as_if(node);
         if (run_list(e, src(e, nif->condition) ? &nif->then_list : &nif->else_list))
            return true;
      } else if (node->type == nir_cf_node_loop) {
         nir_loop *loop = nir_cf_node_as_loop(node);
         while (!run_list(e, &loop->body))
            assert(e->resumes <= e->delay);
      } else {
         assert(node->type == nir_cf_node_block);
         nir_foreach_instr(instr, nir_cf_node_as_block(node)) {
            if (instr->type == nir_instr_type_jump) {
               assert(nir_instr_as_jump(instr)->type == nir_jump_break);
               return true;
            } else if (instr->type == nir_instr_type_load_const) {
               nir_load_const_instr *lc = nir_instr_as_load_const(instr);
               assert(lc->def.num_components == 1);
               e->ssa[lc->def.index] = nir_const_value_as_uint(lc->value[0], lc->def.bit_size);
            } else if (instr->type == nir_instr_type_alu) {
               nir_alu_instr *alu = nir_instr_as_alu(instr);
               uint64_t a = src(e, alu->src[0].src);
               uint64_t b = nir_op_infos[alu->op].num_inputs > 1 ? src(e, alu->src[1].src) : 0;
               uint64_t value;
               switch (alu->op) {
               case nir_op_iadd: value = a + b; break;
               case nir_op_ishl: value = a << b; break;
               case nir_op_iand: value = a & b; break;
               case nir_op_ior: value = a | b; break;
               case nir_op_ieq: value = a == b; break;
               case nir_op_ine: value = a != b; break;
               default: UNREACHABLE("unmodeled wait ALU");
               }
               e->ssa[alu->def.index] = value & (alu->def.bit_size == 64 ? UINT64_MAX :
                  ((UINT64_C(1) << alu->def.bit_size) - 1));
            } else if (instr->type == nir_instr_type_intrinsic) {
               nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
               uint64_t value = 0;
               switch (intr->intrinsic) {
               case nir_intrinsic_load_deref:
                  value = e->vars[nir_src_as_deref(intr->src[0])->var->data.driver_location];
                  break;
               case nir_intrinsic_store_deref:
                  e->vars[nir_src_as_deref(intr->src[0])->var->data.driver_location] = src(e, intr->src[1]);
                  break;
               case nir_intrinsic_load_preamble: break;
               case nir_intrinsic_load_scs_thread_index_agx: value = 7; break;
               case nir_intrinsic_load_scs_base_agx:
                  if (nir_intrinsic_base(intr) == 263)
                     value = e->pending ? 64 : 0;
                  else {
                     assert(nir_intrinsic_base(intr) == 7);
                     value = nir_intrinsic_flags(intr) == 4 || e->resumes < e->delay ? 64 : 0;
                  }
                  break;
               case nir_intrinsic_store_scs_base_agx:
                  assert(!(src(e, intr->src[0]) & 64));
                  break;
               case nir_intrinsic_continue_ray_gen1_agx: ++e->resumes; break;
               default: UNREACHABLE("unmodeled wait intrinsic");
               }
               if (nir_intrinsic_infos[intr->intrinsic].has_dest)
                  e->ssa[intr->def.index] = value;
            } else {
               assert(instr->type == nir_instr_type_deref);
            }
         }
      }
   }
   return false;
}

static void wait_for(unsigned delay, bool pending)
{
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_COMPUTE, NULL, "delayed RayCore");
   nir_def *completed = agx_nir_gen1_wait_traversal(&b, nir_imm_int(&b, 0), nir_imm_int(&b, 0), 128);
   unsigned vars = 0;
   nir_foreach_function_temp_variable(var, b.impl) var->data.driver_location = vars++;
   assert(vars < 16);
   nir_validate_shader(b.shader, "wait fixture");
   struct execution e = {.delay = delay, .pending = pending};
   e.ssa = calloc(b.impl->ssa_alloc, sizeof(*e.ssa));
   assert(e.ssa);
   assert(!run_list(&e, &b.impl->body));
   assert(e.ssa[completed->index] && e.resumes == (pending ? delay : 0));
   free(e.ssa);
   ralloc_free(b.shader);
}

int main(void)
{
   glsl_type_singleton_init_or_ref();
   wait_for(0, false);
   const unsigned delays[] = {1, 65535, 65536, 65537, 1048576};
   for (unsigned i = 0; i < ARRAY_SIZE(delays); ++i) wait_for(delays[i], true);
   glsl_type_singleton_decref();
   puts("PASS: native wait NIR retains pending queries through 1,048,576 resumes");
   return 0;
}
