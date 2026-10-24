/*
 * Copyright © 2020 Intel Corporation
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice (including the next
 * paragraph) shall be included in all copies or substantial portions of the
 * Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#include "nir_test.h"
#include <set>
#include <vector>

class nir_opt_dce_test : public nir_test {
protected:
   nir_opt_dce_test()
      : nir_test::nir_test("nir_opt_dce_test")
   {
   }
};

class nir_opt_dead_cf_test : public nir_test {
protected:
   nir_opt_dead_cf_test()
      : nir_test::nir_test("nir_opt_dead_cf_test")
   {
   }
};

static bool
dce_collect_source(nir_src *src, void *data)
{
   static_cast<std::vector<nir_instr *> *>(data)->push_back(nir_def_instr(src->ssa));
   return true;
}

static void
dce_collect_conditions(struct exec_list *list, std::vector<nir_instr *> &roots)
{
   foreach_list_typed(nir_cf_node, node, node, list) {
      if (node->type == nir_cf_node_if) {
         nir_if *nif = nir_cf_node_as_if(node);
         roots.push_back(nir_def_instr(nif->condition.ssa));
         dce_collect_conditions(&nif->then_list, roots);
         dce_collect_conditions(&nif->else_list, roots);
      } else if (node->type == nir_cf_node_loop) {
         nir_loop *loop = nir_cf_node_as_loop(node);
         dce_collect_conditions(&loop->body, roots);
      }
   }
}

static void
dce_expect_source_closure(nir_builder *build)
{
   std::vector<nir_instr *> pending;
   std::set<nir_instr *> original, expected, actual;
   nir_foreach_block(block, build->impl) {
      nir_foreach_instr(instr, block) {
         original.insert(instr);
         if (instr->type == nir_instr_type_call ||
             instr->type == nir_instr_type_cmat_call ||
             instr->type == nir_instr_type_jump ||
             (instr->type == nir_instr_type_intrinsic &&
              !(nir_intrinsic_infos[nir_instr_as_intrinsic(instr)->intrinsic].flags &
                NIR_INTRINSIC_CAN_ELIMINATE)))
            pending.push_back(instr);
      }
   }
   dce_collect_conditions(&build->impl->body, pending);
   while (!pending.empty()) {
      nir_instr *instr = pending.back();
      pending.pop_back();
      if (expected.insert(instr).second)
         nir_foreach_src(instr, dce_collect_source, &pending);
   }
   ASSERT_LT(expected.size(), original.size());
   nir_validate_shader(build->shader, "before DCE source closure");
   ASSERT_TRUE(nir_opt_dce_impl(build->impl));
   nir_foreach_block(block, build->impl) {
      nir_foreach_instr(instr, block)
         actual.insert(instr);
   }
   ASSERT_EQ(actual, expected);
   nir_validate_shader(build->shader, "after DCE source closure");
   EXPECT_FALSE(nir_opt_dce_impl(build->impl));
}

TEST_F(nir_opt_dce_test, typed_sources_match_generic_reachability)
{
   nir_def *index = nir_iand(b, nir_load_local_invocation_index(b), nir_imm_int(b, 15));
   nir_variable *array = nir_local_variable_create(
      b->impl, glsl_array_type(glsl_uint_type(), 16, 0), "array");
   nir_def *array_index = nir_iand(b, nir_load_local_invocation_index(b), nir_imm_int(b, 15));
   nir_deref_instr *element = nir_build_deref_array(b, nir_build_deref_var(b, array), array_index);
   nir_store_deref(b, element, nir_iadd(b, index, nir_imm_int(b, 3)), 1);
   nir_def *loaded = nir_load_deref(b, element);

   nir_tex_instr *tex = nir_tex_instr_create(b->shader, 2);
   tex->op = nir_texop_txf;
   tex->sampler_dim = GLSL_SAMPLER_DIM_2D;
   tex->coord_components = 2;
   tex->dest_type = nir_type_uint32;
   tex->src[0].src_type = nir_tex_src_coord;
   tex->src[0].src = nir_src_for_ssa(nir_vec2(b, index, loaded));
   tex->src[1].src_type = nir_tex_src_lod;
   tex->src[1].src = nir_src_for_ssa(nir_imm_int(b, 0));
   nir_def_init(&tex->instr, &tex->def, 4, 32);
   nir_builder_instr_insert(b, &tex->instr);
   nir_variable *out = nir_variable_create(
      b->shader, nir_var_shader_out, glsl_vector_type(GLSL_TYPE_UINT, 4), "out");
   nir_store_var(b, out, &tex->def, 15);

   nir_function *callee = nir_function_create(b->shader, "callee");
   callee->num_params = 1;
   callee->params = rzalloc_array(callee, nir_parameter, 1);
   callee->params[0].num_components = 1;
   callee->params[0].bit_size = 32;
   callee->params[0].type = glsl_uint_type();
   nir_call_instr *call = nir_call_instr_create(b->shader, callee);
   call->indirect_callee = nir_src_for_ssa(nir_imm_int64(b, 4096));
   call->params[0] = nir_src_for_ssa(nir_iadd(b, index, nir_imm_int(b, 5)));
   nir_builder_instr_insert(b, &call->instr);
   nir_cmat_call_instr *cmat = nir_cmat_call_instr_create(
      b->shader, nir_cmat_call_op_per_element_op, callee);
   cmat->params[0] = nir_src_for_ssa(nir_iadd(b, nir_undef(b, 1, 32), nir_imm_int(b, 7)));
   nir_builder_instr_insert(b, &cmat->instr);
   nir_undef(b, 4, 32);
   nir_iadd(b, index, nir_imm_int(b, 99));
   nir_jump(b, nir_jump_return);
   dce_expect_source_closure(b);
}

TEST_F(nir_opt_dce_test, live_and_dead_loop_phi_cycles_match_reachability)
{
   nir_def *zero = nir_imm_int(b, 0);
   nir_def *one = nir_imm_int(b, 1);
   nir_loop *loop = nir_push_loop(b);
   nir_block *header = nir_loop_first_block(loop);
   nir_phi_instr *live = nir_phi_instr_create(b->shader);
   nir_phi_instr *dead = nir_phi_instr_create(b->shader);
   nir_def_init(&live->instr, &live->def, 1, 32);
   nir_def_init(&dead->instr, &dead->def, 1, 32);
   nir_phi_instr_add_src(live, nir_def_block(zero), zero);
   nir_phi_instr_add_src(dead, nir_def_block(zero), zero);
   nir_if *nif = nir_push_if(b, nir_ige(b, &live->def, nir_imm_int(b, 4)));
   nir_jump(b, nir_jump_break);
   nir_pop_if(b, nif);
   nir_def *next_live = nir_iadd(b, &live->def, one);
   nir_def *next_dead = nir_iadd(b, &dead->def, one);
   nir_phi_instr_add_src(live, nir_def_block(next_live), next_live);
   nir_phi_instr_add_src(dead, nir_def_block(next_dead), next_dead);
   nir_pop_loop(b, loop);
   nir_variable *out = nir_variable_create(b->shader, nir_var_shader_out, glsl_int_type(), "out");
   nir_store_var(b, out, &live->def, 1);
   nir_cursor end = b->cursor;
   b->cursor = nir_before_block(header);
   nir_builder_instr_insert(b, &live->instr);
   nir_builder_instr_insert(b, &dead->instr);
   b->cursor = end;
   dce_expect_source_closure(b);
}

nir_phi_instr *create_one_source_phi(nir_shader *shader, nir_block *pred,
                                     nir_def *def)
{
   nir_phi_instr *phi = nir_phi_instr_create(shader);
   nir_phi_instr_add_src(phi, pred, def);
   nir_def_init(&phi->instr, &phi->def, def->num_components,
                def->bit_size);

   return phi;
}

TEST_F(nir_opt_dce_test, return_before_loop)
{
   /* Test that nir_opt_dce() works correctly with loops immediately following a jump.
    * nir_opt_dce() has a fast path for loops without continues, and this test ensures that it
    * looks at the actual predecessors of the header instead of just counting.
    *
    *  block block_0:
    *  // preds:
    *  return
    *  // succs: block_3
    *  loop {
    *     block block_1:
    *     // preds: block_1
    *     vec1 32 ssa_1 = phi block_1: ssa_0
    *     vec1 32 ssa_0 = load_const (0x00000001)
    *     vec1 32 ssa_2 = deref_var &out (shader_out int) 
    *     intrinsic store_deref (ssa_2, ssa_1) (1, 0)
    *     // succs: block_1
    *  }
    *  block block_2:
    *  // preds:
    *  // succs: block_3
    *  block block_3:
    *
    * If the fast path is taken here, ssa_0 will be incorrectly DCE'd.
    */

   nir_variable *var = nir_variable_create(b->shader, nir_var_shader_out, glsl_int_type(), "out");

   nir_jump(b, nir_jump_return);

   nir_loop *loop = nir_push_loop(b);

   nir_def *one = nir_imm_int(b, 1);

   nir_phi_instr *phi = create_one_source_phi(b->shader, nir_def_block(one), one);
   nir_instr_insert_before_block(nir_def_block(one), &phi->instr);

   nir_store_var(b, var, &phi->def, 0x1);

   nir_pop_loop(b, loop);

   ASSERT_FALSE(nir_opt_dce(b->shader));

   nir_validate_shader(b->shader, NULL);
}

TEST_F(nir_opt_dead_cf_test, jump_before_constant_if)
{
   /*
    * Test that nir_opt_dead_cf removes everything after the jump, instead of trying and failing to
    * stitch b0 and b3.
    *
    * block b0:  // preds:
    * 1     %0 = load_const (false)
    *            return
    *            // succs: b4
    * if %0 (false) {
    *     block b1:  // preds:
    *     32    %1 = load_const (0x00000001)
    *     32    %2 = deref_var &out (shader_out int)
    *                @store_deref (%2, %1 (0x1)) (wrmask=x, access=0)
    *                // succs: b3
    * } else {
    *     block b2:  // preds: , succs: b3
    * }
    * block b3:  // preds: b1 b2
    * 32    %3 = load_const (0x00000000)
    * 32    %4 = deref_var &out (shader_out int)
    *            @store_deref (%4, %3 (0x0)) (wrmask=x, access=0)
    *            // succs: b4
    * block b4:
    */
   nir_variable *var = nir_variable_create(b->shader, nir_var_shader_out, glsl_int_type(), "out");

   nir_def *cond = nir_imm_false(b);
   nir_jump(b, nir_jump_return);
   nir_push_if(b, cond);
   nir_store_var(b, var, nir_imm_int(b, 1), 0x1);
   nir_pop_if(b, NULL);

   nir_store_var(b, var, nir_imm_int(b, 0), 0x1);

   ASSERT_TRUE(nir_opt_dead_cf(b->shader));

   nir_validate_shader(b->shader, NULL);
}

TEST_F(nir_opt_dead_cf_test, unreachable_phi)
{
   /*
    * Test that nir_opt_dead_cf/nir_remove_after_cf_node removes phis at the start of dead code.
    *
    * block b0:  // preds:
    * 32    %0 = deref_var &cond (shader_in bool)
    * 1     %1 = @load_deref (%0) (access=none)
    *            // succs: b1 b2
    * if %1 {
    *     block b1:// preds: b0
    *     return
    *     // succs: b4
    * } else {
    *     block b2:// preds: b0
    *     return
    *     // succs: b4
    * }
    * block b3:  // preds:
    * 32    %2 = phi
    * 32    %3 = deref_var &out (shader_out int)
    *            @store_deref (%3, %2) (wrmask=x, access=none)
    *            // succs: b4
    * block b4:
    */
   nir_variable *cond = nir_variable_create(b->shader, nir_var_shader_in, glsl_bool_type(), "cond");
   nir_variable *var = nir_variable_create(b->shader, nir_var_shader_out, glsl_int_type(), "out");

   nir_push_if(b, nir_load_var(b, cond));
   nir_jump(b, nir_jump_return);
   nir_push_else(b, NULL);
   nir_jump(b, nir_jump_return);
   nir_pop_if(b, NULL);

   nir_phi_instr *phi = nir_phi_instr_create(b->shader);
   nir_def_init(&phi->instr, &phi->def, 1, 32);
   nir_builder_instr_insert(b, &phi->instr);
   nir_store_var(b, var, &phi->def, 0x1);

   ASSERT_TRUE(nir_opt_dead_cf(b->shader));
   nir_validate_shader(b->shader, NULL);

   ASSERT_TRUE(exec_list_is_empty(&nir_impl_last_block(b->impl)->instr_list));
   ASSERT_FALSE(nir_opt_dead_cf(b->shader));
}

static nir_def *
build_query(nir_builder *b)
{
   nir_variable *q = nir_local_variable_create(b->impl, glsl_uint64_t_type(), "q");
   q->data.ray_query = true;
   nir_def *rq = &nir_build_deref_var(b, q)->def;
   nir_rq_initialize(b, rq, nir_imm_int64(b, 0), nir_imm_int(b, 0),
                     nir_imm_int(b, 255), nir_imm_vec3(b, 0, 0, 0),
                     nir_imm_float(b, 0), nir_imm_vec3(b, 0, 0, 1),
                     nir_imm_float(b, 10));
   return rq;
}

static nir_def *
load_query_type(nir_builder *b, nir_def *rq, bool committed)
{
   nir_intrinsic_instr *load = nir_intrinsic_instr_create(b->shader,
                                                         nir_intrinsic_rq_load);
   load->num_components = 1;
   load->src[0] = nir_src_for_ssa(rq);
   nir_intrinsic_set_ray_query_value(load, nir_ray_query_value_intersection_type);
   nir_intrinsic_set_committed(load, committed);
   nir_def_init(&load->instr, &load->def, 1, 32);
   nir_builder_instr_insert(b, &load->instr);
   return &load->def;
}

TEST_F(nir_opt_dead_cf_test, dead_query_loop)
{
   nir_def *rq = build_query(b);
   nir_push_loop(b);
   nir_break_if(b, nir_inot(b, nir_rq_proceed(b, 1, rq)));
   load_query_type(b, rq, false);
   nir_pop_loop(b, NULL);
   nir_validate_shader(b->shader, NULL);
   ASSERT_FALSE(nir_opt_dead_cf(b->shader));
   ASSERT_TRUE(nir_opt_dead_cf_with_options(b->shader, true));
   ASSERT_FALSE(nir_block_get_following_loop(nir_start_block(b->impl)));
   nir_validate_shader(b->shader, NULL);
}

TEST_F(nir_opt_dead_cf_test, query_state_escapes_loop)
{
   nir_def *rq = build_query(b);
   nir_push_loop(b);
   nir_break_if(b, nir_inot(b, nir_rq_proceed(b, 1, rq)));
   nir_pop_loop(b, NULL);
   nir_variable *var = nir_deref_instr_get_variable(nir_def_as_deref(rq));
   nir_def *again = &nir_build_deref_var(b, var)->def;
   nir_def *type = load_query_type(b, again, true);
   nir_store_global(b, type, nir_imm_int64(b, 4096));
   nir_validate_shader(b->shader, NULL);
   ASSERT_FALSE(nir_opt_dead_cf_with_options(b->shader, true));
   nir_validate_shader(b->shader, NULL);
}

TEST_F(nir_opt_dead_cf_test, query_state_escapes_if)
{
   nir_def *rq = build_query(b);
   nir_push_if(b, nir_ieq_imm(b, nir_load_local_invocation_index(b), 0));
   nir_rq_proceed(b, 1, rq);
   nir_pop_if(b, NULL);
   nir_rq_proceed(b, 1, rq);
   nir_validate_shader(b->shader, NULL);
   ASSERT_FALSE(nir_opt_dead_cf_with_options(b->shader, true));
   nir_validate_shader(b->shader, NULL);
}

TEST_F(nir_opt_dead_cf_test, query_result_escapes_if)
{
   nir_def *rq = build_query(b);
   nir_push_if(b, nir_ieq_imm(b, nir_load_local_invocation_index(b), 0));
   nir_def *more = nir_rq_proceed(b, 1, rq);
   nir_push_else(b, NULL);
   nir_def *no = nir_imm_false(b);
   nir_pop_if(b, NULL);
   nir_def *result = nir_if_phi(b, more, no);
   nir_store_global(b, nir_b2i32(b, result), nir_imm_int64(b, 4096));
   nir_validate_shader(b->shader, NULL);
   ASSERT_FALSE(nir_opt_dead_cf_with_options(b->shader, true));
   nir_validate_shader(b->shader, NULL);
}

TEST_F(nir_opt_dead_cf_test, dead_query_with_live_query_in_same_loop)
{
   nir_def *dead = build_query(b);
   nir_def *live = build_query(b);
   nir_push_loop(b);
   nir_rq_proceed(b, 1, dead);
   nir_break_if(b, nir_inot(b, nir_rq_proceed(b, 1, live)));
   nir_pop_loop(b, NULL);
   nir_def *type = load_query_type(b, live, true);
   nir_store_global(b, type, nir_imm_int64(b, 4096));
   nir_validate_shader(b->shader, NULL);
   ASSERT_FALSE(nir_opt_dead_cf_with_options(b->shader, true));
   nir_validate_shader(b->shader, NULL);
}

TEST_F(nir_opt_dead_cf_test, dead_query_if)
{
   nir_push_if(b, nir_ieq_imm(b, nir_load_local_invocation_index(b), 0));
   nir_def *rq = build_query(b);
   nir_rq_proceed(b, 1, rq);
   load_query_type(b, rq, false);
   nir_pop_if(b, NULL);
   nir_validate_shader(b->shader, NULL);
   ASSERT_FALSE(nir_opt_dead_cf(b->shader));
   ASSERT_TRUE(nir_opt_dead_cf_with_options(b->shader, true));
   ASSERT_FALSE(nir_block_get_following_if(nir_start_block(b->impl)));
   nir_validate_shader(b->shader, NULL);
}
