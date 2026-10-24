/* Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */
#include "hk_rt_inlining.h"
#include "compiler/nir/nir_builder.h"
#include "compiler/glsl_types.h"
#undef NDEBUG
#include <assert.h>
#include <stdio.h>

static nir_shader *
graph(unsigned depth, unsigned calls)
{
   static const nir_shader_compiler_options options = {0};
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_COMPUTE, &options, "RT DAG");
   nir_function *callee = b.impl->function;
   nir_function *levels[32];
   for (unsigned i = 0; i < depth; ++i) {
      levels[i] = nir_function_create(b.shader, "callee");
      nir_function_impl_create(levels[i]);
   }
   for (unsigned i = 0; i <= depth; ++i) {
      nir_builder at = nir_builder_at(nir_after_impl(callee->impl));
      nir_store_global(&at, nir_imm_int(&at, i), nir_imm_int64(&at, 0x10000),
                       .align_mul = 4, .write_mask = 1);
      if (i < depth)
         for (unsigned c = 0; c < calls; ++c)
            nir_build_call(&at, levels[i], 0, NULL);
      if (i < depth)
         callee = levels[i];
   }
   nir_validate_shader(b.shader, "inlining budget fixture");
   return b.shader;
}

int main(void)
{
   glsl_type_singleton_init_or_ref();
   /* Deep single-call pipelines remain valid at the advertised depth. */
   nir_shader *linear = graph(31, 1);
   assert(hk_rt_inlining_fits(linear, 262144));
   NIR_PASS(_, linear, nir_inline_functions);
   nir_validate_shader(linear, "linear inline");
   ralloc_free(linear);

   /* A shared DAG is small, but each call site gets its own inline copy. */
   nir_shader *branch = graph(20, 2);
   assert(!hk_rt_inlining_fits(branch, 262144));
   assert(!hk_rt_inlining_fits(branch, 0));
   ralloc_free(branch);
   nir_shader *small = graph(4, 2);
   assert(hk_rt_inlining_fits(small, 262144));
   NIR_PASS(_, small, nir_inline_functions);
   nir_validate_shader(small, "branch inline");
   ralloc_free(small);

   nir_shader *cycle = graph(0, 0);
   nir_function_impl *impl = nir_shader_get_entrypoint(cycle);
   nir_builder at = nir_builder_at(nir_after_impl(impl));
   nir_build_call(&at, impl->function, 0, NULL);
   assert(!hk_rt_inlining_fits(cycle, 262144));
   ralloc_free(cycle);
   glsl_type_singleton_decref();
   puts("PASS: deep linear and small branching RT DAGs fit; exponential expansion and cycles refused");
   return 0;
}
