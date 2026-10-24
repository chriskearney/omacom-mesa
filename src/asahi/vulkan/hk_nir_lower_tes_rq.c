/* Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */
#include "hk_nir_lower_tes_rq.h"
#include "asahi/compiler/agx_nir_gen1_ray_query.h"
#include "poly/nir/poly_nir.h"
#include "poly/geometry.h"
#include "compiler/nir/nir_builder.h"

/* Run the kernel body only for invocations below the TES-before-GS count
 * that the GPU setup wrote, so it may be launched with a direct grid that
 * bounds the tessellator output (hk_tes_direct_query_threads()). Extra
 * invocations must not query, store, or write vertex outputs. */
static void
hk_guard_tes_kernel_invocations(nir_shader *tes)
{
   nir_function_impl *impl = nir_shader_get_entrypoint(tes);
   nir_cf_list body;
   nir_cf_extract(&body, nir_before_impl(impl), nir_after_impl(impl));

   nir_builder b = nir_builder_at(nir_after_impl(impl));
   nir_def *id = nir_channel(&b, nir_load_global_invocation_id(&b, 32), 0);
   nir_def *count = nir_load_global(&b, 1, 32,
      nir_iadd_imm(&b, nir_load_vertex_param_buffer_poly(&b),
                   offsetof(struct poly_vertex_params, verts_per_instance)),
      .align_mul = 4);
   nir_push_if(&b, nir_ult(&b, id, count));
   nir_cf_reinsert(&body, b.cursor);
   nir_pop_if(&b, NULL);
   nir_progress(true, impl, nir_metadata_none);
}

bool
hk_nir_lower_tes_variant(nir_shader *tes, bool hardware,
                         const struct agx_gen1_ray_query_layout *rq_layout)
{
   poly_nir_lower_tes(tes, hardware);

   if (!rq_layout)
      return true;

   nir_opt_ray_queries(tes);
   nir_remove_dead_variables(tes, nir_var_function_temp | nir_var_shader_temp,
                             NULL);
   nir_shader_gather_info(tes, nir_shader_get_entrypoint(tes));
   if (!tes->info.ray_queries)
      return true;

   if (!hardware)
      hk_guard_tes_kernel_invocations(tes);

   /* A single lowering before the variant split gave both variants the
    * context of whichever stage the pipeline layout predicted. The query
    * context footer differs: four dwords for a vertex program, two for a
    * CDM kernel. */
   struct agx_gen1_ray_query_layout layout = *rq_layout;
   layout.graphics_stage = hardware && rq_layout->graphics_stage;

   nir_opt_ray_query_ranges(tes);
   if (!agx_nir_lower_gen1_ray_query(tes, &layout))
      return false;

   /* Every query is now ordinary temporaries; the backend refuses a shader
    * that still declares any (as in hk_lower_nir()). */
   tes->info.ray_queries = 0;
   return true;
}
