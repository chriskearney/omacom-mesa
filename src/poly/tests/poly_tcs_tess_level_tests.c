/* Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */
#include "nir_builder.h"
#include "poly_nir.h"
#include "poly/tessellator.h"

#include <stddef.h>
#include <stdio.h>

#define PARAMS 0x10000
#define OUTPUT 0x20000
#define READBACK 0x30000

static unsigned
io_slots(const struct glsl_type *type, bool bindless)
{
   return glsl_count_attribute_slots(type, false);
}

/* Specialize execution-time inputs after lowering, so the index is genuinely
 * indirect when nir_lower_io and Poly construct the memory access. */
static bool
specialize(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   unsigned index = *(unsigned *)data;
   b->cursor = nir_before_instr(&intr->instr);
   nir_def *value;
   switch (intr->intrinsic) {
   case nir_intrinsic_load_tess_param_buffer_poly:
      value = nir_imm_int64(b, PARAMS);
      break;
   case nir_intrinsic_load_local_invocation_id:
      value = nir_imm_ivec3(b, index, 0, 0);
      break;
   case nir_intrinsic_load_workgroup_id:
      value = nir_imm_ivec3(b, 2, 1, 0);
      break;
   case nir_intrinsic_load_global_constant:
      if (!nir_src_is_const(intr->src[0]))
         return false;
      switch (nir_src_as_uint(intr->src[0])) {
      case PARAMS + offsetof(struct poly_tess_params, tcs_buffer):
         value = nir_imm_int64(b, OUTPUT);
         break;
      case PARAMS + offsetof(struct poly_tess_params, patches_per_instance):
         value = nir_imm_int(b, 7);
         break;
      default:
         return false;
      }
      break;
   default:
      return false;
   }
   nir_def_replace(&intr->def, value);
   return true;
}

static bool
check_level(gl_varying_slot location, unsigned index, bool indirect, bool read)
{
   static const nir_shader_compiler_options options = {
      .support_indirect_outputs = BITFIELD_BIT(MESA_SHADER_TESS_CTRL),
   };
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_TESS_CTRL, &options, "compact tessellation level");
   b.shader->info.tess.tcs_vertices_out = 4;
   b.shader->info.workgroup_size[0] = 4;
   unsigned length = location == VARYING_SLOT_TESS_LEVEL_OUTER ? 4 : 2;
   nir_variable *level = nir_variable_create(
      b.shader, nir_var_shader_out,
      glsl_array_type(glsl_float_type(), length, 0), "level");
   level->data.location = location;
   level->data.compact = true;
   level->data.patch = true;
   nir_def *idx = indirect ? nir_load_invocation_id(&b)
                           : nir_imm_int(&b, index);
   nir_deref_instr *element = nir_build_deref_array(
      &b, nir_build_deref_var(&b, level), idx);
   if (read) {
      nir_store_global(&b, nir_load_deref(&b, element),
                       nir_imm_int64(&b, READBACK), .write_mask = 1);
   } else {
      nir_store_deref(&b, element, nir_imm_float(&b, 1), 1);
   }
   nir_lower_io(b.shader, nir_var_shader_out, io_slots, 0);
   nir_shader_gather_info(b.shader, b.impl);
   /* Include a per-vertex output to exercise patch stride as well as factors. */
   b.shader->info.outputs_written |= VARYING_BIT_POS;
   poly_nir_lower_tcs_with_options(b.shader, false, true);
   bool progress;
   do {
      progress = nir_shader_intrinsics_pass(
         b.shader, specialize, nir_metadata_control_flow, &index);
      progress |= nir_opt_constant_folding(b.shader);
      progress |= nir_opt_algebraic(b.shader);
      progress |= nir_opt_dead_cf(b.shader);
      progress |= nir_opt_copy_prop(b.shader);
      progress |= nir_opt_dce(b.shader);
   } while (progress);
   nir_validate_shader(b.shader, "specialized TCS factor address");

   uint64_t expected = OUTPUT + 9 * (6 + 4 * 4) * 4 +
      (location == VARYING_SLOT_TESS_LEVEL_INNER ? 4 : 0) * 4 + index * 4;
   unsigned matches = 0;
   nir_foreach_block(block, b.impl) {
      nir_foreach_instr(instr, block) {
         if (instr->type != nir_instr_type_intrinsic)
            continue;
         nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
         if (intr->intrinsic != (read ? nir_intrinsic_load_global
                                     : nir_intrinsic_store_global))
            continue;
         nir_src addr = intr->src[read ? 0 : 1];
         if (nir_src_is_const(addr) && nir_src_as_uint(addr) == expected)
            ++matches;
      }
   }
   if (matches != 1) {
      fprintf(stderr, "%s[%u] %s %s: expected address 0x%llx\n",
              location == VARYING_SLOT_TESS_LEVEL_OUTER ? "outer" : "inner",
              index, indirect ? "indirect" : "constant", read ? "read" : "write",
              (unsigned long long)expected);
      nir_print_shader(b.shader, stderr);
   }
   ralloc_free(b.shader);
   return matches == 1;
}

int
main(void)
{
   glsl_type_singleton_init_or_ref();
   bool valid = true;
   unsigned cases = 0;
   for (unsigned inner = 0; inner < 2; ++inner) {
      for (unsigned index = 0; index < (inner ? 2 : 4); ++index) {
         for (unsigned indirect = 0; indirect < 2; ++indirect) {
            for (unsigned read = 0; read < 2; ++read) {
               valid &= check_level(inner ? VARYING_SLOT_TESS_LEVEL_INNER
                                          : VARYING_SLOT_TESS_LEVEL_OUTER,
                                     index, indirect, read);
               ++cases;
            }
         }
      }
   }
   glsl_type_singleton_decref();
   printf("%u compact TCS factor address checks: %s\n", cases,
          valid ? "PASS" : "FAIL");
   return valid ? 0 : 1;
}
