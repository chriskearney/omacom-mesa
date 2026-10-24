/* Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */
#include "agx_abi.h"
#include "agx_helpers.h"
#include "agx_linker.h"
#include "agx_nir_lower_vbo.h"
#include "asahi/compiler/agx_nir.h"
#include "compiler/nir/nir_builder.h"
#include <stdio.h>
#include <stdlib.h>

#define CHECK(x) do { if (!(x)) { \
 fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #x); abort(); \
} } while (0)

static void
fold(nir_shader *nir)
{
   bool progress;
   do {
      progress = false;
      NIR_PASS(progress, nir, nir_opt_copy_prop);
      NIR_PASS(progress, nir, nir_opt_constant_folding);
      NIR_PASS(progress, nir, nir_opt_dce);
   } while (progress);
}

/* Evaluate the produced addresses and bounds checks with an explicit load
 * model. This tests lowering semantics; it does not execute GPU instructions
 * or establish what a real soft fault returns. */
static void
check_fetch(enum pipe_format format, unsigned components, unsigned size_B,
             unsigned vertex, bool soft_fault, bool native)
{
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_VERTEX,
      &agx_nir_options, "robust vertex fetch");
   nir_def *value = nir_load_input(&b, components, 32, nir_imm_int(&b, 0),
      .base = 0, .dest_type = nir_type_uint32);
   nir_store_global(&b, value, nir_imm_int64(&b, 0x10004000000ull),
      .align_mul = 4, .write_mask = BITFIELD_MASK(components));
   struct agx_velem_key attrib = {.format = format, .stride = 4 * components};
   struct agx_robustness rs = {
      .level = AGX_ROBUSTNESS_D3D, .soft_fault = soft_fault,
   };
   uint64_t address;
   uint32_t clamp = agx_calculate_vbo_clamp(0x10005000000ull, format,
      size_B, attrib.stride, 0, &address);
   CHECK(native ? agx_nir_lower_vbo_raw(b.shader, &attrib, rs, false) :
                  agx_nir_lower_vbo(b.shader, &attrib, rs, false));
   /* One read attribute uses a four-half pointer followed by a two-half
    * inclusive clamp. Exercise the texture and ray-query prefixes too. */
   const unsigned uniform_base = (native ? 8 : 0) + (soft_fault ? 168 : 0);
   CHECK(agx_nir_lower_non_monolithic_uniforms_at(b.shader, 1, uniform_base));
   nir_foreach_block(block, b.impl) {
      nir_foreach_instr_safe(instr, block) {
         if (instr->type != nir_instr_type_intrinsic)
            continue;
         nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
         nir_def *replacement = NULL;
         nir_builder c = nir_builder_at(nir_before_instr(instr));
         switch (intr->intrinsic) {
         case nir_intrinsic_load_preamble:
            if (nir_intrinsic_base(intr) == uniform_base) {
               CHECK(intr->def.bit_size == 64);
               replacement = nir_imm_int64(&c, address);
            } else {
               CHECK(nir_intrinsic_base(intr) == uniform_base + 4);
               CHECK(intr->def.bit_size == 32);
               replacement = nir_imm_int(&c, clamp);
            }
            break;
         case nir_intrinsic_load_attrib_clamp_agx:
            replacement = nir_imm_int(&c, clamp); break;
         case nir_intrinsic_load_vbo_base_agx:
            replacement = nir_imm_int64(&c, address); break;
         case nir_intrinsic_load_vertex_id:
            replacement = nir_imm_int(&c, vertex); break;
         default: break;
         }
         if (replacement) {
            nir_def_rewrite_uses(&intr->def, replacement);
            nir_instr_remove(instr);
         }
      }
   }
   fold(b.shader);
   nir_foreach_block(block, b.impl) {
      nir_foreach_instr_safe(instr, block) {
         if (instr->type != nir_instr_type_intrinsic)
            continue;
         nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
         if (intr->intrinsic != nir_intrinsic_load_constant_agx)
            continue;
         CHECK(nir_src_is_const(intr->src[0]));
         CHECK(nir_src_is_const(intr->src[1]));
         uint64_t base = nir_src_as_uint(intr->src[0]);
         uint32_t index = nir_src_as_uint(intr->src[1]);
         uint64_t offset = (uint64_t)index << (2 + nir_intrinsic_base(intr));
         nir_builder c = nir_builder_at(nir_before_instr(instr));
         nir_def *channels[4];
         for (unsigned i = 0; i < intr->def.num_components; ++i) {
            /* Emulate the zero page and the documented zero-fill soft-fault
             * contract; give live memory a unique value in every component. */
            /* This optional control deliberately violates soft-fault zero
             * fill to distinguish explicit masking from address redirection.
             * It is not a claim about the actual hardware's fault result. */
            bool poison = native && getenv("ASAHI_TEST_POISON_NATIVE_FAULT");
            bool zero = base == AGX_ZERO_PAGE_ADDRESS ||
                        (base == 0 && !poison);
            CHECK(zero || base == 0 || base == 0x10005000000ull);
            channels[i] = nir_imm_int(&c, zero ? 0 : 3 + offset + 4 * i);
         }
         nir_def_rewrite_uses(&intr->def,
            nir_vec(&c, channels, intr->def.num_components));
         nir_instr_remove(instr);
      }
   }
   fold(b.shader);
   unsigned stores = 0;
   nir_foreach_block(block, b.impl) {
      nir_foreach_instr(instr, block) {
         if (instr->type != nir_instr_type_intrinsic)
            continue;
         nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
         if (intr->intrinsic != nir_intrinsic_store_global)
            continue;
         CHECK(nir_src_is_const(intr->src[0]));
         bool in_bounds = (uint64_t)(vertex + 1) * attrib.stride <= size_B;
         for (unsigned i = 0; i < components; ++i)
            CHECK(nir_src_comp_as_uint(intr->src[0], i) ==
                  (in_bounds ? 3 + vertex * attrib.stride + 4 * i : 0));
         ++stores;
      }
   }
   CHECK(stores == 1);
   nir_validate_shader(b.shader, "robust fetch control");
   ralloc_free(b.shader);
}

int main(void)
{
   glsl_type_singleton_init_or_ref();
   const enum pipe_format formats[][4] = {
      {PIPE_FORMAT_R32_FLOAT, PIPE_FORMAT_R32G32_FLOAT,
       PIPE_FORMAT_R32G32B32_FLOAT, PIPE_FORMAT_R32G32B32A32_FLOAT},
      {PIPE_FORMAT_R32_SINT, PIPE_FORMAT_R32G32_SINT,
       PIPE_FORMAT_R32G32B32_SINT, PIPE_FORMAT_R32G32B32A32_SINT},
   };
   unsigned count = 0;
   const unsigned sizes[] = {0, 4, 8, 12, 16, 31, 32, 33, 256};
   const unsigned vertices[] = {0, 1, 2, 15, 16, 17, 999};
   for (unsigned kind = 0; kind < ARRAY_SIZE(formats); ++kind)
      for (unsigned c = 1; c <= 4; ++c)
         for (unsigned size = 0; size < ARRAY_SIZE(sizes); ++size)
            for (unsigned v = 0; v < ARRAY_SIZE(vertices); ++v)
               for (unsigned soft = 0; soft < 2; ++soft)
                  for (unsigned native = 0; native < 2; ++native) {
                     check_fetch(formats[kind][c - 1], c, sizes[size],
                        vertices[v], soft, native);
                     ++count;
                  }
   glsl_type_singleton_decref();
   printf("Robust vertex fetch controls: %u passed\n", count);
   return 0;
}
