/* Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */
#include <gtest/gtest.h>
#include "agx_test.h"
extern "C" {
#include "compiler/nir/nir_builder.h"
#include "agx_nir.h"
}

/* Compile through selection, scheduling, wait insertion and encoding. The
 * shader contains only the barrier, so the 16-bit opcode cannot occur inside
 * some unrelated instruction's operands. These are encoding controls; they
 * cannot establish GPU cache coherence.
 */
TEST(AGX2ImageAcquire, MemoryToTextureVisibility)
{
   for (auto scope : {SCOPE_NONE, SCOPE_INVOCATION, SCOPE_SUBGROUP,
                      SCOPE_WORKGROUP, SCOPE_QUEUE_FAMILY, SCOPE_DEVICE}) {
      for (nir_variable_mode mode :
           {nir_var_image, nir_var_mem_global, nir_var_mem_shared}) {
         for (unsigned semantics :
              {0u, unsigned(NIR_MEMORY_RELEASE), unsigned(NIR_MEMORY_ACQUIRE),
               unsigned(NIR_MEMORY_ACQ_REL),
               unsigned(NIR_MEMORY_MAKE_VISIBLE)}) {
            SCOPED_TRACE(::testing::Message()
                         << "scope=" << scope << " mode=" << mode
                         << " semantics=" << semantics);
            nir_builder b = nir_builder_init_simple_shader(
               MESA_SHADER_COMPUTE, &agx_nir_options, "AGX2 image acquire");
            nir_intrinsic_instr *barrier = nir_barrier(&b);
            nir_intrinsic_set_execution_scope(barrier, SCOPE_NONE);
            nir_intrinsic_set_memory_scope(barrier, scope);
            nir_intrinsic_set_memory_modes(barrier, mode);
            nir_intrinsic_set_memory_semantics(barrier,
                                               nir_memory_semantics(semantics));
            agx_shader_key key = {};
            key.dev.chip = AGX_CHIP_G13G;
            key.dev.encoding = AGX_ISA_ENCODING_AGX2;
            key.no_preamble = true;
            agx_preprocess_nir(b.shader);
            agx_shader_part part = {};
            ASSERT_TRUE(agx_compile_shader_nir(b.shader, &key, &part));
            const uint8_t *code = static_cast<const uint8_t *>(part.binary);
            unsigned flushes = 0;
            for (unsigned i = part.info.main_offset;
                 i + 1 < part.info.main_offset + part.info.main_size; i += 2)
               flushes += code[i] == 0xf5 && code[i + 1] == 0x40;
            const bool acquire =
               semantics & (NIR_MEMORY_ACQUIRE | NIR_MEMORY_MAKE_VISIBLE);
            EXPECT_EQ(flushes, unsigned(scope != SCOPE_NONE &&
                                        mode == nir_var_image && acquire));
            free(part.binary);
            ralloc_free(b.shader);
         }
      }
   }
}
