/* Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */
#include "hk_nir_root_size.h"

/* Follow only constant pointer additions to ordinary root loads. Descriptor
 * indexing, reflection, casts, stores and any unrecognized use keep the whole
 * table. This is a footprint proof, not an assumption about a shader's stage
 * or API descriptor layout. Bound recursion even for adversarial NIR. */
static unsigned
root_extent(nir_def *def, uint64_t offset, unsigned full_size, unsigned depth)
{
   if (depth > 8 || offset >= full_size)
      return full_size;

   unsigned end = 0;
   nir_foreach_use_including_if(src, def) {
      if (nir_src_is_if(src))
         return full_size;
      nir_instr *use = nir_src_use_instr(src);
      unsigned size;
      if (use->type == nir_instr_type_alu) {
         nir_alu_instr *alu = nir_instr_as_alu(use);
         if (alu->op != nir_op_iadd || alu->def.bit_size != 64 ||
             alu->def.num_components != 1)
            return full_size;
         nir_src other = alu->src[alu->src[0].src.ssa == def ? 1 : 0].src;
         if (!nir_src_is_const(other))
            return full_size;
         const uint64_t add = nir_src_as_uint(other);
         if (add >= full_size || offset + add >= full_size)
            return full_size;
         size = root_extent(&alu->def, offset + add, full_size, depth + 1);
      } else if (use->type == nir_instr_type_intrinsic) {
         nir_intrinsic_instr *intr = nir_instr_as_intrinsic(use);
         if (intr->intrinsic != nir_intrinsic_load_global_constant ||
             src != &intr->src[0])
            return full_size;
         /* The first field is a self pointer used for descriptor reflection.
          * A load of it can reach any later root field through that pointer. */
         if (offset < sizeof(uint64_t))
            return full_size;
         const unsigned bytes = intr->def.num_components * intr->def.bit_size / 8;
         if (bytes > full_size - offset)
            return full_size;
         size = offset + bytes;
      } else {
         return full_size;
      }
      end = MAX2(end, size);
   }
   return end;
}

unsigned
hk_nir_root_size(const nir_shader *nir, unsigned full_size)
{
   unsigned end = 0;
   nir_foreach_function_impl(impl, nir) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;
            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            if (intr->intrinsic == nir_intrinsic_load_root_agx)
               end = MAX2(end, root_extent(&intr->def, 0, full_size, 0));
         }
      }
   }
   return end;
}
