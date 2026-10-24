/*
 * Copyright 2024 Valve Corporation
 * SPDX-License-Identifier: MIT
 */

#include "compiler/nir/nir.h"
#include "compiler/nir/nir_builder.h"
#include "agx_nir.h"

/*
 * AGX generally flushes FP32 denorms. However, the min/max instructions do not
 * as they are implemented with cmpsel. We need to flush the results of fp32
 * min/max for correctness. Doing so naively will generate redundant flushes, so
 * this pass tries to be clever and elide flushes when possible.
 *
 */
static bool
normal_bound(nir_scalar value, bool is_max)
{
   value = nir_scalar_chase_movs(value);
   if (!nir_scalar_is_const(value))
      return false;
   const uint32_t bits = nir_scalar_as_uint(value);
   const unsigned exponent = (bits >> 23) & 255;
   return exponent > 0 && exponent < 255 && (bits >> 31) == !is_max;
}

static bool
could_be_denorm(nir_scalar s, unsigned depth, unsigned *budget)
{
   if (!*budget)
      return true;
   --*budget;
   s = nir_scalar_chase_movs(s);
   /* Constants can be denorms only if they are denorms. */
   if (nir_scalar_is_const(s)) {
      /* Classify FP32 bits before promotion to double normalizes them. */
      uint32_t bits = nir_scalar_as_uint(s);
      return (bits & 0x7f800000u) == 0 && (bits & 0x007fffffu) != 0;
   }

   if (!depth)
      return true;
   if (nir_def_instr(s.def)->type == nir_instr_type_phi) {
      nir_phi_instr *phi = nir_instr_as_phi(nir_def_instr(s.def));
      nir_foreach_phi_src(src, phi) {
         if (could_be_denorm(nir_get_scalar(src->src.ssa, s.comp), depth - 1,
                             budget))
            return true;
      }
      return false;
   }
   if (nir_scalar_is_alu(s) && nir_scalar_alu_op(s) == nir_op_bcsel)
      return could_be_denorm(nir_scalar_chase_alu_src(s, 1), depth - 1,
                             budget) ||
             could_be_denorm(nir_scalar_chase_alu_src(s, 2), depth - 1, budget);

   /* Floating-point instructions flush denormals, so ALU results can only be
    * denormal if they are not from a float instruction. Crucially fmin/fmax
    * flushes in NIR, so this pass handles chains of fmin/fmax properly.
    */
   if (nir_scalar_is_alu(s)) {
      nir_op op = nir_scalar_alu_op(s);
      if (op == nir_op_fmin_agx || op == nir_op_fmax_agx) {
         nir_scalar a = nir_scalar_chase_alu_src(s, 0);
         nir_scalar b = nir_scalar_chase_alu_src(s, 1);
         const bool is_max = op == nir_op_fmax_agx;
         if (normal_bound(a, is_max) || normal_bound(b, is_max))
            return false;
         return could_be_denorm(a, depth - 1, budget) ||
                could_be_denorm(b, depth - 1, budget);
      }
      nir_alu_type T = nir_op_infos[op].output_type;
      return nir_alu_type_get_base_type(T) != nir_type_float;
   }

   /* Otherwise, assume it could be denormal (say, loading from a buffer). */
   return true;
}

static bool
lower(nir_builder *b, nir_alu_instr *alu, void *data)
{
   if ((alu->op != nir_op_fmin && alu->op != nir_op_fmax) ||
       (alu->def.bit_size != 32))
      return false;

   /* Lower the op, we'll fix up the denorms right after. */
   if (alu->op == nir_op_fmax)
      alu->op = nir_op_fmax_agx;
   else
      alu->op = nir_op_fmin_agx;

   /* Without a flush-to-zero execution mode, whether a denormal result is
    * flushed is implementation-defined (Vulkan, "Precision and Operation of
    * SPIR-V Instructions": any denormal an instruction produces may be
    * flushed), so the min/max result may stay denormal. Only a shader that
    * asks for flushing needs it. Callers that keep flushing every min/max
    * pass false.
    */
   const bool *ftz_only = data;
   const unsigned mode = b->shader->info.float_controls_execution_mode;
   if (*ftz_only && !(mode & FLOAT_CONTROLS_DENORM_FLUSH_TO_ZERO_FP32))
      return true;

   bool canonicalize = false;
   for (unsigned c = 0; c < alu->def.num_components; ++c) {
      nir_scalar scalar = nir_get_scalar(&alu->def, c);
      nir_scalar src0 = nir_scalar_chase_alu_src(scalar, 0);
      nir_scalar src1 = nir_scalar_chase_alu_src(scalar, 1);
      const bool is_max = alu->op == nir_op_fmax_agx;
      if (normal_bound(src0, is_max) || normal_bound(src1, is_max))
         continue;
      unsigned budget = 64;
      canonicalize |=
         could_be_denorm(src0, 8, &budget) || could_be_denorm(src1, 8, &budget);
   }

   if (canonicalize) {
      b->cursor = nir_after_instr(&alu->instr);
      b->fp_math_ctrl = nir_fp_no_fast_math;

      nir_def *canonicalized = nir_fadd_imm(b, &alu->def, -0.0);
      nir_def_rewrite_uses_after(&alu->def, canonicalized);
   }

   return true;
}

bool
agx_nir_lower_fminmax(nir_shader *s, bool ftz_only)
{
   return nir_shader_alu_pass(s, lower, nir_metadata_control_flow, &ftz_only);
}
