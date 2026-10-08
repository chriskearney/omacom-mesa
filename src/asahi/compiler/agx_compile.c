/*
 * Copyright 2021 Alyssa Rosenzweig
 * Copyright 2020 Collabora Ltd.
 * Copyright 2016 Broadcom
 * SPDX-License-Identifier: MIT
 */

#include "agx_compile.h"
#include "agx_replay_abi.h"
#include "nir_serialize.h"
#include "util/mesa-blake3.h"
#include <dirent.h>
#include <fcntl.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <unistd.h>
#include <limits.h>

/* Only active for opt-in compiler failure capture. Never shared by threads. */
static _Thread_local FILE *agx_failure_ir;

#include "asahi/clc/asahi_clc.h"
#include "asahi/isa/disasm.h"
#include "asahi/layout/layout.h"
#include "asahi/lib/agx_abi.h"
#include "compiler/nir/nir_builder.h"
#include "compiler/nir/nir_format_convert.h"
#include "util/bitset.h"
#include "util/glheader.h"
#include "util/list.h"
#include "util/macros.h"
#include "util/u_debug.h"
#include "util/u_call_once.h"
#include "util/u_dynarray.h"
#include "agx_builder.h"
#include "agx_compiler.h"
#include "agx_debug.h"
#include "agx_nir.h"
#include "agx_opcodes.h"
#include "agx3_pack.h"
#include "glsl_types.h"
#include "nir.h"
#include "nir_builtin_builder.h"
#include "nir_intrinsics.h"
#include "nir_intrinsics_indices.h"
#include "shader_enums.h"

/* Cache-line align shader programs. This matches the prop compiler. */
#define AGX_CODE_ALIGN 0x80

/* clang-format off */
static const struct debug_named_value agx_debug_options[] = {
   {"shaders",   AGX_DBG_SHADERS,	"Dump shaders in NIR and AIR"},
   {"shaderdb",  AGX_DBG_SHADERDB,	"Print statistics"},
   {"verbose",   AGX_DBG_VERBOSE,	"Disassemble verbosely"},
   {"internal",  AGX_DBG_INTERNAL,	"Dump even internal shaders"},
   {"novalidate",AGX_DBG_NOVALIDATE,"Skip IR validation in debug builds"},
   {"noopt",     AGX_DBG_NOOPT,     "Disable backend optimizations"},
   {"wait",      AGX_DBG_WAIT,      "Wait after all async instructions"},
   {"nopreamble",AGX_DBG_NOPREAMBLE,"Do not use shader preambles"},
   {"demand",    AGX_DBG_DEMAND,    "Bound tightly to register demand"},
   {"nosched",   AGX_DBG_NOSCHED,   "Do not schedule the shader"},
   {"spill",     AGX_DBG_SPILL,     "Spill (almost) everything"},
   {"nopromote", AGX_DBG_NOPROMOTE, "Do not promote constants to uniforms"},
   {"remat",     AGX_DBG_REMAT,     "Experiment with AGX3B constant rematerialization"},
   DEBUG_NAMED_VALUE_END
};
/* clang-format on */

DEBUG_GET_ONCE_FLAGS_OPTION(agx_compiler_debug, "AGX_MESA_DEBUG",
                            agx_debug_options, 0)

int agx_compiler_debug = 0;

static void
agx_init_compiler_debug(void)
{
   agx_compiler_debug = debug_get_option_agx_compiler_debug();
}

/*
 * Pad binary to a given alignment and return aligned offset into the binary.
 */
static unsigned
agx_pad_binary(struct util_dynarray *dyn, uint32_t align)
{
   if (dyn->size % align) {
      unsigned ngrow = align - (dyn->size % align);
      memset(util_dynarray_grow_bytes(dyn, ngrow, 1), 0, ngrow);
   }

   assert((dyn->size % align) == 0);
   return dyn->size;
}

uint64_t
agx_get_compiler_debug(void)
{
   static util_once_flag once = UTIL_ONCE_FLAG_INIT;
   util_call_once(&once, agx_init_compiler_debug);
   return agx_compiler_debug;
}

static agx_index
agx_cached_preload(agx_context *ctx, unsigned base, enum agx_size size)
{
   if (agx_is_null(ctx->preloaded[base])) {
      agx_block *block = agx_start_block(ctx);
      agx_builder b = agx_init_builder(ctx, agx_before_block(block));
      ctx->preloaded[base] = agx_preload(&b, agx_register(base, size));
   }

   return ctx->preloaded[base];
}

static agx_index
agx_tess_coord_x(agx_builder *b)
{
   return agx_cached_preload(b->shader, 4, AGX_SIZE_32);
}

static agx_index
agx_tess_coord_y(agx_builder *b)
{
   return agx_cached_preload(b->shader, 6, AGX_SIZE_32);
}

static agx_index
agx_vertex_id_zero_base(agx_builder *b)
{
   return agx_cached_preload(b->shader, AGX_ABI_VIN_VERTEX_ID_ZERO_BASE,
                             AGX_SIZE_32);
}

static agx_index
agx_vertex_id(agx_builder *b)
{
   return agx_cached_preload(b->shader, AGX_ABI_VIN_VERTEX_ID, AGX_SIZE_32);
}

static agx_index
agx_instance_id(agx_builder *b)
{
   return agx_cached_preload(b->shader, AGX_ABI_VIN_INSTANCE_ID, AGX_SIZE_32);
}

#define VARYING_NUM_COMPONENTS (VARYING_SLOT_MAX * 4)

struct coefficient_info {
   BITSET_DECLARE(smooth, VARYING_NUM_COMPONENTS);
   BITSET_DECLARE(flat, VARYING_NUM_COMPONENTS);
   BITSET_DECLARE(noperspective, VARYING_NUM_COMPONENTS);
};

static BITSET_WORD *
bitset_for_interp(struct coefficient_info *info, enum glsl_interp_mode mode)
{
   /* clang-format off */
   switch (mode) {
   case INTERP_MODE_NONE:
   case INTERP_MODE_SMOOTH:         return info->smooth;
   case INTERP_MODE_NOPERSPECTIVE:  return info->noperspective;
   case INTERP_MODE_FLAT:           return info->flat;
   default:                         UNREACHABLE("invalid interp mode");
   }
   /* clang-format on */
}

static bool
gather_cf(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   /* First handle frag coord loads */
   struct coefficient_info *info = data;
   if (intr->intrinsic == nir_intrinsic_load_frag_coord_z) {
      BITSET_SET(info->noperspective, VARYING_SLOT_POS + 2);
      return false;
   }
   if (intr->intrinsic == nir_intrinsic_load_frag_coord_w) {
      BITSET_SET(info->noperspective, VARYING_SLOT_POS + 3);
      return false;
   }

   /* Look for input loads and grab the instruction with the interp mode */
   nir_intrinsic_instr *bary;
   unsigned nr = 1;

   if (intr->intrinsic == nir_intrinsic_load_coefficients_agx) {
      bary = intr;
      /* Always load a scalar */
   } else if (intr->intrinsic == nir_intrinsic_load_interpolated_input) {
      bary = nir_src_as_intrinsic(intr->src[0]);
      nr = intr->num_components;

      /* Perspective interpolation internally reads W */
      if (nir_intrinsic_interp_mode(bary) != INTERP_MODE_NOPERSPECTIVE)
         BITSET_SET(info->noperspective, VARYING_SLOT_POS + 3);
   } else {
      return false;
   }

   BITSET_WORD *set = bitset_for_interp(data, nir_intrinsic_interp_mode(bary));
   nir_io_semantics sem = nir_intrinsic_io_semantics(intr);
   nir_src *offset = nir_get_io_offset_src(intr);

   /* Mark the exact range for direct loads to minimize CF registers, but mark a
    * conservative bounding range for indirect array access.
    */
   if (nir_src_is_const(*offset)) {
      unsigned location = sem.location + nir_src_as_uint(*offset);
      unsigned start_comp = (location * 4) + nir_intrinsic_component(intr);

      BITSET_SET_COUNT(set, start_comp, nr);
   } else {
      unsigned start_comp = (sem.location * 4) + nir_intrinsic_component(intr);
      bool compact = sem.location == VARYING_SLOT_CLIP_DIST0 ||
                     sem.location == VARYING_SLOT_CLIP_DIST1;
      unsigned stride = compact ? 1 : 4;

      /* For now we have to assign CF for the whole vec4 to make indirect
       * indexiing work. This could be optimized later.
       */
      nr = stride;

      for (unsigned i = 0; i < sem.num_slots; ++i) {
         BITSET_SET_COUNT(set, start_comp + (i * stride), nr);
      }
   }

   return false;
}

/*
 * We assign all coefficient registers up front to ensure we have a consistent
 * layout required for indirects to work.
 */
static void
assign_coefficient_regs(nir_shader *nir, struct agx_varyings_fs *var)
{
   struct coefficient_info info = {0};
   nir_shader_intrinsics_pass(nir, gather_cf, nir_metadata_all, &info);

   /* W */
   if (BITSET_TEST(info.noperspective, VARYING_SLOT_POS + 3)) {
      var->bindings[var->nr_bindings++] = (struct agx_cf_binding){
         .cf_base = var->nr_cf++,
         .slot = VARYING_SLOT_POS,
         .offset = 3,
         .count = 1,
         .smooth = true,
      };
   }

   /* Z */
   if (BITSET_TEST(info.noperspective, VARYING_SLOT_POS + 2)) {
      var->bindings[var->nr_bindings++] = (struct agx_cf_binding){
         .cf_base = var->nr_cf++,
         .slot = VARYING_SLOT_POS,
         .offset = 2,
         .count = 1,
         .smooth = true,
      };

      var->reads_z = true;
   }

   static_assert(VARYING_SLOT_POS == 0, "special and handled first");

   for (unsigned i = VARYING_SLOT_POS + 1; i < VARYING_SLOT_MAX; ++i) {
      bool smooth = BITSET_TEST_COUNT(info.smooth, i * 4, 4);
      bool flat = BITSET_TEST_COUNT(info.flat, i * 4, 4);
      bool noperspective = BITSET_TEST_COUNT(info.noperspective, i * 4, 4);

      if (!(smooth || flat || noperspective))
         continue;

      /* From the GLSL 4.60 spec ("Input Layout Qualifiers"):
       *
       *    when location aliasing, the aliases sharing the location must have
       *    the same underlying numerical type and bit width (floating-point or
       *    integer, 32-bit versus 64-bit, etc.) and the same auxiliary storage
       *    and interpolation qualification.
       *
       * SPIR-V should obey this as well although the spec text is muddier.
       */
      assert((smooth + flat + noperspective) == 1 &&
             "slots must have consistent interpolation");

      BITSET_WORD *set = smooth ? info.smooth
                         : flat ? info.flat
                                : info.noperspective;

      /* Find the start offset */
      unsigned offset = 0;
      for (offset = 0; offset < 4 && !BITSET_TEST(set, (i * 4) + offset);
           ++offset)
         ;

      /* Find the end offset. TODO: Do we ever need to split into two bindings
       * to handle e.g. x_zw read masks?
       */
      unsigned count = 0;
      for (unsigned c = offset; c < 4; ++c) {
         if (BITSET_TEST(set, (i * 4) + c))
            count = c - offset + 1;
      }
      assert(count >= 1 && (count + offset) <= 4);

      var->bindings[var->nr_bindings++] = (struct agx_cf_binding){
         .cf_base = var->nr_cf,
         .slot = i,
         .offset = offset,
         .count = count,
         .smooth = !flat,
         .perspective = smooth,
      };

      var->nr_cf += count;
   }
}

static agx_index
agx_get_cf(agx_context *ctx, gl_varying_slot slot, unsigned offset)
{
   struct agx_varyings_fs *varyings = &ctx->out->varyings.fs;

   /* We already have an appropriate binding, find it */
   for (unsigned b = 0; b < varyings->nr_bindings; ++b) {
      if (varyings->bindings[b].slot == slot &&
          (slot != VARYING_SLOT_POS ||
           offset == varyings->bindings[b].offset)) {

         signed cf_offset = offset - varyings->bindings[b].offset;
         assert(cf_offset >= 0);

         return agx_immediate(varyings->bindings[b].cf_base + cf_offset);
      }
   }

   UNREACHABLE("all coefficient registers preassigned");
}

/* Builds a 64-bit hash table key for an index */
static uint64_t
agx_index_to_key(agx_index idx)
{
   STATIC_ASSERT(sizeof(idx) <= sizeof(uint64_t));

   uint64_t key = 0;
   memcpy(&key, &idx, sizeof(idx));
   return key;
}

/*
 * Extract a single channel out of a vector source. We split vectors with
 * p_split so we can use the split components directly, without emitting a
 * machine instruction. This has advantages of RA, as the split can usually be
 * optimized away.
 */
static agx_index
agx_emit_extract(agx_builder *b, agx_index vec, unsigned channel)
{
   agx_index *components = _mesa_hash_table_u64_search(b->shader->allocated_vec,
                                                       agx_index_to_key(vec));

   assert(components != NULL && "missing agx_emit_collect_to");

   return components[channel];
}

static agx_index
agx_extract_nir_src(agx_builder *b, nir_src src, unsigned channel)
{
   agx_index idx = agx_src_index(&src);

   /* We only deal with scalars, extract a single scalar if needed */
   if (nir_src_num_components(src) > 1)
      return agx_emit_extract(b, idx, channel);
   else
      return idx;
}

static void
agx_cache_collect(agx_builder *b, agx_index dst, unsigned nr_srcs,
                  agx_index *srcs)
{
   /* Lifetime of a hash table entry has to be at least as long as the table */
   agx_index *channels = ralloc_array(b->shader, agx_index, nr_srcs);

   for (unsigned i = 0; i < nr_srcs; ++i)
      channels[i] = srcs[i];

   _mesa_hash_table_u64_insert(b->shader->allocated_vec, agx_index_to_key(dst),
                               channels);
}

/*
 * Combine multiple scalars into a vector destination. This corresponds to
 * collect, lowered to moves (a shuffle in general) after register allocation.
 *
 * To optimize vector extractions, we record the individual channels
 */
static agx_instr *
agx_emit_collect_to(agx_builder *b, agx_index dst, unsigned nr_srcs,
                    agx_index *srcs)
{
   agx_cache_collect(b, dst, nr_srcs, srcs);

   if (nr_srcs == 1)
      return agx_mov_to(b, dst, srcs[0]);

   agx_instr *I = agx_collect_to(b, dst, nr_srcs);

   agx_foreach_src(I, s)
      I->src[s] = srcs[s];

   return I;
}

static agx_index
agx_emit_collect(agx_builder *b, unsigned nr_srcs, agx_index *srcs)
{
   agx_index dst = agx_vec_temp(b->shader, srcs[0].size, nr_srcs);
   agx_emit_collect_to(b, dst, nr_srcs, srcs);
   return dst;
}

static agx_index
agx_vec2(agx_builder *b, agx_index s0, agx_index s1)
{
   return agx_emit_collect(b, 2, (agx_index[]){s0, s1});
}

static agx_index
agx_pad_to_32(agx_builder *b, agx_index s)
{
   assert(s.size == AGX_SIZE_16);
   assert(agx_channels(s) == 1);

   agx_index srcs[2] = {s, agx_undef(AGX_SIZE_16)};
   agx_index dst = agx_vec_temp(b->shader, AGX_SIZE_32, 1);
   agx_emit_collect_to(b, dst, 2, srcs);
   return dst;
}

static agx_index
agx_recollect_vector(agx_builder *b, nir_src vec)
{
   agx_index comps[4];
   unsigned nr = nir_src_num_components(vec);

   for (unsigned i = 0; i < nr; ++i)
      comps[i] = agx_extract_nir_src(b, vec, i);

   return agx_emit_collect(b, nr, comps);
}

/*
 * Extract the lower or upper N-bits from a (2*N)-bit quantity. We use a split
 * without null destinations to let us CSE (and coalesce) the splits when both x
 * and y are split.
 */
static agx_instr *
agx_subdivide_to(agx_builder *b, agx_index dst, agx_index s0, unsigned comp)
{
   assert((s0.size == (dst.size + 1)) && "only 2x subdivide handled");
   assert((comp == 0 || comp == 1) && "too many components");

   /* Handle immediates specially so we don't have to constant fold splits. */
   if (s0.type == AGX_INDEX_IMMEDIATE) {
      unsigned bits = 16 * agx_size_align_16(dst.size);
      return agx_mov_imm_to(b, dst, (s0.value >> bits) & BITFIELD64_MASK(bits));
   }

   agx_instr *split = agx_split(b, 2, s0);
   split->dest[comp] = dst;
   split->dest[1 - comp] = agx_temp(b->shader, dst.size);
   return split;
}

void
agx_block_add_successor(agx_block *block, agx_block *successor)
{
   assert(block != NULL && successor != NULL);

   /* Cull impossible edges */
   if (block->unconditional_jumps)
      return;

   for (unsigned i = 0; i < ARRAY_SIZE(block->successors); ++i) {
      if (block->successors[i]) {
         if (block->successors[i] == successor)
            return;
         else
            continue;
      }

      block->successors[i] = successor;
      util_dynarray_append(&successor->predecessors, block);
      return;
   }

   UNREACHABLE("Too many successors");
}

/*
 * Splits an n-component vector (vec) into n scalar destinations (dests) using a
 * split pseudo-instruction.
 *
 * Pre-condition: dests is filled with agx_null().
 */
static void
agx_emit_split(agx_builder *b, agx_index *dests, agx_index vec, unsigned n)
{
   agx_instr *I = agx_split(b, n, vec);

   agx_foreach_dest(I, d) {
      dests[d] = agx_temp(b->shader, vec.size);
      I->dest[d] = dests[d];
   }
}

static void
agx_emit_cached_split(agx_builder *b, agx_index vec, unsigned n)
{
   agx_index dests[4] = {agx_null(), agx_null(), agx_null(), agx_null()};
   agx_emit_split(b, dests, vec, n);
   agx_cache_collect(b, vec, n, dests);
}

static void
agx_emit_load_const(agx_builder *b, nir_load_const_instr *instr)
{
   /* Ensure we've been scalarized and bit size lowered */
   unsigned bit_size = instr->def.bit_size;
   assert(instr->def.num_components == 1);

   /* Emit move, later passes can inline/push if useful */
   agx_mov_imm_to(b, agx_def_index(&instr->def),
                  nir_const_value_as_uint(instr->value[0], bit_size));
}

/*
 * Implement mul_high of 32-bit sources by doing a 32x32->64-bit multiply and
 * extracting only the high word.
 */
static agx_instr *
agx_mul_high_to(agx_builder *b, agx_index dst, agx_index P, agx_index Q,
                bool is_signed)
{
   assert(P.size == Q.size && "source sizes must match");
   assert(P.size == dst.size && "dest size must match");
   assert(P.size != AGX_SIZE_64 && "64x64 multiply should have been lowered");

   static_assert(AGX_SIZE_64 == (AGX_SIZE_32 + 1), "enum wrong");
   static_assert(AGX_SIZE_32 == (AGX_SIZE_16 + 1), "enum wrong");

   if (!is_signed) {
      P = agx_abs(P);
      Q = agx_abs(Q);
   }

   agx_index product = agx_temp(b->shader, P.size + 1);
   agx_imad_to(b, product, P, Q, agx_zero(), 0);

   return agx_subdivide_to(b, dst, product, 1);
}

static enum agx_format
agx_format_for_pipe(enum pipe_format format)
{
#define CASE(x)                                                                \
   if (format == (enum pipe_format)AIL_ISA_FORMAT_##x)                         \
      return AGX_FORMAT_##x;

   CASE(I8);
   CASE(I16);
   CASE(I32);
   CASE(F16);
   CASE(U8NORM);
   CASE(S8NORM);
   CASE(U16NORM);
   CASE(S16NORM);
   CASE(RGB10A2);
   CASE(SRGBA8);
   CASE(RG11B10F);
   CASE(RGB9E5);

#undef CASE
   UNREACHABLE("Invalid format");
}

static agx_index
cf_for_intrinsic(agx_builder *b, nir_intrinsic_instr *intr)
{
   /* Determine the base location, taking into account a constant offset */
   unsigned location = nir_intrinsic_io_semantics(intr).location;
   bool compact = location == VARYING_SLOT_CLIP_DIST0 ||
                  location == VARYING_SLOT_CLIP_DIST1;

   nir_src *offset = nir_get_io_offset_src(intr);
   if (nir_src_is_const(*offset)) {
      /* XXX: NIR is broken and uses constant offsets in slots but dynamic
       * offsets in scalars for compact varyings. This needs to be fixed
       * upstream.
       */
      location += nir_src_as_uint(*offset);
   }

   agx_index I = agx_get_cf(b->shader, location, nir_intrinsic_component(intr));

   /* If we have a non-constant offset, we add it to the CF. Offsets are in
    * vec4 slots (unless we're compact) but the CF is in components, so we need
    * to shift the offset by 2 before adding.
    */
   if (!nir_src_is_const(*offset)) {
      I = agx_iadd(b, I, agx_src_index(offset), compact ? 0 : 2);
   }

   return I;
}

static enum agx_interpolation
agx_interp_for_bary(nir_intrinsic_instr *bary, agx_index *sample_index)
{
   switch (bary->intrinsic) {
   case nir_intrinsic_load_barycentric_pixel:
      return AGX_INTERPOLATION_CENTER;

   case nir_intrinsic_load_barycentric_centroid:
      return AGX_INTERPOLATION_CENTROID;

   case nir_intrinsic_load_barycentric_at_sample:
      *sample_index = agx_src_index(&bary->src[0]);
      return AGX_INTERPOLATION_SAMPLE;

   default:
      UNREACHABLE("should have been lowered");
   }
}

/* AGX3 removed AGX2's fused projective iterator. Build Apple's witnessed
 * scalar software sequence while values are still in SSA so register
 * allocation can place the temporary RHW, reciprocal, and channels safely.
 * Dynamic coefficient indices and explicit at-sample coordinates remain on
 * the legacy IR path so the AGX3 packer rejects them instead of guessing. */
static bool
agx_emit_load_vary_agx3(agx_builder *b, agx_index dest, agx_index cf,
                        agx_index sample_index, unsigned components,
                        enum agx_interpolation interp, bool perspective,
                        nir_alu_type dest_type)
{
   if (b->shader->key->dev.encoding == AGX_ISA_ENCODING_AGX2 ||
       (dest.size != AGX_SIZE_16 && dest.size != AGX_SIZE_32) ||
       cf.type != AGX_INDEX_IMMEDIATE ||
       sample_index.type != AGX_INDEX_IMMEDIATE || sample_index.value != 0 ||
       (interp != AGX_INTERPOLATION_CENTER &&
        interp != AGX_INTERPOLATION_CENTROID))
      return false;

   if (interp == AGX_INTERPOLATION_CENTROID &&
       b->shader->key->dev.encoding == AGX_ISA_ENCODING_AGX3B) {
      /* Native itrcoords takes the original, unshifted coverage word and
       * produces coordinates consumed by every centroid interpolant. Keep
       * the coordinate value in SSA so it survives RHW and channel loads. */
      agx_index coverage = agx_get_sr(b, 16, AGX_SR_INPUT_SAMPLE_MASK);
      sample_index = agx_temp(b->shader, AGX_SIZE_32);
      agx_centroid_coords_agx3_to(b, sample_index, coverage);
   }

   agx_index reciprocal_w = agx_null();
   if (perspective) {
      agx_index rhw_cf = agx_get_cf(b->shader, VARYING_SLOT_POS, 3);
      if (rhw_cf.type != AGX_INDEX_IMMEDIATE)
         return false;

      agx_index rhw = agx_temp(b->shader, AGX_SIZE_32);
      if (b->shader->key->dev.encoding == AGX_ISA_ENCODING_AGX3B)
         agx_rhw_interp_agx3_to(b, rhw, rhw_cf, sample_index, interp);
      else
         agx_interp_agx3_to(b, rhw, rhw_cf, sample_index, interp, true);
      reciprocal_w = b->shader->key->dev.encoding == AGX_ISA_ENCODING_AGX3B
                        ? agx_persp_rcp_agx3(b, rhw, rhw_cf.value)
                        : agx_rcp(b, rhw);
   }

   agx_index channels[4];
   for (unsigned c = 0; c < components; ++c) {
      agx_index channel_cf = cf;
      channel_cf.value += c;

      agx_index raw = agx_temp(b->shader, AGX_SIZE_32);
      agx_interp_agx3_to(b, raw, channel_cf, sample_index, interp,
                         perspective);
      if (perspective &&
          b->shader->key->dev.encoding == AGX_ISA_ENCODING_AGX3B) {
         /* Native f32mul.ep4 identifies the coefficient involved in this
          * projective divide. Preserve it explicitly rather than trying to
          * reconstruct SSA provenance after physical register allocation. */
         channels[c] = agx_persp_mul_agx3(b, raw, reciprocal_w, channel_cf.value);
      } else {
         channels[c] = perspective ? agx_fmul(b, raw, reciprocal_w) : raw;
      }

      /* AGX3 interpolation is natively 32-bit. Vulkan's
       * storageInputOutput16 feature nevertheless permits a logical 16-bit
       * fragment input, so narrow only after interpolation. Vertex output
       * lowering performs the matching type-correct widening before ST_VARY.
       * This preserves integer bits for flat varyings and applies the required
       * floating-point conversion for smooth/noperspective float varyings. */
      if (dest.size == AGX_SIZE_16) {
         agx_index narrowed = agx_temp(b->shader, AGX_SIZE_16);
         if (nir_alu_type_get_base_type(dest_type) == nir_type_float)
            agx_fmov_to(b, narrowed, channels[c]);
         else
            agx_subdivide_to(b, narrowed, channels[c], 0);

         channels[c] = narrowed;
      }
   }

   agx_emit_collect_to(b, dest, components, channels);
   return true;
}

static void
agx_emit_load_vary(agx_builder *b, agx_index dest, nir_intrinsic_instr *instr)
{
   ASSERTED unsigned components = instr->num_components;
   nir_intrinsic_instr *bary = nir_src_as_intrinsic(instr->src[0]);

   assert(components >= 1 && components <= 4);

   agx_index sample_index = agx_zero();
   enum agx_interpolation interp = agx_interp_for_bary(bary, &sample_index);

   bool perspective =
      nir_intrinsic_interp_mode(bary) != INTERP_MODE_NOPERSPECTIVE;

   agx_index I = cf_for_intrinsic(b, instr);

   if (agx_emit_load_vary_agx3(b, dest, I, sample_index, components, interp,
                               perspective, nir_intrinsic_dest_type(instr))) {
      agx_emit_cached_split(b, dest, components);
      return;
   }

   /* For perspective interpolation, we project (multiply by 1/W) */
   if (perspective) {
      agx_index J = agx_get_cf(b->shader, VARYING_SLOT_POS, 3);
      agx_iterproj_to(b, dest, I, J, sample_index, components, interp);
   } else {
      agx_iter_to(b, dest, I, sample_index, components, interp);
   }

   agx_emit_cached_split(b, dest, components);
}

static void
agx_wait_pixel_mask(agx_builder *b, uint32_t mask)
{
   /* Background programs do not need to wait as they are the eldest pixels */
   if (b->shader->key->fs.ignore_tib_dependencies) {
      assert(b->shader->nir->info.internal);
      return;
   }

   /* The native linker holds color and depth/coverage ownership across
    * every sample invocation, then releases once when the pixel is done. */
   if (b->shader->key->dev.encoding == AGX_ISA_ENCODING_AGX3B &&
       b->shader->key->fs.inside_sample_loop)
      mask &= ~0xd;

   /* No need to wait twice on a fence */
   mask &= ~b->shader->already_pixel_waited;
   if (mask == 0) {
      return;
   }

   agx_wait_pix(b, mask);

   /* Only mark the fence as waited if we're not in control flow. Eventually we
    * should do something smarter with a dataflow.
    */
   if (b->shader->total_nesting == 0) {
      b->shader->already_pixel_waited |= mask;
   }
}

/*
 * AGX3's `lstcol` carries only a width, so unlike AGX2's st_tile it cannot
 * normalize on the way out: an 8-bit normalized target has to arrive already
 * packed, two channels to a 16-bit register. `fltnorm` is that pack -- it
 * clamps and scales two 32-bit floats into the two halves of one 16-bit
 * register -- and Apple's own G17P fragment shaders spend exactly
 * ceil(channels / 2) of them before every 8-bit colour store.
 *
 * Returns the number of packed 16-bit registers, or 0 if this store is not one
 * this path can pack, in which case the caller keeps the unpacked value and
 * the AGX3 packer refuses the shader rather than storing raw floats through a
 * width that would truncate them.
 */
static unsigned
agx_pack_normalized_agx3(agx_builder *b, const agx_index *channels,
                         unsigned nr_channels, enum agx_format format,
                         agx_index *packed)
{
   if (b->shader->key->dev.encoding == AGX_ISA_ENCODING_AGX2)
      return 0;

   if (format != AGX_FORMAT_RGB10A2 &&
       format != AGX_FORMAT_U8NORM && format != AGX_FORMAT_S8NORM)
      return 0;

   if (nr_channels < 1 || nr_channels > 4 ||
       (format == AGX_FORMAT_RGB10A2 && nr_channels != 4))
      return 0;

   for (unsigned i = 0; i < nr_channels; ++i)
      if (channels[i].size != AGX_SIZE_16 && channels[i].size != AGX_SIZE_32)
         return 0;

   /* The grounded normalized packs consume f32. Gallium epilogs can
    * receive f16 colours, including the compositor's RGB10A2 output, so
    * widen those lanes explicitly before the physical normalized pack. */
   agx_index wide[4];
   for (unsigned i = 0; i < nr_channels; ++i) {
      wide[i] = channels[i];
      if (channels[i].size == AGX_SIZE_16) {
         wide[i] = agx_temp(b->shader, AGX_SIZE_32);
         agx_fmov_to(b, wide[i], channels[i]);
      }
   }

   if (format == AGX_FORMAT_RGB10A2) {
      packed[0] = agx_pack_rgb10a2_agx3(b, wide[0], wide[1],
                                      wide[2], wide[3]);
      return 1;
   }

   unsigned n = 0;
   for (unsigned i = 0; i < nr_channels; i += 2) {
      const agx_index hi =
         (i + 1 < nr_channels) ? wide[i + 1] : agx_null();
      /* The destination is a 16-bit half, not the sources' 32 bits, so the
       * generated temp-allocating helper cannot be used here. */
      agx_index dest = agx_temp(b->shader, AGX_SIZE_16);
      agx_fltnorm_agx3_to(b, dest, wide[i], hi, format);
      packed[n++] = dest;
   }
   return n;
}

/* Pure 8-bit integer outputs have already been clamped and narrowed to 16-bit
 * NIR lanes. AGX3 tile traffic instead carries two adjacent bytes in each
 * 16-bit register, so explicitly reproduce that physical packing here. */
static unsigned
agx_pack_integer_8_agx3(agx_builder *b, const agx_index *channels,
                        unsigned nr_channels, enum agx_format format,
                        agx_index *packed)
{
   if (b->shader->key->dev.encoding != AGX_ISA_ENCODING_AGX3B ||
       format != AGX_FORMAT_I8 || nr_channels < 1 || nr_channels > 4)
      return 0;

   for (unsigned i = 0; i < nr_channels; ++i)
      if (channels[i].size != AGX_SIZE_16)
         return 0;

   unsigned n = 0;
   for (unsigned i = 0; i < nr_channels; i += 2) {
      if (i + 1 == nr_channels) {
         packed[n++] = channels[i];
         continue;
      }

      agx_index upper = agx_pad_to_32(b, channels[i + 1]);
      agx_index shifted = agx_temp(b->shader, AGX_SIZE_32);
      agx_bfi_to(b, shifted, agx_immediate(0), upper,
                 agx_immediate(8), 0);
      agx_index lower = agx_and(b, channels[i], agx_immediate(0xff));
      agx_index pair = agx_temp(b->shader, AGX_SIZE_16);
      agx_bitop_to(b, pair, lower, shifted, 0xe /* OR */);
      packed[n++] = pair;
   }

   return n;
}

static agx_instr *
agx_emit_local_store_pixel(agx_builder *b, nir_intrinsic_instr *instr)
{
   bool explicit = nir_intrinsic_explicit_coord(instr);

   /* TODO: Reverse-engineer interactions with MRT */
   if (b->shader->stage == MESA_SHADER_FRAGMENT) {
      agx_wait_pixel_mask(b, 0xC);
   }

   /* Compact the registers according to the mask */
   agx_index compacted[4] = {agx_null()};

   unsigned compact_count = 0;
   u_foreach_bit(i, nir_intrinsic_write_mask(instr)) {
      compacted[compact_count++] = agx_extract_nir_src(b, instr->src[0], i);
   }

   const enum agx_format format =
      agx_format_for_pipe(nir_intrinsic_format(instr));

   agx_index packed[4] = {agx_null()};
   unsigned nr_packed =
      agx_pack_normalized_agx3(b, compacted, compact_count, format, packed);
   if (nr_packed == 0)
      nr_packed = agx_pack_integer_8_agx3(
         b, compacted, compact_count, format, packed);

   /* G17P's tile store encodes only the lane width.  Unlike AGX2 it cannot
    * narrow an f32 colour to the f16 lanes named by that width, so make the
    * conversion explicit before collecting the store payload.  Keep AGX3c
    * unchanged until its matching narrowing ALU form is grounded. */
   if (b->shader->key->dev.encoding == AGX_ISA_ENCODING_AGX3B &&
       format == AGX_FORMAT_F16 && nr_packed == 0) {
      nr_packed = compact_count;
      for (unsigned i = 0; i < compact_count; ++i) {
         if (compacted[i].size == AGX_SIZE_32) {
            packed[i] = agx_temp(b->shader, AGX_SIZE_16);
            agx_fmov_to(b, packed[i], compacted[i]);
         } else {
            packed[i] = compacted[i];
         }
      }
   }

   agx_index collected =
      nr_packed ? agx_emit_collect(b, nr_packed, packed)
                : agx_emit_collect(b, compact_count, compacted);
   agx_index coords = explicit ? agx_src_index(&instr->src[2]) : agx_null();

   agx_index samples = agx_src_index(&instr->src[1]);
   if (b->shader->stage == MESA_SHADER_FRAGMENT &&
       b->shader->key->dev.encoding == AGX_ISA_ENCODING_AGX3B) {
      /* SR39's low nibble is the live store coverage. The upper nibble
       * retains the original target set and must not reach LSTCOL. This
       * applies to background clears too: a partially covered pixel invokes
       * its background only for samples not supplied by foreground draws. */
      agx_index coverage = agx_get_sr_coverage(b, 16, AGX_SR_COVERAGE_MASK);
      coverage = agx_and(b, coverage, agx_immediate(15));
      samples = agx_and(b, coverage, samples);
   }
   b->shader->out->tag_write_disable = false;
   return agx_st_tile(b, collected, samples, coords,
                      format, (nr_packed && format == AGX_FORMAT_RGB10A2) ? 1 :
                         nir_intrinsic_write_mask(instr),
                      nir_intrinsic_base(instr), explicit);
}

static agx_instr *
agx_emit_store_zs(agx_builder *b, nir_intrinsic_instr *instr)
{
   unsigned base = nir_intrinsic_base(instr);
   bool write_z = base & 1;
   bool write_s = base & 2;

   agx_index z = agx_src_index(&instr->src[1]);
   agx_index s = agx_src_index(&instr->src[2]);

   assert(!write_z || z.size == AGX_SIZE_32);
   assert(!write_s || s.size == AGX_SIZE_16);

   /* Not necessarily a sample mask but overlapping hw mechanism... Should
    * maybe rename this flag to something more general.
    */
   b->shader->out->writes_sample_mask = true;

   if (b->shader->key->dev.encoding == AGX_ISA_ENCODING_AGX3B &&
       (write_z || write_s)) {
      /* DEPTHF consumes coverage/stencil at staging[0] and optional float
       * depth at staging[2]. Stencil occupies the coverage word's upper half.
       * Build the same current-coverage word as MOVMSK before RA. Keeping
       * depth as float also preserves the NaN used for forced NEVER tests. */
      agx_index input = agx_get_sr_coverage(b, 16, AGX_SR_COVERAGE_MASK);
      agx_index affected = agx_and(b, agx_src_index(&instr->src[0]),
                                    agx_immediate(15));
      agx_index target = agx_and(b, affected, input);
      agx_index coverage = agx_and(b, affected,
                                    agx_ushr(b, input, agx_immediate(4)));
      agx_index mid = agx_iadd(b, coverage, target, 4);
      agx_index word = agx_iadd(b, affected, mid, 4);
      agx_index selected = agx_temp(b->shader, AGX_SIZE_32);
      agx_emit_collect_to(
         b, selected, 2,
         (agx_index[]){word, write_s ? s : agx_mov_imm(b, 16, 0)});
      return agx_zs_emit_agx3(b, selected, write_z ? z : agx_null(), base);
   }

   if (write_z && write_s) {
      agx_index u2u32 = agx_temp(b->shader, AGX_SIZE_32);
      agx_mov_to(b, u2u32, s);
      s = u2u32;
   }

   agx_index zs = (write_z && write_s) ? agx_vec2(b, z, s) : write_z ? z : s;

   agx_wait_pixel_mask(b, 0x1);
   return agx_zs_emit(b, agx_src_index(&instr->src[0]), zs, base);
}


static void
agx_unpack_normalized_halves_agx3(agx_builder *b, agx_index dest,
                                  agx_index *halves, enum agx_format format,
                                  unsigned nr_comps, unsigned read_mask,
                                  bool split_dest)
{
   const bool norm16 =
      format == AGX_FORMAT_U16NORM || format == AGX_FORMAT_S16NORM;
   agx_index channels[4] = {agx_null()};
   const unsigned conversions = DIV_ROUND_UP(nr_comps, 2);
   for (unsigned i = 0; i < conversions; ++i) {
      const unsigned lanes = MIN2(2u, nr_comps - i * 2);
      if (!(read_mask & (BITFIELD_MASK(lanes) << (i * 2)))) {
         for (unsigned c = 0; c < lanes; ++c)
            channels[i * 2 + c] = agx_undef(dest.size);
         continue;
      }

      /* One unpack per packed register; each yields a 32-bit pair, which the
       * split below turns back into the two scalar channels. */
      agx_index pair = agx_temp(b->shader, AGX_SIZE_32);
      pair.channels_m1 = lanes - 1;
      agx_index source = halves[i];
      if (norm16) {
         source = agx_temp(b->shader, AGX_SIZE_16);
         source.channels_m1 = lanes - 1;
         agx_emit_collect_to(b, source, lanes, &halves[i * 2]);
      }
      agx_normflt_agx3_to(b, pair, source, format);

      agx_index unpacked[2] = {agx_null()};
      agx_emit_split(b, unpacked, pair, lanes);
      for (unsigned c = 0; c < lanes; ++c) {
         if (dest.size == AGX_SIZE_16) {
            channels[i * 2 + c] = agx_temp(b->shader, AGX_SIZE_16);
            agx_fmov_to(b, channels[i * 2 + c], unpacked[c]);
         } else {
            channels[i * 2 + c] = unpacked[c];
         }
      }
   }

   agx_emit_collect_to(b, dest, nr_comps, channels);
   if (split_dest)
      agx_emit_cached_split(b, dest, nr_comps);
}

/*
 * The tilebuffer read half of the same story as agx_pack_normalized_agx3():
 * an 8-bit normalized attachment comes back packed, two channels to a 16-bit
 * register, and `normflt` is the unpack -- one 16-bit register in, two 32-bit
 * floats out. Apple's programmable-blending shaders spend exactly
 * ceil(channels / 2) of them straight after the tile load.
 *
 * Returns true if it handled the load.
 */
static bool
agx_unpack_normalized_agx3(agx_builder *b, agx_index dest,
                           nir_intrinsic_instr *instr, enum agx_format format,
                           unsigned nr_comps)
{
   if (b->shader->key->dev.encoding == AGX_ISA_ENCODING_AGX2)
      return false;
   if (format == AGX_FORMAT_RGB10A2) {
      if (nr_comps != 4 || dest.size != AGX_SIZE_32)
         return false;
      agx_index packed = agx_temp(b->shader, AGX_SIZE_32);
      agx_ld_tile_to(b, packed, agx_src_index(&instr->src[0]), agx_null(),
                    format, 1, nir_intrinsic_base(instr), false);
      agx_unpack_rgb10a2_agx3_to(b, dest, packed);
      agx_emit_cached_split(b, dest, 4);
      return true;
   }
   if (format != AGX_FORMAT_U8NORM && format != AGX_FORMAT_S8NORM)
      return false;
   if (nr_comps < 1 || nr_comps > 4 || dest.size != AGX_SIZE_32)
      return false;

   const unsigned nr_packed = DIV_ROUND_UP(nr_comps, 2);

   agx_index packed = agx_temp(b->shader, AGX_SIZE_16);
   packed.channels_m1 = nr_packed - 1;
   agx_ld_tile_to(b, packed, agx_src_index(&instr->src[0]), agx_null(), format,
                  BITFIELD_MASK(nr_comps), nir_intrinsic_base(instr), false);

   agx_index halves[4] = {agx_null()};
   agx_emit_split(b, halves, packed, nr_packed);

   agx_index channels[4] = {agx_null()};
   for (unsigned i = 0; i < nr_packed; ++i) {
      /* One unpack per packed register; each yields a 32-bit pair, which the
       * split below turns back into the two scalar channels. */
      agx_index pair = agx_temp(b->shader, AGX_SIZE_32);
      const unsigned lanes = MIN2(2u, nr_comps - i * 2);
      pair.channels_m1 = lanes - 1;
      agx_normflt_agx3_to(b, pair, halves[i], format);

      agx_index unpacked[2] = {agx_null()};
      agx_emit_split(b, unpacked, pair, lanes);
      for (unsigned c = 0; c < lanes; ++c)
         channels[i * 2 + c] = unpacked[c];
   }

   agx_emit_collect_to(b, dest, nr_comps, channels);
   agx_emit_cached_split(b, dest, nr_comps);
   return true;
}

static void
agx_emit_local_load_pixel(agx_builder *b, agx_index dest,
                          nir_intrinsic_instr *instr)
{
   agx_wait_pixel_mask(b, 0x8);

   unsigned nr_comps = instr->def.num_components;
   const enum agx_format format =
      agx_format_for_pipe(nir_intrinsic_format(instr));

   /* Native normalized loads physically unpack to f32. A half-precision
    * framebuffer-fetch/epilog destination needs explicit narrowing after
    * that unpack, rather than a four-half raw tile load. */
   const bool narrow_normalized =
      b->shader->key->dev.encoding == AGX_ISA_ENCODING_AGX3B &&
      dest.size == AGX_SIZE_16 &&
      (format == AGX_FORMAT_RGB10A2 || format == AGX_FORMAT_U8NORM ||
       format == AGX_FORMAT_S8NORM);
   agx_index load_dest = narrow_normalized
      ? agx_vec_temp(b->shader, AGX_SIZE_32, nr_comps) : dest;
   if (agx_unpack_normalized_agx3(b, load_dest, instr, format, nr_comps)) {
      if (narrow_normalized) {
         agx_index wide[4], half[4];
         agx_emit_split(b, wide, load_dest, nr_comps);
         for (unsigned i = 0; i < nr_comps; ++i) {
            half[i] = agx_temp(b->shader, AGX_SIZE_16);
            agx_fmov_to(b, half[i], wide[i]);
         }
         agx_emit_collect_to(b, dest, nr_comps, half);
         agx_emit_cached_split(b, dest, nr_comps);
      }
      return;
   }

   if (b->shader->key->dev.encoding == AGX_ISA_ENCODING_AGX3B &&
       format == AGX_FORMAT_I8) {
      /* LLDCOL moves packed bytes. Unpack to integer lanes; the tilebuffer
       * NIR lowering applies signed extension for logical SINT formats. */
      const unsigned pairs = DIV_ROUND_UP(nr_comps, 2);
      agx_index packed = agx_vec_temp(b->shader, AGX_SIZE_16, pairs);
      agx_ld_tile_to(b, packed, agx_src_index(&instr->src[0]), agx_null(),
                    format, BITFIELD_MASK(nr_comps),
                    nir_intrinsic_base(instr), false);
      agx_index halves[2], channels[4];
      agx_emit_split(b, halves, packed, pairs);
      for (unsigned c = 0; c < nr_comps; ++c) {
         agx_index wide = agx_pad_to_32(b, halves[c / 2]);
         agx_index byte = agx_temp(b->shader, AGX_SIZE_32);
         agx_bfeil_to(b, byte, agx_zero(), wide,
                      agx_immediate(8 * (c % 2)), 8);
         if (dest.size == AGX_SIZE_16) {
            channels[c] = agx_temp(b->shader, AGX_SIZE_16);
            agx_subdivide_to(b, channels[c], byte, 0);
         } else {
            channels[c] = byte;
         }
      }
      agx_emit_collect_to(b, dest, nr_comps, channels);
      agx_emit_cached_split(b, dest, nr_comps);
      return;
   }

   agx_ld_tile_to(b, dest, agx_src_index(&instr->src[0]), agx_null(), format,
                  BITFIELD_MASK(nr_comps), nir_intrinsic_base(instr), false);
   agx_emit_cached_split(b, dest, nr_comps);
}

static bool
nir_is_coherent(nir_intrinsic_instr *instr)
{
   return nir_intrinsic_access(instr) & (ACCESS_COHERENT | ACCESS_VOLATILE);
}

static void
agx_emit_load(agx_builder *b, agx_index dest, nir_intrinsic_instr *instr)
{
   agx_index addr = agx_src_index(&instr->src[0]);
   agx_index offset = agx_src_index(&instr->src[1]);
   enum agx_format fmt = agx_format_for_pipe(nir_intrinsic_format(instr));
   unsigned shift = nir_intrinsic_base(instr);
   const bool index_bytes = shift & AGX_NIR_MEMORY_INDEX_BYTES;
   assert(!index_bytes || b->shader->key->dev.encoding == AGX_ISA_ENCODING_AGX3B);
   const unsigned packed_count = (shift & AGX_NIR_MEMORY_PACKED_COUNT_MASK) >>
                                 AGX_NIR_MEMORY_PACKED_COUNT_SHIFT;
   assert(packed_count <= 4);
   assert(!packed_count || b->shader->key->dev.encoding == AGX_ISA_ENCODING_AGX3B);
   shift &= ~(AGX_NIR_MEMORY_INDEX_BYTES | AGX_NIR_MEMORY_PACKED_COUNT_MASK);

   assert(shift <= 3);

   /* Zero-extend offset if we're not sign-extending */
   if (!nir_intrinsic_sign_extend(instr))
      offset = agx_abs(offset);

   const bool norm8 = fmt == AGX_FORMAT_U8NORM || fmt == AGX_FORMAT_S8NORM;
   const bool norm16 = fmt == AGX_FORMAT_U16NORM || fmt == AGX_FORMAT_S16NORM;
   if (b->shader->key->dev.encoding == AGX_ISA_ENCODING_AGX3B &&
       (norm8 || norm16)) {
      const unsigned nr = instr->def.num_components;
      assert(nr >= 1 && nr <= 4 && packed_count == 0);
      assert(dest.size == AGX_SIZE_16 || dest.size == AGX_SIZE_32);


      agx_index packed = norm8 ? agx_temp(b->shader, AGX_SIZE_32)
                              : agx_vec_temp(b->shader, AGX_SIZE_16, nr);
      agx_instr *load = agx_device_load_to(
         b, packed, addr, offset, norm8 ? AGX_FORMAT_I8 : AGX_FORMAT_I16,
         BITFIELD_MASK(nr), shift, nir_is_coherent(instr));
      load->memory_index_is_bytes = index_bytes;
      load->memory_is_packed = norm8 && nr > 1;

      agx_index halves[4] = {agx_null()};
      if (norm8) {
         for (unsigned i = 0; i < DIV_ROUND_UP(nr, 2); ++i) {
            halves[i] = agx_temp(b->shader, AGX_SIZE_16);
            agx_subdivide_to(b, halves[i], packed, i);
         }
      } else {
         agx_emit_split(b, halves, packed, nr);
      }
      /* Vertex fetch (agx_nir_lower_vbo, the only producer of normalized
       * constant loads) converts only the lanes the shader reads and leaves
       * each result where `normflt` wrote it. Formatted meta loads keep
       * their collected vector. */
      const bool vertex_fetch =
         instr->intrinsic == nir_intrinsic_load_constant_agx;
      agx_unpack_normalized_halves_agx3(
         b, dest, halves, fmt, nr,
         vertex_fetch ? nir_def_components_read(&instr->def)
                      : BITFIELD_MASK(nr),
         !vertex_fetch);
      return;
   }

   /* AGX3 memory traffic is raw. Convert a normalized packed texel with
    * the same native instruction used by tile loads, after its real load. */
   if (b->shader->key->dev.encoding == AGX_ISA_ENCODING_AGX3B &&
       fmt == AGX_FORMAT_RGB10A2) {
      assert(dest.size == AGX_SIZE_32 && instr->def.num_components == 4);
      assert(packed_count == 0);
      agx_index packed = agx_temp(b->shader, AGX_SIZE_32);
      agx_instr *load = agx_device_load_to(b, packed, addr, offset,
         AGX_FORMAT_I32, 1, shift, nir_is_coherent(instr));
      load->memory_index_is_bytes = index_bytes;
      agx_unpack_rgb10a2_agx3_to(b, dest, packed);
      agx_emit_cached_split(b, dest, 4);
      return;
   }

   if (packed_count && b->shader->key->dev.encoding == AGX_ISA_ENCODING_AGX3B &&
       (fmt == AGX_FORMAT_U8NORM || fmt == AGX_FORMAT_S8NORM)) {
      const unsigned nr = instr->def.num_components;
      assert(nr >= 1 && nr <= 4 && dest.size == AGX_SIZE_32);
      assert(packed_count == nr);
      agx_index packed = agx_temp(b->shader, AGX_SIZE_32);
      agx_instr *load = agx_device_load_to(
         b, packed, addr, offset, AGX_FORMAT_I8, BITFIELD_MASK(nr), shift,
         nir_is_coherent(instr));
      load->memory_index_is_bytes = index_bytes;
      load->memory_is_packed = true;
      agx_index channels[4];
      for (unsigned i = 0; i < DIV_ROUND_UP(nr, 2); ++i) {
         agx_index half = agx_temp(b->shader, AGX_SIZE_16);
         agx_subdivide_to(b, half, packed, i);
         const unsigned lanes = MIN2(2u, nr - 2 * i);
         agx_index pair = agx_vec_temp(b->shader, AGX_SIZE_32, lanes);
         agx_normflt_agx3_to(b, pair, half, fmt);
         agx_index unpacked[2];
         agx_emit_split(b, unpacked, pair, lanes);
         for (unsigned c = 0; c < lanes; ++c)
            channels[2 * i + c] = unpacked[c];
      }
      agx_emit_collect_to(b, dest, nr, channels);
      agx_emit_cached_split(b, dest, nr);
      return;
   }

   if (b->shader->key->dev.encoding == AGX_ISA_ENCODING_AGX3B &&
       fmt == AGX_FORMAT_I8 && !packed_count && instr->def.num_components > 1) {
      /* Native byte vectors are densely packed, unlike the compiler's
       * 16-bit scalar representation of 8-bit NIR values. */
      const unsigned nr = instr->def.num_components;
      assert(nr <= 4 && dest.size == AGX_SIZE_16);
      agx_index packed = agx_temp(b->shader, AGX_SIZE_32);
      agx_instr *load = agx_device_load_to(b, packed, addr, offset, fmt,
         BITFIELD_MASK(nr), shift, nir_is_coherent(instr));
      load->memory_index_is_bytes = index_bytes;
      load->memory_is_packed = true;
      agx_index channels[4];
      for (unsigned c = 0; c < nr; ++c) {
         agx_index byte = agx_temp(b->shader, AGX_SIZE_32);
         agx_bfeil_to(b, byte, agx_zero(), packed, agx_immediate(8 * c), 8);
         channels[c] = agx_temp(b->shader, AGX_SIZE_16);
         agx_subdivide_to(b, channels[c], byte, 0);
      }
      agx_emit_collect_to(b, dest, nr, channels);
      return;
   }

   agx_instr *load = agx_device_load_to(b, dest, addr, offset, fmt,
                      BITFIELD_MASK(packed_count ? packed_count : instr->def.num_components), shift,
                      nir_is_coherent(instr));
   load->memory_index_is_bytes = index_bytes;
   load->memory_is_packed = packed_count != 0;
   agx_emit_cached_split(b, dest, instr->def.num_components);
}

static void
agx_emit_store(agx_builder *b, nir_intrinsic_instr *instr)
{
   agx_index addr = agx_src_index(&instr->src[1]);
   agx_index offset = agx_src_index(&instr->src[2]);
   enum agx_format fmt = agx_format_for_pipe(nir_intrinsic_format(instr));
   unsigned shift = nir_intrinsic_base(instr);
   const bool index_bytes = shift & AGX_NIR_MEMORY_INDEX_BYTES;
   assert(!index_bytes || b->shader->key->dev.encoding == AGX_ISA_ENCODING_AGX3B);
   shift &= ~AGX_NIR_MEMORY_INDEX_BYTES;

   assert(shift <= 3);

   /* Zero-extend offset if we're not sign-extending */
   if (!nir_intrinsic_sign_extend(instr))
      offset = agx_abs(offset);

   agx_index value;
   unsigned mask;
   if (b->shader->key->dev.encoding == AGX_ISA_ENCODING_AGX3B &&
       fmt == AGX_FORMAT_RGB10A2) {
      assert(nir_src_num_components(instr->src[0]) == 4 &&
             nir_src_bit_size(instr->src[0]) == 32);
      value = agx_pack_rgb10a2_agx3(b,
         agx_extract_nir_src(b, instr->src[0], 0),
         agx_extract_nir_src(b, instr->src[0], 1),
         agx_extract_nir_src(b, instr->src[0], 2),
         agx_extract_nir_src(b, instr->src[0], 3));
      fmt = AGX_FORMAT_I32;
      mask = 1;
   } else {
      value = agx_recollect_vector(b, instr->src[0]);
      mask = BITFIELD_MASK(nir_src_num_components(instr->src[0]));
   }
   const bool packed_bytes =
      b->shader->key->dev.encoding == AGX_ISA_ENCODING_AGX3B &&
      fmt == AGX_FORMAT_I8 && nir_src_num_components(instr->src[0]) > 1;
   if (packed_bytes) {
      const unsigned nr = nir_src_num_components(instr->src[0]);
      assert(nr <= 4);
      for (unsigned c = 0; c < nr; ++c) {
         agx_index byte = agx_pad_to_32(b, agx_extract_nir_src(b, instr->src[0], c));
         byte = agx_and(b, byte, agx_immediate(0xff));
         if (c) {
            agx_index shifted = agx_temp(b->shader, AGX_SIZE_32);
            agx_bfi_to(b, shifted, agx_zero(), byte, agx_immediate(8 * c), 0);
            value = agx_or(b, value, shifted);
         } else {
            value = byte;
         }
      }
   }
   agx_instr *store = agx_device_store(b, value, addr, offset, fmt, mask,
                                      shift, nir_is_coherent(instr));
   store->memory_index_is_bytes = index_bytes;
   store->memory_is_packed = packed_bytes;
}

/*
 * Hardware bindless texture sources are specified as a 64-bit uniform base
 * address summed with a 32-bit register index. We model in NIR with the
 * bindless_image_agx intrinsic.
 */
static agx_index
agx_translate_bindless_handle(agx_builder *b, nir_src *handle, agx_index *base)
{
   nir_intrinsic_instr *intr = nir_src_as_intrinsic(*handle);
   assert(intr->intrinsic == nir_intrinsic_bindless_image_agx ||
          intr->intrinsic == nir_intrinsic_bindless_sampler_agx);

   *base = agx_uniform(nir_intrinsic_desc_set(intr), AGX_SIZE_64);
   return agx_src_index(&intr->src[0]);
}

/* Preambles write directly to uniform registers, so move from uniform to GPR */
static agx_instr *
agx_emit_load_preamble(agx_builder *b, agx_index dst,
                       nir_intrinsic_instr *instr)
{
   agx_index srcs[4] = {agx_null()};
   unsigned dim = instr->def.num_components;
   assert(dim <= ARRAY_SIZE(srcs) && "shouldn't see larger vectors");

   unsigned base = nir_intrinsic_base(instr);
   unsigned stride = agx_size_align_16(dst.size);

   for (unsigned i = 0; i < dim; ++i)
      srcs[i] = agx_uniform(base + i * stride, dst.size);

   return agx_emit_collect_to(b, dst, dim, srcs);
}

static agx_instr *
agx_emit_store_preamble(agx_builder *b, nir_intrinsic_instr *instr)
{
   nir_preamble_class cls = nir_intrinsic_preamble_class(instr);
   unsigned base = nir_intrinsic_base(instr);

   if (cls != nir_preamble_class_general) {
      agx_index heap, offset;
      offset = agx_translate_bindless_handle(b, &instr->src[0], &heap);

      /* base is 32-bit units for images but 16-bit for samplers, hence the
       * division difference to convert into texture/sampler state units.
       */
      if (cls == nir_preamble_class_image)
         return agx_tex_state_store(b, heap, offset, base / 2);
      else
         return agx_sampler_state_store(b, heap, offset, base);
   }

   agx_index vec = agx_src_index(&instr->src[0]);
   unsigned stride = agx_size_align_16(vec.size);
   unsigned nr = nir_src_num_components(instr->src[0]);

   for (unsigned i = 0; i < nr; i += (4 / stride)) {
      agx_index data[4] = {0};
      unsigned count = MIN2(4 / stride, nr - i);

      for (unsigned c = 0; c < count; ++c) {
         data[c] = agx_extract_nir_src(b, instr->src[0], i + c);
      }

      agx_uniform_store(b, agx_emit_collect(b, count, data),
                        agx_immediate(base + i * stride), BITFIELD_MASK(count));
   }

   return NULL;
}

static enum agx_dim
agx_tex_dim(enum glsl_sampler_dim dim, bool array)
{
   switch (dim) {
   case GLSL_SAMPLER_DIM_1D:
      return array ? AGX_DIM_1D_ARRAY : AGX_DIM_1D;

   case GLSL_SAMPLER_DIM_2D:
   case GLSL_SAMPLER_DIM_RECT:
   case GLSL_SAMPLER_DIM_EXTERNAL:
      return array ? AGX_DIM_2D_ARRAY : AGX_DIM_2D;

   case GLSL_SAMPLER_DIM_MS:
      return array ? AGX_DIM_2D_MS_ARRAY : AGX_DIM_2D_MS;

   case GLSL_SAMPLER_DIM_3D:
      assert(!array && "3D arrays unsupported");
      return AGX_DIM_3D;

   case GLSL_SAMPLER_DIM_CUBE:
      return array ? AGX_DIM_CUBE_ARRAY : AGX_DIM_CUBE;

   case GLSL_SAMPLER_DIM_BUF:
      UNREACHABLE("Buffer textures should have been lowered");

   default:
      UNREACHABLE("Invalid sampler dim\n");
   }
}

static agx_instr *
agx_emit_block_image_store(agx_builder *b, nir_intrinsic_instr *instr)
{
   agx_index offset = agx_src_index(&instr->src[1]);
   agx_index coords = agx_src_index(&instr->src[2]);
   enum agx_format format = agx_format_for_pipe(nir_intrinsic_format(instr));

   bool ms = nir_intrinsic_image_dim(instr) == GLSL_SAMPLER_DIM_MS;
   bool array = nir_intrinsic_image_array(instr);
   enum agx_dim dim = agx_tex_dim(nir_intrinsic_image_dim(instr), array);
   bool explicit = nir_intrinsic_explicit_coord(instr);

   /* Modified coordinate descriptor */
   if (!explicit) {
      if (array) {
         agx_index layer = coords;
         coords = agx_temp(b->shader, AGX_SIZE_32);
         agx_emit_collect_to(b, coords, 2,
                             (agx_index[]){
                                ms ? agx_mov_imm(b, 16, 0) : layer,
                                ms ? layer : agx_undef(AGX_SIZE_16),
                             });
      } else {
         coords = agx_null();
      }
   }

   agx_index base, index;
   if (instr->intrinsic == nir_intrinsic_bindless_image_store_block_agx) {
      index = agx_translate_bindless_handle(b, &instr->src[0], &base);

      assert(base.size == AGX_SIZE_64);
      assert(index.size == AGX_SIZE_32);
   } else {
      base = agx_zero();
      index = agx_src_index(&instr->src[0]);

      assert(index.size == AGX_SIZE_16);
   }

   /* Preserve the typed F16 store for native AGX3 EOT selection. */
   if (format == AGX_FORMAT_F16 &&
       b->shader->key->dev.encoding != AGX_ISA_ENCODING_AGX3B)
      format = AGX_FORMAT_I16;

   return agx_block_image_store(b, base, index, offset, coords, format, dim,
                                explicit);
}

static agx_instr *
agx_get_sr_native_width_to(agx_builder *b, agx_index dst, enum agx_sr sr)
{
   bool narrow_agx3 = false;

   switch (sr) {
   case AGX_SR_THREADS_PER_THREADGROUP_X:
   case AGX_SR_THREADS_PER_THREADGROUP_Y:
   case AGX_SR_THREADS_PER_THREADGROUP_Z:
   case AGX_SR_DISPATCH_THREADS_PER_THREADGROUP_X:
   case AGX_SR_DISPATCH_THREADS_PER_THREADGROUP_Y:
   case AGX_SR_DISPATCH_THREADS_PER_THREADGROUP_Z:
   case AGX_SR_THREAD_POSITION_IN_THREADGROUP_X:
   case AGX_SR_THREAD_POSITION_IN_THREADGROUP_Y:
   case AGX_SR_THREAD_POSITION_IN_THREADGROUP_Z:
   case AGX_SR_THREAD_INDEX_IN_THREADGROUP:
   case AGX_SR_THREAD_INDEX_IN_SUBGROUP:
   case AGX_SR_SUBGROUP_INDEX_IN_THREADGROUP:
   case AGX_SR_RENDER_TARGET_LAYER:
      narrow_agx3 = true;
      break;
   default:
      break;
   }

   if (b->shader->key->dev.encoding != AGX_ISA_ENCODING_AGX2 && narrow_agx3 &&
       dst.size == AGX_SIZE_32) {
      /* Apple's AGX3 get_sr forms expose these builtins as witnessed 16-bit
       * values. NIR presents the Vulkan/D3D builtins as uint32_t, so read the
       * native half register and explicitly zero-extend it. This also makes
       * the async dependency visible to the scoreboard allocator. */
      agx_index native = agx_temp(b->shader, AGX_SIZE_16);
      agx_get_sr_to(b, native, sr);
      return agx_mov_to(b, dst, native);
   }

   return agx_get_sr_to(b, dst, sr);
}

/* Apple reads the active-thread index/count of the SIMD-group (SR 34/36)
 * into half registers (witness-sr59); NIR presents them as uint32_t. */
static agx_instr *
agx_get_sr_coverage_native_width_to(agx_builder *b, agx_index dst,
                                    enum agx_sr sr)
{
   if (b->shader->key->dev.encoding != AGX_ISA_ENCODING_AGX2 &&
       dst.size == AGX_SIZE_32 &&
       (sr == AGX_SR_ACTIVE_THREAD_INDEX_IN_SUBGROUP ||
        sr == AGX_SR_TOTAL_ACTIVE_THREADS_IN_SUBGROUP)) {
      agx_index native = agx_temp(b->shader, AGX_SIZE_16);
      agx_get_sr_coverage_to(b, native, sr);
      return agx_mov_to(b, dst, native);
   }

   return agx_get_sr_coverage_to(b, dst, sr);
}

static agx_instr *
agx_load_compute_dimension(agx_builder *b, agx_index dst,
                           nir_intrinsic_instr *instr, enum agx_sr base)
{
   unsigned dim = instr->def.num_components;
   unsigned size = instr->def.bit_size;
   assert(size == 16 || size == 32);

   agx_index srcs[3];
   for (unsigned i = 0; i < ARRAY_SIZE(srcs); ++i) {
      srcs[i] = agx_temp(b->shader, agx_size_for_bits(size));
      agx_get_sr_native_width_to(b, srcs[i], base + i);
   }

   return agx_emit_collect_to(b, dst, dim, srcs);
}

static enum agx_atomic_opc
translate_atomic_opcode(nir_atomic_op op)
{
   /* clang-format off */
   switch (op) {
   case nir_atomic_op_iadd:    return AGX_ATOMIC_OPC_ADD;
   case nir_atomic_op_imin:    return AGX_ATOMIC_OPC_IMIN;
   case nir_atomic_op_umin:    return AGX_ATOMIC_OPC_UMIN;
   case nir_atomic_op_imax:    return AGX_ATOMIC_OPC_IMAX;
   case nir_atomic_op_umax:    return AGX_ATOMIC_OPC_UMAX;
   case nir_atomic_op_iand:    return AGX_ATOMIC_OPC_AND;
   case nir_atomic_op_ior:     return AGX_ATOMIC_OPC_OR;
   case nir_atomic_op_ixor:    return AGX_ATOMIC_OPC_XOR;
   case nir_atomic_op_xchg:    return AGX_ATOMIC_OPC_XCHG;
   case nir_atomic_op_cmpxchg: return AGX_ATOMIC_OPC_CMPXCHG;
   default: UNREACHABLE("unknown atomic opcode");
   }
   /* clang-format on */
}

/*
 * The "base" of a local load/store/atomic can be zero but no other immediates.
 * This would be a little silly to handle when inlining immediates, so we
 * instead exclude these ops from immediate inlining and just handle 0 specially
 * when translating.
 */
static agx_index
agx_local_base(nir_src src)
{
   if (nir_src_is_const(src) && nir_src_as_uint(src) == 0)
      return agx_zero();
   else
      return agx_src_index(&src);
}

static void
agx_emit_atomic(agx_builder *b, agx_index dst, nir_intrinsic_instr *instr,
                bool local)
{
   enum agx_atomic_opc op =
      translate_atomic_opcode(nir_intrinsic_atomic_op(instr));
   agx_index base =
      local ? agx_local_base(instr->src[0]) : agx_src_index(&instr->src[0]);
   agx_index value = agx_src_index(&instr->src[local ? 1 : 2]);
   agx_index index = local ? agx_zero() : agx_src_index(&instr->src[1]);

   /* cmpxchg (only) takes 2 sources, passed in consecutive registers */
   if (op == AGX_ATOMIC_OPC_CMPXCHG) {
      agx_index value2 = agx_src_index(&instr->src[local ? 2 : 3]);
      value = agx_vec2(b, value2, value);
   }

   if (local) {
      assert(base.size == AGX_SIZE_16);
      agx_local_atomic_to(b, dst, value, base, index, op);
   } else {
      assert(base.size == AGX_SIZE_64);
      agx_atomic_to(b, dst, value, base, index, op);
   }
}

static enum agx_format
format_for_bitsize(unsigned bitsize)
{
   switch (bitsize) {
   case 8:
      return AGX_FORMAT_I8;
   case 16:
      return AGX_FORMAT_I16;
   case 32:
      return AGX_FORMAT_I32;
   default:
      UNREACHABLE("should've been lowered");
   }
}

static void
agx_emit_local_load(agx_builder *b, agx_index dst, nir_intrinsic_instr *instr)
{
   agx_index base = agx_local_base(instr->src[0]);
   agx_index index = agx_zero(); /* TODO: optimize address arithmetic */
   assert(base.size == AGX_SIZE_16);

   enum agx_format format = format_for_bitsize(instr->def.bit_size);
   unsigned nr = instr->def.num_components;
   unsigned mask = BITFIELD_MASK(nr);

   agx_local_load_to(b, dst, base, index, format, mask);
   agx_emit_cached_split(b, dst, nr);
}

static void
agx_emit_local_store(agx_builder *b, nir_intrinsic_instr *instr)
{
   agx_index value = agx_src_index(&instr->src[0]);
   agx_index base = agx_local_base(instr->src[1]);
   agx_index index = agx_zero(); /* TODO: optimize address arithmetic */
   assert(base.size == AGX_SIZE_16);

   enum agx_format format = format_for_bitsize(nir_src_bit_size(instr->src[0]));
   unsigned mask = BITFIELD_MASK(
      nir_src_num_components(instr->src[0])); /* XXX: there's a write mask */

   agx_local_store(b, value, base, index, format, mask);
}

/* Scalarize static 32-bit stack vectors after RA so both NIR scratch and
 * allocator spills use the native scalar TPS copy/rld protocol. No temporary
 * GPR is required: component registers and frame offsets are already fixed. */
static void
agx_lower_agx3_stack_vectors(agx_context *ctx)
{
   if (ctx->key->dev.encoding != AGX_ISA_ENCODING_AGX3B)
      return;

   agx_foreach_instr_global_safe(ctx, I) {
      const bool store = I->op == AGX_OPCODE_STACK_STORE;
      if ((!store && I->op != AGX_OPCODE_STACK_LOAD) ||
          (I->format != AGX_FORMAT_I32 && I->format != AGX_FORMAT_I16) ||
          I->mask == 1)
         continue;
      /* Spilled 16-bit vectors take the same route one half at a time
       * (subgroups.ballot_broadcast.*.subgroupbroadcast_{f,i,u}16vec*). */
      const bool halves = I->format == AGX_FORMAT_I16;
      agx_index data = store ? I->src[0] : I->dest[0];
      agx_index offset = I->src[store ? 1 : 0];
      if (data.type != AGX_INDEX_REGISTER || data.memory ||
          (halves ? data.size != AGX_SIZE_16
                  : (data.size != AGX_SIZE_32 && data.size != AGX_SIZE_64)) ||
          offset.type != AGX_INDEX_IMMEDIATE)
         continue;
      agx_builder b = agx_init_builder(ctx, agx_before_instr(I));
      unsigned packed = 0;
      u_foreach_bit(c, I->mask) {
         agx_index component = data;
         component.value += (halves ? 1 : 2) * packed++;
         component.channels_m1 = 0;
         component.size = halves ? AGX_SIZE_16 : AGX_SIZE_32;
         agx_index address = agx_immediate(offset.value + (halves ? 2 : 4) * c);
         if (store)
            agx_stack_store(&b, component, address, I->format, 1);
         else
            agx_stack_load_to(&b, component, address, I->format, 1);
      }
      agx_remove_instruction(I);
   }
}

static agx_index
agx_scratch_offset(agx_builder *b, nir_src *src)
{
   agx_index offset = agx_src_index(src);
   if (!b->shader->atomic_tps_reserved)
      return offset;
   if (nir_src_is_const(*src))
      return agx_immediate(nir_src_as_uint(*src) + 16);
   return agx_iadd(b, offset, agx_immediate(16), 0);
}

static void
agx_emit_load_scratch(agx_builder *b, agx_index dst, nir_intrinsic_instr *instr)
{
   agx_index offset = agx_scratch_offset(b, &instr->src[0]);
   enum agx_format format = format_for_bitsize(instr->def.bit_size);
   unsigned nr = instr->def.num_components;
   unsigned mask = BITFIELD_MASK(nr);

   agx_stack_load_to(b, dst, offset, format, mask);
   agx_emit_cached_split(b, dst, nr);
   b->shader->any_scratch = true;
}

static void
agx_emit_store_scratch(agx_builder *b, nir_intrinsic_instr *instr)
{
   agx_index value = agx_recollect_vector(b, instr->src[0]);
   agx_index offset = agx_scratch_offset(b, &instr->src[1]);
   enum agx_format format = format_for_bitsize(nir_src_bit_size(instr->src[0]));
   unsigned mask = BITFIELD_MASK(nir_src_num_components(instr->src[0]));

   agx_stack_store(b, value, offset, format, mask);
   b->shader->any_scratch = true;
}

static unsigned
agx_expand_tex_to(agx_builder *b, nir_def *def, agx_index src, bool masked)
{
   unsigned nr_channels = def->num_components;
   nir_component_mask_t mask = nir_def_components_read(def);

   if (!masked)
      mask = (nir_component_mask_t)BITFIELD_MASK(nr_channels);

   agx_index packed_channels[8] = {agx_null()};
   agx_index unpacked_channels[8] = {agx_null()};

   /* Hardware writes the masked components contiguously, expand out for NIR */
   /* Sparse depth comparisons still reserve the sparse operation's full
    * clobber span even though only colour + residency are consumed. */
   agx_emit_split(b, packed_channels, src, agx_channels(src));

   for (unsigned i = 0; i < nr_channels; ++i) {
      unpacked_channels[i] =
         (mask & BITFIELD_BIT(i))
            ? packed_channels[util_bitcount(mask & BITFIELD_MASK(i))]
            : agx_undef(src.size);
   }

   agx_emit_collect_to(b, agx_def_index(def), nr_channels, unpacked_channels);
   return mask;
}

static agx_index
agx_pack_ms_aux(agx_builder *b, agx_index sample, agx_index layer)
{
   agx_index packed_layer = agx_zero();
   if (!agx_is_null(layer)) {
      packed_layer = agx_temp(b->shader, AGX_SIZE_32);
      agx_pack_layer_agx3_to(b, packed_layer, layer);
   }
   agx_index aux = agx_temp(b->shader, AGX_SIZE_32);
   agx_pack_sample_agx3_to(b, aux, sample, packed_layer);
   return aux;
}

static agx_instr *
agx_emit_image_load(agx_builder *b, agx_index dst, nir_intrinsic_instr *intr)
{
   agx_index ms_index = agx_src_index(&intr->src[2]);
   agx_index lod = agx_src_index(&intr->src[3]);
   enum agx_lod_mode lod_mode = AGX_LOD_MODE_LOD_MIN;
   bool sparse = intr->intrinsic == nir_intrinsic_image_sparse_load ||
                 intr->intrinsic == nir_intrinsic_bindless_image_sparse_load;

   agx_index bindless = agx_immediate(0), texture;
   if (intr->intrinsic == nir_intrinsic_bindless_image_load ||
       intr->intrinsic == nir_intrinsic_bindless_image_sparse_load) {

      texture = agx_translate_bindless_handle(b, &intr->src[0], &bindless);
   } else if (nir_src_is_const(intr->src[0]) &&
              nir_src_as_uint(intr->src[0]) < 0x100) {
      texture = agx_immediate(nir_src_as_uint(intr->src[0]));
   } else {
      texture = agx_src_index(&intr->src[0]);
   }

   assert(nir_src_num_components(intr->src[1]) == 4);
   agx_index coord[4] = {
      agx_extract_nir_src(b, intr->src[1], 0),
      agx_extract_nir_src(b, intr->src[1], 1),
      agx_extract_nir_src(b, intr->src[1], 2),
      agx_extract_nir_src(b, intr->src[1], 3),
   };

   /* Get the image dimension. Cubes are lowered to 2D, since they are logically
    * equivalent for imageLoad, but out-of-bounds behaviour for cubes on G13
    * is wrong according to Piglit's arb_shader_image_load_store-invalid.
    *
    * This requires a matching transform in the driver.
    */
   enum glsl_sampler_dim dim = nir_intrinsic_image_dim(intr);
   bool is_array = nir_intrinsic_image_array(intr);

   if (dim == GLSL_SAMPLER_DIM_CUBE) {
      dim = GLSL_SAMPLER_DIM_2D;
      is_array = true;
   }

   bool is_ms = dim == GLSL_SAMPLER_DIM_MS;
   unsigned coord_comps = glsl_get_sampler_dim_coordinate_components(dim);
   if (is_ms && b->shader->key->dev.encoding == AGX_ISA_ENCODING_AGX3B) {
      /* Coherent/volatile loads retain their memory opcode, but need the
       * same native sample/layer packing as ordinary multisample fetches. */
      agx_index layer = agx_null();
      if (is_array) {
         layer = agx_temp(b->shader, AGX_SIZE_16);
         agx_subdivide_to(b, layer, coord[coord_comps], 0);
      }
      coord[coord_comps++] = agx_pack_ms_aux(b, ms_index, layer);
   } else if (is_array && is_ms) {
      agx_index layer = agx_temp(b->shader, AGX_SIZE_16);
      agx_subdivide_to(b, layer, coord[coord_comps], 0);

      assert(ms_index.size == AGX_SIZE_16);
      agx_index tmp = agx_temp(b->shader, AGX_SIZE_32);
      agx_emit_collect_to(b, tmp, 2, (agx_index[]){ms_index, layer});
      coord[coord_comps++] = tmp;
   } else if (is_ms) {
      agx_index tmp = agx_temp(b->shader, AGX_SIZE_32);
      agx_mov_to(b, tmp, ms_index);
      coord[coord_comps++] = tmp;
   } else if (is_array) {
      coord_comps++;
   }

   /* Multisampled images do not support mipmapping */
   if (is_ms) {
      lod_mode = AGX_LOD_MODE_AUTO_LOD;
      lod = agx_zero();
   } else if (b->shader->key->dev.encoding == AGX_ISA_ENCODING_AGX3B &&
              (intr->intrinsic == nir_intrinsic_image_load ||
               intr->intrinsic == nir_intrinsic_image_sparse_load) &&
              dim == GLSL_SAMPLER_DIM_2D &&
              nir_src_is_const(intr->src[3]) &&
              nir_src_as_uint(intr->src[3]) == 0) {
      /* Native heap-indexed storage reads use the immediate LOD0 form.
       * Keep qualified memory reads as IMAGE_LOAD, including their ordering
       * metadata, instead of converting them to pure texture operations. */
      lod_mode = AGX_LOD_MODE_AUTO_LOD;
      lod = agx_zero();
   }

   agx_index coords = agx_emit_collect(b, coord_comps, coord);
   agx_index tmp = agx_vec_temp(b->shader, dst.size, sparse ? 8 : 4);

   agx_instr *I = agx_image_load_to(b, tmp, coords, lod, bindless, texture,
                                    agx_immediate(0), agx_null(),
                                    agx_tex_dim(dim, is_array), lod_mode, 0,
                                    false, sparse, nir_is_coherent(intr));
   I->mask = agx_expand_tex_to(b, &intr->def, tmp, !sparse);

   b->shader->out->uses_txf = true;
   b->shader->out->sampler_state_count =
      MAX2(b->shader->out->sampler_state_count, 1);
   return NULL;
}

static agx_instr *
agx_emit_export(agx_builder *b, unsigned base, nir_src src)
{
   agx_builder b_ = *b;
   agx_cursor after_cursor = agx_after_block(agx_exit_block(b->shader));
   b_.cursor = after_cursor;

   for (unsigned c = 0; c < nir_src_num_components(src); ++c) {
      agx_index chan = agx_extract_nir_src(b, src, c);
      unsigned stride = agx_size_align_16(chan.size);

      agx_export(&b_, chan, base + (c * stride));
   }

   if (agx_cursors_equal(b->cursor, after_cursor)) {
      b->cursor = agx_after_block_logical(b->cursor.block);
   }

   return NULL;
}

static agx_instr *
agx_load_exported_to(agx_builder *b, agx_index dst, unsigned base, unsigned nr)
{
   agx_index chans[4] = {0};
   unsigned stride = agx_size_align_16(dst.size);

   for (unsigned c = 0; c < nr; ++c) {
      chans[c] = agx_cached_preload(b->shader, base + c * stride, dst.size);
   }

   return agx_emit_collect_to(b, dst, nr, chans);
}

static agx_instr *
agx_emit_image_store(agx_builder *b, nir_intrinsic_instr *instr)
{
   /* See remarks in agx_emit_image_load */
   enum glsl_sampler_dim glsl_dim = nir_intrinsic_image_dim(instr);
   bool is_array = nir_intrinsic_image_array(instr);

   if (glsl_dim == GLSL_SAMPLER_DIM_CUBE) {
      glsl_dim = GLSL_SAMPLER_DIM_2D;
      is_array = true;
   }

   enum agx_dim dim = agx_tex_dim(glsl_dim, is_array);
   assert(glsl_dim != GLSL_SAMPLER_DIM_MS && "needs to be lowered");

   agx_index base, index;
   if (instr->intrinsic == nir_intrinsic_bindless_image_store) {
      index = agx_translate_bindless_handle(b, &instr->src[0], &base);

      assert(base.size == AGX_SIZE_64);
      assert(index.size == AGX_SIZE_32);
   } else {
      base = agx_zero();
      index = agx_src_index(&instr->src[0]);

      assert(index.size == AGX_SIZE_16 ||
             (b->shader->key->dev.encoding == AGX_ISA_ENCODING_AGX3B &&
              index.size == AGX_SIZE_64));
   }

   agx_index coords4 = agx_src_index(&instr->src[1]);
   agx_index lod = agx_src_index(&instr->src[4]);
   assert(lod.size == AGX_SIZE_16);

   int coord_components = glsl_get_sampler_dim_coordinate_components(glsl_dim);
   if (is_array)
      coord_components++;

   agx_index coord_comps[4] = {};
   for (unsigned i = 0; i < coord_components; ++i)
      coord_comps[i] = agx_emit_extract(b, coords4, i);

   agx_index coords = agx_emit_collect(b, coord_components, coord_comps);
   agx_index data = agx_src_index(&instr->src[3]);

   /* If the image format has less than 4 components, nir_opt_shrink_stores can
    * shrink the store. But the IR still expects 4 components: pad with undef.
    */
   if (nir_src_num_components(instr->src[3]) < 4) {
      agx_index chan[4] = {agx_null()};

      for (unsigned i = 0; i < 4; ++i) {
         if (i < nir_src_num_components(instr->src[3]))
            chan[i] = agx_extract_nir_src(b, instr->src[3], i);
         else
            chan[i] = agx_undef(data.size);
      }

      data = agx_emit_collect(b, 4, chan);
   }

   /* Image stores act like tilebuffer stores when used for tib spilling */
   b->shader->out->tag_write_disable = false;

   agx_instr *write = agx_image_write(b, data, coords, lod, base, index, dim,
                                      nir_is_coherent(instr));
   if (b->shader->key->dev.encoding == AGX_ISA_ENCODING_AGX3B) {
      switch (nir_intrinsic_format(instr)) {
      case PIPE_FORMAT_NONE:
         /* Formatless writes supply four 32-bit lanes. The PBE supplies
          * the destination width; native imgwci distinguishes floating
          * conversion from raw integer input, not I8/I16/I32 payloads. */
         write->format = nir_intrinsic_src_type(instr) == nir_type_float32
                            ? AGX_FORMAT_F16 : AGX_FORMAT_I32;
         break;
      case PIPE_FORMAT_R10G10B10A2_UINT:
      case PIPE_FORMAT_B10G10R10A2_UINT:
      case PIPE_FORMAT_R16_UINT:
      case PIPE_FORMAT_R16_SINT:
      case PIPE_FORMAT_R16G16_UINT:
      case PIPE_FORMAT_R16G16_SINT:
      case PIPE_FORMAT_R16G16B16A16_SINT:
      case PIPE_FORMAT_R16G16B16A16_UINT:
         write->format = AGX_FORMAT_I16;
         break;
      case PIPE_FORMAT_R8_UINT:
      case PIPE_FORMAT_R8G8_UINT:
      case PIPE_FORMAT_R8G8B8A8_UINT:
      case PIPE_FORMAT_R8_SINT:
      case PIPE_FORMAT_R8G8_SINT:
      case PIPE_FORMAT_R8G8B8A8_SINT:
         write->format = AGX_FORMAT_I8;
         break;
      case PIPE_FORMAT_R8G8_UNORM:
      case PIPE_FORMAT_R8_UNORM:
      case PIPE_FORMAT_R8G8B8A8_UNORM:
      case PIPE_FORMAT_B8G8R8A8_UNORM:
         write->format = AGX_FORMAT_U8NORM;
         break;
      case PIPE_FORMAT_R10G10B10A2_UNORM:
      /* Same RGB10A2 lanes; the PBE swizzles B and R as for the UINT twin. */
      case PIPE_FORMAT_B10G10R10A2_UNORM:
         /* Formatted float writes share the native normalized imgwci path;
          * the PBE descriptor supplies the packed destination format. */
         write->format = AGX_FORMAT_RGB10A2;
         break;
      case PIPE_FORMAT_R32_UINT:
      case PIPE_FORMAT_R32_SINT:
      case PIPE_FORMAT_R32G32_UINT:
      case PIPE_FORMAT_R32G32_SINT:
      case PIPE_FORMAT_R32G32B32A32_SINT:
      case PIPE_FORMAT_R32G32B32A32_UINT:
         write->format = AGX_FORMAT_I32;
         break;
      case PIPE_FORMAT_R5G6B5_UNORM:
      case PIPE_FORMAT_B5G6R5_UNORM:
      case PIPE_FORMAT_R11G11B10_FLOAT:
      case PIPE_FORMAT_R8_SNORM:
      case PIPE_FORMAT_R8G8_SNORM:
      case PIPE_FORMAT_R8G8B8A8_SNORM:
      case PIPE_FORMAT_R32_FLOAT:
      case PIPE_FORMAT_R32G32_FLOAT:
      case PIPE_FORMAT_R32G32B32A32_FLOAT:
         /* AGX3 uses this tag for floating input lanes; the PBE
          * descriptor controls whether storage is F16 or F32. */
         write->format = AGX_FORMAT_F16;
         break;
      case PIPE_FORMAT_R16_UNORM:
      case PIPE_FORMAT_R16G16_UNORM:
      case PIPE_FORMAT_R16_SNORM:
      case PIPE_FORMAT_R16G16_SNORM:
      case PIPE_FORMAT_R16G16B16A16_UNORM:
      case PIPE_FORMAT_R16G16B16A16_SNORM:
         /* Native formatted stores take F32 lanes for both normalized
          * signs; the PBE descriptor selects the destination conversion. */
         write->format = AGX_FORMAT_U16NORM;
         break;
      case PIPE_FORMAT_R16_FLOAT:
      case PIPE_FORMAT_R16G16_FLOAT:
      case PIPE_FORMAT_R16G16B16A16_FLOAT:
         /* imgwci consumes float32 lanes; the PBE converts to half storage. */
         write->format = AGX_FORMAT_F16;
         break;
      default:
         /* The AGX3 packer remains fail-closed for unqualified formats. */
         write->format = AGX_NUM_FORMATS;
         break;
      }
   }
   return write;
}

static enum agx_simd_op
translate_simd_op(nir_op op)
{
#define CASE(agx_, nir_)                                                       \
   case nir_op_##nir_:                                                         \
      return AGX_SIMD_OP_##agx_;

   switch (op) {
      CASE(AND, iand)
      CASE(FADD, fadd)
      CASE(OR, ior)
      CASE(FMUL, fmul)
      CASE(XOR, ixor)
      CASE(FMIN, fmin)
      CASE(FMAX, fmax)
      CASE(IADD, iadd)
      CASE(SMIN, imin)
      CASE(SMAX, imax)
      CASE(UMIN, umin)
      CASE(UMAX, umax)
   default:
      UNREACHABLE("unknown simd op");
   }
#undef CASE
}

static agx_instr *
agx_emit_intrinsic(agx_builder *b, nir_intrinsic_instr *instr)
{
   agx_index dst = nir_intrinsic_infos[instr->intrinsic].has_dest
                      ? agx_def_index(&instr->def)
                      : agx_null();
   mesa_shader_stage stage = b->shader->stage;

   switch (instr->intrinsic) {
   case nir_intrinsic_load_barycentric_pixel:
   case nir_intrinsic_load_barycentric_centroid:
   case nir_intrinsic_load_barycentric_at_sample:
   case nir_intrinsic_load_barycentric_at_offset:
      /* handled later via load_vary */
      return NULL;
   case nir_intrinsic_load_interpolated_input:
      assert(stage == MESA_SHADER_FRAGMENT);
      agx_emit_load_vary(b, dst, instr);
      return NULL;

   case nir_intrinsic_load_coefficients_agx:
      assert(stage == MESA_SHADER_FRAGMENT);
      agx_ldcf_to(b, dst, cf_for_intrinsic(b, instr), 1);
      agx_emit_cached_split(b, dst, 3);
      return NULL;

   case nir_intrinsic_load_agx:
   case nir_intrinsic_load_constant_agx:
      agx_emit_load(b, dst, instr);
      return NULL;

   case nir_intrinsic_store_uvs_agx:
      assert(stage == MESA_SHADER_VERTEX || stage == MESA_SHADER_TESS_EVAL);
      return agx_st_vary(b, agx_src_index(&instr->src[1]),
                         agx_src_index(&instr->src[0]));

   case nir_intrinsic_store_agx:
      agx_emit_store(b, instr);
      return NULL;

   case nir_intrinsic_store_shared:
      agx_emit_local_store(b, instr);
      return NULL;

   case nir_intrinsic_load_shared:
      agx_emit_local_load(b, dst, instr);
      return NULL;

   case nir_intrinsic_global_atomic_agx:
   case nir_intrinsic_global_atomic_swap_agx:
      agx_emit_atomic(b, dst, instr, false);
      return NULL;

   case nir_intrinsic_shared_atomic:
   case nir_intrinsic_shared_atomic_swap:
      agx_emit_atomic(b, dst, instr, true);
      return NULL;

   case nir_intrinsic_store_zs_agx:
      assert(stage == MESA_SHADER_FRAGMENT);
      return agx_emit_store_zs(b, instr);

   case nir_intrinsic_store_local_pixel_agx:
      return agx_emit_local_store_pixel(b, instr);

   case nir_intrinsic_load_local_pixel_agx:
      assert(stage == MESA_SHADER_FRAGMENT);
      agx_emit_local_load_pixel(b, dst, instr);
      return NULL;

   case nir_intrinsic_load_pixel_coord:
      if (b->shader->key->dev.encoding == AGX_ISA_ENCODING_AGX2) {
         return agx_emit_collect_to(
            b, dst, 2,
            (agx_index[2]){
               agx_get_sr(b, 16, AGX_SR_THREAD_POSITION_IN_GRID_X),
               agx_get_sr(b, 16, AGX_SR_THREAD_POSITION_IN_GRID_Y),
            });
      } else {
         agx_index xy[2];
         for (unsigned i = 0; i < 2; i++) {
            agx_index native = agx_temp(b->shader, AGX_SIZE_32);
            agx_get_sr_native_width_to(
               b, native, AGX_SR_THREAD_POSITION_IN_GRID_X + i);
            xy[i] = agx_temp(b->shader, AGX_SIZE_16);
            agx_subdivide_to(b, xy[i], native, 0);
         }
         return agx_emit_collect_to(b, dst, 2, xy);
      }

   case nir_intrinsic_load_frag_coord_z: {
      agx_index cf = agx_get_cf(b->shader, VARYING_SLOT_POS, 2);

      if (b->shader->key->dev.encoding == AGX_ISA_ENCODING_AGX3B)
         return agx_interp_agx3_to(b, dst, cf, agx_zero(),
                                     AGX_INTERPOLATION_CENTER, false);

      return agx_iter_to(b, dst, cf, agx_zero(), 1, AGX_INTERPOLATION_CENTER);
   }

   case nir_intrinsic_load_frag_coord_w: {
      agx_index cf = agx_get_cf(b->shader, VARYING_SLOT_POS, 3);

      /* Native position.w uses ordinary interpolation (f54=0). The
       * projective RHW form is private to reciprocal interpolation and
       * cannot be reused as a public arithmetic value. */
      if (b->shader->key->dev.encoding == AGX_ISA_ENCODING_AGX3B)
         return agx_interp_agx3_to(b, dst, cf, agx_zero(),
                                     AGX_INTERPOLATION_CENTER, false);

      return agx_iter_to(b, dst, cf, agx_zero(), 1, AGX_INTERPOLATION_CENTER);
   }

   case nir_intrinsic_sample_mask_agx: {
      assert(stage == MESA_SHADER_FRAGMENT);
      b->shader->out->writes_sample_mask = true;

      /* We need to wait_pix before running Z/S tests, but we don't need to
       * wait_pix before merely discarding. Omit the wait_pix when the affected
       * samples are unconditionally killed.
       */
      bool no_tests =
         nir_src_is_const(instr->src[1]) && nir_src_as_uint(instr->src[1]) == 0;

      /* Only affected samples can run tests. A static pipeline mask uses
       * complementary affected/live masks to kill excluded samples in the
       * prolog; its nonzero live operand does not test any of them. */
      if (nir_src_is_const(instr->src[0]) &&
          nir_src_is_const(instr->src[1]))
         no_tests = !(nir_src_as_uint(instr->src[0]) &
                      nir_src_as_uint(instr->src[1]));

      if (b->shader->key->dev.encoding == AGX_ISA_ENCODING_AGX3B) {
         /* Read current coverage for every update. The immutable input mask
          * can resurrect samples killed by an earlier discard. Preserve the
          * target and covered nibbles independently, restricting each by the
          * samples affected by this operation. */
         agx_index input = agx_get_sr_coverage(b, 16, AGX_SR_COVERAGE_MASK);
         agx_index affected = agx_and(b, agx_src_index(&instr->src[0]),
                                        agx_immediate(15));
         agx_index target = agx_and(b, affected, input);
         agx_index coverage = agx_and(b, affected,
                                      agx_ushr(b, input, agx_immediate(4)));
         agx_index live = agx_and(b, affected, agx_src_index(&instr->src[1]));
         agx_index mid = agx_iadd(b, coverage, target, 4);
         agx_index word = agx_iadd(b, live, mid, 4);
         agx_index zero = agx_mov_imm(b, 16, 0);
         agx_index selected = agx_temp(b->shader, AGX_SIZE_32);
         agx_emit_collect_to(b, selected, 2, (agx_index[]){word, zero});
         return agx_sample_mask_agx3(b, target, selected, !no_tests ? 2 : 0);
      }

      if (!no_tests)
         agx_wait_pixel_mask(b, 0x1);

      return agx_sample_mask(b, agx_src_index(&instr->src[0]),
                             agx_src_index(&instr->src[1]));
   }

   case nir_intrinsic_load_back_face_agx:
      if (b->shader->key->dev.encoding == AGX_ISA_ENCODING_AGX3B) {
         /* Native front_facing is original SR41 == 0. Return a canonical
          * NIR boolean for the back-facing intrinsic, not the raw half. */
         return agx_icmp_to(b, dst, agx_get_sr(b, 16, AGX_SR_BACKFACING),
                            agx_zero(), AGX_ICOND_UEQ, true);
      }
      return agx_get_sr_to(b, dst, AGX_SR_BACKFACING);

   case nir_intrinsic_load_samples_log2_agx:
      return agx_get_sr_to(b, dst, AGX_SR_SAMPLES_LOG2);

   case nir_intrinsic_load_sample_mask_in:
      if (b->shader->key->dev.encoding == AGX_ISA_ENCODING_AGX3B) {
         /* Original SR39 packs coverage and target masks into nibbles. */
         agx_index input = agx_get_sr(b, 16, AGX_SR_INPUT_SAMPLE_MASK);
         agx_index mask = agx_and(b, agx_ushr(b, input, agx_immediate(4)),
                                  agx_immediate(15));
         return agx_mov_to(b, dst, mask);
      }
      return agx_get_sr_to(b, dst, AGX_SR_INPUT_SAMPLE_MASK);

   case nir_intrinsic_load_sample_mask:
      if (b->shader->key->dev.encoding == AGX_ISA_ENCODING_AGX3B) {
         agx_index input = agx_get_sr_coverage(b, 16, AGX_SR_COVERAGE_MASK);
         return agx_mov_to(b, dst, agx_and(b, input, agx_immediate(15)));
      }
      return agx_get_sr_coverage_to(b, dst, AGX_SR_COVERAGE_MASK);

   case nir_intrinsic_load_helper_invocation:
      if (b->shader->key->dev.encoding == AGX_ISA_ENCODING_AGX3B) {
         /* G16G's simd_is_helper_thread reads original SR2 as a half and
          * compares it with zero. NIR tracks subsequent demotes separately. */
         return agx_icmp_to(b, dst,
                            agx_get_sr_coverage(b, 16, AGX_SR_IS_ACTIVE_THREAD),
                            agx_zero(), AGX_ICOND_UEQ, false);
      }
      /* Compare special register to zero. We could lower this in NIR (letting
       * us fold in an inot) but meh?
       */
      return agx_icmp_to(b, dst,
                         agx_get_sr_coverage(b, 32, AGX_SR_IS_ACTIVE_THREAD),
                         agx_zero(), AGX_ICOND_UEQ, false);

   case nir_intrinsic_load_vertex_id_zero_base:
      assert(stage == MESA_SHADER_COMPUTE && "only for SW VS");
      return agx_mov_to(b, dst, agx_abs(agx_vertex_id_zero_base(b)));

   case nir_intrinsic_load_vertex_id:
      /* AGX3 hardware vertex shaders read the invocation index explicitly.
       * Apple's isolated G17P vertex-attribute shaders begin with
       * `0cdd1006`, a 32-bit rld from SR 0xdd. Retaining AGX2's preloaded r5
       * ABI made every invocation observe the same index and collapse a
       * triangle to one vertex. Software vertex shaders still use the preload
       * ABI populated by their compute launcher. */
      if (stage == MESA_SHADER_VERTEX &&
          b->shader->key->dev.encoding != AGX_ISA_ENCODING_AGX2)
         return agx_get_sr_to(b, dst, AGX_SR_VERTEX_ID);

      return agx_mov_to(b, dst, agx_abs(agx_vertex_id(b)));

   case nir_intrinsic_load_primitive_id:
      assert(stage == MESA_SHADER_TESS_EVAL);
      return agx_mov_to(b, dst, agx_abs(agx_vertex_id(b)));

   case nir_intrinsic_load_instance_id:
      /* The matching Apple hardware-vertex witness is `0cd81006`, reading
       * instance ID from SR 0xd8. */
      if (stage == MESA_SHADER_VERTEX &&
          b->shader->key->dev.encoding != AGX_ISA_ENCODING_AGX2)
         return agx_get_sr_to(b, dst, AGX_SR_INSTANCE_ID);

      return agx_mov_to(b, dst, agx_abs(agx_instance_id(b)));

   case nir_intrinsic_load_tess_coord_xy: {
      assert(stage == MESA_SHADER_TESS_EVAL);

      agx_index coords[] = {agx_tess_coord_x(b), agx_tess_coord_y(b)};
      return agx_emit_collect_to(b, dst, 2, coords);
   }

   case nir_intrinsic_load_preamble:
      return agx_emit_load_preamble(b, dst, instr);

   case nir_intrinsic_store_preamble:
      return agx_emit_store_preamble(b, instr);

   case nir_intrinsic_image_load:
   case nir_intrinsic_image_sparse_load:
   case nir_intrinsic_bindless_image_load:
   case nir_intrinsic_bindless_image_sparse_load:
      return agx_emit_image_load(b, dst, instr);

   case nir_intrinsic_image_store:
   case nir_intrinsic_bindless_image_store:
      return agx_emit_image_store(b, instr);

   case nir_intrinsic_image_store_block_agx:
   case nir_intrinsic_bindless_image_store_block_agx:
      return agx_emit_block_image_store(b, instr);

   case nir_intrinsic_load_workgroup_id:
      return agx_load_compute_dimension(b, dst, instr,
                                        AGX_SR_THREADGROUP_POSITION_IN_GRID_X);

   case nir_intrinsic_load_workgroup_size:
      return agx_load_compute_dimension(b, dst, instr,
                                        AGX_SR_THREADS_PER_THREADGROUP_X);

   case nir_intrinsic_load_global_invocation_id:
      return agx_load_compute_dimension(b, dst, instr,
                                        AGX_SR_THREAD_POSITION_IN_GRID_X);

   case nir_intrinsic_load_local_invocation_id:
      return agx_load_compute_dimension(
         b, dst, instr, AGX_SR_THREAD_POSITION_IN_THREADGROUP_X);

   case nir_intrinsic_load_local_invocation_index:
      return agx_get_sr_native_width_to(
         b, dst, AGX_SR_THREAD_INDEX_IN_THREADGROUP);

   case nir_intrinsic_load_layer_id:
      if (b->shader->key->dev.encoding == AGX_ISA_ENCODING_AGX3B &&
          b->shader->stage == MESA_SHADER_FRAGMENT)
         return agx_get_sr_native_width_to(b, dst, AGX_SR_RENDER_TARGET_LAYER);
      return agx_get_sr_to(b, dst, AGX_SR_THREADGROUP_POSITION_IN_GRID_Z);

   case nir_intrinsic_barrier: {
      assert(!b->shader->is_preamble && "invalid");

      if (b->shader->key->dev.encoding != AGX_ISA_ENCODING_AGX2) {
         bool emitted = false;

         if (nir_intrinsic_memory_scope(instr) != SCOPE_NONE) {
            const nir_variable_mode modes =
               nir_intrinsic_memory_modes(instr);

            /* The AGX2 cache-operation train collapses on AGX3 into one
             * exact barrier per semantic memory scope. Keep Apple's observed
             * device -> texture -> threadgroup ordering. */
            if (modes & (nir_var_mem_global | nir_var_image)) {
               const bool fence =
                  b->shader->key->dev.encoding == AGX_ISA_ENCODING_AGX3B &&
                  nir_intrinsic_execution_scope(instr) < SCOPE_WORKGROUP;
               agx_scoped_barrier_agx3(b, fence ? AGX3_BARRIER_SCOPE_DEVICE_FENCE
                                                : AGX3_BARRIER_SCOPE_DEVICE);
               emitted = true;
            }

            if (modes & nir_var_image) {
               const bool fence =
                  b->shader->key->dev.encoding == AGX_ISA_ENCODING_AGX3B &&
                  nir_intrinsic_execution_scope(instr) < SCOPE_WORKGROUP;
               agx_scoped_barrier_agx3(
                  b, fence ? AGX3_BARRIER_SCOPE_TEXTURE_FENCE
                           : AGX3_BARRIER_SCOPE_TEXTURE);
               emitted = true;
            }

            if (modes & nir_var_mem_shared) {
               /* A shared-memory fence or subgroup barrier must not wait
                * for the entire workgroup. Other subgroups may have exited
                * or may be executing a different control-flow path. Use the
                * non-rendezvous fence, which also drains outstanding local
                * reads, unless NIR requests a workgroup execution barrier.
                */
               const bool fence =
                  b->shader->key->dev.encoding == AGX_ISA_ENCODING_AGX3B &&
                  nir_intrinsic_execution_scope(instr) < SCOPE_WORKGROUP;
               agx_scoped_barrier_agx3(
                  b, fence ? AGX3_BARRIER_SCOPE_DEVICE_FENCE
                           : AGX3_BARRIER_SCOPE_THREADGROUP);
               emitted = true;
            }
         }

         /* A workgroup execution barrier with no memory scope is the
          * recovered mem_none form. Scoped memory barriers already carry the
          * execution rendezvous, as shown by each Metal mem_flags contrast. */
         if (nir_intrinsic_execution_scope(instr) >= SCOPE_WORKGROUP &&
             !emitted) {
            assert(mesa_shader_stage_is_compute(b->shader->nir->info.stage));
            agx_scoped_barrier_agx3(b, AGX3_BARRIER_SCOPE_EXECUTION);
         }

         return NULL;
      }

      bool needs_image_barriers = false;

      if (nir_intrinsic_memory_scope(instr) != SCOPE_NONE) {
         nir_variable_mode modes = nir_intrinsic_memory_modes(instr);

         if (modes & (nir_var_mem_global | nir_var_image)) {
            agx_memory_barrier(b);

            /* Pull out all the big hammers to make cross-workgroup memory
             * barriers work.
             */
            if (nir_intrinsic_memory_scope(instr) >= SCOPE_QUEUE_FAMILY) {
               agx_unknown_barrier_1(b);
               agx_memory_barrier_2(b);

               /* These are observed on G13D. At some point we should figure out
                * what the individual opcodes do.
                */
               agx_device_barrier_2(b);
               agx_unknown_barrier_2(b);
               agx_memory_barrier_3(b);
               agx_device_barrier_1(b);
            }
         }

         if (modes & nir_var_image) {
            agx_image_barrier_1(b);
            agx_image_barrier_2(b);
            needs_image_barriers = true;
         }
      }

      /* Nothing to do for subgroup barriers */
      if (nir_intrinsic_execution_scope(instr) >= SCOPE_WORKGROUP) {
         assert(mesa_shader_stage_is_compute(b->shader->nir->info.stage));

         agx_threadgroup_barrier(b);
      }

      if (needs_image_barriers) {
         agx_image_barrier_3(b);
         agx_image_barrier_4(b);
      }

      return NULL;
   }

   case nir_intrinsic_fence_pbe_to_tex_agx: {
      if (b->shader->key->dev.encoding != AGX_ISA_ENCODING_AGX2) {
         /* Compiler-inserted fences order one invocation's accesses. They
          * may occur after a predicated store, so must not rendezvous with
          * other invocations which never execute the store. */
         agx_scoped_barrier_agx3(
            b, b->shader->key->dev.encoding == AGX_ISA_ENCODING_AGX3B
                  ? AGX3_BARRIER_SCOPE_TEXTURE_FENCE
                  : AGX3_BARRIER_SCOPE_TEXTURE);
         return NULL;
      }

      agx_image_barrier_1(b);
      agx_image_barrier_2(b);
      agx_image_barrier_3(b);
      agx_image_barrier_4(b);
      return NULL;
   }

   case nir_intrinsic_fence_mem_to_tex_agx: {
      if (b->shader->key->dev.encoding != AGX_ISA_ENCODING_AGX2) {
         const bool fence =
            b->shader->key->dev.encoding == AGX_ISA_ENCODING_AGX3B;
         agx_scoped_barrier_agx3(b, fence ? AGX3_BARRIER_SCOPE_DEVICE_FENCE
                                        : AGX3_BARRIER_SCOPE_DEVICE);
         agx_scoped_barrier_agx3(b, fence ? AGX3_BARRIER_SCOPE_TEXTURE_FENCE
                                        : AGX3_BARRIER_SCOPE_TEXTURE);
         return NULL;
      }

      /* Flush out the atomic to main memory... Found experimentally... */
      agx_memory_barrier(b);
      agx_memory_barrier_2(b);

      /* TODO: Which ones do we actually need? */
      agx_image_barrier_1(b);
      agx_image_barrier_2(b);
      agx_image_barrier_3(b);
      agx_image_barrier_4(b);

      /* Flush out the texture cache */
      agx_flush_memory_to_texture(b);
      return NULL;
   }

   case nir_intrinsic_fence_pbe_to_tex_pixel_agx: {
      if (b->shader->key->dev.encoding != AGX_ISA_ENCODING_AGX2) {
         agx_scoped_barrier_agx3(
            b, b->shader->key->dev.encoding == AGX_ISA_ENCODING_AGX3B
                  ? AGX3_BARRIER_SCOPE_TEXTURE_FENCE
                  : AGX3_BARRIER_SCOPE_TEXTURE);
         return NULL;
      }

      agx_image_barrier_1(b);
      agx_image_barrier_2(b);
      agx_flush_memory_to_texture(b);
      agx_image_barrier_3(b);
      return NULL;
   }

   case nir_intrinsic_fence_helper_exit_agx: {
      assert(b->shader->key->is_helper);
      if (b->shader->key->dev.encoding != AGX_ISA_ENCODING_AGX2) {
         agx_scoped_barrier_agx3(b, AGX3_BARRIER_SCOPE_DEVICE);
         return NULL;
      }

      agx_memory_barrier(b);
      agx_unknown_barrier_1(b);
      agx_memory_barrier_2(b);
      agx_unknown_barrier_2(b);
      agx_memory_barrier_3(b);
      return NULL;
   }

   case nir_intrinsic_begin_invocation_interlock: {
      /* Native ordered storage acquires pixel ownership before memory access.
       * Keep ownership through the fragment/sample-loop epilog, which performs
       * the normal final release. Tilebuffer spills retain their existing wait.
       */
      if (b->shader->key->dev.encoding == AGX_ISA_ENCODING_AGX3B &&
          (b->shader->nir->info.fs.pixel_interlock_ordered ||
           b->shader->nir->info.fs.pixel_interlock_unordered)) {
         agx_wait_pixel_mask(b, 8);
      } else {
         agx_wait_pixel_mask(b, 0xC);
      }
      return NULL;
   }

   case nir_intrinsic_end_invocation_interlock:
      assert(b->shader->key->dev.encoding == AGX_ISA_ENCODING_AGX3B);
      agx_scoped_barrier_agx3(b, AGX3_BARRIER_SCOPE_INTERLOCK);
      return NULL;

   case nir_intrinsic_ddx:
   case nir_intrinsic_ddx_coarse:
   case nir_intrinsic_ddx_fine:
      return agx_dfdx_to(b, dst, agx_src_index(&instr->src[0]));

   case nir_intrinsic_ddy:
   case nir_intrinsic_ddy_coarse:
   case nir_intrinsic_ddy_fine:
      return agx_dfdy_to(b, dst, agx_src_index(&instr->src[0]));

   case nir_intrinsic_load_subgroup_invocation:
      return agx_get_sr_native_width_to(b, dst,
                                        AGX_SR_THREAD_INDEX_IN_SUBGROUP);

   case nir_intrinsic_load_subgroup_id:
      return agx_get_sr_native_width_to(
         b, dst, AGX_SR_SUBGROUP_INDEX_IN_THREADGROUP);

   case nir_intrinsic_load_active_subgroup_invocation_agx:
      return agx_get_sr_coverage_native_width_to(
         b, dst, AGX_SR_ACTIVE_THREAD_INDEX_IN_SUBGROUP);

   case nir_intrinsic_load_active_subgroup_count_agx:
      return agx_get_sr_coverage_native_width_to(
         b, dst, AGX_SR_TOTAL_ACTIVE_THREADS_IN_SUBGROUP);

   case nir_intrinsic_reduce: {
      assert((instr->def.bit_size == 1 || instr->def.bit_size == 16 ||
              instr->def.bit_size == 32) &&
             "should've been lowered");

      unsigned cluster_size = nir_intrinsic_cluster_size(instr);
      assert(cluster_size == 0 || cluster_size == 4 || cluster_size >= 32);

      enum agx_simd_op op =
         translate_simd_op(nir_intrinsic_reduction_op(instr));

      agx_index src0 = agx_src_index(&instr->src[0]);

      if (cluster_size == 4)
         return agx_quad_reduce_to(b, dst, src0, op);
      else
         return agx_simd_reduce_to(b, dst, src0, op);
   }

   case nir_intrinsic_exclusive_scan: {
      assert((instr->def.bit_size == 1 || instr->def.bit_size == 16 ||
              instr->def.bit_size == 32) &&
             "should've been lowered");

      return agx_simd_prefix_to(
         b, dst, agx_src_index(&instr->src[0]),
         translate_simd_op(nir_intrinsic_reduction_op(instr)));
   }

   case nir_intrinsic_read_invocation: {
      /* TODO: Check if we're actually inside divergent control flow */
      b->shader->any_quad_divergent_shuffle |= b->shader->any_cf;

      /* Lane ID guaranteed to be uniform */
      return agx_shuffle_to(b, dst, agx_src_index(&instr->src[0]),
                            agx_src_index(&instr->src[1]));
   }

   case nir_intrinsic_quad_broadcast: {
      /* TODO: Check if we're actually inside divergent control flow */
      b->shader->any_quad_divergent_shuffle |= b->shader->any_cf;

      /* Lane ID guaranteed to be uniform */
      return agx_quad_shuffle_to(b, dst, agx_src_index(&instr->src[0]),
                                 agx_src_index(&instr->src[1]));
   }

   case nir_intrinsic_quad_swap_horizontal: {
      return agx_quad_shuffle_xor_to(b, dst, agx_src_index(&instr->src[0]),
                                     agx_immediate(1));
   }

   case nir_intrinsic_quad_swap_vertical: {
      return agx_quad_shuffle_xor_to(b, dst, agx_src_index(&instr->src[0]),
                                     agx_immediate(2));
   }

   case nir_intrinsic_quad_swap_diagonal: {
      return agx_quad_shuffle_xor_to(b, dst, agx_src_index(&instr->src[0]),
                                     agx_immediate(3));
   }

   case nir_intrinsic_ballot: {
      return agx_ballot_to(b, dst, agx_src_index(&instr->src[0]));
   }

   case nir_intrinsic_quad_ballot_agx: {
      if (b->shader->key &&
          b->shader->key->dev.encoding == AGX_ISA_ENCODING_AGX3B &&
          dst.size == AGX_SIZE_16) {
         /* The native ballot writes a word even when only the four quad
          * bits are needed. Allocate that word before extracting its half. */
         agx_index ballot = agx_temp(b->shader, AGX_SIZE_32);
         agx_quad_ballot_to(b, ballot, agx_src_index(&instr->src[0]));
         return agx_subdivide_to(b, dst, ballot, 0);
      }
      return agx_quad_ballot_to(b, dst, agx_src_index(&instr->src[0]));
   }

   case nir_intrinsic_doorbell_agx: {
      return agx_doorbell(b, nir_src_as_uint(instr->src[0]));
   }

   case nir_intrinsic_stack_map_agx: {
      return agx_stack_map(b, agx_src_index(&instr->src[1]),
                           nir_src_as_uint(instr->src[0]));
   }

   case nir_intrinsic_stack_unmap_agx: {
      return agx_stack_unmap_to(b, dst, nir_src_as_uint(instr->src[0]));
   }

   case nir_intrinsic_load_scratch:
      agx_emit_load_scratch(b, dst, instr);
      return NULL;

   case nir_intrinsic_store_scratch:
      agx_emit_store_scratch(b, instr);
      return NULL;

   case nir_intrinsic_load_core_id:
      return agx_get_sr_to(b, dst, AGX_SR_CORE_ID);

   case nir_intrinsic_load_helper_op_id_agx:
      assert(b->shader->key->is_helper);
      return agx_get_sr_barrier_to(b, dst, AGX_SR_HELPER_OP);

   case nir_intrinsic_load_helper_arg_lo_agx:
      assert(b->shader->key->is_helper);
      return agx_get_sr_barrier_to(b, dst, AGX_SR_HELPER_ARG_L);

   case nir_intrinsic_load_helper_arg_hi_agx:
      assert(b->shader->key->is_helper);
      return agx_get_sr_barrier_to(b, dst, AGX_SR_HELPER_ARG_H);

   case nir_intrinsic_load_exported_agx:
      return agx_load_exported_to(b, dst, nir_intrinsic_base(instr),
                                  instr->def.num_components);

   case nir_intrinsic_export_agx:
      return agx_emit_export(b, nir_intrinsic_base(instr), instr->src[0]);

   case nir_intrinsic_bindless_image_agx:
   case nir_intrinsic_bindless_sampler_agx:
      /* These must always be chased */
      return NULL;

   case nir_intrinsic_load_barycentric_sample:
   case nir_intrinsic_load_sample_id:
   case nir_intrinsic_load_sample_pos:
      UNREACHABLE("Sample shading should have been lowered");

   default:
      fprintf(stderr, "Unhandled intrinsic %s\n",
              nir_intrinsic_infos[instr->intrinsic].name);
      UNREACHABLE("Unhandled intrinsic");
   }
}

static agx_index
agx_alu_src_index(agx_builder *b, nir_alu_src src)
{
   /* Check well-formedness of the input NIR */
   ASSERTED unsigned bitsize = nir_src_bit_size(src.src);
   unsigned comps = nir_src_num_components(src.src);
   unsigned channel = src.swizzle[0];

   assert(bitsize == 1 || bitsize == 8 || bitsize == 16 || bitsize == 32 ||
          bitsize == 64);
   assert(channel < comps);

   return agx_extract_nir_src(b, src.src, channel);
}

/*
 * Emit an instruction translating (s0 * s1) + (s2 << s3). Assuming s3 is
 * constant, this is an imad instruction. If s1 == 1, then this is optimized to
 * an iadd instruction, which is faster.
 */
static agx_instr *
agx_emit_imadshl_agx(agx_builder *b, nir_alu_instr *alu, agx_index dst,
                     agx_index s0, agx_index s1, agx_index s2, agx_index s3)
{
   /* The immediate iadd/imad encoding accepts only shifts 0..4.  NIR normally
    * guarantees that domain, but this backend is also a process boundary for
    * arbitrary DXIL- and SPIR-V-derived shaders.  Keep every out-of-domain
    * value on the general shift path instead of letting the packer's
    * always-live domain check take down a release process.
    */
   unsigned shift;
   bool constant_shift = nir_src_is_const(alu->src[3].src);

   if (constant_shift)
      shift = nir_alu_src_as_uint(alu->src[3]);

   if (!constant_shift || shift > 4) {
      s2 = agx_bfi(b, agx_immediate(0), s2, s3, 0);
      shift = 0;
   }

   /* Only the immediate-encoding path reaches this assertion. */
   assert(shift <= 4 && "domain restriction on the input NIR");

   /* Emit iadd if possible, else imad */
   if (nir_src_is_const(alu->src[1].src) &&
       nir_alu_src_as_uint(alu->src[1]) == 1) {

      return agx_iadd_to(b, dst, s0, s2, shift);
   } else {
      /*
       * AGX3 has no shifted imad. Apple's own compiler does not emit one:
       * across `o[i] = a[i]*b[i] + (c[i] << N)` kernels for N = 1, 2 and 3,
       * the imadd is BYTE-IDENTICAL and the shift appears on a following
       * iadd instead. So split it the same way rather than inventing a shift
       * field for an instruction that has none.
       *
       * imad(a, b, c, s) is a*b + (c << s), and integer wraparound makes the
       * split exact: the product is computed unshifted, then added with the
       * shift on the addend. Both halves are witnessed byte-exactly -- the
       * pure-multiply imadd and the shifted iadd -- in test-agx3-iadd-core.c.
       */
      const enum agx_isa_encoding enc = b->shader->key->dev.encoding;
      const bool agx3 = enc != AGX_ISA_ENCODING_AGX2;

      if (agx3 && shift != 0) {
         agx_index product = agx_temp(b->shader, dst.size);
         agx_imad_to(b, product, s0, s1, agx_immediate(0), 0);
         return agx_iadd_to(b, dst, product, s2, shift);
      }

      return agx_imad_to(b, dst, s0, s1, s2, shift);
   }
}

static bool
is_conversion_to_8bit(nir_op op)
{
   switch (op) {
   case nir_op_i2i8:
   case nir_op_u2u8:
   case nir_op_f2i8:
   case nir_op_f2u8:
   case nir_op_b2i8:
      return true;
   default:
      return false;
   }
}

static agx_instr *
agx_fminmax_to(agx_builder *b, agx_index dst, agx_index s0, agx_index s1,
               nir_alu_instr *alu)
{
   /* The hardware gtn/ltn modes are unfortunately incorrect for signed zeros */
   assert(!nir_alu_instr_is_signed_zero_preserve(alu) &&
          "should've been lowered");

   assert((alu->def.bit_size == 16) ==
             (alu->op == nir_op_fmin || alu->op == nir_op_fmax) &&
          "fp32 should be lowered");

   bool fmax = alu->op == nir_op_fmax || alu->op == nir_op_fmax_agx;
   enum agx_fcond fcond = fmax ? AGX_FCOND_GTN : AGX_FCOND_LTN;

   /* Calculate min/max with the appropriate hardware instruction. This will not
    * handle denorms, but we were already lowered for that. On AGX3 the tied
    * fcmpsel is the native fsel condition 3/7 min/max (IEEE minNum/maxNum,
    * the semantics of NIR's fmin/fmax and fmin_agx/fmax_agx); the optimizer
    * keeps both sides tied and the packer never rewrites LTN/GTN into an
    * ordered GE/LE select, which would return the NaN when s0 is NaN.
    */
   return agx_fcmpsel_to(b, dst, s0, s1, s0, s1, fcond);
}

void
agx_lower_agx3_half_converts(agx_context *ctx)
{
   if (ctx->key->dev.encoding != AGX_ISA_ENCODING_AGX3B)
      return;

   agx_foreach_instr_global_safe(ctx, I) {
      if (I->op != AGX_OPCODE_CONVERT || I->nr_dests != 1 ||
          I->nr_srcs != 2 ||
          I->src[0].type != AGX_INDEX_IMMEDIATE || I->saturate)
         continue;

      if (I->src[0].value == AGX_CONVERT_U8_TO_F &&
          I->round == AGX_ROUND_RTE) {
         /* Byte integers occupy a half register; discard the other byte
          * before using the native unsigned-half conversion. */
         agx_builder before = agx_init_builder(ctx, agx_before_instr(I));
         agx_index byte = agx_temp(ctx, AGX_SIZE_16);
         agx_and_to(&before, byte, I->src[1], agx_immediate(0xff));
         I->src[1] = byte;
         I->src[0].value = AGX_CONVERT_U16_TO_F;
      }
      if (I->src[0].value == AGX_CONVERT_S8_TO_F &&
          I->round == AGX_ROUND_RTE) {
         /* Sign-extend the byte inside its half register with the grounded
          * half bit operations and add (x ^ 0x80) - 0x80, then use the
          * native signed-half conversion
          * (spirv_assembly.instruction.*.convertstof.int8_to_float*). */
         agx_builder before = agx_init_builder(ctx, agx_before_instr(I));
         agx_index byte = agx_temp(ctx, AGX_SIZE_16);
         agx_and_to(&before, byte, I->src[1], agx_immediate(0xff));
         agx_index flipped = agx_temp(ctx, AGX_SIZE_16);
         agx_bitop_to(&before, flipped, byte, agx_mov_imm(&before, 16, 0x80),
                      0x6);
         agx_index extended = agx_temp(ctx, AGX_SIZE_16);
         agx_iadd_to(&before, extended, flipped,
                     agx_mov_imm(&before, 16, 0xff80), 0);
         I->src[1] = extended;
         I->src[0].value = AGX_CONVERT_S16_TO_F;
      }
      if (I->dest[0].size == AGX_SIZE_32 &&
          (I->src[0].value == AGX_CONVERT_F_TO_S32 ||
           I->src[0].value == AGX_CONVERT_F_TO_U32) &&
          I->round == AGX_ROUND_RTZ &&
          (I->src[1].size == AGX_SIZE_16 || I->src[1].abs || I->src[1].neg)) {
         /* Every binary16 value is exactly representable in binary32.
          * Widen first and preserve the original integer rounding mode. */
         agx_builder b = agx_init_builder(ctx, agx_before_instr(I));
         agx_index wide = agx_temp(ctx, AGX_SIZE_32);
         agx_fmov_to(&b, wide, I->src[1]);
         I->src[1] = wide;
         continue;
      }
      if (I->dest[0].size != AGX_SIZE_16)
         continue;

      if ((I->src[0].value == AGX_CONVERT_F_TO_S16 ||
           I->src[0].value == AGX_CONVERT_F_TO_U16) &&
          I->round == AGX_ROUND_RTZ) {
         /* NIR f2i16/f2u16 is defined for representable integer results.
          * Convert to the corresponding 32-bit integer then retain its low
          * half. This is exact throughout that defined range, including
          * negative signed results; do not use a floating-point narrowing. */
         agx_builder before = agx_init_builder(ctx, agx_before_instr(I));
         if (I->src[1].size == AGX_SIZE_16 || I->src[1].abs || I->src[1].neg) {
            agx_index wide = agx_temp(ctx, AGX_SIZE_32);
            agx_fmov_to(&before, wide, I->src[1]);
            I->src[1] = wide;
         }
         I->src[0].value = I->src[0].value == AGX_CONVERT_F_TO_S16 ?
            AGX_CONVERT_F_TO_S32 : AGX_CONVERT_F_TO_U32;
         agx_index half = I->dest[0];
         I->dest[0] = agx_temp(ctx, AGX_SIZE_32);
         agx_builder after = agx_init_builder(ctx, agx_after_instr(I));
         agx_subdivide_to(&after, half, I->dest[0], 0);
         continue;
      }

      if (I->round != AGX_ROUND_RTE)
         continue;
      switch (I->src[0].value) {
      case AGX_CONVERT_S32_TO_F:
      case AGX_CONVERT_U32_TO_F:
      case AGX_CONVERT_S16_TO_F:
      case AGX_CONVERT_U16_TO_F:
         break;
      default:
         continue;
      }

      /* Native integer->F32 and F32->F16 are grounded independently. This
       * composition cannot double-round an integer in the finite F16 range:
       * every such integer (including the overflow midpoint) is exact in F32.
       * Larger magnitudes overflow F16 in either route. Run after copyprop so
       * its destination narrowing cannot recreate the unsupported cast.
       */
      agx_index half = I->dest[0];
      I->dest[0] = agx_temp(ctx, AGX_SIZE_32);
      agx_builder b = agx_init_builder(ctx, agx_after_instr(I));
      agx_fmov_to(&b, half, I->dest[0]);
   }
}

static void
agx_lower_agx3_half_integer_sources(agx_context *ctx)
{
   if (ctx->key->dev.encoding != AGX_ISA_ENCODING_AGX3B)
      return;

   agx_foreach_instr_global_safe(ctx, I) {
      if ((I->op == AGX_OPCODE_SIMD_REDUCE ||
           I->op == AGX_OPCODE_QUAD_REDUCE ||
           I->op == AGX_OPCODE_SIMD_PREFIX ||
           I->op == AGX_OPCODE_QUAD_PREFIX) &&
          I->nr_dests == 1 && I->nr_srcs == 1 &&
          I->dest[0].size == AGX_SIZE_16 &&
          I->src[0].size == AGX_SIZE_16 &&
          (I->simd_op == AGX_SIMD_OP_IADD ||
           I->simd_op == AGX_SIMD_OP_AND || I->simd_op == AGX_SIMD_OP_OR ||
           I->simd_op == AGX_SIMD_OP_XOR ||
           I->simd_op == AGX_SIMD_OP_SMIN || I->simd_op == AGX_SIMD_OP_SMAX ||
           I->simd_op == AGX_SIMD_OP_UMIN || I->simd_op == AGX_SIMD_OP_UMAX)) {
         /* Native subgroup reductions operate on words. Addition and bitwise
          * operations commute with truncation; min/max need the appropriate
          * signedness when extending their inputs. */
         agx_builder b = agx_init_builder(ctx, agx_before_instr(I));
         agx_index wide = agx_temp(ctx, AGX_SIZE_32);
         if (I->simd_op == AGX_SIMD_OP_SMIN || I->simd_op == AGX_SIMD_OP_SMAX)
            agx_signext_to(&b, wide, I->src[0]);
         else
            agx_mov_to(&b, wide, I->src[0]);
         I->src[0] = wide;
         agx_index half = I->dest[0];
         I->dest[0] = agx_temp(ctx, AGX_SIZE_32);
         b.cursor = agx_after_instr(I);
         agx_mov_to(&b, half, I->dest[0]);
         continue;
      }

      if (I->op == AGX_OPCODE_EXTR && I->nr_srcs == 3) {
         /* Copy propagation folds zero extensions into EXTR, but the native
          * data operands read whole words. Materialize their high zeroes
          * before RA instead of reading an unrelated neighboring half. */
         agx_builder b = agx_init_builder(ctx, agx_before_instr(I));
         for (unsigned s = 0; s < 2; ++s) {
            if (I->src[s].size == AGX_SIZE_16 &&
                I->src[s].type != AGX_INDEX_IMMEDIATE &&
                I->src[s].type != AGX_INDEX_NULL &&
                !I->src[s].abs && !I->src[s].neg) {
               agx_index wide = agx_temp(ctx, AGX_SIZE_32);
               agx_mov_to(&b, wide, I->src[s]);
               I->src[s] = wide;
            }
         }
      }

      if ((I->op == AGX_OPCODE_SHUFFLE ||
           I->op == AGX_OPCODE_SHUFFLE_XOR ||
           I->op == AGX_OPCODE_SHUFFLE_UP ||
           I->op == AGX_OPCODE_SHUFFLE_DOWN ||
           I->op == AGX_OPCODE_QUAD_SHUFFLE ||
           I->op == AGX_OPCODE_QUAD_SHUFFLE_XOR ||
           I->op == AGX_OPCODE_QUAD_SHUFFLE_UP ||
           I->op == AGX_OPCODE_QUAD_SHUFFLE_DOWN) &&
          I->nr_dests == 1 && I->nr_srcs == 2) {
         /* Native shuffles move full words. Preserve narrow bit patterns
          * with a zero-extended source and truncate after lane selection. */
         agx_builder b = agx_init_builder(ctx, agx_before_instr(I));
         agx_foreach_src(I, s) {
            if (I->src[s].size == AGX_SIZE_16 &&
                I->src[s].type != AGX_INDEX_IMMEDIATE &&
                I->src[s].type != AGX_INDEX_NULL &&
                !I->src[s].abs && !I->src[s].neg) {
               agx_index wide = agx_temp(ctx, AGX_SIZE_32);
               agx_mov_to(&b, wide, I->src[s]);
               I->src[s] = wide;
            }
         }
         if (I->dest[0].size == AGX_SIZE_16) {
            agx_index half = I->dest[0];
            I->dest[0] = agx_temp(ctx, AGX_SIZE_32);
            b.cursor = agx_after_instr(I);
            agx_mov_to(&b, half, I->dest[0]);
         }
         continue;
      }

      if (I->nr_dests == 1 && I->dest[0].size == AGX_SIZE_16 &&
          !I->saturate &&
          ((I->op == AGX_OPCODE_NOT && I->nr_srcs == 1) ||
           (I->op == AGX_OPCODE_BITOP && I->nr_srcs == 2 &&
            I->truth_table == UTIL_LUT2(~a))) &&
          !I->src[0].abs && !I->src[0].neg &&
          I->src[0].size == AGX_SIZE_16) {
         /* Complement commutes with taking the low half. Use the native
          * 32-bit operation without changing the neighboring half-register.
          */
         agx_builder b = agx_init_builder(ctx, agx_before_instr(I));
         agx_index wide = agx_temp(ctx, AGX_SIZE_32);
         agx_mov_to(&b, wide, I->src[0]);
         I->src[0] = wide;
         agx_index half = I->dest[0];
         I->dest[0] = agx_temp(ctx, AGX_SIZE_32);
         b.cursor = agx_after_instr(I);
         agx_mov_to(&b, half, I->dest[0]);
         continue;
      }

      if (I->op == AGX_OPCODE_ASR && I->nr_srcs == 2 &&
          I->src[0].type == AGX_INDEX_IMMEDIATE &&
          !I->src[0].abs && !I->src[0].neg) {
         /* The native signed shift takes its value from a register. Keep
          * immediate folding out of that operand before register allocation. */
         agx_builder b = agx_init_builder(ctx, agx_before_instr(I));
         I->src[0] = agx_mov_imm(&b, 32, I->src[0].value);
      }

      if ((I->op == AGX_OPCODE_BFEIL ||
           (I->op == AGX_OPCODE_BFI && I->bfi_mask == 0)) &&
          I->nr_dests == 1 && I->nr_srcs == 3 &&
          I->dest[0].size == AGX_SIZE_16 && I->src[1].size == AGX_SIZE_16 &&
          I->src[0].type == AGX_INDEX_IMMEDIATE && I->src[0].value == 0 &&
          !I->src[0].abs && !I->src[0].neg && !I->src[1].abs &&
          !I->src[1].neg && !I->saturate) {
         /* Unsigned extraction and left shifts from a half register have
          * no high source bits. Widen with zeroes, retain the original mask
          * and shift, then truncate the result to the original width. */
         agx_builder b = agx_init_builder(ctx, agx_before_instr(I));
         agx_index wide = agx_temp(ctx, AGX_SIZE_32);
         agx_mov_to(&b, wide, I->src[1]);
         I->src[1] = wide;
         agx_index half = I->dest[0];
         I->dest[0] = agx_temp(ctx, AGX_SIZE_32);
         b.cursor = agx_after_instr(I);
         agx_mov_to(&b, half, I->dest[0]);
         continue;
      }
      if (I->op == AGX_OPCODE_ASR && I->nr_dests == 1 && I->nr_srcs == 2 &&
          (I->dest[0].size == AGX_SIZE_16 || I->dest[0].size == AGX_SIZE_32) &&
          I->src[0].size == AGX_SIZE_16 &&
          I->src[0].type != AGX_INDEX_IMMEDIATE && !I->src[0].abs &&
          !I->src[0].neg &&
          !I->src[1].abs && !I->src[1].neg && !I->saturate) {
         /* Arithmetic shifts sign-extend their value operand independently
          * of the destination width. This also handles register shift counts
          * and the 16-to-32-bit form used by signed 8-bit conversions. */
         agx_builder b = agx_init_builder(ctx, agx_before_instr(I));
         agx_index wide = agx_temp(ctx, AGX_SIZE_32);
         agx_signext_to(&b, wide, I->src[0]);
         I->src[0] = wide;
         if (I->dest[0].size == AGX_SIZE_16) {
            agx_index half = I->dest[0];
            I->dest[0] = agx_temp(ctx, AGX_SIZE_32);
            b.cursor = agx_after_instr(I);
            agx_mov_to(&b, half, I->dest[0]);
         }
         continue;
      }
      if (I->op != AGX_OPCODE_IMAD || I->nr_dests != 1 || I->nr_srcs != 3 ||
          (I->dest[0].size != AGX_SIZE_16 && I->dest[0].size != AGX_SIZE_32) ||
          I->shift || I->saturate)
         continue;
      bool plain_half = true;
      agx_foreach_src(I, s) {
         plain_half &= (I->src[s].size == AGX_SIZE_16 ||
                        I->src[s].size == AGX_SIZE_32) &&
                       I->src[s].type != AGX_INDEX_NULL &&
                       !I->src[s].neg;
      }
      if (!plain_half)
         continue;

      /* Multiplication and addition modulo 2^16 are unchanged by zero
       * extending the inputs, computing modulo 2^32 and truncating once. */
      agx_builder b = agx_init_builder(ctx, agx_before_instr(I));
      const bool narrow_result = I->dest[0].size == AGX_SIZE_16;
      agx_foreach_src(I, s) {
         if (I->src[s].type == AGX_INDEX_IMMEDIATE) {
            /* IMAD's multiplier and a nonnegative byte addend already have
             * native immediate forms. Preserve them while widening half
             * register operands; materializing them needlessly increases
             * pressure and can put IMAD operands outside its register bank. */
            const bool byte = I->src[s].value <= UINT8_MAX &&
                              !I->src[s].abs && !I->src[s].neg;
            const bool immediate_multiplier =
               I->src[1].type == AGX_INDEX_IMMEDIATE &&
               I->src[1].value <= UINT8_MAX &&
               !I->src[1].abs && !I->src[1].neg;
            if (byte && (s == 1 ||
                         (s == 2 && (I->src[s].value == 0 ||
                                     immediate_multiplier))))
               continue;
            I->src[s] = agx_mov_imm(&b, 32, I->src[s].value);
            continue;
         }
         if (I->src[s].size == AGX_SIZE_32)
            continue;
         agx_index wide = agx_temp(ctx, AGX_SIZE_32);
         agx_index src = I->src[s];
         bool zero_extend = narrow_result || src.abs;
         src.abs = false;
         if (zero_extend)
            agx_mov_to(&b, wide, src);
         else
            agx_signext_to(&b, wide, src);
         I->src[s] = wide;
      }
      if (!narrow_result)
         continue;
      agx_index half = I->dest[0];
      I->dest[0] = agx_temp(ctx, AGX_SIZE_32);
      b.cursor = agx_after_instr(I);
      agx_mov_to(&b, half, I->dest[0]);
   }
}

bool
agx_native_minifloat_fma_negate_enabled(void)
{
   return debug_get_bool_option("AGX_NATIVE_MINIFLOAT_FMA_NEGATE", false);
}

void
agx_lower_agx3_fma_sources(agx_context *ctx)
{
   if (ctx->key->dev.encoding != AGX_ISA_ENCODING_AGX3B)
      return;

   agx_foreach_instr_global_safe(ctx, I) {
      if ((I->op == AGX_OPCODE_ROUNDEVEN || I->op == AGX_OPCODE_FLOOR ||
           I->op == AGX_OPCODE_CEIL || I->op == AGX_OPCODE_TRUNC ||
           I->op == AGX_OPCODE_DFDX || I->op == AGX_OPCODE_DFDY ||
           I->op == AGX_OPCODE_RCP || I->op == AGX_OPCODE_RSQRT ||
           I->op == AGX_OPCODE_SRSQRT || I->op == AGX_OPCODE_LOG2 || I->op == AGX_OPCODE_EXP2) &&
          I->dest[0].size == AGX_SIZE_16) {
         /* Native unary math and derivatives use F32. Widen the input
          * exactly and round the result to half. Integral rounding is exact in both widths;
          * approximate reciprocal/transcendental operations retain the F32
          * unit's higher internal precision before the final half rounding. */
         agx_builder before = agx_init_builder(ctx, agx_before_instr(I));
         agx_index source = agx_temp(ctx, AGX_SIZE_32);
         agx_fmov_to(&before, source, I->src[0]);
         I->src[0] = source;
         agx_index half = I->dest[0];
         I->dest[0] = agx_temp(ctx, AGX_SIZE_32);
         agx_builder after = agx_init_builder(ctx, agx_after_instr(I));
         agx_fmov_to(&after, half, I->dest[0]);
         continue;
      }
      if ((I->op == AGX_OPCODE_ROUNDEVEN || I->op == AGX_OPCODE_FLOOR ||
           I->op == AGX_OPCODE_CEIL || I->op == AGX_OPCODE_TRUNC ||
           I->op == AGX_OPCODE_DFDX || I->op == AGX_OPCODE_DFDY ||
           I->op == AGX_OPCODE_RCP || I->op == AGX_OPCODE_RSQRT ||
           I->op == AGX_OPCODE_SRSQRT || I->op == AGX_OPCODE_LOG2 ||
           I->op == AGX_OPCODE_EXP2) &&
          I->dest[0].size == AGX_SIZE_32 &&
          I->src[0].size == AGX_SIZE_16 &&
          I->src[0].type != AGX_INDEX_IMMEDIATE &&
          I->src[0].type != AGX_INDEX_NULL) {
         /* Copyprop may fold an exact F16 widening into a F32 unary op.
          * Materialize it before RA: these native forms require F32 inputs
          * and must retain the F32 result's precision and range. */
         agx_builder b = agx_init_builder(ctx, agx_before_instr(I));
         agx_index wide = agx_temp(ctx, AGX_SIZE_32);
         agx_fmov_to(&b, wide, I->src[0]);
         I->src[0] = wide;
         continue;
      }
      if ((I->op == AGX_OPCODE_RCP || I->op == AGX_OPCODE_RSQRT ||
           I->op == AGX_OPCODE_SRSQRT) &&
          I->src[0].size == AGX_SIZE_32 &&
          I->src[0].type != AGX_INDEX_IMMEDIATE &&
          I->src[0].type != AGX_INDEX_NULL &&
          (I->src[0].abs || I->src[0].neg)) {
         /* Native reciprocal forms consume an unmodified F32 register.
          * Materialize abs/neg as sign-bit operations before RA, preserving
          * signed zeros and infinities without an extra floating rounding. */
         agx_builder b = agx_init_builder(ctx, agx_before_instr(I));
         agx_index src = I->src[0];
         const bool absolute = src.abs, negate = src.neg;
         src.abs = src.neg = false;
         if (absolute) {
            agx_index plain = agx_temp(ctx, AGX_SIZE_32);
            agx_bitop_to(&b, plain, src, agx_mov_imm(&b, 32, 0x7fffffff), 0x8);
            src = plain;
         }
         if (negate) {
            agx_index plain = agx_temp(ctx, AGX_SIZE_32);
            agx_bitop_to(&b, plain, src, agx_mov_imm(&b, 32, 0x80000000), 0x6);
            src = plain;
         }
         I->src[0] = src;
         continue;
      }
      if ((I->op == AGX_OPCODE_FADD && I->saturate &&
           I->dest[0].size == AGX_SIZE_16) ||
          I->op == AGX_OPCODE_FLTNORM_AGX3) {
         /* Saturated narrowing ADD and normalized packing take plain F32
          * sources. Apply sign modifiers exactly before the conversion;
          * retain its clamp and final rounding in the original operation. */
         agx_builder b = agx_init_builder(ctx, agx_before_instr(I));
         agx_foreach_src(I, s) {
            agx_index src = I->src[s];
            if (!src.abs && !src.neg)
               continue;
            if (src.type == AGX_INDEX_IMMEDIATE) {
               float value = agx_minifloat_decode(src.value);
               if (src.abs)
                  value = fabsf(value);
               if (src.neg)
                  value = -value;
               I->src[s] = agx_mov_imm(&b, 32, fui(value));
               continue;
            }
            if (src.size != AGX_SIZE_32)
               continue;
            const bool absolute = src.abs, negate = src.neg;
            src.abs = src.neg = false;
            if (absolute) {
               agx_index plain = agx_temp(ctx, AGX_SIZE_32);
               agx_bitop_to(&b, plain, src, agx_mov_imm(&b, 32, 0x7fffffff), 0x8);
               src = plain;
            }
            if (negate) {
               agx_index plain = agx_temp(ctx, AGX_SIZE_32);
               agx_bitop_to(&b, plain, src, agx_mov_imm(&b, 32, 0x80000000), 0x6);
               src = plain;
            }
            I->src[s] = src;
         }
      }
      if (I->op == AGX_OPCODE_FMUL && I->dest[0].size == AGX_SIZE_16) {
         agx_builder b = agx_init_builder(ctx, agx_before_instr(I));
         agx_foreach_src(I, s) {
            agx_index src = I->src[s];
            if (src.type == AGX_INDEX_IMMEDIATE) {
               float value = agx_minifloat_decode(src.value);
               if (src.abs) value = fabsf(value);
               if (src.neg) value = -value;
               I->src[s] = agx_mov_imm(&b, 32, fui(value));
            } else if (src.size == AGX_SIZE_16 || src.abs || src.neg) {
               agx_index wide = agx_temp(ctx, AGX_SIZE_32);
               agx_fmov_to(&b, wide, src);
               I->src[s] = wide;
            }
         }
         if (I->saturate) {
            /* FMA(a,b,-0) preserves multiplication, including signed zero,
             * and has a native saturated half-result form. */
            agx_instr *fma = agx_ffma_to(&b, I->dest[0], I->src[0], I->src[1],
                                        agx_mov_imm(&b, 32, 0x80000000));
            fma->saturate = true;
            agx_remove_instruction(I);
         }
         continue;
      }
      if ((I->op == AGX_OPCODE_FADD || I->op == AGX_OPCODE_FMUL) &&
          (I->dest[0].size == AGX_SIZE_32 ||
           (I->op == AGX_OPCODE_FMUL && I->dest[0].size == AGX_SIZE_16)) &&
          ((I->src[0].type != AGX_INDEX_IMMEDIATE && I->src[0].size == AGX_SIZE_16) ||
           (I->src[1].type != AGX_INDEX_IMMEDIATE && I->src[1].size == AGX_SIZE_16))) {
         /* F16 inputs widen exactly. Keep the F32 operation and its rounding
          * when copy propagation folds a half-width source into arithmetic. */
         agx_builder b = agx_init_builder(ctx, agx_before_instr(I));
         agx_foreach_src(I, s) {
            if (I->src[s].size == AGX_SIZE_16 &&
                I->src[s].type != AGX_INDEX_IMMEDIATE) {
               agx_index wide = agx_temp(ctx, AGX_SIZE_32);
               agx_fmov_to(&b, wide, I->src[s]);
               I->src[s] = wide;
            }
         }
         continue;
      }

      if (I->op != AGX_OPCODE_FFMA || I->nr_srcs != 3)
         continue;

      agx_builder b = agx_init_builder(ctx, agx_before_instr(I));
      if (I->src[0].type == AGX_INDEX_IMMEDIATE &&
          I->src[1].type == AGX_INDEX_IMMEDIATE) {
         /* Native immediate FMA forms require a register multiplicand.
          * Keep the fused operation (and signed-zero behavior) intact when
          * copy propagation exposes two constants, as in Heaven's shaders. */
         agx_index src = I->src[0];
         float value = agx_minifloat_decode(src.value);
         if (src.abs)
            value = fabsf(value);
         if (src.neg)
            value = -value;
         I->src[0] = agx_mov_imm(&b, 32, fui(value));
      }
      bool half_fma_source = false;
      agx_foreach_src(I, s)
         half_fma_source |= I->src[s].size == AGX_SIZE_16 &&
                            I->src[s].type != AGX_INDEX_IMMEDIATE &&
                            I->src[s].type != AGX_INDEX_NULL;
      if (half_fma_source && I->dest[0].size == AGX_SIZE_32 && !I->saturate) {
         /* Native F32 FMA widens each half operand independently while
          * preserving its ordinary F32 arithmetic. Keep the half registers;
          * materialize minifloats to use the qualified three-register form. */
         agx_foreach_src(I, s) {
            if (I->src[s].type != AGX_INDEX_IMMEDIATE)
               continue;
            agx_index src = I->src[s];
            float value = agx_minifloat_decode(src.value);
            if (src.abs) value = fabsf(value);
            if (src.neg) value = -value;
            I->src[s] = agx_mov_imm(&b, 32, fui(value));
         }
         continue;
      }
      if (I->dest[0].size == AGX_SIZE_16) {
         /* Native F32 FMA can round directly to half. Keep that single
          * operation and make its full-width register operands explicit. */
         agx_foreach_src(I, s) {
            agx_index src = I->src[s];
            if (src.type == AGX_INDEX_IMMEDIATE) {
               float value = agx_minifloat_decode(src.value);
               if (src.abs) value = fabsf(value);
               if (src.neg) value = -value;
               I->src[s] = agx_mov_imm(&b, 32, fui(value));
            } else if (src.size == AGX_SIZE_16) {
               agx_index wide = agx_temp(ctx, AGX_SIZE_32);
               agx_fmov_to(&b, wide, src);
               I->src[s] = wide;
            }
         }
         continue;
      }
      if (I->saturate) {
         if (I->dest[0].size != AGX_SIZE_32)
            continue;

         /* The grounded saturated FMA form accepts F32 registers, including
          * source sign/absolute modifiers. Materialize minifloats and widen
          * half inputs exactly before RA, keeping the original fused rounding
          * and final saturation. Firefox's WebRender emits -x*y + 0.5 here.
          */
         agx_foreach_src(I, s) {
            agx_index src = I->src[s];
            if (src.type == AGX_INDEX_IMMEDIATE && src.value <= UINT8_MAX) {
               float value = agx_minifloat_decode(src.value);
               if (src.abs)
                  value = fabsf(value);
               if (src.neg)
                  value = -value;
               I->src[s] = agx_mov_imm(&b, 32, fui(value));
            } else if (src.size == AGX_SIZE_16 &&
                       src.type != AGX_INDEX_IMMEDIATE &&
                       src.type != AGX_INDEX_NULL) {
               agx_index wide = agx_temp(ctx, AGX_SIZE_32);
               agx_fmov_to(&b, wide, src);
               I->src[s] = wide;
            }
         }
         continue;
      }

      agx_foreach_src(I, s) {
         if (I->src[s].size != AGX_SIZE_16 ||
             I->src[s].type == AGX_INDEX_IMMEDIATE ||
             I->src[s].type == AGX_INDEX_NULL)
            continue;

         /* Widen signed half inputs too: FMOV applies their abs/neg
          * before the F32 FMA, without changing its fused rounding. */
         agx_index widened = agx_temp(ctx, AGX_SIZE_32);
         agx_fmov_to(&b, widened, I->src[s]);
         I->src[s] = widened;
      }

      /* The existing native minifloat-multiplier encoder already supports
       * a negated F32 GPR addend. Do not turn that qualified modifier into
       * a separate sign-bit XOR. Keep all other mixed-source legalization. */
      if (ctx->stage == MESA_SHADER_FRAGMENT && !ctx->is_preamble &&
          !ctx->key->secondary && !ctx->nir->info.internal &&
          agx_native_minifloat_fma_negate_enabled()) {
         const bool imm_first = I->src[0].type == AGX_INDEX_IMMEDIATE;
         const agx_index imm = I->src[imm_first ? 0 : 1];
         const agx_index reg = I->src[imm_first ? 1 : 0];
         const agx_index add = I->src[2];
         if (I->dest[0].size == AGX_SIZE_32 &&
             agx_channels(I->dest[0]) == 1 &&
             imm.type == AGX_INDEX_IMMEDIATE && imm.value <= UINT8_MAX &&
             !imm.abs && !imm.neg && !imm.memory &&
             (reg.type == AGX_INDEX_NORMAL || reg.type == AGX_INDEX_REGISTER) &&
             reg.size == AGX_SIZE_32 && agx_channels(reg) == 1 &&
             !reg.abs && !reg.neg && !reg.memory &&
             (add.type == AGX_INDEX_NORMAL || add.type == AGX_INDEX_REGISTER) &&
             add.size == AGX_SIZE_32 && agx_channels(add) == 1 &&
             !add.abs && add.neg && !add.memory)
            continue;
      }

      bool mixed = false, has_modifier = false, unsupported_modifier = false;
      agx_foreach_src(I, s) {
         mixed |= I->src[s].type == AGX_INDEX_IMMEDIATE ||
                  I->src[s].size != AGX_SIZE_32;
         has_modifier |= I->src[s].abs || I->src[s].neg;
         unsupported_modifier |= ((I->src[s].abs || I->src[s].neg) &&
                                  (I->src[s].type == AGX_INDEX_IMMEDIATE ||
                                   I->src[s].size != AGX_SIZE_32));
      }

      /* The native modifier form takes three full F32 registers. Preserve
       * exact fabs/fneg semantics for mixed/minifloat FMA shapes by clearing
       * and then flipping the source sign bit before register allocation. The resulting FMA has no
       * modifier and stays on the already-grounded mixed-source form. */
      if (!mixed || !has_modifier || unsupported_modifier)
         continue;

      agx_foreach_src(I, s) {
         const bool absolute = I->src[s].abs, negate = I->src[s].neg;
         if (!absolute && !negate)
            continue;

         agx_index source = I->src[s];
         source.abs = source.neg = false;
         if (absolute) {
            agx_index mask = agx_mov_imm(&b, 32, UINT32_C(0x7fffffff));
            agx_index cleared = agx_temp(ctx, AGX_SIZE_32);
            agx_bitop_to(&b, cleared, source, mask, 0x8);
            source = cleared;
         }
         if (negate) {
            agx_index sign = agx_mov_imm(&b, 32, UINT32_C(0x80000000));
            agx_index flipped = agx_temp(ctx, AGX_SIZE_32);
            agx_bitop_to(&b, flipped, source, sign, 0x6);
            source = flipped;
         }
         I->src[s] = source;
      }
   }
}

/* The native half-width branch encoder supports register/register and
 * register/zero comparisons. Materialize other constants while SSA temporaries
 * can still be allocated; never invent a scratch register in the packer. */
bool
agx_lower_agx3_cf_constants(agx_context *ctx)
{
   if (ctx->key->dev.encoding != AGX_ISA_ENCODING_AGX3B)
      return false;

   bool progress = false;
   agx_foreach_instr_global_safe(ctx, I) {
      bool floating = false;
      switch (I->op) {
      case AGX_OPCODE_IF_FCMP:
      case AGX_OPCODE_WHILE_FCMP:
      case AGX_OPCODE_BREAK_IF_FCMP:
         floating = true;
         break;
      case AGX_OPCODE_IF_ICMP:
      case AGX_OPCODE_ELSE_ICMP:
      case AGX_OPCODE_WHILE_ICMP:
      case AGX_OPCODE_BREAK_IF_ICMP:
         break;
      default:
         continue;
      }
      if (I->nr_srcs != 2)
         continue;
      /* Constant propagation may leave a constant predicate inside
       * structured control flow. The native integer branch still requires
       * one register comparator; keep the branch's mask-stack semantics.
       */
      if (!floating && I->src[0].type == AGX_INDEX_IMMEDIATE &&
          I->src[1].type == AGX_INDEX_IMMEDIATE &&
          !I->src[0].abs && !I->src[0].neg) {
         agx_builder b = agx_init_builder(ctx, agx_before_instr(I));
         I->src[0] = agx_mov_imm(&b, 32, I->src[0].value);
         progress = true;
      }
      /* Native FCMP branches compare full floats. Widen actual half
       * operands after copy propagation, which could otherwise fold this
       * conversion back into the branch. Binary16 values are exact in F32.
       * Immediate zero keeps the separately supported register/zero form. */
      if (floating) {
         for (unsigned s = 0; s < 2; ++s) {
            if (I->src[s].type == AGX_INDEX_IMMEDIATE ||
                I->src[s].size != AGX_SIZE_16)
               continue;
            agx_builder b = agx_init_builder(ctx, agx_before_instr(I));
            agx_index wide = agx_temp(ctx, AGX_SIZE_32);
            agx_fmov_to(&b, wide, I->src[s]);
            I->src[s] = wide;
            progress = true;
         }
      }
      /* Native branch predicates have no qualified absolute-value operand.
       * Clear the IEEE sign bit before RA, preserving zeros, infinities and
       * NaN payloads. If abs is followed by negation, materialize that bit too
       * so the register/zero predicate remains a plain operand. Negated
       * equality and unordered relations likewise need plain operands;
       * only strict LT/GT currently have a native negated encoding. */
      if (floating) {
         for (unsigned s = 0; s < 2; ++s) {
            agx_index src = I->src[s];
            const bool native_neg =
               (I->fcond == AGX_FCOND_LT || I->fcond == AGX_FCOND_GT) &&
               !I->src[0].abs && !I->src[1].abs &&
               I->src[0].type != AGX_INDEX_IMMEDIATE &&
               I->src[1].type != AGX_INDEX_IMMEDIATE &&
               (I->src[0].neg != I->src[1].neg);
            if ((!src.abs && (!src.neg || native_neg)) ||
                src.type == AGX_INDEX_IMMEDIATE ||
                src.size != AGX_SIZE_32 || agx_channels(src) != 1)
               continue;
            agx_builder b = agx_init_builder(ctx, agx_before_instr(I));
            const bool negate = src.neg, clear_sign = src.abs;
            src.abs = src.neg = false;
            agx_index absolute = src;
            if (clear_sign) {
               absolute = agx_temp(ctx, AGX_SIZE_32);
               agx_bitop_to(&b, absolute, src,
                            agx_mov_imm(&b, 32, 0x7fffffff), 0x8);
            }
            if (negate) {
               agx_index negative = agx_temp(ctx, AGX_SIZE_32);
               agx_bitop_to(&b, negative, absolute,
                            agx_mov_imm(&b, 32, 0x80000000), 0x6);
               absolute = negative;
            }
            I->src[s] = absolute;
            progress = true;
         }
      }
      for (unsigned s = 0; s < 2; ++s) {
         agx_index src = I->src[s], other = I->src[1 - s];
         if (floating) {
            /* FCMP has a register/zero form, but arbitrary minifloats must
             * use its register/register form. Preserve IEEE value and source
             * modifiers while allocating the temporary through normal SSA. */
            if (src.type != AGX_INDEX_IMMEDIATE || src.value == 0 ||
                src.value > UINT8_MAX || other.type == AGX_INDEX_IMMEDIATE ||
                other.size != AGX_SIZE_32 || agx_channels(other) != 1)
               continue;
            float value = agx_minifloat_decode(src.value);
            if (src.abs)
               value = fabsf(value);
            if (src.neg)
               value = -value;
            agx_builder b = agx_init_builder(ctx, agx_before_instr(I));
            I->src[s] = agx_mov_imm(&b, 32, fui(value));
            progress = true;
            continue;
         }
         if (src.type != AGX_INDEX_IMMEDIATE || src.value == 0 ||
             src.value > UINT8_MAX || src.abs || src.neg ||
             other.type == AGX_INDEX_IMMEDIATE || other.size != AGX_SIZE_16 ||
             agx_channels(other) != 1)
            continue;
         agx_builder b = agx_init_builder(ctx, agx_before_instr(I));
         I->src[s] = agx_mov_imm(&b, 16, src.value);
         progress = true;
      }
   }
   return progress;
}

/* Half-width shift-add has register operands. Fold an immediate's integer
 * negation modulo 2^16 and allocate it normally before RA. This also covers
 * decrementing 16-bit loop counters without guessing a half subtract opcode. */
bool
agx_lower_agx3_half_iadd_constants(agx_context *ctx)
{
   if (ctx->key->dev.encoding != AGX_ISA_ENCODING_AGX3B)
      return false;

   bool progress = false;
   agx_foreach_instr_global_safe(ctx, I) {
      /* 64-bit adds have register operands too: no grounded form takes an
       * immediate (float_controls.fp32.input_args.rounding_rte_conv_from_
       * {int,uint}64_*_vert, whose constants stay immediates in vertex
       * shaders). */
      if (I->op != AGX_OPCODE_IADD || I->nr_dests != 1 || I->nr_srcs != 2 ||
          (I->dest[0].size != AGX_SIZE_16 && I->dest[0].size != AGX_SIZE_64) ||
          agx_channels(I->dest[0]) != 1 || I->saturate)
         continue;
      const bool wide = I->dest[0].size == AGX_SIZE_64;
      agx_foreach_src(I, s) {
         agx_index src = I->src[s];
         if (src.type != AGX_INDEX_IMMEDIATE || src.value > UINT8_MAX || src.abs)
            continue;
         agx_builder b = agx_init_builder(ctx, agx_before_instr(I));
         if (wide) {
            uint64_t value = src.neg ? -(uint64_t)src.value : src.value;
            I->src[s] = agx_mov_imm(&b, 64, value);
         } else {
            uint16_t value = src.neg ? -src.value : src.value;
            I->src[s] = agx_mov_imm(&b, 16, value);
         }
         progress = true;
      }
   }
   return progress;
}

/* The native full-width subtract replaces the carry chain. Addition uses
 * a - (-b), modulo 64 bits, with no signed-overflow assumption. */
static bool
agx_lower_agx3_wide_iadd(agx_context *ctx)
{
   if (ctx->key->dev.encoding != AGX_ISA_ENCODING_AGX3B)
      return false;
   bool progress = false;
   agx_foreach_instr_global_safe(ctx, I) {
      if (I->op != AGX_OPCODE_IADD || I->nr_dests != 1 || I->nr_srcs != 2 ||
          I->dest[0].size != AGX_SIZE_64 || agx_channels(I->dest[0]) != 1 ||
          I->shift != 0 || I->saturate)
         continue;
      bool wide_regs = true;
      agx_foreach_src(I, s) {
         if ((I->src[s].type != AGX_INDEX_NORMAL &&
              I->src[s].type != AGX_INDEX_UNIFORM) ||
             I->src[s].size != AGX_SIZE_64 || I->src[s].abs ||
             (s == 0 && I->src[s].neg) || agx_channels(I->src[s]) != 1)
            wide_regs = false;
      }
      if (!wide_regs || I->src[1].neg)
         continue;
      agx_builder b = agx_init_builder(ctx, agx_before_instr(I));
      agx_index neg = agx_temp(ctx, AGX_SIZE_64);
      agx_iadd_to(&b, neg, agx_zero(), agx_neg(I->src[1]), 0);
      I->src[1] = agx_neg(neg);
      progress = true;
   }
   return progress;
}

/* Adding zero with IADD computes an extension and, for source 1, a left
 * shift. AGX IR plain half sources are signed; abs selects zero extension.
 * Legalize before SIGNEXT lowering and allocation, preserving shift/order. */
bool
agx_lower_agx3_widening_iadd(agx_context *ctx)
{
   if (ctx->key->dev.encoding != AGX_ISA_ENCODING_AGX3B)
      return false;

   bool progress = false;
   agx_foreach_instr_global_safe(ctx, I) {
      if (I->op != AGX_OPCODE_IADD || I->nr_dests != 1 || I->nr_srcs != 2 ||
          I->dest[0].size != AGX_SIZE_32 || agx_channels(I->dest[0]) != 1 ||
          I->saturate)
         continue;
      bool imm_first = I->src[0].type == AGX_INDEX_IMMEDIATE;
      agx_index zero = I->src[imm_first ? 0 : 1];
      agx_index half = I->src[imm_first ? 1 : 0];
      if (zero.type != AGX_INDEX_IMMEDIATE || zero.value || zero.abs || zero.neg ||
          half.type == AGX_INDEX_IMMEDIATE || half.type == AGX_INDEX_NULL ||
          (half.size != AGX_SIZE_16 && half.size != AGX_SIZE_32) ||
          agx_channels(half) != 1 || half.neg) {
         /* General mixed-width addition needs the same signed/unsigned
          * extension even when the other operand is nonzero. Keep the add
          * and its source-1 shift after materializing the extension. */
         agx_builder b = agx_init_builder(ctx, agx_before_instr(I));
         agx_foreach_src(I, s) {
            agx_index src = I->src[s];
            if (src.size != AGX_SIZE_16 ||
                src.type == AGX_INDEX_IMMEDIATE || src.type == AGX_INDEX_NULL ||
                agx_channels(src) != 1)
               continue;
            bool is_unsigned = src.abs;
            bool subtract = src.neg;
            src.abs = false;
            src.neg = false;
            agx_index wide = agx_temp(ctx, AGX_SIZE_32);
            if (is_unsigned)
               agx_mov_to(&b, wide, src);
            else
               agx_signext_to(&b, wide, src);
            /* Negation is subtraction, independent of extension. */
            I->src[s] = subtract ? agx_neg(wide) : wide;
            progress = true;
         }
         continue;
      }
      agx_builder b = agx_init_builder(ctx, agx_before_instr(I));
      bool is_unsigned = half.abs || half.size == AGX_SIZE_32;
      unsigned shift = imm_first ? I->shift : 0;
      agx_index extended = shift ? agx_temp(ctx, AGX_SIZE_32) : I->dest[0];
      half.abs = false;
      if (is_unsigned)
         agx_mov_to(&b, extended, half);
      else
         agx_signext_to(&b, extended, half);
      if (shift)
         agx_bfi_to(&b, I->dest[0], agx_zero(), extended,
                     agx_mov_imm(&b, 32, shift), 0);
      agx_remove_instruction(I);
      progress = true;
   }
   return progress;
}

/* Native register subtraction has no shifted-source form in the current
 * encoder. Keep modulo-32-bit semantics by shifting before subtracting. */
static bool
agx_lower_agx3_shifted_subtract(agx_context *ctx)
{
   if (ctx->key->dev.encoding != AGX_ISA_ENCODING_AGX3B)
      return false;
   bool progress = false;
   agx_foreach_instr_global_safe(ctx, I) {
      if (I->op != AGX_OPCODE_IADD || !I->shift || I->saturate ||
          I->dest[0].size != AGX_SIZE_32 || !I->src[1].neg ||
          I->src[1].size != AGX_SIZE_32 || I->src[1].abs)
         continue;
      agx_builder b = agx_init_builder(ctx, agx_before_instr(I));
      agx_index src = I->src[1];
      src.neg = false;
      agx_index shifted = agx_temp(ctx, AGX_SIZE_32);
      agx_bfi_to(&b, shifted, agx_zero(), src,
                 agx_mov_imm(&b, 32, I->shift), 0);
      I->src[1] = agx_neg(shifted);
      I->shift = 0;
      progress = true;
   }
   return progress;
}

static void
agx_materialize_agx3_alu_constants(agx_context *ctx)
{
   agx_foreach_instr_global_safe(ctx, I) {
      agx_builder b = agx_init_builder(ctx, agx_before_instr(I));
      if (ctx->key->dev.encoding == AGX_ISA_ENCODING_AGX3B &&
          I->op == AGX_OPCODE_IMAD && I->dest[0].size == AGX_SIZE_64) {
         for (unsigned s = 0; s < 2; ++s) {
            if (I->src[s].type == AGX_INDEX_IMMEDIATE) {
               bool is_unsigned = I->src[s].abs;
               I->src[s] = agx_mov_imm(&b, 32, I->src[s].value);
               I->src[s].abs = is_unsigned;
            }
         }
      }
      if (ctx->key->dev.encoding == AGX_ISA_ENCODING_AGX3B &&
          (I->op == AGX_OPCODE_ROUNDEVEN || I->op == AGX_OPCODE_FLOOR ||
           I->op == AGX_OPCODE_CEIL || I->op == AGX_OPCODE_TRUNC ||
           I->op == AGX_OPCODE_DFDX || I->op == AGX_OPCODE_DFDY ||
           I->op == AGX_OPCODE_RCP || I->op == AGX_OPCODE_RSQRT ||
           I->op == AGX_OPCODE_SRSQRT || I->op == AGX_OPCODE_LOG2 ||
           I->op == AGX_OPCODE_EXP2) &&
          I->src[0].type == AGX_INDEX_IMMEDIATE) {
         /* Unary math has register-only native forms. Late image-format
          * lowering and copy propagation can expose minifloats even when
          * NIR constant folding ran earlier. Preserve their floating value
          * and sign modifiers, including signed zero, before RA. */
         agx_index src = I->src[0];
         float value = agx_minifloat_decode(src.value);
         if (src.abs)
            value = fabsf(value);
         if (src.neg)
            value = -value;
         I->src[0] = agx_mov_imm(&b, 32, fui(value));
      }
      if (ctx->key->dev.encoding == AGX_ISA_ENCODING_AGX3B &&
          I->op == AGX_OPCODE_PACK_RGB10A2_AGX3) {
         /* The pack consumes plain F32 lanes; blend arithmetic can leave
          * negation or absolute-value modifiers on any component. */
         for (unsigned s = 0; s < I->nr_srcs; ++s) {
            if (I->src[s].abs || I->src[s].neg) {
               agx_index value = agx_temp(ctx, AGX_SIZE_32);
               agx_fmov_to(&b, value, I->src[s]);
               I->src[s] = value;
            }
         }
      }
      if (ctx->key->dev.encoding == AGX_ISA_ENCODING_AGX3B &&
          I->op == AGX_OPCODE_BITREV && I->src[0].size == AGX_SIZE_16) {
         /* Copy propagation may narrow an otherwise word-wide reversal. */
         agx_index wide = agx_temp(ctx, AGX_SIZE_32);
         agx_mov_to(&b, wide, I->src[0]);
         I->src[0] = wide;
      }
      if ((I->op == AGX_OPCODE_PACK_LOD_OFFSET_AGX3 ||
           I->op == AGX_OPCODE_PACK_MIN_LOD_AGX3) &&
          I->src[0].type == AGX_INDEX_IMMEDIATE) {
         /* Copyprop folds the half LOD back to a raw immediate. The native
          * pack reads a half register, so restore its bit pattern before RA. */
         I->src[0] = agx_mov_imm(&b, 16, I->src[0].value);
      }
      if (I->op == AGX_OPCODE_PACK_LOD_OFFSET_AGX3 &&
          I->src[1].type == AGX_INDEX_IMMEDIATE)
         I->src[1] = agx_mov_imm(&b, 32, I->src[1].value);
      /* ISEL accepts a register comparator and an immediate comparator,
       * including ICMP pseudos lowered to ISEL after register allocation.
       */
      if (ctx->key->dev.encoding == AGX_ISA_ENCODING_AGX3B &&
          (I->op == AGX_OPCODE_ICMP || I->op == AGX_OPCODE_ICMPSEL) &&
          I->src[0].type == AGX_INDEX_IMMEDIATE &&
          I->src[1].type == AGX_INDEX_IMMEDIATE &&
          !I->src[0].abs && !I->src[0].neg)
         I->src[0] = agx_mov_imm(&b, 32, I->src[0].value);
      /* A negated full-width IADD source means subtraction, not signed
       * extension. Materialize the minuend so the native register ISUB can
       * implement constant - value, including the negation in udiv refinement.
       * Shifted subtraction has already been lowered to a separate shift. */
      if (ctx->key->dev.encoding == AGX_ISA_ENCODING_AGX3B &&
          I->op == AGX_OPCODE_IADD && I->nr_dests == 1 && I->nr_srcs == 2 &&
          !I->saturate && !I->shift && I->dest[0].size == AGX_SIZE_32 &&
          I->src[0].type == AGX_INDEX_IMMEDIATE && !I->src[0].abs &&
          !I->src[0].neg && I->src[1].size == AGX_SIZE_32 &&
          I->src[1].neg && !I->src[1].abs &&
          I->src[1].type != AGX_INDEX_IMMEDIATE) {
         I->src[0] = agx_mov_imm(&b, 32, I->src[0].value);
      }
      /* SIGNEXT becomes a signed half IADD on AGX2. Keep the AGX3 lowering
       * in SSA, where temporary registers are still available: preserve the
       * low 16 bits, move the sign bit to bit 31, then shift arithmetically.
       */
      if (I->op == AGX_OPCODE_SIGNEXT &&
          ctx->key->dev.encoding == AGX_ISA_ENCODING_AGX3B) {
         agx_index wide = agx_temp(ctx, AGX_SIZE_32);
         agx_mov_to(&b, wide, I->src[0]);
         agx_index shifted = agx_temp(ctx, AGX_SIZE_32);
         agx_bfi_to(&b, shifted, agx_zero(), wide,
                    agx_mov_imm(&b, 32, 16), 0);
         agx_asr_to(&b, I->dest[0], shifted, agx_immediate(16));
         agx_remove_instruction(I);
         continue;
      }
      /* General FSEL has register comparators or an implicit zero. Nonzero
       * minifloats in compare slots must first become floating registers;
       * selected values in slots 2/3 have integer bit-pattern semantics.
       * FCMP is lowered to a boolean FSEL later, so normalize it here too.
       */
      if (I->op == AGX_OPCODE_FCMPSEL || I->op == AGX_OPCODE_FCMP) {
         if (ctx->key->dev.encoding == AGX_ISA_ENCODING_AGX3B &&
             I->op == AGX_OPCODE_FCMP) {
            /* Boolean FCMP becomes an immediate-alternative FSEL after this
             * pass; that form also needs full-width comparators. */
            for (unsigned s = 0; s < 2; ++s) {
               if (I->src[s].type != AGX_INDEX_IMMEDIATE &&
                   I->src[s].size == AGX_SIZE_16) {
                  agx_index wide = agx_temp(ctx, AGX_SIZE_32);
                  agx_fmov_to(&b, wide, I->src[s]);
                  I->src[s] = wide;
               }
            }
         }
         if (ctx->key->dev.encoding == AGX_ISA_ENCODING_AGX3B &&
             I->op == AGX_OPCODE_FCMPSEL && I->dest[0].size == AGX_SIZE_16) {
            /* Select alternatives are bit patterns, not minifloats. */
            for (unsigned s = 2; s < 4; ++s) {
               if (I->src[s].type == AGX_INDEX_IMMEDIATE)
                  I->src[s] = agx_mov_imm(&b, 16, I->src[s].value);
            }
         }
         /* The native F32/F16 route is qualified for EQ/LT and two full
          * register alternatives. Other full-result shapes use the general
          * F32 comparator route. F16->F32 is exact, including signed zero;
          * retain the condition, inversion, and selected bit patterns. */
         if (ctx->key->dev.encoding == AGX_ISA_ENCODING_AGX3B &&
             I->op == AGX_OPCODE_FCMPSEL && I->dest[0].size == AGX_SIZE_32) {
            const bool native_mixed =
               I->src[0].size == AGX_SIZE_32 &&
               I->src[0].type != AGX_INDEX_IMMEDIATE &&
               I->src[1].size == AGX_SIZE_16 &&
               I->src[1].type != AGX_INDEX_IMMEDIATE &&
               !I->invert_cond &&
               (I->fcond == AGX_FCOND_EQ || I->fcond == AGX_FCOND_LT) &&
               I->src[2].type != AGX_INDEX_IMMEDIATE &&
               I->src[3].type != AGX_INDEX_IMMEDIATE &&
               I->src[2].size == AGX_SIZE_32 && I->src[3].size == AGX_SIZE_32;
            if (!native_mixed) {
               for (unsigned c = 0; c < 2; ++c) {
                  if (I->src[c].type == AGX_INDEX_IMMEDIATE ||
                      I->src[c].size != AGX_SIZE_16)
                     continue;
                  agx_index wide = agx_temp(ctx, AGX_SIZE_32);
                  agx_fmov_to(&b, wide, I->src[c]);
                  I->src[c] = wide;
               }
            }
         }
         if (ctx->key->dev.encoding == AGX_ISA_ENCODING_AGX3B &&
             I->op == AGX_OPCODE_FCMPSEL &&
             I->dest[0].size == AGX_SIZE_16 &&
             !(agx_is_equiv(I->src[0], I->src[2]) &&
               agx_is_equiv(I->src[1], I->src[3]) &&
               !I->invert_cond &&
               (I->fcond == AGX_FCOND_LTN || I->fcond == AGX_FCOND_GTN))) {
            /* Only the tied native half min/max form accepts two half
             * comparators. Widen general comparisons exactly, independently
             * of the bit-pattern width of their selected alternatives. */
            for (unsigned c = 0; c < 2; ++c) {
               if (I->src[c].type == AGX_INDEX_IMMEDIATE ||
                   I->src[c].size != AGX_SIZE_16)
                  continue;
               agx_index wide = agx_temp(ctx, AGX_SIZE_32);
               agx_fmov_to(&b, wide, I->src[c]);
               I->src[c] = wide;
            }
         }
         const bool modified = I->src[0].abs || I->src[0].neg ||
                               I->src[1].abs || I->src[1].neg;
         for (unsigned s = 0; s < 2; ++s) {
            agx_index src = I->src[s];
            if (src.type != AGX_INDEX_IMMEDIATE && modified &&
                (src.abs || src.neg || src.size == AGX_SIZE_16)) {
               agx_index plain = agx_temp(ctx, AGX_SIZE_32);
               if (src.size == AGX_SIZE_32 && src.abs && src.neg) {
                  /* -|x| has no grounded F32 move; clear then flip the
                   * sign bit (robustness2.*.vertex_attribute_fetch.
                   * no_fmt_qual.len_25{2,6}.*.1d.vert). */
                  agx_index bare = src;
                  bare.abs = bare.neg = false;
                  agx_index cleared = agx_temp(ctx, AGX_SIZE_32);
                  agx_bitop_to(&b, cleared, bare,
                               agx_mov_imm(&b, 32, 0x7fffffff), 0x8);
                  agx_bitop_to(&b, plain, cleared,
                               agx_mov_imm(&b, 32, 0x80000000), 0x6);
               } else {
                  agx_fmov_to(&b, plain, src);
               }
               I->src[s] = plain;
               continue;
            }
            if (src.type != AGX_INDEX_IMMEDIATE || src.value == 0)
               continue;

            float value = agx_minifloat_decode(src.value);
            if (src.abs)
               value = fabsf(value);
            if (src.neg)
               value = -value;
            I->src[s] = agx_mov_imm(&b, 32, fui(value));
         }
      }

      /* Same-width F16 fmov is an FADD with negative zero. Apply sign
       * modifiers to the bits, preserving the half value and signed zero. */
      /* The 32-bit form has grounded |x| and -|x|+1 encodings only; -|x|
       * takes the same bit route (robustness2.*.vertex_attribute_fetch.
       * no_fmt_qual.len_25{2,6}.*.1d.vert). */
      if (ctx->key->dev.encoding == AGX_ISA_ENCODING_AGX3B &&
          I->op == AGX_OPCODE_FADD && !I->saturate &&
          ((I->dest[0].size == AGX_SIZE_16 &&
            (I->src[0].size == AGX_SIZE_16 ||
             (I->src[0].size == AGX_SIZE_32 &&
              (I->src[0].abs || I->src[0].neg)))) ||
           (I->dest[0].size == AGX_SIZE_32 && I->src[0].size == AGX_SIZE_32 &&
            I->src[0].abs && I->src[0].neg)) &&
          I->src[0].type != AGX_INDEX_IMMEDIATE &&
          I->src[1].type == AGX_INDEX_IMMEDIATE &&
          ((I->src[1].value == 0 && I->src[1].neg) ||
           (I->src[1].value == agx_minifloat_encode(-0.0f) &&
            !I->src[1].neg && !I->src[1].abs))) {
         agx_index source = I->src[0];
         const bool absolute = source.abs, negate = source.neg;
         source.abs = source.neg = false;
         if (absolute) {
            agx_index cleared = agx_temp(ctx, source.size);
            agx_bitop_to(&b, cleared, source,
                         agx_mov_imm(&b, agx_size_align_16(source.size) * 16,
                            source.size == AGX_SIZE_16 ? 0x7fff : 0x7fffffff), 0x8);
            source = cleared;
         }
         if (negate) {
            agx_index flipped = agx_temp(ctx, source.size);
            agx_bitop_to(&b, flipped, source,
                         agx_mov_imm(&b, agx_size_align_16(source.size) * 16,
                            source.size == AGX_SIZE_16 ? 0x8000 : 0x80000000), 0x6);
            source = flipped;
         }
         if (source.size == AGX_SIZE_16)
            agx_mov_to(&b, I->dest[0], source);
         else
            agx_fmov_to(&b, I->dest[0], source);
         agx_remove_instruction(I);
         continue;
      }

      /* Half arithmetic uses register operands. Materialize minifloats
       * without changing the operation, fused behavior or half rounding. */
      if (ctx->key->dev.encoding == AGX_ISA_ENCODING_AGX3B &&
          (I->op == AGX_OPCODE_HFMA || I->op == AGX_OPCODE_HADD ||
           I->op == AGX_OPCODE_HMUL)) {
         agx_foreach_src(I, s) {
            agx_index src = I->src[s];
            if (src.type != AGX_INDEX_IMMEDIATE) {
               const bool lower_neg = I->op == AGX_OPCODE_HMUL ||
                  (I->op == AGX_OPCODE_HADD && I->src[0].neg && I->src[1].neg);
               if (!src.abs && !(src.neg && lower_neg))
                  continue;
               const bool absolute = src.abs, negate = src.neg;
               src.abs = src.neg = false;
               if (absolute) {
                  agx_index cleared = agx_temp(ctx, AGX_SIZE_16);
                  agx_bitop_to(&b, cleared, src, agx_mov_imm(&b, 16, 0x7fff), 0x8);
                  src = cleared;
               }
               if (negate && lower_neg) {
                  agx_index flipped = agx_temp(ctx, AGX_SIZE_16);
                  agx_bitop_to(&b, flipped, src, agx_mov_imm(&b, 16, 0x8000), 0x6);
                  src = flipped;
               } else {
                  src.neg = negate;
               }
               I->src[s] = src;
               continue;
            }
            float value = agx_minifloat_decode(src.value);
            if (src.abs)
               value = fabsf(value);
            if (src.neg)
               value = -value;
            I->src[s] = agx_mov_imm(&b, 16, _mesa_float_to_half(value));
         }
         if (I->op == AGX_OPCODE_HADD && I->saturate) {
            /* a*1+b has the same half rounding as a+b and a native clamp. */
            agx_instr *fma = agx_hfma_to(&b, I->dest[0], I->src[0],
                                        agx_mov_imm(&b, 16, 0x3c00), I->src[1]);
            fma->saturate = true;
            agx_remove_instruction(I);
         }
      }

      /* Compact add/multiply immediates cannot carry absolute-value
       * modifiers, and multiply's compact form cannot saturate. The native
       * register form supports these combinations without splitting the
       * arithmetic or changing its rounding.
       */
      if ((I->op == AGX_OPCODE_FADD || I->op == AGX_OPCODE_FMUL) &&
          (I->dest[0].size == AGX_SIZE_32 ||
           (I->op == AGX_OPCODE_FMUL && I->dest[0].size == AGX_SIZE_16)) &&
          (I->src[0].abs || I->src[1].abs ||
           (I->op == AGX_OPCODE_FMUL && I->saturate))) {
         for (unsigned s = 0; s < 2; ++s) {
            agx_index src = I->src[s];
            if (src.type != AGX_INDEX_IMMEDIATE)
               continue;

            float value = agx_minifloat_decode(src.value);
            if (src.abs)
               value = fabsf(value);
            if (src.neg)
               value = -value;
            I->src[s] = agx_mov_imm(&b, 32, fui(value));
         }
      }

      /* The native IMADD extension selector does not negate a full-width
       * addend. Split a * b - c into multiply and subtract, keeping the
       * modulo-2^32 result for register as well as immediate addends.
       * Preamble operands are still uniforms here; they become registers
       * in agx_lower_uniform_sources after this legalization pass. */
      if (I->op == AGX_OPCODE_IMAD && I->nr_dests == 1 && I->nr_srcs == 3 &&
          I->dest[0].size == AGX_SIZE_32 && I->shift == 0 &&
          !I->src[1].abs && !I->src[1].neg &&
          ((I->src[1].type == AGX_INDEX_IMMEDIATE &&
            I->src[1].value <= UINT8_MAX) ||
           ((I->src[1].type == AGX_INDEX_NORMAL ||
             I->src[1].type == AGX_INDEX_UNIFORM) &&
            I->src[1].size == AGX_SIZE_32)) &&
          !I->src[2].abs && I->src[2].neg &&
          ((I->src[2].type == AGX_INDEX_IMMEDIATE &&
            I->src[2].value <= UINT8_MAX) ||
           ((I->src[2].type == AGX_INDEX_NORMAL ||
             I->src[2].type == AGX_INDEX_UNIFORM) &&
            I->src[2].size == AGX_SIZE_32))) {
         agx_index product = agx_temp(ctx, AGX_SIZE_32);
         agx_imad_to(&b, product, I->src[0], I->src[1], agx_zero(), 0);
         I->op = AGX_OPCODE_IADD;
         I->src[0] = product;
         I->src[1] = I->src[2];
         I->nr_srcs = 2;
      }

      if ((I->op == AGX_OPCODE_BFI || I->op == AGX_OPCODE_BFEIL) &&
          I->nr_dests == 1 && I->nr_srcs == 3 &&
          I->dest[0].size == AGX_SIZE_32 &&
          I->src[2].type == AGX_INDEX_IMMEDIATE) {
         I->src[2] = agx_mov_imm(&b, 32, I->src[2].value);
      }

      /* G17P's register-selected shift has no immediate value operand.  NIR
       * uses BFI(0, 1, shift) for 1u << shift, so materialize that value just
       * like the variable shift amount above. */
      if ((I->op == AGX_OPCODE_BFI || I->op == AGX_OPCODE_BFEIL) &&
          I->nr_dests == 1 && I->nr_srcs == 3 &&
          I->dest[0].size == AGX_SIZE_32 &&
          I->src[1].type == AGX_INDEX_IMMEDIATE) {
         I->src[1] = agx_mov_imm(&b, 32, I->src[1].value);
      } else if (I->op == AGX_OPCODE_BITOP && I->nr_dests == 1 &&
                 (I->dest[0].size == AGX_SIZE_32 ||
                  (I->dest[0].size == AGX_SIZE_16 &&
                   (I->truth_table == 0x8 || I->truth_table == 0xe ||
                    I->truth_table == 0x6)))) {
         agx_foreach_src(I, s) {
            if (I->src[s].type == AGX_INDEX_IMMEDIATE)
               I->src[s] = agx_mov_imm(&b, I->dest[0].size == AGX_SIZE_16 ? 16 : 32,
                                      I->src[s].value);
         }
      }
   }
}

/* Half-precision ALU instructions cannot encode 32-bit register sources.
 * Convert them explicitly. Immediate and null sources have no size bit. */
static agx_index
agx_coerce_f16_src(agx_builder *b, agx_index src)
{
   if (src.size == AGX_SIZE_16 || src.type == AGX_INDEX_IMMEDIATE ||
       src.type == AGX_INDEX_NULL)
      return src;

   agx_index tmp = agx_temp(b->shader, AGX_SIZE_16);
   agx_fmov_to(b, tmp, src);
   return tmp;
}

static agx_index
agx_g17p_ushr_imm(agx_builder *b, agx_index value, unsigned shift)
{
   assert(shift >= 1 && shift <= 31);
   return agx_ushr(b, value, agx_mov_imm(b, 32, shift));
}

static agx_index
agx_g17p_and_imm(agx_builder *b, agx_index value, uint32_t mask)
{
   return agx_and(b, value, agx_mov_imm(b, 32, mask));
}

static agx_index
agx_g17p_popcount(agx_builder *b, agx_index scan)
{
   assert(scan.size == AGX_SIZE_32);
   agx_index odd = agx_g17p_and_imm(
      b, agx_g17p_ushr_imm(b, scan, 1), UINT32_C(0x55555555));
   scan = agx_iadd(b, scan, agx_neg(odd), 0);

   agx_index pairs_lo = agx_g17p_and_imm(b, scan, UINT32_C(0x33333333));
   agx_index pairs_hi = agx_g17p_and_imm(
      b, agx_g17p_ushr_imm(b, scan, 2), UINT32_C(0x33333333));
   scan = agx_iadd(b, pairs_lo, pairs_hi, 0);

   scan = agx_g17p_and_imm(
      b, agx_iadd(b, scan, agx_g17p_ushr_imm(b, scan, 4), 0),
      UINT32_C(0x0f0f0f0f));
   scan = agx_iadd(b, scan, agx_g17p_ushr_imm(b, scan, 8), 0);
   scan = agx_iadd(b, scan, agx_g17p_ushr_imm(b, scan, 16), 0);

   return agx_g17p_and_imm(b, scan, 0x3f);
}

static agx_instr *
agx_emit_g17p_ufind_msb(agx_builder *b, agx_index dst, agx_index value)
{
   /* Smear the highest bit, count the prefix, then subtract one. Zero
    * naturally returns -1. Use the same integer SWAR sequence as bitCount.
    */
   assert(value.size == AGX_SIZE_32 && dst.size == AGX_SIZE_32);
   agx_index scan = value;
   static const unsigned smear_shifts[] = {1, 2, 4, 8, 16};
   for (unsigned i = 0; i < ARRAY_SIZE(smear_shifts); ++i)
      scan = agx_or(b, scan, agx_g17p_ushr_imm(b, scan, smear_shifts[i]));
   agx_index count = agx_g17p_popcount(b, scan);
   return agx_iadd_to(b, dst, count, agx_neg(agx_immediate(1)), 0);
}

static agx_instr *
agx_emit_alu(agx_builder *b, nir_alu_instr *instr)
{
   unsigned srcs = nir_op_infos[instr->op].num_inputs;
   unsigned sz = instr->def.bit_size;
   unsigned src_sz = srcs ? nir_src_bit_size(instr->src[0].src) : 0;
   ASSERTED unsigned comps = instr->def.num_components;

   assert(comps == 1 || nir_op_is_vec_or_mov(instr->op));
   assert(sz == 1 ||
          ((nir_op_is_vec_or_mov(instr->op) ||
            is_conversion_to_8bit(instr->op) || instr->op == nir_op_bcsel) &&
           sz == 8) ||
          sz == 16 || sz == 32 || sz == 64);

   agx_index dst = agx_def_index(&instr->def);
   agx_index s0 = srcs > 0 ? agx_alu_src_index(b, instr->src[0]) : agx_null();
   agx_index s1 = srcs > 1 ? agx_alu_src_index(b, instr->src[1]) : agx_null();
   agx_index s2 = srcs > 2 ? agx_alu_src_index(b, instr->src[2]) : agx_null();
   agx_index s3 = srcs > 3 ? agx_alu_src_index(b, instr->src[3]) : agx_null();

   agx_index i0 = agx_immediate(0);
   agx_index i1 = agx_immediate(1);

#define UNOP(nop, aop)                                                         \
   case nir_op_##nop:                                                          \
      return agx_##aop##_to(b, dst, s0);
#define BINOP(nop, aop)                                                        \
   case nir_op_##nop:                                                          \
      return agx_##aop##_to(b, dst, s0, s1);

   switch (instr->op) {
      UNOP(f2f16, fmov);
      UNOP(f2f16_rtne, fmov);
      UNOP(f2f32, fmov);
      UNOP(fround_even, roundeven);
      UNOP(ftrunc, trunc);
      UNOP(ffloor, floor);
      UNOP(fceil, ceil);
      UNOP(frcp, rcp);
      UNOP(frsq, rsqrt);
      UNOP(flog2, log2);
      UNOP(fexp2, exp2);

      UNOP(mov, mov);
      UNOP(u2u32, mov);
   case nir_op_bitfield_reverse:
      if (b->shader->key->dev.encoding == AGX_ISA_ENCODING_AGX3B && sz == 16) {
         /* Native reversal is word-wide. Move the reversed low half back
          * down before truncating the result to the NIR width. */
         agx_index wide = agx_temp(b->shader, AGX_SIZE_32);
         agx_mov_to(b, wide, s0);
         agx_index reversed = agx_bitrev(b, wide);
         return agx_subdivide_to(b, dst, reversed, 1);
      }
      return agx_bitrev_to(b, dst, s0);
   case nir_op_bit_count:
      if (b->shader->key->dev.encoding == AGX_ISA_ENCODING_AGX3B &&
          s0.size <= AGX_SIZE_32) {
         if (s0.size == AGX_SIZE_16) {
            agx_index wide = agx_temp(b->shader, AGX_SIZE_32);
            agx_mov_to(b, wide, s0);
            s0 = wide;
         }
         return agx_mov_to(b, dst, agx_g17p_popcount(b, s0));
      }
      return agx_popcount_to(b, dst, s0);
   case nir_op_ufind_msb:
      if (b->shader->key->dev.encoding == AGX_ISA_ENCODING_AGX3B &&
          src_sz == 32)
         return agx_emit_g17p_ufind_msb(b, dst, s0);
      return agx_ffs_to(b, dst, s0);
      BINOP(iand, and);
      BINOP(ior, or);
      BINOP(ixor, xor);
      BINOP(interleave, intl);

   case nir_op_fadd:
      if (instr->def.bit_size == 16)
         return agx_hadd_to(b, dst, agx_coerce_f16_src(b, s0),
                            agx_coerce_f16_src(b, s1));
      else
         return agx_fadd_to(b, dst, s0, s1);

   case nir_op_fmul:
      if (instr->def.bit_size == 16)
         return agx_hmul_to(b, dst, agx_coerce_f16_src(b, s0),
                            agx_coerce_f16_src(b, s1));
      else
         return agx_fmul_to(b, dst, s0, s1);

   case nir_op_ffma:
      if (instr->def.bit_size == 16)
         return agx_hfma_to(b, dst, agx_coerce_f16_src(b, s0),
                            agx_coerce_f16_src(b, s1),
                            agx_coerce_f16_src(b, s2));
      else
         return agx_ffma_to(b, dst, s0, s1, s2);

   case nir_op_fsat: {
      agx_instr *I = agx_fadd_to(b, dst, s0, agx_negzero());
      if (instr->def.bit_size == 16)
         I->op = AGX_OPCODE_HADD;

      I->saturate = true;
      return I;
   }

   case nir_op_feq:
      return agx_fcmp_to(b, dst, s0, s1, AGX_FCOND_EQ, false);
   case nir_op_flt:
      return agx_fcmp_to(b, dst, s0, s1, AGX_FCOND_LT, false);
   case nir_op_fge:
      return agx_fcmp_to(b, dst, s0, s1, AGX_FCOND_GE, false);
   case nir_op_fneu:
      return agx_fcmp_to(b, dst, s0, s1, AGX_FCOND_EQ, true);

   case nir_op_ieq:
      return agx_icmp_to(b, dst, s0, s1, AGX_ICOND_UEQ, false);
   case nir_op_ine:
      return agx_icmp_to(b, dst, s0, s1, AGX_ICOND_UEQ, true);
   case nir_op_ilt:
      return agx_icmp_to(b, dst, s0, s1, AGX_ICOND_SLT, false);
   case nir_op_ige:
      return agx_icmp_to(b, dst, s0, s1, AGX_ICOND_SLT, true);
   case nir_op_ult:
      return agx_icmp_to(b, dst, s0, s1, AGX_ICOND_ULT, false);
   case nir_op_uge:
      return agx_icmp_to(b, dst, s0, s1, AGX_ICOND_ULT, true);

   case nir_op_inot:
      if (sz == 1)
         return agx_xor_to(b, dst, s0, i1);
      else
         return agx_not_to(b, dst, s0);

   case nir_op_b2b1:
      return agx_icmp_to(b, dst, s0, i0, AGX_ICOND_UEQ, true);

   case nir_op_fsqrt:
      return agx_fmul_to(b, dst, s0, agx_srsqrt(b, s0));
   case nir_op_fabs:
      return agx_fmov_to(b, dst, agx_abs(s0));
   case nir_op_fneg:
      return agx_fmov_to(b, dst, agx_neg(s0));

   case nir_op_fmin:
   case nir_op_fmax:
   case nir_op_fmin_agx:
   case nir_op_fmax_agx:
      return agx_fminmax_to(b, dst, s0, s1, instr);

   case nir_op_iabs: {
      assert(sz == 8 || sz == 16 || sz == 32);
      agx_index value = s0;
      if (sz < 32) {
         value = agx_temp(b->shader, AGX_SIZE_32);
         if (sz == 8) {
            agx_index shifted = agx_bfi(b, i0, s0, agx_immediate(8), 0);
            agx_asr_to(b, value, shifted, agx_immediate(8));
         } else {
            agx_signext_to(b, value, s0);
         }
      }
      /* Keep G17P's 32-bit absolute value independent of signed isel.  The
       * usual two's-complement identity is exact even for INT_MIN. Narrow
       * inputs are sign-extended first and truncated after the arithmetic. */
      agx_index sign = agx_asr(b, value, agx_immediate(31));
      agx_index magnitude = agx_xor(b, value, sign);
      if (sz < 32) {
         magnitude = agx_iadd(b, magnitude, agx_neg(sign), 0);
         return agx_subdivide_to(b, dst, magnitude, 0);
      }
      return agx_iadd_to(b, dst, magnitude, agx_neg(sign), 0);
   }

   case nir_op_imin:
      return agx_icmpsel_to(b, dst, s0, s1, s0, s1, AGX_ICOND_SLT);
   case nir_op_imax:
      return agx_icmpsel_to(b, dst, s0, s1, s0, s1, AGX_ICOND_SGT);
   case nir_op_umin:
      return agx_icmpsel_to(b, dst, s0, s1, s0, s1, AGX_ICOND_ULT);
   case nir_op_umax:
      return agx_icmpsel_to(b, dst, s0, s1, s0, s1, AGX_ICOND_UGT);
   case nir_op_bounds_agx:
      /* end offset > bound ? 0 : data */
      return agx_icmpsel_to(b, dst, s1, s2, agx_zero(), s0, AGX_ICOND_UGT);

   case nir_op_iadd:
      return agx_iadd_to(b, dst, s0, s1, 0);
   case nir_op_imadshl_agx:
      return agx_emit_imadshl_agx(b, instr, dst, s0, s1, s2, s3);
   case nir_op_imsubshl_agx:
      return agx_emit_imadshl_agx(b, instr, dst, s0, s1, agx_neg(s2), s3);
   case nir_op_isub:
      return agx_iadd_to(b, dst, s0, agx_neg(s1), 0);
   case nir_op_ineg:
      return agx_iadd_to(b, dst, i0, agx_neg(s0), 0);
   case nir_op_imul:
      return agx_imad_to(b, dst, s0, s1, i0, 0);
   case nir_op_umul_2x32_64:
      return agx_imad_to(b, dst, agx_abs(s0), agx_abs(s1), i0, 0);
   case nir_op_imul_2x32_64:
      return agx_imad_to(b, dst, s0, s1, i0, 0);
   case nir_op_umul_high:
      return agx_mul_high_to(b, dst, s0, s1, false);
   case nir_op_imul_high:
      return agx_mul_high_to(b, dst, s0, s1, true);

   case nir_op_ishl:
      return agx_bfi_to(b, dst, i0, s0, s1, 0);
   case nir_op_ushr:
      return agx_ushr_to(b, dst, s0, s1);
   case nir_op_ishr:
      return agx_asr_to(b, dst, s0, s1);

   case nir_op_extr_agx:
      return agx_extr_to(b, dst, s0, s1, s2,
                         nir_alu_src_as_uint(instr->src[3]));

   case nir_op_ubitfield_extract: {
      unsigned m = nir_alu_src_as_uint(instr->src[2]);
      assert(m != 0 && "should've been optimized");

      /* Disable masking if the whole thing is used */
      if (m >= 32)
         m = 0;

      return agx_bfeil_to(b, dst, i0, s0, s1, m);
   }

   case nir_op_bcsel:
      return agx_icmpsel_to(b, dst, s0, i0, s2, s1, AGX_ICOND_UEQ);

   case nir_op_i2i32: {
      if (src_sz == 8) {
         /* Sign extend in software, NIR likes 8-bit conversions */
         agx_index ishl16 = agx_bfi(b, i0, s0, agx_immediate(8), 0);
         return agx_asr_to(b, dst, ishl16, agx_immediate(8));
      } else {
         assert(s0.size == AGX_SIZE_16 && "other conversions lowered");
         return agx_signext_to(b, dst, s0);
      }
   }

   case nir_op_i2i16: {
      if (src_sz == 8) {
         /* Sign extend in software, NIR likes 8-bit conversions */
         agx_index ishl16 = agx_bfi(b, i0, s0, agx_immediate(8), 0);
         return agx_asr_to(b, dst, ishl16, agx_immediate(8));
      } else {
         assert(s0.size == AGX_SIZE_32 && "other conversions lowered");
         return agx_subdivide_to(b, dst, s0, 0);
      }
   }

   case nir_op_u2u16: {
      if (s0.size == AGX_SIZE_32)
         return agx_subdivide_to(b, dst, s0, 0);
      else
         return agx_mov_to(b, dst, s0);
   }

   /* It will be put into a 16-bit register, but zero out the garbage. We could
    * optimize this in the future but it ensures correctness for u2u16(u2u8(x))
    * sequences.
    */
   case nir_op_u2u8:
   case nir_op_i2i8:
      return agx_and_to(b, dst, s0, agx_immediate(0xFF));

   case nir_op_iadd_sat: {
      agx_instr *I = agx_iadd_to(b, dst, s0, s1, 0);
      I->saturate = true;
      return I;
   }

   case nir_op_isub_sat: {
      agx_instr *I = agx_iadd_to(b, dst, s0, agx_neg(s1), 0);
      I->saturate = true;
      return I;
   }

   case nir_op_uadd_sat: {
      agx_instr *I = agx_iadd_to(b, dst, agx_abs(s0), agx_abs(s1), 0);
      I->saturate = true;
      return I;
   }

   case nir_op_usub_sat: {
      agx_instr *I = agx_iadd_to(b, dst, agx_abs(s0), agx_neg(agx_abs(s1)), 0);
      I->saturate = true;
      return I;
   }

   case nir_op_fsin_agx: {
      if (b->shader->key->dev.encoding == AGX_ISA_ENCODING_AGX3B) {
         /* G17P has no sin_pt_1 quadrant fixup. Use the standard Mesa
          * minimax sine fallback on the already-normalized quadrant input
          * instead. This is the same polynomial used by lower_sincos, with
          * radians / (2 pi) reduced to quadrants / 4:
          *
          *   t = 2 fract(q / 4 + 1/2) - 1
          *   x = 4 (t - t |t|)
          *   sin(q pi/2) ~= x + .225 (x |x| - x)
          *
          * It uses only the grounded G17P FMA and floor paths and avoids
          * assigning AGX2 semantics to G17P's unqualified f32sinc funop.
          */
         agx_index u = agx_ffma(b, s0, agx_immediate_f(0.25f),
                                agx_immediate_f(0.5f));
         agx_index fract = agx_ffma(b, agx_neg(agx_floor(b, u)),
                                    agx_immediate_f(1.0f), u);
         agx_index t = agx_ffma(b, fract, agx_immediate_f(2.0f),
                                agx_immediate_f(-1.0f));
         agx_index x = agx_ffma(b, agx_neg(t), agx_abs(t), t);
         x = agx_fmul(b, x, agx_immediate_f(4.0f));
         agx_index correction = agx_ffma(b, x, agx_abs(x), agx_neg(x));
         agx_index slope = agx_mov_imm(b, 32, UINT32_C(0x3e666666));
         return agx_ffma_to(b, dst, correction, slope, x);
      }

      agx_index fixup = agx_sin_pt_1(b, s0);
      agx_index sinc = agx_sin_pt_2(b, fixup);
      return agx_fmul_to(b, dst, sinc, fixup);
   }

   case nir_op_f2i16:
      return agx_convert_to(b, dst, agx_immediate(AGX_CONVERT_F_TO_S16), s0,
                            AGX_ROUND_RTZ);

   case nir_op_f2i32:
      return agx_convert_to(b, dst, agx_immediate(AGX_CONVERT_F_TO_S32), s0,
                            AGX_ROUND_RTZ);

   case nir_op_f2u16:
      return agx_convert_to(b, dst, agx_immediate(AGX_CONVERT_F_TO_U16), s0,
                            AGX_ROUND_RTZ);

   case nir_op_f2u32:
      return agx_convert_to(b, dst, agx_immediate(AGX_CONVERT_F_TO_U32), s0,
                            AGX_ROUND_RTZ);

   case nir_op_u2f16:
   case nir_op_u2f32: {
      if (src_sz == 64)
         UNREACHABLE("64-bit conversions unimplemented");

      enum agx_convert mode = (src_sz == 32)   ? AGX_CONVERT_U32_TO_F
                              : (src_sz == 16) ? AGX_CONVERT_U16_TO_F
                                               : AGX_CONVERT_U8_TO_F;

      return agx_convert_to(b, dst, agx_immediate(mode), s0, AGX_ROUND_RTE);
   }

   case nir_op_i2f16:
   case nir_op_i2f32: {
      if (src_sz == 64)
         UNREACHABLE("64-bit conversions unimplemented");

      enum agx_convert mode = (src_sz == 32)   ? AGX_CONVERT_S32_TO_F
                              : (src_sz == 16) ? AGX_CONVERT_S16_TO_F
                                               : AGX_CONVERT_S8_TO_F;

      return agx_convert_to(b, dst, agx_immediate(mode), s0, AGX_ROUND_RTE);
   }

   case nir_op_pack_32_2x16_split:
   case nir_op_pack_64_2x32_split: {
      agx_index idx[] = {s0, s1};
      return agx_emit_collect_to(b, dst, 2, idx);
   }

   case nir_op_unpack_64_2x32_split_x:
   case nir_op_unpack_32_2x16_split_x:
      return agx_subdivide_to(b, dst, s0, 0);

   case nir_op_unpack_64_2x32_split_y:
   case nir_op_unpack_32_2x16_split_y:
      return agx_subdivide_to(b, dst, s0, 1);

   case nir_op_vec2:
   case nir_op_vec3:
   case nir_op_vec4: {
      agx_index idx[] = {s0, s1, s2, s3};
      return agx_emit_collect_to(b, dst, srcs, idx);
   }

   case nir_op_vec8:
   case nir_op_vec16:
      UNREACHABLE("should've been lowered");

   default:
      fprintf(stderr, "Unhandled ALU op %s\n", nir_op_infos[instr->op].name);
      UNREACHABLE("Unhandled ALU instruction");
   }
}

static enum agx_lod_mode
agx_lod_mode_for_nir(nir_texop op, bool biased, bool min_lod, bool lod_is_zero)
{
   switch (op) {
   case nir_texop_tex:
      /* Zero-bias MinLod operations can retain implicit sampling after NIR
       * folds the bias, including dynamically loaded image-view minima. */
      return min_lod ? AGX_LOD_MODE_AUTO_LOD_BIAS_MIN : AGX_LOD_MODE_AUTO_LOD;
   case nir_texop_tg4:
      assert(!min_lod);
      return AGX_LOD_MODE_AUTO_LOD;
   case nir_texop_txb:
      return min_lod ? AGX_LOD_MODE_AUTO_LOD_BIAS_MIN
                     : AGX_LOD_MODE_AUTO_LOD_BIAS;
   case nir_texop_lod:
      assert(!min_lod);
      return biased ? AGX_LOD_MODE_AUTO_LOD_BIAS : AGX_LOD_MODE_AUTO_LOD;
   case nir_texop_txd:
      return min_lod ? AGX_LOD_MODE_LOD_GRAD_MIN : AGX_LOD_MODE_LOD_GRAD;
   case nir_texop_txl:
   case nir_texop_txf:
      assert(!min_lod);
      return lod_is_zero ? AGX_LOD_MODE_AUTO_LOD : AGX_LOD_MODE_LOD_MIN;
   case nir_texop_txf_ms:
      assert(!min_lod);
      assert(lod_is_zero && "no mipmapping");
      return AGX_LOD_MODE_AUTO_LOD;
   default:
      UNREACHABLE("Unhandled texture op");
   }
}

static enum agx_gather
agx_gather_for_nir(nir_tex_instr *tex)
{
   if (tex->op == nir_texop_tg4) {
      enum agx_gather components[] = {
         AGX_GATHER_R,
         AGX_GATHER_G,
         AGX_GATHER_B,
         AGX_GATHER_A,
      };

      assert(tex->component < ARRAY_SIZE(components));
      return components[tex->component];
   } else {
      return AGX_GATHER_NONE;
   }
}

static void
agx_emit_tex(agx_builder *b, nir_tex_instr *instr)
{
   agx_index coords = agx_null(), bindless = agx_immediate(0),
             texture = agx_immediate(instr->texture_index),
             sampler = agx_immediate(instr->sampler_index),
             lod = agx_immediate(0), compare = agx_null(),
             packed_offset = agx_null();

   /* Default to the txf sampler at ss0 */
   if (!nir_tex_instr_need_sampler(instr)) {
      sampler = agx_immediate(0);
   }

   bool lod_is_zero = true;

   for (unsigned i = 0; i < instr->num_srcs; ++i) {
      agx_index index = agx_src_index(&instr->src[i].src);

      switch (instr->src[i].src_type) {
      case nir_tex_src_backend1:
         coords = index;
         break;

      case nir_tex_src_backend2:
         packed_offset = index;
         if (b->shader->key->dev.encoding == AGX_ISA_ENCODING_AGX3B &&
             !instr->is_shadow &&
             (instr->sampler_dim == GLSL_SAMPLER_DIM_2D ||
              instr->sampler_dim == GLSL_SAMPLER_DIM_3D) &&
             nir_src_is_const(instr->src[i].src) &&
             nir_src_as_uint(instr->src[i].src) <=
                (instr->sampler_dim == GLSL_SAMPLER_DIM_3D ? 0xfff : 0xff)) {
            packed_offset = agx_immediate(nir_src_as_uint(instr->src[i].src));
         }
         break;

      case nir_tex_src_lod:
      case nir_tex_src_bias:
         lod = index;
         lod_is_zero = nir_src_is_const(instr->src[i].src) &&
                       nir_src_as_uint(instr->src[i].src) == 0;
         if (b->shader->key->dev.encoding == AGX_ISA_ENCODING_AGX3B &&
             instr->op == nir_texop_txl &&
             instr->src[i].src_type == nir_tex_src_lod &&
             nir_src_is_const(instr->src[i].src)) {
            const float level = nir_src_as_float(instr->src[i].src);
            /* An immediate LOD leaves the auxiliary staging operand free
             * for an array layer. Shadow immediates are qualified only at
             * zero; keep nonzero comparisons in the witnessed register form. */
            if (level >= 0.0f && level * 64.0f <= 0xfff &&
                (!instr->is_shadow || level == 0.0f) &&
                level * 64.0f == (float)(uint16_t)(level * 64.0f) &&
                agx_minifloat_exact(level))
               lod = agx_immediate_f(level);
         }
         break;

      case nir_tex_src_lod_bias_min_agx:
         assert(index.size == AGX_SIZE_32);
         lod = index;
         break;

      case nir_tex_src_comparator:
         assert(index.size == AGX_SIZE_32);
         compare = index;
         break;

      case nir_tex_src_texture_offset:
         texture = index;
         break;
      case nir_tex_src_sampler_handle:
         nir_intrinsic_instr *intr = nir_src_as_intrinsic(instr->src[i].src);
         if (intr && intr->intrinsic == nir_intrinsic_bindless_sampler_agx) {
            b->shader->out->uses_sampler_heap = true;
            /* Heap offset */
            sampler = agx_src_index(&intr->src[1]);
         } else {
            sampler = index;
         }
         break;

      case nir_tex_src_texture_handle:
         texture =
            agx_translate_bindless_handle(b, &instr->src[i].src, &bindless);
         break;

      case nir_tex_src_min_lod:
         assert(instr->op == nir_texop_txd && "other cases lowered");
         break;

      case nir_tex_src_ddx: {
         int y_idx = nir_tex_instr_src_index(instr, nir_tex_src_ddy);
         assert(y_idx >= 0 && "we only handle gradients");

         int min_idx = nir_tex_instr_src_index(instr, nir_tex_src_min_lod);
         bool has_min = min_idx >= 0;
         agx_index min;

         unsigned n = nir_tex_instr_src_size(instr, y_idx);
         assert((n == 2 || n == 3) && "other sizes not supported");

         agx_index index2 = agx_src_index(&instr->src[y_idx].src);

         if (has_min) {
            min = agx_src_index(&instr->src[min_idx].src);

            /* Undef extend to 32-bit since our IR is iffy */
            min = agx_pad_to_32(b, min);
         }

         /* We explicitly don't cache about the split cache for this */
         unsigned chans = (2 * n) + (has_min ? 1 : 0);
         lod = agx_vec_temp(b->shader, AGX_SIZE_32, chans);
         agx_instr *I = agx_collect_to(b, lod, chans);

         for (unsigned i = 0; i < n; ++i) {
            I->src[(2 * i) + 0] = agx_emit_extract(b, index, i);
            I->src[(2 * i) + 1] = agx_emit_extract(b, index2, i);
         }

         if (has_min)
            I->src[2 * n] = min;

         break;
      }

      case nir_tex_src_ddy:
         /* handled above */
         break;

      default:
         UNREACHABLE("Unexpected texture source");
      }
   }

   bool has_min_lod =
      nir_tex_instr_src_index(instr, nir_tex_src_lod_bias_min_agx) >= 0 ||
      nir_tex_instr_src_index(instr, nir_tex_src_min_lod) >= 0;

   bool has_bias = (has_min_lod && instr->op != nir_texop_txd) ||
                   (nir_tex_instr_src_index(instr, nir_tex_src_bias) >= 0);

   enum agx_lod_mode lod_mode =
      agx_lod_mode_for_nir(instr->op, has_bias, has_min_lod, lod_is_zero);

   /* AGX2's AUTO_LOD encoding also covers txl(..., 0). G17P's AUTO form is
    * specifically the fragment derivative form, while Apple's kernel
    * witnesses use the explicit immediate-level encoding. Keep txl at level
    * zero explicit in every stage; fragment txl must not acquire derivatives. */
   if (b->shader->key->dev.encoding == AGX_ISA_ENCODING_AGX3B &&
       instr->op == nir_texop_txl && lod_is_zero)
      lod_mode = AGX_LOD_MODE_LOD_MIN;

   /* Register offsets use the native bias/LOD auxiliary word. Preserve
    * implicit derivatives with bias zero at instruction selection, after
    * NIR constant folding (which correctly turns txb(..., 0) into tex). */
   if (b->shader->key->dev.encoding == AGX_ISA_ENCODING_AGX3B &&
       instr->op == nir_texop_tex && !has_min_lod && !agx_is_null(packed_offset) &&
       packed_offset.type != AGX_INDEX_IMMEDIATE) {
      lod_mode = AGX_LOD_MODE_AUTO_LOD_BIAS;
      lod = agx_mov_imm(b, 16, 0);
   }

   if (lod_mode == AGX_LOD_MODE_AUTO_LOD) {
      /* Ignored logically but asserted 0 */
      lod = agx_immediate(0);
   }

   if (b->shader->key->dev.encoding == AGX_ISA_ENCODING_AGX3B &&
       !agx_is_null(packed_offset) && packed_offset.type != AGX_INDEX_IMMEDIATE &&
       (lod_mode == AGX_LOD_MODE_LOD_MIN ||
        (lod_mode == AGX_LOD_MODE_AUTO_LOD_BIAS ||
         lod_mode == AGX_LOD_MODE_AUTO_LOD_BIAS_MIN))) {
      /* Native G16G PACK_SMPIR inserts the F16 LOD as signed fixed-point
       * bits 16..27, preserving the three offset nibbles in the base word. */
      agx_index level = lod;
      if (lod_mode == AGX_LOD_MODE_AUTO_LOD_BIAS_MIN) {
         level = agx_temp(b->shader, AGX_SIZE_16);
         agx_subdivide_to(b, level, lod, 0);
      }
      if (level.type == AGX_INDEX_IMMEDIATE)
         level = agx_mov_imm(b, 16, _mesa_float_to_half(agx_minifloat_decode(level.value)));
      if (instr->is_array) {
         /* The auxiliary word holds the layer in bits 0..11, offsets in
          * 12..15/28..31 and LOD in 16..27. Keep all temporaries in SSA. */
         agx_index layer_word = agx_emit_extract(b, coords, 2);
         agx_index layer_half = agx_temp(b->shader, AGX_SIZE_16);
         agx_subdivide_to(b, layer_half, layer_word, 0);
         agx_index layer = agx_temp(b->shader, AGX_SIZE_32);
         agx_pack_layer_agx3_to(b, layer, layer_half);
         packed_offset = agx_or(b, packed_offset, layer);
      }
      agx_index auxiliary = agx_temp(b->shader, AGX_SIZE_32);
      agx_pack_lod_offset_agx3_to(b, auxiliary, level, packed_offset);
      packed_offset = auxiliary;
   }

   if (b->shader->key->dev.encoding == AGX_ISA_ENCODING_AGX3B &&
       lod_mode == AGX_LOD_MODE_LOD_GRAD_MIN && instr->is_array &&
       !agx_is_null(packed_offset) && packed_offset.type != AGX_INDEX_IMMEDIATE) {
      agx_index layer_half = agx_temp(b->shader, AGX_SIZE_16);
      agx_subdivide_to(b, layer_half, agx_emit_extract(b, coords, 2), 0);
      agx_index layer = agx_temp(b->shader, AGX_SIZE_32);
      agx_pack_layer_agx3_to(b, layer, layer_half);
      packed_offset = agx_or(b, packed_offset, layer);
   }

   /* Native array depth+bias and 3D offset+bias consume the minimum as a
    * packed scalar GPR because their staging operand bank is already full. */
   if (b->shader->key->dev.encoding == AGX_ISA_ENCODING_AGX3B &&
       lod_mode == AGX_LOD_MODE_AUTO_LOD_BIAS_MIN &&
       ((instr->is_shadow && instr->is_array) ||
        (instr->sampler_dim == GLSL_SAMPLER_DIM_3D &&
         !agx_is_null(packed_offset) && packed_offset.type != AGX_INDEX_IMMEDIATE))) {
      agx_index minimum = agx_temp(b->shader, AGX_SIZE_16);
      agx_subdivide_to(b, minimum, lod, 1);
      /* PACK_SMPIR and SMP consume a full word, even though the packed
       * minimum occupies only its low bits. A half-sized allocation can
       * overlap the next live value and corrupt repeated integer samples. */
      bindless = agx_temp(b->shader, AGX_SIZE_32);
      agx_pack_min_lod_agx3_to(b, bindless, minimum);
   }

   agx_index dst = agx_def_index(&instr->def);

   /* Pack shadow reference value (compare) and packed offset together */
   agx_index compare_offset = agx_null();

   if (!agx_is_null(compare) && !agx_is_null(packed_offset))
      compare_offset = agx_vec2(b, compare, packed_offset);
   else if (!agx_is_null(packed_offset))
      compare_offset = packed_offset;
   else if (!agx_is_null(compare))
      compare_offset = compare;

   if (b->shader->key->dev.encoding == AGX_ISA_ENCODING_AGX3B &&
       instr->op == nir_texop_txf_ms) {
      /* The common lowering carries sample/layer as two 16-bit halves.
       * G16 uses PACK_SMPIR's native sample/array auxiliary bank instead. */
      agx_index xy_aux[3] = {agx_null(), agx_null(), agx_null()};
      agx_emit_split(b, xy_aux, coords, 3);
      agx_index sample = agx_temp(b->shader, AGX_SIZE_16);
      agx_subdivide_to(b, sample, xy_aux[2], 0);
      agx_index layer = agx_null();
      if (instr->is_array) {
         layer = agx_temp(b->shader, AGX_SIZE_16);
         agx_subdivide_to(b, layer, xy_aux[2], 1);
      }
      xy_aux[2] = agx_pack_ms_aux(b, sample, layer);
      coords = agx_emit_collect(b, 3, xy_aux);
   }

   if (b->shader->key->dev.encoding == AGX_ISA_ENCODING_AGX3B &&
       instr->sampler_dim == GLSL_SAMPLER_DIM_CUBE && instr->is_array &&
       (instr->op == nir_texop_txl || instr->op == nir_texop_txb || has_min_lod) &&
       lod.type != AGX_INDEX_IMMEDIATE) {
      /* Native three-source PACK_SMPIR consumes an already packed layer.
       * Keep this temporary visible to allocation instead of stealing a GPR. */
      agx_index xy_face_layer[4];
      agx_emit_split(b, xy_face_layer, coords, 4);
      agx_index layer = agx_temp(b->shader, AGX_SIZE_16);
      agx_subdivide_to(b, layer, xy_face_layer[3], 0);
      xy_face_layer[3] = agx_temp(b->shader, AGX_SIZE_32);
      agx_pack_layer_agx3_to(b, xy_face_layer[3], layer);
      coords = agx_emit_collect(b, 4, xy_face_layer);
   }

   agx_index tmp = agx_vec_temp(b->shader, dst.size, instr->is_sparse ? 8 : 4);
   agx_instr *I = agx_texture_sample_to(
      b, tmp, coords, lod, bindless, texture, sampler, compare_offset,
      agx_tex_dim(instr->sampler_dim, instr->is_array), lod_mode, 0,
      !agx_is_null(packed_offset), !agx_is_null(compare),
      instr->op == nir_texop_lod, agx_gather_for_nir(instr), instr->is_sparse);

   if (instr->op == nir_texop_txf || instr->op == nir_texop_txf_ms) {
      I->op = AGX_OPCODE_TEXTURE_LOAD;
      b->shader->out->uses_txf = true;
      b->shader->out->sampler_state_count =
         MAX2(b->shader->out->sampler_state_count, 1);
   }

   /* Destination masking doesn't seem to work properly for gathers (because
    * it's mostly pointless), but it does show up in the lowering of
    * textureGatherOffsets. Don't try to mask the destination for gathers.
    *
    * TODO: Check if it works with sparse.
    */
   /* Native LOD queries publish both words even when only one is used. */
   bool masked = instr->op != nir_texop_tg4 && instr->op != nir_texop_lod &&
                 !instr->is_sparse;
   I->mask = agx_expand_tex_to(b, &instr->def, tmp, masked);
   /* The residency result is not a colour component. For sparse depth
    * comparisons it is the second word; only the first is in the SMP mask. */
   if (instr->is_sparse)
      I->mask &= BITFIELD_MASK(instr->def.num_components - 1);
}

static void
agx_emit_jump(agx_builder *b, nir_jump_instr *instr)
{
   if (instr->type == nir_jump_halt) {
      agx_stop(b);
   } else {
      assert(instr->type == nir_jump_break);
      agx_block_add_successor(b->shader->current_block, b->shader->break_block);
      agx_break(b, b->shader->loop_nesting + 1, b->shader->break_block);
   }

   b->shader->current_block->unconditional_jumps = true;
}

static void
agx_emit_phi(agx_builder *b, nir_phi_instr *instr)
{
   agx_instr *I =
      agx_phi_to(b, agx_def_index(&instr->def), exec_list_length(&instr->srcs));

   /* Deferred */
   I->phi = instr;
}

/* Look up the AGX block corresponding to a given NIR block. Used when
 * translating phi nodes after emitting all blocks.
 */
static agx_block *
agx_from_nir_block(agx_context *ctx, nir_block *block)
{
   return ctx->indexed_nir_blocks[block->index];
}

static void
agx_emit_phi_deferred(agx_context *ctx, agx_block *block, agx_instr *I)
{
   nir_phi_instr *phi = I->phi;
   I->phi = NULL;

   /* Guaranteed by lower_phis_to_scalar */
   assert(phi->def.num_components == 1);

   nir_foreach_phi_src(src, phi) {
      agx_block *pred = agx_from_nir_block(ctx, src->pred);
      unsigned i = agx_predecessor_index(block, pred);
      assert(i < I->nr_srcs);

      I->src[i] = agx_src_index(&src->src);
   }
}

static void
agx_emit_phis_deferred(agx_context *ctx)
{
   agx_foreach_block(ctx, block) {
      agx_foreach_phi_in_block(block, I)
         agx_emit_phi_deferred(ctx, block, I);
   }
}

static void
agx_emit_undef(agx_builder *b, nir_undef_instr *instr)
{
   /* For now, just lower undefs to zero. This doesn't matter too much, since
    * the lowering happens in NIR and this just allows for late lowering passes
    * to result in undefs.
    */
   if (instr->def.num_components > 1) {
      assert(instr->def.num_components <= 4);
      agx_index zero = agx_mov_imm(b, instr->def.bit_size, 0);

      agx_emit_collect_to(b, agx_def_index(&instr->def),
                          instr->def.num_components,
                          (agx_index[4]){zero, zero, zero, zero});
   } else {
      agx_mov_imm_to(b, agx_def_index(&instr->def), 0);
   }
}

static void
agx_emit_instr(agx_builder *b, struct nir_instr *instr)
{
   switch (instr->type) {
   case nir_instr_type_load_const:
      agx_emit_load_const(b, nir_instr_as_load_const(instr));
      break;

   case nir_instr_type_intrinsic:
      agx_emit_intrinsic(b, nir_instr_as_intrinsic(instr));
      break;

   case nir_instr_type_alu:
      agx_emit_alu(b, nir_instr_as_alu(instr));
      break;

   case nir_instr_type_tex:
      agx_emit_tex(b, nir_instr_as_tex(instr));
      break;

   case nir_instr_type_jump:
      agx_emit_jump(b, nir_instr_as_jump(instr));
      break;

   case nir_instr_type_phi:
      agx_emit_phi(b, nir_instr_as_phi(instr));
      break;

   case nir_instr_type_undef:
      agx_emit_undef(b, nir_instr_as_undef(instr));
      break;

   default:
      UNREACHABLE("should've been lowered");
   }
}

static agx_block *
agx_create_block(agx_context *ctx)
{
   agx_block *blk = rzalloc(ctx, agx_block);

   /* This is conservative, TODO: divergence analysis */
   blk->divergent = ctx->total_nesting > 0;

   util_dynarray_init(&blk->predecessors, blk);

   return blk;
}

static agx_block *
emit_block(agx_context *ctx, nir_block *block)
{
   if (ctx->after_block) {
      ctx->current_block = ctx->after_block;
      ctx->after_block = NULL;
   } else {
      ctx->current_block = agx_create_block(ctx);
   }

   agx_block *blk = ctx->current_block;
   if (ctx->key->dev.encoding == AGX_ISA_ENCODING_AGX3B &&
       (ctx->stage == MESA_SHADER_COMPUTE || ctx->stage == MESA_SHADER_FRAGMENT))
      blk->divergent = block->divergent;
   list_addtail(&blk->link, &ctx->blocks);
   list_inithead(&blk->instructions);

   ctx->indexed_nir_blocks[block->index] = blk;

   agx_builder _b = agx_init_builder(ctx, agx_after_block(blk));

   nir_foreach_instr(instr, block) {
      agx_emit_instr(&_b, instr);
   }

   return blk;
}

static agx_block *emit_cf_list(agx_context *ctx, struct exec_list *list);

static void
emit_if(agx_context *ctx, nir_if *nif)
{
   agx_block *first_block = ctx->current_block;
   agx_builder _b = agx_init_builder(ctx, agx_after_block(first_block));
   agx_index cond = agx_src_index(&nif->condition);

   agx_instr *if_ = agx_if_icmp(&_b, cond, agx_zero(), 1, AGX_ICOND_UEQ, true,
                                NULL /* filled in later */);
   ctx->loop_nesting++;
   ctx->total_nesting++;

   /* Emit the two subblocks. */
   agx_block *if_block = emit_cf_list(ctx, &nif->then_list);
   agx_block *end_then = ctx->current_block;

   _b.cursor = agx_after_block(ctx->current_block);

   agx_block *else_block = emit_cf_list(ctx, &nif->else_list);
   agx_block *end_else = ctx->current_block;

   /* If the "if" fails, we fallthrough to the else */
   if_->target = else_block;

   /* Insert an else instruction at the beginning of the else block. We use
    * "else_fcmp 0.0, 0.0, eq" as unconditional else, matching the blob.
    *
    * If it fails, we fall through to the logical end of the last else block.
    */
   _b.cursor = agx_before_block(else_block);
   agx_else_fcmp(&_b, agx_zero(), agx_zero(), 1, AGX_FCOND_EQ, false, end_else);

   ctx->after_block = agx_create_block(ctx);

   agx_block_add_successor(first_block, if_block);
   agx_block_add_successor(first_block, else_block);
   agx_block_add_successor(end_then, ctx->after_block);
   agx_block_add_successor(end_else, ctx->after_block);

   _b.cursor = agx_after_block(ctx->current_block);
   agx_pop_exec(&_b, 1);
   ctx->loop_nesting--;
   ctx->total_nesting--;
}

/* Contended global atomics can keep a retrying workgroup runnable while the
 * lock owner needs execution time. Native YIELD supplies scheduler backoff
 * without bounding the application's retry loop or changing its memory work.
 * Satisfactory's luminance reduction is a small reproducible instance.
 */
static bool
loop_has_global_atomic(nir_loop *loop)
{
   nir_foreach_block_in_cf_node(block, &loop->cf_node) {
      nir_foreach_instr(instr, block) {
         if (instr->type != nir_instr_type_intrinsic)
            continue;
         switch (nir_instr_as_intrinsic(instr)->intrinsic) {
         case nir_intrinsic_global_atomic:
         case nir_intrinsic_global_atomic_swap:
         case nir_intrinsic_global_atomic_agx:
         case nir_intrinsic_global_atomic_swap_agx:
            return true;
         default:
            break;
         }
      }
   }
   return false;
}

static void
emit_loop(agx_context *ctx, nir_loop *nloop)
{
   assert(!nir_loop_has_continue_construct(nloop));
   /* We only track nesting within the innermost loop, so push and reset */
   unsigned pushed_nesting = ctx->loop_nesting;
   ctx->loop_nesting = 0;
   ctx->total_nesting++;

   agx_block *popped_break = ctx->break_block;

   ctx->break_block = agx_create_block(ctx);
   agx_block *loop_header = agx_create_block(ctx);

   /* If we are emitting a loop inside other control flow, there might be
    * threads masked off (TODO: divergence analysis), so push_exec them so
    * we get the lower nesting count values to ourselves.
    */
   agx_builder _b = agx_init_builder(ctx, agx_after_block(ctx->current_block));
   if (ctx->total_nesting > 1)
      agx_push_exec(&_b, 1);

   /* Fallthrough to loop header */
   agx_block_add_successor(ctx->current_block, loop_header);

   /* Emit the loop */
   ctx->after_block = loop_header;
   loop_header->loop_header = true;
   agx_block *start_block = emit_cf_list(ctx, &nloop->body);

   /* The backoff has ordinary SSA operands, so it must precede the block's
    * logical terminators, including a break at the end of the loop body. */
   _b.cursor = agx_after_block_logical(ctx->current_block);
   if (ctx->key->dev.encoding == AGX_ISA_ENCODING_AGX3B &&
       ctx->stage == MESA_SHADER_COMPUTE && loop_has_global_atomic(nloop))
      agx_yield_agx3(&_b, agx_mov_imm(&_b, 16, 8));
   _b.cursor = agx_after_block(ctx->current_block);
   agx_jmp_exec_any(&_b, start_block);
   agx_pop_exec(&_b, 1);
   agx_block_add_successor(ctx->current_block, loop_header);

   /* Pop off */
   ctx->after_block = ctx->break_block;
   ctx->break_block = popped_break;

   /* Update shader-db stats */
   ++ctx->loop_count;

   /* All nested control flow must have finished */
   assert(ctx->loop_nesting == 0);

   /* Restore loop nesting (we might be inside an if inside an outer loop) */
   ctx->loop_nesting = pushed_nesting;
   ctx->total_nesting--;
}

/* Before the first control flow structure, the nesting counter needs to be
 * zeroed for correct operation. This only happens at most once, since by
 * definition this occurs at the end of the first block, which dominates the
 * rest of the program. */

static void
emit_first_cf(agx_context *ctx)
{
   if (ctx->any_cf)
      return;

   agx_builder _b = agx_init_builder(ctx, agx_after_block(ctx->current_block));
   agx_begin_cf(&_b);
   ctx->any_cf = true;
}

static agx_block *
emit_cf_list(agx_context *ctx, struct exec_list *list)
{
   agx_block *start_block = NULL;

   foreach_list_typed(nir_cf_node, node, node, list) {
      switch (node->type) {
      case nir_cf_node_block: {
         agx_block *block = emit_block(ctx, nir_cf_node_as_block(node));

         if (!start_block)
            start_block = block;

         break;
      }

      case nir_cf_node_if:
         emit_first_cf(ctx);
         emit_if(ctx, nir_cf_node_as_if(node));
         break;

      case nir_cf_node_loop:
         emit_first_cf(ctx);
         emit_loop(ctx, nir_cf_node_as_loop(node));
         break;

      default:
         UNREACHABLE("Unknown control flow");
      }
   }

   return start_block;
}

static void
agx_set_st_vary_final(agx_context *ctx)
{
   agx_foreach_instr_global_rev(ctx, I) {
      if (I->op == AGX_OPCODE_ST_VARY) {
         I->last = true;
         return;
      }
   }

   /* If we got here, there was no varying written. We need to mark that. */
   agx_block *last_block = list_last_entry(&ctx->blocks, agx_block, link);
   agx_builder _b = agx_init_builder(ctx, agx_after_block_logical(last_block));
   agx_no_varyings(&_b);
}

static void
agx_calc_stats(agx_context *ctx, unsigned size, struct agx2_stats *stats)
{
   struct agx_cycle_estimate cycles = agx_estimate_cycles(ctx);
   uint32_t old_preamble_inst = stats->preamble_inst;

   *stats = (struct agx2_stats){
      .alu = cycles.alu,
      .fscib = cycles.f_scib,
      .ic = cycles.ic,
      .code_size = size,
      .gprs = ctx->max_reg,
      .uniforms = ctx->out->push_count,
      .scratch = ctx->scratch_size_B,
      .threads = agx_occupancy_for_register_count(ctx->max_reg).max_threads,
      .loops = ctx->loop_count,
      .preamble_inst = old_preamble_inst,
   };

   /* Count instructions */
   agx_foreach_instr_global(ctx, I) {
      stats->instrs++;

      if (I->op == AGX_OPCODE_STACK_STORE)
         stats->spills++;
      else if (I->op == AGX_OPCODE_STACK_LOAD)
         stats->fills++;
   }
}

static bool
agx_lower_sincos_filter(const nir_instr *instr, UNUSED const void *_)
{
   if (instr->type != nir_instr_type_alu)
      return false;

   nir_alu_instr *alu = nir_instr_as_alu(instr);
   return alu->op == nir_op_fsin || alu->op == nir_op_fcos;
}

/* Sine and cosine are implemented via the sin_pt_1 and sin_pt_2 opcodes for
 * heavy lifting. sin_pt_2 implements sinc in the first quadrant, expressed in
 * turns (sin (tau x) / x), while sin_pt_1 implements a piecewise sign/offset
 * fixup to transform a quadrant angle [0, 4] to [-1, 1]. The NIR opcode
 * fsin_agx models the fixup, sinc, and multiply to obtain sine, so we just
 * need to change units from radians to quadrants modulo turns. Cosine is
 * implemented by shifting by one quadrant: cos(x) = sin(x + tau/4).
 */

static nir_def *
agx_lower_sincos_impl(struct nir_builder *b, nir_instr *instr, void *data)
{
   nir_alu_instr *alu = nir_instr_as_alu(instr);
   nir_def *x = nir_mov_alu(b, alu->src[0], 1);
   if (*(const bool *)data) {
      /* AGX3B has no qualified sin_pt_1 fixup. Reduce and evaluate in f32,
       * including half inputs: rounding the phase in f16 loses significant
       * angle bits before the sine is even evaluated. Fold to [-pi/2, pi/2]
       * and use the odd degree-13 Taylor polynomial on that interval. */
      const unsigned bits = x->bit_size;
      x = nir_f2f32(b, x);
      nir_def *phase = nir_ffma(b, x, nir_imm_float(b, M_1_PI * 0.5),
                               nir_imm_float(b, alu->op == nir_op_fcos ? 0.75 : 0.5));
      nir_def *t = nir_ffma(b, nir_ffract(b, phase), nir_imm_float(b, 2.0),
                           nir_imm_float(b, -1.0));
      nir_def *a = nir_fabs(b, t);
      nir_def *r = nir_fmul_imm(b, nir_fmin(b, a, nir_fsub(b, nir_imm_float(b, 1.0), a)), M_PI);
      nir_def *r2 = nir_fmul(b, r, r);
      nir_def *p = nir_imm_float(b, 1.0 / 6227020800.0);
      const double coefficients[] = {-1.0 / 39916800.0, 1.0 / 362880.0,
                                      -1.0 / 5040.0, 1.0 / 120.0,
                                      -1.0 / 6.0, 1.0};
      for (unsigned i = 0; i < ARRAY_SIZE(coefficients); ++i)
         p = nir_ffma(b, p, r2, nir_imm_float(b, coefficients[i]));
      nir_def *result = nir_fmul(b, r, p);
      /* The polynomial rounds to 1.000002 at r = pi/2, which cos(x) for
       * tiny |x| hits directly (glsl.builtin.precision.cos.* expect at most
       * 1.0). True sine never leaves [-1, 1]; clamp the magnitude. */
      result = nir_fmin(b, result, nir_imm_float(b, 1.0));
      result = nir_bcsel(b, nir_flt_imm(b, t, 0.0), nir_fneg(b, result), result);
      /* The magnitude clamp uses minNum semantics and would turn NaN into
       * one. Preserve the original operation's non-finite-input behavior
       * when its float controls require it (including sin/cos(infinity)). */
      if (alu->fp_math_ctrl & (nir_fp_preserve_inf | nir_fp_preserve_nan)) {
         /* The builder defaults to fast math. This comparison deliberately
          * has an infinite operand, so it must not inherit the assumption
          * that both operands are finite (which can fold it to false).
          */
         const unsigned saved_fp_math_ctrl = b->fp_math_ctrl;
         b->fp_math_ctrl |= nir_fp_preserve_inf | nir_fp_preserve_nan;
         nir_def *finite = nir_flt(b, nir_fabs(b, x), nir_imm_float(b, INFINITY));
         b->fp_math_ctrl = saved_fp_math_ctrl;
         result = nir_bcsel(b, finite, result, nir_imm_float(b, NAN));
      }
      return nir_f2fN(b, result, bits);
   }
   nir_def *turns = nir_fmul_imm(b, x, M_1_PI * 0.5f);

   if (alu->op == nir_op_fcos)
      turns = nir_fadd_imm(b, turns, 0.25f);

   nir_def *quadrants = nir_fmul_imm(b, nir_ffract(b, turns), 4.0);
   return nir_fsin_agx(b, quadrants);
}

static bool
agx_lower_sincos(nir_shader *shader, bool g17p)
{
   return nir_shader_lower_instructions(shader, agx_lower_sincos_filter,
                                        agx_lower_sincos_impl, &g17p);
}

static bool
agx_lower_front_face(struct nir_builder *b, nir_intrinsic_instr *intr,
                     UNUSED void *data)
{
   if (intr->intrinsic != nir_intrinsic_load_front_face)
      return false;

   nir_def *def = &intr->def;
   assert(def->bit_size == 1);

   b->cursor = nir_before_instr(&intr->instr);
   nir_def_rewrite_uses(def, nir_inot(b, nir_load_back_face_agx(b, 1)));
   return true;
}

/*
 * Standard NIR optimization loop. This is run in agx_preprocess_nir, then once
 * again at shader variant compile time. Unless there was a complex shader key,
 * the latter run should be almost a no-op.
 */
/* Large flattened branches eagerly execute both arms for every fragment.
 * Preserve expensive conditional work in the native backend experiment;
 * small selects still avoid branch overhead. This is shader-independent. */
static unsigned
agx_select_limit(const nir_shader *nir, bool native)
{
   return native && nir->info.stage == MESA_SHADER_FRAGMENT &&
          debug_get_bool_option("AGX_NATIVE_PRESERVE_BRANCHES", false) ? 8 : 64;
}

static void
agx_optimize_loop_nir(nir_shader *nir, bool native)
{
   bool progress;

   do {
      progress = false;

      NIR_PASS(progress, nir, nir_opt_copy_prop);
      NIR_PASS(progress, nir, nir_opt_remove_phis);
      NIR_PASS(progress, nir, nir_opt_dce);
      NIR_PASS(progress, nir, nir_opt_dead_cf);
      NIR_PASS(progress, nir, nir_opt_cse);

      nir_opt_peephole_select_options peephole_select_options = {
         .limit = agx_select_limit(nir, native),
         .expensive_alu_ok = true,
      };
      NIR_PASS(progress, nir, nir_opt_peephole_select,
               &peephole_select_options);
      NIR_PASS(progress, nir, nir_opt_phi_precision);
      NIR_PASS(progress, nir, nir_opt_algebraic);
      NIR_PASS(progress, nir, nir_opt_constant_folding);
      NIR_PASS(progress, nir, nir_opt_undef);
      NIR_PASS(progress, nir, nir_opt_loop_unroll);
   } while (progress);
}

static bool
agx_nir_lower_imul64_filter(const nir_instr *instr, const void *lower_full)
{
   if (instr->type != nir_instr_type_alu)
      return false;

   const nir_alu_instr *alu = nir_instr_as_alu(instr);
   return lower_full && alu->op == nir_op_imul &&
          alu->def.bit_size == 64;
}

static nir_def *
agx_nir_lower_imul64_instr(nir_builder *b, nir_instr *instr, void *_)
{
   nir_alu_instr *alu = nir_instr_as_alu(instr);
   assert(alu->op == nir_op_imul && alu->def.bit_size == 64);
   nir_def *x = nir_ssa_for_alu_src(b, alu, 0);
   nir_def *y = nir_ssa_for_alu_src(b, alu, 1);
   nir_def *xl = nir_unpack_64_2x32_split_x(b, x);
   nir_def *xh = nir_unpack_64_2x32_split_y(b, x);
   nir_def *yl = nir_unpack_64_2x32_split_x(b, y);
   nir_def *yh = nir_unpack_64_2x32_split_y(b, y);
   nir_def *hi = nir_iadd(
      b, nir_umul_high(b, xl, yl),
      nir_iadd(b, nir_imul(b, xl, yh), nir_imul(b, xh, yl)));

   return nir_pack_64_2x32_split(b, nir_imul(b, xl, yl), hi);
}

static bool
agx_nir_lower_imul64(nir_shader *nir, bool lower_full)
{
   return nir_shader_lower_instructions(
      nir, agx_nir_lower_imul64_filter, agx_nir_lower_imul64_instr,
      (void *)(uintptr_t)lower_full);
}

/* Extract signed packed bytes directly in 32-bit arithmetic before the
 * generic 8-bit lowering introduces intermediate 16-bit conversions. */
static bool
agx_nir_unpack_signed_bytes(nir_builder *b, nir_alu_instr *alu, void *data)
{
   if (alu->op != nir_op_i2i32 || alu->src[0].src.ssa->bit_size != 8)
      return false;

   nir_scalar bytes[NIR_MAX_VEC_COMPONENTS];
   for (unsigned c = 0; c < alu->def.num_components; ++c) {
      bytes[c] = nir_scalar_chase_movs(
         nir_scalar_chase_alu_src(nir_get_scalar(&alu->def, c), 0));
      if (!nir_scalar_is_alu(bytes[c]) ||
          nir_scalar_alu_op(bytes[c]) != nir_op_unpack_32_4x8)
         return false;
   }

   b->cursor = nir_before_instr(&alu->instr);
   nir_def *result[NIR_MAX_VEC_COMPONENTS];
   for (unsigned c = 0; c < alu->def.num_components; ++c) {
      nir_scalar word = nir_scalar_chase_alu_src(bytes[c], 0);
      nir_def *value = nir_channel(b, word.def, word.comp);
      result[c] = nir_ishr_imm(b, nir_ishl_imm(b, value,
                                24 - 8 * bytes[c].comp), 24);
   }
   nir_def_rewrite_uses(&alu->def, nir_vec(b, result, alu->def.num_components));
   nir_instr_remove(&alu->instr);
   return true;
}

static bool
agx_nir_has_alu64(const nir_shader *nir, nir_op opcode)
{
   nir_foreach_function_impl(impl, nir) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_alu)
               continue;

            const nir_alu_instr *alu = nir_instr_as_alu(instr);
            if (alu->op == opcode && alu->def.bit_size == 64)
               return true;
         }
      }
   }

   return false;
}

bool
agx_mem_vectorize_cb(unsigned align_mul, unsigned align_offset,
                     unsigned bit_size, unsigned num_components,
                     int64_t hole_size, nir_intrinsic_instr *low,
                     nir_intrinsic_instr *high, void *data)
{
   if (hole_size > 0)
      return false;

   /* Must be aligned to the size of the load */
   unsigned align = nir_combined_align(align_mul, align_offset);
   if ((bit_size / 8) > align)
      return false;

   if (num_components > 4)
      return false;

   if (bit_size > 32)
      return false;

   return true;
}

static bool
set_speculate(nir_builder *b, nir_instr *instr, UNUSED void *_)
{
   if (instr->type == nir_instr_type_intrinsic &&
       nir_intrinsic_has_access(nir_instr_as_intrinsic(instr))) {
      nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
      nir_intrinsic_set_access(
         intr, ACCESS_CAN_SPECULATE | nir_intrinsic_access(intr));
      return true;
   }

   if (instr->type == nir_instr_type_tex) {
      nir_instr_as_tex(instr)->can_speculate = true;
      return true;
   }

   return false;
}

static void
agx_optimize_nir(nir_shader *nir, bool soft_fault, bool uniform_atomics,
                 bool byte_indices,
                 uint16_t *preamble_size, uint8_t *ts_count, uint8_t *ss_count)
{
   /* This runs only once up front since other optimizations don't affect it */
   NIR_PASS(_, nir, nir_opt_shrink_stores, true);

   agx_optimize_loop_nir(nir, byte_indices);

   /* If soft fault is enabled, we can freely speculate everything. That lets us
    * peephole select and form preambles more aggressively.
    */
   if (soft_fault) {
      NIR_PASS(_, nir, nir_shader_instructions_pass, set_speculate,
               nir_metadata_control_flow, NULL);
   }

   /* Peephole select again after setting the speculate flag but before
    * vectorizing. This cleans up short-circuit loads in unrolled loops.
    *
    * XXX: Set indirect_load_ok once we can investigate CTS flakes.
    */
   nir_opt_peephole_select_options peephole_select_options = {
      .limit = agx_select_limit(nir, byte_indices),
      .expensive_alu_ok = true,
   };
   NIR_PASS(_, nir, nir_opt_peephole_select, &peephole_select_options);

   NIR_PASS(_, nir, nir_opt_load_store_vectorize,
            &(const nir_load_store_vectorize_options){
               .modes = nir_var_mem_global | nir_var_mem_constant |
                        nir_var_shader_temp,
               .callback = agx_mem_vectorize_cb,
            });
   NIR_PASS(_, nir, nir_lower_pack);
   NIR_PASS(_, nir, nir_opt_algebraic);

   nir_divergence_analysis(nir);
   bool progress = false;

   static const nir_lower_subgroups_options subgroups_options = {
      .ballot_bit_size = 32,
      .ballot_components = 1,
      .lower_elect = true,
      .lower_subgroup_masks = true,
   };

   /*
    * Rewrites a uniform-address atomic into "one lane of the subgroup performs
    * the whole operation", which needs ACTIVE_THREAD_INDEX_IN_SUBGROUP and
    * TOTAL_ACTIVE_THREADS_IN_SUBGROUP. Their AGX3 numbers (SR 34/36) are now
    * witnessed and grounded in agx3_map_sr(), but the optimized form is kept
    * off on AGX3 until a CTS campaign validates it; before the witness every
    * shader containing a uniform-address atomic failed to compile.
    *
    * Skipping the optimization keeps atomics CORRECT -- each lane performs its
    * own atomic, which is what the unoptimized program means -- at the cost of
    * one atomic per lane instead of one per subgroup on uniform addresses.
    * Re-enable once those two SR numbers are witnessed rather than guessed; a
    * wrong SR number would silently elect the wrong lane.
    */
   if (uniform_atomics)
      NIR_PASS(progress, nir, nir_opt_uniform_atomics, true);

   /* Lower addressing modes. The sooner we do this, the sooner we get rid of
    * amul/aadd instructions and can let nir_opt_algebraic do its job. But we
    * want to vectorize first since nir_opt_load_store_vectorize doesn't know
    * how to handle our loads. Likewise for uniform atomic optimization.
    */
   NIR_PASS(_, nir, agx_nir_lower_address, byte_indices);

   NIR_PASS(progress, nir, nir_opt_uniform_subgroup, &subgroups_options);
   if (progress) {
      if (byte_indices)
         NIR_PASS(_, nir, agx_nir_lower_subgroups_g17p);
      else
         NIR_PASS(_, nir, agx_nir_lower_subgroups);
   }

   /* The above create operations that need lowering/optimizing */
   do {
      progress = false;

      NIR_PASS(progress, nir, nir_opt_algebraic);
      NIR_PASS(progress, nir, nir_opt_constant_folding);
      NIR_PASS(progress, nir, nir_opt_dce);
   } while (progress);

   progress = false;

   /* If address lowering made progress, clean up before forming preambles.
    * Otherwise the optimized preambles might just be constants! Do it before
    * lowering int64 too, to avoid lowering constant int64 arithmetic.
    */
   if (progress) {
      NIR_PASS(_, nir, nir_opt_constant_folding);
      NIR_PASS(_, nir, nir_opt_dce);
   }

   /* Only lower int64 after optimizing address arithmetic, so that u2u64/i2i64
    * conversions remain.
    */
   progress = false;
   NIR_PASS(progress, nir, nir_lower_int64);

   /* If we lowered actual int64 arithmetic (not folded into the address
    * calculations), then clean up after the lowering.
    */
   if (progress) {
      do {
         progress = false;

         NIR_PASS(progress, nir, nir_opt_algebraic);
         NIR_PASS(progress, nir, nir_opt_constant_folding);
         NIR_PASS(progress, nir, nir_opt_dce);
      } while (progress);
   }

   /* Reassociate before forming preambles because it makes preambles more
    * effective. Clean up after.
    */
   nir_opt_reassociate_loop(
      nir, nir_reassociate_scalar_math | nir_reassociate_cse_heuristic);

   /* Lower fmin/fmax before optimizing preambles so we can see across uniform
    * expressions. Do it after nir_opt_reassociate because nir_opt_reassociate
    * should work on unlowered fmin/fmax.
    */
   NIR_PASS(_, nir, agx_nir_lower_fminmax);

   /* Force preamble generation for internal shaders since hk's meta copies
    * depend on it for correctness with imgwblk intrinsics.
    */
   if (preamble_size &&
       (!(agx_compiler_debug & AGX_DBG_NOPREAMBLE) || nir->info.internal)) {
      unsigned sizes[] = {
         *preamble_size,
         ts_count ? *ts_count : 1000 /* large finite */,
         ss_count ? *ss_count : 1000 /* large finite */,
      };

      /* Don't clobber txf sampler */
      if (sizes[2] == 0)
         sizes[2]++;

      NIR_PASS(_, nir, agx_nir_opt_preamble, sizes, byte_indices);

      *preamble_size = sizes[0];

      if (ts_count)
         *ts_count = sizes[1];

      /* if something other than the txf sampler is written... */
      if (ss_count && sizes[2] > 1)
         *ss_count = sizes[2];
   }

   /* Forming preambles may dramatically reduce the instruction count
    * in certain blocks, causing some if-else statements to become
    * trivial. We want to peephole select those, given that control flow
    * prediction instructions are costly.
    *
    * We need to lower int64 again to deal with the resulting 64-bit csels.
    */
   NIR_PASS(_, nir, nir_opt_peephole_select, &peephole_select_options);
   NIR_PASS(_, nir, nir_lower_int64);

   /* We need to lower fmin/fmax again after nir_opt_algebraic_late due to f2fmp
    * wackiness. This is usually a no-op but is required for correctness in
    * GLES.
    */
   NIR_PASS(_, nir, nir_opt_algebraic_late);
   NIR_PASS(_, nir, agx_nir_lower_fminmax);

   /* The AGX3 byte-address backend cannot encode 64-bit shifted adds.
    * Lower remaining multiplies before fusion turns x * 3 into one; the
    * normal late multiply pass cannot recognize fused backend opcodes. */
   if (byte_indices)
      NIR_PASS(_, nir, agx_nir_lower_imul64, true);

   if (byte_indices)
      NIR_PASS(_, nir, agx_nir_opt_native_integer);

   /* Fuse add/sub/multiplies/shifts after running opt_algebraic_late to fuse
    * isub but before shifts are lowered.
    */
   do {
      progress = false;

      NIR_PASS(progress, nir, nir_opt_dce);
      NIR_PASS(progress, nir, nir_opt_cse);
      NIR_PASS(progress, nir, agx_nir_fuse_algebraic_late);
   } while (progress);

   /* Before optimizing bounds checks, we need to clean up and index defs so
    * optimize_bounds does the right thing.
    */
   NIR_PASS(_, nir, nir_opt_copy_prop);
   NIR_PASS(_, nir, nir_opt_dce);

   nir_foreach_function_impl(impl, nir) {
      nir_index_ssa_defs(impl);
   }

   /* Do remaining lowering late, since this inserts &s for shifts so we want to
    * do it after fusing constant shifts. Constant folding will clean up.
    */
   NIR_PASS(_, nir, agx_nir_lower_algebraic_late);
   NIR_PASS(_, nir, agx_nir_fuse_selects);
   NIR_PASS(_, nir, nir_opt_constant_folding);
   NIR_PASS(_, nir, nir_opt_combine_barriers, NULL, NULL);
   NIR_PASS(_, nir, nir_opt_copy_prop);
   NIR_PASS(_, nir, nir_opt_dce);
   NIR_PASS(_, nir, nir_opt_cse);
   NIR_PASS(_, nir, nir_lower_alu_to_scalar, NULL, NULL);
   NIR_PASS(_, nir, nir_lower_load_const_to_scalar);

   /* Cleanup optimizations */
   nir_move_options move_all = nir_move_const_undef | nir_move_load_ubo |
                               nir_move_load_input | nir_move_load_frag_coord |
                               nir_move_comparisons | nir_move_copies |
                               nir_move_load_ssbo | nir_move_alu;

   NIR_PASS(_, nir, nir_opt_sink, move_all);
   NIR_PASS(_, nir, nir_opt_move, move_all);
   NIR_PASS(progress, nir, nir_lower_all_phis_to_scalar);

   /* After lowering phis to scalar, we must copy prop/DCE for correctness. RA
    * can't deal with dead phis.
    */
   do {
      NIR_PASS(progress, nir, nir_opt_copy_prop);
      NIR_PASS(progress, nir, nir_opt_dce);
      progress = false;
   } while (progress);
}

/*
 * Varyings that are used as texture coordinates should be kept at fp32, because
 * fp16 does not have enough precision for large textures. It's technically
 * conformant not to, but every app gets this wrong.
 */
static bool
gather_texcoords(nir_builder *b, nir_tex_instr *tex, void *data)
{
   nir_def *coord = nir_get_tex_src(tex, nir_tex_src_coord);
   if (!coord)
      return false;

   nir_scalar x = nir_scalar_resolved(coord, 0);
   nir_scalar y = nir_scalar_resolved(coord, 1);
   nir_intrinsic_instr *intr = nir_scalar_as_intrinsic(x);

   if (x.def == y.def && intr &&
       intr->intrinsic == nir_intrinsic_load_interpolated_input) {

      uint64_t *mask = data;
      *mask |= BITFIELD64_BIT(nir_intrinsic_io_semantics(intr).location);
   }

   return false;
}

/*
 * Build a bit mask of varyings (by location) that are used as texture
 * coordinates. This information is needed by lower_mediump_io.
 */
uint64_t
agx_gather_texcoords(nir_shader *nir)
{
   assert(nir->info.stage == MESA_SHADER_FRAGMENT);

   uint64_t mask = 0;
   nir_shader_tex_pass(nir, gather_texcoords, nir_metadata_all, &mask);
   return mask;
}

static nir_mem_access_size_align
mem_access_size_align_cb(nir_intrinsic_op intrin, uint8_t bytes,
                         uint8_t bit_size, uint32_t align,
                         uint32_t align_offset, bool offset_is_const,
                         enum gl_access_qualifier access, const void *cb_data)
{
   align = nir_combined_align(align, align_offset);

   assert(util_is_power_of_two_nonzero(align));

   if ((bytes & 1) || (align == 1))
      bit_size = 8;
   else if ((bytes & 2) || (align == 2))
      bit_size = 16;
   else if (bit_size >= 32)
      bit_size = 32;

   return (nir_mem_access_size_align){
      .num_components = MIN2(bytes / (bit_size / 8), 4),
      .bit_size = bit_size,
      .align = bit_size / 8,
      .shift = nir_mem_access_shift_method_scalar,
   };
}

static unsigned
lower_bit_size_callback(const nir_instr *instr, UNUSED void *_)
{
   if (instr->type == nir_instr_type_intrinsic) {
      /* Handle small subgroup ops */
      nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);

      switch (intr->intrinsic) {
      case nir_intrinsic_reduce:
      case nir_intrinsic_exclusive_scan:
      case nir_intrinsic_inclusive_scan:
         /* The identity for iand doesn't work for lowered 1-bit booleans, so
          * lower that explicitly.
          */
         if (nir_intrinsic_reduction_op(intr) == nir_op_iand &&
             intr->def.bit_size == 1)
            return 16;

         /* In general, we have 16-bit ops instead of 8-bit, so lower those. */
         return intr->def.bit_size == 8 ? 16 : 0;
      default:
         return 0;
      }
   } else if (instr->type == nir_instr_type_alu) {
      /* Lower 8-bit ALU to 16-bit. We check the destination, as we do not want
       * to lower conversions from 8-bit to larger types. Those conversions get
       * implemented natively.
       */
      nir_alu_instr *alu = nir_instr_as_alu(instr);

      switch (alu->op) {
      case nir_op_bitfield_reverse:
         return alu->def.bit_size < 32 ? 32 : 0;
      default:
         break;
      }

      if (alu->def.bit_size == 8 && !is_conversion_to_8bit(alu->op))
         return 16;
      else if (alu->def.bit_size == 1 && alu->src[0].src.ssa->bit_size == 8)
         return 16 /* comparisons */;
   }

   return 0;
}

static bool
lower_load_from_texture_handle(nir_builder *b, nir_intrinsic_instr *intr,
                               void *data)
{
   if (intr->intrinsic != nir_intrinsic_load_from_texture_handle_agx)
      return false;

   nir_intrinsic_instr *handle = nir_src_as_intrinsic(intr->src[0]);

   b->cursor = nir_instr_remove(&intr->instr);
   nir_def *base = nir_load_preamble(b, 1, 64, nir_intrinsic_desc_set(handle));
   nir_def *offset = nir_u2u64(b, handle->src[0].ssa);

   nir_def_rewrite_uses(&intr->def, nir_iadd(b, base, offset));
   return true;
}

static bool
lower_sink_address(nir_builder *b, nir_intrinsic_instr *intr, UNUSED void *data)
{
   switch (intr->intrinsic) {
   case nir_intrinsic_load_ro_sink_address_poly:
      b->cursor = nir_before_instr(&intr->instr);
      nir_def_replace(&intr->def, nir_imm_int64(b, AGX_ZERO_PAGE_ADDRESS));
      return true;
   case nir_intrinsic_ro_to_rw_poly: {
      b->cursor = nir_before_instr(&intr->instr);
      nir_def_replace(
         &intr->def,
         nir_bcsel(b, nir_ieq_imm(b, intr->src[0].ssa, AGX_ZERO_PAGE_ADDRESS),
                   nir_imm_int64(b, AGX_SCRATCH_PAGE_ADDRESS),
                   intr->src[0].ssa));
      return true;
   }
   default:
      break;
   }

   return false;
}

static void
agx_remove_unreachable_block(agx_block *block)
{
   /* Delete the edges */
   agx_foreach_successor(block, succ) {
      unsigned block_idx = agx_predecessor_index(succ, block);

      /* Remove the corresponding predecessor from the successor */
      struct util_dynarray *blocks = &succ->predecessors;
      int remaining = agx_num_predecessors(succ) - (block_idx + 1);
      assert(remaining >= 0);

      memcpy(util_dynarray_element(blocks, agx_block *, block_idx),
             util_dynarray_element(blocks, agx_block *, block_idx + 1),
             remaining * sizeof(agx_block *));
      blocks->size -= sizeof(agx_block *);

      /* Remove the corresponding source from the phis */
      agx_foreach_phi_in_block(succ, phi) {
         assert(block_idx + 1 <= phi->nr_srcs);

         memcpy(phi->src + block_idx, phi->src + block_idx + 1,
                (phi->nr_srcs - (block_idx + 1)) * sizeof(phi->src[0]));

         phi->nr_srcs--;

         /* This might cause phis to become trivial. Lower 1-source phis to
          * moves and let copyprop take it from here.
          */
         if (phi->nr_srcs == 1) {
            phi->op = AGX_OPCODE_MOV;
         }
      }
   }

   /* Remove the successor from the predecessor. */
   block->successors[0] = NULL;
   block->successors[1] = NULL;

   /* Note: we do not remove the block itself, although it is now fully orphaned
    * in the control flow graph. We still need it in source order if it has any
    * pop_exec instructions, for a loop continue block.
    *
    * TODO: Is there a better way to handle this?
    *
    * Affects: dEQP-VK.graphicsfuzz.cov-matching-if-always-true-inside-loop
    */
}

/*
 * NIR sometimes contains unreachable blocks (e.g. due to infinite loops). These
 * blocks have no predecessors, but do have successors and can contribute to
 * phis. They are dead and do not need to be here. Further, they violate the IR
 * invariant:
 *
 *    Live-in sources are live-out in all predecessors.
 *
 * ...which RA depends on when handling live range splits. The simplest solution
 * is to simply delete these dead blocks. Fortunately, because they are
 * unreachable, this does not have any ill effects. Notably, this cannot
 * introduce critical edges.
 *
 * Deleting a block may cause a successor to become unreachable, so we use a
 * fixed-point algorithm to converge.
 */
static void
agx_remove_unreachable_blocks(agx_context *ctx)
{
   agx_block *start = agx_start_block(ctx);
   bool progress;

   do {
      progress = false;

      agx_foreach_block_safe(ctx, pred) {
         if (pred != start && agx_num_predecessors(pred) == 0 &&
             agx_num_successors(pred) > 0) {

            agx_remove_unreachable_block(pred);
            progress = true;
         }
      }
   } while (progress);
}

static bool
agx_should_dump(nir_shader *nir, unsigned agx_dbg_bit)
{
   return (agx_compiler_debug & agx_dbg_bit) &&
          !(nir->info.internal && !(agx_compiler_debug & AGX_DBG_INTERNAL));
}

#define AGX_PASS(shader, pass, ...)                                            \
   do {                                                                        \
      pass(shader, ##__VA_ARGS__);                                             \
      agx_validate(ctx, #pass);                                                \
   } while (0)

/* Keep scalar UVS values live until a native-shaped terminal writeout.
 * Run after pressure scheduling but before RA: moving physical-register stores
 * would lose values when later instructions reuse their sources. */
bool
agx3_sink_vertex_outputs(agx_context *ctx)
{
   if (ctx->key->dev.encoding != AGX_ISA_ENCODING_AGX3B ||
       ctx->stage != MESA_SHADER_VERTEX || ctx->is_preamble ||
       ctx->key->secondary)
      return false;

   /* An earlier STOP terminates only that path and does not observe exports
    * in the final block. Keep it in place, just like other control flow.
    * Outputs already belonging to the final block can be bundled locally
    * even when earlier blocks contain instance transforms. Never move an
    * output across a branch or change its execution mask. */
   agx_block *output_block = NULL;
   agx_foreach_block(ctx, block)
      output_block = block;
   if (output_block == NULL)
      return false;
   agx_foreach_block(ctx, block) {
      if (block == output_block)
         break;
      agx_foreach_instr_in_block(block, I) {
         if (I->op == AGX_OPCODE_ST_VARY)
            return false;
      }
   }

   agx_instr *stores[AGX3_MAX_VERTEX_OUTPUTS] = {0}, *stop = NULL;
   bool saw_store = false;
   agx_foreach_instr_in_block(output_block, I) {
      if (stop)
         return false;
      switch (I->op) {
      case AGX_OPCODE_STOP:
         stop = I;
         break;
      case AGX_OPCODE_ST_VARY: {
         if (I->nr_dests || I->nr_srcs != 2 ||
             I->src[0].type != AGX_INDEX_IMMEDIATE || I->src[0].value >= ARRAY_SIZE(stores) ||
             I->src[1].type != AGX_INDEX_NORMAL || I->src[1].memory ||
             I->src[1].size != AGX_SIZE_32 || agx_channels(I->src[1]) != 1 ||
             stores[I->src[0].value])
            return false;
         stores[I->src[0].value] = I;
         saw_store = true;
         break;
      }
      case AGX_OPCODE_GET_SR:
         if (saw_store && I->sr != AGX_SR_VERTEX_ID && I->sr != AGX_SR_INSTANCE_ID)
            return false;
         break;
      default:
         /* Pure SSA producers may move relative to vertex writeout. Their
          * actual encodings remain validated by the normal compiler path;
          * side effects and control flow require a separate ordering proof. */
         if (saw_store && !agx_opcodes_info[I->op].can_eliminate)
            return false;
         break;
      }
   }
   if (!stop)
      return false;
   /* Constant position components need no UVS store. A layer or point-size
    * export can therefore follow holes in the position vector. Sort the live
    * SSA exports just as for dense outputs, without manufacturing values for
    * those holes. The packer's sparse protocol retains their actual indices.
    */
   bool user_output = false;
   for (unsigned i = 4; i < ARRAY_SIZE(stores); ++i)
      user_output |= stores[i] != NULL;
   if (!user_output)
      return false;
   for (unsigned i = 0; i < ARRAY_SIZE(stores); ++i) {
      if (!stores[i])
         continue;
      list_del(&stores[i]->link);
      list_addtail(&stores[i]->link, &stop->link);
   }
   return true;
}

static bool
agx_compile_function_nir(nir_shader *nir, nir_function_impl *impl,
                         struct agx_shader_key *key,
                         struct util_dynarray *binary,
                         struct agx_shader_info *out, unsigned *offset_out)
{
   const unsigned initial_binary_size = binary->size;
   nir_index_blocks(impl);
   nir_index_ssa_defs(impl);

   agx_context *ctx = rzalloc(NULL, agx_context);
   ctx->nir = nir;
   ctx->is_preamble = impl->function->is_preamble;
   ctx->out = out;
   ctx->key = key;
   ctx->stage = nir->info.stage;
   ctx->allocated_vec = _mesa_hash_table_u64_create(ctx);
   ctx->indexed_nir_blocks = rzalloc_array(ctx, agx_block *, impl->num_blocks);
   list_inithead(&ctx->blocks);

   if (ctx->stage == MESA_SHADER_FRAGMENT && !ctx->is_preamble) {
      ctx->any_cf = key->fs.inside_sample_loop;
   }

   /* CLR's grounded compact form addresses a low TPS word. Reserve it
    * before assigning NIR scratch and RA spill addresses, rather than
    * appending it after a potentially large spill frame. */
   if (key->dev.encoding == AGX_ISA_ENCODING_AGX3B) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;
            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            if ((intr->intrinsic == nir_intrinsic_global_atomic_agx ||
                 intr->intrinsic == nir_intrinsic_global_atomic_swap_agx) &&
                !nir_def_is_unused(&intr->def))
               ctx->atomic_tps_reserved = true;
         }
      }
   }

   ctx->alloc = impl->ssa_alloc;
   emit_cf_list(ctx, &impl->body);
   agx_emit_phis_deferred(ctx);

   /* Index blocks now that we're done emitting so the order is consistent. Do
    * this before agx_remove_unreachable_blocks so we match NIR indexing. This
    * makes for easier debugging.
    */
   agx_foreach_block(ctx, block) {
      block->index = ctx->num_blocks++;
   }

   agx_remove_unreachable_blocks(ctx);

   /* Only allocate scratch if it's statically used, regardless of if the NIR
    * info claims otherwise.
    */
   if (ctx->any_scratch) {
      assert(!ctx->is_preamble && "preambles don't use scratch");
      ctx->scratch_size_B = align(nir->scratch_size, 16);
   }
   if (ctx->atomic_tps_reserved)
      ctx->scratch_size_B += 16;

   /* Stop the main shader or preamble shader after the exit block. For real
    * functions, we would return here.
    */
   if (!ctx->key->no_stop || ctx->is_preamble) {
      agx_block *last_block = list_last_entry(&ctx->blocks, agx_block, link);
      agx_builder _b = agx_init_builder(ctx, agx_after_block(last_block));
      agx_stop(&_b);
   }

   agx_validate(ctx, "IR translation");

   if (likely(!(agx_compiler_debug & AGX_DBG_NOOPT))) {
      /* Eliminate dead instructions before CSE to avoid silly scheduling */
      AGX_PASS(ctx, agx_dce, false);

      /* CSE before eliminating dead destinations so that subdivision is
       * optimized properly.
       */
      AGX_PASS(ctx, agx_opt_cse);

      /* After DCE, use counts are right so we can run the optimizer. */
      AGX_PASS(ctx, agx_optimizer_backward);
      AGX_PASS(ctx, agx_optimizer_forward);
      AGX_PASS(ctx, agx_opt_compact_constants);

      /* After inlining constants, promote what's left */
      if (key->promote_constants && !key->secondary &&
          !(agx_compiler_debug & AGX_DBG_NOPROMOTE)) {
         AGX_PASS(ctx, agx_opt_promote_constants);
      }
   }

   agx_lower_agx3_half_converts(ctx);
   agx_lower_agx3_half_integer_sources(ctx);
   AGX_PASS(ctx, agx_lower_agx3_widening_iadd);
   AGX_PASS(ctx, agx_lower_agx3_shifted_subtract);

   if (key->dev.encoding != AGX_ISA_ENCODING_AGX2)
      agx_materialize_agx3_alu_constants(ctx);

   /* Constant materialization can introduce mixed-width FADD operands. */
   agx_lower_agx3_fma_sources(ctx);

   AGX_PASS(ctx, agx_lower_agx3_cf_constants);
   AGX_PASS(ctx, agx_lower_agx3_half_iadd_constants);
   AGX_PASS(ctx, agx_lower_agx3_wide_iadd);

   /* For correctness, lower uniform sources after copyprop (for correctness,
    * as copyprop creates uniform sources). To keep register pressure in
    * check, lower after CSE, since moves are cheaper than registers.
    */
   AGX_PASS(ctx, agx_lower_uniform_sources);

   /* Native ALU legalization creates copies of uniform sources after the
    * normal CSE pass. Reuse those values within each block of a fragment,
    * compute, or driver-qualified vertex shader; its uniform file is immutable.
    * Constant programs may write uniforms, so keep their reads ordered. Keep
    * the separately tuned internal clear/reload programs unchanged.
    */
   if (key->dev.encoding != AGX_ISA_ENCODING_AGX2 &&
       (ctx->stage == MESA_SHADER_FRAGMENT || ctx->stage == MESA_SHADER_COMPUTE ||
        (ctx->stage == MESA_SHADER_VERTEX &&
         key->native_vertex_uniform_cse)) &&
       !ctx->is_preamble && !nir->info.internal)
      AGX_PASS(ctx, agx_opt_cse);

   /* RA correctness depends on DCE */
   AGX_PASS(ctx, agx_dce, true);
   /* Full-width native IMAD now handles the division-by-three product.
    * The old fixed-form shortcut only accepts wait slot 1 and rejects
    * internal geometry helpers whose operands have another dependency. */


   if (agx_should_dump(nir, AGX_DBG_SHADERS))
      agx_print_shader(ctx, stdout);

   if (likely(!(agx_compiler_debug & AGX_DBG_NOSCHED))) {
      AGX_PASS(ctx, agx_pressure_schedule);
      if (agx_native_early_textures_enabled())
         AGX_PASS(ctx, agx_schedule_native_samples);
   }

   if (agx_should_dump(nir, AGX_DBG_SHADERS))
      agx_print_shader(ctx, stdout);

   AGX_PASS(ctx, agx3_sink_vertex_outputs);

   if (!agx_ra(ctx)) {
      binary->size = initial_binary_size;
      ralloc_free(ctx);
      return false;
   }
   agx_validate(ctx, "agx_ra");
   agx_lower_64bit_postra(ctx);
   agx_lower_agx3_stack_vectors(ctx);

   if (ctx->scratch_size_B > 0) {
      /* Apple always allocate 40 more bytes in the entrypoint and align to 4. */
      uint64_t stack_size = ctx->key->dev.encoding == AGX_ISA_ENCODING_AGX2
         ? align(DIV_ROUND_UP(ctx->scratch_size_B, 4) + 10, 4)
         : ctx->scratch_size_B;

      if (ctx->key->dev.encoding == AGX_ISA_ENCODING_AGX2) {
         assert(stack_size < INT16_MAX);
         agx_block *start_block = agx_start_block(ctx);
         agx_builder _b =
            agx_init_builder(ctx, agx_before_block(start_block));
         agx_stack_adjust(&_b, stack_size);

         /* If we're going to execute multiple times, make sure we clean up
          * after ourselves, else the hardware faults. */
         if (ctx->stage == MESA_SHADER_FRAGMENT && !ctx->is_preamble &&
             ctx->key->fs.inside_sample_loop) {
            _b = agx_init_builder(ctx, agx_after_block(agx_end_block(ctx)));
            agx_stack_adjust(&_b, -stack_size);
         }
      }

      /* AGX3 static spills are copy/rld accesses to the launch-allocated TPS
       * frame. The pinned G15/G16/G17 compute spill corpus emits no incStk;
       * that distinct instruction belongs to the ray-traversal stack. Keep
       * publishing the frame size below, but never inject an AGX2 stack
       * instruction into an AGX3 shader. */

      if (ctx->is_preamble)
         out->preamble_scratch_size = stack_size;
      else
         out->scratch_size = stack_size;
   }

   if ((ctx->stage == MESA_SHADER_VERTEX ||
        ctx->stage == MESA_SHADER_TESS_EVAL) &&
       !impl->function->is_preamble && !ctx->key->secondary)
      agx_set_st_vary_final(ctx);

   /* AGX3 has no scoreboard `wait`; dependency fences ride on the terminal
    * `end`, and every real kernel ends with mask 0. Wait insertion is
    * therefore an AGX2-only pass. */
   if (ctx->key->dev.encoding == AGX_ISA_ENCODING_AGX2)
      agx_insert_waits(ctx);
   agx_opt_empty_else(ctx);
   agx_opt_break_if(ctx);
   agx_opt_jmp_none(ctx);

   if (ctx->any_quad_divergent_shuffle)
      agx_lower_divergent_shuffle(ctx);

   agx_lower_pseudo(ctx);

   if (ctx->key->dev.encoding == AGX_ISA_ENCODING_AGX3B &&
       (ctx->stage == MESA_SHADER_COMPUTE ||
        (ctx->stage == MESA_SHADER_FRAGMENT &&
         agx_native_fragment_loads_enabled())) && !ctx->is_preamble &&
       !nir->info.internal)
      agx_schedule_native_loads(ctx);

   /* Set last-use bits after lowering pseudo for best results. */
   /* Causal control for the M4 vertex-fog reproducer. This changes only
    * register lifetime hints; it is not a production workaround. */
   if (!(ctx->stage == MESA_SHADER_VERTEX &&
         ctx->key->dev.encoding == AGX_ISA_ENCODING_AGX3B &&
         debug_get_bool_option("AGX3_VERTEX_NO_CACHE_HINTS", false)))
      agx_opt_register_cache(ctx);

   if (agx_should_dump(nir, AGX_DBG_SHADERS))
      agx_print_shader(ctx, stdout);

   /* Upload constants before the binary instead after to reduce the chance they
    * get prefetched into the i-cache, when we want them only in the d-cache.
    * Also helps fill the padding space for small preambles.
    */
   if (ctx->out->rodata.size_16 && !impl->function->is_preamble) {
      ctx->out->rodata.offset = agx_pad_binary(binary, 4);
      unsigned size_16 = ctx->out->rodata.size_16;

      uint16_t *ro = util_dynarray_grow(binary, uint16_t, size_16);
      memcpy(ro, ctx->rodata, size_16 * 2);
   }

   unsigned offset = agx_pad_binary(binary, AGX_CODE_ALIGN);
   if (ctx->key->dev.encoding == AGX_ISA_ENCODING_AGX2) {
      agx_pack_binary(ctx, binary);
   } else {
      /* AGX3 (M3/M4/M5). Device open still refuses these parts
       * (agx_chip_codegen_supported), so this runs only when a caller builds an
       * AGX3 device key deliberately. agx3_pack_binary fails the shader -- never
       * aborts -- on any ungrounded op and rolls the emitted shader prefix back
       * before returning false. */
      if (!agx3_pack_binary(ctx, binary)) {
         if (agx_failure_ir)
            agx_print_shader(ctx, agx_failure_ir);
         binary->size = initial_binary_size;
         ralloc_free(ctx);
         return false;
      }
   }

   unsigned nr_gprs = ctx->max_reg + 1;

   /* If the preamble uses scratch (due to spilling), we need to set maximal
    * GPRs. Do it here so the driver doesn't have to worry about it.
    */
   if (impl->function->is_preamble)
      out->nr_preamble_gprs = ctx->scratch_size_B ? 256 : nr_gprs;
   else {
      out->nr_gprs = nr_gprs;
      out->agx3_staging_halfs = ctx->agx3_staging_halfs;
      out->agx3_staging_known = ctx->agx3_staging_known;
      out->agx3_fltnorm_start = ctx->agx3_fltnorm_start;
      out->agx3_colour_link = ctx->agx3_colour_link;
   }

   if (impl->function->is_preamble) {
      uint32_t instrs = 0;
      agx_foreach_instr_global(ctx, I) {
         instrs++;
      }

      ctx->out->stats.preamble_inst = instrs;
   } else {
      agx_calc_stats(ctx, binary->size, &ctx->out->stats);
   }

   /*
    * Disassemble if the environment variable is set. Also disassemble to
    * /dev/null in debug builds as a smoke test - a legally encoded shader
    * should not hit any disassembler errors.
    */
   bool dump_shaders = agx_should_dump(nir, AGX_DBG_SHADERS);
   if (dump_shaders && ctx->key->dev.encoding != AGX_ISA_ENCODING_AGX2) {
      /* There is no AGX3 instruction printer here yet. Retain the exact
       * packed bytes alongside the IR for offline comparison and decoding.
       * Print each binary in one call so parallel compiler threads cannot
       * interleave the individual bytes. */
      const size_t size = binary->size - offset;
      char *hex = ralloc_array(ctx, char, size * 2 + 1);
      static const char digits[] = "0123456789abcdef";
      const uint8_t *code = (const uint8_t *)binary->data + offset;
      for (size_t i = 0; i < size; ++i) {
         hex[2 * i] = digits[code[i] >> 4];
         hex[2 * i + 1] = digits[code[i] & 0xf];
      }
      hex[2 * size] = '\0';
      fprintf(stderr,
              "AGX3 binary: stage=%u preamble=%u encoding=%u size=%zu hex=%s\n",
              ctx->stage, ctx->is_preamble, ctx->key->dev.encoding, size, hex);
   }
#ifndef NDEBUG
   bool selftest = !dump_shaders;
#else
   bool selftest = false;
#endif

#ifdef _WIN32
   /* Windows' null device is NUL; /dev/null may fail to open and feeding a
    * NULL FILE to the disassembler would turn a debug self-test into an
    * access violation during shader compilation. */
   if (selftest) {
      FILE *probe = fopen("NUL", "w");
      if (!probe) {
         fprintf(stderr, "agx_compile: unable to open NUL; skipping "
                         "disassembler self-test\n");
         selftest = false;
      } else {
         fclose(probe);
      }
   }
#endif

   if ((dump_shaders || selftest) &&
       ctx->key->dev.encoding == AGX_ISA_ENCODING_AGX2) {
#ifdef _WIN32
      FILE *fp = selftest ? fopen("NUL", "w") : stdout;
#else
      FILE *fp = selftest ? fopen("/dev/null", "w") : stdout;
#endif
      bool errors =
         agx2_disassemble(binary->data + offset, binary->size - offset, fp);

      if (errors) {
         /* In a self-test, disassemble again to get something the user can look
          * at to figure out what went wrong.
          */
         if (selftest) {
            agx2_disassemble(binary->data + offset, binary->size - offset,
                             stderr);
         }

         UNREACHABLE("Disassembly error hit.");
      }

      if (selftest) {
         fclose(fp);
      }
   }

   *offset_out = offset;
   ralloc_free(ctx);
   return true;
}

/*
 * The hardware frcp instruction is sometimes off by 1 ULP. For correctly
 * rounded frcp, a refinement step is required. This routine has been
 * exhaustively tested with a modified math_bruteforce.
 *
 * While Khronos APIs allow 2.5 ULP error for divides, nir_lower_idiv relies on
 * correctly rounded frcp. This is therefore load bearing for integer division
 * on all APIs.
 */
static nir_def *
libagx_frcp(nir_builder *b, nir_def *x)
{
   nir_def *u = nir_frcp(b, x);

   /* Do 1 Newton-Raphson refinement step.
    *
    * Define f(u) = xu - 1. Then f(u) = 0 iff u = 1/x. Newton's method gives:
    *
    * u_2 = u - f(u) / f'(u) = u - (xu - 1) / x
    *
    * Our original guess is close, so we approximate (1 / x) by u:
    *
    * u_2 = u - u(xu - 1) = u + u(1 - xu)
    *     = fma(fma(-x, u, 1), u, u)
    */
   nir_def *one = nir_imm_float(b, 1.0);
   nir_def *u_2 = nir_ffma(b, nir_ffma(b, nir_fneg(b, x), u, one), u, u);

   /* If the original value was infinite, frcp will generate the correct zero.
    * However, the Newton-Raphson step would multiply 0 * Inf and get a NaN. So
    * skip the refinement step for infinite inputs. We do this backwards,
    * checking whether the refined result is NaN, since we can implement this
    * check in a single fcmpsel instruction. The other case where the refinement
    * is NaN is a NaN input, in which skipping refinement is acceptable.
    */
   return nir_bcsel(b, nir_fisnan(b, u_2), u, u_2);
}

static bool
agx_nir_lower_fdiv(nir_builder *b, nir_alu_instr *alu, void *_)
{
   if (alu->op != nir_op_frcp || !nir_alu_instr_is_exact(alu) ||
       alu->def.bit_size != 32)
      return false;

   b->cursor = nir_before_instr(&alu->instr);
   nir_def_replace(&alu->def, libagx_frcp(b, nir_ssa_for_alu_src(b, alu, 0)));
   return true;
}

/* Shared preprocessing with target-bounded subgroup choices. */
static nir_mem_access_size_align
agx3_scratch_word_size_align_cb(nir_intrinsic_op intrin, uint8_t bytes,
                                uint8_t bit_size, uint32_t align,
                                uint32_t align_offset, bool offset_is_const,
                                enum gl_access_qualifier access,
                                const void *cb_data)
{
   return (nir_mem_access_size_align){
      .num_components = MIN2(MAX2(bytes / 4, 1), 4),
      .bit_size = 32,
      .align = 4,
      .shift = nir_mem_access_shift_method_scalar,
   };
}

static void
agx_preprocess_nir_internal(nir_shader *nir, bool g17p)
{
   if (!nir)
      return;

   NIR_PASS(_, nir, nir_lower_vars_to_ssa);

   /* Lower large arrays to scratch and small arrays to csel */
   NIR_PASS(_, nir, nir_lower_vars_to_scratch, 256,
            glsl_get_natural_size_align_bytes, glsl_get_word_size_align_bytes);

   /* The native stack protocol is grounded for 32-bit words and single
    * static halves only. Access scratch in whole words on AGX3: loads
    * over-read within the private frame and narrow stores become the
    * pass's read-modify-write of the containing word
    * (subgroups.ballot_broadcast.*.subgroupbroadcast_{f,i,u}16vec*,
    * {i,u}8vec*). */
   if (g17p) {
      nir_lower_mem_access_bit_sizes_options scratch_words = {
         .modes = nir_var_function_temp,
         .callback = agx3_scratch_word_size_align_cb,
      };
      NIR_PASS(_, nir, nir_lower_mem_access_bit_sizes, &scratch_words);
   }
   NIR_PASS(_, nir, nir_lower_indirect_derefs_to_if_else_trees,
            nir_var_function_temp, ~0);
   NIR_PASS(_, nir, nir_split_var_copies);
   NIR_PASS(_, nir, nir_lower_global_vars_to_local);
   NIR_PASS(_, nir, nir_lower_var_copies);

   if (nir->info.stage == MESA_SHADER_FRAGMENT) {
      NIR_PASS(_, nir, agx_nir_lower_frag_sidefx, g17p);
      NIR_PASS(_, nir, nir_lower_is_helper_invocation);
   }

   /* Clean up deref gunk after lowering I/O */
   NIR_PASS(_, nir, nir_opt_dce);

   NIR_PASS(_, nir, nir_lower_alu_to_scalar, NULL, NULL);
   NIR_PASS(_, nir, nir_lower_frexp);
   NIR_PASS(_, nir, nir_lower_alu);
   NIR_PASS(_, nir, nir_lower_load_const_to_scalar);
   NIR_PASS(_, nir, nir_lower_flrp, 16 | 32 | 64, false);
   NIR_PASS(_, nir, agx_lower_sincos, g17p);
   NIR_PASS(_, nir, nir_shader_intrinsics_pass, agx_lower_front_face,
            nir_metadata_control_flow, NULL);
   if (g17p)
      NIR_PASS(_, nir, agx_nir_lower_subgroups_g17p);
   else
      NIR_PASS(_, nir, agx_nir_lower_subgroups);
   NIR_PASS(_, nir, nir_lower_all_phis_to_scalar);

   /* After lowering, run through the standard suite of NIR optimizations. We
    * will run through the loop later, once we have the shader key, but if we
    * run now, that run will ideally be almost a no-op.
    */
   agx_optimize_loop_nir(nir, g17p);

   /* Lower idiv after an optimization loop so we can constant fold more before
    * nir_opt_idiv_const.
    */
   NIR_PASS(_, nir, nir_opt_idiv_const, 16);

   nir_lower_idiv_options idiv_options = {
      .allow_fp16 = true,
   };

   NIR_PASS(_, nir, nir_lower_idiv, &idiv_options);

   /* Has to run after nir_lower_idiv */
   NIR_PASS(_, nir, nir_shader_alu_pass, agx_nir_lower_fdiv,
            nir_metadata_control_flow, NULL);

   NIR_PASS(_, nir, nir_opt_deref);
   NIR_PASS(_, nir, nir_lower_vars_to_ssa);

   /* We're lowered away all variables. Remove them all for smaller shaders. */
   NIR_PASS(_, nir, nir_remove_dead_variables, nir_var_all, NULL);
   nir->info.io_lowered = true;

   /* Move before lowering */
   nir_move_options move_all = nir_move_const_undef | nir_move_load_ubo |
                               nir_move_load_input | nir_move_load_frag_coord |
                               nir_move_comparisons | nir_move_copies |
                               nir_move_load_ssbo | nir_move_alu;

   NIR_PASS(_, nir, nir_opt_sink, move_all);
   NIR_PASS(_, nir, nir_opt_move, move_all);
   NIR_PASS(_, nir, agx_nir_lower_shared_bitsize);
}

void
agx_preprocess_nir(nir_shader *nir)
{
   agx_preprocess_nir_internal(nir, false);
}

void
agx_preprocess_nir_g17p(nir_shader *nir)
{
   agx_preprocess_nir_internal(nir, true);
}

static bool
lower_printf_buffer(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   const struct agx_device_key *dev = data;
   uint64_t value = 0;
   if (intr->intrinsic == nir_intrinsic_load_printf_buffer_address) {
      /* AGX3 frontends must publish the process-local printf address through
       * runtime state.  Baking it here would poison serialized shader caches;
       * the preflight below rejects an address intrinsic that escaped that
       * lowering.  AGX2 retains its established fixed-address ABI. */
      if (dev->encoding != AGX_ISA_ENCODING_AGX2)
         return false;
      value = LIBAGX_PRINTF_BUFFER_ADDRESS;
   } else if (intr->intrinsic == nir_intrinsic_load_printf_buffer_size) {
      value = LIBAGX_PRINTF_BUFFER_SIZE;
   } else {
      return false;
   }

   b->cursor = nir_before_instr(&intr->instr);
   nir_def_replace(&intr->def, nir_imm_intN_t(b, value, intr->def.bit_size));
   return true;
}

static bool
has_unlowered_printf_address(const nir_shader *nir)
{
   nir_foreach_function_impl(impl, nir) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type == nir_instr_type_intrinsic &&
                nir_instr_as_intrinsic(instr)->intrinsic ==
                   nir_intrinsic_load_printf_buffer_address)
               return true;
         }
      }
   }

   return false;
}

/* AGX3 byte vectors occupy packed physical storage, whereas AGX IR gives
 * each byte its own half register. Scalar byte operations preserve layout
 * and byte alignment without reading or overwriting neighbouring fields. */
static bool
lower_agx3_shared_byte_vectors(nir_builder *b, nir_intrinsic_instr *intr,
                               UNUSED void *data)
{
   bool load = intr->intrinsic == nir_intrinsic_load_shared;
   bool store = intr->intrinsic == nir_intrinsic_store_shared;
   if ((!load && !store) || intr->num_components <= 1 ||
       (load ? intr->def.bit_size : nir_src_bit_size(intr->src[0])) != 8)
      return false;
   b->cursor = nir_before_instr(&intr->instr);
   nir_def *offset = intr->src[store ? 1 : 0].ssa;
   nir_def *channels[NIR_MAX_VEC_COMPONENTS];
   for (unsigned c = 0; c < intr->num_components; ++c) {
      nir_def *address = nir_iadd_imm(b, offset, c);
      if (load) {
         channels[c] = nir_load_shared(b, 1, 8, address,
            .base = nir_intrinsic_base(intr), .align_mul = 1,
            .access = nir_intrinsic_access(intr));
      } else if (nir_intrinsic_write_mask(intr) & BITFIELD_BIT(c)) {
         nir_store_shared(b, nir_channel(b, intr->src[0].ssa, c), address,
            .base = nir_intrinsic_base(intr), .align_mul = 1,
            .write_mask = 1, .access = nir_intrinsic_access(intr));
      }
   }
   if (load)
      nir_def_rewrite_uses(&intr->def, nir_vec(b, channels, intr->num_components));
   nir_instr_remove(&intr->instr);
   return true;
}

/* Native G16 framebuffer reads name a constant sample mask. Within the
 * per-sample loop, the dynamic mask has at most one bit set. Select the
 * corresponding constant-mask read, keeping inactive samples unaccessed. */
static bool
lower_agx3_sample_tile_read(nir_builder *b, nir_intrinsic_instr *intr,
                           UNUSED void *data)
{
   if (intr->intrinsic != nir_intrinsic_load_local_pixel_agx ||
       nir_src_is_const(intr->src[0]))
      return false;

   b->cursor = nir_before_instr(&intr->instr);
   nir_def *mask = intr->src[0].ssa;
   nir_def *value = nir_undef(b, intr->def.num_components, intr->def.bit_size);
   /* G16's qualified raster sample counts are 1, 2 and 4. */
   for (unsigned sample = 0; sample < 4; ++sample) {
      nir_push_if(b, nir_ine_imm(b, nir_iand_imm(b, mask, 1u << sample), 0));
      nir_def *loaded = nir_load_local_pixel_agx(
         b, intr->def.num_components, intr->def.bit_size,
         nir_imm_intN_t(b, 1u << sample, mask->bit_size),
         .base = nir_intrinsic_base(intr), .format = nir_intrinsic_format(intr));
      nir_push_else(b, NULL);
      nir_pop_if(b, NULL);
      value = nir_if_phi(b, loaded, value);
   }
   nir_def_replace(&intr->def, value);
   return true;
}

/* AGX3 tile instructions transfer raw words. Packed HDR pixels occupy one
 * W32 word; their RGB arithmetic is performed on unpacked floating values. */
static bool
lower_agx3_packed_tile(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   bool load = intr->intrinsic == nir_intrinsic_load_local_pixel_agx;
   bool store = intr->intrinsic == nir_intrinsic_store_local_pixel_agx;
   if (!load && !store)
      return false;

   const enum pipe_format format = nir_intrinsic_format(intr);
   if (load && format == PIPE_FORMAT_R16_UINT && intr->def.bit_size == 32) {
      /* Native LLDCOL transfers tightly packed halfword lanes. AGX2 can
       * widen while loading, so its NIR may request W32 results from I16
       * storage. Load the physical lanes before zero-extending each one;
       * signed logical formats already extend in the tilebuffer lowering. */
      b->cursor = nir_before_instr(&intr->instr);
      nir_def *raw = nir_load_local_pixel_agx(
         b, intr->def.num_components, 16, intr->src[0].ssa,
         .base = nir_intrinsic_base(intr), .format = PIPE_FORMAT_R16_UINT);
      nir_def_replace(&intr->def, nir_u2u32(b, raw));
      return true;
   }
   if (format == PIPE_FORMAT_R16_UNORM || format == PIPE_FORMAT_R16_SNORM) {
      /* LSTCOL/LLDCOL carry a lane width, not AGX2's typed conversion.
       * Normalize explicitly, then transfer the raw 16-bit integer lanes. */
      const bool snorm = format == PIPE_FORMAT_R16_SNORM;
      const unsigned bits[4] = {16, 16, 16, 16};
      b->cursor = nir_before_instr(&intr->instr);
      if (load) {
         nir_def *raw = nir_load_local_pixel_agx(
            b, intr->def.num_components, 16, intr->src[0].ssa,
            .base = nir_intrinsic_base(intr), .format = PIPE_FORMAT_R16_UINT);
         raw = snorm ? nir_i2i32(b, raw) : nir_u2u32(b, raw);
         nir_def *value = snorm ? nir_format_snorm_to_float(b, raw, bits)
                               : nir_format_unorm_to_float(b, raw, bits);
         nir_def_rewrite_uses(&intr->def,
                              nir_f2fN(b, value, intr->def.bit_size));
      } else {
         nir_def *value = nir_f2f32(b, intr->src[0].ssa);
         nir_def *raw = snorm ? nir_format_float_to_snorm(b, value, bits)
                             : nir_format_float_to_unorm(b, value, bits);
         nir_store_local_pixel_agx(
            b, nir_u2u16(b, raw), intr->src[1].ssa, intr->src[2].ssa,
            .base = nir_intrinsic_base(intr), .format = PIPE_FORMAT_R16_UINT,
            .write_mask = nir_intrinsic_write_mask(intr),
            .explicit_coord = nir_intrinsic_explicit_coord(intr));
      }
      nir_instr_remove(&intr->instr);
      return true;
   }

   /* RGB9E5's fourth format channel describes the shared exponent. NIR
    * carries four color lanes, but only RGB is packed and alpha reads as 1. */
   const bool shared_exponent = format == PIPE_FORMAT_R9G9B9E5_FLOAT;
   if ((format != PIPE_FORMAT_R11G11B10_FLOAT && !shared_exponent) ||
       (load && (intr->def.bit_size != 32 ||
                 intr->def.num_components > (shared_exponent ? 4 : 3))) ||
       (store && (nir_intrinsic_explicit_coord(intr) ||
                  nir_intrinsic_write_mask(intr) != (shared_exponent ? 15 : 7))))
      return false;

   b->cursor = nir_before_instr(&intr->instr);
   if (load) {
      nir_def *packed = nir_load_local_pixel_agx(
         b, 1, 32, intr->src[0].ssa, .base = nir_intrinsic_base(intr),
         .format = PIPE_FORMAT_R32_UINT);
      nir_def *rgb = shared_exponent ? nir_format_unpack_r9g9b9e5(b, packed)
                                     : nir_format_unpack_11f11f10f(b, packed);
      if (intr->def.num_components == 4)
         rgb = nir_vec4(b, nir_channel(b, rgb, 0), nir_channel(b, rgb, 1),
                        nir_channel(b, rgb, 2), nir_imm_float(b, 1.0));
      nir_def_rewrite_uses(
         &intr->def, nir_trim_vector(b, rgb, intr->def.num_components));
   } else {
      nir_def *packed = shared_exponent
         ? nir_format_pack_r9g9b9e5(b, intr->src[0].ssa)
         : nir_format_pack_11f11f10f(b, intr->src[0].ssa);
      nir_store_local_pixel_agx(
         b, packed, intr->src[1].ssa, intr->src[2].ssa,
         .base = nir_intrinsic_base(intr), .format = PIPE_FORMAT_R32_UINT,
         .write_mask = 1);
   }
   nir_instr_remove(&intr->instr);
   return true;
}

bool
agx_nir_lower_agx3_tile_formats(nir_shader *nir)
{
   return nir_shader_intrinsics_pass(nir, lower_agx3_packed_tile,
                                     nir_metadata_control_flow, NULL);
}

/* AGX2 has a Morton interleave instruction. AGX3 uses ordinary integer
 * arithmetic: spread each unsigned 16-bit coordinate into the even bits,
 * then place the second coordinate in the odd bits. */
static bool
lower_agx3_interleave(nir_builder *b, nir_alu_instr *alu, UNUSED void *data)
{
   if (alu->op != nir_op_interleave)
      return false;

   b->cursor = nir_before_instr(&alu->instr);
   static const unsigned shifts[] = {8, 4, 2, 1};
   static const uint32_t masks[] = {
      0x00ff00ff, 0x0f0f0f0f, 0x33333333, 0x55555555,
   };
   nir_def *spread[2];
   for (unsigned c = 0; c < 2; ++c) {
      nir_def *x = nir_u2u32(b, nir_ssa_for_alu_src(b, alu, c));
      for (unsigned i = 0; i < ARRAY_SIZE(shifts); ++i)
         x = nir_iand_imm(b, nir_ior(b, x, nir_ishl_imm(b, x, shifts[i])),
                          masks[i]);
      spread[c] = x;
   }
   nir_def_replace(&alu->def,
      nir_ior(b, spread[0], nir_ishl_imm(b, spread[1], 1)));
   return true;
}

bool
agx_nir_lower_interleave(nir_shader *nir)
{
   return nir_shader_alu_pass(nir, lower_agx3_interleave,
                              nir_metadata_control_flow, NULL);
}

/* Exported ABI registers are consumed by the next shader part at exit, not
 * at the source-level store. Late address lowering may split an originally
 * terminal output sequence with control flow. Materialize the final values
 * through SSA so conditional/repeated writes also keep their execution masks.
 */
static bool
agx_nir_export_at_exit(nir_shader *nir)
{
   bool progress = false;
   nir_foreach_function_impl(impl, nir) {
      nir_block *exit = nir_impl_last_block(impl);
      bool needs_lowering = false;
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type == nir_instr_type_intrinsic &&
                nir_instr_as_intrinsic(instr)->intrinsic == nir_intrinsic_export_agx)
               needs_lowering |= block != exit;
         }
      }
      if (!needs_lowering)
         continue;

      nir_variable *values[AGX_NUM_REGS] = {0};
      nir_builder b = nir_builder_create(impl);
      nir_foreach_block(block, impl) {
         nir_foreach_instr_safe(instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;
            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            if (intr->intrinsic != nir_intrinsic_export_agx)
               continue;
            b.cursor = nir_before_instr(instr);
            nir_def *src = intr->src[0].ssa;
            for (unsigned c = 0; c < src->num_components; ++c) {
               unsigned reg = nir_intrinsic_base(intr) + c * (src->bit_size / 16);
               assert(reg < ARRAY_SIZE(values));
               if (!values[reg])
                  values[reg] = nir_local_variable_create(
                     impl, glsl_uintN_t_type(src->bit_size), "export_value");
               assert(glsl_get_bit_size(values[reg]->type) == src->bit_size);
               nir_store_var(&b, values[reg], nir_channel(&b, src, c), 1);
            }
            nir_instr_remove(instr);
         }
      }
      b.cursor = nir_after_impl(impl);
      for (unsigned reg = 0; reg < ARRAY_SIZE(values); ++reg) {
         if (values[reg])
            nir_export_agx(&b, nir_load_var(&b, values[reg]), .base = reg);
      }
      nir_progress(true, impl, nir_metadata_control_flow);
      progress = true;
   }
   if (progress) {
      nir_lower_vars_to_ssa(nir);
      nir_opt_dce(nir);
      nir_remove_dead_variables(nir, nir_var_function_temp, NULL);
   }
   return progress;
}

static bool
lower_prefetched_uniform(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   if (intr->intrinsic != nir_intrinsic_load_prefetched_agx)
      return false;

   nir_def_replace(&intr->def, intr->src[1].ssa);
   return true;
}

bool
agx_native_late_loop_enabled(void)
{
   return debug_get_bool_option("AGX_NATIVE_LATE_LOOP", false);
}

bool
agx_native_fragment_loads_enabled(void)
{
   return debug_get_bool_option("AGX_NATIVE_FRAGMENT_LOADS", false);
}

bool
agx_native_early_compare_enabled(void)
{
   return debug_get_bool_option("AGX_NATIVE_EARLY_COMPARE", false);
}

bool
agx_native_parallel_lod_enabled(void)
{
   return debug_get_bool_option("AGX_NATIVE_PARALLEL_LOD", false);
}

bool
agx_native_early_textures_enabled(void)
{
   return debug_get_bool_option("AGX_NATIVE_EARLY_TEXTURES", false);
}

static bool
agx_compile_shader_nir_inner(nir_shader *nir, struct agx_shader_key *key,
                       struct agx_shader_part *out)
{
   /* Publish flags once, rather than racing identical global writes from
    * concurrent pipeline compilation workers. */
   agx_get_compiler_debug();
   struct agx_shader_info *info = &out->info;

   /* Address and resource lowering can expose a constant initial loop
    * predicate after the frontend's loop pass. Use NIR's existing peeling
    * transform to move that first predicate out of the loop. Run once:
    * repeatedly peeling arbitrary loops is neither necessary nor bounded.
    * This remains an opt-in until hardware control-flow and application
    * regression checks qualify the changed instruction schedule. */
   if (key->dev.encoding == AGX_ISA_ENCODING_AGX3B &&
       agx_native_late_loop_enabled() && nir_opt_loop(nir)) {
      NIR_PASS(_, nir, nir_lower_vars_to_ssa);
      bool progress;
      do {
         progress = false;
         NIR_PASS(progress, nir, nir_opt_copy_prop);
         NIR_PASS(progress, nir, nir_opt_cse);
         NIR_PASS(progress, nir, nir_opt_dce);
         NIR_PASS(progress, nir, nir_opt_constant_folding);
         NIR_PASS(progress, nir, nir_opt_dead_cf);
      } while (progress);
   }

   /* Native fragment shaders have no constant program yet. Descriptor and
    * sampler emulation therefore leaves substantial repeated work across
    * blocks. GCM shares that work before backend lowering expands addresses
    * and filtering arithmetic. Keep an override for bounded A/B diagnosis. */
   if (key->dev.encoding == AGX_ISA_ENCODING_AGX3B) {
      const char *motion = debug_get_option(
         "AGX_CODE_MOTION", nir->info.stage == MESA_SHADER_FRAGMENT ? "gcm" : "none");
      if (!strcmp(motion, "gcm") || !strcmp(motion, "gcm-no-vn")) {
         NIR_PASS(_, nir, nir_opt_gcm, !strcmp(motion, "gcm"));
      } else if (!strcmp(motion, "licm")) {
         NIR_PASS(_, nir, nir_opt_licm);
         NIR_PASS(_, nir, nir_opt_cse);
         NIR_PASS(_, nir, nir_opt_dce);
      }
   }

   struct util_dynarray binary = UTIL_DYNARRAY_INIT;

   memset(out, 0, sizeof *out);

   /* Record API/source arithmetic before backend lowering introduces its own
    * 64-bit address and invocation-ID calculations. */
   const bool source_imul64 = agx_nir_has_alu64(nir, nir_op_imul) ||
                              agx_nir_has_alu64(nir, nir_op_umul_2x32_64) ||
                              agx_nir_has_alu64(nir, nir_op_imul_2x32_64);

   if ((key->dev.encoding == AGX_ISA_ENCODING_AGX3B ||
        key->dev.encoding == AGX_ISA_ENCODING_AGX3C) &&
       has_unlowered_printf_address(nir)) {
      fprintf(stderr, "AGX3 preflight: unlowered runtime printf buffer address\n");
      return false;
   }

   assert(nir->info.io_lowered &&
          "agx_preprocess_nir is called first, then the shader is specalized,"
          "then the specialized shader is compiled");

   /* The AGX3 backend has byte-grounded traversal/SCS codecs, but no complete
    * NIR ray-query lowering yet.  Reject before instruction selection reaches
    * its fatal unhandled-intrinsic path, and keep the output transactional.
    * This gate is removed only together with the full lowering and scheduling
    * contract; it does not prevent direct internal encoder tests. */
   if (nir->info.ray_queries != 0 &&
       (key->dev.encoding == AGX_ISA_ENCODING_AGX3B ||
        key->dev.encoding == AGX_ISA_ENCODING_AGX3C)) {
      fprintf(stderr, "AGX3 preflight: NIR ray-query lowering is unavailable\n");
      return false;
   }

   /* If required, tag writes will be enabled by instruction selection */
   if (nir->info.stage == MESA_SHADER_FRAGMENT)
      info->tag_write_disable = !nir->info.writes_memory;

   NIR_PASS(_, nir, nir_shader_intrinsics_pass, lower_printf_buffer,
            nir_metadata_control_flow, &key->dev);

   if (key->dev.encoding == AGX_ISA_ENCODING_AGX3B) {
      NIR_PASS(_, nir, nir_shader_intrinsics_pass, lower_agx3_sample_tile_read,
               nir_metadata_none, NULL);
      NIR_PASS(_, nir, agx_nir_lower_agx3_tile_formats);
   }
   if (key->dev.encoding == AGX_ISA_ENCODING_AGX3B)
      NIR_PASS(_, nir, agx_nir_lower_interleave);

   NIR_PASS(_, nir, nir_lower_frag_coord_to_pixel_coord);
   NIR_PASS(_, nir, nir_lower_vars_to_ssa);

   /* Late sysval lowering creates large loads. Load lowering creates unpacks */
   nir_lower_mem_access_bit_sizes_options lower_mem_access_options = {
      .modes = nir_var_mem_ssbo | nir_var_mem_constant |
               nir_var_mem_task_payload | nir_var_shader_temp |
               nir_var_function_temp | nir_var_mem_global | nir_var_mem_shared,
      .callback = mem_access_size_align_cb,
   };
   NIR_PASS(_, nir, nir_lower_mem_access_bit_sizes, &lower_mem_access_options);

   /* Optimize scratch access for silly internal CL shaders */
   if (nir->info.internal) {
      NIR_PASS(_, nir, nir_lower_scratch_to_var);
      NIR_PASS(_, nir, nir_lower_vars_to_scratch, 256,
               glsl_get_natural_size_align_bytes,
               glsl_get_natural_size_align_bytes);
      NIR_PASS(_, nir, nir_lower_indirect_derefs_to_if_else_trees,
               nir_var_function_temp, ~0);
   }

   if (key->dev.encoding == AGX_ISA_ENCODING_AGX3B)
      NIR_PASS(_, nir, nir_shader_alu_pass, agx_nir_unpack_signed_bytes,
               nir_metadata_control_flow, NULL);

   /* Cleanup 8-bit math before lowering */
   bool progress;
   do {
      progress = false;

      NIR_PASS(progress, nir, nir_opt_algebraic);
      NIR_PASS(progress, nir, nir_opt_constant_folding);
      NIR_PASS(progress, nir, nir_opt_dce);
   } while (progress);

   NIR_PASS(_, nir, nir_lower_bit_size, lower_bit_size_callback, NULL);

   /* Late blend lowering creates vectors */
   NIR_PASS(_, nir, nir_lower_alu_to_scalar, NULL, NULL);
   NIR_PASS(_, nir, nir_lower_load_const_to_scalar);

   /* Late VBO lowering creates constant udiv instructions */
   NIR_PASS(_, nir, nir_opt_idiv_const, 16);

   NIR_PASS(_, nir, nir_opt_constant_folding);
   NIR_PASS(_, nir, nir_shader_intrinsics_pass, lower_load_from_texture_handle,
            nir_metadata_control_flow, NULL);

   NIR_PASS(_, nir, nir_shader_intrinsics_pass, lower_sink_address,
            nir_metadata_control_flow, NULL);

   info->push_count = key->reserved_preamble;
   const bool form_preamble = !key->secondary && !key->no_preamble;
   const bool is_agx3 = key->dev.encoding == AGX_ISA_ENCODING_AGX3B ||
                       key->dev.encoding == AGX_ISA_ENCODING_AGX3C;
   const bool is_g17p = key->dev.encoding == AGX_ISA_ENCODING_AGX3B;

   /* AGX2 has native 64-bit add and 32x32->64 multiply forms, so the shared
    * compiler options preserve them. AGX3B has grounded wide multiplication and subtraction,
    * but still decomposes 64x64 products. Keep the override local to this
    * compile so the other encodings retain their paths and frontends continue
    * sharing agx_nir_options. */
   const nir_shader_compiler_options *original_options = nir->options;
   nir_shader_compiler_options agx3_options;
   if (is_g17p) {
      agx3_options = *original_options;
      agx3_options.lower_iabs = false;
      agx3_options.lower_iadd_sat = true;
      agx3_options.lower_uadd_sat = true;
      agx3_options.lower_usub_sat = true;
      agx3_options.lower_int64_options &= ~nir_lower_iadd64;
      if (source_imul64)
         agx3_options.lower_int64_options &= ~nir_lower_imul64;
      nir->options = &agx3_options;
   }

   agx_optimize_nir(
      nir, key->dev.soft_fault, !is_agx3,
      key->dev.encoding == AGX_ISA_ENCODING_AGX3B,
      form_preamble ? &info->push_count : NULL,
      (form_preamble && key->promote_textures) ? &info->texture_state_count
                                              : NULL,
      (form_preamble && key->promote_textures) ? &info->sampler_state_count
                                              : NULL);

   /* Preserve the original read's convergence and arithmetic decisions until
    * optimization is complete. The driver qualifies either an input-loader
    * word or an eligibility-controlled memory fallback as its replacement.
    * Only clean up dead address work afterwards; do not reassociate floating-
    * point values using the replacement's potentially different convergence. */
   bool prefetched = false;
   NIR_PASS(prefetched, nir, nir_shader_intrinsics_pass, lower_prefetched_uniform,
            nir_metadata_control_flow, NULL);
   if (prefetched) {
      bool cleanup;
      do {
         cleanup = false;
         NIR_PASS(cleanup, nir, nir_opt_dce);
         NIR_PASS(cleanup, nir, nir_opt_dead_cf);
      } while (cleanup);
   }

   if (is_g17p)
      NIR_PASS(_, nir, nir_shader_intrinsics_pass,
               lower_agx3_shared_byte_vectors, nir_metadata_control_flow, NULL);

   if (is_g17p)
      NIR_PASS(_, nir, agx_nir_lower_imul64, source_imul64);

   if (is_g17p)
      nir->options = original_options;

   NIR_PASS(_, nir, agx_nir_export_at_exit);

   /* Register-cache discards need execution divergence, not simply syntactic
    * nesting: descriptor and preamble-controlled branches can be uniform. */
   if (key->dev.encoding == AGX_ISA_ENCODING_AGX3B)
      nir_divergence_analysis(nir);

   if (nir->info.stage == MESA_SHADER_FRAGMENT) {
      info->varyings.fs.nr_cf = key->fs.cf_base;
      assign_coefficient_regs(nir, &info->varyings.fs);
   }

   if (agx_should_dump(nir, AGX_DBG_SHADERS))
      nir_print_shader(nir, stdout);

   info->local_size = nir->info.shared_size;
   info->uses_resource_info_query = nir->info.uses_resource_info_query;
   info->uses_ray_traversal = nir->info.ray_queries != 0;
   /* NIR ray_queries counts inline intersection_query variables, matching the
    * G17S next()/committed witness whose ShaderInfo ID32 is nonzero.  Future
    * one-shot/full traversal lowerings must set uses_ray_traversal without
    * automatically setting this committed-path property. */
   info->ray_traversal_depth_nonzero = nir->info.ray_queries != 0;

   nir_foreach_function_with_impl(func, impl, nir) {
      unsigned offset;
      if (!agx_compile_function_nir(nir, impl, key, &binary, &out->info,
                                    &offset)) {
         util_dynarray_fini(&binary);
         memset(out, 0, sizeof(*out));
         return false;
      }

      if (func->is_preamble) {
         info->preamble_offset = offset;
         info->has_preamble = true;
      } else if (func->is_entrypoint) {
         info->main_offset = offset;
         info->main_size = binary.size - offset;
      } else {
         UNREACHABLE("General functions not yet supported");
      }
   }

   if (agx_should_dump(nir, AGX_DBG_SHADERDB)) {
      agx2_stats_fprintf(stderr, _mesa_shader_stage_to_abbrev(nir->info.stage),
                         &out->info.stats);
   }

   info->stage = nir->info.stage;

   /* Check these outside the stage check since nir->info.stage is the hardware
    * stage and these are read in the vertex *software* stage.
    */
   info->uses_draw_id =
      BITSET_TEST(nir->info.system_values_read, SYSTEM_VALUE_DRAW_ID);

   info->uses_base_param =
      BITSET_TEST(nir->info.system_values_read, SYSTEM_VALUE_FIRST_VERTEX) ||
      BITSET_TEST(nir->info.system_values_read, SYSTEM_VALUE_BASE_VERTEX) ||
      BITSET_TEST(nir->info.system_values_read, SYSTEM_VALUE_BASE_INSTANCE);

   if (nir->info.stage == MESA_SHADER_VERTEX ||
       nir->info.stage == MESA_SHADER_TESS_EVAL) {
      info->nonzero_viewport = nir->info.outputs_written & VARYING_BIT_VIEWPORT;

      info->writes_layer_viewport =
         nir->info.outputs_written & (VARYING_BIT_LAYER | VARYING_BIT_VIEWPORT);

   } else if (nir->info.stage == MESA_SHADER_FRAGMENT) {
      info->disable_tri_merging =
         nir->info.uses_wide_subgroup_intrinsics ||
         nir->info.fs.needs_coarse_quad_helper_invocations ||
         nir->info.writes_memory;

      /* Writing the sample mask requires tag writes */
      info->tag_write_disable &= !info->writes_sample_mask;

      /* Report a canonical depth layout. This happens at the end because the
       * sample mask lowering affects it.
       */
      enum gl_frag_depth_layout layout = nir->info.fs.depth_layout;

      if (!(nir->info.outputs_written & BITFIELD64_BIT(FRAG_RESULT_DEPTH)))
         info->depth_layout = FRAG_DEPTH_LAYOUT_UNCHANGED;
      else if (layout == FRAG_DEPTH_LAYOUT_NONE)
         info->depth_layout = FRAG_DEPTH_LAYOUT_ANY;
      else
         info->depth_layout = layout;

      /* Interlocked storage requires a dependent pass even when it does
       * not read the tilebuffer. The linker also uses this dependency to
       * retain ownership across a per-sample invocation loop. */
      info->reads_tib = nir->info.fs.uses_fbfetch_output ||
         (key->dev.encoding == AGX_ISA_ENCODING_AGX3B &&
          (nir->info.fs.pixel_interlock_ordered ||
           nir->info.fs.pixel_interlock_unordered ||
           nir->info.fs.sample_interlock_ordered ||
           nir->info.fs.sample_interlock_unordered));
      info->early_fragment_tests = nir->info.fs.early_fragment_tests;
   } else if (nir->info.stage == MESA_SHADER_COMPUTE) {
      info->imageblock_stride = nir->info.cs.image_block_size_per_thread_agx;

      for (unsigned i = 0; i < 3; ++i) {
         info->workgroup_size[i] = nir->info.workgroup_size[i];
      }
   }

   out->binary = binary.data;
   info->binary_size = binary.size;
   return true;
}

/* Development-only, same-serialization-ABI reproducer. The caller creates a
 * private directory; no process environment or application credentials are
 * copied. Publication is atomic, deduplicated, and capped at 32 bundles (8 MiB input and 8 MiB IR each).
 * Capture errors must never alter the compiler result. */
static void
agx_publish_capture(const char *directory, struct blob *input, FILE *ir, int success)
{
   if (input->size > 8 * 1024 * 1024)
      return;
   char lockpath[PATH_MAX], final[PATH_MAX], temporary[PATH_MAX];
   if (snprintf(lockpath, sizeof(lockpath), "%s/.lock", directory) >= sizeof(lockpath))
      return;
   int lock = open(lockpath, O_CREAT | O_RDWR | O_CLOEXEC | O_NOFOLLOW, 0600);
   if (lock < 0)
      return;
   if (flock(lock, LOCK_EX | LOCK_NB)) {
      close(lock);
      return;
   }
   DIR *dir = opendir(directory);
   unsigned count = 0;
   if (!dir) {
      close(lock);
      return;
   }
   struct dirent *entry;
   while ((entry = readdir(dir)))
      count += strcmp(entry->d_name, ".") && strcmp(entry->d_name, "..") &&
               strcmp(entry->d_name, ".lock");
   closedir(dir);
   if (count >= 32) {
      close(lock);
      return;
   }
   blake3_hash hash;
   char hex[65];
   _mesa_blake3_compute(input->data, input->size, hash);
   _mesa_blake3_format(hex, hash);
   if (snprintf(final, sizeof(final), "%s/%s", directory, hex) >= sizeof(final) ||
       snprintf(temporary, sizeof(temporary), "%s/.pending-XXXXXX", directory) >= sizeof(temporary) ||
       access(final, F_OK) == 0 || !mkdtemp(temporary)) {
      close(lock);
      return;
   }
   char path[PATH_MAX];
   bool ok = false;
   snprintf(path, sizeof(path), "%s/input.bin", temporary);
   FILE *f = fopen(path, "wb");
   if (f) {
      ok = fwrite(input->data, 1, input->size, f) == input->size;
      ok &= fflush(f) == 0;
      ok &= fsync(fileno(f)) == 0;
      ok &= fclose(f) == 0;
   }
   if (ok && ir) {
      fflush(ir);
      long length = ftell(ir);
      snprintf(path, sizeof(path), "%s/backend-ir.txt", temporary);
      f = fopen(path, "wb");
      if (f && length >= 0 && length < 8 * 1024 * 1024) {
         rewind(ir);
         char buffer[4096];
         size_t n;
         while ((n = fread(buffer, 1, sizeof(buffer), ir)))
            if (fwrite(buffer, 1, n, f) != n) break;
      }
      if (f) { fflush(f); fsync(fileno(f)); fclose(f); }
   }
   if (ok) {
      snprintf(path, sizeof(path), "%s/manifest.json", temporary);
      f = fopen(path, "w");
      if (f) {
         fprintf(f, "{\"version\":1,\"serialization_abi\":\"%s\","
                    "\"input_blake3\":\"%s\",\"input_bytes\":%zu,"
                    "\"entrypoint\":\"agx_compile_shader_nir\",\"success\":%s}\n",
                 AGX_REPLAY_ABI, hex, input->size,
                 success < 0 ? "null" : success ? "true" : "false");
         fflush(f); fsync(fileno(f)); fclose(f);
      }
      if (f && rename(temporary, final) == 0)
         fprintf(stderr, "AGX compiler %s saved: %s\n",
                 success < 0 ? "pending input" : success ? "input" : "failure", final);
   }
   /* On publication failure do not leave partial evidence consuming quota. */
   const char *names[] = {"input.bin", "backend-ir.txt", "manifest.json"};
   for (unsigned i = 0; i < ARRAY_SIZE(names); ++i) {
      snprintf(path, sizeof(path), "%s/%s", temporary, names[i]);
      unlink(path);
   }
   rmdir(temporary);
   int directory_fd = open(directory, O_RDONLY | O_DIRECTORY | O_CLOEXEC);
   if (directory_fd >= 0) { fsync(directory_fd); close(directory_fd); }
   close(lock);
}

bool
agx_compile_shader_nir(nir_shader *nir, struct agx_shader_key *key,
                       struct agx_shader_part *out)
{
   const char *directory = getenv("AGX_COMPILER_CAPTURE_DIR");
   const char *capture_success = getenv("AGX_COMPILER_CAPTURE_SUCCESS");
   bool save_success = capture_success &&
      (!strcmp(capture_success, "all") ||
       (!strcmp(capture_success, "vertex") && nir->info.stage == MESA_SHADER_VERTEX) ||
       (!strcmp(capture_success, "fragment") && nir->info.stage == MESA_SHADER_FRAGMENT) ||
       (!strcmp(capture_success, "compute") && nir->info.stage == MESA_SHADER_COMPUTE));
   /* Keep browser startup shaders from exhausting the bounded capture quota
    * before the large shader being profiled is compiled. Failures are still
    * captured regardless of this successful-capture filter. */
   if (save_success) {
      const char *source_hash = getenv("AGX_COMPILER_CAPTURE_SOURCE_HASH");
      if (source_hash) {
         char hex[65];
         _mesa_blake3_format(hex, nir->info.source_blake3);
         save_success = !strcmp(hex, source_hash);
      }
      unsigned minimum = debug_get_num_option("AGX_COMPILER_CAPTURE_MIN_INSTRUCTIONS", 0);
      unsigned instructions = 0;
      if (minimum) {
         nir_foreach_function_impl(impl, nir)
            nir_foreach_block(block, impl)
               nir_foreach_instr(instr, block)
                  ++instructions;
         save_success &= instructions >= minimum;
      }
   }
   struct blob input;
   blob_init(&input);
   FILE *previous = agx_failure_ir;
   FILE *ir = NULL;
   /* The initial consumer uses the standard Asahi options. Reject capture
    * of other options (including callbacks) rather than serializing pointers. */
   if (directory && !memcmp(nir->options, &agx_nir_options, sizeof(agx_nir_options))) {
      blob_write_bytes(&input, "AGXREPLAYv1", 12);
      blob_write_bytes(&input, AGX_REPLAY_ABI, 65);
      blob_write_uint32(&input, sizeof(*key));
      blob_write_bytes(&input, key, sizeof(*key));
      nir_serialize(&input, nir, false);
      ir = tmpfile();
      agx_failure_ir = ir;
   }
   /* Opt-in crash diagnosis: publish the input before an assertion or abort
    * can prevent normal failure capture. Keep it separate from completed
    * results and explicitly record an unknown outcome. */
   if (getenv("AGX_COMPILER_CAPTURE_BEFORE") && input.size &&
       !input.out_of_memory) {
      char pending[PATH_MAX];
      if (snprintf(pending, sizeof(pending), "%s/before", directory) < sizeof(pending) &&
          (mkdir(pending, 0700) == 0 || errno == EEXIST))
         agx_publish_capture(pending, &input, NULL, -1);
   }
   bool result = agx_compile_shader_nir_inner(nir, key, out);
   agx_failure_ir = previous;
   /* Select one executable observed by HK_M4_SHADER_PROFILE without filling
    * the capture quota with unrelated startup variants. This only filters
    * successful diagnostics; failures are always retained as before. */
   const char *capture_hash = getenv("AGX_COMPILER_CAPTURE_BINARY_HASH");
   if (result && save_success && capture_hash) {
      char hash[9];
      snprintf(hash, sizeof(hash), "%08x",
               _mesa_hash_data(out->binary, out->info.binary_size));
      save_success = false;
      for (const char *p = capture_hash; p; ) {
         if (!strncmp(hash, p, 8) && (p[8] == '\0' || p[8] == ','))
            save_success = true;
         p = strchr(p, ',');
         if (p)
            ++p;
      }
   }
   if (result && save_success) {
      unsigned size = debug_get_num_option("AGX_COMPILER_CAPTURE_BINARY_SIZE", 0);
      if (size && out->info.binary_size != size)
         save_success = false;
   }
   if ((!result || save_success) && input.size && !input.out_of_memory)
      agx_publish_capture(directory, &input, ir, result);
   if (ir) fclose(ir);
   blob_finish(&input);
   return result;
}
