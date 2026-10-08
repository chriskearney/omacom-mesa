/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include "hk_agx3_ioto_program_core.h"

#include "asahi/compiler/agx3_binder.h"
#include "asahi/lib/agx_ioto_core.h"
#include "asahi/lib/agx_tilebuffer.h"

#include <stdlib.h>
#include <string.h>
#include "util/format/u_format.h"

bool
hk_agx3_ioto_program_generate(enum agx_chip chip,
                              const struct hk_agx3_ioto_program_desc *desc,
                              void *output, size_t capacity, size_t *written)
{
   if (!desc || !agx_ioto_target_supported(chip))
      return false;

   const struct agx_ioto_profile_control profile = {
      .chip = chip,
      .payload0 = desc->profile_payload0,
      .payload1 = desc->profile_payload1,
   };

   return agx_ioto_straight_line_program_compose(
      chip, desc->program_va, desc->body, desc->body_size, &desc->tail,
      &profile, output, capacity, written);
}

bool
hk_agx3_ioto_program_generate_no_pixrel(
   enum agx_chip chip, const struct hk_agx3_ioto_program_desc *desc,
   const struct agx_ioto_preshader_desc *preshader, void *output,
   size_t capacity, size_t *written)
{
   if (!desc || !agx_ioto_target_supported(chip))
      return false;

   const struct agx_ioto_profile_control profile = {
      .chip = chip,
      .payload0 = desc->profile_payload0,
      .payload1 = desc->profile_payload1,
   };

   return agx_ioto_program_compose_no_pixrel(
      chip, desc->program_va, desc->body, desc->body_size, &desc->tail,
      &profile, preshader, output, capacity, written);
}

bool
hk_agx3_g17p_no_resource_program_generate(
   enum agx_chip chip, uint64_t program_va,
   const struct agx_ioto_state_tail_output *packed_tail, void *output,
   size_t capacity, size_t *written)
{
   if (!agx_chip_uses_hal200_ioto(chip))
      return false;

   return agx_ioto_g17p_no_resource_program_compose(program_va, packed_tail,
                                                    output, capacity, written);
}

bool
hk_agx3_g17p_single_layer_rendering(uint32_t view_mask, uint32_t layer_count)
{
   return view_mask == 0 && layer_count == 1;
}

static bool
hk_native_single_target_layout_supported(struct agx_tilebuffer_layout *tib)
{
   if (!tib ||
       (tib->nr_samples != 1 && tib->nr_samples != 2 && tib->nr_samples != 4) ||
       (tib->tile_size != 32 * 32 && tib->tile_size != 32 * 16 &&
        tib->tile_size != 16 * 16) ||
       tib->spilled[0])
      return false;

   for (unsigned rt = 1; rt < AGX_MAX_RENDER_TARGETS; ++rt) {
      if (tib->logical_format[rt] != PIPE_FORMAT_NONE)
         return false;
   }

   /* Depth-only and attachmentless passes have no color tile storage. */
   if (tib->logical_format[0] == PIPE_FORMAT_NONE)
      return tib->sample_size_B == 0;

   const bool scalar_r32f = tib->logical_format[0] == PIPE_FORMAT_R32_FLOAT;
   const bool scalar_r32i = tib->logical_format[0] == PIPE_FORMAT_R32_SINT;
   const bool pair_r32f = tib->logical_format[0] == PIPE_FORMAT_R32G32_FLOAT;
   const bool pair_r32i = tib->logical_format[0] == PIPE_FORMAT_R32G32_SINT;
   const bool pair_r32u = tib->logical_format[0] == PIPE_FORMAT_R32G32_UINT;
   const bool scalar_r8u = tib->logical_format[0] == PIPE_FORMAT_R8_UINT;
   const bool scalar_r8i = tib->logical_format[0] == PIPE_FORMAT_R8_SINT;
   const bool rgba16u =
      tib->logical_format[0] == PIPE_FORMAT_R16G16B16A16_UINT;
   const bool rgba16i =
      tib->logical_format[0] == PIPE_FORMAT_R16G16B16A16_SINT;

   const bool rgba32 =
      tib->sample_size_B == 16 &&
      (tib->logical_format[0] == PIPE_FORMAT_R32G32B32A32_FLOAT ||
       tib->logical_format[0] == PIPE_FORMAT_R32G32B32A32_SINT ||
       tib->logical_format[0] == PIPE_FORMAT_R32G32B32A32_UINT);

   return rgba32 ||
          (tib->sample_size_B == 8 &&
           (agx_tilebuffer_physical_format(tib, 0) == PIPE_FORMAT_R8_UNORM ||
            agx_tilebuffer_physical_format(tib, 0) ==
               PIPE_FORMAT_R8G8B8A8_SRGB ||
            agx_tilebuffer_physical_format(tib, 0) == PIPE_FORMAT_R16_FLOAT ||
            agx_tilebuffer_physical_format(tib, 0) ==
               PIPE_FORMAT_R10G10B10A2_UNORM ||
            tib->logical_format[0] == PIPE_FORMAT_R16G16_UNORM ||
            tib->logical_format[0] == PIPE_FORMAT_R16G16_SNORM ||
            tib->logical_format[0] == PIPE_FORMAT_R32_UINT ||
            tib->logical_format[0] == PIPE_FORMAT_R8_SNORM ||
            tib->logical_format[0] == PIPE_FORMAT_R8G8_SNORM ||
            tib->logical_format[0] == PIPE_FORMAT_R8G8_UINT ||
            tib->logical_format[0] == PIPE_FORMAT_R8G8_SINT ||
            tib->logical_format[0] == PIPE_FORMAT_R8G8B8A8_SNORM ||
            tib->logical_format[0] == PIPE_FORMAT_B8G8R8A8_SNORM ||
            tib->logical_format[0] == PIPE_FORMAT_R8G8B8A8_UINT ||
            tib->logical_format[0] == PIPE_FORMAT_B8G8R8A8_UINT ||
            tib->logical_format[0] == PIPE_FORMAT_R8G8B8A8_SINT ||
            tib->logical_format[0] == PIPE_FORMAT_B8G8R8A8_SINT ||
            tib->logical_format[0] == PIPE_FORMAT_R10G10B10A2_UINT ||
            tib->logical_format[0] == PIPE_FORMAT_B10G10R10A2_UINT ||
            tib->logical_format[0] == PIPE_FORMAT_R16_UINT ||
            tib->logical_format[0] == PIPE_FORMAT_R16_SINT ||
            tib->logical_format[0] == PIPE_FORMAT_R16_UNORM ||
            tib->logical_format[0] == PIPE_FORMAT_R16_SNORM ||
            tib->logical_format[0] == PIPE_FORMAT_R16G16B16A16_SNORM ||
            tib->logical_format[0] == PIPE_FORMAT_R16G16_UINT ||
            tib->logical_format[0] == PIPE_FORMAT_R16G16_SINT ||
            tib->logical_format[0] == PIPE_FORMAT_R16G16B16A16_UNORM ||
            tib->logical_format[0] == PIPE_FORMAT_R11G11B10_FLOAT ||
            tib->logical_format[0] == PIPE_FORMAT_R9G9B9E5_FLOAT ||
            scalar_r32f || scalar_r32i || pair_r32f || pair_r32i || pair_r32u ||
            scalar_r8u || scalar_r8i || rgba16u || rgba16i));
}

bool
hk_agx3_g17p_rt0_tile_layout_supported(struct agx_tilebuffer_layout *tib)
{
   if (!tib ||
       (tib->nr_samples != 1 && tib->nr_samples != 2 && tib->nr_samples != 4) ||
       (tib->tile_size != 1024 && tib->tile_size != 512 &&
        tib->tile_size != 256) ||
       (tib->sample_size_B & 7) || tib->sample_size_B > 248 ||
       tib->sample_size_B * tib->nr_samples * tib->tile_size >= 32768)
      return false;
   unsigned begin[AGX_MAX_RENDER_TARGETS], end[AGX_MAX_RENDER_TARGETS],
      count = 0;
   for (unsigned rt = 0; rt < AGX_MAX_RENDER_TARGETS; ++rt) {
      const enum pipe_format logical = tib->logical_format[rt];
      if (logical == PIPE_FORMAT_NONE)
         continue;
      /* BG/EOT compact active bindings while preserving these tile offsets. */
      if (tib->spilled[rt])
         continue;
      const enum pipe_format physical = agx_tilebuffer_physical_format(tib, rt);
      const unsigned alignment = util_format_get_blocksize(physical);
      unsigned bytes = alignment;
      if (util_format_get_nr_components(physical) == 1)
         bytes *= util_format_get_nr_components(logical);
      struct agx_tilebuffer_layout single = {
         .logical_format = {logical},
         .nr_samples = tib->nr_samples,
         .sample_size_B = ALIGN_POT(bytes, 8),
         .tile_size = tib->tile_size,
         .layered = tib->layered,
      };
      if (!bytes || !alignment ||
          !hk_native_single_target_layout_supported(&single) ||
          tib->_offset_B[rt] % alignment)
         return false;
      begin[count] = tib->_offset_B[rt];
      end[count] = begin[count] + bytes;
      if (end[count] > tib->sample_size_B)
         return false;
      for (unsigned j = 0; j < count; ++j)
         if (begin[count] < end[j] && begin[j] < end[count])
            return false;
      count++;
   }
   /* MSAA reserves an eight-byte sample stride even without color targets. */
   return count || tib->sample_size_B == (tib->nr_samples > 1 ? 8 : 0);
}

static bool
hk_agx3_g17p_graphics_state_tail(
   const struct hk_agx3_g17p_graphics_program_desc *desc,
   struct agx_ioto_state_tail_output *tail, bool ordinary)
{
   if (!desc || !tail || desc->shader_binary_size == 0 ||
       desc->nr_gprs == 0 ||
       (desc->sample_size_B & 7) || desc->sample_size_B > 248)
      return false;

   static const uint8_t vertex_speclm[AGX_IOTO_SPECLM_SIZE] = {
      0x77, 0x00, 0x2a, 0x41, 0x00, 0x00, 0x00, 0x00,
   };
   static const uint8_t fragment_speclm[AGX_IOTO_SPECLM_SIZE] = {
      0x77, 0x00, 0x2a, 0x43, 0x4b, 0x00, 0x00, 0x00,
   };

   struct agx_ioto_state ldshdr = {0};
   ldshdr.f[0] = desc->shader_va;
   /* Native G17P graphics code containers reserve an 0x80-byte framing area
    * before the 64-byte-aligned main code. LDSHDR f1 is the saturated number
    * of complete 0x80-byte spans in that padded main-code region. Valid
    * captures cover container sizes 0xc0/0x100/0x140/0x1c0/0x240 and yield
    * f1 0/1/1/2/3 respectively, including the cold 88-byte VS and 128-byte
    * EOT. Compute uses a separate state builder and remains unchanged.
    */
   const uint64_t padded_shader_size =
      ((uint64_t)desc->shader_binary_size + 63) & ~UINT64_C(63);
   const uint64_t fetch_span = padded_shader_size >> 7;
   ldshdr.f[1] = fetch_span > 3 ? 3 : fetch_span;
   ldshdr.f[23] = 1;

   struct agx_ioto_state spectpr = {0};
   /* The valid cold G1a no-vary fragment and the matched Apple Metal
    * 32023.917 constant-colour probe both omit ProgramVariant field 38.
    * FragmentProgramVariantESLState supplies D=0, whose rounded SPECTPR
    * payload is entirely zero. Mesa's nr_gprs is not this Apple metadata
    * field; using even a two-GPR shader here incorrectly sets f22=2.
    *
    * Resource fragments retain the compiler-footprint path below until their
    * ProgramVariant metadata contract is independently recovered.
   */
   const bool exact_no_resource_fragment =
      ordinary && desc->fragment && desc->uniform_range_count == 0 &&
      desc->scratch_size == 0;
   const uint32_t compiler_d =
      desc->compiler_d_valid ? desc->compiler_d : desc->nr_gprs;
   if ((desc->compiler_d_valid || !exact_no_resource_fragment) &&
       !agx_ioto_g17p_compiler_spectpr_build(desc->scratch_size,
                                             compiler_d, &spectpr)) {
      return false;
   }

   struct agx_ioto_state_tail_output staged = {
      .length = AGX_IOTO_STATE_TAIL_TYPE0_SIZE,
   };
   memcpy(staged.bytes, desc->fragment ? fragment_speclm : vertex_speclm,
          AGX_IOTO_SPECLM_SIZE);
   if (desc->fragment && desc->sample_size_B) {
      /* G16G SpecLM::encodeImageBlockData (0x3155ac..0x3155bc)
       * produces the stride in 8-byte units in semantic hi[39:35].
       * AGX3EncodedInstr<SPECLM_0> (0x2a79a0..0x2a79cc) copies that
       * field to encoded bits[29:25]. The old constant declared 8 bytes
       * for every target and truncated a 128-bit tile store to 64 bits.
       * Geometry and coverage retain the existing 32x32 pixel-rate ABI.
       */
      staged.bytes[3] = (staged.bytes[3] & 0xc1) |
                        ((desc->sample_size_B / 8) << 1);
   }
   if (desc->fragment) {
      const unsigned samples = desc->nr_samples ?: 1;
      const unsigned tile = desc->tile_size ?: 32 * 32;
      if ((samples != 1 && samples != 2 && samples != 4) ||
          (tile != 32 * 32 && tile != 32 * 16 && tile != 16 * 16))
         return false;
      /* Native SpecLM::encodeImageBlockData + SPECLM_0 encoder, independently
       * executed by tools/macos-native/speclm-imageblock.py. Sample codes
       * 1x/2x/4x map to bits[24:23] = 2/0/1. The utile geometry changes both
       * word1 bit14 and word2 bits[3:1], not the sample stride. */
      staged.bytes[2] = (staged.bytes[2] & 0x7f) | (samples == 4 ? 0x80 : 0);
      staged.bytes[3] = (staged.bytes[3] & 0xbe) |
                        (samples == 1 ? 1 : 0) | (tile == 32 * 32 ? 0x40 : 0);
      staged.bytes[4] = (staged.bytes[4] & 0xf5) |
                        (tile == 32 * 32 ? 0xa : tile == 32 * 16 ? 2 : 0);
   }
   if (!agx_ioto_state_pack_ldshdr(AGX_CHIP_G17P, &ldshdr,
                                   staged.bytes + AGX_IOTO_SPECLM_SIZE,
                                   AGX_IOTO_LDSHDR_SIZE) ||
       !agx_ioto_state_pack_spectpr(
          AGX_CHIP_G17P, &spectpr,
          staged.bytes + AGX_IOTO_SPECLM_SIZE + AGX_IOTO_LDSHDR_SIZE,
          AGX_IOTO_SPECTPR_SIZE))
      return false;

   *tail = staged;
   return true;
}

bool
hk_agx3_g17p_graphics_program_bound(
   enum agx_chip chip, const struct agx3_ioto_uniform_range *ranges,
   size_t range_count, size_t *size)
{
   size_t body, chunks;
   if (!size || !agx_chip_uses_hal200_ioto(chip) ||
       range_count > HK_AGX3_G17P_GRAPHICS_MAX_RANGES ||
       !agx3_ioto_uniform_program_body_bound(chip, ranges, range_count,
                                              &body, &chunks) ||
       chunks > HK_AGX3_G17P_GRAPHICS_MAX_LOAD_CHUNKS ||
       body > HK_AGX3_G17P_GRAPHICS_PROGRAM_MAX_SIZE -
                 AGX_IOTO_G17P_NO_RESOURCE_PROGRAM_SIZE)
      return false;

   /* The vertex envelope is four bytes larger than the fragment envelope.
    * Preserve the native composition and allocate only this shader's bound. */
   *size = body + AGX_IOTO_G17P_NO_RESOURCE_PROGRAM_SIZE;
   return true;
}

bool
hk_agx3_fragment_preamble_append(
   enum agx_chip chip, bool writes_sample_mask, void *output, size_t capacity,
   size_t *length, const void *preamble, size_t preamble_size)
{
   const uint8_t terminal[] = {
      0x07, 0x02, 0x54, writes_sample_mask ? 0xf2 : 0xf3, 0x02, 0x00,
      0x0e, 0x00, 0x00, 0x00,
   };
   const size_t stop_size = 4;
   if (!agx_chip_uses_g15_g16_shader_abi(chip) || !output || !length ||
       !preamble || preamble_size < stop_size || *length < sizeof(terminal) ||
       *length > capacity)
      return false;

   const size_t body_size = preamble_size - stop_size;
   if (body_size > capacity - *length ||
       HK_AGX3_FRAGMENT_PREAMBLE_DRAIN_SIZE > capacity - *length - body_size)
      return false;

   uint8_t *dst = output;
   const uint8_t *src = preamble;
   const size_t insertion = *length - sizeof(terminal);
   if (memcmp(dst + insertion, terminal, sizeof(terminal)) ||
       memcmp(src + body_size, terminal + 6, stop_size))
      return false;

   /* The resource loader publishes root/descriptor pointers through uniform
    * COPYs. Matched G15S/G16G replays isolate intermittent corruption to this
    * producer/consumer boundary, consistent with incomplete publication of
    * those inputs. Padding the loader's start or merely reserving more space
    * does not resolve the matched replay. Keep this conservative while native
    * memory preambles remain opt-in; see dx11-nopromote-preamble.md.
    *
    * Move the body before writing padding, allowing an overlapping input.
    * PIXREL follows the final preamble exports, not just the loader's COPYs.
    */
   memmove(dst + insertion + HK_AGX3_FRAGMENT_PREAMBLE_DRAIN_SIZE,
           src, body_size);
   for (unsigned i = 0; i < HK_AGX3_FRAGMENT_PREAMBLE_DRAIN_SIZE;
        i += AGX_IOTO_LOAD_NOP_SIZE)
      agx_ioto_load_pack_nop(0, dst + insertion + i);
   memcpy(dst + insertion + HK_AGX3_FRAGMENT_PREAMBLE_DRAIN_SIZE + body_size,
          terminal, sizeof(terminal));
   *length += HK_AGX3_FRAGMENT_PREAMBLE_DRAIN_SIZE + body_size;
   return true;
}

bool
hk_agx3_vertex_preamble_branch(
   enum agx_chip chip, void *output, size_t capacity, size_t *length,
   const void *preamble, size_t preamble_size, uint64_t preamble_va,
   uint64_t usc_base, uint64_t program_va)
{
   const uint8_t terminal[] = {
      0x03, 0x00, 0x07, 0x00, 0x02, 0x00, 0x00, 0x00, 0x60, 0x00,
      0x0e, 0x00, 0x00, 0x00,
   };
   const size_t stop_size = 4;
   const size_t suffix_size = HK_AGX3_FRAGMENT_PREAMBLE_DRAIN_SIZE +
      AGX_IOTO_PRESHADER_BRANCH_SIZE;
   if (chip != AGX_CHIP_G15S || !output || !length || !preamble ||
       preamble_size < stop_size || preamble_size > UINT32_MAX ||
       *length < sizeof(terminal) || *length > capacity ||
       usc_base > (UINT64_C(1) << 48) - (UINT64_C(1) << 32) ||
       (program_va & 1) || (preamble_va & 1) ||
       preamble_va < usc_base || preamble_va - usc_base >= (UINT64_C(1) << 32) ||
       preamble_size > (UINT64_C(1) << 32) - (preamble_va - usc_base) ||
       program_va < usc_base || program_va - usc_base >= (UINT64_C(1) << 32) ||
       *length > (UINT64_C(1) << 32) - (program_va - usc_base) ||
       suffix_size - stop_size > (UINT64_C(1) << 32) -
          (program_va - usc_base) - *length ||
       suffix_size - stop_size > capacity - *length)
      return false;

   const uint8_t *src = preamble;
   uint8_t *dst = output;
   uint8_t branch[AGX_IOTO_PRESHADER_BRANCH_SIZE];
   const uint64_t branch_va = program_va + *length - stop_size +
      HK_AGX3_FRAGMENT_PREAMBLE_DRAIN_SIZE;
   /* B carries the signed byte displacement from its own instruction start.
    * Both validated locations share one 4 GiB USC window, so the difference
    * fits the native signed 48-bit field without truncating an address. */
   const int64_t displacement = (int64_t)(preamble_va - usc_base) -
      (int64_t)(branch_va - usc_base);
   const uint64_t encoded = (uint64_t)displacement & ((UINT64_C(1) << 48) - 1);
   if (memcmp(dst + *length - sizeof(terminal), terminal, sizeof(terminal)) ||
       memcmp(src + preamble_size - stop_size, terminal + sizeof(terminal) -
          stop_size, stop_size) ||
       !agx_ioto_preshader_pack_branch(chip, encoded, false, branch))
      return false;

   const size_t insertion = *length - stop_size;
   for (unsigned i = 0; i < HK_AGX3_FRAGMENT_PREAMBLE_DRAIN_SIZE;
        i += AGX_IOTO_LOAD_NOP_SIZE)
      agx_ioto_load_pack_nop(0, dst + insertion + i);
   memcpy(dst + insertion + HK_AGX3_FRAGMENT_PREAMBLE_DRAIN_SIZE,
          branch, sizeof(branch));
   *length += suffix_size - stop_size;
   return true;
}

bool
hk_agx3_g17p_basic_graphics_program_generate(
   enum agx_chip chip, const struct hk_agx3_g17p_graphics_program_desc *desc,
   struct hk_agx3_g17p_graphics_program *program)
{
   if (!program || !agx_chip_uses_hal200_ioto(chip) || !desc ||
       (desc->writes_sample_mask && !desc->fragment) ||
       desc->uniform_range_count > HK_AGX3_G17P_GRAPHICS_MAX_RANGES ||
       (desc->uniform_range_count != 0 && !desc->uniform_ranges))
      return false;

   struct agx_ioto_state_tail_output tail;
   if (!hk_agx3_g17p_graphics_state_tail(desc, &tail, true))
      return false;

   struct hk_agx3_g17p_graphics_program staged;
   size_t written = 0;
   if (desc->uniform_range_count == 0) {
      const bool composed =
         desc->fragment && desc->writes_sample_mask
            ? agx_ioto_g17p_coverage_fragment_program_compose(
                 desc->program_va, NULL, 0, NULL, 0, &tail, staged.bytes,
                 sizeof(staged.bytes), &written)
            : desc->fragment
            ? agx_ioto_g17p_fragment_no_resource_program_compose(
                 desc->program_va, &tail, staged.bytes, sizeof(staged.bytes),
                 &written)
            : agx_ioto_g17p_no_resource_program_compose(
                 desc->program_va, &tail, staged.bytes, sizeof(staged.bytes),
                 &written);
      const size_t expected =
         desc->fragment ? AGX_IOTO_G17P_FRAGMENT_NO_RESOURCE_PROGRAM_SIZE
                        : AGX_IOTO_G17P_NO_RESOURCE_PROGRAM_SIZE;
      if (!composed || written != expected)
         return false;
   } else {
      size_t bound;
      if (!hk_agx3_g17p_graphics_program_bound(
             chip, desc->uniform_ranges, desc->uniform_range_count, &bound))
         return false;
      uint8_t body[HK_AGX3_G17P_GRAPHICS_PROGRAM_MAX_SIZE];
      struct agx_ioto_inflight_load loads[HK_AGX3_G17P_GRAPHICS_MAX_LOAD_CHUNKS];
      struct agx_ioto_inflight_load sort[HK_AGX3_G17P_GRAPHICS_MAX_LOAD_CHUNKS];
      struct agx_ioto_load_cache_entry cache[HK_AGX3_G17P_GRAPHICS_MAX_LOAD_CHUNKS];
      const struct agx3_ioto_uniform_program_desc load_desc = {
         .program_va = desc->program_va,
         .ranges = desc->uniform_ranges,
         .range_count = desc->uniform_range_count,
      };
      const struct agx3_ioto_uniform_program_workspace workspace = {
         .body = body,
         .body_capacity = sizeof(body),
         .loads = loads,
         .load_capacity = sizeof(loads) / sizeof(loads[0]),
         .sort = sort,
         .sort_capacity = sizeof(sort) / sizeof(sort[0]),
         .cache = cache,
         .cache_capacity = sizeof(cache) / sizeof(cache[0]),
      };

      size_t body_size, copy_offset;
      if (!agx3_ioto_uniform_program_body_generate_cached(
             chip, &load_desc, workspace, &body_size, &copy_offset,
             desc->uniform_template) ||
          copy_offset > body_size ||
          !agx_ioto_g17p_generated_resource_program_compose(
             desc->program_va, body, body_size, copy_offset, &tail,
             desc->fragment, desc->writes_sample_mask, staged.bytes,
             sizeof(staged.bytes), &written))
         return false;
   }

   if (written > UINT16_MAX)
      return false;

   memcpy(program->bytes, staged.bytes, written);
   program->length = (uint16_t)written;
   return true;
}

bool
hk_agx3_g17p_combined_sampler_pointers(
   const struct hk_agx3_g17p_combined_sampler_heap *heap,
   uint64_t pointers[2])
{
   if (!heap || !pointers || heap->descriptor_set_va == 0 ||
       heap->sampler_heap_va == 0 || (heap->texture_offset_B & 7) != 0 ||
       heap->sampler_element_size_B != 32 ||
       heap->sampler_index < HK_AGX3_G17P_BINDLESS_SAMPLER_BASE_INDEX)
      return false;

   const uint32_t native_index =
      heap->sampler_index - HK_AGX3_G17P_BINDLESS_SAMPLER_BASE_INDEX;
   if (native_index >= heap->sampler_count)
      return false;

   const uint64_t texture_pointer_va =
      heap->descriptor_set_va + heap->texture_offset_B;
   const uint64_t sampler_offset =
      (uint64_t)native_index * heap->sampler_element_size_B;
   const uint64_t sampler_pointer_va = heap->sampler_heap_va + sampler_offset;
   if (texture_pointer_va < heap->descriptor_set_va ||
       sampler_pointer_va < heap->sampler_heap_va ||
       texture_pointer_va > UINT64_MAX - 31 || sampler_pointer_va > UINT64_MAX - 31)
      return false;

   pointers[0] = texture_pointer_va;
   pointers[1] = sampler_pointer_va;
   return true;
}

bool
hk_agx3_g17p_sampled_storage_pointers(
   const struct hk_agx3_g17p_sampled_storage_heap *heap,
   uint64_t pointers[4])
{
   if (!heap || !pointers || !heap->pbe_descriptor_set_va ||
       (heap->pbe_offset_B & 7) != 0 ||
       heap->pbe_descriptor_set_va > UINT64_MAX - heap->pbe_offset_B ||
       heap->pbe_descriptor_set_va + heap->pbe_offset_B > UINT64_MAX - 31)
      return false;

   uint64_t staged[4] = {0};
   if (!hk_agx3_g17p_combined_sampler_pointers(&heap->sampled, staged))
      return false;

   staged[2] = heap->pbe_descriptor_set_va + heap->pbe_offset_B;
   staged[3] = UINT64_C(0xffff);
   memcpy(pointers, staged, sizeof(staged));
   return true;
}

bool
hk_agx3_g17p_indexed_storage_pointers(
   const struct hk_agx3_g17p_indexed_storage_heap *heap,
   uint64_t pointers[4])
{
   if (!heap || !pointers || !heap->texture_heap_va ||
       !heap->sampler_heap_va || !heap->pbe_descriptor_set_va ||
       (heap->pbe_offset_B & 7) != 0 ||
       heap->pbe_descriptor_set_va > UINT64_MAX - heap->pbe_offset_B ||
       heap->pbe_descriptor_set_va + heap->pbe_offset_B > UINT64_MAX - 31)
      return false;

   const uint64_t staged[4] = {
      heap->texture_heap_va,
      heap->sampler_heap_va,
      heap->pbe_descriptor_set_va + heap->pbe_offset_B,
      UINT64_C(0xffff),
   };
   memcpy(pointers, staged, sizeof(staged));
   return true;
}

bool
hk_agx3_g17p_root_range(uint16_t start_halfs, uint8_t set_count,
                        uint64_t root_va,
                        struct agx3_ioto_uniform_range *range)
{
   if (!range || root_va == 0 || (start_halfs & 1) != 0 ||
       set_count > HK_AGX3_G17P_MAX_ROOT_SETS)
      return false;

   *range = (struct agx3_ioto_uniform_range){
      .start_halfs = start_halfs,
      .size_halfs = 4 * (1 + set_count),
      .address = root_va,
   };
   return true;
}

bool
hk_agx3_g17p_rt0_bg_resource_spec(
   enum agx_chip chip, enum hk_agx3_g17p_rt0_bg_op op, uint64_t *resource_spec)
{
   if (!agx_chip_uses_hal200_ioto(chip) || !resource_spec)
      return false;

   uint64_t pass_mask;
   switch (op) {
   case HK_AGX3_G17P_RT0_BG_CLEAR:
   case HK_AGX3_G17P_RT0_BG_RELOAD:
   case HK_AGX3_BG_MIXED:
      /* AGXMetalG17P agxsGetShaderPassInfo (0x3ababc, extension at
       * 0x3abe60..0x3abe8c) repeats the highest active pass's write bit to
       * fill its 17-bit result. agx_bg_eot.c emits a full-component RT0
       * store for these two bounded shaders, so the result is all ones,
       * not the unextended four-component mask 0xf from a different native
       * framebuffer. The exact WBOIT MRT clear/reload also ends in a live RT1
       * scalar pass, so the same highest-pass extension remains all ones.
       * calculateBackgroundObjectBits (0x3f0d90) places this
       * pass mask at bit 47, independently of the USC pointer and resource
       * fields. This is not a general MRT or partial-component encoder.
       * It is the ordinary pass-control mode: native loader +0xb0 starts
       * at 0 (0x4bdb7c). calculateBackgroundObjectBits mode 0 uses all 17
       * bits, mode 1 reserves the low 8, and mode 2 suppresses the mask.
       * Do not reuse this helper for those specialized modes.
       */
      pass_mask = (UINT64_C(1) << 17) - 1;
      break;
   case HK_AGX3_G17P_RT0_BG_NONE:
      pass_mask = 0;
      break;
   default:
      return false;
   }

   const uint64_t resources =
      AGX_IOTO_G17P_BG_RESOURCE_SPEC & ((UINT64_C(1) << 47) - 1);
   *resource_spec = resources | (pass_mask << 47);
   return true;
}

bool
hk_agx3_g17p_rt0_bg_shader_resources_match(
   enum hk_agx3_g17p_rt0_bg_op op, unsigned push_count, bool uses_txf)
{
   switch (op) {
   case HK_AGX3_G17P_RT0_BG_CLEAR:
      return push_count >= 6 &&
             push_count <= 4 + 8 * HK_AGX3_MAX_COLOR_ATTACHMENTS &&
             !(push_count & 1) && !uses_txf;
   case HK_AGX3_G17P_RT0_BG_RELOAD:
      return push_count == 0 && uses_txf;
   case HK_AGX3_BG_MIXED:
      return push_count >= 10 &&
             push_count <= 8 + 8 * HK_AGX3_MAX_COLOR_ATTACHMENTS &&
             !(push_count & 1) && uses_txf;
   default:
      return false;
   }
}

bool
hk_agx3_g17p_rt0_bg_program_generate(
   enum agx_chip chip, const struct hk_agx3_g17p_rt0_bg_program_desc *desc,
   struct hk_agx3_g17p_rt0_bg_program *program)
{
   if (!program || !agx_chip_uses_hal200_ioto(chip) || !desc)
      return false;

   struct agx3_ioto_uniform_range ranges[3];
   size_t range_count;
   switch (desc->op) {
   case HK_AGX3_BG_MIXED:
      if (!desc->clear_value_va || !desc->texture_pointer_va ||
          !desc->sampler_pointer_va ||
          !hk_agx3_g17p_rt0_bg_shader_resources_match(
             desc->op, desc->clear_push_halfs, true))
         return false;
      /* Immediate TEX keeps its two table pointers in halfs0..7. Clear
       * values begin at half8 so neither resource overwrites the other. */
      ranges[0] = (struct agx3_ioto_uniform_range){
         .start_halfs = 0,
         .size_halfs = 4,
         .address = desc->texture_pointer_va,
      };
      ranges[1] = (struct agx3_ioto_uniform_range){
         .start_halfs = 4,
         .size_halfs = 4,
         .address = desc->sampler_pointer_va,
      };
      ranges[2] = (struct agx3_ioto_uniform_range){
         .start_halfs = 8,
         .size_halfs = desc->clear_push_halfs - 8,
         .address = desc->clear_value_va,
      };
      range_count = 3;
      break;
   case HK_AGX3_G17P_RT0_BG_CLEAR: {
      const uint16_t clear_push_halfs = desc->clear_push_halfs
                                           ? desc->clear_push_halfs
                                           : HK_AGX3_G17P_RT0_CLEAR_PUSH_HALFS;
      if (!desc->clear_value_va || desc->texture_pointer_va ||
          desc->sampler_pointer_va ||
          (clear_push_halfs < 6 ||
           clear_push_halfs > 4 + 8 * HK_AGX3_MAX_COLOR_ATTACHMENTS ||
           (clear_push_halfs & 1)))
         return false;
      ranges[0] = (struct agx3_ioto_uniform_range){
         .start_halfs = 4,
         .size_halfs = clear_push_halfs - 4,
         .address = desc->clear_value_va,
      };
      range_count = 1;
      break;
   }
   case HK_AGX3_G17P_RT0_BG_RELOAD:
      if (desc->clear_value_va || !desc->texture_pointer_va || !desc->sampler_pointer_va)
         return false;
      ranges[0] = (struct agx3_ioto_uniform_range){
         .start_halfs = 0,
         .size_halfs = 4,
         .address = desc->texture_pointer_va,
      };
      ranges[1] = (struct agx3_ioto_uniform_range){
         .start_halfs = 4,
         .size_halfs = 4,
         .address = desc->sampler_pointer_va,
      };
      range_count = 2;
      break;
   default:
      return false;
   }

   const struct hk_agx3_g17p_graphics_program_desc shader = {
      .sample_size_B = desc->sample_size_B,
      .tile_size = desc->tile_size,
      .nr_samples = desc->nr_samples,
      .program_va = desc->program_va,
      .shader_va = desc->shader_va,
      .shader_binary_size = desc->shader_binary_size,
      .scratch_size = desc->scratch_size,
      .nr_gprs = desc->nr_gprs,
      /* BG uses the same independent compiler staging requirement as the
       * main graphics stages. In particular a uniform-fed clear needs D=0
       * despite using GPRs. Unknown texel-fetch staging keeps the existing
       * fallback instead of falsely advertising zero. */
      .compiler_d_valid = desc->staging_known,
      .compiler_d = (uint32_t)desc->staging_halfs * 2,
      .fragment = true,
   };
   struct agx_ioto_state_tail_output tail;
   if (!hk_agx3_g17p_graphics_state_tail(&shader, &tail, false))
      return false;

   size_t body_capacity, chunk_count;
   if (!agx3_ioto_uniform_program_body_bound(chip, ranges, range_count,
                                             &body_capacity, &chunk_count) ||
       chunk_count > 64)
      return false;
   uint8_t *body = malloc(body_capacity);
   if (!body)
      return false;
   struct agx_ioto_inflight_load loads[64];
   struct agx_ioto_inflight_load sort[64];
   struct agx_ioto_load_cache_entry cache[64];
   const struct agx3_ioto_uniform_program_desc load_desc = {
      .program_va = desc->program_va,
      .ranges = ranges,
      .range_count = range_count,
   };
   const struct agx3_ioto_uniform_program_workspace workspace = {
      .body = body,
      .body_capacity = body_capacity,
      .loads = loads,
      .load_capacity = sizeof(loads) / sizeof(loads[0]),
      .sort = sort,
      .sort_capacity = sizeof(sort) / sizeof(sort[0]),
      .cache = cache,
      .cache_capacity = sizeof(cache) / sizeof(cache[0]),
   };

   size_t body_size, copy_offset;
   if (!agx3_ioto_uniform_program_body_generate(chip, &load_desc, workspace,
                                                &body_size, &copy_offset) ||
       copy_offset > body_size) {
      free(body);
      return false;
   }

   /*
    * The clear program's length is a bound, not a constant. It embeds
    * clear_value_va as two LDIMM records, and agx_ioto_load_pack_ldimm() is
    * variable length: a record whose destination half is below 32 and whose
    * immediate is below 128 takes a 2-byte compact form instead of the 8-byte
    * long one. The address's high word is `clear_value_va >> 32`, so whenever
    * the clear value happens to live below the 128 GiB mark the program comes
    * out 6 bytes shorter -- 140 versus 134 with the background PIXREL terminal
    * protocol.
    *
    * AGX_IOTO_G17P_RT0_CLEAR_PROGRAM_SIZE is therefore the long-form maximum,
    * which is also what the caller allocates (hk_cmd_draw.c:349-351), so a
    * shorter program is written and used as-is. Demanding equality rejected
    * every clear whose colour landed at a low address, which is to say the
    * background program of an entire class of render passes.
    */
   struct hk_agx3_g17p_rt0_bg_program staged;
   size_t written = 0;
   /* Both bounded auxiliary shaders are compiled by agx_bg_eot.c with
    * ignore_tib_dependencies. Their release descriptor is native BG 0x200;
    * ordinary FS 0x2f3 releases a different set of pixel classes.
    */
   if (!agx_ioto_g17p_background_resource_program_compose(
          desc->program_va, body, copy_offset, body + copy_offset,
          body_size - copy_offset, &tail, staged.bytes, sizeof(staged.bytes),
          &written) ||
       written > UINT16_MAX) {
      free(body);
      return false;
   }
   free(body);

   staged.length = (uint16_t)written;
   *program = staged;
   return true;
}

bool
hk_agx3_g17p_rt0_eot_program_generate(
   enum agx_chip chip, const struct hk_agx3_g17p_rt0_eot_program_desc *desc,
   struct hk_agx3_g17p_rt0_eot_program *program)
{
   const unsigned count = desc ? MAX2(desc->attachment_count, 1) : 0;
   if (!program || !agx_chip_uses_hal200_ioto(chip) || !desc ||
       count > HK_AGX3_MAX_COLOR_ATTACHMENTS || !desc->shader_binary_size ||
       desc->nr_gprs < 8 || desc->staging_halfs < count * 8 ||
       (desc->staging_halfs & 7) ||
       !desc->pbe_argument_va ||
       desc->pbe_argument_va > UINT64_C(0xffffffffffff) - (32 * count - 1) ||
       (desc->pbe_argument_va & 31))
      return false;

   const struct hk_agx3_g17p_graphics_program_desc shader = {
      .sample_size_B = desc->sample_size_B,
      .tile_size = desc->tile_size,
      .nr_samples = desc->nr_samples,
      .program_va = desc->program_va,
      .shader_va = desc->shader_va,
      .shader_binary_size = desc->shader_binary_size,
      .scratch_size = desc->scratch_size,
      .nr_gprs = desc->nr_gprs,
      .compiler_d_valid = true,
      .compiler_d = desc->staging_halfs * 2,
      .fragment = true,
   };
   struct agx_ioto_state_tail_output tail;
   if (!hk_agx3_g17p_graphics_state_tail(&shader, &tail, false))
      return false;

   /* The EOT consumes a PBE table pointer, not inline descriptor words.
    * Matching G17P eot_uber metadata binds driver argument 0x32 through
    * the qword gather to uniform halves 0..3. The driver supplies a raw,
    * 32-byte-aligned GPU VA to its flat PBE32 table, separately from the
    * image address encoded inside each descriptor.
    *
    * Materialize the pointer in GPR halves4..7: the unchanged profile loader
    * uses0..3 before the final
    * COPYs. No descriptor-memory load or load-completion wait is needed.
    */
   uint8_t body[(2 + HK_AGX3_MAX_COLOR_ATTACHMENTS) * AGX_IOTO_LOAD_LDIMM_MAX_SIZE];
   size_t body_size = 0;
   for (unsigned i = 0; i < 2; ++i) {
      uint8_t bytes[AGX_IOTO_LOAD_LDIMM_MAX_SIZE], length;
      if (!agx_ioto_load_pack_ldimm(chip, 4 + 2 * i,
                                     (uint32_t)(desc->pbe_argument_va >> (32 * i)),
                                     bytes, &length))
         return false;
      memcpy(body + body_size, bytes, length);
      body_size += length;
   }
   /* Native kind5 supplies the byte offset inside an imageblock sample.
    * Keep it in CProg GPR8 across profile scratch0..3, independently of the
    * fragment program's register allocation. */
   for (unsigned rt = 0; rt < count; ++rt) {
      uint8_t offset_length;
      if (!agx_ioto_load_pack_ldimm(chip, 8 + 2 * rt,
                                      desc->imageblock_offsets_B[rt],
                                      body + body_size, &offset_length))
         return false;
      body_size += offset_length;
   }

   /* Preserve the proven EOT COPY source-file and no-kill policy. Publish
    * pointer dwords in halves0/2 and the imageblock byte offset in half4.
    */
   uint8_t copies[4 * (2 * HK_AGX3_MAX_COLOR_ATTACHMENTS + 1)];
   const struct agx3_field_kv baseline[] = {{9,3},{10,2},{23,2},{24,2}};
   const unsigned copy_count = 2 * count + 1;
   for (unsigned i = 0; i < copy_count; ++i) {
      const struct agx3_field_kv operands[] = {
         {3, 2u * i}, {16, i < 2 ? 4 + 2 * i : 8 + 2 * ((i - 2) / 2)},
      };
      uint8_t bytes[AGX3_MAX_WORDS * 2], length = 0;
      if (!agx3_bind_compose(AGX_ISA_ENCODING_AGX3B, "copy", 0,
                               baseline, 4, operands, 2, bytes, &length) ||
          length != 4)
         return false;
      memcpy(copies + 4 * i, bytes, length);
   }

   /* Capacity covers the API attachment limit. The generated body and
    * uniform copies grow with active attachments, not shader text size. */
   struct hk_agx3_g17p_rt0_eot_program staged = {0};
   size_t written = 0;
   if (!agx_ioto_g17p_resource_program_compose(
          desc->program_va, body, body_size, copies, 4 * copy_count, &tail,
          staged.bytes, sizeof(staged.bytes), &written) ||
       written > sizeof(staged.bytes))
      return false;

   staged.length = (uint16_t)written;
   *program = staged;
   return true;
}

bool
hk_agx3_g17p_direct_add3_program_upload_prepare(
   enum agx_chip chip,
   const struct agx3_g17p_direct_add3_program_desc *desc,
   const struct agx3_compute_program_upload *upload, size_t *written,
   struct agx3_compute_program *program)
{
   return agx3_g17p_direct_add3_program_upload_prepare(
      chip, desc, upload, written, program);
}

bool
hk_agx3_g17p_direct_add3_dispatch_append(
   const struct agx3_compute_program *program,
   const struct agx3_compute_direct_grid *grid,
   struct agx3_compute_dispatch_sink *sink)
{
   return agx3_g17p_direct_add3_dispatch_append(program, grid, sink);
}

bool
hk_agx3_g17p_native_indirect_compute_dispatch_append(
   const struct agx3_compute_program *program,
   const struct agx3_g17p_native_indirect_dispatch *dispatch,
   struct agx3_compute_dispatch_sink *sink)
{
   return agx3_g17p_native_indirect_compute_dispatch_append(program, dispatch,
                                                             sink);
}

static bool
hk_agx3_g17p_compute_state_tail(
   const struct hk_agx3_g17p_compute_program_desc *desc,
   struct agx_ioto_state_tail_output *tail)
{
   if (!desc || !tail || desc->shader_binary_size == 0 ||
       desc->nr_gprs == 0)
      return false;

   /* The direct-dispatch program kind, exactly as the add3 witness announces
    * its own compute shader.  See the graphics path above for why f1 is a
    * selector rather than a size: deriving it from `shader_binary_size` made
    * this 118-byte shader announce itself as GRID_HELPER, the kind m1n1 uses
    * for the indirect grid-setup helper, and made every shader over 256 bytes
    * fail the compose. */
   const enum agx_ioto_g17p_compute_ldshdr_f1 fetch =
      AGX_IOTO_G17P_COMPUTE_LDSHDR_F1_DIRECT_ADD3;
   struct agx_ioto_state_tail_output staged;
   const bool tail_ok = desc->threadgroup_memory_size_B
      ? agx_ioto_g17p_runtime_compute_tail_build_shared(
           desc->shader_va, fetch, desc->threadgroup_memory_size_B, &staged)
      : agx_ioto_g17p_runtime_compute_tail_build(desc->shader_va, fetch,
                                                 &staged);
   if (!tail_ok)
      return false;

   /* Declare the shader's compiler-D footprint.
    *
    * agx_ioto_g17p_runtime_compute_tail_build() leaves SPECTPR entirely zero,
    * because it was recovered from the hand-written add3 witness, whose
    * register use fits the default allocation. Apple's ShaderInfo field 38 is
    * staging bytes, independently of the ordinary GPR allocation; the G17P
    * graphics path already supplies the same `agx3_staging_halfs * 2` value.
    * Do not substitute nr_gprs here: storage image staging uses 16 halves even
    * when the compiler happens to allocate a different main-register count.
    */
   struct agx_ioto_state spectpr;
   if (!desc->compiler_d_valid ||
       !agx_ioto_g17p_compiler_spectpr_build(desc->scratch_size,
                                             desc->compiler_d,
                                             &spectpr) ||
       !agx_ioto_state_pack_spectpr(
          AGX_CHIP_G17P, &spectpr,
          staged.bytes + AGX_IOTO_SPECLM_SIZE + AGX_IOTO_LDSHDR_SIZE,
          AGX_IOTO_SPECTPR_SIZE))
      return false;

   *tail = staged;
   return true;
}

bool
hk_agx3_g17p_compute_program_generate(
   enum agx_chip chip, const struct hk_agx3_g17p_compute_program_desc *desc,
   struct hk_agx3_g17p_compute_program *program)
{
   if (!program || !agx_chip_uses_hal200_ioto(chip) || !desc ||
       (desc->uniform_range_count != 0 && !desc->uniform_ranges))
      return false;

   struct agx_ioto_state_tail_output tail;
   if (!hk_agx3_g17p_compute_state_tail(desc, &tail))
      return false;

   struct hk_agx3_g17p_compute_program staged = {0};
   size_t written = 0;
   if (agx_chip_is_m3_runtime(chip) && desc->uniform_range_count == 0) {
      /* The qualified RTKit 2419 M3 compute launch loads the body and
       * terminates, without the newer default-profile protocol. Bound this
       * initial form to the register-only shader footprint witnessed there.
       */
      if (desc->scratch_size || desc->threadgroup_memory_size_B || desc->nr_gprs > 8)
         return false;
      memcpy(staged.bytes, tail.bytes + AGX_IOTO_SPECLM_SIZE, AGX_IOTO_LDSHDR_SIZE);
      staged.bytes[AGX_IOTO_LDSHDR_SIZE] = 0x0e;
      written = AGX_IOTO_LDSHDR_SIZE + 4;
   } else if (desc->uniform_range_count == 0) {
      if (!agx_ioto_g17p_no_resource_program_compose(
             desc->program_va, &tail, staged.bytes, sizeof(staged.bytes),
             &written))
         return false;
   } else {
      uint8_t body[HK_AGX3_G17P_COMPUTE_PROGRAM_MAX_SIZE];
      struct agx_ioto_inflight_load
         loads[HK_AGX3_G17P_COMPUTE_MAX_LOAD_CHUNKS];
      struct agx_ioto_inflight_load
         sort[HK_AGX3_G17P_COMPUTE_MAX_LOAD_CHUNKS];
      struct agx_ioto_load_cache_entry
         cache[HK_AGX3_G17P_COMPUTE_MAX_LOAD_CHUNKS];
      size_t body_bound, chunk_count;
      if (!agx3_ioto_uniform_program_body_bound(
             chip, desc->uniform_ranges, desc->uniform_range_count,
             &body_bound, &chunk_count) ||
          body_bound > sizeof(body) ||
          chunk_count > HK_AGX3_G17P_COMPUTE_MAX_LOAD_CHUNKS)
         return false;

      const struct agx3_ioto_uniform_program_desc load_desc = {
         .program_va = desc->program_va,
         .ranges = desc->uniform_ranges,
         .range_count = desc->uniform_range_count,
      };
      const struct agx3_ioto_uniform_program_workspace workspace = {
         .body = body,
         .body_capacity = sizeof(body),
         .loads = loads,
         .load_capacity = sizeof(loads) / sizeof(loads[0]),
         .sort = sort,
         .sort_capacity = sizeof(sort) / sizeof(sort[0]),
         .cache = cache,
         .cache_capacity = sizeof(cache) / sizeof(cache[0]),
      };
      size_t body_size, copy_offset;
      if (!agx3_ioto_uniform_program_body_generate(
             chip, &load_desc, workspace, &body_size, &copy_offset) ||
          copy_offset > body_size ||
          !agx_ioto_g17p_resource_program_compose(
             desc->program_va, body, copy_offset, body + copy_offset,
             body_size - copy_offset, &tail, staged.bytes,
             sizeof(staged.bytes), &written))
         return false;
   }

   if (written > UINT16_MAX)
      return false;

   const struct agx3_g17p_native_compute_program_desc launch_desc = {
      .program_va = desc->program_va,
      .shader_va = desc->shader_va,
      .launch_sources = desc->launch_sources,
   };
   if (!agx3_g17p_native_compute_program_bind(&launch_desc, &staged.launch))
      return false;

   staged.launch.chip = chip;
   staged.length = (uint16_t)written;
   *program = staged;
   return true;
}

bool
hk_agx3_g17p_native_compute_dispatch_append(
   const struct agx3_compute_program *program,
   const struct agx3_compute_direct_grid *grid,
   struct agx3_compute_dispatch_sink *sink)
{
   return agx3_g17p_native_compute_dispatch_append(program, grid, sink);
}

bool
hk_agx3_g17s_minimal_compute_program_prepare(
   enum agx_chip chip,
   const struct agx3_g17s_minimal_compute_program_desc *desc, void *output,
   size_t capacity, size_t *written, struct agx3_compute_program *program)
{
   if (chip != AGX_CHIP_G17S)
      return false;

   return agx3_g17s_minimal_compute_program_prepare(
      desc, output, capacity, written, program);
}

bool
hk_agx3_g17s_minimal_compute_program_upload_prepare(
   enum agx_chip chip,
   const struct agx3_g17s_minimal_compute_program_desc *desc,
   const struct agx3_compute_program_upload *upload, size_t *written,
   struct agx3_compute_program *program)
{
   return agx3_g17s_minimal_compute_program_upload_prepare(
      chip, desc, upload, written, program);
}

bool
hk_agx3_g17s_minimal_compute_program_live_upload_prepare(
   enum agx_chip chip,
   const struct agx3_g17s_minimal_compute_program_desc *desc,
   const struct agx_ioto_g17s_minimal_compute_resources *resources,
   const struct agx3_compute_program_upload *upload, size_t *written,
   struct agx3_compute_program *program)
{
   return agx3_g17s_minimal_compute_program_live_upload_prepare(
      chip, desc, resources, upload, written, program);
}

bool
hk_agx3_g17s_minimal_compute_dispatch_append(
   const struct agx3_compute_program *program,
   const struct agx3_compute_direct_grid *grid,
   struct agx3_compute_dispatch_sink *sink)
{
   return agx3_g17s_minimal_compute_dispatch_append(program, grid, sink);
}
