/* Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */
#include "hk_execution_allowance.h"
#include "hk_agx3_vdm_decode.h"
#include "hk_cmd_buffer.h"
#undef NDEBUG
#include <assert.h>

/* The production merge copies or links bytes before combining metadata. A
 * synthetic link emitter keeps this a CPU-only test of both merge paths.
 */
static unsigned links;
void *
hk_cmd_buffer_emit_cs_jump(struct hk_cmd_buffer *cmd, void *current,
                          uint64_t target, bool vdm)
{
   assert(cmd == NULL && !vdm);
   links++;
   return (uint8_t *)current + AGX_CDM_STREAM_LINK_LENGTH;
}

static void
check_merge(bool first, bool second, bool linked)
{
   uint8_t a_bytes[HK_CS_SPACE_PADDING + 128] = {0}, b_bytes[32] = {0};
   struct hk_cs a = {
      .type = HK_CS_CDM, .start = a_bytes, .current = a_bytes + 8,
      .end = a_bytes + sizeof(a_bytes), .as_build = first,
      .stream_linked = linked,
   };
   struct hk_cs b = {
      .type = HK_CS_CDM, .start = b_bytes, .current = b_bytes + 8,
      .end = b_bytes + sizeof(b_bytes), .as_build = second,
   };
   const unsigned before = links;
   hk_cs_merge_cdm(&a, &b);
   assert(a.as_build == (first || second));
   assert(b.as_build == second);
   assert(links == before + linked);
}

int main(void)
{
   for (enum agx_chip chip = AGX_CHIP_G13G; chip < AGX_CHIP_UNKNOWN; ++chip)
      assert(hk_shader_binary_version(chip) ==
             (chip == AGX_CHIP_G17P ? 35 :
              chip <= AGX_CHIP_G15P ? 36 : 34));
   assert(hk_as_build_flags(AGX_CHIP_G17P, 0, true) == 0);
   assert(hk_as_build_flags(AGX_CHIP_G17P, DRM_ASAHI_FEATURE_RENDER_RAY_QUERY, true) == 0);
   assert(hk_as_build_flags(AGX_CHIP_G17P, DRM_ASAHI_FEATURE_BOUNDED_AS_BUILD, false) == 0);
   assert(hk_as_build_flags(AGX_CHIP_G17P, DRM_ASAHI_FEATURE_BOUNDED_AS_BUILD, true) ==
          DRM_ASAHI_COMPUTE_AS_BUILD);
   assert(hk_as_build_flags(AGX_CHIP_G17P, UINT64_MAX, true) == DRM_ASAHI_COMPUTE_AS_BUILD);
   /* Even a future classic kernel advertising every bit keeps zero flags. */
   for (enum agx_chip chip = AGX_CHIP_G13G; chip <= AGX_CHIP_G17A; ++chip)
      if (chip != AGX_CHIP_G17P)
         assert(hk_as_build_flags(chip, UINT64_MAX, true) == 0);
   for (unsigned a = 0; a < 2; ++a)
      for (unsigned b = 0; b < 2; ++b)
         for (unsigned link = 0; link < 2; ++link)
            check_merge(a, b, link);
   return 0;
}
