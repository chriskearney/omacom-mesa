/* SPDX-License-Identifier: MIT */
#include "agx_g15_wire.h"
#include "agx_chip.h"
#include "drm-uapi/asahi_drm.h"
#include <string.h>

/* Public contract: omacom/linux-aurora@7b969982306e, g15/wire.rs.
 * Its render/compute prefixes end at 256/64, while canonical packets insert
 * eight timestamp-copy records at those offsets. All remaining fields move
 * by 64 bytes. Never reinterpret feature bits without also converting every
 * hardware command. Air/G17 and virtio continue using the canonical wire. */
_Static_assert(sizeof(struct drm_asahi_cmd_header) == 8, "command header");
_Static_assert(sizeof(struct drm_asahi_cmd_render) == 320, "render base");
_Static_assert(offsetof(struct drm_asahi_cmd_render, ts_frag_end_copies) == 256,
               "render timestamp copies");
_Static_assert(offsetof(struct drm_asahi_cmd_render_with_prior, prior_vdm) == 320,
               "canonical render prior");
_Static_assert(sizeof(struct drm_asahi_cmd_render_with_ray_query) == 448,
               "canonical render extension");
_Static_assert(sizeof(struct drm_asahi_cmd_compute) == 128, "compute base");
_Static_assert(offsetof(struct drm_asahi_cmd_compute, ts_end_copies) == 64,
               "compute timestamp copies");
_Static_assert(offsetof(struct drm_asahi_cmd_compute_with_prior, scs_layout) == 128,
               "canonical compute scratch");
_Static_assert(sizeof(struct drm_asahi_cmd_compute_with_prior) == 200,
               "canonical compute extension");

bool
agx_g15_wire_features(const struct drm_asahi_params_global *p, uint64_t *features)
{
   if (!p || !features || !agx_params_driver_supported(p))
      return false;

   enum agx_chip chip = agx_chip_from_params(p);
   if (chip != AGX_CHIP_G15S && chip != AGX_CHIP_G15C)
      return false;

   /* Exact signatures from feature_word(false/true) in g15/wire.rs. In
    * particular, canonical bits 3..9 must not leak into normalized features. */
   if (p->features != UINT64_C(0x3e) && p->features != UINT64_C(0xbe))
      return false;

   *features = DRM_ASAHI_FEATURE_VM_STATUS |
               DRM_ASAHI_FEATURE_INDEPENDENT_QUEUES |
               DRM_ASAHI_FEATURE_PER_COMMAND_INPUT_SYNCS |
               DRM_ASAHI_FEATURE_VM_STATUS_MIRROR |
               DRM_ASAHI_FEATURE_EXACT_PRIOR_BARRIERS;
   if (p->features & UINT64_C(0x80))
      *features |= DRM_ASAHI_FEATURE_FRAGMENT_BARRIERS;
   return true;
}

bool
agx_g15_wire_pack(const void *input, size_t input_size,
                  void *output, size_t output_capacity, size_t *output_size)
{
   if (!input || !output || !output_size || !input_size)
      return false;

   const uint8_t *src = input;
   uint8_t *dst = output;
   size_t in = 0, out = 0;
   while (in < input_size) {
      struct drm_asahi_cmd_header h;
      if (input_size - in < sizeof(h))
         return false;
      memcpy(&h, src + in, sizeof(h));
      in += sizeof(h);
      size_t size = h.size, prefix = 0;
      if (size > input_size - in || size % 8)
         return false;

      switch (h.cmd_type) {
      case DRM_ASAHI_CMD_RENDER:
         if (size != sizeof(struct drm_asahi_cmd_render) &&
             size != sizeof(struct drm_asahi_cmd_render_with_prior) &&
             size != sizeof(struct drm_asahi_cmd_render_with_ray_query))
            return false;
         prefix = 256;
         break;
      case DRM_ASAHI_CMD_COMPUTE:
         if (size != sizeof(struct drm_asahi_cmd_compute) &&
             size != sizeof(struct drm_asahi_cmd_compute_with_prior))
            return false;
         prefix = 64;
         break;
      case DRM_ASAHI_SET_VERTEX_ATTACHMENTS:
      case DRM_ASAHI_SET_FRAGMENT_ATTACHMENTS:
      case DRM_ASAHI_SET_COMPUTE_ATTACHMENTS:
         if (size % sizeof(struct drm_asahi_attachment))
            return false;
         break;
      default:
         return false;
      }

      if (prefix) {
         for (size_t i = prefix; i < prefix + 64; ++i) {
            if (src[in + i])
               return false;
         }
         h.size -= 64;
      }
      if (output_capacity - out < sizeof(h) ||
          output_capacity - out - sizeof(h) < h.size)
         return false;
      memcpy(dst + out, &h, sizeof(h));
      out += sizeof(h);
      if (prefix) {
         memcpy(dst + out, src + in, prefix);
         memcpy(dst + out + prefix, src + in + prefix + 64,
                size - prefix - 64);
      } else {
         memcpy(dst + out, src + in, size);
      }
      in += size;
      out += h.size;
   }
   *output_size = out;
   return true;
}
