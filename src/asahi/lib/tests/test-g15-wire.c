/* SPDX-License-Identifier: MIT */
#undef NDEBUG
#include "agx_g15_wire.h"
#include "agx_chip.h"
#include "drm-uapi/asahi_drm.h"
#include <assert.h>
#include <stdio.h>
#include <string.h>

static void
check_features(void)
{
   struct drm_asahi_params_global p = {
      .gpu_generation = 15, .gpu_variant = 'S', .chip_id = 0x6030,
      .num_dies = 1, .num_clusters_total = 2, .usc_generation = 3,
      .gpu_hal_generation = 0,
   };
   uint64_t f;
   for (unsigned i = 0; i < 65536; ++i) {
      p.features = i;
      bool ok = agx_g15_wire_features(&p, &f);
      assert(ok == (i == 0x3e || i == 0xbe));
      if (ok) {
         assert(f == (i == 0x3e ? 0x1c06 : 0x5c06));
         struct drm_asahi_params_global normalized = p;
         normalized.features = f;
         assert(agx_params_native_kernel_abi_supported(&normalized));
      }
   }
   p.features = 0x3e;
   p.gpu_variant = 'C'; p.chip_id = 0x6031; p.num_clusters_total = 4;
   assert(agx_g15_wire_features(&p, &f));
   struct drm_asahi_params_global bad = p;
   bad.chip_id = 0x6030; assert(!agx_g15_wire_features(&bad, &f));
   bad = p; bad.num_dies = 2; assert(!agx_g15_wire_features(&bad, &f));
   bad = p; bad.usc_generation = 2; assert(!agx_g15_wire_features(&bad, &f));
   bad = p; bad.gpu_hal_generation = 200; assert(!agx_g15_wire_features(&bad, &f));
   bad = p; bad.gpu_variant = 'G'; bad.chip_id = 0x8122;
   bad.num_clusters_total = 1; assert(!agx_g15_wire_features(&bad, &f));
   bad = p; bad.features |= UINT64_C(1) << 40;
   assert(!agx_g15_wire_features(&bad, &f));
   assert(!agx_g15_wire_features(NULL, &f));
   assert(!agx_g15_wire_features(&p, NULL));
}

static void
check_packet(unsigned type, size_t size, size_t prefix)
{
   unsigned char source[512] = {0}, saved[512], output[514];
   struct drm_asahi_cmd_header h = {
      .cmd_type = type, .size = size, .vdm_barrier = 7, .cdm_barrier = 11,
   };
   memcpy(source, &h, sizeof(h));
   for (unsigned i = 0; i < size; ++i) {
      if (i < prefix || i >= prefix + 64)
         source[8 + i] = (i * 73 + 5) & 255;
   }
   memcpy(saved, source, sizeof(source));
   memset(output, 0xaa, sizeof(output));
   size_t n = 0;
   assert(agx_g15_wire_pack(source, size + 8, output + 1, size + 8, &n));
   assert(n == size + 8 - 64);
   assert(!memcmp(saved, source, sizeof(source)));
   assert(output[0] == 0xaa && output[n + 1] == 0xaa);
   struct drm_asahi_cmd_header got;
   memcpy(&got, output + 1, sizeof(got));
   assert(got.cmd_type == type && got.size == size - 64);
   assert(got.vdm_barrier == 7 && got.cdm_barrier == 11);
   assert(!memcmp(output + 9, source + 8, prefix));
   assert(!memcmp(output + 9 + prefix, source + 8 + prefix + 64,
                  size - prefix - 64));
   /* The public Rust wire places render priors at 256/264, compute scratch
    * at 64 and priors at 72/80. Sentinels above cover those exact bytes,
    * including extension tails; no nonzero field may be discarded. */
   assert(!agx_g15_wire_pack(source, size + 8, output, n - 1, &n));
   for (unsigned i = 0; i < 64; ++i) {
      source[8 + prefix + i] = 1;
      assert(!agx_g15_wire_pack(source, size + 8, output, sizeof(output), &n));
      source[8 + prefix + i] = 0;
   }
   for (unsigned i = 1; i < size + 8; ++i)
      assert(!agx_g15_wire_pack(source, i, output, sizeof(output), &n));
}

static void
check_stream(void)
{
   unsigned char src[1024] = {0}, dst[1024] = {0};
   struct drm_asahi_cmd_header h[3] = {
      {DRM_ASAHI_SET_COMPUTE_ATTACHMENTS, 24, 0xffff, 0xffff},
      {DRM_ASAHI_CMD_RENDER, 336, 0xffff, 0xffff},
      {DRM_ASAHI_CMD_COMPUTE, 200, 1, 0},
   };
   size_t in = 0;
   for (unsigned i = 0; i < 3; ++i) {
      memcpy(src + in, &h[i], 8);
      if (!i) memset(src + in + 8, 0x5a, 24);
      in += 8 + h[i].size;
   }
   size_t n;
   assert(agx_g15_wire_pack(src, in, dst, sizeof(dst), &n));
   assert(n == in - 128);
   assert(!memcmp(src, dst, 32));
   struct drm_asahi_cmd_header c;
   memcpy(&c, dst + 32, 8); assert(c.cmd_type == 0 && c.size == 272);
   memcpy(&c, dst + 312, 8); assert(c.cmd_type == 1 && c.size == 136);
   assert(c.vdm_barrier == 1 && c.cdm_barrier == 0);
   src[0] = 99; assert(!agx_g15_wire_pack(src, in, dst, sizeof(dst), &n));
   src[0] = 2; src[2] = 8;
   assert(!agx_g15_wire_pack(src, in, dst, sizeof(dst), &n));
   assert(!agx_g15_wire_pack(src, 0, dst, sizeof(dst), &n));
   assert(!agx_g15_wire_pack(NULL, in, dst, sizeof(dst), &n));
}

int main(void)
{
   check_features();
   check_packet(DRM_ASAHI_CMD_RENDER, 320, 256);
   check_packet(DRM_ASAHI_CMD_RENDER, 336, 256);
   check_packet(DRM_ASAHI_CMD_RENDER, 448, 256);
   check_packet(DRM_ASAHI_CMD_COMPUTE, 128, 64);
   check_packet(DRM_ASAHI_CMD_COMPUTE, 200, 64);
   check_stream();
   puts("G15 retained-wire feature and packet controls passed");
   return 0;
}
