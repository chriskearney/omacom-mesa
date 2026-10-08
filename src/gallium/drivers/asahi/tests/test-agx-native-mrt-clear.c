/* Copyright 2026 Chris Kearney
 * SPDX-License-Identifier: MIT */
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include "compiler/agx3_graphics_program.h"
#include "pipe/p_context.h"
#include "pipe/p_screen.h"
#include "util/driconf.h"
#include "util/hash_table.h"
#include "util/u_inlines.h"
#include "util/xmlconfig.h"
#include "agx_public.h"
#include "agx_state.h"

#define CHECK(condition, message)                                              \
   do {                                                                        \
      if (!(condition)) {                                                      \
         fprintf(stderr, "%s\n", message);                                     \
         return 1;                                                             \
      }                                                                        \
   } while (0)

int
main(int argc, char **argv)
{
   CHECK(argc == 2, "choose fresh-fresh or fresh-valid");
   const bool valid = !strcmp(argv[1], "fresh-valid");
   CHECK(valid || !strcmp(argv[1], "fresh-fresh"), "unknown case");
   int fd = open("/dev/dri/renderD128", O_RDWR | O_CLOEXEC);
   CHECK(fd >= 0, "render node unavailable");
   static const driOptionDescription descriptions[] = {
#include "driinfo_asahi.h"
   };
   driOptionCache info, options;
   driParseOptionInfo(&info, descriptions, ARRAY_SIZE(descriptions));
   struct pipe_screen_config config = {.options = &options,
                                       .options_info = &info};
   struct pipe_screen *screen = agx_screen_create(fd, NULL, &config);
   CHECK(screen && screen->caps.max_render_targets == 8,
         "native MRT admission");
   CHECK(agx_device(screen)->chip == AGX_CHIP_G15G,
         "test requires native G15G");
   struct pipe_context *pipe = screen->context_create(screen, NULL, 0);
   CHECK(pipe, "context unavailable");
   struct agx_context *ctx = agx_context(pipe);
   struct pipe_resource *rt[2];
   struct pipe_framebuffer_state fb = {
      .width = 64,
      .height = 64,
      .layers = 1,
      .samples = 1,
      .nr_cbufs = 2,
   };
   const struct pipe_resource templ = {
      .target = PIPE_TEXTURE_2D,
      .format = PIPE_FORMAT_R8G8B8A8_UNORM,
      .width0 = 64,
      .height0 = 64,
      .depth0 = 1,
      .array_size = 1,
      .nr_samples = 1,
      .nr_storage_samples = 1,
      .usage = PIPE_USAGE_DEFAULT,
      .bind = PIPE_BIND_RENDER_TARGET | PIPE_BIND_SAMPLER_VIEW,
   };
   for (unsigned i = 0; i < 2; ++i) {
      rt[i] = screen->resource_create(screen, &templ);
      CHECK(rt[i], "RGBA8 resource unavailable");
      fb.cbufs[i] =
         (struct pipe_surface){.texture = rt[i], .format = templ.format};
   }
   struct agx_resource *other = agx_resource(rt[1]);
   if (valid) {
      unsigned char pixels[64 * 64 * 4];
      for (unsigned i = 0; i < sizeof(pixels); i += 4) {
         pixels[i] = 0x31;
         pixels[i + 1] = 0x72;
         pixels[i + 2] = 0xb4;
         pixels[i + 3] = 0xff;
      }
      const struct pipe_box box = {.width = 64, .height = 64, .depth = 1};
      pipe->texture_subdata(pipe, rt[1], 0, PIPE_MAP_WRITE, &box, pixels,
                            64 * 4, sizeof(pixels));
      CHECK(agx_resource_valid(other, 0), "RT1 upload did not initialize it");
   }
   void *saved = malloc(other->bo->size);
   CHECK(saved, "snapshot allocation failed");
   memcpy(saved, agx_bo_map(other->bo), other->bo->size);
   pipe->set_framebuffer_state(pipe, &fb);
   const unsigned skipped = ctx->skipped_no_program;
   const union pipe_color_union color = {.f = {1.0, 0.0, 0.0, 1.0}};
   pipe->clear(pipe, PIPE_CLEAR_COLOR0, 0xf, 0, NULL, &color, 0.0, 0);
   struct agx_batch *batch = agx_get_batch(ctx);
   CHECK(batch->initialized, "clear batch was not initialized");

   /* Background operations must match the attachments before submission. */
   struct asahi_bg_eot bg = agx_build_bg_eot(batch, false, false);
   CHECK(bg.usc, "background program was not generated");
   CHECK(ctx->skipped_no_program == skipped, "clear shader was dropped");
   CHECK(batch->draws > 0 && batch->draw == PIPE_CLEAR_COLOR0,
         "clear must emit a draw writing RT0 only");
   CHECK(!(batch->clear & PIPE_CLEAR_COLOR), "MRT clear was deferred");
   CHECK(!memcmp(saved, agx_bo_map(other->bo), other->bo->size),
         "uncleared RT1 storage was overwritten on the CPU");
   struct agx_bg_eot_key expected = {.tib = batch->tilebuffer_layout};
   expected.op[0] = expected.op[1] = valid ? AGX_BG_LOAD : AGX_BG_EOT_NONE;
   CHECK(_mesa_hash_table_search(ctx->bg_eot.ht, &expected),
         "background did not retain NONE/NONE or normalize LOAD/LOAD");
   CHECK(agx_build_bg_eot(batch, false, true).usc,
         "partial background program was not generated");
   CHECK(agx_build_bg_eot(batch, true, false).usc,
         "end-of-tile program was not generated");
   struct agx_bg_eot_key eot_key = {.tib = batch->tilebuffer_layout};
   eot_key.op[0] = eot_key.op[1] = AGX_EOT_STORE;
   struct agx_bg_eot_shader *eot =
      agx_get_bg_eot_shader(&ctx->bg_eot, &eot_key);
   CHECK(eot && eot->info.binary_size == 136 && eot->info.nr_gprs == 8 &&
            eot->info.agx3_staging_halfs == 16 && !eot->info.scratch_size,
         "unexpected G15G two-RGBA8 compiler ABI");
   struct agx3_g17p_rt0_eot_program_desc desc = {
      .program_va = agx_device(screen)->shader_base + 0x200000,
      .shader_va = eot->ptr + eot->info.main_offset,
      .pbe_argument_va = other->bo->va->addr,
      .attachment_count = 2,
      .imageblock_offset1_B = 4,
      .mixed_mrt = true,
      .shader_binary_size = eot->info.binary_size,
      .nr_gprs = eot->info.nr_gprs,
      .staging_halfs = eot->info.agx3_staging_halfs,
      .imageblock = {.sample_size_B = 8, .nr_samples = 1, .tile_size_px = 1024},
   };
   struct agx3_g17p_rt0_eot_program program;
   CHECK(agx3_g17p_rt0_eot_program_generate(AGX_CHIP_G15G, &desc, &program) &&
            program.length && program.length <= sizeof(program.bytes),
         "EOT descriptor exceeds its program capacity");
   const unsigned eot_length = program.length;
   desc.scratch_size = 16;
   CHECK(!agx3_g17p_rt0_eot_program_generate(AGX_CHIP_G15G, &desc, &program),
         "scratch EOT was admitted");
   desc.scratch_size = 0;
   desc.shader_binary_size = 4097;
   CHECK(!agx3_g17p_rt0_eot_program_generate(AGX_CHIP_G15G, &desc, &program),
         "oversized EOT was admitted");
   desc.shader_binary_size = 137;
   CHECK(!agx3_g17p_rt0_eot_program_generate(AGX_CHIP_G15G, &desc, &program),
         "unsupported EOT was admitted");
   desc.shader_binary_size = 136;
   CHECK(!agx3_g17p_rt0_eot_program_generate(AGX_CHIP_G17P, &desc, &program),
         "G15G EOT size was admitted on another chip");
   pipe->flush(pipe, NULL, 0);
   CHECK(ctx->skipped_no_program == skipped, "flush dropped a shader");
   printf("eot-shader=136 program=%u capacity=%zu guards=pass\n", eot_length,
          sizeof(program.bytes));
   printf(
      "native-mrt-clear=%s case=%s draws=%u write-mask=0x%x bg-eot=accepted\n",
      "pass", argv[1], batch->draws, batch->draw);
   free(saved);
   /* The DRM shim records userspace construction; it does not execute pixels. */
   pipe->destroy(pipe);
   screen->destroy(screen);
   close(fd);
   return 0;
}
