/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * The metadata tile indexing the G17P compute mip path writes with
 * (hk_g17p_metadata_tile_index) against libagx_fast_clear's index_metadata,
 * and against the metadata AIL allocates for each compressed level: every
 * tile of a level has its own word inside that level's metadata.
 */
#include "hk_g17p_meta_copy_core.h"
#include "layout.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static unsigned failures;

#define EXPECT(cond, ...)                                                      \
   do {                                                                        \
      if (!(cond)) {                                                           \
         fprintf(stderr, __VA_ARGS__);                                         \
         fprintf(stderr, "\n");                                                \
         failures++;                                                           \
      }                                                                        \
   } while (0)

/* libagx/compression.cl index_metadata(), with a bit-by-bit interleave */
static uint32_t
reference_index(uint32_t x, uint32_t y, uint32_t width, uint32_t height)
{
   const uint32_t major = width > height ? x : y;
   const uint32_t minor = width < height ? width : height;
   uint32_t bits = 0;
   while ((1u << bits) < minor)
      bits++;
   const uint32_t mask = (1u << bits) - 1;

   uint32_t interleaved = 0;
   for (uint32_t i = 0; i < 16; ++i) {
      interleaved |= ((x & mask) >> i & 1) << (2 * i);
      interleaved |= ((y & mask) >> i & 1) << (2 * i + 1);
   }
   return interleaved + ((major & ~mask) << bits);
}

static void
check_grid(uint32_t width_tl, uint32_t height_tl)
{
   struct hk_g17p_metadata_grid grid;
   EXPECT(hk_g17p_metadata_grid_init(width_tl, height_tl, &grid),
          "grid %ux%u refused", width_tl, height_tl);
   for (uint32_t y = 0; y < height_tl; ++y) {
      for (uint32_t x = 0; x < width_tl; ++x) {
         const uint32_t got = hk_g17p_metadata_tile_index(&grid, x, y);
         const uint32_t want = reference_index(x, y, width_tl, height_tl);
         if (got != want) {
            EXPECT(false, "grid %ux%u tile (%u,%u): index %u, want %u",
                   width_tl, height_tl, x, y, got, want);
            return;
         }
      }
   }
}

/* Every logically compressed level of an RGBA8 image: distinct words, all
 * inside the level's own metadata. */
static void
check_layout(uint32_t width, uint32_t height)
{
   struct ail_layout layout = {
      .width_px = width,
      .height_px = height,
      .depth_px = 1,
      .sample_count_sa = 1,
      .levels = util_logbase2(MAX2(width, height)) + 1,
      .tiling = AIL_TILING_GPU,
      .format = PIPE_FORMAT_R8G8B8A8_UNORM,
      .compressed = true,
   };
   ail_make_miptree(&layout);

   for (uint32_t l = 0; l < layout.levels; ++l) {
      if (!ail_is_level_logically_compressed(&layout, l))
         continue;

      const uint64_t end_B =
         l + 1 < layout.levels && ail_is_level_allocated_compressed(&layout, l + 1)
            ? layout.level_offsets_compressed_B[l + 1]
            : layout.compression_layer_stride_B;
      const uint32_t words =
         (uint32_t)((end_B - layout.level_offsets_compressed_B[l]) / 8);

      struct hk_g17p_metadata_grid grid;
      const uint32_t w_tl = ail_metadata_width_tl(&layout, l);
      const uint32_t h_tl = ail_metadata_height_tl(&layout, l);
      EXPECT(hk_g17p_metadata_grid_init(w_tl, h_tl, &grid),
             "%ux%u level %u: grid refused", width, height, l);

      uint8_t *seen = calloc(words, 1);
      for (uint32_t y = 0; y < h_tl; ++y) {
         for (uint32_t x = 0; x < w_tl; ++x) {
            const uint32_t index = hk_g17p_metadata_tile_index(&grid, x, y);
            if (index >= words || seen[index]) {
               EXPECT(false,
                      "%ux%u level %u tile (%u,%u): word %u of %u%s", width,
                      height, l, x, y, index, words,
                      index < words ? " written twice" : "");
               free(seen);
               return;
            }
            seen[index] = 1;
         }
      }
      free(seen);
   }
}

/* Which levels of common chains are stored compressed by a render pass */
static void
check_mip_split(void)
{
   EXPECT(hk_g17p_mip_level_through_pbe(1920, 1080),
          "3840x2160 level 1 should take a render pass");
   EXPECT(hk_g17p_mip_level_through_pbe(2048, 2048),
          "4096x4096 level 1 should take a render pass");
   EXPECT(hk_g17p_mip_level_through_pbe(1280, 720),
          "2560x1440 level 1 should take a render pass");
   EXPECT(hk_g17p_mip_level_through_pbe(960, 540),
          "1920x1080 level 1 should take a render pass");
   EXPECT(hk_g17p_mip_level_through_pbe(640, 360),
          "1280x720 level 1 should take a render pass");
   EXPECT(!hk_g17p_mip_level_through_pbe(480, 270),
          "1920x1080 level 2 should take compute");
   EXPECT(!hk_g17p_mip_level_through_pbe(256, 256),
          "512x512 level 1 should take compute");
   EXPECT(!hk_g17p_mip_level_through_pbe(1, 1), "1x1 should take compute");
}

int
main(void)
{
   check_mip_split();
   for (uint32_t h = 1; h <= 70; ++h) {
      for (uint32_t w = 1; w <= 70; ++w)
         check_grid(w, h);
   }
   check_grid(240, 135);
   check_grid(135, 240);
   check_grid(4096, 1);

   static const uint32_t sizes[][2] = {
      {1920, 1080}, {3840, 2160}, {256, 256}, {129, 127}, {1000, 40},
      {40, 1000},   {300, 17},    {16, 16},   {17, 4096}, {960, 540},
   };
   for (unsigned i = 0; i < sizeof(sizes) / sizeof(sizes[0]); ++i)
      check_layout(sizes[i][0], sizes[i][1]);

   if (failures) {
      fprintf(stderr, "%u failure(s)\n", failures);
      return 1;
   }
   printf("hk_g17p_metadata_grid: OK\n");
   return 0;
}
