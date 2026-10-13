/*
 * Copyright 2022 Alyssa Rosenzweig
 * SPDX-License-Identifier: MIT
 */

#include "util/format/u_format.h"
#include "agx_tilebuffer.h"

#include <gtest/gtest.h>

struct test {
   const char *name;
   uint8_t nr_samples;
   enum pipe_format formats[8];
   struct agx_tilebuffer_layout layout;
   uint32_t total_size;
};

/* clang-format off */
struct test tests[] = {
   {
      "Simple test",
      1,
      { PIPE_FORMAT_R8G8B8A8_UNORM },
      {
         ._offset_B = { 0 },
         .sample_size_B = 8,
         .nr_samples = 1,
         .tile_size = 32 * 32,
      },
      8192
   },
   {
      "MSAA 2x",
      2,
      { PIPE_FORMAT_R8G8B8A8_UNORM },
      {
         ._offset_B = { 0 },
         .sample_size_B = 8,
         .nr_samples = 2,
         .tile_size = 32 * 32,
      },
      16384
   },
   {
      "MSAA 4x",
      4,
      { PIPE_FORMAT_R8G8B8A8_UNORM },
      {
         ._offset_B = { 0 },
         .sample_size_B = 8,
         .nr_samples = 4,
         .tile_size = 32 * 16,
      },
      16384
   },
   {
      "MRT",
      1,
      {
         PIPE_FORMAT_R16_SINT,
         PIPE_FORMAT_R32G32_FLOAT,
         PIPE_FORMAT_R8_SINT,
         PIPE_FORMAT_R32G32_SINT,
      },
      {
         ._offset_B = { 16, 0, 18, 8 },
         .sample_size_B = 24,
         .nr_samples = 1,
         .tile_size = 32 * 32,
      },
      24576
   },
   {
      "MRT with MSAA 2x",
      2,
      {
         PIPE_FORMAT_R16_SINT,
         PIPE_FORMAT_R32G32_FLOAT,
         PIPE_FORMAT_R8_SINT,
         PIPE_FORMAT_R32G32_SINT,
      },
      {
         ._offset_B = { 16, 0, 18, 8 },
         .sample_size_B = 24,
         .nr_samples = 2,
         .tile_size = 32 * 16,
      },
      24576
   },
   {
      "MRT with MSAA 4x",
      4,
      {
         PIPE_FORMAT_R16_SINT,
         PIPE_FORMAT_R32G32_FLOAT,
         PIPE_FORMAT_R8_SINT,
         PIPE_FORMAT_R32G32_SINT,
      },
      {
         ._offset_B = { 16, 0, 18, 8 },
         .sample_size_B = 24,
         .nr_samples = 4,
         .tile_size = 16 * 16,
      },
      24576
   },
   {
      "MRT test requiring 2 alignment on the second RT",
      1,
      { PIPE_FORMAT_R8_UNORM, PIPE_FORMAT_R16G16_SNORM },
      {
         ._offset_B = { 4, 0 },
         .sample_size_B = 8,
         .nr_samples = 1,
         .tile_size = 32 * 32,
      },
      8192
   },
   {
      "Simple MRT test requiring 4 alignment on the second RT",
      1,
      { PIPE_FORMAT_R8_UNORM, PIPE_FORMAT_R10G10B10A2_UNORM },
      {
         ._offset_B = { 4, 0 },
         .sample_size_B = 8,
         .nr_samples = 1,
         .tile_size = 32 * 32,
      },
      8192
   },
   {
      "MRT test that requires spilling to consider alignment requirements",
      4,
      {
         PIPE_FORMAT_R32_FLOAT,
         PIPE_FORMAT_R32_FLOAT,
         PIPE_FORMAT_R32_FLOAT,
         PIPE_FORMAT_R32_FLOAT,
         PIPE_FORMAT_R32_FLOAT,
         PIPE_FORMAT_R32_FLOAT,
         PIPE_FORMAT_R32_FLOAT,
         PIPE_FORMAT_R32_FLOAT,
      },
      {
         .spilled = { false, false, false, false, false, false, true, true },
         ._offset_B = { 0, 4, 8, 12, 16, 20, 0, 0},
         .sample_size_B = 24,
         .nr_samples = 4,
         .tile_size = 16 * 16,
      },
      24576
   },

};
/* clang-format on */

TEST(Tilebuffer, Layouts)
{
   for (unsigned i = 0; i < ARRAY_SIZE(tests); ++i) {
      unsigned nr_cbufs;

      for (nr_cbufs = 0; nr_cbufs < ARRAY_SIZE(tests[i].formats) &&
                         tests[i].formats[nr_cbufs] != PIPE_FORMAT_NONE;
           ++nr_cbufs)
         ;

      struct agx_tilebuffer_layout actual = agx_build_tilebuffer_layout(
         tests[i].formats, nr_cbufs, tests[i].nr_samples, false);

      ASSERT_EQ(tests[i].layout.sample_size_B, actual.sample_size_B)
         << tests[i].name;
      ASSERT_EQ(tests[i].layout.nr_samples, actual.nr_samples) << tests[i].name;
      ASSERT_EQ(tests[i].layout.tile_size, actual.tile_size) << tests[i].name;
      ASSERT_EQ(tests[i].total_size, agx_tilebuffer_total_size(&tests[i].layout))
         << tests[i].name;

      for (unsigned j = 0; j < 8; ++j) {
         ASSERT_EQ(tests[i].layout._offset_B[j], actual._offset_B[j])
            << tests[i].name << ", render target " << j;
      }
   }
}

TEST(Tilebuffer, NativeFullAllocation)
{
   const enum pipe_format formats[] = {
      PIPE_FORMAT_R32G32_FLOAT, PIPE_FORMAT_R32G32_FLOAT,
      PIPE_FORMAT_R32G32_FLOAT, PIPE_FORMAT_R32G32_FLOAT,
   };
   auto native = agx_build_tilebuffer_layout_with_limits(formats, 4, 4, true,
                                                       32768, 2048, 248);
   ASSERT_FALSE(agx_tilebuffer_spills(&native));
   ASSERT_EQ(32768u, agx_tilebuffer_total_size(&native));
   ASSERT_EQ(256u, native.tile_size);

   auto deferred = agx_build_tilebuffer_layout_with_limits(formats, 4, 0, true,
                                                         32768, 2048, 248);
   agx_tilebuffer_set_samples(&deferred, 4);
   ASSERT_FALSE(agx_tilebuffer_spills(&deferred));
   ASSERT_EQ(32768u, agx_tilebuffer_total_size(&deferred));

   auto legacy = agx_build_tilebuffer_layout(formats, 4, 4, true);
   ASSERT_TRUE(legacy.spilled[3]);
   ASSERT_EQ(24576u, agx_tilebuffer_total_size(&legacy));
}

TEST(Tilebuffer, NativeRasterSampleBudget)
{
   const enum pipe_format format = PIPE_FORMAT_R8G8B8A8_UNORM;
   auto four = agx_build_tilebuffer_layout_with_limits(&format, 1, 4, false,
                                                       32768, 2048, 248);
   ASSERT_EQ(512u, four.tile_size);
   ASSERT_EQ(16384u, agx_tilebuffer_total_size(&four));
   auto two = agx_build_tilebuffer_layout_with_limits(&format, 1, 2, false,
                                                      32768, 2048, 248);
   ASSERT_EQ(1024u, two.tile_size);
   auto deferred = agx_build_tilebuffer_layout_with_limits(nullptr, 0, 0,
                                                           false, 32768, 2048, 248);
   agx_tilebuffer_set_samples(&deferred, 4);
   ASSERT_EQ(512u, deferred.tile_size);
}

TEST(Tilebuffer, NativeWideSample)
{
   const enum pipe_format formats[] = {
      PIPE_FORMAT_R32G32B32A32_FLOAT, PIPE_FORMAT_R32G32B32A32_FLOAT,
      PIPE_FORMAT_R32G32B32A32_FLOAT, PIPE_FORMAT_R32G32B32A32_FLOAT,
      PIPE_FORMAT_R32G32B32A32_FLOAT, PIPE_FORMAT_R32G32B32A32_FLOAT,
   };
   auto native = agx_build_tilebuffer_layout_with_limits(formats, 6, 1, false,
                                                        0, 2048, 248);
   ASSERT_FALSE(agx_tilebuffer_spills(&native));
   ASSERT_EQ(96u, native.sample_size_B);
   ASSERT_EQ(256u, native.tile_size);
   ASSERT_EQ(80u, native._offset_B[5]);
   auto deferred = agx_build_tilebuffer_layout_with_limits(formats, 6, 0, false,
                                                          0, 2048, 248);
   agx_tilebuffer_set_samples(&deferred, 1);
   ASSERT_EQ(native.sample_size_B, deferred.sample_size_B);
   ASSERT_FALSE(agx_tilebuffer_spills(&deferred));
   auto legacy = agx_build_tilebuffer_layout(formats, 6, 1, false);
   ASSERT_TRUE(legacy.spilled[4]);
}

TEST(Tilebuffer, NativeWideMultisampleBoundary)
{
   const enum pipe_format format = PIPE_FORMAT_R32G32B32A32_UINT;
   auto tib = agx_build_tilebuffer_layout_with_limits(&format, 1, 2, true,
                                                     0, 2048, 248);
   ASSERT_EQ(16u, tib.sample_size_B);
   ASSERT_EQ(512u, tib.tile_size);
   ASSERT_LT(agx_tilebuffer_total_size(&tib), 32768u);
   auto deferred = agx_build_tilebuffer_layout_with_limits(&format, 1, 0, true,
                                                          0, 2048, 248);
   agx_tilebuffer_set_samples(&deferred, 2);
   ASSERT_EQ(tib.tile_size, deferred.tile_size);
}

TEST(Tilebuffer, NativeRgba8MultisampleMrt)
{
   const enum pipe_format formats[] = {
      PIPE_FORMAT_R8G8B8A8_UNORM, PIPE_FORMAT_R8G8B8A8_UNORM,
      PIPE_FORMAT_R8G8B8A8_UNORM, PIPE_FORMAT_R8G8B8A8_UNORM,
   };
   for (unsigned samples : {1u, 2u, 4u}) {
      for (unsigned count : {2u, 3u, 4u}) {
         auto tib = agx_build_tilebuffer_layout_with_limits(
            formats, count, samples, false, 32768, 2048, 248);
         ASSERT_FALSE(agx_tilebuffer_spills(&tib));
         ASSERT_LE(agx_tilebuffer_total_size(&tib), 32768u);
         const unsigned mask = agx_tilebuffer_total_size(&tib) < 32768
                                  ? BITFIELD_MASK(count) : 0;
         ASSERT_EQ(mask, agx_g17p_rgba8_mrt_mask(&tib));

         auto invalid = tib;
         invalid.spilled[1] = true;
         ASSERT_EQ(0u, agx_g17p_rgba8_mrt_mask(&invalid));
         invalid = tib;
         invalid._offset_B[1] += 4;
         ASSERT_EQ(0u, agx_g17p_rgba8_mrt_mask(&invalid));
         invalid = tib;
         invalid.nr_samples = 8;
         ASSERT_EQ(0u, agx_g17p_rgba8_mrt_mask(&invalid));
         invalid = tib;
         invalid.layered = true;
         ASSERT_EQ(0u, agx_g17p_rgba8_mrt_mask(&invalid));
         invalid = tib;
         invalid.nr_samples = 4;
         invalid.tile_size = 1024;
         invalid.sample_size_B = 8;
         ASSERT_EQ(0u, agx_g17p_rgba8_mrt_mask(&invalid));
      }
   }
}
