/* Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */
#pragma once
#include <stdbool.h>

/* The native vertex template must execute observable stores even when PPP
 * discards rasterization. This includes the TES/GS raster variant, not just
 * software transform feedback. This selector does not enable fragments. */
static inline bool
hk_native_vertex_template_enabled(bool rasterizer_discard,
                                  bool writes_external_memory,
                                  bool xfb_vertex_stores)
{
   return !rasterizer_discard || writes_external_memory || xfb_vertex_stores;
}
