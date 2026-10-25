/* SPDX-License-Identifier: MIT */
#ifndef AGX_G15_WIRE_H
#define AGX_G15_WIRE_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

struct drm_asahi_params_global;

/* Omacom's retained G15 backend uses the Mesa-751 wire, not the canonical
 * header's feature assignments or timestamp-copy slots. Identify only its
 * exact supported parameter signatures, before VM creation. */
bool agx_g15_wire_features(const struct drm_asahi_params_global *params,
                          uint64_t *features);

/* Convert canonical packets into that wire in a separate output buffer.
 * Removed timestamp-copy slots must be zero. Unknown/truncated packets fail;
 * the caller must not submit the output on failure. No GPU access occurs. */
bool agx_g15_wire_pack(const void *input, size_t input_size,
                      void *output, size_t output_capacity, size_t *output_size);

#endif
