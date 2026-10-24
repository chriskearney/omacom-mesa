/* Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */
#pragma once
#include <stdbool.h>
#include <stdint.h>
struct nir_shader;

/* Count instructions in the expanded entrypoint, charging every call site
 * even when callees share an implementation. Saturate before allocating IR. */
bool hk_rt_inlining_fits(struct nir_shader *shader, uint32_t limit);
