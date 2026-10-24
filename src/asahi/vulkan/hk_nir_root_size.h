/* Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */
#pragma once
#include "nir.h"

/* Required prefix of the root table. Unknown/escaping accesses require full_size. */
unsigned hk_nir_root_size(const nir_shader *nir, unsigned full_size);
