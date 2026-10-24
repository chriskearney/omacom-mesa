/* Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */
#ifndef HK_NIR_LOWER_TES_RQ_H
#define HK_NIR_LOWER_TES_RQ_H

#include "nir.h"

struct agx_gen1_ray_query_layout;

/* Lower one TES execution variant for Poly, then lower its native ray queries
 * for the stage that runs it: the hardware variant is a vertex program, the
 * software variant a CDM kernel feeding a geometry stage. rq_layout may be
 * NULL when queries are not native. Returns success, not pass progress.
 */
bool hk_nir_lower_tes_variant(nir_shader *tes, bool hardware,
                              const struct agx_gen1_ray_query_layout *rq_layout);

#endif
