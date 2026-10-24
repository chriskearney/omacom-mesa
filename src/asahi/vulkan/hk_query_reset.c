/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include "hk_cmd_buffer.h"

/* Reset dispatches may overlap one another: they write the same reset value
 * even if their ranges overlap. Make them visible once before any consumer,
 * or before closing a stream that later streams/submissions can depend on.
 * Keep the existing queue dependencies; only the redundant per-reset cache
 * flush is deferred. Host resets and multiview availability publication do
 * not use this path.
 */
void
hk_flush_query_resets(struct hk_cmd_buffer *cmd, struct hk_cs *cs)
{
   if (!cs->query_reset_pending)
      return;

   hk_ensure_cs_has_space(cmd, cs, AGX3_CDM_BARRIER_SIZE);
   hk_cdm_cache_flush(hk_cmd_buffer_device(cmd), cs);
   cs->query_reset_pending = false;
   cs->stats.query_reset_flushes++;
}
