/* Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

/* CPU regression: production batch retirement, real BOs and cache, replacing
 * only kernel submission/completion. No command executes on a GPU. Model the
 * public G17 Outputs/aggregate contract, including failures before publication.
 */
#include <dlfcn.h>
#include <fcntl.h>
#include <stdio.h>
#include <unistd.h>
#include "util/driconf.h"
#include "util/xmlconfig.h"
#include "agx_state.h"
#include "agx_fence.h"
#include "agx_public.h"

#define CHECK(x)                                                               \
   do {                                                                        \
      if (!(x)) {                                                              \
         fprintf(stderr, "%s:%u: %s\n", __FILE__, __LINE__, #x);               \
         abort();                                                              \
      }                                                                        \
   } while (0)

static const size_t probe_size = 8 * 1024 * 1024;
static const enum agx_bo_flags probe_flags = AGX_BO_READONLY | AGX_BO_WRITEBACK;
enum mode {
   SPLIT_PENDING,
   SPLIT_COMPLETE,
   MIXED_PENDING,
   MIXED_COMPLETE,
   FIRST_FAILURE,
   PREPUBLICATION,
   STALE_GENERATION,
   COMPLETED_ERROR,
   FIRST_PARTIAL,
   SPLIT_WAIT_ERROR,
   MIXED_WAIT_ERROR,
   PROBE_WAIT_ERROR,
   INTERRUPTED_WAIT,
   SUCCESS,
};
enum completion { ABSENT, PENDING, COMPLETE };
static struct {
   uint32_t handle;
   enum completion state;
   bool live;
} fences[128];
static unsigned fence_count, calls, blocking_waits, resets, interruptions;
static enum mode mode;
static bool split, wait_error;
static struct agx_device *device;
static struct agx_bo *probe, *during_wait;
static uint32_t compute_fence, private_fence, aggregate_batch;

static unsigned
track(uint32_t handle)
{
   for (unsigned i = 0; i < fence_count; ++i) {
      if (fences[i].handle == handle && fences[i].live)
         return i;
   }
   CHECK(fence_count < ARRAY_SIZE(fences));
   unsigned i = fence_count++;
   fences[i].handle = handle;
   fences[i].state = ABSENT;
   fences[i].live = true;
   return i;
}

/* The generic no-op shim returns handle 1 for every syncobj. Give this
 * completion fixture distinct object identities so a batch's old fence cannot
 * stand in for a newly published aggregate or a split compute fence.
 */
int
drmSyncobjCreate(int fd, uint32_t flags, uint32_t *handle)
{
   static uint32_t next_handle = 100;
   *handle = next_handle++;
   return 0;
}

int
drmSyncobjDestroy(int fd, uint32_t handle)
{
   for (unsigned i = 0; i < fence_count; ++i) {
      if (fences[i].handle == handle && fences[i].live) {
         CHECK(fences[i].state != PENDING);
         fences[i].live = false;
      }
   }
   int (*real_destroy)(int, uint32_t) = dlsym(RTLD_NEXT, "drmSyncobjDestroy");
   CHECK(real_destroy);
   return real_destroy(fd, handle);
}

int
drmSyncobjWait(int fd, uint32_t *handles, unsigned count, int64_t timeout,
               unsigned flags, uint32_t *first_signaled)
{
   /* Driver setup/ordinary successful retirement belongs to the shim. */
   for (unsigned j = 0; j < count; ++j) {
      for (unsigned i = 0; i < fence_count; ++i) {
         if (!fences[i].live || fences[i].handle != handles[j])
            continue;
         if (fences[i].state == ABSENT)
            return -EINVAL;
         if (fences[i].state == COMPLETE)
            continue;
         CHECK(count == 1);
         CHECK(probe && probe->refcnt == 1);
         if (!during_wait) {
            during_wait = agx_bo_create(device, probe_size, 0x4000, probe_flags,
                                        "retirement allocation while pending");
            CHECK(during_wait && during_wait != probe);
         }
         if (!timeout)
            return mode == PROBE_WAIT_ERROR ? -ENOMEM : -ETIME;
         CHECK(timeout == INT64_MAX);
         blocking_waits++;
         if (wait_error)
            return -EIO;
         if (mode == INTERRUPTED_WAIT && !interruptions++)
            return -EINTR;
         fences[i].state = COMPLETE;
         if (handles[j] == private_fence && aggregate_batch)
            fences[track(aggregate_batch)].state = COMPLETE;
      }
   }
   if (first_signaled)
      *first_signaled = 0;
   return 0;
}

static int
submit(struct agx_device *dev, struct drm_asahi_submit *s,
       struct agx_submit_virt *virt)
{
   calls++;
   const struct drm_asahi_sync *out =
      (const void *)((uintptr_t)s->syncs + s->in_sync_count * sizeof(*out));
   CHECK(s->out_sync_count == 1 || s->out_sync_count == 3);
   unsigned idx;
   if (s->out_sync_count == 1) {
      CHECK(split && calls == 1);
      compute_fence = out[0].handle;
      idx = track(compute_fence);
   } else {
      private_fence = out[2].handle;
      aggregate_batch = out[0].handle;
      CHECK(private_fence != out[0].handle);
      idx = track(private_fence);
   }
   CHECK(fences[idx].state == ABSENT); /* fresh, never a prior generation */
   if (mode == SUCCESS) {
      fences[idx].state = COMPLETE;
      if (s->out_sync_count == 3)
         fences[track(out[0].handle)].state = COMPLETE;
      return 0;
   }
   if (mode == FIRST_FAILURE || mode == PREPUBLICATION ||
       mode == STALE_GENERATION || (split && calls == 2))
      return -EINVAL; /* no output published by this ioctl */

   CHECK(s->out_sync_count == 1 || calls == 1);
   fences[idx].state = (mode == SPLIT_COMPLETE || mode == MIXED_COMPLETE ||
                        mode == COMPLETED_ERROR)
                          ? COMPLETE
                          : PENDING;
   if (s->out_sync_count == 3)
      fences[track(out[0].handle)].state = fences[idx].state;
   /* Split compute succeeds; modern mixed returns an error after publishing
    * the actual aggregate and enqueueing compute, before render enqueue.
    */
   return split && mode != FIRST_PARTIAL ? 0 : -ENOMEM;
}

static void
reset(void *data, enum pipe_reset_status status)
{
   CHECK(status == PIPE_GUILTY_CONTEXT_RESET);
   resets++;
}

static void
run(struct pipe_screen *screen, enum mode scenario)
{
   mode = scenario;
   fence_count = calls = blocking_waits = resets = interruptions = 0;
   compute_fence = private_fence = aggregate_batch = 0;
   during_wait = NULL;
   split = mode == SPLIT_PENDING || mode == SPLIT_COMPLETE ||
           mode == FIRST_FAILURE || mode == FIRST_PARTIAL ||
           mode == SPLIT_WAIT_ERROR;
   wait_error = mode == SPLIT_WAIT_ERROR || mode == MIXED_WAIT_ERROR;
   bool quarantine = wait_error || mode == PROBE_WAIT_ERROR;
   device = agx_device(screen);
   device->params.max_commands_per_submission = split ? 1 : 64;
   device->ops.submit = submit;
   agx_bo_cache_evict_all(device);
   struct pipe_context *pctx = screen->context_create(screen, NULL, 0);
   CHECK(pctx);
   struct agx_context *ctx = agx_context(pctx);
   struct pipe_device_reset_callback cb = {.reset = reset};
   pctx->set_device_reset_callback(pctx, &cb);
   struct agx_batch *batch = agx_get_batch(ctx);
   struct drm_asahi_cmd_compute c = {0};
   struct drm_asahi_cmd_render r = {0};

   if (mode == STALE_GENERATION) {
      mode = SUCCESS;
      agx_batch_submit(ctx, batch, &c, &r);
      agx_sync_all(ctx, "previous output generation");
      struct agx_batch *next = agx_get_batch(ctx);
      CHECK(next == batch); /* retain the previous signalled batch output */
      CHECK(ctx->syncobj == batch->syncobj);
      mode = scenario;
      calls = 0;
   }

   probe = agx_bo_create(device, probe_size, 0x4000, probe_flags,
                         "accepted compute sole batch owner");
   CHECK(probe && probe->refcnt == 1);
   uintptr_t original = (uintptr_t)probe;
   agx_batch_add_bo(batch, probe);
   agx_bo_unreference(device, probe);
   CHECK(probe->refcnt == 1);
   uint32_t last_fence = ctx->syncobj;
   uint64_t last_seqid = ctx->flush_last_seqid;
   agx_batch_submit(ctx, batch, &c, &r);
   CHECK(calls ==
         (split && mode != FIRST_FAILURE && mode != FIRST_PARTIAL ? 2 : 1));
   CHECK(ctx->any_faults && resets == 1);
   CHECK(pctx->get_device_reset_status(pctx) == PIPE_GUILTY_CONTEXT_RESET);
   CHECK(ctx->syncobj == last_fence && ctx->flush_last_seqid == last_seqid);
   CHECK(!agx_fence_create(ctx)); /* never export an older success for render */
   CHECK(!agx_batch_is_active(batch));
   CHECK(agx_batch_is_submitted(batch) == quarantine);
   CHECK(batch->submit_quarantined == quarantine);
   if (quarantine) {
      CHECK(probe->refcnt == 1 && ctx->submit_quarantined);
      CHECK(agx_cleanup_batches(ctx) == -1);
      CHECK(!agx_poll_batch(ctx, batch));
   } else if (mode == SPLIT_PENDING || mode == MIXED_PENDING ||
              mode == FIRST_PARTIAL) {
      CHECK(blocking_waits == 1 && during_wait);
   } else if (mode == INTERRUPTED_WAIT) {
      CHECK(blocking_waits == 2 && during_wait);
   } else {
      CHECK(blocking_waits == 0);
   }
   struct agx_bo *next = agx_bo_create(device, probe_size, 0x4000, probe_flags,
                                       "retirement allocation after failure");
   CHECK(next && (((uintptr_t)next == original) != quarantine));

   agx_sync_all(ctx, "failed batch must not replay");
   pctx->flush(pctx, NULL, 0);
   unsigned old_calls = calls;
   /* Once lost, further batches are discarded without submitting or another
    * reset callback, including a context with quarantined accepted work. */
   agx_batch_submit(ctx, agx_get_batch(ctx), &c, &r);
   CHECK(calls == old_calls && resets == 1);
   pctx->destroy(pctx);
   if (quarantine) {
      CHECK(probe->refcnt == 1);
      CHECK(batch->failed_syncobjs[0] || batch->failed_syncobjs[1]);
   }

   /* A new context still uses the positive mixed/dense path on this device. */
   mode = SUCCESS;
   split = false;
   device->params.max_commands_per_submission = 64;
   struct pipe_context *fresh = screen->context_create(screen, NULL, 0);
   CHECK(fresh);
   struct agx_batch *fb = agx_get_batch(agx_context(fresh));
   agx_batch_add_bo(fb, next);
   agx_batch_add_bo(fb, next);
   CHECK(next->refcnt == 2 && fb->bo_list.set->entries == 1);
   agx_batch_submit(agx_context(fresh), fb, &c, &r);
   agx_sync_all(agx_context(fresh), "fresh context dense BO reuse");
   CHECK(next->refcnt == 1 && !agx_context(fresh)->any_faults);
   fresh->destroy(fresh);
   agx_bo_unreference(device, next);
   agx_bo_unreference(device, during_wait);
   probe = NULL;
   printf("retirement scenario %u: PASS (waits=%u quarantine=%u)\n", scenario,
          blocking_waits, quarantine);
}

int
main(int argc, char **argv)
{
   CHECK(argc == 2);
   unsigned scenario = strtoul(argv[1], NULL, 0);
   CHECK(scenario < SUCCESS);
   int fd = open("/dev/dri/renderD128", O_RDWR | O_CLOEXEC);
   CHECK(fd >= 0);
   static const driOptionDescription options[] = {
#include "driinfo_asahi.h"
   };
   driOptionCache info, cache;
   driParseOptionInfo(&info, options, ARRAY_SIZE(options));
   struct pipe_screen_config config = {.options = &cache,
                                       .options_info = &info};
   struct pipe_screen *screen = agx_screen_create(fd, NULL, &config);
   CHECK(screen && agx_device(screen)->chip == AGX_CHIP_G17P);
   run(screen, scenario);
   bool quarantined = scenario == SPLIT_WAIT_ERROR ||
                      scenario == MIXED_WAIT_ERROR ||
                      scenario == PROBE_WAIT_ERROR;
   int device_fd = agx_device(screen)->fd;
   screen->destroy(screen);
   if (quarantined)
      CHECK(fcntl(device_fd, F_GETFD) >= 0);
   driDestroyOptionInfo(&info);
   close(fd);
   return 0;
}
