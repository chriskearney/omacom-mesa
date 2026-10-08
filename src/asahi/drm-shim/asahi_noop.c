/*
 * Copyright 2022 Alyssa Rosenzweig
 * Copyright 2018 Broadcom
 * SPDX-License-Identifier: MIT
 */

#include <errno.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "util/macros.h"
#include "drm-shim/drm_shim.h"
#include "drm-uapi/asahi_drm.h"

bool drm_shim_driver_prefers_first_render_node = true;

static const struct drm_asahi_params_global params_g13g = {
   .gpu_generation = 13,
   .gpu_variant = 'G',
   .gpu_revision = 0,
   .vm_start = 0x4000,
   .vm_end = 0x7fffff8000ull,
   .command_timestamp_frequency_hz = 1000000000,
};

/*
 * T8140 / A18 Pro, exactly as our G17P kernel adapter reports it
 * (drivers/gpu/drm/asahi/g17_drm.rs:24-27,118-140 and file.rs:295-324): a
 * 42-bit user root two pages below the top, one die, one cluster, USC
 * generation 3, HAL 200, and soft faults on because fault_control defaults to
 * 0xb.  Select with ASAHI_SHIM_CHIP=g17p.  This exists so the Honeykrisp G17P
 * shader and command-stream paths can be exercised without hardware.
 */
static const struct drm_asahi_params_global params_g17p = {
   .features = DRM_ASAHI_FEATURE_SOFT_FAULTS,
   .gpu_generation = 17,
   .gpu_variant = 'P',
   .gpu_revision = 0,
   .chip_id = 0x8140,
   .num_dies = 1,
   .num_clusters_total = 1,
   .num_cores_per_cluster = 5,
   .core_masks = {0x1f},
   .vm_start = 0x4000,
   .vm_end = (1ull << 42) - 0x8000ull,
   .vm_kernel_min_size = 0x20000000,
   .max_commands_per_submission = 1,
   .max_attachments = 16,
   .max_frequency_khz = 1500000,
   .command_timestamp_frequency_hz = 1000000000,
   .usc_generation = 3,
   .gpu_hal_generation = DRM_ASAHI_GPU_HAL_200,
};

/* Synthetic full-core T8132 profile for offline compilation and recording.
 * GPU identity, address ranges and limits follow g16_drm.rs/file.rs. The
 * physical core mask is fused per machine; this models all ten cores.
 * Unlike the hardware, this shim does not implement VM-status queries, so
 * it must not advertise that extension. Neither profile executes GPU work.
 */
static const struct drm_asahi_params_global params_g16g = {
   .gpu_generation = 16,
   .gpu_variant = 'G',
   .gpu_revision = 0x10, /* B0, UABI enum (not firmware enum). */
   .chip_id = 0x8132,
   .num_dies = 1,
   .num_clusters_total = 1,
   .num_cores_per_cluster = 10,
   .core_masks = {0x3ff},
   .vm_start = 0x4000,
   .vm_end = (1ull << 42) - 0x8000ull,
   .vm_kernel_min_size = 0x20000000,
   .max_commands_per_submission = 64,
   .max_attachments = 16,
   .max_frequency_khz = 1470000,
   .command_timestamp_frequency_hz = 1000000000,
   .usc_generation = 3,
   .gpu_hal_generation = DRM_ASAHI_GPU_HAL_200,
};

/* J613 / MacBook Air M3 as published by the M3 kernel runtime's J613 board
 * record (m3_drm.rs params, file.rs get_params): T8122 G15G C0, one cluster
 * of ten cores, USC 3, legacy HAL. max_frequency_khz is the J613 InitData
 * top p-state (1338 MHz). Features are left clear: the shim implements
 * neither VM_STATUS nor scheduled queues. For offline compile/submit smoke
 * only; nothing here has run on G15G hardware. */
static const struct drm_asahi_params_global params_g15g_j613 = {
   .gpu_generation = 15,
   .gpu_variant = 'G',
   .gpu_revision = 0x20, /* C0, UABI enum. */
   .chip_id = 0x8122,
   .num_dies = 1,
   .num_clusters_total = 1,
   .num_cores_per_cluster = 10,
   .core_masks = {0x3ff},
   .vm_start = 0x4000,
   .vm_end = 0x2ffffff8000ull,
   .vm_kernel_min_size = 0x20000000,
   .max_commands_per_submission = 64,
   .max_attachments = 16,
   .max_frequency_khz = 1338000,
   .command_timestamp_frequency_hz = 1000000000,
   .usc_generation = 3,
   .gpu_hal_generation = DRM_ASAHI_GPU_HAL_LEGACY,
};

/* Synthetic J613 identity with the 25G83 USC3/HAL200 contract. Other
 * geometry is inherited from the existing shim fixture; this validates
 * userspace command construction only, never execution or clock state. */
static const struct drm_asahi_params_global params_g15g_j613_hal200 = {
   .gpu_generation = 15,
   .gpu_variant = 'G',
   .gpu_revision = 0x20, /* C0, UABI enum. */
   .chip_id = 0x8122,
   .num_dies = 1,
   .num_clusters_total = 1,
   .num_cores_per_cluster = 10,
   .core_masks = {0x3ff},
   .vm_start = 0x4000,
   .vm_end = (1ull << 42) - 0x8000ull,
   .vm_kernel_min_size = 0x20000000,
   .max_commands_per_submission = 64,
   .max_attachments = 16,
   .max_frequency_khz = 1338000,
   .command_timestamp_frequency_hz = 1000000000,
   .usc_generation = 3,
   .gpu_hal_generation = DRM_ASAHI_GPU_HAL_200,
};

static const struct drm_asahi_params_global *
shim_params(void)
{
   const char *chip = getenv("ASAHI_SHIM_CHIP");

   if (chip && !strcmp(chip, "g17p"))
      return &params_g17p;
   if (chip && !strcmp(chip, "g16g"))
      return &params_g16g;
   if (chip && !strcmp(chip, "g15g-hal200"))
      return &params_g15g_j613_hal200;
   if (chip && !strcmp(chip, "g15g"))
      return &params_g15g_j613;

   return &params_g13g;
}

struct asahi_bo {
   struct shim_bo base;
   uint32_t offset;
};

static struct asahi_bo *
asahi_bo(struct shim_bo *bo)
{
   return (struct asahi_bo *)bo;
}

struct asahi_device {
   uint64_t next_offset;
};

static struct asahi_device asahi = {
   .next_offset = 0x1000,
};

static int
asahi_ioctl_noop(int fd, unsigned long request, void *arg)
{
   return 0;
}

static int
asahi_ioctl_gem_bind_object(int fd, unsigned long request, void *arg)
{
   struct drm_asahi_gem_bind_object *bind = arg;

   switch (bind->op) {
   case DRM_ASAHI_BIND_OBJECT_OP_BIND:
      if (!bind->handle ||
          bind->flags != DRM_ASAHI_BIND_OBJECT_USAGE_TIMESTAMPS) {
         errno = EINVAL;
         return -1;
      }

      /* The no-op winsys has no firmware object table.  Preserve a stable,
       * nonzero identity so timestamp-pool recording exercises the same
       * driver lifetime path without pretending an object was submitted. */
      bind->object_handle = bind->handle;
      return 0;

   case DRM_ASAHI_BIND_OBJECT_OP_UNBIND:
      if (!bind->object_handle) {
         errno = EINVAL;
         return -1;
      }
      return 0;

   default:
      errno = EINVAL;
      return -1;
   }
}

static int
asahi_ioctl_gem_create(int fd, unsigned long request, void *arg)
{
   struct shim_fd *shim_fd = drm_shim_fd_lookup(fd);
   struct drm_asahi_gem_create *create = arg;
   struct asahi_bo *bo = calloc(1, sizeof(*bo));

   drm_shim_bo_init(&bo->base, create->size);

   assert(UINT64_MAX - asahi.next_offset > create->size);
   bo->offset = asahi.next_offset;
   asahi.next_offset += create->size;

   create->handle = drm_shim_bo_get_handle(shim_fd, &bo->base);

   drm_shim_bo_put(&bo->base);

   return 0;
}

static int
asahi_ioctl_gem_mmap_offset(int fd, unsigned long request, void *arg)
{
   struct shim_fd *shim_fd = drm_shim_fd_lookup(fd);
   struct drm_asahi_gem_mmap_offset *map = arg;
   struct shim_bo *bo = drm_shim_bo_lookup(shim_fd, map->handle);

   map->offset = drm_shim_bo_get_mmap_offset(shim_fd, bo);

   drm_shim_bo_put(bo);

   return 0;
}

static int
asahi_ioctl_get_param(int fd, unsigned long request, void *arg)
{
   struct drm_asahi_get_params *gp = arg;

   switch (gp->param_group) {
   case 0: {
      /* Mirror the kernel, which copies min(sizeof(params), data.size) and
       * reports back what it wrote (file.rs get_params).  Asserting equality
       * instead made the shim unusable against any caller whose header
       * differs by a field. */
      size_t size = MIN2((size_t)gp->size, sizeof(struct drm_asahi_params_global));
      memcpy((void *)gp->pointer, shim_params(), size);
      gp->size = size;
      return 0;
   }

   default:
      fprintf(stderr, "Unknown DRM_IOCTL_ASAHI_GET_PARAMS %d\n",
              gp->param_group);
      return -1;
   }
}

/* The kernel's VA rules, mirrored so a bad address layout fails here instead
 * of costing a boot on hardware.
 *
 * drivers/gpu/drm/asahi/file.rs rejects a bind that is not contained in the
 * device user range, or that overlaps the VM's kernel range, both with EINVAL.
 * A no-op VM_BIND accepts every address, so an allocator geometry that the
 * real driver refuses looks perfectly healthy under the shim.  These two
 * checks are the whole difference between reproducing that on a laptop and
 * discovering it on the target. */
static struct {
   uint64_t kernel_start, kernel_end;
   bool valid;
} shim_vm;

static int
asahi_ioctl_vm_create(int fd, unsigned long request, void *arg)
{
   struct drm_asahi_vm_create *create = arg;

   shim_vm.kernel_start = create->kernel_start;
   shim_vm.kernel_end = create->kernel_end;
   shim_vm.valid = true;
   create->vm_id = 1;
   return 0;
}

static int
asahi_ioctl_vm_bind(int fd, unsigned long request, void *arg)
{
   const struct drm_asahi_vm_bind *bind = arg;
   const struct drm_asahi_params_global *params = shim_params();

   for (uint32_t i = 0; i < bind->num_binds; ++i) {
      const struct drm_asahi_gem_bind_op *op =
         (const struct drm_asahi_gem_bind_op *)(uintptr_t)(bind->userptr +
                                                          (uint64_t)i *
                                                             bind->stride);
      uint64_t start = op->addr;
      uint64_t end = start + op->range;

      if (end < start || start < params->vm_start || end > params->vm_end) {
         fprintf(stderr,
                 "shim: gem_bind: Invalid map range %#llx..%#llx "
                 "(not contained in user range %#llx..%#llx)\n",
                 (unsigned long long)start, (unsigned long long)end,
                 (unsigned long long)params->vm_start,
                 (unsigned long long)params->vm_end);
         return -EINVAL;
      }

      if (shim_vm.valid && start < shim_vm.kernel_end &&
          shim_vm.kernel_start < end) {
         fprintf(stderr,
                 "shim: gem_bind: Invalid map range %#llx..%#llx "
                 "(intrudes in kernel range %#llx..%#llx)\n",
                 (unsigned long long)start, (unsigned long long)end,
                 (unsigned long long)shim_vm.kernel_start,
                 (unsigned long long)shim_vm.kernel_end);
         return -EINVAL;
      }
   }

   return 0;
}

static ioctl_fn_t driver_ioctls[] = {
   [DRM_ASAHI_GET_PARAMS] = asahi_ioctl_get_param,
   [DRM_ASAHI_VM_CREATE] = asahi_ioctl_vm_create,
   [DRM_ASAHI_VM_DESTROY] = asahi_ioctl_noop,
   [DRM_ASAHI_VM_BIND] = asahi_ioctl_vm_bind,
   [DRM_ASAHI_GEM_CREATE] = asahi_ioctl_gem_create,
   [DRM_ASAHI_GEM_MMAP_OFFSET] = asahi_ioctl_gem_mmap_offset,
   [DRM_ASAHI_QUEUE_CREATE] = asahi_ioctl_noop,
   [DRM_ASAHI_QUEUE_DESTROY] = asahi_ioctl_noop,
   [DRM_ASAHI_GEM_BIND_OBJECT] = asahi_ioctl_gem_bind_object,
   [DRM_ASAHI_SUBMIT] = asahi_ioctl_noop,
};

void
drm_shim_driver_init(void)
{
   shim_device.bus_type = DRM_BUS_PLATFORM;
   shim_device.driver_name = "asahi";
   shim_device.driver_ioctls = driver_ioctls;
   shim_device.driver_ioctl_count = ARRAY_SIZE(driver_ioctls);

   const char *chip = getenv("ASAHI_SHIM_CHIP");
   const bool g17p = chip && !strcmp(chip, "g17p");
   const bool g16g = chip && !strcmp(chip, "g16g");

   drm_shim_override_file(g16g ? "DRIVER=asahi\n"
                                 "OF_FULLNAME=/soc/agx\n"
                                 "OF_COMPATIBLE_0=apple,gpu-g16g\n"
                                 "OF_COMPATIBLE_N=1\n"
                          : g17p ? "DRIVER=asahi\n"
                                 "OF_FULLNAME=/soc/agx\n"
                                 "OF_COMPATIBLE_0=apple,gpu-g17p\n"
                                 "OF_COMPATIBLE_N=1\n"
                               : "DRIVER=asahi\n"
                                 "OF_FULLNAME=/soc/agx\n"
                                 "OF_COMPATIBLE_0=apple,gpu-g13g\n"
                                 "OF_COMPATIBLE_N=1\n",
                          "/sys/dev/char/%d:%d/device/uevent", DRM_MAJOR,
                          render_node_minor);
}
