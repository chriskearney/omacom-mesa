/* Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */
#pragma once

#include <stdbool.h>
#include <stdint.h>

/* Return the smallest remaining hard memory limit in the calling process's
 * cgroup v2 ancestry. The proc paths are arguments so host tests can use a
 * synthetic hierarchy without changing the host's cgroups. */
bool hk_get_cgroup_memory_available(const char *proc_cgroup,
                                  const char *proc_mountinfo,
                                  uint64_t *available);
