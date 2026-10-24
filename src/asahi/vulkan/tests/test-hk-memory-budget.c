/* Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */
#undef NDEBUG
#include "hk_memory_budget.h"
#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

static void
write_file(const char *path, const char *value)
{
   FILE *file = fopen(path, "w");
   assert(file);
   assert(fputs(value, file) >= 0);
   assert(!fclose(file));
}

int
main(void)
{
   char temp[] = "/tmp/hk-memory-budget-XXXXXX";
   assert(mkdtemp(temp));
   assert(!chdir(temp));
   assert(!mkdir("group mount", 0700));
   assert(!mkdir("group mount/parent", 0700));
   assert(!mkdir("group mount/parent/leaf", 0700));
   write_file("cgroup", "0::/parent/leaf\n");
   char mountinfo[4096];
   snprintf(mountinfo, sizeof(mountinfo),
            "1 0 0:1 / %s/group\\040mount rw - cgroup2 cgroup rw\n", temp);
   write_file("mountinfo", mountinfo);
   write_file("group mount/parent/memory.max", "2147483648\n");
   write_file("group mount/parent/memory.current", "1073741824\n");
   write_file("group mount/parent/leaf/memory.max", "max\n");
   write_file("group mount/parent/leaf/memory.current", "536870912\n");
   uint64_t available = 42;
   assert(hk_get_cgroup_memory_available("cgroup", "mountinfo", &available));
   assert(available == UINT64_C(1073741824));

   /* The leaf is unlimited; the ancestor counts sibling charges too. */
   write_file("group mount/parent/memory.current", "2147483648\n");
   assert(hk_get_cgroup_memory_available("cgroup", "mountinfo", &available));
   assert(available == 0);
   write_file("group mount/parent/memory.current", "2147483649\n");
   assert(hk_get_cgroup_memory_available("cgroup", "mountinfo", &available));
   assert(available == 0);

   write_file("group mount/parent/memory.current", "1073741824\n");
   write_file("group mount/parent/leaf/memory.max", "805306368\n");
   assert(hk_get_cgroup_memory_available("cgroup", "mountinfo", &available));
   assert(available == UINT64_C(268435456));

   /* A subtree mount's root is stripped from the process's cgroup path. */
   snprintf(mountinfo, sizeof(mountinfo),
            "1 0 0:1 /outside %s/group\\040mount rw - cgroup2 cgroup rw\n", temp);
   write_file("mountinfo", mountinfo);
   write_file("cgroup", "0::/outside/parent/leaf\n");
   assert(hk_get_cgroup_memory_available("cgroup", "mountinfo", &available));
   assert(available == UINT64_C(268435456));
   write_file("cgroup", "0::/outside-other/parent/leaf\n");
   assert(!hk_get_cgroup_memory_available("cgroup", "mountinfo", &available));

   write_file("cgroup", "0::/outside\n");
   write_file("group mount/memory.max", "100\n");
   write_file("group mount/memory.current", "75\n");
   assert(hk_get_cgroup_memory_available("cgroup", "mountinfo", &available));
   assert(available == 25);
   write_file("group mount/memory.max", "max\n");
   assert(!hk_get_cgroup_memory_available("cgroup", "mountinfo", &available));
   write_file("group mount/memory.max", "18446744073709551616\n");
   assert(!hk_get_cgroup_memory_available("cgroup", "mountinfo", &available));
   write_file("group mount/memory.max", "-1\n");
   assert(!hk_get_cgroup_memory_available("cgroup", "mountinfo", &available));
   write_file("group mount/memory.max", "100garbage\n");
   assert(!hk_get_cgroup_memory_available("cgroup", "mountinfo", &available));
   assert(!hk_get_cgroup_memory_available("missing", "mountinfo", &available));
   write_file("cgroup", "0::/../outside\n");
   assert(!hk_get_cgroup_memory_available("cgroup", "mountinfo", &available));
   write_file("cgroup", "2:memory:/outside\n");
   assert(!hk_get_cgroup_memory_available("cgroup", "mountinfo", &available));

   const char *files[] = {"cgroup", "mountinfo", "group mount/memory.max",
      "group mount/memory.current", "group mount/parent/memory.max",
      "group mount/parent/memory.current", "group mount/parent/leaf/memory.max",
      "group mount/parent/leaf/memory.current"};
   for (unsigned i = 0; i < sizeof(files) / sizeof(files[0]); ++i)
      assert(!unlink(files[i]));
   assert(!rmdir("group mount/parent/leaf"));
   assert(!rmdir("group mount/parent"));
   assert(!rmdir("group mount"));
   assert(!chdir("/"));
   assert(!rmdir(temp));
   puts("PASS: cgroup memory headroom, ancestors, subtree mounts and malformed input");
   return 0;
}
