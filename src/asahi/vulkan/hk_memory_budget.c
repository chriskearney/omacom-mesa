/* Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */
#include "hk_memory_budget.h"

#include <errno.h>
#include <inttypes.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* mountinfo escapes whitespace and backslashes with octal sequences. */
static void
unescape(char *s)
{
   char *out = s;
   while (*s) {
      if (s[0] == '\\' && s[1] >= '0' && s[1] <= '7' &&
          s[2] >= '0' && s[2] <= '7' && s[3] >= '0' && s[3] <= '7') {
         *out++ = (s[1] - '0') * 64 + (s[2] - '0') * 8 + s[3] - '0';
         s += 4;
      } else {
         *out++ = *s++;
      }
   }
   *out = 0;
}

static bool
read_counter(const char *directory, const char *name, uint64_t *value)
{
   char path[PATH_MAX];
   if (snprintf(path, sizeof(path), "%s/%s", directory, name) >= sizeof(path))
      return false;

   FILE *file = fopen(path, "re");
   if (!file)
      return false;
   char text[64];
   bool read = fgets(text, sizeof(text), file) != NULL;
   fclose(file);
   if (!read || text[0] < '0' || text[0] > '9')
      return false;

   errno = 0;
   char *end;
   uint64_t result = strtoull(text, &end, 10);
   if (errno || (*end != '\n' && *end != 0))
      return false;
   *value = result;
   return true;
}

static bool
available_in_hierarchy(const char *mount, const char *relative,
                       uint64_t *available)
{
   char path[PATH_MAX];
   if (snprintf(path, sizeof(path), "%s%s", mount, relative) >= sizeof(path))
      return false;

   size_t root_len = strlen(mount);
   while (strlen(path) > root_len && path[strlen(path) - 1] == '/')
      path[strlen(path) - 1] = 0;

   bool limited = false;
   uint64_t remaining = UINT64_MAX;
   for (;;) {
      uint64_t max, current;
      /* An unlimited node, or the hierarchy root without memory.max, can
       * still have a limited descendant or ancestor. */
      if (read_counter(path, "memory.max", &max) &&
          read_counter(path, "memory.current", &current)) {
         uint64_t free = current < max ? max - current : 0;
         if (free < remaining)
            remaining = free;
         limited = true;
      }

      if (strlen(path) <= root_len)
         break;
      char *slash = strrchr(path, '/');
      if (!slash || slash < path + root_len)
         path[root_len] = 0;
      else
         *slash = 0;
   }

   if (limited)
      *available = remaining;
   return limited;
}

bool
hk_get_cgroup_memory_available(const char *proc_cgroup,
                               const char *proc_mountinfo,
                               uint64_t *available)
{
   FILE *file = fopen(proc_cgroup, "re");
   if (!file)
      return false;

   char group[PATH_MAX] = {0};
   char *line = NULL;
   size_t capacity = 0;
   while (getline(&line, &capacity, file) >= 0) {
      if (!strncmp(line, "0::/", 4)) {
         size_t length = strcspn(line + 3, "\n");
         if (length < sizeof(group)) {
            memcpy(group, line + 3, length);
            group[length] = 0;
         }
         break;
      }
   }
   fclose(file);

   /* A cgroup namespace outside the mounted hierarchy may expose ../ paths.
    * Do not follow those out of the cgroup mount. */
   if (!group[0] || strstr(group, "/../") ||
       !strcmp(group, "/..") ||
       (strlen(group) >= 3 && !strcmp(group + strlen(group) - 3, "/.."))) {
      free(line);
      return false;
   }

   file = fopen(proc_mountinfo, "re");
   if (!file) {
      free(line);
      return false;
   }

   bool limited = false;
   uint64_t remaining = UINT64_MAX;
   while (getline(&line, &capacity, file) >= 0) {
      if (!strstr(line, " - cgroup2 "))
         continue;

      char *save;
      char *root = NULL, *mount = NULL;
      char *token = strtok_r(line, " ", &save);
      for (unsigned field = 1; token && field <= 5; ++field) {
         if (field == 4)
            root = token;
         if (field == 5)
            mount = token;
         token = strtok_r(NULL, " ", &save);
      }
      if (!root || !mount)
         continue;
      unescape(root);
      unescape(mount);
      size_t root_len = strlen(root);
      const char *relative;
      if (!strcmp(root, "/")) {
         relative = group;
      } else if (!strncmp(group, root, root_len) &&
                 (group[root_len] == '/' || group[root_len] == 0)) {
         relative = group + root_len;
      } else {
         continue;
      }

      uint64_t free;
      if (available_in_hierarchy(mount, relative, &free)) {
         if (free < remaining)
            remaining = free;
         limited = true;
      }
   }
   free(line);
   fclose(file);
   if (limited)
      *available = remaining;
   return limited;
}
