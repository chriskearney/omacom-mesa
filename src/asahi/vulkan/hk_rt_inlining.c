/* Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */
#include "hk_rt_inlining.h"
#include "compiler/nir/nir.h"
#include "util/hash_table.h"

static uint64_t
expanded_size(nir_function *function, struct hash_table *sizes, uint32_t limit)
{
   struct hash_entry *entry = _mesa_hash_table_search(sizes, function);
   if (entry)
      return entry->data ? (uintptr_t)entry->data - 1 : (uint64_t)limit + 1;

   /* NULL is an in-progress mark: a cycle must never reach the inliner. */
   if (!_mesa_hash_table_insert(sizes, function, NULL) || !function->impl)
      return (uint64_t)limit + 1;

   uint64_t size = 0;
   nir_foreach_block(block, function->impl) {
      nir_foreach_instr(instr, block) {
         ++size;
         if (instr->type == nir_instr_type_call)
            size += expanded_size(nir_instr_as_call(instr)->callee, sizes, limit);
         if (size > limit)
            return (uint64_t)limit + 1;
      }
   }
   entry = _mesa_hash_table_search(sizes, function);
   entry->data = (void *)(uintptr_t)(size + 1);
   return size;
}

bool
hk_rt_inlining_fits(nir_shader *shader, uint32_t limit)
{
   struct hash_table *sizes = _mesa_pointer_hash_table_create(NULL);
   if (!sizes)
      return false;
   uint64_t size = expanded_size(nir_shader_get_entrypoint(shader)->function,
                                 sizes, limit);
   _mesa_hash_table_destroy(sizes, NULL);
   return size <= limit;
}
