/*
 * Copyright 2026 Chris Kearney
 * SPDX-License-Identifier: MIT
 *
 * G15G scheduling-stall pass (agx3_g15g_stalls.c).
 *
 * For each vector in agx3_g15g_stall_vectors.h (parts this driver compiled):
 *  - the pass reproduces the recorded output and stall count;
 *  - the output is checked against the input independently of the pass:
 *    every input instruction sits at its mapped offset, either alone or
 *    behind one 8-byte stall (06 W<<2 06 00 06 00 06 00, W nonzero), with
 *    the same bytes; a 2-byte waiting nop may instead carry three 06 00
 *    words itself; every b/bn lands on the mapped offset of its input
 *    target with all other fields unchanged; nothing else is in the output;
 *  - running the pass again over its output inserts nothing.
 * It also checks that malformed input is refused rather than passed through.
 */

#include <inttypes.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "agx3_compose/agx3_compose.h"
#include "agx3_compose/agx3_decompose.h"
#include "agx3_g15g_stalls.h"
#include "agx3_g15g_stall_vectors.h"

static unsigned failures;

#define CHECK(cond, ...)                                                       \
   do {                                                                        \
      if (!(cond)) {                                                           \
         printf("FAIL %s: ", name);                                            \
         printf(__VA_ARGS__);                                                  \
         printf("\n");                                                         \
         failures++;                                                           \
         goto out;                                                             \
      }                                                                        \
   } while (0)

static size_t
unhex(const char *h, uint8_t **out)
{
   const size_t n = strlen(h) / 2;
   uint8_t *b = malloc(n ? n : 1);
   for (size_t i = 0; i < n; ++i) {
      unsigned v;
      sscanf(h + 2 * i, "%2x", &v);
      b[i] = (uint8_t)v;
   }
   *out = b;
   return n;
}

static bool
is_stall(const uint8_t *p, size_t avail)
{
   static const uint8_t tail[6] = {0x06, 0x00, 0x06, 0x00, 0x06, 0x00};
   return avail >= AGX3_G15G_STALL_SIZE && p[0] == 0x06 && p[1] != 0 &&
          (p[1] & 3) == 0 && !memcmp(p + 2, tail, sizeof(tail));
}

static int64_t
disp48(uint64_t raw)
{
   raw &= (1ull << 48) - 1;
   return (raw >> 47) ? (int64_t)(raw | ~((1ull << 48) - 1)) : (int64_t)raw;
}

/* Check out[] against in[] with the pass's offset map, without using the
 * pass's own layout logic. Returns the number of stalls found. */
static int
check_layout(const char *name, const uint8_t *in, size_t ni,
             const uint8_t *out, size_t no, const uint32_t *map)
{
   const agx3_table *t = &agx3b_table;
   int stalls = 0;
   size_t pos = 0;

   for (size_t off = 0; off < ni;) {
      struct agx3_decoded d;
      CHECK(agx3_decompose(t, in + off, ni - off, &d),
            "input does not decode at %zu", off);
      const char *m = t->forms[d.form].mnemonic;
      const bool branch = !strcmp(m, "b") || !strcmp(m, "bn");
      const bool nop2 = !strcmp(m, "nop") && d.size == 2;

      CHECK(map[off / 2] == pos, "instruction at %zu maps to %" PRIu32
            ", expected %zu", off, map[off / 2], pos);
      size_t q = pos;
      if (!nop2 && is_stall(out + q, no - q)) {
         stalls++;
         q += AGX3_G15G_STALL_SIZE;
      }
      CHECK(q + d.size <= no, "output too short at %zu", q);

      if (branch) {
         struct agx3_decoded o;
         CHECK(agx3_decompose(t, out + q, no - q, &o) && o.form == d.form &&
               o.size == d.size, "branch at %zu changed form", off);
         CHECK(!memcmp(o.fields + 1, d.fields + 1,
                       sizeof(d.fields) - sizeof(d.fields[0])),
               "branch at %zu changed a field other than its displacement", off);
         const int64_t target = (int64_t)off + disp48(d.fields[0]);
         CHECK(target >= 0 && target < (int64_t)ni && !(target & 1),
               "input branch at %zu has a bad target", off);
         CHECK((int64_t)q + disp48(o.fields[0]) == (int64_t)map[target / 2],
               "branch at %zu lands at %" PRId64 ", its target maps to %" PRIu32,
               off, (int64_t)q + disp48(o.fields[0]), map[target / 2]);
      } else {
         CHECK(!memcmp(out + q, in + off, d.size),
               "instruction at %zu changed", off);
      }
      q += d.size;

      /* A waiting nop that covered pending slots became the stall itself. */
      if (nop2 && in[off + 1] != 0 && is_stall(out + pos, no - pos)) {
         stalls++;
         q += AGX3_G15G_STALL_SIZE - 2;
      }
      pos = q;
      off += d.size;
   }
   CHECK(map[ni / 2] == pos && pos == no,
         "output has %zu bytes, the instructions cover %zu", no, pos);
   return stalls;
out:
   return -1;
}

static void
check_vector(const struct g15g_stall_vector *v)
{
   const char *name = v->name;
   uint8_t *in, *exp, *out = NULL, *again = NULL;
   const size_t ni = unhex(v->input, &in), ne = unhex(v->expected, &exp);
   size_t no = 0, na = 0;
   uint32_t *map = calloc(ni / 2 + 1, sizeof(*map));
   uint32_t *map2 = NULL;

   const int stalls = agx3_g15g_insert_stalls(in, ni, &out, &no, map);
   CHECK(stalls >= 0, "pass refused the input");
   CHECK(no == ne && !memcmp(out, exp, ne),
         "output differs from the recorded output (%zu vs %zu bytes)", no, ne);
   CHECK(stalls == v->stalls, "%d stalls, expected %d", stalls, v->stalls);

   const int found = check_layout(name, in, ni, out, no, map);
   if (found < 0)
      goto out;
   CHECK(found == stalls, "layout check found %d stalls, the pass reports %d",
         found, stalls);

   map2 = calloc(no / 2 + 1, sizeof(*map2));
   const int again_stalls = agx3_g15g_insert_stalls(out, no, &again, &na, map2);
   CHECK(again_stalls == 0 && na == no && !memcmp(again, out, no),
         "a second run changed the output (%d stalls)", again_stalls);

   printf("ok   %-15s %5zu -> %5zu bytes, %2d stalls  %s\n", name, ni, no,
          stalls, v->what);
out:
   free(in);
   free(exp);
   free(out);
   free(again);
   free(map);
   free(map2);
}

static void
check_refused(const char *name, const uint8_t *in, size_t n)
{
   uint8_t *out = (uint8_t *)1;
   size_t no = 1;
   const int r = agx3_g15g_insert_stalls(in, n, &out, &no, NULL);
   CHECK(r < 0 && out == NULL && no == 0, "accepted (%d)", r);
   printf("ok   %-15s refused\n", name);
out:
   if (r >= 0)
      free(out);
}

int
main(void)
{
   const unsigned n = sizeof(g15g_stall_vectors) / sizeof(g15g_stall_vectors[0]);
   for (unsigned i = 0; i < n; ++i)
      check_vector(&g15g_stall_vectors[i]);

   /* Odd length: never a whole instruction stream. */
   static const uint8_t odd[] = {0x0e, 0x00, 0x00};
   check_refused("odd-length", odd, sizeof(odd));

   /* A stream whose last instruction is cut short. */
   uint8_t *cut;
   const size_t ncut = unhex(g15g_stall_vectors[1].input, &cut);
   check_refused("truncated", cut, ncut >= 6 ? 6 : ncut);
   free(cut);

   /* Bytes that are not an AGX3b instruction. */
   static const uint8_t junk[] = {0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff};
   check_refused("undecodable", junk, sizeof(junk));

   printf("%u vectors, %u failures\n", n, failures);
   return failures ? 1 : 0;
}
