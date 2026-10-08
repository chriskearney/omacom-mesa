/*
 * Copyright 2026 Satya Benson
 * SPDX-License-Identifier: MIT
 *
 * G15G scheduling-stall insertion over a finished AGX3b instruction stream.
 *
 * The rule:
 *
 *  - An instruction is a register producer of slot s when its signal slot
 *    (f12) is s (1..6) and its destination file (f9) is 2, the general
 *    registers: device and threadgroup loads, special-register reads,
 *    texture results, coefficient loads and threadgroup atomics with a
 *    return. Producers into the staging banks (f9 = 4 or 5) are not.
 *  - A forward may-dataflow tracks the set P of slots that may still owe a
 *    register result: a producer adds its slot, a wait mask removes its
 *    slots. Edges are fallthrough (not after `end`), b/bn targets, and an
 *    edge from each exec-mask push/else to the else/pop that closes its
 *    region, because a region with no active lanes runs as a no-op. Each
 *    program starts with P empty: the stream start, and the instruction
 *    after an `end` plus zero-nop filler.
 *  - Every instruction whose wait mask W meets P gets a stall in front of it
 *    with the value W & P. `end` counts as waiting on all six slots. A
 *    2-byte waiting nop that covers pending slots becomes the stall itself.
 *  - b/bn displacements (48-bit, from the branch start) are recomputed over
 *    the padded stream and land on the target's stall, so the wait also runs
 *    on the branch path.
 *
 * It works on bytes, not IR, so every emitter -- including hand-assembled
 * sequences -- is covered. Each instruction is decomposed into its field
 * vector with the composer tables (agx3_decompose), which also yields its
 * length. test/test-agx3-g15g-stalls.c checks it. The fields used are:
 *
 *    wait mask  six consecutive per-mnemonic field ids (see wait_ids())
 *    f12        signal slot 1..6 of an asynchronous producer (push count on
 *               cndst, depth elsewhere in control flow)
 *    f9         destination file: 2 = general registers, 4/5 = the staging
 *               banks read by smp/imgwci/atomics/uvsw, 0/1 = control/select
 *    f10        pop count of cndend
 *    f0         48-bit signed displacement of b/bn from the branch's start
 */

#include "agx3_g15g_stalls.h"

#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#include "agx3_compose/agx3_compose.h"
#include "agx3_compose/agx3_decompose.h"

enum kind {
   K_OTHER,
   K_NOP,
   K_END,
   K_B,       /* branch if any lane active: backward loop edge */
   K_BN,      /* branch if no lane active: forward skip */
   K_CNDST,   /* exec-mask push (if / loop entry) */
   K_CNDEF,   /* else */
   K_CNDEND,  /* pop */
   K_CNDLT,   /* loop/break test */
   K_CNDCF,   /* break across nested conditionals */
};

struct ins {
   uint32_t off;
   uint8_t size;
   uint8_t kind;
   uint8_t wait;       /* scoreboard wait mask, slot n = bit n-1 */
   uint8_t gpr_slot;   /* bit of the slot this signals with a GPR result */
   uint8_t push, pop;  /* exec-mask stack effect of cndst / cndend */
   uint32_t form;
   int64_t disp;       /* b/bn */
   int32_t target;     /* b/bn target instruction index */
   int32_t succ[3];
   uint8_t nsucc;
   uint8_t in;         /* slots with a possibly-outstanding GPR producer */
   uint8_t value;      /* wait mask of the inserted stall */
   bool reached;
   bool stall;         /* insert a stall in front */
   bool nop_stall;     /* a waiting nop that becomes the stall itself */
};

/*
 * Per-mnemonic id of the wait field for slot 1; slots 2..6 follow. Long
 * forms deposit these at encoded bits 12..17, which agx3_wait_fields() finds
 * automatically. The mnemonics below compress or relocate the mask in every
 * form, so their ids are listed here; they agree with Mesa's own packers
 * (agx3_alu_table.h, agx3_sr_core.c, agx3_tex_core.c, agx3_stage_core.c).
 */
static const struct {
   const char *mnemonic;
   uint8_t base;
} wait_bases[] = {
   {"bit_and", 44},  {"bit_andc", 44}, {"bit_nand", 44}, {"bit_nor", 44},
   {"bit_or", 44},   {"bit_xor", 44},  {"clr", 16},      {"copy", 30},
   {"f16add", 51},   {"f16fma", 68},   {"f16mul", 51},   {"f32add", 52},
   {"f32fma", 97},   {"f32mul", 80},   {"fcmp", 38},     {"fsel", 102},
   {"icmp", 34},     {"inv", 30},      {"isel", 98},     {"sel", 98},
   {"ldimm", 17},    {"rld", 30},      {"smp", 62},
};

static bool
wait_ids(const agx3_table *t, uint32_t form, uint16_t ids[6])
{
   const char *mnemonic = t->forms[form].mnemonic;
   for (unsigned i = 0; i < sizeof(wait_bases) / sizeof(wait_bases[0]); ++i) {
      if (!strcmp(wait_bases[i].mnemonic, mnemonic)) {
         for (unsigned j = 0; j < 6; ++j)
            ids[j] = wait_bases[i].base + j;
         return true;
      }
   }
   return agx3_wait_fields(t, form, ids);
}

static enum kind
classify(const char *m)
{
   static const struct { const char *m; enum kind k; } kinds[] = {
      {"nop", K_NOP},       {"end", K_END},       {"b", K_B},
      {"bn", K_BN},         {"cndst", K_CNDST},   {"cndef", K_CNDEF},
      {"cndend", K_CNDEND}, {"cndlt", K_CNDLT},   {"cndcf", K_CNDCF},
   };
   for (unsigned i = 0; i < sizeof(kinds) / sizeof(kinds[0]); ++i)
      if (!strcmp(kinds[i].m, m))
         return kinds[i].k;
   return K_OTHER;
}

static bool
is_zero_nop(const struct ins *I)
{
   return I->kind == K_NOP && I->wait == 0 && I->size == 2;
}

static int32_t
find_ins(const struct ins *v, unsigned n, int64_t off)
{
   unsigned lo = 0, hi = n;
   while (lo < hi) {
      unsigned mid = (lo + hi) / 2;
      if (v[mid].off < off)
         lo = mid + 1;
      else
         hi = mid;
   }
   return (lo < n && v[lo].off == off) ? (int32_t)lo : -1;
}

static void
add_succ(struct ins *I, int32_t s)
{
   for (unsigned i = 0; i < I->nsucc; ++i)
      if (I->succ[i] == s)
         return;
   if (I->nsucc < 3)
      I->succ[I->nsucc++] = s;
}

int
agx3_g15g_insert_stalls(const uint8_t *in, size_t in_size, uint8_t **out,
                        size_t *out_size, uint32_t *offset_map)
{
   const agx3_table *t = &agx3b_table;
   *out = NULL;
   *out_size = 0;
   if (in_size & 1)
      return -1;

   /* Decode. */
   const unsigned cap = in_size / 2 + 1;
   struct ins *v = calloc(cap, sizeof(*v));
   uint32_t *start = calloc(cap, sizeof(*start)); /* stall-or-instruction */
   uint32_t *self = calloc(cap, sizeof(*self));   /* the instruction itself */
   uint8_t *buf = NULL;
   int ret = -1;
   if (!v || !start || !self)
      goto done;

   unsigned n = 0;
   for (size_t off = 0; off < in_size;) {
      struct agx3_decoded d;
      if (!agx3_decompose(t, in + off, in_size - off, &d))
         goto done;

      struct ins *I = &v[n++];
      I->off = off;
      I->size = d.size;
      I->form = d.form;
      I->kind = classify(t->forms[d.form].mnemonic);
      I->target = -1;

      uint16_t ids[6];
      if (wait_ids(t, d.form, ids)) {
         for (unsigned j = 0; j < 6; ++j)
            I->wait |= (d.fields[ids[j]] & 1) << j;
      }

      /* Asynchronous producers carry their signal slot in f12 and their
       * destination file in f9: 2 is the general register file; 4 and 5
       * are the staging banks read by the texture, image, atomic and
       * varying units, whose consumers never need a stall. The exec-mask ops
       * and selects reuse f12 for nesting with f9 = 0 or 1. */
      const uint64_t slot = d.fields[12], file = d.fields[9];
      if (I->kind == K_OTHER && slot >= 1 && slot <= 6 && file == 2)
         I->gpr_slot = 1u << (slot - 1);

      if (I->kind == K_CNDST)
         I->push = slot ? slot : 1;
      if (I->kind == K_CNDEND)
         I->pop = d.fields[10] ? d.fields[10] : 1;
      if (I->kind == K_B || I->kind == K_BN) {
         uint64_t raw = d.fields[0] & ((1ull << 48) - 1);
         I->disp = (raw >> 47) ? (int64_t)(raw | ~((1ull << 48) - 1))
                               : (int64_t)raw;
      }
      off += d.size;
   }

   /* Control flow. */
   for (unsigned k = 0; k < n; ++k) {
      struct ins *I = &v[k];
      if (I->kind == K_B || I->kind == K_BN) {
         I->target = find_ins(v, n, (int64_t)I->off + I->disp);
         if (I->target < 0 || I->disp == 0)
            goto done;
         add_succ(I, I->target);
      }
      if (I->kind != K_END && k + 1 < n)
         add_succ(I, k + 1);

      /* The exec-mask stack runs a region with no active lanes as a no-op,
       * so a wait inside it may not have happened after it closes: model the
       * skip as an edge to the region's else/pop. */
      if (I->kind == K_CNDST || I->kind == K_CNDEF || I->kind == K_CNDLT ||
          I->kind == K_CNDCF) {
         int depth = 0;
         for (unsigned j = k + 1; j < n; ++j) {
            if (v[j].kind == K_CNDST) {
               depth += v[j].push;
            } else if (v[j].kind == K_CNDEF && depth == 0 &&
                       I->kind == K_CNDST) {
               add_succ(I, j);
               break;
            } else if (v[j].kind == K_CNDEND) {
               depth -= v[j].pop;
               if (depth < 0) {
                  add_succ(I, j);
                  break;
               }
            }
         }
      }
   }

   /* Entries: the stream start and anything after `end` plus nop filler
    * (the code container places the main program after the constant one). */
   v[0].reached = n > 0;
   for (unsigned k = 1; k < n; ++k) {
      if (v[k].kind == K_NOP)
         continue;
      int j = (int)k - 1;
      while (j >= 0 && is_zero_nop(&v[j]))
         --j;
      if (j >= 0 && v[j].kind == K_END)
         v[k].reached = true;
   }

   /* Forward may-dataflow of GPR-producing slots not yet waited on. */
   bool changed = true;
   while (changed) {
      changed = false;
      for (unsigned k = 0; k < n; ++k) {
         const struct ins *I = &v[k];
         if (!I->reached)
            continue;
         const uint8_t o = (uint8_t)((I->in & ~I->wait) | I->gpr_slot);
         for (unsigned s = 0; s < I->nsucc; ++s) {
            struct ins *S = &v[I->succ[s]];
            if (!S->reached || (S->in | o) != S->in) {
               S->reached = true;
               S->in |= o;
               changed = true;
            }
         }
      }
   }

   /* Placement: in front of every instruction whose wait covers a slot that
    * may still owe a register result, waiting on exactly those slots. The
    * program's `end` implicitly waits on everything, so it drains whatever
    * register producers are still in flight. */
   unsigned extra = 0, stalls = 0;
   for (unsigned k = 0; k < n; ++k) {
      struct ins *I = &v[k];
      const uint8_t wait = I->kind == K_END ? 0x3f : I->wait;
      I->value = wait & I->in;
      if (!I->value)
         continue;
      if (I->kind == K_NOP && I->size == 2) {
         /* Already a stall (e.g. re-running over padded code)? */
         if (k + 3 < n && is_zero_nop(&v[k + 1]) && is_zero_nop(&v[k + 2]) &&
             is_zero_nop(&v[k + 3]))
            continue;
         I->nop_stall = true;
         extra += AGX3_G15G_STALL_SIZE - 2;
      } else {
         I->stall = true;
         extra += AGX3_G15G_STALL_SIZE;
      }
      stalls++;
   }

   /* Layout. */
   uint32_t pos = 0;
   for (unsigned k = 0; k < n; ++k) {
      struct ins *I = &v[k];
      start[k] = pos;
      if (I->stall)
         pos += AGX3_G15G_STALL_SIZE;
      self[k] = pos;
      pos += I->size + (I->nop_stall ? AGX3_G15G_STALL_SIZE - 2 : 0);
   }
   if (pos != in_size + extra)
      goto done;

   buf = malloc(pos ? pos : 1);
   if (!buf)
      goto done;

   for (unsigned k = 0; k < n; ++k) {
      struct ins *I = &v[k];
      uint8_t *p = buf + start[k];
      if (I->stall) {
         p[0] = 0x06;
         p[1] = (uint8_t)(I->value << 2);
         for (unsigned i = 2; i < AGX3_G15G_STALL_SIZE; i += 2) {
            p[i] = 0x06;
            p[i + 1] = 0x00;
         }
      }
      uint8_t *q = buf + self[k];
      memcpy(q, in + I->off, I->size);
      if (I->nop_stall) {
         for (unsigned i = 2; i < AGX3_G15G_STALL_SIZE; i += 2) {
            q[i] = 0x06;
            q[i + 1] = 0x00;
         }
      }
      if (I->target >= 0) {
         /* Displacements are from the branch itself to the target's stall,
          * so the wait runs on the branch path too. */
         const int64_t disp = (int64_t)start[I->target] - (int64_t)self[k];
         if (disp != I->disp) {
            struct agx3_decoded d;
            if (!agx3_decompose(t, in + I->off, I->size, &d) ||
                d.form != I->form)
               goto done;
            d.fields[0] = (uint64_t)disp & ((1ull << 48) - 1);
            uint8_t enc[2 * AGX3_MAX_WORDS];
            if (!agx3_compose(t, d.form, d.fields, enc))
               goto done;
            memcpy(q, enc, I->size);
         }
      }
   }

   if (offset_map) {
      for (unsigned k = 0; k < n; ++k)
         offset_map[v[k].off / 2] = start[k];
      offset_map[in_size / 2] = pos;
   }

   *out = buf;
   *out_size = pos;
   buf = NULL;
   ret = (int)stalls;

done:
   free(buf);
   free(v);
   free(start);
   free(self);
   return ret;
}
