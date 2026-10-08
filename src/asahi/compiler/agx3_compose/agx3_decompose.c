/*
 * agx3_decompose.c -- structural decomposer over the AGX3 encoder tables.
 * See agx3_decompose.h. Based on the standalone decomposer (decompose.c)
 * by DjDeveloperr, MIT.
 * SPDX-License-Identifier: MIT
 */
#include "agx3_decompose.h"

#include <string.h>

#define MAXLUT  24
#define MAXPEND 64

struct sol {
   uint64_t fields[AGX3_MAX_FIELDS];
   uint8_t known[AGX3_MAX_FIELDS];
   const agx3_cmd *lut[MAXLUT];
   unsigned nlut;
};

struct pending {
   const agx3_group *g[MAXPEND];
   unsigned n;
};

struct dctx {
   const agx3_table *t;
   uint16_t want[AGX3_MAX_WORDS];
   unsigned nwords;
   bool found;
   uint32_t form;
   uint64_t fields[AGX3_MAX_FIELDS];
};

static bool
fixed_ok(const struct dctx *c, const agx3_node *n)
{
   for (uint16_t i = 0; i < n->fixed_n; i++) {
      const agx3_fixed *f = &c->t->fixeds[n->fixed_off + i];
      if (f->word >= c->nwords)
         return false;
      if ((c->want[f->word] & f->mask) != (f->value & f->mask))
         return false;
   }
   return true;
}

static void
apply_preds(const struct dctx *c, struct sol *s, uint32_t off, uint16_t n)
{
   for (uint16_t i = 0; i < n; i++) {
      const agx3_pred *p = &c->t->preds[off + i];
      if (p->field >= AGX3_MAX_FIELDS)
         continue;
      if (p->kind == 0) {
         s->fields[p->field] = p->a;
         s->known[p->field] = 1;
      } else if (p->kind == 3) {
         s->fields[p->field] |= (p->a << p->shift);
      }
   }
}

static uint64_t
cmd_mask(const agx3_cmd *cmd)
{
   return cmd->width >= 64 ? ~0ull : ((1ull << cmd->width) - 1);
}

static bool
apply_cmds(const struct dctx *c, struct sol *s, const agx3_node *n)
{
   for (uint16_t i = 0; i < n->cmd_n; i++) {
      const agx3_cmd *cmd = &c->t->cmds[n->cmd_off + i];
      if (cmd->field >= AGX3_MAX_FIELDS || cmd->word >= c->nwords)
         return false;
      const uint64_t mask = cmd_mask(cmd);
      const uint64_t bits = (c->want[cmd->word] >> cmd->dst_shift) & mask;
      if (cmd->lut_off >= 0) {
         if (s->known[cmd->field]) {
            const uint64_t v = s->fields[cmd->field];
            if (v >= cmd->lut_n)
               return false;
            if ((((uint64_t)c->t->lutpool[cmd->lut_off + v] >>
                  cmd->src_shift) & mask) != bits)
               return false;
            continue;
         }
         if (s->nlut >= MAXLUT)
            return false;
         s->lut[s->nlut++] = cmd;
      } else {
         s->fields[cmd->field] |= bits << cmd->src_shift;
      }
   }
   return true;
}

static void
resolve_luts(struct dctx *c, struct sol *s, unsigned k, uint32_t form)
{
   if (c->found)
      return;
   if (k == s->nlut) {
      uint8_t out[2 * AGX3_MAX_WORDS];
      if (!agx3_compose(c->t, form, s->fields, out))
         return;
      for (unsigned w = 0; w < c->nwords; w++)
         if (((uint16_t)out[2 * w] | ((uint16_t)out[2 * w + 1] << 8)) !=
             c->want[w])
            return;
      c->found = true;
      c->form = form;
      memcpy(c->fields, s->fields, sizeof(c->fields));
      return;
   }
   const agx3_cmd *cmd = s->lut[k];
   const uint64_t mask = cmd_mask(cmd);
   const uint64_t bits = (c->want[cmd->word] >> cmd->dst_shift) & mask;
   const uint64_t save = s->fields[cmd->field];
   for (uint16_t v = 0; v < cmd->lut_n && !c->found; v++) {
      if ((((uint64_t)c->t->lutpool[cmd->lut_off + v] >> cmd->src_shift) &
           mask) != bits)
         continue;
      s->fields[cmd->field] = v;
      resolve_luts(c, s, k + 1, form);
      s->fields[cmd->field] = save;
   }
}

static void
push_groups(const struct dctx *c, struct pending *p, const agx3_node *n)
{
   for (uint16_t i = 0; i < n->grp_n && p->n < MAXPEND; i++)
      p->g[p->n++] = &c->t->groups[n->grp_off + i];
}

static void
solve(struct dctx *c, const struct pending *p, const struct sol *s,
      uint32_t form)
{
   if (c->found)
      return;
   if (p->n == 0) {
      struct sol s2 = *s;
      resolve_luts(c, &s2, 0, form);
      return;
   }
   struct pending rest = *p;
   const agx3_group *g = rest.g[--rest.n];
   for (uint16_t a = 0; a < g->alt_n && !c->found; a++) {
      const uint32_t alt = c->t->altpool[g->alt_off + a];
      const agx3_node *an = &c->t->nodes[alt];
      if (!fixed_ok(c, an))
         continue;
      struct sol s2 = *s;
      struct pending p2 = rest;
      apply_preds(c, &s2, an->pred_off, an->pred_n);
      if (!apply_cmds(c, &s2, an))
         continue;
      push_groups(c, &p2, an);
      solve(c, &p2, &s2, form);
   }
}

bool
agx3_decompose(const agx3_table *t, const uint8_t *code, size_t avail,
               struct agx3_decoded *out)
{
   struct dctx c;
   c.t = t;
   for (uint32_t f = 0; f < t->num_forms; f++) {
      const unsigned size = t->forms[f].size;
      if (size > avail || size > 2 * AGX3_MAX_WORDS || (size & 1))
         continue;
      c.nwords = size / 2;
      for (unsigned w = 0; w < c.nwords; w++)
         c.want[w] = (uint16_t)code[2 * w] | ((uint16_t)code[2 * w + 1] << 8);
      const agx3_node *root = &t->nodes[t->forms[f].node];
      if (!fixed_ok(&c, root))
         continue;
      struct sol s;
      memset(&s, 0, sizeof(s));
      apply_preds(&c, &s, root->pred_off, root->pred_n);
      if (!apply_cmds(&c, &s, root))
         continue;
      struct pending p;
      p.n = 0;
      push_groups(&c, &p, root);
      c.found = false;
      solve(&c, &p, &s, f);
      if (c.found) {
         out->form = f;
         out->size = size;
         memcpy(out->fields, c.fields, sizeof(out->fields));
         return true;
      }
   }
   return false;
}

/* Record every single-bit direct deposit into encoded bits 12..17. */
static void
collect_wait_bits(const agx3_table *t, uint32_t node_id,
                  uint64_t cand[6][(AGX3_MAX_FIELDS + 63) / 64], unsigned depth)
{
   if (depth > 16)
      return;
   const agx3_node *n = &t->nodes[node_id];
   for (uint16_t i = 0; i < n->cmd_n; i++) {
      const agx3_cmd *cmd = &t->cmds[n->cmd_off + i];
      const unsigned bit = cmd->word * 16u + cmd->dst_shift;
      if (cmd->lut_off >= 0 || cmd->width != 1 || cmd->src_shift != 0 ||
          bit < 12 || bit > 17 || cmd->field >= AGX3_MAX_FIELDS)
         continue;
      cand[bit - 12][cmd->field / 64] |= 1ull << (cmd->field % 64);
   }
   for (uint16_t g = 0; g < n->grp_n; g++) {
      const agx3_group *grp = &t->groups[n->grp_off + g];
      for (uint16_t a = 0; a < grp->alt_n; a++)
         collect_wait_bits(t, t->altpool[grp->alt_off + a], cand, depth + 1);
   }
}

bool
agx3_wait_fields(const agx3_table *t, uint32_t form, uint16_t ids[6])
{
   const char *mnemonic = t->forms[form].mnemonic;

   if (!strcmp(mnemonic, "end") || !strcmp(mnemonic, "nop")) {
      for (unsigned j = 0; j < 6; ++j)
         ids[j] = 2 + j;
      return true;
   }

   uint64_t cand[6][(AGX3_MAX_FIELDS + 63) / 64];
   memset(cand, 0, sizeof(cand));
   for (uint32_t f = 0; f < t->num_forms; f++) {
      if (strcmp(t->forms[f].mnemonic, mnemonic))
         continue;
      collect_wait_bits(t, t->forms[f].node, cand, 0);
   }

   /* The six wait bits are six consecutive field ids. */
   for (unsigned base = 0; base + 6 <= AGX3_MAX_FIELDS; ++base) {
      bool all = true;
      for (unsigned j = 0; j < 6 && all; ++j)
         all = (cand[j][(base + j) / 64] >> ((base + j) % 64)) & 1;
      if (all) {
         for (unsigned j = 0; j < 6; ++j)
            ids[j] = base + j;
         return true;
      }
   }
   return false;
}
