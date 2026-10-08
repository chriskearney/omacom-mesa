/*
 * agx3_decompose.h -- inverse of agx3_compose: recover a form and its field
 * vector from encoded AGX3 instruction bytes.
 *
 * Structural search over the same tables the encoder uses: enumerate each
 * candidate form's choice-group alternatives, derive the field values the
 * target bytes imply (predicates pin selectors, direct deposits recover
 * operands, LUT deposits are inverted by enumeration), then keep a candidate
 * only if agx3_compose() re-encodes it to exactly the same bytes. A reported
 * decomposition therefore always round-trips through the encoder.
 *
 * Based on the standalone decomposer (decompose.c) by DjDeveloperr, MIT;
 * made reentrant and extended to find the instruction length itself.
 */
#ifndef AGX3_DECOMPOSE_H
#define AGX3_DECOMPOSE_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "agx3_compose.h"

#ifdef __cplusplus
extern "C" {
#endif

struct agx3_decoded {
   uint32_t form;
   uint8_t size;
   uint64_t fields[AGX3_MAX_FIELDS];
};

/* Decode the instruction at the start of code (avail bytes available). The
 * first form in table order that round-trips wins, which also fixes the
 * instruction length. Returns false if no form reproduces the bytes. */
bool agx3_decompose(const agx3_table *t, const uint8_t *code, size_t avail,
                    struct agx3_decoded *out);

/* Field ids of the six scoreboard wait bits (slot 1..6 = bit 0..5) of a form,
 * or false if the form's mnemonic has no wait field. Long forms deposit the
 * wait mask directly into encoded bits 12..17; short forms of the same
 * mnemonic compress it through a LUT but keep the field ids, so the ids are
 * taken from any form of the mnemonic that deposits them directly. `end` and
 * `nop` carry theirs at bits 10..15 (fields 2..7). */
bool agx3_wait_fields(const agx3_table *t, uint32_t form, uint16_t ids[6]);

#ifdef __cplusplus
}
#endif

#endif
