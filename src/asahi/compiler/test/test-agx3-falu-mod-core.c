/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * The f32 modifier encoders must reproduce the composer's bytes -- the same
 * recipe (baseline + operands + modifiers) the RE proved byte-exact against
 * Apple -- across the abs/neg/saturate combinations of add, mul and fma.
 */

#include "agx3_falu_mod_core.h"
#include "agx3_falu_mod_core_vectors.c"

#include <stdio.h>
#include <string.h>

static unsigned
uniform_binary_vectors(void)
{

   const struct {
      bool multiply; uint16_t dest, uniform, reg;
      bool uniform_neg, reg_neg, dest_cache, cache, discard;
      uint8_t size, bytes[16];
   } vectors[] = {
      {1,12,42,12,0,0,1,0,1,6,{0x69,0x0d,0x2d,0x2b,0x00,0x02}},
      {0,18,44,2,1,0,1,1,0,6,{0x99,0x83,0x24,0x2d,0x00,0x0a}},
      {0,14,46,0,1,0,1,1,0,6,{0x79,0x81,0x24,0x2f,0x00,0x0a}},
      {0,14,50,0,1,0,1,1,0,6,{0x79,0x81,0x24,0x33,0x00,0x0a}},
      {0,18,48,2,1,0,1,1,0,6,{0x99,0x83,0x24,0x31,0x00,0x0a}},
      {1,90,32,2,0,0,1,0,1,6,{0xd9,0x03,0xad,0x21,0x00,0x02}},
      {1,88,34,0,0,0,1,0,1,6,{0xc9,0x01,0xad,0x23,0x00,0x02}},
      {1,88,38,88,0,0,1,0,1,6,{0xc9,0x59,0xad,0x27,0x00,0x02}},
      {1,90,36,90,0,0,1,0,1,6,{0xd9,0x5b,0xad,0x25,0x00,0x02}},
      {0,52,40,62,1,0,1,0,1,6,{0xa9,0x3f,0x6c,0x29,0x00,0x0a}},
      {1,18,32,2,0,0,1,1,0,6,{0x99,0x83,0x25,0x21,0x00,0x02}},
      {1,16,34,0,0,0,1,1,0,6,{0x89,0x81,0x25,0x23,0x00,0x02}},
      {1,18,36,18,0,0,1,0,1,6,{0x99,0x13,0x2d,0x25,0x00,0x02}},
      {1,16,38,16,0,0,1,0,1,6,{0x89,0x11,0x2d,0x27,0x00,0x02}},
      {0,14,20,18,1,0,1,1,0,6,{0x79,0x93,0x24,0x15,0x00,0x0a}},
      {1,14,28,14,0,0,1,0,1,6,{0x79,0x0f,0x2d,0x1d,0x00,0x02}},
      {0,16,24,12,1,0,1,1,0,6,{0x89,0x8d,0x24,0x19,0x00,0x0a}},
      {1,16,30,16,0,0,1,0,1,6,{0x89,0x11,0x2d,0x1f,0x00,0x02}},
      {1,12,62,12,0,0,1,0,1,6,{0x69,0x0d,0x2d,0x3f,0x00,0x02}},
   };
   unsigned fail = 0;
   uint8_t bytes[16], size = 0;
   for (unsigned i = 0; i < sizeof(vectors)/sizeof(vectors[0]); ++i) {
      const struct agx3_falu_runtime rt = {
         .dest_cache=vectors[i].dest_cache,
         .cache={false,vectors[i].cache},
         .discard={false,vectors[i].discard},
      };
      if (!agx3_falu_binary_uniform(AGX_ISA_ENCODING_AGX3B,
              vectors[i].multiply, vectors[i].dest, vectors[i].uniform,
              vectors[i].reg, vectors[i].uniform_neg, vectors[i].reg_neg,
              &rt, bytes, &size) || size != vectors[i].size ||
          memcmp(bytes,vectors[i].bytes,size)) {
         fprintf(stderr,"native uniform binary vector %u failed\n",i);
         fprintf(stderr,"actual (%u): ",size);
         for (unsigned b=0; b<size; ++b) fprintf(stderr,"%02x",bytes[b]);
         fprintf(stderr,"\n");
         fail++;
      }
   }
   struct agx3_falu_runtime rt = {.wait_mask=63};
   for (unsigned mul = 0; mul < 2; ++mul) {
      rt.wait_mask = 63;
      fail += !agx3_falu_binary_uniform(AGX_ISA_ENCODING_AGX3B,mul,
         254,510,254,true,true,&rt,bytes,&size);
      fail += agx3_falu_binary_uniform(AGX_ISA_ENCODING_AGX3C,mul,
         0,0,0,false,false,&rt,bytes,&size);
      for (unsigned s = 0; s < 3; ++s) {
         uint16_t v[3] = {0}; v[s] = 1;
         fail += agx3_falu_binary_uniform(AGX_ISA_ENCODING_AGX3B,mul,
            v[0],v[1],v[2],false,false,&rt,bytes,&size);
         v[s] = s == 1 ? 512 : 256;
         fail += agx3_falu_binary_uniform(AGX_ISA_ENCODING_AGX3B,mul,
            v[0],v[1],v[2],false,false,&rt,bytes,&size);
      }
      rt.wait_mask = 64;
      fail += agx3_falu_binary_uniform(AGX_ISA_ENCODING_AGX3B,mul,
         0,0,0,false,false,&rt,bytes,&size);
      rt.wait_mask = 0;
      rt.cache[0] = true;
      fail += agx3_falu_binary_uniform(AGX_ISA_ENCODING_AGX3B,mul,
         0,0,0,false,false,&rt,bytes,&size);
      rt.cache[0] = false; rt.discard[0] = true;
      fail += agx3_falu_binary_uniform(AGX_ISA_ENCODING_AGX3B,mul,
         0,0,0,false,false,&rt,bytes,&size);
      rt.discard[0] = false; rt.cache[1] = rt.discard[1] = true;
      fail += agx3_falu_binary_uniform(AGX_ISA_ENCODING_AGX3B,mul,
         0,0,0,false,false,&rt,bytes,&size);
      rt.cache[1] = rt.discard[1] = false;
   }
   printf("native uniform ADD/MUL: %zu Apple vectors, %u failures\n",
      sizeof(vectors)/sizeof(vectors[0]),fail);
   return fail;
}

static unsigned
mixed_fma_vectors(void)
{
   const struct mixed_vector { unsigned mask; uint16_t a,b,c; uint8_t bytes[8]; } vectors[] = {
      {0,4,0,8,{0x09,0x01,0x1e,0x05,0x81,0x08,0x02,0xc0}},
      {2,0,8,4,{0x09,0x08,0x1e,0x01,0x81,0x04,0x02,0xc0}},
      {1,8,0,4,{0x09,0x01,0x1e,0x08,0x81,0x04,0x02,0xc0}},
      {3,5,4,0,{0x09,0x04,0x1e,0x04,0x81,0x00,0x12,0xc0}},
      {4,4,0,8,{0x09,0x01,0x1e,0x05,0x81,0x08,0x00,0xc0}},
      {6,0,4,5,{0x09,0x04,0x1e,0x01,0x81,0x05,0x00,0xc0}},
      {5,4,0,5,{0x09,0x01,0x1e,0x04,0x81,0x05,0x00,0xc0}},
      {7,1,0,4,{0x09,0x00,0x1e,0x00,0x81,0x04,0x10,0xc0}},
      {3,1,0,4,{0x09,0x00,0x1e,0x00,0x81,0x04,0x12,0xc0}},
   };
   const struct agx3_falu_mods mods = {0};
   struct agx3_falu_runtime rt = {.discard={true,true,true}, .wait_mask=32};
   unsigned fail = 0;
   for (unsigned i=0; i<sizeof(vectors)/sizeof(vectors[0]); ++i) {
      uint8_t out[16], length=0;
      const struct mixed_vector *v=&vectors[i];
      if (!agx3_falu_fma_mixed_mod_runtime(AGX_ISA_ENCODING_AGX3B,
             0,false,v->mask,v->a,v->b,v->c,&mods,&rt,out,&length) ||
          length!=8 || memcmp(out,v->bytes,8)) {
         fprintf(stderr,"mixed FMA Apple vector %u failed\n",i);
         ++fail;
      }
   }
   uint8_t out[16], length = 0;
   rt = (struct agx3_falu_runtime){.wait_mask = 63};
   fail += !agx3_falu_fma_mixed_mod_runtime(AGX_ISA_ENCODING_AGX3B,
      254, false, 7, 255, 255, 255, &mods, &rt, out, &length);
   fail += agx3_falu_fma_mixed_mod_runtime(AGX_ISA_ENCODING_AGX3B,
      0, false, 8, 0, 0, 0, &mods, &rt, out, &length);
   fail += agx3_falu_fma_mixed_mod_runtime(AGX_ISA_ENCODING_AGX3B,
      0, false, 7, 256, 0, 0, &mods, &rt, out, &length);
   for (unsigned s = 0; s < 3; ++s) {
      unsigned src[3] = {0};
      src[s] = 1;
      fail += agx3_falu_fma_mixed_mod_runtime(AGX_ISA_ENCODING_AGX3B,
         0, false, 7 ^ (1u << s), src[0], src[1], src[2],
         &mods, &rt, out, &length);
      rt.cache[s] = rt.discard[s] = true;
      fail += agx3_falu_fma_mixed_mod_runtime(AGX_ISA_ENCODING_AGX3B,
         0, false, 7, 0, 0, 0, &mods, &rt, out, &length);
      rt.cache[s] = rt.discard[s] = false;
   }
   rt.wait_mask = 64;
   fail += agx3_falu_fma_mixed_mod_runtime(AGX_ISA_ENCODING_AGX3B,
      0, false, 7, 0, 0, 0, &mods, &rt, out, &length);
   printf("native mixed FMA: 9 Apple vectors and invalid operands, %u failures\n",fail);
   return fail;
}

static unsigned
uniform_fma_negate_vectors(void)
{

   const struct {
      unsigned mask;
      uint16_t dest, src[3];
      struct agx3_falu_mods mods;
      struct agx3_falu_runtime runtime;
      uint8_t bytes[8];
   } vectors[] = {
      {1,2,{4,2,0},{.addend_neg=true},
       {.dest_cache=true,.cache={false,false,true},
        .discard={false,true,false}},
       {0x19,0x03,0x2e,0x05,0x11,0x80,0x02,0x02}},
      {4,0,{4,0,6},{.src0_neg=true},
       {.dest_cache=true,.cache={true,false,false},
        .discard={false,true,false}},
       {0x09,0x01,0x2e,0x85,0x21,0x06,0x02,0x08}},
   };
   unsigned fail = 0;
   uint8_t bytes[16], size = 0;
   for (unsigned i=0; i<sizeof(vectors)/sizeof(vectors[0]); ++i) {
      if (!agx3_falu_fma_uniform_negate(AGX_ISA_ENCODING_AGX3B,
            vectors[i].dest,vectors[i].mask,vectors[i].src,&vectors[i].mods,
            &vectors[i].runtime,bytes,&size) || size != 8 ||
          memcmp(bytes,vectors[i].bytes,8)) {
         fprintf(stderr,"native uniform FMA negate vector %u failed\n",i);
         ++fail;
      }
   }
   const uint16_t src[] = {510,254,508};
   for (unsigned neg=0; neg<8; ++neg) {
      struct agx3_falu_mods mods = {
         .src0_neg=neg&1,.src1_neg=neg&2,.addend_neg=neg&4,
      };
      for (unsigned wait=0; wait<64; ++wait) {
         const struct agx3_falu_runtime rt = {.wait_mask=wait};
         fail += !agx3_falu_fma_uniform_negate(AGX_ISA_ENCODING_AGX3B,
                    254,5,src,&mods,&rt,bytes,&size);
      }
      for (unsigned mod=0; mod<4; ++mod) {
         struct agx3_falu_mods invalid = mods;
         if (mod==0) invalid.src0_abs=true;
         if (mod==1) invalid.src1_abs=true;
         if (mod==2) invalid.addend_abs=true;
         if (mod==3) invalid.saturate=true;
         const struct agx3_falu_runtime rt = {0};
         fail += agx3_falu_fma_uniform_negate(AGX_ISA_ENCODING_AGX3B,
                    254,5,src,&invalid,&rt,bytes,&size);
      }
   }
   printf("native uniform FMA negate: two Apple vectors, modifier/wait domain: %u failures\n",fail);
   return fail;
}

static unsigned
uniform_fma_vectors(void)
{

   const struct {
      unsigned mask; uint16_t s0, add; uint8_t size, bytes[4][8];
   } vectors[] = {
      {1,14,10,6,{{0x09,0x01,0x0e,0x0f,0x20,0x0a},
                  {0x19,0x03,0x0e,0x0d,0x20,0x0c},
                  {0x29,0x05,0x0e,0x0b,0x20,0x0e},
                  {0x39,0x07,0x0e,0x09,0x20,0x10}}},
      {4,10,8,8,{{0x09,0x01,0x0e,0x0b,0x21,0x08,0x02,0x00},
                 {0x19,0x03,0x0e,0x0d,0x21,0x0a,0x02,0x00},
                 {0x29,0x05,0x0e,0x0f,0x21,0x0c,0x02,0x00},
                 {0x39,0x07,0x0e,0x11,0x21,0x0e,0x02,0x00}}},
      {5,22,8,8,{{0x09,0x01,0x0e,0x17,0x21,0x08,0x02,0x02},
                 {0x19,0x03,0x0e,0x15,0x21,0x0a,0x02,0x02},
                 {0x29,0x05,0x0e,0x13,0x21,0x0c,0x02,0x02},
                 {0x39,0x07,0x0e,0x11,0x21,0x0e,0x02,0x02}}},
   };
   unsigned fail = 0;
   for (unsigned i = 0; i < sizeof(vectors)/sizeof(vectors[0]); ++i) {
      for (unsigned c = 0; c < 4; ++c) {
         const uint16_t src[] = {
            vectors[i].s0 + ((vectors[i].mask & 1) ? -2 * (int)c : 2 * (int)c),
            2*c, vectors[i].add + 2*c,
         };
         const struct agx3_falu_runtime rt = {.discard={false,true,false}};
         uint8_t bytes[16], size = 0;
         if (!agx3_falu_fma_uniform(AGX_ISA_ENCODING_AGX3B, 2*c,
                 vectors[i].mask, src, &rt, bytes, &size) ||
             size != vectors[i].size || memcmp(bytes,vectors[i].bytes[c],size))
            fail++;
      }
   }
   uint16_t src[] = {510,254,508};
   struct agx3_falu_runtime rt = {.wait_mask=63};
   uint8_t bytes[16], size = 0;
   fail += !agx3_falu_fma_uniform(AGX_ISA_ENCODING_AGX3B,254,5,src,&rt,bytes,&size);
   for (unsigned mask = 0; mask < 8; ++mask)
      if (mask != 1 && mask != 4 && mask != 5)
         fail += agx3_falu_fma_uniform(AGX_ISA_ENCODING_AGX3B,0,mask,src,&rt,bytes,&size);
   fail += agx3_falu_fma_uniform(AGX_ISA_ENCODING_AGX3C,0,5,src,&rt,bytes,&size);
   for (unsigned s = 0; s < 3; ++s) {
      uint16_t old = src[s]; src[s] = old | 1;
      fail += agx3_falu_fma_uniform(AGX_ISA_ENCODING_AGX3B,0,5,src,&rt,bytes,&size);
      src[s] = s == 1 ? 256 : 512;
      fail += agx3_falu_fma_uniform(AGX_ISA_ENCODING_AGX3B,0,5,src,&rt,bytes,&size);
      src[s] = old;
      rt.cache[s] = rt.discard[s] = true;
      fail += agx3_falu_fma_uniform(AGX_ISA_ENCODING_AGX3B,0,5,src,&rt,bytes,&size);
      rt.cache[s] = rt.discard[s] = false;
   }
   rt.wait_mask = 64;
   fail += agx3_falu_fma_uniform(AGX_ISA_ENCODING_AGX3B,0,5,src,&rt,bytes,&size);
   printf("native uniform FMA: 12 Apple vectors and invalid operands: %u failures\n",fail);
   return fail;
}

static unsigned
narrow_runtime_vectors(void)
{
   const struct {
      uint16_t dest, src;
      bool discard, dest_cache, saturate;
      uint8_t wait, size, bytes[8];
   } vectors[] = {
      {0,0,true,false,false,32,6,{0x01,0x01,0x1c,0x81,0,0xc2}},
      {1,2,true,false,false,0,6,{0x01,0x03,0x1c,0x81,0x04,0x02}},
      {3,6,true,false,false,0,6,{0x11,0x07,0x1c,0x81,0x04,0x02}},
      {0,2,false,true,false,32,6,{0x01,0x03,0x34,0x81,0,0xc2}},
      {2,2,true,false,true,32,8,{0x11,0x03,0x1c,0x81,0x01,0x02,0,0x82}},
   };
   unsigned fail = 0;
   for (unsigned i=0; i<sizeof(vectors)/sizeof(vectors[0]); ++i) {
      struct agx3_falu_runtime rt = {
         .discard={vectors[i].discard}, .dest_cache=vectors[i].dest_cache,
         .wait_mask=vectors[i].wait,
      };
      uint8_t bytes[16], length;
      if (!agx3_falu_narrow_f32(AGX_ISA_ENCODING_AGX3B, vectors[i].dest,
             vectors[i].src, vectors[i].saturate, &rt, bytes, &length) ||
          length!=vectors[i].size || memcmp(bytes,vectors[i].bytes,length)) {
         fprintf(stderr,"native narrowing vector %u failed\n",i);
         fail++;
      }
   }
   struct agx3_falu_runtime rt = {.cache={true},.wait_mask=63};
   uint8_t bytes[16], length=0;
   fail += !agx3_falu_narrow_f32(AGX_ISA_ENCODING_AGX3B,255,254,false,&rt,bytes,&length);
   fail += length>10;
   fail += agx3_falu_narrow_f32(AGX_ISA_ENCODING_AGX3B,256,254,false,&rt,bytes,&length);
   fail += agx3_falu_narrow_f32(AGX_ISA_ENCODING_AGX3B,0,255,false,&rt,bytes,&length);
   rt.discard[0]=true;
   fail += agx3_falu_narrow_f32(AGX_ISA_ENCODING_AGX3B,0,2,false,&rt,bytes,&length);
   printf("native f32-to-f16: odd/even halves, live/dead input, saturate, waits: %u failures\n",fail);
   return fail;
}

static unsigned
add_runtime_vectors(void)
{
   struct add_vector {
      uint16_t src0, src1;
      struct agx3_falu_mods mods;
      struct agx3_falu_runtime runtime;
      unsigned length;
      uint8_t bytes[10];
   };
   const struct add_vector vectors[] = {
      /* Original x-y ordering commutes to Apple's shorter negate-first form. */
      {0, 2, {.src1_neg=true}, {.discard={true,true}, .wait_mask=32}, 6,
       {0x09,0x01,0x1c,0x03,0,0xc8}},
      {2, 0, {.src0_neg=true}, {.discard={true,true}, .wait_mask=32}, 6,
       {0x09,0x01,0x1c,0x03,0,0xc8}},
      {4, 2, {.src0_neg=true}, {.wait_mask=32}, 6,
       {0x09,0x03,0x04,0x05,0,0xc8}},
      {2, 0, {.src0_abs=true, .src0_neg=true, .src1_abs=true},
       {.discard={true,true}, .wait_mask=32}, 10,
       {0x09,0x01,0x1c,0x03,0x02,0x08,0,0x80,0x03,0}},
      {2, 0, {.src0_neg=true, .saturate=true},
       {.discard={true,true}, .wait_mask=32}, 8,
       {0x09,0x01,0x1c,0x03,0x01,0x08,0,0x82}},
      {2, 0, {.src0_neg=true, .src1_neg=true},
       {.discard={true,true}, .wait_mask=32}, 8,
       {0x09,0x01,0x1c,0x03,0x01,0x08,0x02,0x80}},
   };
   unsigned fail = 0;
   for (unsigned i = 0; i < sizeof(vectors)/sizeof(vectors[0]); ++i) {
      const struct add_vector *v = &vectors[i];
      uint8_t out[16], length;
      if (!agx3_falu_add_mod_runtime(AGX_ISA_ENCODING_AGX3B, 0,
             v->src0, v->src1, &v->mods, &v->runtime, out, &length) ||
          length != v->length || memcmp(out, v->bytes, length)) {
         fprintf(stderr, "native modified ADD vector %u failed\n", i);
         fail++;
      }
   }
   struct agx3_falu_mods m = {.src0_neg=true, .src1_abs=true};
   struct agx3_falu_runtime rt = {.wait_mask=63};
   uint8_t out[16], length = 0;
   /* Long forms must carry multiple dependency slots without a separate
    * arithmetic operation or discarding either live input. */
   fail += !agx3_falu_add_mod_runtime(AGX_ISA_ENCODING_AGX3B, 254, 252, 250,
                                       &m, &rt, out, &length);
   fail += length > 10;
   memset(out, 0xa5, sizeof(out)); length = 0xa5;
   rt.cache[0] = rt.discard[0] = true;
   fail += agx3_falu_add_mod_runtime(AGX_ISA_ENCODING_AGX3B, 0, 2, 4,
                                      &m, &rt, out, &length);
   rt.cache[0] = false; rt.wait_mask = 64;
   fail += agx3_falu_add_mod_runtime(AGX_ISA_ENCODING_AGX3B, 0, 2, 4,
                                      &m, &rt, out, &length);
   rt.wait_mask = 0;
   fail += agx3_falu_add_mod_runtime(AGX_ISA_ENCODING_AGX3B, 0, 255, 4,
                                      &m, &rt, out, &length);
   fail += agx3_falu_add_mod_runtime(AGX_ISA_ENCODING_AGX3C, 0, 2, 4,
                                      &m, &rt, out, &length);
   fail += length != 0xa5;
   for (unsigned i=0; i<sizeof(out); ++i) fail += out[i] != 0xa5;
   printf("native modified ADD: five Apple vectors plus commuted subtraction, live operands, waits and bounds: %u failures\n",fail);
   return fail;
}

static unsigned
runtime_vectors(void)
{
   unsigned fail = 0;
   {
      const uint8_t expected[] = {0x09, 0x01, 0x1e, 0x03, 0x91, 0x04, 0x02, 0xc0};
      const struct agx3_falu_mods m = {.src0_abs = 0, .src0_neg = 0, .src1_abs = 0, .src1_neg = 0, .addend_abs = 0, .addend_neg = 1, .saturate = 0};
      const struct agx3_falu_runtime rt = {.cache = {0, 0, 0}, .discard = {1, 1, 1}, .dest_cache = 0, .wait_mask = 32};
      uint8_t out[16], len;
      if (!agx3_falu_fma_mod_runtime(AGX_ISA_ENCODING_AGX3B, 0, false, 2, 0, 4,
                                      &m, &rt, out, &len) ||
          len != sizeof(expected) || memcmp(out, expected, len)) {
         fprintf(stderr, "native modified FMA vector 0 failed\n");
         fail++;
      }
   }
   {
      const uint8_t expected[] = {0x09, 0x03, 0x06, 0x05, 0x11, 0x06, 0x02, 0xc0};
      const struct agx3_falu_mods m = {.src0_abs = 0, .src0_neg = 0, .src1_abs = 0, .src1_neg = 0, .addend_abs = 0, .addend_neg = 1, .saturate = 0};
      const struct agx3_falu_runtime rt = {.cache = {0, 0, 0}, .discard = {0, 0, 0}, .dest_cache = 0, .wait_mask = 32};
      uint8_t out[16], len;
      if (!agx3_falu_fma_mod_runtime(AGX_ISA_ENCODING_AGX3B, 0, false, 4, 2, 6,
                                      &m, &rt, out, &len) ||
          len != sizeof(expected) || memcmp(out, expected, len)) {
         fprintf(stderr, "native modified FMA vector 1 failed\n");
         fail++;
      }
   }
   {
      const uint8_t expected[] = {0x09, 0x01, 0x1e, 0x03, 0x83, 0x04, 0x02, 0x00, 0x00, 0x80, 0x01, 0x00};
      const struct agx3_falu_mods m = {.src0_abs = 0, .src0_neg = 0, .src1_abs = 1, .src1_neg = 0, .addend_abs = 0, .addend_neg = 0, .saturate = 0};
      const struct agx3_falu_runtime rt = {.cache = {0, 0, 0}, .discard = {1, 1, 1}, .dest_cache = 0, .wait_mask = 32};
      uint8_t out[16], len;
      if (!agx3_falu_fma_mod_runtime(AGX_ISA_ENCODING_AGX3B, 0, false, 2, 0, 4,
                                      &m, &rt, out, &len) ||
          len != sizeof(expected) || memcmp(out, expected, len)) {
         fprintf(stderr, "native modified FMA vector 2 failed\n");
         fail++;
      }
   }
   {
      const uint8_t expected[] = {0x09, 0x03, 0x1e, 0x01, 0x81, 0x04, 0x02, 0xc8};
      const struct agx3_falu_mods m = {.src0_abs = 0, .src0_neg = 1, .src1_abs = 0, .src1_neg = 0, .addend_abs = 0, .addend_neg = 0, .saturate = 0};
      const struct agx3_falu_runtime rt = {.cache = {0, 0, 0}, .discard = {1, 1, 1}, .dest_cache = 0, .wait_mask = 32};
      uint8_t out[16], len;
      if (!agx3_falu_fma_mod_runtime(AGX_ISA_ENCODING_AGX3B, 0, false, 0, 2, 4,
                                      &m, &rt, out, &len) ||
          len != sizeof(expected) || memcmp(out, expected, len)) {
         fprintf(stderr, "native modified FMA vector 3 failed\n");
         fail++;
      }
   }
   {
      const uint8_t expected[] = {0x09, 0x01, 0x1e, 0x03, 0x89, 0x04, 0x02, 0xc0};
      const struct agx3_falu_mods m = {.src0_abs = 0, .src0_neg = 0, .src1_abs = 0, .src1_neg = 0, .addend_abs = 1, .addend_neg = 0, .saturate = 0};
      const struct agx3_falu_runtime rt = {.cache = {0, 0, 0}, .discard = {1, 1, 1}, .dest_cache = 0, .wait_mask = 32};
      uint8_t out[16], len;
      if (!agx3_falu_fma_mod_runtime(AGX_ISA_ENCODING_AGX3B, 0, false, 2, 0, 4,
                                      &m, &rt, out, &len) ||
          len != sizeof(expected) || memcmp(out, expected, len)) {
         fprintf(stderr, "native modified FMA vector 4 failed\n");
         fail++;
      }
   }
   {
      const uint8_t expected[] = {0x09, 0x01, 0x1e, 0x03, 0x82, 0x04, 0x02, 0x00, 0x00, 0x82};
      const struct agx3_falu_mods m = {.src0_abs = 0, .src0_neg = 0, .src1_abs = 0, .src1_neg = 0, .addend_abs = 0, .addend_neg = 0, .saturate = 1};
      const struct agx3_falu_runtime rt = {.cache = {0, 0, 0}, .discard = {1, 1, 1}, .dest_cache = 0, .wait_mask = 32};
      uint8_t out[16], len;
      if (!agx3_falu_fma_mod_runtime(AGX_ISA_ENCODING_AGX3B, 0, false, 2, 0, 4,
                                      &m, &rt, out, &len) ||
          len != sizeof(expected) || memcmp(out, expected, len)) {
         fprintf(stderr, "native modified FMA vector 5 failed\n");
         fail++;
      }
   }
   /* The witnessed abs source forces the long form. Independently inspect
    * all six physical wait bits and source hints, not the encoder's recipe. */
   for (unsigned waits = 0; waits < 64; ++waits) {
      const struct agx3_falu_mods m = {.src1_abs = true};
      struct agx3_falu_runtime rt = {.wait_mask = waits};
      uint8_t out[16], len;
      const unsigned wait_bits[] = {61, 62, 63, 77, 78, 79};
      if (!agx3_falu_fma_mod_runtime(AGX_ISA_ENCODING_AGX3B, 0, false, 2, 0, 4,
                                    &m, &rt, out, &len) || len != 12) {
         fail++;
         continue;
      }
      for (unsigned i = 0; i < 6; ++i)
         fail += ((out[wait_bits[i] / 8] >> (wait_bits[i] % 8)) & 1) !=
                 ((waits >> i) & 1);
      const unsigned cache_bits[] = {31, 15, 47};
      const unsigned discard_bits[] = {20, 19, 39};
      for (unsigned hints = 0; hints < 27; ++hints) {
         unsigned v = hints;
         for (unsigned i = 0; i < 3; ++i, v /= 3) {
            rt.cache[i] = (v % 3) == 1;
            rt.discard[i] = (v % 3) == 2;
         }
         if (!agx3_falu_fma_mod_runtime(AGX_ISA_ENCODING_AGX3B, 0, false, 2, 0, 4,
                                       &m, &rt, out, &len) || len != 12) {
            fail++;
            continue;
         }
         for (unsigned i = 0; i < 3; ++i) {
            fail += ((out[cache_bits[i] / 8] >> (cache_bits[i] % 8)) & 1)
                    != rt.cache[i];
            fail += ((out[discard_bits[i] / 8] >> (discard_bits[i] % 8)) & 1)
                    != rt.discard[i];
         }
      }
   }
   struct agx3_falu_mods m = {.addend_neg = true};
   struct agx3_falu_runtime rt = {.wait_mask = 64};
   uint8_t out[16], len = 0xa5;
   memset(out, 0xa5, sizeof(out));
   fail += agx3_falu_fma_mod_runtime(AGX_ISA_ENCODING_AGX3B, 0, false, 2, 4, 6,
                                    &m, &rt, out, &len);
   rt.wait_mask = 0;
   rt.cache[0] = rt.discard[0] = true;
   fail += agx3_falu_fma_mod_runtime(AGX_ISA_ENCODING_AGX3B, 0, false, 2, 4, 6,
                                    &m, &rt, out, &len);
   rt.cache[0] = false;
   fail += agx3_falu_fma_mod_runtime(AGX_ISA_ENCODING_AGX3B, 0, false, 2, 4, 255,
                                    &m, &rt, out, &len);
   fail += agx3_falu_fma_mod_runtime(AGX_ISA_ENCODING_AGX3C, 0, false, 2, 4, 6,
                                    &m, &rt, out, &len);
   fail += len != 0xa5;
   for (unsigned i = 0; i < sizeof(out); ++i)
      fail += out[i] != 0xa5;
   {
      const struct agx3_falu_mods narrow_mod = {.src0_neg=true};
      const struct agx3_falu_runtime narrow_rt = {.discard={1,1,1}, .wait_mask=32};
      const uint8_t expected[] = {0x11,0x03,0x1e,0x01,0x81,0x04,0x02,0xc8};
      fail += !agx3_falu_fma_mod_runtime(AGX_ISA_ENCODING_AGX3B,2,true,0,2,4,
                    &narrow_mod,&narrow_rt,out,&len) || len != sizeof(expected) ||
                    memcmp(out,expected,sizeof(expected));
   }
   printf("native modified FMA: six Apple byte vectors, 64 wait masks, "
          "27 independent lifetime combinations: %u failures\n", fail);
   return fail;
}

static unsigned
mul_runtime_vectors(void)
{
   const struct {
      uint16_t dst, s0, s1;
      struct agx3_falu_mods m;
      struct agx3_falu_runtime rt;
      unsigned size;
      uint8_t bytes[16];
   } vectors[] = {
      {0,6,4,{.src0_abs=0,.src0_neg=0,.src1_abs=0,.src1_neg=0,.saturate=1}, {.cache={0,0},.discard={1,1},.dest_cache=0,.wait_mask=32},8,{0x09,0x05,0x1d,0x07,0x01,0x00,0x00,0x82}},
      {0,4,2,{.src0_abs=0,.src0_neg=0,.src1_abs=0,.src1_neg=0,.saturate=1}, {.cache={0,0},.discard={0,0},.dest_cache=0,.wait_mask=32},8,{0x09,0x03,0x05,0x05,0x01,0x00,0x00,0x82}},
      {0,6,4,{.src0_abs=0,.src0_neg=0,.src1_abs=1,.src1_neg=0,.saturate=0}, {.cache={0,0},.discard={1,1},.dest_cache=0,.wait_mask=32},12,{0x09,0x05,0x1d,0x07,0x02,0x00,0x00,0x80,0x01,0x00,0x00,0x00}},
      {0,4,2,{.src0_abs=0,.src0_neg=0,.src1_abs=1,.src1_neg=0,.saturate=0}, {.cache={0,0},.discard={0,0},.dest_cache=0,.wait_mask=32},12,{0x09,0x03,0x05,0x05,0x02,0x00,0x00,0x80,0x01,0x00,0x00,0x00}},
      {0,4,6,{.src0_abs=0,.src0_neg=1,.src1_abs=0,.src1_neg=0,.saturate=0}, {.cache={0,0},.discard={1,1},.dest_cache=0,.wait_mask=32},6,{0x09,0x07,0x1d,0x05,0x00,0xc8}},
      {0,2,4,{.src0_abs=0,.src0_neg=1,.src1_abs=0,.src1_neg=0,.saturate=0}, {.cache={0,0},.discard={0,0},.dest_cache=0,.wait_mask=32},6,{0x09,0x05,0x05,0x03,0x00,0xc8}},
      {0,4,6,{.src0_abs=1,.src0_neg=1,.src1_abs=1,.src1_neg=0,.saturate=0}, {.cache={0,0},.discard={1,1},.dest_cache=0,.wait_mask=32},12,{0x09,0x07,0x1d,0x05,0x02,0x08,0x00,0x80,0x03,0x00,0x00,0x00}},
      {0,2,4,{.src0_abs=1,.src0_neg=1,.src1_abs=1,.src1_neg=0,.saturate=0}, {.cache={0,0},.discard={0,0},.dest_cache=0,.wait_mask=32},12,{0x09,0x05,0x05,0x03,0x02,0x08,0x00,0x80,0x03,0x00,0x00,0x00}},
   };
   unsigned fail = 0;
   for (unsigned i = 0; i < sizeof(vectors)/sizeof(vectors[0]); ++i) {
      uint8_t out[16], size;
      if (!agx3_falu_mul_mod_runtime(AGX_ISA_ENCODING_AGX3B,
              vectors[i].dst, vectors[i].s0, vectors[i].s1,
              &vectors[i].m, &vectors[i].rt, out, &size) ||
          size != vectors[i].size || memcmp(out, vectors[i].bytes, size)) {
         fprintf(stderr, "native modified FMUL vector %u failed size=%u: ", i, size);
         for (unsigned j=0;j<size;++j) fprintf(stderr,"%02x",out[j]);
         fprintf(stderr,"\n");
         fail++;
      }
   }
   printf("native modified FMUL: eight Apple live/dead byte vectors: %u failures\n", fail);
   return fail;
}

static unsigned
fma_addend_vectors(void)
{
   const struct {
      uint16_t dst,s0,s1; uint8_t imm;
      struct agx3_falu_runtime rt;
      unsigned size; uint8_t bytes[16];
   } vectors[] = {
      {0,6,4,48,{.cache={0,0},.discard={1,1},.dest_cache=0,.wait_mask=32},8,{0x09,0x05,0x1e,0x07,0x21,0xb0,0x02,0xc0}},
      {0,4,2,48,{.cache={0,0},.discard={0,0},.dest_cache=0,.wait_mask=32},8,{0x09,0x03,0x06,0x05,0x21,0xb0,0x02,0xc0}},
      {0,6,4,176,{.cache={0,0},.discard={1,1},.dest_cache=0,.wait_mask=32},8,{0x09,0x05,0x1e,0x07,0xa1,0xb0,0x02,0xc0}},
      {0,4,2,176,{.cache={0,0},.discard={0,0},.dest_cache=0,.wait_mask=32},8,{0x09,0x03,0x06,0x05,0xa1,0xb0,0x02,0xc0}},
      {0,6,4,32,{.cache={0,0},.discard={1,1},.dest_cache=0,.wait_mask=32},8,{0x09,0x05,0x1e,0x07,0x21,0xa0,0x02,0xc0}},
      {0,4,2,32,{.cache={0,0},.discard={0,0},.dest_cache=0,.wait_mask=32},8,{0x09,0x03,0x06,0x05,0x21,0xa0,0x02,0xc0}},
   };
   unsigned fail=0;
   for(unsigned i=0;i<sizeof(vectors)/sizeof(vectors[0]);++i){
      uint8_t out[16],size=0;
      if(!agx3_falu_fma_addend_imm(AGX_ISA_ENCODING_AGX3B,vectors[i].dst,
             vectors[i].s0,vectors[i].s1,vectors[i].imm,&vectors[i].rt,out,&size)||
         size!=vectors[i].size||memcmp(out,vectors[i].bytes,size)){
         fprintf(stderr,"FMA minifloat addend vector %u failed\n",i);fail++;
      }
   }
   {
      const struct {
         uint16_t dst, src0, src1;
         struct agx3_falu_runtime rt;
         uint8_t bytes[8];
      } negative_multiplier[] = {
         {0, 0, 2, {.discard={1,1}},
          {0x09,0x03,0x1e,0x01,0x21,0x98,0x02,0xc8}},
         {0, 2, 4, {.cache={0,0}, .discard={0,0}},
          {0x09,0x05,0x06,0x03,0x21,0x98,0x02,0xc8}},
      };
      for (unsigned i = 0;
           i < sizeof(negative_multiplier) / sizeof(negative_multiplier[0]);
           ++i) {
         uint8_t out[16], size = 0;
         if (!agx3_falu_fma_addend_imm_mod(
                AGX_ISA_ENCODING_AGX3B, negative_multiplier[i].dst,
                negative_multiplier[i].src0, negative_multiplier[i].src1,
                24, true, false, &negative_multiplier[i].rt, out, &size) ||
             size != sizeof(negative_multiplier[i].bytes) ||
             memcmp(out, negative_multiplier[i].bytes, size)) {
            fprintf(stderr,
                    "FMA minifloat addend negated-multiplier vector %u failed\n",
                    i);
            for (unsigned j = 0; j < size; ++j)
               fprintf(stderr, "%02x", out[j]);
            fprintf(stderr, " size=%u\n", size);
            fail++;
         }
      }
   }
   printf("native FMA minifloat addend: eight Apple live/dead/modifier vectors: %u failures\n",fail);
   return fail;
}

static unsigned
fma_two_vectors(void)
{
   const struct {
      uint16_t dst,src; uint8_t mul,add;
      struct agx3_falu_runtime rt;
      unsigned size; uint8_t bytes[16];
   } vectors[] = {
      {0,0,64,176,{.cache={0},.discard={1},.dest_cache=0,.wait_mask=32},8,{0x09,0x01,0x0e,0xc1,0xa1,0xb0,0x02,0xc2}},
      {0,2,64,176,{.cache={0},.discard={0},.dest_cache=0,.wait_mask=32},8,{0x09,0x03,0x06,0xc1,0xa1,0xb0,0x02,0xc2}},
      {0,0,32,16,{.cache={0},.discard={1},.dest_cache=0,.wait_mask=32},8,{0x09,0x01,0x0e,0xa1,0x21,0x90,0x02,0xc2}},
      {0,2,32,16,{.cache={0},.discard={0},.dest_cache=0,.wait_mask=32},8,{0x09,0x03,0x06,0xa1,0x21,0x90,0x02,0xc2}},
      {0,0,56,160,{.cache={0},.discard={1},.dest_cache=0,.wait_mask=32},8,{0x09,0x01,0x0e,0xb9,0xa1,0xa0,0x02,0xc2}},
      {0,2,56,160,{.cache={0},.discard={0},.dest_cache=0,.wait_mask=32},8,{0x09,0x03,0x06,0xb9,0xa1,0xa0,0x02,0xc2}},
   };
   unsigned fail=0;
   for(unsigned i=0;i<sizeof(vectors)/sizeof(vectors[0]);++i){
      uint8_t out[16],size=0;
      if(!agx3_falu_fma_two_imm(AGX_ISA_ENCODING_AGX3B,vectors[i].dst,false,
             vectors[i].src,vectors[i].mul,vectors[i].add,&vectors[i].rt,out,&size)||
         size!=vectors[i].size||memcmp(out,vectors[i].bytes,size)){
         fprintf(stderr,"FMA two-immediate vector %u failed\n",i);fail++;
      }
   }
   {
      uint8_t out[16], size = 0;
      const uint8_t expected[] = {0x11,0x03,0x0e,0x91,0x21,0xa0,0x02,0xc2};
      const struct agx3_falu_runtime rt = {.discard={1}, .wait_mask=32};
      if (!agx3_falu_fma_two_imm(AGX_ISA_ENCODING_AGX3B,2,true,2,16,32,&rt,out,&size) ||
          size != sizeof(expected) || memcmp(out,expected,size))
         fail++;
   }
   printf("native FMA two minifloats: six Apple live/dead vectors: %u failures\n",fail);
   return fail;
}

int
main(void)
{
   unsigned pass = 0, fail = uniform_fma_vectors() + uniform_fma_negate_vectors() + uniform_binary_vectors() + mixed_fma_vectors() + runtime_vectors() + add_runtime_vectors() + narrow_runtime_vectors() + mul_runtime_vectors() + fma_addend_vectors() + fma_two_vectors();
   for (size_t i = 0; i < mod_vec_count; ++i) {
      const struct mod_vec *v = &mod_vecs[i];
      const enum agx_isa_encoding e =
         v->enc == 0 ? AGX_ISA_ENCODING_AGX3B : AGX_ISA_ENCODING_AGX3C;
      const struct agx3_falu_mods m = {
         .src0_abs = v->s0a, .src0_neg = v->s0n, .src1_abs = v->s1a,
         .src1_neg = v->s1n, .addend_abs = v->ada, .addend_neg = v->adn,
         .saturate = v->sat,
      };
      uint8_t out[AGX3_FALU_MOD_MAX_LENGTH];
      uint8_t len = 0;
      bool ok;
      if (!strcmp(v->mnem, "f32add"))
         ok = agx3_falu_add_mod(e, v->dest, v->s0, v->s1, &m, out, &len);
      else if (!strcmp(v->mnem, "f32mul"))
         ok = agx3_falu_mul_mod(e, v->dest, v->s0, v->s1, &m, out, &len);
      else
         ok = agx3_falu_fma_mod(e, v->dest, v->s0, v->s1, v->ad, &m, out, &len);

      if (!ok || len != v->nbytes || memcmp(out, v->bytes, v->nbytes) != 0) {
         if (fail < 10)
            fprintf(stderr, "FAIL %s enc=%d\n", v->mnem, v->enc);
         fail++;
      } else {
         pass++;
      }
   }
   if (fail) {
      fprintf(stderr, "agx3_falu_mod_core: %u of %zu failed\n", fail,
              mod_vec_count);
      return 1;
   }
   printf("agx3_falu_mod_core: all %u modifier forms reproduced\n", pass);
   return 0;
}
