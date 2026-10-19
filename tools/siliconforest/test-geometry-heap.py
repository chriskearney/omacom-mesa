#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Execute the real heap callback with BO allocation/map failures and drirc sizes."""
from pathlib import Path
import subprocess
import os
import shlex
import tempfile
repo = Path(__file__).resolve().parents[2]
src = (repo / "src/asahi/vulkan/hk_cmd_draw.c").read_text()
start = src.index("static void\nhk_init_heap(")
callback = src[start:src.index("\n}", start) + 2]
geometry = (repo / "src/poly/geometry.h").read_text()
layout_start = geometry.index("struct poly_heap {")
heap_layout = geometry[layout_start:geometry.index("} PACKED;", layout_start) + len("} PACKED;")]
prefix = r'''
#include <assert.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <limits.h>
#define HAVE_FUNC_ATTRIBUTE_PACKED 1
#include "util/macros.h"
#define DEVICE(T) uint64_t
struct hk_instance { struct { struct { int geometry_heap_size_mb; } misc; } drirc; };
struct hk_physical_device { struct hk_instance *instance; };
struct va { uint64_t addr; };
struct agx_bo { struct va *va; };
HEAP_LAYOUT
_Static_assert(sizeof(struct poly_heap) == 16, "GPU heap layout");
struct hk_device { int dev; struct { void *heap_bo; } rodata; struct agx_bo *heap; };
static struct hk_instance instance;
static struct hk_physical_device physical = { &instance };
static struct va address = { 0x123456780000ULL };
static struct agx_bo allocation = { &address };
static struct poly_heap state;
static size_t requested;
static int fail_bo, fail_map, releases;
static struct hk_physical_device *hk_device_physical(struct hk_device *d) { return &physical; }
static struct hk_instance *hk_physical_device_instance(struct hk_physical_device *p) { return p->instance; }
#define perf_debug_dev(...) ((void)0)
static struct agx_bo *agx_bo_create(void *d, size_t size, int a, int b, const char *n)
{ requested = size; return fail_bo ? NULL : &allocation; }
static struct poly_heap *agx_bo_map(void *b) { return fail_map ? NULL : &state; }
static void agx_bo_unreference(void *d, struct agx_bo *b) { assert(b == &allocation); releases++; }
'''
prefix = prefix.replace('HEAP_LAYOUT', heap_layout)
suffix = r'''
int main(void)
{
   const int sizes[] = { INT_MIN, -1, 0, 15, 16, 128, 256, 1024, 1025, 4096, 5000, INT_MAX };
   const unsigned expected_mb[] = { 16, 16, 16, 16, 16, 128, 256, 1024, 1024, 1024, 1024, 1024 };
   for (unsigned i = 0; i < sizeof(sizes) / sizeof(sizes[0]); i++) {
      struct hk_device dev = {0}; instance.drirc.misc.geometry_heap_size_mb = sizes[i];
      hk_init_heap(&dev);
      assert(dev.heap == &allocation && state.base == address.addr);
      assert(requested == (size_t)expected_mb[i] * 1024 * 1024 && state.size == requested);
      assert(state.bottom == 0);
   }
   for (int scenario = 0; scenario < 2; scenario++) {
      struct hk_device dev = {0}; fail_bo = scenario == 0; fail_map = scenario == 1;
      state.base = 99; state.size = 101; releases = 0;
      hk_init_heap(&dev);
      assert(!dev.heap && state.base == 99 && state.size == 101);
      assert(releases == scenario);
   }
   puts("PASS real hk_init_heap: u32 layout, 12 XML sizes clamped, BO/map failure and cleanup");
}
'''
with tempfile.TemporaryDirectory() as t:
    binary = Path(t) / "heap"
    subprocess.run(shlex.split(os.environ.get("CC", "cc")) + ["-I", str(repo / "src"), "-std=c11", "-Wall", "-Wextra", "-Werror", "-Wno-unused-parameter", "-fsanitize=address,undefined", "-x", "c", "-", "-o", str(binary)], input=prefix+callback+suffix, text=True, check=True)
    subprocess.run([str(binary)], check=True)
