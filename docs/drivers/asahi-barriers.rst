Borealis render-pass barriers
=============================

``hk_barrier_classify()`` decides whether an in-pass dependency needs a
physical render boundary. Its result is a set of reasons, rather than a
test for a nonzero destination access mask. It has no GPU or DRM dependency;
``hk_barrier`` in the Asahi Meson test suite exercises the production function.

The API render-pass instance remains open across physical splits. Dependencies
still go through ``hk_sync_pass_barrier()`` and expire at its end. A continued
render waits for the preceding physical render. A software vertex stream
excludes in-pass fragment-only destinations; otherwise introducing geometry or
tessellation work after a local barrier could create a forward dependency on
its consuming render. Global requirements remain in force for internal
producers such as attachment decompression.

Dependencies kept in the tile pass
----------------------------------

* Framebuffer-space attachment operations with ``VK_DEPENDENCY_BY_REGION_BIT``,
  no tile-buffer spill and ordinary optimal attachments without aliasing. This
  includes color blending and depth/stencil read/write dependencies. The
  presence of a query or software vertex stream does not alone require ending
  the render.
* Framebuffer-local execution dependencies and vertex-to-fragment execution
  dependencies already satisfied by the tiling/fragment boundary.
* Memory scopes with no writes to publish in the pass. The classifier retains
  execution ordering requirements even when no store is needed. It knows
  whether any recorded graphics shader/XFB may write external memory, across
  all physical renders of the API instance. With read-only shaders, an image
  outside the attachments cannot have been written by the pass. Aliasable,
  sparse, external-memory, linear or modifier-based attachments disable that
  image-disjoint proof. Comparisons cover whole images, not subresource ranges.
  Loaded, unresolved depth/stencil aspects in read-only layouts are omitted from the
  possibly-written attachment set, so their texture/input reads need no store
  when shaders cannot write them either. Clears, resolves and non-load operations remain
  conservative; a writable stencil aspect keeps its whole image in the set.

A read-only or empty source access mask alone is not an unwritten-memory
proof: a visibility operation can expose writes made available by an earlier
dependency. For example, a local color-write to color-read dependency can
stay in tile memory, but a later color-read to input-read dependency must
still store it. Resource scope and cumulative write knowledge determine the
fallback, including visibility-only dependencies.

Conservative boundaries
-----------------------

``BY_REGION`` alone is insufficient for keeping a barrier. Input attachments
are lowered by ``nir_lower_input_attachments`` to texture fetches in this
driver. Input-attachment and sampled feedback reads of attachment writes still
need a tile store; their reason is ``attachment_memory``. Removing these
boundaries requires a compiler/descriptor path that reads the tile buffer.

Storage writes, framebuffer-global dependencies, vertex destinations requiring
an execution fence, compute/transfer scopes, real layout changes, ownership
transfers, and unknown scopes/extensions also retain boundaries. All layout
changes are conservative, including changes which a future layout-specific
analysis might prove to require no data movement. Equal layouts, including
equal ``UNDEFINED`` layouts with synchronization2, are not transitions.

Resuming an instance conservatively assumes its earlier part may have written
storage memory. A shader-write bit records possible writes, not exact resource
addresses, so barriers on unrelated storage resources may still split.

Performance counters
--------------------

With ``ASAHI_MESA_DEBUG=perf``, each submitted command buffer reports::

   Barrier renders: kept=N splits=M
   Barrier split reason framebuffer_global=N
   Barrier split reason attachment_memory=N
   Barrier split reason shader_memory=N
   Barrier split reason side_stream=N
   Barrier split reason vertex_order=N
   Barrier split reason compute_transfer=N
   Barrier split reason layout=N
   Barrier split reason ownership=N
   Barrier split reason unknown=N

``splits`` counts physical renders finalized at barriers. Empty renders removed
or replaced by compute clears do not contribute. Reasons overlap: a single
boundary can increment several reasons, so their sum can exceed ``splits``.
``kept`` counts barriers encountered with an open render which stayed open;
it includes execution barriers that the previous implementation also kept.
Counters reset on command-buffer reset and are reported again on each submit
of a reused command buffer. They measure boundaries, not attachment bytes or
hardware partial renders. Normalize them and submitted VDM/CDM counts by
presented frames when comparing workloads. Capture diagnostic counters in
separate runs from uninstrumented frame-time measurements.

Specification and implementation references
-------------------------------------------

The Vulkan specification's `Framebuffer Region Dependencies
<https://docs.vulkan.org/spec/latest/chapters/synchronization.html#synchronization-framebuffer-regions>`_
and `Pipeline Barriers
<https://docs.vulkan.org/spec/latest/chapters/synchronization.html#synchronization-pipeline-barriers>`_
define the relevant scopes. In particular, the in-pass rules
``07890``, ``07891``, and ``07892`` restrict framebuffer destinations, require
``BY_REGION`` when a source is framebuffer-space, and restrict stages to
graphics. Synchronization2 permits equal image layouts without a transition.

The implementation was written after studying Mesa's Panvk CSF
``collect_cs_deps()``, Turnip ``tu_barrier()`` and its GMEM decision, and v3dv
``is_tlb_color_input_barrier()``. These illustrate why tile-local dependencies
need both appropriate synchronization scopes and a compatible attachment read
path. Their hardware-specific commands are not reused here.
