# Robust indexed indirect draws on native AGX3

The repair kernel preserves the software zero-index path required by
`robustBufferAccess2`. The public Index List representation has a buffer-size
word, but it does not establish the zero-index behavior needed to replace this
path. See the [Vulkan drawing specification](https://docs.vulkan.org/spec/latest/chapters/drawing.html)
and `libagx/draws.cl`.

An eligible `CmdDrawIndexedIndirect` call records one 32-lane workgroup per draw
in one robustness dispatch, followed by one strong CDM publication barrier. The
GPU still reads each indirect record at execution time. The VDM stream retains
API order, per-record draw parameters and state selectors; workgroup execution
order does not affect drawing order. No repair is pending when the API call
returns. A count draw retains its separate count-predication dispatch, followed
by the same batch repair.

The first draw prepares state through the general path. Later records join only
if the existing indirect-record eligibility check rules out other lowering and
stream splits. Conditional rendering, geometry/tessellation, adjacency,
multiview and pipeline statistics retain the general path. If state is still
ineligible after the first draw, its repair is published before the remaining
draws use the general path. Single-record API calls retain one repair dispatch
each. This change does not coalesce separate API calls across barriers.

## Shadow allocation failure

A partial index-buffer overread needs a shadow allocation. If its size cannot
be represented or the bounded CAS reservation cannot fit, the repair kernel
sets the device-owned printf/abort buffer's sticky abort word. Both single and
batch entrypoints receive this address at dispatch time. No device address is
embedded in cached machine code. Empty-instance draws require no shadow.

The kernel initializes a safe empty VDM continuation so execution can finish
without an invalid index-buffer access. This continuation represents an
execution failure: the existing `vk_check_printf_status()` contract reports
`VK_ERROR_DEVICE_LOST` through host completion waits and status polls, including
already-available query results. Later heap reclamation and successful draws
do not clear the failure. Initial asynchronous submission success is not a
successful execution result. No deliberate GPU fault or new kernel ABI is used.

The CPU regression executes the extracted production repair body, single and
batch wrappers, and the actual abort/status consumers. It covers all index
widths, legacy/native selectors, points and line/triangle topologies, size and
reservation boundaries, no-allocation cases, and competing workgroups. It
retains the exact prior silent-empty failure and failure-publication, point-count
and query-status negative controls:

```
python3 src/asahi/vulkan/tests/robust-index-exhaustion.py --output /path/to/evidence
```

This requires Clang with address/undefined sanitizers and pthreads. Its emitter
and transport are shims and its subgroup lanes run serially. It does not prove
GPU execution, completion ordering, or visibility of the abort word on hardware.
Pressure and successful-rendering qualification remain required on each target.

## Host checks

Build both Gallium Asahi and Borealis with `-Dbuild-tests=true`, then run:

```
ninja -C build -j8
meson test -C build --no-rebuild --suite asahi --print-errorlogs
```

`asahi-agx3-packing`
executes the shared shadow-copy routine for all three index widths, 16 source
alignments, 1..33 available indices, and 1..32 appended zero indices. It checks
all output bytes and surrounding canaries. It also covers wrapped first/count
arithmetic, existing restart packets, state selectors and empty-slot links.
These host checks do not execute the GPU kernel or establish GPU correctness.

## Hardware correctness qualification (pending)

Test the exact candidate build on a G17P device. Preserve the existing kernel, session and
driver configuration. Use per-process driver selection and keep an immediately
available baseline launch configuration. Do not change the desktop driver.

Run the applicable Vulkan CTS robustness2 indexed-draw and indirect-count cases
discovered from the installed CTS case list. Record that list, skips, logs and
image/readback results. Exercise the following targeted cases as well:

| Dimension | Cases |
| --- | --- |
| API | indexed indirect and indexed indirect count |
| Record count | 0, 1, 2, 31, 32, 33, 1024, maximum advertised |
| Record stride | packed 20 bytes, padded 32 bytes, large valid stride |
| Index format | uint8, uint16, uint32 |
| Index range | in bounds, one-element overread, partial overread with 32 zero indices, fully out of bounds, null binding |
| Arithmetic | firstIndex at the range end and near UINT32_MAX; requested count near UINT32_MAX |
| Counts | zero indexCount, zero instanceCount, mixed empty/nonempty records; runtime count 0, 1, max-1, max, above max |
| Draw parameters | distinct signed vertexOffset, firstInstance, instanceCount and draw ID per record |
| Topology | points, lines, triangles, strips/fans; restart on/off and restart sentinel in each width |
| Fallback | statistics query, conditional rendering/inversion, multiview, adjacency, geometry/tessellation, depth bias |
| Ordering/lifetime | compute/transfer-written records and indices with proper barriers, barrier between two multi-draw calls, secondary command buffers, repeated submissions, VDM chunk links |

Compare output with a CPU-produced reference that explicitly zero-extends the
index input; an unpatched robust build alone is not a sufficient oracle. In
particular verify short unaligned shadows, uint32 zero padding beyond byte 32,
empty-record state/link handling, and distinct jobs that allocate shadows at
the same time. Monitor GPU faults and allocation failures. Exercise heap
exhaustion separately: it must not wrap the allocator or overlap successful
reservations, and completed host waits/polls must report `VK_ERROR_DEVICE_LOST`
instead of successful empty output. Check the abort word's GPU-to-host visibility
and persistence across later successful draws and heap reclamation. Also qualify
the single-draw fallback on M1/M2; host chip selectors do not prove its execution.
Zero-count and zero-instance controls must remain successful without allocating.

## Counters and performance A/B/C (pending)

Choose an installed vkd3d-proton title with a deterministic benchmark or saved
scene. Record title/build, Proton/vkd3d revision, resolution/settings, driver
SHA, kernel SHA, architecture and enabled robustness features. Confirm that it
actually issues multi-record indexed-indirect draws; a workload consisting of
single-record calls is a control and is not expected to improve.

| Arm | Mesa | Robustness |
| --- | --- | --- |
| A | unmodified base | enabled, normal title configuration |
| B | unmodified base | `HK_PERFTEST=norobust` |
| C | candidate | enabled, normal title configuration |

Keep other environment flags identical, including any flags required by the
baseline. `norobust` is a diagnostic ceiling, not a correctness configuration.
For translated games, select matching native and guest driver builds and verify
which library the process actually loaded.

Use alternating ABC/CBA order, at least five runs per arm, a 30-second warmup
and 60-second fixed measurement window. Begin each run below 85 C. Require
MemAvailable >= 1.5 GiB, swap <= 1 GiB and root free space >= 3 GiB; stop the
profiler at >95 C on two samples or >=100 C once,. A stop must remove only this test's instrumentation and leave the game
and session alone.

Capture fps, frame-time median/p99/max, submit count per frame and submit
p50/p99, CPU time blocked on GPU, and CPU/GPU frame-budget estimates. Collect
driver counters in separate diagnostic runs with `ASAHI_MESA_DEBUG=perf` to
avoid log overhead in the fps result. VDM submission lines report:

- `robust_index_draws`: repair slots, including execution-time empty records;
- `robust_index_dispatches`: repair launches and their strong publication tails;
- `robust_index_batches`: launches through the batch entry point.

For an eligible N-record call, C should report N repair slots, one repair
dispatch and one batch. Its count variant additionally has one count-predicate
dispatch. A single-record call keeps one repair dispatch. B should have no
robustness repairs. Compare summed counters over the same presented frames;
include fallback workloads so counter changes cannot conceal lost draws.

Retain raw samples, variance/CV, 95% bootstrap confidence intervals and Welch's
t results. Report a performance change only if its confidence interval excludes
zero, and reject a candidate with image differences, faults, or a material
frame-time regression. Restore the baseline per-process launch configuration
on failure and preserve the failing logs, exact SHAs and reproduction steps.
No target execution or fps result is claimed by this plan.
