# Omacom Mesa CI

Proposed integration CI, vetted first in `chriskearney/omacom-mesa`.
No CODEOWNERS, Omacom repository settings, package source pins, or driver code are changed.

## PR checks

`ci / result` is the single required check. It always runs and fails on failures,
cancellations, absent planner output or unexpected skips. The workflow has no
path/branch filters on PRs, supports `merge_group`, and tests GitHub's checked-out
merge commit rather than replacing it with the PR head. New PR commits cancel
older PR runs; queued merges and post-merge builds do not cancel one another.

Every PR runs production CPU fixtures, pinned kernel UAPI-layout checks, workflow lint and negative CI-control tests.
Only a small documentation allowlist can skip native builds on PRs. Unknown paths,
CI files, dependency metadata and every source change build both architectures.
Merge-queue, manual and integration-branch push runs always build both.

The parallel x86_64 and native aarch64 jobs use hosted Ubuntu 26.04, a matching
LLVM/Clang/translator 22 toolchain and hash-pinned Meson 1.10.2. Four build workers
match the hosted CPUs while bounding peak memory. Compiler tools come from the same source; the final
Asahi/Zink/Honeykrisp driver build disables LLVM and SPIRV-Tools linkage. It runs
all configured Meson tests, then the existing no-submit drm-shim admission,
pipeline and memory-map probes with Vulkan validation. Complete shader-compilation probes also cover float-to-int64 and perspective interpolateAtOffset on G13G, G15S, G15C, G17P and G15G. Installed ELF dependencies
are checked before a manifest is created. CPU/shim success is not GPU or CTS proof.

Ccache speeds up subsequent PR revisions. Keys include architecture and build
configuration, and GitHub isolates PR cache writes to each PR ref. Post-merge
builds disable ccache and do not restore PR caches. No build directory, prebuilt
shader tool or driver binary is shared between runs. Timeouts, bounded cache size
and artifact retention limit cost. Exact cold/warm timings are measured in the
personal-fork trial, not promised in advance.

## Delivery and release

A successful push to `integration/aurora` retains a staged ARM driver candidate
and manifest, not an Arch package or automatic release. The manifest identifies
the exact commit, runner architecture, run/attempt, per-file SHA-256, tested suites
and installed toolchain packages. Failure artifacts retain Meson configure logs,
test logs and JUnit evidence. Artifacts have 14-day retention.

These are integration/debug builds with assertions. Shipping remains in
`omarchy-pkgs`: pin an exact Omacom source commit, build the actual split packages
with its production flags, obtain relevant hardware/CTS qualification and the
normal maintainer release approval. Do not publish a PR artifact or assume a
hosted ARM CPU is an Apple GPU. A new compiler/kernel contract needs new hardware
results. Existing M3 Air and full CTS limitations remain.

PR code runs only on disposable GitHub-hosted runners with a read-only token,
no secrets, no retained checkout credentials and full-SHA action pins. There is no
`pull_request_target`, privileged `workflow_run`, automatic lab access, OIDC or
package-write permission. Action pin updates should be reviewed like source.

## Maintainer rollout after the trial

1. Establish Omacom's intended `integration/aurora` base and land this workflow there.
2. Require the exact `ci / result` check from the GitHub Actions app (ID 15368).
   Do not require optional matrix job names or the old lightweight check.
3. Require PR review and approval of the latest push; disallow branch deletion and
   force-pushes. Merge queues can use this workflow's `merge_group` coverage.
4. Confirm a failing PR is blocked, a docs-only PR succeeds without build-pending
   checks, and a merge-group run produces the same required context before release.

CODEOWNERS and owner-review settings are deliberately outside this change, per
Chris's hold. Workflow files and CI-control code are PR-editable, so the required
status check alone is not tamper-proof: human review of workflow/gate changes or
an organization-controlled required workflow is still necessary. Required status
checks cannot enforce a numeric hardware/CTS result that this workflow never ran.

Public design references:
- https://docs.github.com/en/actions/reference/security/secure-use
- https://docs.github.com/en/pull-requests/how-tos/merge-and-close-pull-requests/troubleshooting-required-status-checks
- https://github.com/actions/runner-images
