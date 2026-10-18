# Maintaining SiliconForest Mesa

The default integration is `siliconforest/next`. Keep original contributor
commits and licensing intact; use merges or `git cherry-pick -x` with a written
record of any conflict resolution. Label newly generated commits as described
in `AGENTS.md`.

## Reviewing a source update

1. Check the source repository's visibility before including its refs or code in
   a publication. For GitHub, use `gh api repos/OWNER/REPO --jq .visibility`.
   Private-source work stays in local research refs and notes. This public
   integration contains public-source code only.
2. Pin the exact source SHA, compare divergence against the current integration,
   and review dependencies and duplicate implementations. A newer topic name is
   not proof that it includes another branch's fixes.
3. Keep identity, topology, USC, HAL and kernel-ABI admission checks intact.
   Hardware recognition and shader encoding alone do not establish support.
4. Update `SILICONFOREST-SOURCES.json` and the review ledger. Source tags preserve
   evidence; their existence does not mean every experiment is approved for the
   default build.

## Checking the integration

Run the quick checks, build the affected drivers and all registered Asahi test
dependencies, and run the host suite. For changes around device setup, mappings
or synchronization, also run the Vulkan drm-shim regression probe with Khronos
validation enabled. Commands and the exact baseline results are in
`SILICONFOREST-VALIDATION.md`.

`check.py --check-git` works with local maintainer branches or the corresponding
`origin/*` refs in a fresh full clone. It verifies exact source tags and the
limited changes to compatibility profiles. The quick GitHub workflow runs
without requiring all historical source tags or a complete Mesa build.

## Publishing source

Push explicit branch/tag refs from the reviewed source manifest. Do not use an
unrestricted `--mirror`, `--all` or `--tags` push from a research checkout that
also holds private or unreviewed refs. For a new repository, keep it private until
its remote ref set and source access checks match the intended public inventory.
Cancelling a local wrapper process is not proof that its upload children stopped.

Keep experimental firmware/driver profiles separate. Before publishing installable
packages, qualify the exact board, firmware, m1n1 handoffs and kernel ABI on
hardware, run the relevant CTS and compositor/application checks, and record the
results against a pinned source. Host CPU tests and drm-shim checks do not replace
that hardware qualification. Source maintenance must not install Mesa or alter a
working boot stack as a side effect.
