#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
set -euo pipefail
# Official v1.7.12 release; source commit 914e7df21a07ef503a81201c76d2b11c789d3fca.
# Verify immutable bytes instead of recompiling Go dependencies on every PR.
ci_lint_directory=$(mktemp -d "${RUNNER_TEMP:-/tmp}/mesa-actionlint.XXXXXX")
trap 'rm -rf -- "$ci_lint_directory"' EXIT
curl --fail --silent --show-error --location --retry 3 --proto '=https' --proto-redir '=https' \
  https://github.com/rhysd/actionlint/releases/download/v1.7.12/actionlint_1.7.12_linux_amd64.tar.gz \
  -o "$ci_lint_directory/actionlint.tar.gz"
printf '%s  %s\n' \
  8aca8db96f1b94770f1b0d72b6dddcb1ebb8123cb3712530b08cc387b349a3d8 \
  "$ci_lint_directory/actionlint.tar.gz" | sha256sum --check
tar -xzf "$ci_lint_directory/actionlint.tar.gz" -C "$ci_lint_directory" actionlint
"$ci_lint_directory/actionlint" -shellcheck="${ACTIONLINT_SHELLCHECK-shellcheck}" .github/workflows/mesa.yml
