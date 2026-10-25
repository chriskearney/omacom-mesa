#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Only a narrow allowlist of documentation changes can skip driver compilation."""
import os
from pathlib import PurePosixPath
import re
import subprocess


def docs_only(paths):
    return bool(paths) and all(
        (p.startswith('docs/') and PurePosixPath(p).suffix in {'.md', '.rst', '.txt', '.svg', '.png'})
        or p in {'README.md', 'README.rst', 'CONTRIBUTING.md', 'CODE_OF_CONDUCT.md', 'SECURITY.md', 'COPYING'}
        or ('/' not in p and p.startswith('SILICONFOREST-') and p.endswith('.md'))
        for p in paths
    )


def plan(base):
    if not re.fullmatch(r'[0-9a-f]{40}', base or '') or base == '0' * 40:
        return True
    subprocess.run(['git', 'fetch', '--no-tags', '--depth=1', 'origin', base], check=True)
    names = subprocess.check_output(
        ['git', 'diff', '--name-only', '--no-renames', '-z', base, 'HEAD'])
    paths = names.decode('utf-8', errors='strict').rstrip('\0').split('\0') if names else []
    return not docs_only(paths)


if __name__ == '__main__':
    build = True if os.environ.get('GITHUB_EVENT_NAME') in {'push', 'merge_group', 'workflow_dispatch'} else plan(os.environ.get('BASE_SHA', ''))
    with open(os.environ['GITHUB_OUTPUT'], 'a') as out:
        print(f'build={str(build).lower()}', file=out)
    print('Full driver builds required:', build)
