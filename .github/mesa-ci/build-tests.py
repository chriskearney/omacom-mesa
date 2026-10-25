#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build every registered test dependency, including build_by_default:false wrappers."""
import json
import os
from pathlib import Path
import subprocess
import sys


def outputs(build, tests, targets):
    needed = {dep for test in tests for dep in test.get('depends', [])}
    selected = {target['id']: target for target in targets if target['id'] in needed}
    if missing := needed - selected.keys():
        raise ValueError('Missing introspection targets: ' + str(missing))
    return sorted({str(Path(filename).relative_to(build))
                   for target in selected.values() for filename in target['filename']})


if __name__ == '__main__':
    build = Path(sys.argv[1]).resolve()
    info = build / 'meson-info'
    tests = json.loads((info / 'intro-tests.json').read_text())
    targets = json.loads((info / 'intro-targets.json').read_text())
    files = outputs(build, tests, targets)
    if not tests or not files:
        raise SystemExit('No configured tests or test executables')
    print(f'Building {len(files)} outputs for {len(tests)} configured tests', flush=True)
    if '--dry-run' in sys.argv[2:]:
        print('\n'.join(files))
    else:
        subprocess.run(['ninja', '-C', str(build), '-j', os.environ.get('CI_JOBS', '4'), *files], check=True)
