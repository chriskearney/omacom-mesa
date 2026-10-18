#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build the Apple driver outputs and Asahi host tests, not other drivers' tests."""
import json
import os
from pathlib import Path
import subprocess
import sys

build = Path(sys.argv[1]).resolve()
info = build / 'meson-info'
outputs = set()
dependencies = set()
for test in json.loads((info / 'intro-tests.json').read_text()):
    if any(suite == 'asahi' or suite.endswith(':asahi') for suite in test['suite']):
        dependencies.update(test.get('depends', []))
        executable = Path(test['cmd'][0])
        if executable.is_relative_to(build):
            outputs.add(str(executable.relative_to(build)))
required = {'vulkan_asahi', 'asahi_noop_drm_shim', 'GL', 'EGL', 'gbm',
            'asahi_icd.x86_64.json', 'asahi_devenv_icd.x86_64.json'}
seen = set()
for target in json.loads((info / 'intro-targets.json').read_text()):
    name = target['name']
    if target['id'] in dependencies or name in required or name.startswith('gallium-') or (name.startswith('asahi') and 'icd.' in name):
        seen.add(name)
        outputs.update(str(Path(filename).relative_to(build)) for filename in target['filename'])
# ICD architecture varies for cross/native AArch64 builds.
assert required - {'asahi_icd.x86_64.json', 'asahi_devenv_icd.x86_64.json'} <= seen, seen
jobs = int(os.environ.get('JOBS', '8'))
assert jobs > 0
subprocess.run(['ninja', '-C', str(build), '-j', str(jobs), *sorted(outputs)], check=True)
