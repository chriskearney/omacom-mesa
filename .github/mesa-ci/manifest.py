#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Bind staged driver bytes, test results and dependency inventory to an exact commit."""
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess

from policy import test_summary

root = Path('.ci-stage')
elfs = []
files = []
for path in sorted(root.rglob('*')):
    if path.is_symlink():
        if not path.resolve().is_relative_to(root.resolve()):
            raise SystemExit(f'Staged symlink escapes install root: {path}')
        files.append({'path': str(path.relative_to(root)), 'symlink': os.readlink(path)})
        continue
    if not path.is_file():
        continue
    data = path.read_bytes()
    files.append({'path': str(path.relative_to(root)), 'sha256': hashlib.sha256(data).hexdigest()})
    if data[:4] == b'\x7fELF':
        dynamic = subprocess.check_output(['readelf', '-d', str(path)], text=True)
        if any('NEEDED' in line and ('libLLVM' in line or 'libSPIRV-Tools' in line)
               for line in dynamic.splitlines()):
            raise SystemExit(f'Unexpected compiler runtime dependency in {path}')
        elfs.append(str(path.relative_to(root)))
if not elfs or not any(Path(p).name == 'libvulkan_asahi.so' for p in elfs):
    raise SystemExit('Missing staged driver ELF payload')
logs = Path('.ci-build/meson-logs/testlog.json')
results = [json.loads(line) for line in logs.read_text().splitlines()]
summary = test_summary(results)
provenance = {
    'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
    'architecture': platform.machine(),
    'kernel_contract': {k: v for k, v in json.loads(Path('SILICONFOREST-SOURCES.json').read_text())['integration_followups_20261024'].items() if k.startswith('kernel_')},
    'event': os.environ.get('GITHUB_EVENT_NAME', 'local'),
    'run_id': os.environ.get('GITHUB_RUN_ID'),
    'run_attempt': os.environ.get('GITHUB_RUN_ATTEMPT'),
    'hardware_qualified': False,
    'claim': 'CPU build/tests and no-submit drm-shim only; not a release approval or Arch package',
    'test_results': summary,
    'elfs': elfs, 'files': files,
    'packages': subprocess.check_output(['dpkg-query', '-W', '-f=${Package}=${Version}\n'], text=True).splitlines(),
}
Path('.ci-manifest.json').write_text(json.dumps(provenance, indent=2) + '\n')
