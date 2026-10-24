#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Run production CPU checks without a GPU or a Mesa build directory."""
import argparse
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]

def run(*args):
    return subprocess.check_output(args, cwd=ROOT, text=True).strip()

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-git', action='store_true',
                        help='also verify every pinned tag and profile in a full clone')
    args = parser.parse_args()
    data = json.loads((ROOT / 'SILICONFOREST-SOURCES.json').read_text())
    tags = set()
    for source in data['source_snapshots']:
        assert re.fullmatch(r'[0-9a-f]{40}', source['commit']), source
        assert source['tag'] not in tags, source['tag']
        assert source['url'].startswith(('https://github.com/', 'https://gitlab.freedesktop.org/'))
        tags.add(source['tag'])
        if args.check_git:
            assert run('git', 'rev-parse', 'refs/tags/' + source['tag']) == source['commit']
    if args.check_git:
        for profile in data['profiles']:
            local = subprocess.run(['git', 'rev-parse', '--verify', '--quiet',
                                    'refs/heads/' + profile['branch']],
                                   cwd=ROOT, text=True, capture_output=True)
            actual = local.stdout.strip() if local.returncode == 0 else run(
                'git', 'rev-parse', 'refs/remotes/origin/' + profile['branch'])
            assert actual == profile['commit']
            subprocess.run(['git', 'merge-base', '--is-ancestor', profile['source_commit'], profile['commit']], cwd=ROOT, check=True)
            paths = run('git', 'diff', '--name-only', profile['source_commit'], profile['commit']).splitlines()
            assert set(paths) <= {'AGENTS.md', 'README.md', '.github/workflows/macos.yml'}, paths
    cases = [
        ('chip-admission', ['src/asahi/lib/agx_chip.c',
                            'src/asahi/lib/agx_va_runtime_hole.c',
                            'src/util/vma.c',
                            'src/asahi/lib/tests/test-chip-admission.c']),
        ('compression', ['src/asahi/vulkan/tests/test-hk-g17p-compression.c']),
        ('image-plane-layout', ['src/asahi/vulkan/hk_image_layout_core.c',
                                'src/asahi/vulkan/tests/test-hk-image-layout-core.c']),
        ('scratch-layout', ['src/asahi/lib/agx_scs_layout.c',
                            'src/asahi/lib/tests/test-scs-layout.c']),
    ]
    cc = shlex.split(os.environ.get('CC', 'cc'))
    with tempfile.TemporaryDirectory(prefix='siliconforest-mesa-check-') as directory:
        for name, sources in cases:
            binary = str(Path(directory) / name)
            subprocess.run(cc + ['-std=c11', '-D_POSIX_C_SOURCE=200809L', '-D_DEFAULT_SOURCE',
                                  '-DHAVE_ENDIAN_H', '-O2', '-Wall', '-Wextra', '-Werror',
                                  '-Wno-unused-function', '-Iinclude', '-Isrc',
                                  '-Isrc/asahi/lib', '-Isrc/asahi/vulkan'] +
                           sources + ['-o', binary], cwd=ROOT, check=True)
            subprocess.run([binary], check=True)
    subprocess.run([sys.executable, str(ROOT / 'tools/siliconforest/test-geometry-heap.py')], check=True)
    print(f'{len(tags)} source records and {len(cases) + 1} production CPU test programs passed')

if __name__ == '__main__':
    main()
