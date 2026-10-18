#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Exercise host driver setup and mappings through drm-shim, without GPU submits."""
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile

root = Path(__file__).resolve().parents[2]
build = Path(sys.argv[1]).resolve()
shim = build / 'src/asahi/drm-shim/libasahi_noop_drm_shim.so'
icds = list((build / 'src/asahi/vulkan').glob('asahi_devenv_icd.*.json'))
assert shim.is_file() and len(icds) == 1, 'Build the driver, development ICD and drm-shim first'
with tempfile.TemporaryDirectory(prefix='siliconforest-vulkan-smoke-') as directory:
    binary = str(Path(directory) / 'vulkan-smoke')
    subprocess.run(shlex.split(os.environ.get('CC', 'cc')) +
                   ['-std=c11', '-O2', '-Wall', '-Wextra', '-Werror',
                    '-I' + str(root / 'include'),
                    str(root / 'tools/siliconforest/vulkan-smoke.c'),
                    '-lvulkan', '-ldl', '-o', binary], check=True)
    for chip in ['g13g', 'g15s', 'g15c', 'g17p']:
        env = os.environ.copy()
        env.update(LD_PRELOAD=str(shim), VK_DRIVER_FILES=str(icds[0]),
                   ASAHI_SHIM_CHIP=chip, ASAHI_SHIM_G15S_EXPECT_VALID='1')
        for key in ['VK_ICD_FILENAMES', 'VK_LOADER_DRIVERS_SELECT', 'VK_LOADER_DRIVERS_DISABLE']:
            env.pop(key, None)
        print('drm-shim profile:', chip, flush=True)
        subprocess.run([binary], env=env, check=True, timeout=90)
