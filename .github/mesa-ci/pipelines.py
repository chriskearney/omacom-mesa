#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Exercise complete shader compilation on emulated device profiles; no queue submits."""
import os
from pathlib import Path
import subprocess
import tempfile

root = Path(__file__).resolve().parent
build = Path('.ci-build').resolve()
icds = list((build / 'src/asahi/vulkan').glob('asahi_devenv_icd.*.json'))
assert len(icds) == 1
with tempfile.TemporaryDirectory(prefix='mesa-ci-pipelines-') as directory:
    work = Path(directory)
    shaders = ['conversion.comp', 'interpolation.vert', 'interpolation.frag']
    binaries = [work / (s + '.spv') for s in shaders]
    for shader, binary in zip(shaders, binaries):
        subprocess.run(['glslangValidator', '-V', '--target-env', 'vulkan1.3',
                        str(root / shader), '-o', str(binary)], check=True)
    executable = work / 'pipeline'
    subprocess.run(['cc', '-std=c11', '-O2', '-Wall', '-Wextra', '-Werror',
                    '-Iinclude', str(root / 'pipeline.c'), '-lvulkan', '-ldl',
                    '-o', str(executable)], check=True)
    for chip in ['g13g', 'g15s', 'g15c', 'g17p', 'g15g']:
        env = os.environ.copy()
        for key in ['VK_ICD_FILENAMES', 'VK_LOADER_DRIVERS_SELECT',
                    'VK_LOADER_DRIVERS_DISABLE', 'ASAHI_M3_EXPERIMENTAL']:
            env.pop(key, None)
        env.update(LD_PRELOAD=str(build / 'src/asahi/drm-shim/libasahi_noop_drm_shim.so'),
                   VK_DRIVER_FILES=str(icds[0]), ASAHI_SHIM_CHIP=chip,
                   ASAHI_SHIM_G15S_EXPECT_VALID='1', MESA_SHADER_CACHE_DISABLE='true',
                   VK_INSTANCE_LAYERS='VK_LAYER_KHRONOS_validation')
        result = subprocess.run([str(executable), *map(str, binaries)], env=env,
                                text=True, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, timeout=90)
        print(chip, result.stdout, flush=True)
        result.check_returncode()
        if 'Validation Error:' in result.stdout:
            raise RuntimeError('Invalid compiler regression probe')
