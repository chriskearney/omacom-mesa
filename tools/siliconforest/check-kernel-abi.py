#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Compare native Mesa wire layouts with the experimental SiliconForest header."""
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile

root = Path(__file__).resolve().parents[2]
kernel = Path(sys.argv[1]).resolve()
common = {
    'params_size': 'sizeof(struct drm_asahi_params_global)',
    'params_hal_offset': 'offsetof(struct drm_asahi_params_global, gpu_hal_generation)',
    'queue_size': 'sizeof(struct drm_asahi_queue_create)',
    'vm_size': 'sizeof(struct drm_asahi_vm_create)',
    'submit_size': 'sizeof(struct drm_asahi_submit)',
    'vm_ioctl': 'DRM_IOCTL_ASAHI_VM_CREATE',
    'queue_ioctl': 'DRM_IOCTL_ASAHI_QUEUE_CREATE',
    'submit_ioctl': 'DRM_IOCTL_ASAHI_SUBMIT',
    'per_command_syncs': 'DRM_ASAHI_FEATURE_PER_COMMAND_INPUT_SYNCS',
    'vm_status_mirror': 'DRM_ASAHI_FEATURE_VM_STATUS_MIRROR',
    'prior_barriers': 'DRM_ASAHI_FEATURE_EXACT_PRIOR_BARRIERS',
    'fragment_barriers': 'DRM_ASAHI_FEATURE_FRAGMENT_BARRIERS',
    'ray_query': 'DRM_ASAHI_FEATURE_RENDER_RAY_QUERY',
}
mesa = dict(common, render_size='sizeof(struct drm_asahi_cmd_render_with_ray_query)',
            render_prior_offset='offsetof(struct drm_asahi_cmd_render_with_prior, prior_vdm)',
            compute_size='sizeof(struct drm_asahi_cmd_compute_with_prior)',
            compute_scs_offset='offsetof(struct drm_asahi_cmd_compute_with_prior, scs_layout)',
            compute_prior_offset='offsetof(struct drm_asahi_cmd_compute_with_prior, prior_vdm)')
native = dict(common, render_size='sizeof(struct drm_asahi_cmd_render)',
              render_prior_offset='offsetof(struct drm_asahi_cmd_render, prior_vdm)',
              compute_size='sizeof(struct drm_asahi_cmd_compute)',
              compute_scs_offset='offsetof(struct drm_asahi_cmd_compute, scs_layout)',
              compute_prior_offset='offsetof(struct drm_asahi_cmd_compute, prior_vdm)')

def compile_and_run(directory, name, header, includes, fields):
    source = Path(directory) / (name + '.c')
    binary = Path(directory) / name
    body = '\n'.join(f'printf("{key} %llu\\n", (unsigned long long)({expr}));'
                     for key, expr in fields.items())
    source.write_text('#define __user\n#include <stddef.h>\n#include <stdio.h>\n#include "' + header + '"\nint main(void) {\n' + body + '\nreturn 0; }\n')
    subprocess.run(shlex.split(os.environ.get('CC', 'cc')) + ['-std=c11', '-Werror'] +
                   ['-I' + str(path) for path in includes] +
                   [str(source), '-o', str(binary)], check=True)
    return {key: int(value) for key, value in
            (line.split() for line in subprocess.check_output([binary], text=True).splitlines())}

with tempfile.TemporaryDirectory(prefix='siliconforest-abi-') as directory:
    a = compile_and_run(directory, 'mesa', 'drm-uapi/asahi_drm.h', [root / 'include'], mesa)
    b = compile_and_run(directory, 'kernel', 'asahi_m3_drm.h', [kernel / 'include/uapi/drm'], native)
    differences = {key: {'mesa': a[key], 'kernel': b[key]} for key in a if a[key] != b[key]}
    print(json.dumps({'matched': not differences, 'values': a, 'differences': differences}, indent=2))
    raise SystemExit(bool(differences))
