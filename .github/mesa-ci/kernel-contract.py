#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Fetch only the public, exact kernel contract declared by this source tree."""
import json
from pathlib import Path
import re

contract = json.loads(Path('SILICONFOREST-SOURCES.json').read_text())['integration_followups_20261024']
assert contract['kernel_repository'] == 'https://github.com/omacom/linux-aurora'
assert contract['kernel_header'] == 'include/uapi/drm/asahi_drm.h'
assert re.fullmatch(r'[0-9a-f]{40}', contract['kernel_commit'])
print('sha=' + contract['kernel_commit'])
