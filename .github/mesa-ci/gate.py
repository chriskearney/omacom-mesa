#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Fail closed on failed, cancelled, missing or unexpectedly skipped work."""
import json
import os
import sys


def passes(needs):
    if set(needs) != {'plan', 'source', 'build'}:
        return False
    if any(needs[j].get('result') != 'success' for j in ('plan', 'source')):
        return False
    required = needs['plan'].get('outputs', {}).get('build')
    return (required == 'true' and needs['build'].get('result') == 'success') or (
        required == 'false' and needs['build'].get('result') == 'skipped')


if __name__ == '__main__':
    needs = json.loads(os.environ['NEEDS_JSON'])
    print(json.dumps(needs, indent=2))
    sys.exit(0 if passes(needs) else 1)
