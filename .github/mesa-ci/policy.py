# SPDX-License-Identifier: MIT
"""Evidence required before a staged build can become a delivery candidate."""
REQUIRED_SUITES = {'agx_tests', 'nir_tests', 'spirv_tests', 'asahi-g15-wire'}


def test_summary(results):
    if not results or any(r.get('result') not in {'OK', 'SKIP', 'EXPECTEDFAIL'} for r in results):
        raise ValueError('Missing or failed Meson tests')
    passed = {r['name'].rsplit(':', 1)[-1] for r in results if r['result'] == 'OK'}
    if not REQUIRED_SUITES <= passed:
        raise ValueError('Required compiler/wire suites did not pass: ' + str(REQUIRED_SUITES - passed))
    return {key: sum(r['result'] == key for r in results) for key in sorted({r['result'] for r in results})}
