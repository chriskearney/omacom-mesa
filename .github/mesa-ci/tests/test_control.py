# SPDX-License-Identifier: MIT
import importlib.util
from pathlib import Path
import unittest

root = Path(__file__).resolve().parents[1]

def load(name):
    spec = importlib.util.spec_from_file_location(name, root / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

plan, gate, policy, dependencies = load('plan'), load('gate'), load('policy'), load('build-tests')

class ControlTests(unittest.TestCase):
    def needs(self, build='true', result='success'):
        return {'plan': {'result': 'success', 'outputs': {'build': build}},
                'source': {'result': 'success'}, 'build': {'result': result}}

    def test_doc_only(self):
        self.assertTrue(plan.docs_only(['README.md', 'docs/asahi.rst']))
        for paths in [[], ['docs/build.py'], ['.github/workflows/mesa.yml'],
                      ['src/asahi/compiler/agx_compile.c'], ['meson.build'], ['requirements.txt'],
                      ['docs/guide.rst', 'include/vulkan/vulkan.h'], ['unknown']]:
            self.assertFalse(plan.docs_only(paths))

    def test_python_wrapped_executable_is_built(self):
        build = Path('/tmp/build')
        tests = [{'cmd': ['python3', 'wrapper.py', 'hidden-test'], 'depends': ['hidden@exe']}]
        targets = [{'id': 'hidden@exe', 'build_by_default': False,
                    'filename': ['/tmp/build/path/hidden-test']}]
        self.assertEqual(dependencies.outputs(build, tests, targets), ['path/hidden-test'])
        with self.assertRaises(ValueError):
            dependencies.outputs(build, tests, [])

    def test_required_test_evidence_cannot_disappear(self):
        good = [{'name': 'suite - mesa:' + n, 'result': 'OK'} for n in policy.REQUIRED_SUITES]
        self.assertEqual(policy.test_summary(good), {'OK': 4})
        for bad in [[], good[:-1], [dict(r, result='SKIP') for r in good],
                    good + [{'name': 'another-test', 'result': 'FAIL'}]]:
            with self.assertRaises(ValueError):
                policy.test_summary(bad)

    def test_required_work_fails_closed(self):
        self.assertTrue(gate.passes(self.needs()))
        self.assertTrue(gate.passes(self.needs('false', 'skipped')))
        for result in ['failure', 'cancelled', 'skipped', '', 'neutral']:
            self.assertFalse(gate.passes(self.needs(result=result)))
        for flag in ['', 'True', 'false ', 'unknown']:
            self.assertFalse(gate.passes(self.needs(flag)))
        for job in ['plan', 'source']:
            for result in ['failure', 'cancelled', 'skipped']:
                needs = self.needs('false', 'skipped')
                needs[job]['result'] = result
                self.assertFalse(gate.passes(needs))
        self.assertFalse(gate.passes({}))
        self.assertFalse(gate.passes(self.needs('false', 'failure')))

if __name__ == '__main__':
    unittest.main()
