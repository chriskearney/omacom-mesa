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

plan, gate = load('plan'), load('gate')

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
