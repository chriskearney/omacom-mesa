# CI validation control only. Never merge this branch.
import unittest

class DeliberateFailure(unittest.TestCase):
    def test_required_gate_blocks_a_failure(self):
        self.fail('Deliberate CI failure: the required check must block this PR')
