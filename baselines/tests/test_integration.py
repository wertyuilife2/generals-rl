"""Package boundaries and compatibility with interactive controller assembly."""
import subprocess
import sys
import unittest

from generals.controllers import make_controllers as interactive_controllers
from generals.protocol.controller import Controller
from generals_baselines import AI_KINDS, make_controllers


class IntegrationTests(unittest.TestCase):
    def test_interactive_and_baseline_factories_preserve_seat_streams(self):
        ai = make_controllers(AI_KINDS, seed=1729)
        interactive = interactive_controllers(AI_KINDS, seed=1729)
        for left, right in zip(ai, interactive):
            self.assertIsInstance(left, Controller)
            self.assertIs(type(left), type(right))
            self.assertEqual(left.rng.bit_generator.state, right.rng.bit_generator.state)
        mixed = interactive_controllers(["human", "random"], seed=1729)
        original = make_controllers(["aggressive", "random"], seed=1729)
        self.assertIsInstance(mixed[0], Controller)
        self.assertEqual(mixed[1].rng.bit_generator.state, original[1].rng.bit_generator.state)

    def test_human_and_engine_import_without_baselines(self):
        code = '''
import sys
from generals.controllers import create_controller
from generals.core.engine import CoreEngine
create_controller("human")
assert not any(n.startswith(("generals_baselines", "torch", "tkinter")) for n in sys.modules)
'''
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_baselines_do_not_import_training_or_gui(self):
        code = '''
import sys
from generals_baselines import make_controllers
make_controllers(["aggressive", "random"])
assert not any(n.startswith(("generals_rl", "rsl_rl", "torch", "tkinter")) for n in sys.modules)
'''
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
