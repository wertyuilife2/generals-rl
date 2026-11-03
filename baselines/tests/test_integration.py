"""Package boundaries and compatibility with simulation controller assembly."""
import subprocess
import sys
import unittest

from generals.controllers import make_controllers as simulation_controllers
from generals.protocol.controller import Controller
from generals_baselines import AI_KINDS, make_controllers


class IntegrationTests(unittest.TestCase):
    def test_simulation_and_baseline_factories_preserve_seat_streams(self):
        ai = make_controllers(AI_KINDS, seed=1729)
        simulation = simulation_controllers(AI_KINDS, seed=1729)
        for left, right in zip(ai, simulation):
            self.assertIsInstance(left, Controller)
            self.assertIs(type(left), type(right))
            self.assertEqual(left.rng.bit_generator.state, right.rng.bit_generator.state)

    def test_controller_factory_and_engine_import_without_baselines(self):
        code = '''
import sys
from generals.controllers import create_controller
from generals.core.engine import CoreEngine
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
