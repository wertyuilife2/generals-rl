"""Retain tests of rejected economic and column-policy hypotheses."""
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parent
SDK = ROOT.parents[2] / 'arena/src/generals_arena/sdk'

class EconomicTrialRegressions(unittest.TestCase):
    def compile_trial(self, name):
        with tempfile.TemporaryDirectory() as tmp:
            binary = Path(tmp) / 'trial'
            subprocess.run(['c++', '-std=c++17', '-O2', '-I', str(SDK),
                            str(ROOT / ('test_v8_' + name + '_trial.cpp')),
                            '-o', str(binary)], check=True)
            subprocess.run([str(binary)], check=True)

    def test_short_tenure_history(self):
        self.compile_trial('land_tenure')

    def test_delivered_refill_and_friendly_transport(self):
        self.compile_trial('refill')

    def test_capture_before_and_after_the_income_pulse(self):
        self.compile_trial('harvest')

    def test_uniform_terminal_column_valuation(self):
        self.compile_trial('column')

if __name__ == '__main__':
    unittest.main()
