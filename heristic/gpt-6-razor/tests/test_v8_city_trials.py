"""Archive checked mechanics without promoting their unsuccessful match results."""
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parent
SDK = ROOT.parents[2] / 'arena/src/generals_arena/sdk'

class CityTrialRegressions(unittest.TestCase):
    def test_archived_city_and_scout_hypotheses(self):
        for name in ('city_control', 'capability_scout'):
            with self.subTest(trial=name), tempfile.TemporaryDirectory() as tmp:
                binary = Path(tmp) / 'trial'
                subprocess.run(['c++', '-std=c++17', '-O2', '-I', str(SDK),
                                str(ROOT / ('test_v8_' + name + '_trial.cpp')),
                                '-o', str(binary)], check=True)
                subprocess.run([str(binary)], check=True)

if __name__ == '__main__':
    unittest.main()
