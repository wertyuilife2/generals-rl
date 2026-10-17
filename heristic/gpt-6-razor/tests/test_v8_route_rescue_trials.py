"""Route reward and rescue semantics on immutable experimental sources."""
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parent
SDK = ROOT.parents[2] / 'arena/src/generals_arena/sdk'


class RouteRescueTrials(unittest.TestCase):
    def test_remaining_rewards_and_shared_rescue_troops(self):
        for case in ('route_value', 'rescue_ledger', 'rescue_readiness'):
            with self.subTest(case=case), tempfile.TemporaryDirectory() as tmp:
                binary = Path(tmp) / case
                subprocess.run(['c++', '-std=c++17', '-O2', '-I', str(SDK),
                                str(ROOT / f'test_v8_{case}_trial.cpp'),
                                '-o', str(binary)], check=True)
                subprocess.run([str(binary)], check=True)


if __name__ == '__main__':
    unittest.main()
