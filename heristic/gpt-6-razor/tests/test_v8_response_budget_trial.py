"""Check a bounded response scheduler; this is not a strength test."""
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parent
SDK = ROOT.parents[2] / 'arena/src/generals_arena/sdk'


class ResponseBudgetTrial(unittest.TestCase):
    def test_wait_and_recall_share_real_troops_and_action_deadlines(self):
        for case in ('response_budget', 'response_counter', 'response_guard'):
            with self.subTest(case=case), tempfile.TemporaryDirectory() as tmp:
                binary = Path(tmp) / case
                subprocess.run(['c++', '-std=c++17', '-O2', '-I', str(SDK),
                                str(ROOT / f'test_v8_{case}_trial.cpp'),
                                '-o', str(binary)], check=True)
                subprocess.run([str(binary)], check=True)


if __name__ == '__main__':
    unittest.main()
