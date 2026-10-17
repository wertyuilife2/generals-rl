"""Preserve tested hypotheses without promoting rejected strategy variants."""
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parent
SDK = ROOT.parents[2] / 'arena/src/generals_arena/sdk'

class CampaignTrialRegressions(unittest.TestCase):
    def test_archived_campaign_hypotheses(self):
        for name in ('campaign_ledger', 'king_race', 'king_race_bound'):
            with self.subTest(trial=name), tempfile.TemporaryDirectory() as tmp:
                binary = Path(tmp) / 'trial'
                subprocess.run(['c++', '-std=c++17', '-O2', '-I', str(SDK),
                                str(ROOT / ('test_v8_' + name + '_trial.cpp')),
                                '-o', str(binary)], check=True)
                subprocess.run([str(binary)], check=True)

if __name__ == '__main__':
    unittest.main()
