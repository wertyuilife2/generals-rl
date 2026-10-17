"""Preserve mechanics of an unsuccessful regional-defense experiment."""
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parent
SDK = ROOT.parents[2] / 'arena/src/generals_arena/sdk'

class RegionalTrial(unittest.TestCase):
    def test_archived_joint_response_conserves_troops_and_checks_routes(self):
        with tempfile.TemporaryDirectory() as tmp:
            binary = Path(tmp) / 'regional'
            subprocess.run(['c++', '-std=c++17', '-O2', '-I', str(SDK),
                            str(ROOT / 'test_v8_regional_trial.cpp'),
                            '-o', str(binary)], check=True)
            subprocess.run([str(binary)], check=True)

if __name__ == '__main__':
    unittest.main()
