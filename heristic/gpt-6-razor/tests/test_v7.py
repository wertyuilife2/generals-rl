"""Razor v7: regression coverage for transport, split accounting and replies."""
from pathlib import Path
import subprocess
import tempfile
import unittest
import test_v6 as v6
PROJECT, ROOT = v6.PROJECT, v6.ROOT

class RazorV7Regressions(v6.RazorV6Regressions):
    version = 'v7'

    def test_supply_program_conserves_army_and_does_not_reward_pure_relocation(self):
        with tempfile.TemporaryDirectory() as tmp:
            binary = Path(tmp) / 'planning'
            subprocess.run(['c++', '-std=c++17', '-O2', '-I',
                str(PROJECT / 'arena/src/generals_arena/sdk'),
                str(ROOT / 'test_v7_planning.cpp'), '-o', str(binary)], check=True)
            subprocess.run([str(binary)], check=True)

if __name__ == '__main__':
    unittest.main()
