"""Mechanics regressions for archived, unsuccessful province trials.

These checks are not strength evidence and do not promote their policies.
"""
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parent
SDK = ROOT.parents[2] / 'arena/src/generals_arena/sdk'


class ProvinceTrials(unittest.TestCase):
    def test_archived_mechanics(self):
        for case in ('province_trial', 'province_king_path', 'half_supply'):
            with self.subTest(case=case), tempfile.TemporaryDirectory() as tmp:
                binary = Path(tmp) / case
                subprocess.run(['c++', '-std=c++17', '-O2', '-I', str(SDK),
                                str(ROOT / f'test_v8_{case}.cpp'),
                                '-o', str(binary)], check=True)
                subprocess.run([str(binary)], check=True)


if __name__ == '__main__':
    unittest.main()
