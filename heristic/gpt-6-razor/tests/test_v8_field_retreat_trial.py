"""Check move coverage of an archived field-tactics experiment."""
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parent
SDK = ROOT.parents[2] / 'arena/src/generals_arena/sdk'


class FieldRetreatTrial(unittest.TestCase):
    def test_principal_army_can_retreat_despite_negative_immediate_gain(self):
        with tempfile.TemporaryDirectory() as tmp:
            binary = Path(tmp) / 'field'
            subprocess.run(['c++', '-std=c++17', '-O2', '-I', str(SDK),
                            str(ROOT / 'test_v8_field_retreat_trial.cpp'),
                            '-o', str(binary)], check=True)
            subprocess.run([str(binary)], check=True)


if __name__ == '__main__':
    unittest.main()
