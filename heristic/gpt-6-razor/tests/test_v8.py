"""V8 preserves the observed tactical safety requirements of earlier releases."""
import unittest
import subprocess
import tempfile
from pathlib import Path
import test_v6


class RazorV8Regressions(test_v6.RazorV6Regressions):
    version = 'v8'

    def test_supply_program_conserves_army_and_does_not_reward_pure_relocation(self):
        # Compile against v8 rather than accidentally checking the frozen v6.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(__file__).resolve().parent
            source = Path(tmp) / 'supply.cpp'
            old = (root / 'test_v6_supply.cpp').read_text()
            source.write_text(old.replace('../v6/policy.cpp', str(root.parent / 'v8/policy.cpp')))
            binary = Path(tmp) / 'supply'
            subprocess.run(['c++', '-std=c++17', '-O2', '-I',
                            str(test_v6.PROJECT / 'arena/src/generals_arena/sdk'),
                            str(source), '-o', str(binary)], check=True)
            subprocess.run([str(binary)], check=True)

    def test_production_retention_and_reconnaissance_routes(self):
        with tempfile.TemporaryDirectory() as tmp:
            binary = Path(tmp) / 'operations'
            subprocess.run(['c++', '-std=c++17', '-O2', '-I',
                            str(test_v6.PROJECT / 'arena/src/generals_arena/sdk'),
                            str(Path(__file__).with_name('test_v8_operations.cpp')),
                            '-o', str(binary)], check=True)
            subprocess.run([str(binary)], check=True)


if __name__ == '__main__':
    unittest.main()
