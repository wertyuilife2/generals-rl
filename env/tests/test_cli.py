import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


class CLITests(unittest.TestCase):
    def command(self, *args):
        return subprocess.run([sys.executable, '-m', 'generals_env', *args],
                              capture_output=True, text=True, timeout=30)

    def test_import_and_headless_without_tk_torch(self):
        code = "import sys; import generals_env; import generals_env.cli; assert 'tkinter' not in sys.modules; assert 'torch' not in sys.modules; assert 'generals_rl' not in sys.modules"
        result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        result = self.command('simulate', '--players', '2', '--max-ticks', '20', '--size', '15')
        self.assertEqual(result.returncode, 0, result.stderr)
        data = json.loads(result.stdout)
        self.assertEqual(data['total_ticks'], 20)
        self.assertEqual(data['truncated'], 1)

    def test_bad_configuration_is_clear(self):
        for args in [('simulate', '--players', '9'), ('play', '--human-seat', '4'),
                     ('simulate', '--games', '0'), ('simulate', '--max-ticks', '0'),
                     ('simulate', '--ais', 'random')]:
            with self.subTest(args=args):
                result = self.command(*args)
                self.assertEqual(result.returncode, 2)
                self.assertIn('error:', result.stderr)

    def test_cli_record_replay_and_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            trace, summary = Path(tmp)/'game.jsonl', Path(tmp)/'summary.json'
            result = self.command('simulate', '--players', '2', '--max-ticks', '10',
                                  '--record', str(trace), '--output', str(summary))
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads(summary.read_text())['total_ticks'], 10)
            checked = self.command('replay', str(trace))
            self.assertEqual(checked.returncode, 0, checked.stderr)
            self.assertTrue(json.loads(checked.stdout)['verified'])


if __name__ == '__main__': unittest.main()
