"""Archived complete-defence trial; not the active policy."""
import unittest,json,gzip
from pathlib import Path
import test_v6
class RazorDefenseTrial(unittest.TestCase):
    version="experiments/58d2ab5a3e045d40329a8fa3d79afd8526bf53c26ebaf854ec0373c74782b8c1"
    run_history=test_v6.RazorV6Regressions.run_history
    def test_recalled_column_finishes_its_defense_before_relaunch(self):
        fixture = json.loads(gzip.decompress(
            Path(__file__).with_name('unfinished_defense_history.json.gz').read_bytes()))
        obs, action = self.run_history(fixture['observations'], fixture['seat'], fixture['seed'])
        if int(action.kind):
            target = action.source + (-obs.width, 1, obs.width, -1)[action.direction]
            self.assertNotEqual((action.source, target), (238, 237))


if __name__=="__main__":
    unittest.main()
