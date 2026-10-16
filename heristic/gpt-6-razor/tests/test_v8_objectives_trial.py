"""Archived common-terminal trial; not the active policy."""
import unittest,json,gzip
from pathlib import Path
import test_v6
class RazorObjectivesTrial(unittest.TestCase):
    version="experiments/95f9311f025b8b1e2fc51daf16d8990bea2fe962caeb5ddaa3f87a77b9856a93"
    run_history=test_v6.RazorV6Regressions.run_history
    def test_does_not_relaunch_an_unsupported_general_capture(self):
        fixture = json.loads(gzip.decompress(
            Path(__file__).with_name('unfinished_defense_history.json.gz').read_bytes()))
        obs, action = self.run_history(fixture['observations'], fixture['seat'], fixture['seed'])
        # The nearby city can intercept the route even though it lacks an
        # enemy-owned road to the general. No alternative move is executed.
        if int(action.kind):
            target = action.source + (-obs.width, 1, obs.width, -1)[action.direction]
            self.assertNotEqual((action.source, target), (238, 237))


if __name__=="__main__":
    unittest.main()
