"""Archived resource-claim trial regressions, not the active policy."""
import unittest,json,gzip
from pathlib import Path
import test_v6

class RazorCommandTrial(unittest.TestCase):
    version="experiments/c348e64f58be13e1fc31f4bd410476c5d64b7727accef4b853f5941379293b0a"
    run_history=test_v6.RazorV6Regressions.run_history

    def test_home_return_does_not_immediately_relaunch_the_same_column(self):
        fixture = json.loads(gzip.decompress(
            Path(__file__).with_name('command_home_role_history.json.gz').read_bytes()))
        obs, action = self.run_history(fixture['observations'], fixture['seat'], fixture['seed'])
        if int(action.kind):
            target = action.source + (-obs.width, 1, obs.width, -1)[action.direction]
            self.assertNotEqual((action.source, target), (241, 240))

    def test_directs_a_recorded_column_toward_an_information_goal(self):
        fixture = json.loads(gzip.decompress(
            Path(__file__).with_name('queuet_low_information_raid.json.gz').read_bytes()))
        obs, action = self.run_history(fixture['observations'], fixture['seat'], fixture['seed'])
        # Only the legal history through tick158 is supplied. Both empty
        # landings can open a route north; the historical detour turned south
        # into enemy garrisons. This is a decision regression, not an alternate
        # game or proof that the opponent's general would be discovered.
        self.assertNotEqual(int(action.kind), 0)
        target = action.source + (-obs.width, 1, obs.width, -1)[action.direction]
        self.assertEqual(action.source, 209)
        self.assertIn(target, (189, 210))


if __name__=="__main__":
    unittest.main()
