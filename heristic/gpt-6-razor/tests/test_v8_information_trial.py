"""Regression for the archived information-pricing trial; not the active policy."""
import unittest
import gzip
import json
from pathlib import Path
import test_v6

class RazorInformationTrial(unittest.TestCase):
    version = "experiments/568981b061254ce23a12241af0377bc3b68c982a4fa33a96d9e6ed389667d336"
    run_history = test_v6.RazorV6Regressions.run_history

    def test_does_not_repeat_the_recorded_low_information_raid_detour(self):
        fixture = json.loads(gzip.decompress(
            Path(__file__).with_name('queuet_low_information_raid.json.gz').read_bytes()))
        obs, action = self.run_history(fixture['observations'], fixture['seat'], fixture['seed'])
        # At this decision the 47-troop column can explore through empty land.
        # The old action turned south into a winding chain of enemy two-troop
        # squares. Equal newly revealed information must not be worth more
        # merely because the landing square is enemy-owned. Later frames and
        # the eventual counterattack are deliberately not supplied.
        if int(action.kind):
            target = action.source + (-obs.width, 1, obs.width, -1)[action.direction]
            self.assertNotEqual((action.source, target), (209, 229))


if __name__ == "__main__":
    unittest.main()
