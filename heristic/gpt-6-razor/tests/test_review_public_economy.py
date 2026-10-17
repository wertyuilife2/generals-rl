"""Keep missing frames and player identity out of economic conclusions."""
from types import SimpleNamespace
import unittest

import numpy as np

from review_public_economy import public_economy


def frame(tick, player, own, enemy):
    armies = [own, enemy] if player == 0 else [enemy, own]
    obs = SimpleNamespace(
        tick=tick, player_id=player, size=2, general_pos=0,
        alive=np.array([True, True]), visible=np.array([True, False]),
        owner=np.array([player, -3]), army=np.array([3, 0]),
        structure=np.array([2, 255]),
        stats=SimpleNamespace(army=np.array(armies), territory=np.array([1, 1]),
                              cities=np.array([0, -1] if player == 0 else [-1, 0])),
    )
    return obs, []


class PublicEconomyReview(unittest.TestCase):
    def test_gap_sign_uses_the_recorded_seat(self):
        for player in [0, 1]:
            result = public_economy([frame(125, player, 100, 90),
                                     frame(126, player, 101, 92)])
            change = result['nonzero_gap_changes'][0]
            self.assertEqual(change['change'], 1)
            self.assertEqual(change['gap_after'], -9)
            self.assertEqual(change['phase'], 'ordinary_growth_boundary')
            self.assertIsNone(result['timeline'][-1]['largest_visible_enemy'])

    def test_skipped_frames_do_not_become_one_tick_income(self):
        result = public_economy([frame(148, 0, 100, 100),
                                 frame(150, 0, 130, 145)])
        self.assertEqual(result['nonzero_gap_changes'][0]['phase'], 'gap_in_record')
        self.assertFalse(result['gap_change_histograms']['land_growth_boundary'])

    def test_preserves_one_time_spending_and_zero_changes(self):
        result = public_economy([frame(121, 1, 112, 120),
                                 frame(122, 1, 113, 76),
                                 frame(123, 1, 113, 76)])
        self.assertEqual(result['nonzero_gap_changes'][0]['change'], -45)
        self.assertEqual(result['gap_change_histograms']['between_growth_boundaries'], {0: 1})


if __name__ == '__main__':
    unittest.main()
