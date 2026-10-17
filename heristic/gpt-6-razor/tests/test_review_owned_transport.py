import unittest
from types import SimpleNamespace
import numpy as np
from review_owned_transport import owned_transport


def board(armies, enemy=None):
    owner = np.full((3, 3), -1, dtype=int)
    army = np.zeros((3, 3), dtype=int)
    for c, n in armies.items():
        owner.flat[c], army.flat[c] = 0, n
    for c, n in (enemy or {}).items():
        owner.flat[c], army.flat[c] = 1, n
    return SimpleNamespace(general_pos=0, player_id=0, width=3, height=3,
                           size=9, owner=owner, army=army)


class OwnedTransportTests(unittest.TestCase):
    def test_shared_corridor_consumes_orders_and_counts_each_donor_once(self):
        obs = board({0: 3, 1: 1, 2: 11, 4: 21})
        value = owned_transport(obs, (0, 1, 2, 3), minimum_cut_tiles=1)
        # One order cannot teleport either distant donor home. Two orders
        # can bring one; three can merge both through their shared corridor.
        self.assertEqual(value['quiet_tree_home_force_by_orders'],
                         {'0': 3, '1': 3, '2': 23, '3': 33})
        self.assertEqual(value['critical_owned_connections'], [
            {'cell': 1, 'guard': 1, 'owned_tiles_cut_off': 2,
             'spare_army_cut_off': 30}])

    def test_non_owned_corridor_does_not_supply_an_imaginary_defense(self):
        obs = board({0: 3, 2: 11, 4: 21}, enemy={1: 5})
        value = owned_transport(obs, (0, 4, 8), minimum_cut_tiles=1)
        self.assertEqual(value['home_connected_tiles'], 1)
        self.assertEqual(value['disconnected_spare_army'], 30)
        self.assertEqual(value['quiet_tree_home_force_by_orders'],
                         {'0': 3, '4': 3, '8': 3})
        obs.army.flat[1] = 1000000
        self.assertEqual(value, owned_transport(obs, (0, 4, 8), minimum_cut_tiles=1))

    def test_second_owned_route_removes_the_connection_failure(self):
        obs = board({0: 3, 1: 1, 2: 11, 3: 1, 4: 21, 5: 1})
        value = owned_transport(obs, (0, 4), minimum_cut_tiles=1)
        self.assertEqual(value['critical_owned_connections'], [])

    def test_lost_general_is_explicit(self):
        self.assertEqual(owned_transport(board({}, enemy={0: 5})),
                         {'status': 'general_not_owned'})


if __name__ == '__main__':
    unittest.main()
