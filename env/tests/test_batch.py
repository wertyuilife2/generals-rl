"""Cold state conversion must validate shape and preserve independent ownership."""
import unittest

import numpy as np

from generals.core.batch import generate_states, pack_states, unpack_states
from generals.protocol import MapConfig, STATE_DTYPES


class BatchConversionTests(unittest.TestCase):
    def test_round_trip_is_exact_and_does_not_alias_inputs_or_siblings(self):
        states = generate_states(MapConfig(15, 15, 4), 2, seed=91)
        arrays, width, height = pack_states(iter(states))
        restored = unpack_states(arrays, width=width, height=height, validate=True)
        repacked, _, _ = pack_states(restored)
        for name in arrays:
            np.testing.assert_array_equal(arrays[name], repacked[name], err_msg=name)
            self.assertEqual(arrays[name].dtype, np.dtype(STATE_DTYPES[name]))
        original = arrays['army'].copy()
        restored[0].army.flat[0] += 17
        states[1].army.flat[0] += 9
        np.testing.assert_array_equal(arrays['army'], original)
        np.testing.assert_array_equal(restored[1].army.ravel(), original[1])
        self.assertFalse(np.shares_memory(restored[0].layout.terrain, arrays['terrain']))

    def test_invalid_or_mixed_states_are_rejected_before_device_upload(self):
        valid = generate_states(MapConfig(15, 15, 2), 1)[0]
        for others in ([], [valid, generate_states(MapConfig(17, 15, 2))[0]],
                       [valid, generate_states(MapConfig(15, 15, 4))[0]]):
            with self.subTest(count=len(others)), self.assertRaises(ValueError):
                pack_states(others)
        valid.army.flat[0] = -1
        with self.assertRaises(ValueError):
            pack_states([valid])

    def test_batch_extension_preserves_lane_seeds_and_rejects_invalid_seeds(self):
        config = MapConfig(15, 15, 4)
        small, _, _ = pack_states(generate_states(config, 2, seed=73))
        large, _, _ = pack_states(generate_states(config, 5, seed=73))
        for name in small:
            np.testing.assert_array_equal(small[name], large[name][:2])
        for options in ({'seed': -1}, {'seed': 1.5}, {'batch_size': 0}):
            with self.subTest(options=options), self.assertRaises((TypeError, ValueError)):
                generate_states(config, **options)


if __name__ == '__main__':
    unittest.main()
