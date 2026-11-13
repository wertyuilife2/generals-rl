"""Local own-general relative coordinates, missing markers and dtype."""
import unittest

import torch

from generals_rl.policy.layout import own_king_coordinates


class OwnKingGeometryTests(unittest.TestCase):
    def test_relative_xy_local_frame_guards_and_dtype(self):
        for dtype in (torch.float32, torch.float64):
            with self.subTest(dtype=dtype):
                maps = torch.zeros(2, 13, 9, 9, dtype=dtype)
                for lane, (row, column) in enumerate(((7, 3), (1, 6))):
                    maps[lane, 4:6, row, column] = 1
                    # A visible enemy general and a self-owned plain cell are
                    # insufficient individually: both markers must intersect.
                    maps[lane, 4, 0, 8] = 1
                    maps[lane, 5, 8, 0] = 1
                actual = own_king_coordinates(maps)
                self.assertEqual((actual.dtype, actual.device), (maps.dtype, maps.device))
                self.assertEqual(tuple(actual.shape), (2, 2, 9, 9))
                rows, columns = torch.meshgrid(torch.arange(9, dtype=dtype),
                                               torch.arange(9, dtype=dtype), indexing='ij')
                expected = torch.stack([torch.stack(((columns-column)/8, (rows-row)/8))
                                        for row, column in ((7, 3), (1, 6))])
                torch.testing.assert_close(actual[:, -2:], expected, rtol=0, atol=0)
                changed = maps.clone()
                changed[:, 4] = 0
                self.assertTrue(bool(own_king_coordinates(changed).eq(0).all()))
                changed = maps.clone()
                changed[:, 4:6, 4, 4] = 1
                self.assertTrue(bool(own_king_coordinates(changed)[:, -2:].eq(0).all()))
