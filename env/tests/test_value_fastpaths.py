"""Value conversion fast paths retain public validation and snapshot isolation."""
from dataclasses import dataclass
import numpy as np
import pytest

from generals import Action
from generals.core.state import PlayerStats
from generals.protocol.actions import ActionKind, Direction, MoveMode


def test_decode_matches_validated_construction_and_retains_subclass_initialization():
    for code in [0, *range(1, 4096 * 8 + 1), 2**100]:
        if code:
            cell, rest = divmod(code - 1, 8)
            expected = Action(ActionKind.MOVE, cell, Direction(rest // 2), MoveMode(rest % 2))
        else:
            expected = Action(ActionKind.WAIT)
        actual = Action.decode(code)
        assert actual == expected and actual.encode() == code
        assert actual is not Action.decode(code)
    for bad in (True, False, 1., '1'):
        with pytest.raises(TypeError):
            Action.decode(bad)
    with pytest.raises(ValueError):
        Action.decode(-1)
    assert Action.decode(np.int64(17)) == Action.decode(17)
    @dataclass(frozen=True)
    class Custom(Action):
        marker: int = 3
    assert isinstance(Custom.wait(), Custom) and Custom.decode(9).marker == 3


@pytest.mark.parametrize('array', [np.arange(12, dtype=np.int64)[::2],
    np.arange(6, dtype=np.int32), np.arange(6, dtype='>i8'), np.arange(6, dtype=np.uint64)])
def test_statistics_copy_noncontiguous_and_other_integer_representations(array):
    expected = array.copy()
    stats = PlayerStats(array, array, array)
    masked = stats._for_player(0)
    array[:] = 0
    for value in (stats.army, stats.territory, stats.cities, masked.army, masked.territory):
        np.testing.assert_array_equal(value, expected)
        assert value.dtype == np.int64
        with pytest.raises(ValueError):
            value.flags.writeable = True
    np.testing.assert_array_equal(masked.cities, [expected[0], -1, -1, -1, -1, -1])
    with pytest.raises(ValueError):
        masked.cities.flags.writeable = True


def test_statistics_still_reject_float_and_integer_overflow():
    for value, error in [(np.zeros(2), TypeError), (np.array([2**63], np.uint64), ValueError)]:
        with pytest.raises(error):
            PlayerStats(value, value, value)
