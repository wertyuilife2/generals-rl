"""Parameters for standard random maps (generals.io v31.4.3 defaults).

Density/size values are the official 0..1 sliders, not tile percentages.
Only ordinary, non-team maps are covered; special terrain/modifiers are not
enabled by the official defaults. MapConfig without these options retains the
fixed-size training generator and its historical seed behavior.
"""
from dataclasses import dataclass
from math import floor, isfinite
from numbers import Integral, Real


@dataclass(frozen=True)
class MapGenerationConfig:
    width: float = 0.5
    height: float = 0.5
    mountain_density: float = 0.5
    city_density: float = 0.5
    spawn_fairness: float = 0.5
    city_fairness: float = 0.5
    min_general_spacing: int = 9
    city_army_min: int = 40
    city_army_max: int = 50
    close_city_distance: int = 6
    require_close_city: bool = True
    max_attempts: int = 1000
    spawn_attempts: int = 1000
    city_attempts: int = 8

    def __post_init__(self):
        for name in ('width', 'height', 'mountain_density', 'city_density',
                     'spawn_fairness', 'city_fairness'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Real):
                raise TypeError(f'{name} must be a real number')
            if not isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f'{name} must be between 0 and 1')
            object.__setattr__(self, name, float(value))
        for name in ('min_general_spacing', 'city_army_min', 'city_army_max',
                     'close_city_distance', 'max_attempts', 'spawn_attempts', 'city_attempts'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise TypeError(f'{name} must be an integer')
            minimum = 1 if name.endswith('attempts') else 0
            if value < minimum:
                raise ValueError(f'{name} must be at least {minimum}')
            object.__setattr__(self, name, int(value))
        if self.city_army_max < self.city_army_min or self.city_army_max > 2**31 - 1:
            raise ValueError('city army bounds must satisfy 0 <= min <= max <= 2**31-1')
        if not isinstance(self.require_close_city, bool):
            raise TypeError('require_close_city must be a boolean')

    def _side(self, players, slider, jitter):
        base = 16 + min(8, players) + max(0, (players - 6) / 2)
        adjustment = ((slider - .5) * (3 + players / 2) if slider < .5 else
                      2 * (slider - .5) * (2 + players))
        return floor(base + adjustment + jitter * (4 + players / 2) + .5)

    def side_range(self, players, axis='width'):
        slider = getattr(self, axis)
        return self._side(players, slider, 0), self._side(players, slider, 1)

    def sample_size(self, players, rng):
        return (self._side(players, self.width, float(rng.random()) ** 2),
                self._side(players, self.height, float(rng.random()) ** 2))

    def spawn_spacing(self, players):
        return max(self.min_general_spacing, self.min_general_spacing * (1.8 - .1 * players))
