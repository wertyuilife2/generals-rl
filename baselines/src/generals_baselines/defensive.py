"""DefensiveAI configuration; implementation lives in policy.cpp."""
from .base import BaseAI


class DefensiveAI(BaseAI):
    kind = 2
    weights = {'fortify': 1.5, 'gather': 1., 'expand': .3}
    temperature = .7
    path_lifetime = 15
