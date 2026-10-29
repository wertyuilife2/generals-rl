"""AggressiveAI configuration; implementation lives in policy.cpp."""
from .base import BaseAI


class AggressiveAI(BaseAI):
    kind = 0
    weights = {'merge': .5, 'explore': .8, 'attack': 1.5}
    temperature = .5
    path_lifetime = 20
