"""ExpansionAI configuration; implementation lives in policy.cpp."""
from .base import BaseAI


class ExpansionAI(BaseAI):
    kind = 1
    weights = {'expand': 1.5, 'capture': 1.2, 'gather': .5}
    temperature = .6
    path_lifetime = 25
