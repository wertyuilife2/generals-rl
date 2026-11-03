"""Hand-written Generals strategies, independent of training and the GUI."""
from .aggressive import AggressiveAI
from .defensive import DefensiveAI
from .expansion import ExpansionAI
from .random_policy import RandomAI
from .registry import AI_KINDS, create_controller, make_controllers

__all__ = ["AggressiveAI", "DefensiveAI", "ExpansionAI", "RandomAI",
           "AI_KINDS", "create_controller", "make_controllers"]
