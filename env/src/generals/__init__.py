"""Small public API; optional backends and GUI load only when selected.

Reference rules, controllers and diagnostics use explicit submodule imports.
"""
from .api import make_env
from .protocol import Action, ActionKind, Direction, MapConfig, MoveMode
from .protocol.vector import BatchStep
from .vector import VecEnv

__version__ = "0.1.0"
__all__ = ["make_env", "VecEnv", "MapConfig", "Action", "ActionKind",
           "Direction", "MoveMode", "BatchStep"]
