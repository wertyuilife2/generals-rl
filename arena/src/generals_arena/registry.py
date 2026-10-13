"""Lazy factory loading; every implementation obeys the environment protocol."""
from importlib import import_module

from generals.protocol.controller import Controller
from .entrants import verify_artifacts


def load_controller(entrant, rules):
    verify_artifacts(entrant)
    module, name = entrant["factory"].split(":")
    factory = getattr(import_module(module), name, None)
    if not callable(factory):
        raise ValueError(f"controller factory is not callable: {entrant['factory']}")
    controller = factory(player_id=0, seed=0, **entrant["factory_kwargs"])
    if not isinstance(controller, Controller) or not all(
            callable(getattr(controller, method, None)) for method in ("reset", "act", "on_result")):
        raise TypeError(f"{entrant['name']} must implement reset, act, and on_result")
    validate = getattr(controller, "validate_rules", None)
    if validate is not None:
        validate(rules.map, rules.visibility)
    verify_artifacts(entrant)
    return controller
