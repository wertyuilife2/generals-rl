"""A native WAIT reference, registered through the ordinary policy pipeline."""
from pathlib import Path


def wait_spec(store):
    from .research import snapshot_policy
    return {**snapshot_policy(store, Path(__file__).with_name('sdk') / 'wait.cpp'), 'name': 'WAIT'}


def register_wait_reference(store):
    entrant = store.register(wait_spec(store))
    store.set_rating_reference(entrant['entrant_id'])
    return entrant
