"""One ordered lifecycle for batch jobs and persistent isolated matches."""
from copy import deepcopy
from dataclasses import asdict
import threading

import pytest

from generals import Action, MapConfig
from generals_arena.policies.entries import prepare_entrant
from generals_arena.matches.runner import MatchRunner
from policy_test_utils import make_bundle


def plans_for(entry, *, width=9, count=3):
    return [{"game_id": str(index), "entrant_ids": [entry["entrant_id"]] * 2,
             "instance_ids": [0, 1], "controller_seeds": [31, 47], "map_seed": 17 + index,
             "map": asdict(MapConfig(width, 9, 2, 0, 0))} for index in range(count)]


def test_ordered_callbacks_homogeneous_chunks_and_controller_ownership(monkeypatch):
    entry = prepare_entrant({"name": "random", "kind": "baseline", "policy": "random"})
    plans = plans_for(entry)
    plans[1]["instance_ids"] = [1, 0]  # Rotation keeps the same instance assignment.
    plans[2]["map"] = asdict(MapConfig(10, 9, 2, 0, 0))
    calls, callbacks, closed = [], [], []

    class Cached:
        def close(self):
            closed.append(True)

    def execute(entries, chunk, **options):
        calls.append([plan["game_id"] for plan in chunk])
        options["runtime_cache"].setdefault((entry["entrant_id"], 0), Cached())
        return [(dict(plan), b"replay") for plan in chunk], {"step_seconds": 2.}

    monkeypatch.setattr("generals_arena.matches.batch.execute_batch", execute)
    with MatchRunner({entry["entrant_id"]: entry}, max_ticks=3, batch_size=0) as runner:
        results, timing = runner.run(plans, on_result=lambda value, blob: callbacks.append((value, blob)))
        assert not closed
    assert calls == [["0", "1"], ["2"]]
    assert results == [pair[0] for pair in callbacks] == plans
    assert all(blob == b"replay" for _, blob in callbacks)
    assert timing == {"step_seconds": 4.}
    assert closed == [True]


def test_whole_schedule_capability_check_precedes_any_execution(tmp_path, monkeypatch):
    bundle = make_bundle(tmp_path / "bundle")
    entry = prepare_entrant({"name": "package", "kind": "package", "bundle": str(bundle)})
    plans = plans_for(entry, count=2)
    plans[1]["map"] = asdict(MapConfig(10, 9, 2, 0, 0))

    def unexpected(*args, **kwargs):
        pytest.fail("the incompatible suffix must be rejected before the first batch")

    monkeypatch.setattr("generals_arena.matches.batch.execute_batch", unexpected)
    callbacks = []
    with MatchRunner({entry["entrant_id"]: entry}, max_ticks=2, batch_size=1) as runner:
        with pytest.raises(ValueError, match="does not support"):
            runner.run(plans, on_result=lambda *args: callbacks.append(args))
    assert callbacks == []


def test_registered_implementation_change_is_rejected_before_batch(monkeypatch):
    entry = prepare_entrant({"name": "random", "kind": "baseline", "policy": "random"})
    monkeypatch.setattr("generals_arena.matches.runner.source_digest", lambda package: "changed")
    with MatchRunner({entry["entrant_id"]: entry}, max_ticks=2) as runner:
        with pytest.raises(ValueError, match="registered source changed"):
            runner.run(plans_for(entry))


@pytest.mark.parametrize("field,value", [
    ("instance_ids", [0, 0]), ("instance_ids", [0]),
    ("controller_seeds", [17]), ("controller_seeds", [17, -1]),
    ("controller_seeds", [17, 2**64]),
    ("map_seed", -1), ("map_seed", 1.5), ("map_seed", True),
])
def test_invalid_schedule_suffix_is_rejected_before_first_callback(field, value):
    entry = prepare_entrant({"name": "random", "kind": "baseline", "policy": "random"})
    plans = plans_for(entry)
    plans[-1][field] = value
    callbacks = []
    with MatchRunner({entry["entrant_id"]: entry}, max_ticks=2, batch_size=1) as runner:
        with pytest.raises(ValueError):
            runner.run(plans, on_result=lambda *args: callbacks.append(args))
    assert callbacks == []


def test_isolated_stop_keeps_current_plan_uncommitted_and_closes_controllers(monkeypatch):
    entry = prepare_entrant({"name": "random", "kind": "baseline", "policy": "random"})
    plans = plans_for(entry, count=1)
    original = deepcopy(plans)
    stop, closed, decisions = threading.Event(), [], []

    class Controller:
        def reset(self, player_id, seed):
            pass

        def act(self, observation):
            decisions.append(observation.tick)
            stop.set()
            return Action.wait()

        def on_result(self, result):
            pass

        def close(self):
            closed.append(True)

    monkeypatch.setattr("generals_arena.matches.runner.build_controllers",
                        lambda *args: [Controller(), Controller()])
    callbacks = []
    with MatchRunner({entry["entrant_id"]: entry}, max_ticks=4, execution="isolated") as runner:
        results, _ = runner.run(plans, on_result=lambda *args: callbacks.append(args), stop=stop)
    assert not results and not callbacks
    assert decisions == [0, 0] and closed == [True, True]
    assert plans == original


def test_callback_failure_closes_batch_runtime(monkeypatch):
    entry = prepare_entrant({"name": "random", "kind": "baseline", "policy": "random"})
    closed = []

    class Cached:
        def close(self):
            closed.append(True)

    def execute(entries, chunk, **options):
        options["runtime_cache"][(entry["entrant_id"], 0)] = Cached()
        return [(dict(plan), None) for plan in chunk], {}

    def reject(*args):
        raise RuntimeError("commit failed")

    monkeypatch.setattr("generals_arena.matches.batch.execute_batch", execute)
    with pytest.raises(RuntimeError, match="commit failed"):
        with MatchRunner({entry["entrant_id"]: entry}, max_ticks=3) as runner:
            runner.run(plans_for(entry), on_result=reject)
    assert closed == [True]


@pytest.mark.parametrize("options", [
    {"backend": "invented"}, {"backend": "gpu", "device": "cpu"}, {"device": "other"},
    {"batch_size": -1}, {"batch_size": True}, {"cpu_threads": 0}, {"execution": "other"},
])
def test_invalid_execution_options_are_rejected(options):
    with pytest.raises(ValueError):
        MatchRunner({}, max_ticks=3, **options)
