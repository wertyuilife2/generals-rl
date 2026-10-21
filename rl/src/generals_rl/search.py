"""Algorithm-independent CPU branching at the pre-growth joint-tick boundary.

The game snapshot is privileged simulator input. A deployed policy must receive
only its permitted observation from the MDP observation encoder; passing this
full state to a policy in local visibility would reveal hidden information.
Local visibility search additionally needs a belief/determinization model and
observation history, neither of which is provided by this simulator.

This module intentionally implements no MCTS selection, backup, learned model,
reward, opponent, or autoreset. Rules are delegated to ``CoreEngine.step``.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral
from typing import Protocol, TypeVar, runtime_checkable

from generals.core.engine import CoreEngine
from generals.core.state import Action, GameState, TickResult, compute_stats


ObservationT = TypeVar("ObservationT", contravariant=True)


@runtime_checkable
class PolicyValue(Protocol[ObservationT]):
    """A future search network, independent of rsl_rl and its action distribution.

    Priors map legal encoded actions to nonnegative probabilities summing to
    one. The scalar value is from the observation player's perspective; the
    eventual search implementation chooses its reward scale and backup rules.
    An adapter may use the same MDP encoder/mask as the training environment.
    """

    def policy_value(self, observation: ObservationT) -> tuple[Mapping[int, float], float]: ...


@dataclass(frozen=True)
class JointTickInfo:
    """Independent reference-rule result and episode status for one branch."""

    result: TickResult
    terminated: bool
    truncated: bool
    advanced: bool

    @property
    def done(self) -> bool:
        return self.terminated or self.truncated


class JointTickSimulator:
    """Copy and advance a complete preselected joint action using CPU rules.

    ``episode_ticks`` is the same absolute tick limit as CPUEnv; zero disables
    it. Natural termination takes precedence over truncation. Finished branches
    remain frozen, and every call returns independent dynamic storage.

    Only whole-tick snapshots are supported. A GameState copied while a
    CoreEngine transaction is active does not contain its cursor or phase and
    must not be supplied here. Each call selects all player actions before
    growth, then the engine executes them in player-slot order. This is the
    vector training environment's timing, rather than Runner's per-turn timing.

    Encoded inputs must be nonnegative integers, or canonical Action objects.
    Negative raw native action codes are malformed search input and rejected.
    Dead player slots are ignored, matching the native vector environments.
    Numeric overflow raises, rather than returning the native errors buffer.
    """

    def __init__(self, *, episode_ticks: int = 0):
        if isinstance(episode_ticks, bool) or not isinstance(episode_ticks, Integral):
            raise TypeError("episode_ticks must be an integer")
        if not 0 <= episode_ticks <= 2**63 - 1:
            raise ValueError("episode_ticks must be in the int64 nonnegative range")
        self.episode_ticks = int(episode_ticks)

    @staticmethod
    def snapshot(state: GameState) -> GameState:
        """Detach all mutable game arrays while sharing immutable map topology."""
        if not isinstance(state, GameState):
            raise TypeError("state must be GameState")
        state.validate()
        return state.copy()

    def advance(
        self, state: GameState, joint_actions: Sequence[int | Action]
    ) -> tuple[GameState, JointTickInfo]:
        """Return an isolated child and its result without mutating the parent.

        A one-dimensional NumPy integer array is also accepted as a player-slot
        sequence, although ndarray does not register with collections.Sequence.
        """
        child = self.snapshot(state)
        try:
            action_count = len(joint_actions)
        except TypeError as exc:
            raise TypeError("joint_actions must be indexed by player slot") from exc
        if action_count != child.player_count:
            raise ValueError("provide exactly one action per player slot")

        terminated = child.terminated
        truncated = not terminated and self.episode_ticks > 0 and child.tick >= self.episode_ticks
        if terminated or truncated:
            result = TickResult(child.tick, (), compute_stats(child), terminated, child.winner_id)
            return child, JointTickInfo(result, terminated, truncated, advanced=False)

        actions = []
        for player_id, encoded in enumerate(joint_actions):
            if not child.alive[player_id]:
                actions.append(Action.wait())
                continue
            if isinstance(encoded, Action):
                action = encoded
            else:
                if isinstance(encoded, Integral) and not isinstance(encoded, bool) and encoded > 2**63 - 1:
                    raise ValueError("encoded actions must fit int64")
                action = Action.decode(encoded)
            actions.append(action)
        result = CoreEngine(child).step(actions)
        truncated = not result.terminated and self.episode_ticks > 0 and child.tick >= self.episode_ticks
        return child, JointTickInfo(result, result.terminated, truncated, advanced=True)
