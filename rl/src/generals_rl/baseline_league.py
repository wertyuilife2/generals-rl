"""Opt-in sparse original aggressive league over historical self-play."""
from dataclasses import asdict, dataclass
from time import perf_counter

import numpy as np
import torch

from .league_bridge import OriginalBaselineBridge
from .opponents import SelfPlayOpponent


@dataclass(frozen=True)
class BaselineLeagueConfig:
    baseline_envs: int = 0
    seed: int = 19000042
    native_candidates: bool = False

    def __post_init__(self):
        if (isinstance(self.baseline_envs, bool) or not isinstance(self.baseline_envs, int)
                or not 0 <= self.baseline_envs <= 256):
            raise ValueError("baseline_league.baseline_envs must be an integer in [0,256]")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("baseline_league.seed must be a nonnegative integer")
        if not isinstance(self.native_candidates, bool):
            raise TypeError("baseline_league.native_candidates must be boolean")

    def validate_task(self, task):
        if not self.baseline_envs:
            return
        if (task.opponent != "self_play" or (task.players, task.width, task.height) != (2, 9, 9)):
            raise ValueError("baseline league requires self_play on a 9x9 two-player task")
        if self.baseline_envs * 50 > task.num_envs:
            raise ValueError("baseline league must retain at least 98% historical self-play lanes")


def make_league_opponent(task, settings):
    """None preserves the exact original task opponent construction path."""
    settings.validate_task(task)
    if not settings.baseline_envs:
        return None
    historical = SelfPlayOpponent(history_length=task.history_length, seed=task.seed)
    return BaselineLeagueOpponent(historical, settings, num_envs=task.num_envs)


class BaselineLeagueOpponent:
    """Delegate all historical inference; overlay fixed original-AI lanes only.

    No old-pool mutation, distribution/storage/vendor changes or hidden policy
    input. Decisions and result callbacks are owned by the original controller
    bridge. Timing values are host wall time, not isolated CUDA kernel timings.
    """
    def __init__(self, historical, settings, *, num_envs):
        if settings.baseline_envs * 50 > num_envs:
            raise ValueError("baseline league must retain at least 98% historical self-play lanes")
        self.historical, self.settings, self.num_envs = historical, settings, num_envs
        lanes = np.random.default_rng(settings.seed).choice(
            num_envs, settings.baseline_envs, replace=False).tolist()
        self.bridge = OriginalBaselineBridge(sorted(lanes), kind="aggressive", seed=settings.seed)
        self.native = None
        if settings.native_candidates:
            # Optional dependency is imported/compiled only when explicitly on.
            from .league_native_candidates import NativeCandidateBackend
            self.native = NativeCandidateBackend()
        self.lane_mask = self.totals = self._logged_totals = None
        self.joint_ticks = self._logged_ticks = 0
        self.times = {key: 0.0 for key in ("historical_actions", "baseline_actions", "baseline_results")}
        self._logged_times = self.times.copy()

    @property
    def episode_assignments(self):
        return self.historical.episode_assignments

    def attach(self, *args, **kwargs):
        return self.historical.attach(*args, **kwargs)

    def snapshot_counts(self):
        return self.historical.snapshot_counts()

    def on_reset(self, batch, learner_ids, action_space, encoder, mask=None):
        self.historical.on_reset(batch, learner_ids, action_space, encoder, mask=mask)
        if self.lane_mask is None:
            self.lane_mask = torch.zeros(batch.num_envs, dtype=torch.bool, device=batch.device)
            self.lane_mask[list(self.bridge.lanes)] = True
            self.totals = torch.zeros(2, 3, dtype=torch.long, device=batch.device)
            self._logged_totals = self.totals.cpu().clone()
            self.bridge.reset(learner_ids)
        else:
            self.bridge.reset(learner_ids, mask)
        if self.native is not None:
            # Instance-only shim; the factory and evaluator remain original.
            from types import MethodType
            for controller in self.bridge.controllers:
                if "_candidates" not in controller.__dict__:
                    controller._candidates = MethodType(self.native.method(), controller)

    def on_step(self, *args, **kwargs):
        return self.historical.on_step(*args, **kwargs)

    def joint_actions(self, batch, learner_ids, action_space, encoder):
        start = perf_counter()
        joint = self.historical.joint_actions(batch, learner_ids, action_space, encoder)
        self.times["historical_actions"] += perf_counter() - start
        start = perf_counter()
        self.bridge.patch_actions(joint, batch, learner_ids)
        self.times["baseline_actions"] += perf_counter() - start
        self.joint_ticks += 1
        return joint

    def on_result(self, result, submitted, batch, learner_ids):
        """Called after native step/check_errors, before any task reset."""
        start = perf_counter()
        self.bridge.on_result(result, submitted)
        self.times["baseline_results"] += perf_counter() - start
        rows = torch.arange(batch.num_envs, device=batch.device)
        raw_done = torch.as_tensor(result.done, device=batch.device)
        alive = batch.state["alive"][rows, learner_ids]
        terminated = raw_done[:, 0] | ~alive
        truncated = raw_done[:, 1] & ~terminated
        done = terminated | truncated
        wins = batch.state["winner"] == learner_ids
        for source, lanes in enumerate((self.lane_mask, ~self.lane_mask)):
            ended = done & lanes
            self.totals[source] += torch.stack((ended.sum(), (ended & wins).sum(),
                                                (ended & truncated).sum()))

    def take_log_metrics(self):
        # Tiny counters cross the host boundary once per PPO update, not all
        # 16k state/moves or per-tick episode vectors.
        current = self.totals.detach().cpu()
        delta = current - self._logged_totals
        self._logged_totals = current.clone()
        ticks = self.joint_ticks - self._logged_ticks
        self._logged_ticks = self.joint_ticks
        result = {"League/baseline_envs": self.settings.baseline_envs,
                  "League/baseline_fraction": self.settings.baseline_envs / self.num_envs}
        for index, (source, count) in enumerate((
                ("aggressive", self.settings.baseline_envs),
                ("historical", self.num_envs - self.settings.baseline_envs))):
            episodes, wins, truncated = delta[index].tolist()
            result[f"League/{source}_decisions"] = count * ticks
            result[f"League/{source}_episodes"] = episodes
            if episodes:
                result[f"League/{source}_win_rate"] = wins / episodes
                result[f"League/{source}_truncated_rate"] = truncated / episodes
        for key, total in self.times.items():
            result[f"League/{key}_wall_seconds"] = total - self._logged_times[key]
            self._logged_times[key] = total
        if self.native is not None:
            result["League/native_candidate_calls"] = self.native.native_calls
            result["League/native_fallback_calls"] = self.native.fallback_calls
        return result

    def describe(self):
        rows = self.totals.detach().cpu().tolist()
        return {"config": asdict(self.settings), "fixed_lanes": list(self.bridge.lanes),
                "decision_timing": "joint_pre_growth_then_absolute_seat_order",
                "baseline_implementation": "original generals.controllers.make_controllers aggressive",
                "all_historical_inference_retained": True,
                "controller_episodes": self.bridge.episode.tolist(),
                "native_candidate_calls": self.native.native_calls if self.native else 0,
                "native_fallback_calls": self.native.fallback_calls if self.native else 0,
                "source_totals": {name: dict(decisions=self.joint_ticks * count,
                    completed_episodes=values[0], wins=values[1], truncated=values[2])
                    for (name, count), values in zip((
                        ("aggressive", self.settings.baseline_envs),
                        ("historical", self.num_envs - self.settings.baseline_envs)), rows)},
                "decision_wall_seconds": self.times.copy()}
