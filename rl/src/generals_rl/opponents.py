"""Vector opponent interface, independent of CTS and partial/full critic views."""
from inspect import signature

import torch

from .policy.knowledge import KnowledgeState
from .policy.observations import ObservationHistory


class WaitOpponent:
    def act(self, batch, player_ids, action_space, encoder, generator):
        return torch.zeros(batch.num_envs, dtype=torch.long, device=batch.device)


class RandomOpponent:
    def act(self, batch, player_ids, action_space, encoder, generator):
        mask = action_space.mask(batch, player_ids)
        return torch.multinomial(mask.float(), 1, generator=generator).squeeze(1)


class CurrentSelfPlayOpponent:
    """Two-player self-play with the learner's live local actor.

    Each lane has one opposing seat, fixed until its episode resets. Keeping
    only that seat's history makes every decision one full N-row actor call.
    Knowledge storage can be shared with the task: each role updates only its
    own seat, so sharing never transfers facts between players.
    """

    def __init__(self, *, history_length=4, seed=42, deterministic=False,
                 persistent_knowledge=False):
        if not isinstance(persistent_knowledge, bool):
            raise TypeError("persistent_knowledge must be boolean")
        self.history = ObservationHistory(history_length)
        self.seed = seed
        self.deterministic = deterministic
        self.persistent_knowledge = persistent_knowledge
        self.knowledge = None
        self._shared_knowledge = False
        self.policy = None
        self.generator = None
        self.observations = None
        self._probability_options = {"is_teacher": False}

    def bind_knowledge(self, knowledge):
        """Share task-owned storage before the initial reset hook."""
        if self.history.frames is not None:
            raise RuntimeError("bind knowledge before initializing current self-play")
        if (knowledge is not None) != self.persistent_knowledge:
            raise ValueError("task and opponent persistent knowledge must agree")
        self.knowledge = knowledge
        self._shared_knowledge = True

    def attach_policy(self, policy, *, stochastic=True, seed=None):
        """Bind the loaded learner model after runner initialization."""
        if self.policy is not None:
            raise RuntimeError("current self-play policy is already attached")
        if self.generator is None:
            raise RuntimeError("initialize the task before attaching its current policy")
        if getattr(policy, "is_recurrent", False):
            raise ValueError("current self-play requires a nonrecurrent local actor")
        self.policy = policy
        self.deterministic = not stochastic
        if seed is not None:
            self.generator.manual_seed(seed)
        # Built-in action spaces guarantee a legal WAIT in each row. Defer the
        # selected-action check to one batched validation after sampling.
        if "validate_mask" in signature(policy.action_probabilities).parameters:
            self._probability_options["validate_mask"] = False

    def _cache_observations(self, batch, action_space):
        values = {"policy": self.history.frames.flatten(1),
                  "single_obs": self.history.frames[:, -1],
                  "action_mask": action_space.mask(batch, self._player_ids)}
        if self.knowledge is not None:
            values["knowledge"] = self.knowledge.encode(batch.state["tick"], self._player_ids).flatten(1)
        self.observations = values

    def on_reset(self, batch, learner_ids, action_space, encoder, mask=None):
        if batch.players != 2:
            raise ValueError("current self-play requires two players")
        if self.generator is None:
            self.generator = torch.Generator(device=batch.device).manual_seed(self.seed + 217645177)
            self._rows = torch.arange(batch.num_envs, device=batch.device)
            if self.persistent_knowledge and self.knowledge is None:
                self.knowledge = KnowledgeState(batch.num_envs, batch.width, batch.height,
                                                batch.players, batch.device)
        self._player_ids = 1 - learner_ids
        self.history.reset(encoder.encode(batch, self._player_ids), mask)
        if self.knowledge is not None:
            if not self._shared_knowledge:
                self.knowledge.reset(mask)
            self.knowledge.update(batch, self._player_ids, mask=mask)
        self._cache_observations(batch, action_space)

    def on_step(self, batch, learner_ids, action_space, encoder):
        self.history.push(encoder.encode(batch, self._player_ids))
        if self.knowledge is not None:
            self.knowledge.update(batch, self._player_ids)
        self._cache_observations(batch, action_space)

    @torch.no_grad()
    def joint_actions(self, batch, learner_ids, action_space, encoder):
        if self.policy is None:
            raise RuntimeError("attach the current learner policy before self-play stepping")
        from tensordict import TensorDict

        observations = TensorDict(self.observations, batch_size=[batch.num_envs], device=batch.device)
        had_distribution = hasattr(self.policy, "distribution")
        learner_distribution = getattr(self.policy, "distribution", None)
        try:
            if self.deterministic:
                actions = self.policy.act_inference(observations).reshape(-1)
            else:
                probabilities = self.policy.action_probabilities(observations, **self._probability_options)
                actions = torch.multinomial(probabilities, 1, generator=self.generator).squeeze(1)
        finally:
            # PPO still owns the learner's action log-probability/entropy state.
            if had_distribution:
                self.policy.distribution = learner_distribution
            elif hasattr(self.policy, "distribution"):
                del self.policy.distribution
        actions = action_space.validate(actions, self.observations["action_mask"])
        joint = torch.zeros((batch.num_envs, 2), dtype=torch.long, device=batch.device)
        joint[self._rows, self._player_ids] = actions
        return joint


class SelfPlayOpponent:
    """Episode-fixed opponents drawn from a frozen historical policy pool.

    Histories belong to absolute game seats, not to the learner's rotating
    role. Only local observations enter historical policies. The pool and its
    policy implementation are injected so this module does not depend on CTS.
    """

    def __init__(self, *, history_length=4, seed=42, deterministic=False,
                 persistent_knowledge=False):
        if not isinstance(persistent_knowledge, bool):
            raise TypeError("persistent_knowledge must be boolean")
        self.history_length = history_length
        self.seed = seed
        self.deterministic = deterministic
        self.pool = None
        self.histories = []
        self.persistent_knowledge = persistent_knowledge
        self.knowledge = None
        self.assignment_ids = None
        self.episode_assignments = 0
        self._snapshots = {}
        self._learner_ids = None
        self.generator = None

    def on_reset(self, batch, learner_ids, action_space, encoder, mask=None):
        """Reset finished lanes after the task rotates their learner seats."""
        if mask is None:
            mask = torch.ones(batch.num_envs, dtype=torch.bool, device=batch.device)
        if self.assignment_ids is None:
            self.assignment_ids = torch.full((batch.num_envs, batch.players), -1,
                                             dtype=torch.long, device=batch.device)
            self.histories = [ObservationHistory(self.history_length) for _ in range(batch.players)]
            if self.persistent_knowledge:
                self.knowledge = KnowledgeState(batch.num_envs, batch.width, batch.height,
                                                batch.players, batch.device)
            # Independent of both the learner's global PyTorch RNG and the
            # task's heuristic-opponent generator.
            self.generator = torch.Generator(device=batch.device).manual_seed(self.seed + 217645177)
        self._learner_ids = learner_ids.clone()
        if self.knowledge is not None:
            self.knowledge.reset(mask)
        for pid, history in enumerate(self.histories):
            seats = torch.full_like(learner_ids, pid)
            history.reset(encoder.encode(batch, seats), mask)
            if self.knowledge is not None:
                self.knowledge.update(batch, seats, mask=mask)
        if self.pool is not None:
            self._assign(mask)

    def attach(self, pool, *, stochastic=True, seed=None):
        """Bind a populated pool before the first rollout."""
        if self.pool is not None:
            raise RuntimeError("self-play pool is already attached")
        if self.assignment_ids is None:
            raise RuntimeError("initialize the task before attaching its self-play pool")
        if not pool.ids:
            raise ValueError("self-play requires at least one historical policy snapshot")
        self.pool = pool
        self.deterministic = not stochastic
        if seed is not None:
            self.generator.manual_seed(seed)
        self._assign(torch.ones(len(self._learner_ids), dtype=torch.bool,
                                device=self.assignment_ids.device))

    def _assign(self, reset_mask):
        seats = torch.arange(self.assignment_ids.shape[1], device=self.assignment_ids.device)
        active = reset_mask[:, None] & (seats[None] != self._learner_ids[:, None])
        self.assignment_ids[reset_mask] = -1
        snapshots = self.pool.sample(int(active.sum()), generator=self.generator)
        if len(snapshots) != int(active.sum()):
            raise RuntimeError("historical pool returned the wrong number of episode assignments")
        if snapshots:
            ids = torch.tensor([s.snapshot_id for s in snapshots], dtype=torch.long,
                               device=self.assignment_ids.device)
            self.assignment_ids[active] = ids
            self._snapshots.update({s.snapshot_id: s for s in snapshots})
        self.episode_assignments += len(snapshots)
        active_ids = set(self.assignment_ids[self.assignment_ids >= 0].unique().tolist())
        # Retain evicted snapshots until their last ongoing episode finishes.
        self._snapshots = {key: value for key, value in self._snapshots.items() if key in active_ids}

    def on_step(self, batch, learner_ids, action_space, encoder):
        """Advance every seat's history exactly once per completed joint tick."""
        for pid, history in enumerate(self.histories):
            seats = torch.full_like(learner_ids, pid)
            history.push(encoder.encode(batch, seats))
            if self.knowledge is not None:
                self.knowledge.update(batch, seats)

    @torch.no_grad()
    def joint_actions(self, batch, learner_ids, action_space, encoder):
        if self.pool is None:
            raise RuntimeError("attach a historical policy pool before self-play stepping")
        # Keep the TensorDict import out of heuristic and MDP-only execution.
        from tensordict import TensorDict

        joint = torch.zeros((batch.num_envs, batch.players), dtype=torch.long, device=batch.device)
        for pid in range(batch.players):
            active = learner_ids != pid
            if not bool(active.any()):
                continue
            seats = torch.full_like(learner_ids, pid)
            current = encoder.encode(batch, seats)
            mask = action_space.mask(batch, seats)
            knowledge = None if self.knowledge is None else self.knowledge.encode(batch.state["tick"], seats).flatten(1)
            for snapshot_id in self.assignment_ids[active, pid].unique().tolist():
                lanes = active & (self.assignment_ids[:, pid] == snapshot_id)
                rows = torch.where(lanes)[0]
                snapshot = self._snapshots[snapshot_id]
                values = {"policy": self.histories[pid].frames.index_select(0, rows).flatten(1),
                          "single_obs": current.index_select(0, rows),
                          "action_mask": mask.index_select(0, rows)}
                if knowledge is not None:
                    values["knowledge"] = knowledge.index_select(0, rows)
                observations = TensorDict(values, batch_size=[rows.numel()], device=batch.device)
                if self.deterministic:
                    actions = snapshot.model.act_inference(observations).reshape(-1)
                else:
                    probabilities = snapshot.model.action_probabilities(observations, is_teacher=False)
                    actions = torch.multinomial(probabilities, 1, generator=self.generator).squeeze(1)
                actions = action_space.validate(actions, values["action_mask"])
                joint[:, pid].index_copy_(0, rows, actions)
        return joint

    def snapshot_counts(self):
        if self.assignment_ids is None:
            return {}
        ids, counts = self.assignment_ids[self.assignment_ids >= 0].unique(return_counts=True)
        return {int(key): int(count) for key, count in zip(ids.tolist(), counts.tolist())}


def make_opponent(name, *, history_length=4, seed=42, persistent_knowledge=False):
    if name == "current_self_play":
        return CurrentSelfPlayOpponent(history_length=history_length, seed=seed,
                                       persistent_knowledge=persistent_knowledge)
    if name == "self_play":
        return SelfPlayOpponent(history_length=history_length, seed=seed,
                                persistent_knowledge=persistent_knowledge)
    classes = {"random": RandomOpponent, "wait": WaitOpponent}
    if name not in classes:
        raise ValueError(f"opponent must be one of {(*classes, 'self_play', 'current_self_play')}")
    return classes[name]()
