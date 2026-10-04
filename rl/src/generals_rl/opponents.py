"""Vector opponent interface, independent of CTS and partial/full critic views."""
import torch

from .mdp import ObservationHistory


class WaitOpponent:
    def act(self, batch, player_ids, action_space, encoder, generator):
        return torch.zeros(batch.num_envs, dtype=torch.long, device=batch.device)


class RandomOpponent:
    def act(self, batch, player_ids, action_space, encoder, generator):
        mask = action_space.mask(batch, player_ids)
        return torch.multinomial(mask.float(), 1, generator=generator).squeeze(1)


class SelfPlayOpponent:
    """Episode-fixed opponents drawn from a frozen historical policy pool.

    Histories belong to absolute game seats, not to the learner's rotating
    role. Only local observations enter historical policies. The pool and its
    policy implementation are injected so this module does not depend on CTS.
    """

    def __init__(self, *, history_length=4, seed=42, deterministic=False):
        self.history_length = history_length
        self.seed = seed
        self.deterministic = deterministic
        self.pool = None
        self.histories = []
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
            # Independent of both the learner's global PyTorch RNG and the
            # task's heuristic-opponent generator.
            self.generator = torch.Generator(device=batch.device).manual_seed(self.seed + 217645177)
        self._learner_ids = learner_ids.clone()
        for pid, history in enumerate(self.histories):
            history.reset(encoder.encode(batch, torch.full_like(learner_ids, pid)), mask)
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
            history.push(encoder.encode(batch, torch.full_like(learner_ids, pid)))

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
            for snapshot_id in self.assignment_ids[active, pid].unique().tolist():
                lanes = active & (self.assignment_ids[:, pid] == snapshot_id)
                snapshot = self._snapshots[snapshot_id]
                values = {"policy": self.histories[pid].frames[lanes].flatten(1),
                          "single_obs": current[lanes], "action_mask": mask[lanes]}
                observations = TensorDict(values, batch_size=[int(lanes.sum())], device=batch.device)
                if self.deterministic:
                    actions = snapshot.model.act_inference(observations).reshape(-1)
                else:
                    probabilities = snapshot.model.action_probabilities(observations, is_teacher=False)
                    actions = torch.multinomial(probabilities, 1, generator=self.generator).squeeze(1)
                actions = action_space.validate(actions, values["action_mask"])
                joint[lanes, pid] = actions
        return joint

    def snapshot_counts(self):
        if self.assignment_ids is None:
            return {}
        ids, counts = self.assignment_ids[self.assignment_ids >= 0].unique(return_counts=True)
        return {int(key): int(count) for key, count in zip(ids.tolist(), counts.tolist())}


def make_opponent(name, *, history_length=4, seed=42):
    if name == "self_play":
        return SelfPlayOpponent(history_length=history_length, seed=seed)
    classes = {"random": RandomOpponent, "wait": WaitOpponent}
    if name not in classes:
        raise ValueError(f"opponent must be one of {(*classes, 'self_play')}")
    return classes[name]()
