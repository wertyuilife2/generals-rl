"""Algorithm-neutral MDP composition over the existing CPU/CUDA game backend."""
from dataclasses import asdict, dataclass, field

import torch

from generals import MapConfig, make_env

from .mdp import (EncodedActionSpace, GameBatch, LocalObservationEncoder,
                  ObservationHistory, RewardConfig, ShapedReward)
from .opponents import make_opponent


@dataclass(frozen=True)
class TaskConfig:
    width: int = 9
    height: int = 9
    players: int = 2
    num_envs: int = 64
    backend: str = "cpu"
    device: str = "cpu"
    max_episode_steps: int = 512
    history_length: int = 4
    seed: int = 42
    threads: int = 1
    learner_seat: int = 0
    rotate_seats: bool = True
    opponent: str = "random"
    mountain_density: float = 0.1
    city_density: float = 0.05
    reward: RewardConfig = field(default_factory=RewardConfig)

    def __post_init__(self):
        MapConfig(self.width, self.height, self.players, self.mountain_density, self.city_density)
        if self.num_envs < 1 or self.max_episode_steps < 1 or self.history_length < 1:
            raise ValueError("num_envs, episode steps, and history must be positive")
        if self.backend not in ("cpu", "gpu", "mps"):
            raise ValueError("backend must be cpu, gpu or mps")
        if self.backend == "gpu" and torch.device(self.device).type not in ("cuda", "mps"):
            raise ValueError("GPU simulation requires a CUDA or MPS task device")
        if self.backend == "mps" and torch.device(self.device).type != "mps":
            raise ValueError("MPS simulation requires an MPS task device")
        if not 0 <= self.learner_seat < self.players:
            raise ValueError("learner seat is out of range")


@dataclass(frozen=True)
class TaskStep:
    observations: dict[str, torch.Tensor]
    reward: torch.Tensor
    terminated: torch.Tensor
    truncated: torch.Tensor
    terminal_observations: dict[str, torch.Tensor]
    metrics: dict[str, torch.Tensor]


class GeneralsTask:
    """One learner per lane, replaceable opponent and MDP components.

    All decisions use the pre-growth joint tick boundary. Reset is explicitly
    owned here so terminal rewards and next observations precede reset.
    """

    def __init__(self, config: TaskConfig, *, backend=None, action_space=None,
                 observation_encoder=None, reward_model=None, opponent=None):
        self.config = config
        self.device = torch.device(config.device)
        self.num_envs = config.num_envs
        sim_device = "cpu" if config.backend == "cpu" else config.device
        options = {"threads": config.threads} if config.backend == "cpu" else {}
        self.backend = backend if backend is not None else make_env(
            MapConfig(config.width, config.height, config.players,
                      config.mountain_density, config.city_density),
            batch_size=config.num_envs, device=sim_device, backend=config.backend,
            seed=config.seed, auto_reset=False, episode_ticks=config.max_episode_steps, **options)
        if (self.backend.batch_size, self.backend.players, self.backend.width, self.backend.height) != (
                config.num_envs, config.players, config.width, config.height):
            raise ValueError("injected backend dimensions do not match TaskConfig")
        if self.backend.unwrapped.auto_reset:
            raise ValueError("task requires simulator auto_reset=False")
        self.action_space = action_space or EncodedActionSpace(config.width, config.height)
        self.observation_encoder = observation_encoder or LocalObservationEncoder(config.max_episode_steps)
        self.reward_model = reward_model or ShapedReward(config.reward)
        self.opponent = opponent or make_opponent(config.opponent, history_length=config.history_length,
                                                  seed=config.seed)
        self.history = ObservationHistory(config.history_length)
        self.generator = torch.Generator(device=self.device).manual_seed(config.seed)
        self.player_ids = torch.full((config.num_envs,), config.learner_seat,
                                     dtype=torch.long, device=self.device)
        if config.rotate_seats:
            self.player_ids = (self.player_ids + torch.arange(config.num_envs, device=self.device)) % config.players
        self.episode_length_buf = torch.zeros(config.num_envs, dtype=torch.long, device=self.device)
        self.episode_return = torch.zeros(config.num_envs, device=self.device)
        self.completed_episodes = 0
        self.truncated_episodes = 0
        self.terminated_episodes = 0
        self.history.reset(self.observation_encoder.encode(self.batch(), self.player_ids))
        self._notify_opponent("on_reset")
        self.observations = self._observations()

    def batch(self):
        return GameBatch.from_backend(self.backend, self.device)

    def _observations(self):
        batch = self.batch()
        return {
            "policy": self.history.flattened(),
            "single_obs": self.observation_encoder.encode(batch, self.player_ids),
            "critic": self.observation_encoder.encode(batch, self.player_ids, privileged=True),
            "action_mask": self.action_space.mask(batch, self.player_ids),
        }

    def get_observations(self):
        return self.observations

    def _notify_opponent(self, method, mask=None):
        hook = getattr(self.opponent, method, None)
        if hook is not None:
            arguments = (self.batch(), self.player_ids, self.action_space, self.observation_encoder)
            if method == "on_reset":
                hook(*arguments, mask=mask)
            else:
                hook(*arguments)

    def _backend_mask(self, mask):
        return mask.cpu().numpy() if str(self.backend.device) == "cpu" else mask.to(self.backend.device)

    def reset(self, mask=None, *, rotate=False):
        if mask is None:
            mask = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        self.backend.reset(self._backend_mask(mask))
        if rotate and self.config.rotate_seats:
            self.player_ids[mask] = (self.player_ids[mask] + 1) % self.config.players
        self.episode_length_buf[mask] = 0
        self.episode_return[mask] = 0
        self.history.reset(self.observation_encoder.encode(self.batch(), self.player_ids), mask)
        self._notify_opponent("on_reset", mask)
        self.observations = self._observations()
        return self.observations

    def step(self, actions):
        batch = self.batch()
        learner_actions = self.action_space.validate(actions.to(self.device), self.observations["action_mask"])
        before = batch.learner_stats(self.player_ids)
        if hasattr(self.opponent, "joint_actions"):
            joint = self.opponent.joint_actions(batch, self.player_ids, self.action_space,
                                                self.observation_encoder)
        else:
            joint = torch.empty((self.num_envs, self.config.players), dtype=torch.long, device=self.device)
            for pid in range(self.config.players):
                seats = torch.full_like(self.player_ids, pid)
                joint[:, pid] = self.opponent.act(batch, seats, self.action_space,
                                                 self.observation_encoder, self.generator)
        rows = torch.arange(self.num_envs, device=self.device)
        joint[rows, self.player_ids] = learner_actions
        submitted = joint.cpu().numpy() if str(self.backend.device) == "cpu" else joint.to(self.backend.device)
        result = self.backend.step(submitted)
        # Check simulator overflow/numerical errors before using its outputs.
        self.backend.check_errors()
        batch = self.batch()
        raw_done = torch.as_tensor(result.done, device=self.device).clone()
        alive = batch.state["alive"][rows, self.player_ids]
        terminated = raw_done[:, 0] | ~alive
        truncated = raw_done[:, 1] & ~terminated
        done = terminated | truncated
        reward = self.reward_model.evaluate(before, batch.learner_stats(self.player_ids),
                                           terminated, truncated, batch.state["winner"],
                                           alive, self.player_ids)
        self.episode_length_buf += 1
        self.episode_return += reward.total
        self.history.push(self.observation_encoder.encode(batch, self.player_ids))
        self._notify_opponent("on_step")
        terminal_obs = self._observations()
        metrics = {f"Reward/{key}": value.detach() for key, value in reward.terms.items()}
        metrics.update({"Episode/return": self.episode_return[done].clone(),
                        "Episode/length": self.episode_length_buf[done].float().clone(),
                        "Episode/win": (batch.state["winner"][done] == self.player_ids[done]).float(),
                        "Episode/truncated": truncated[done].float()})
        self.completed_episodes += int(done.sum())
        self.truncated_episodes += int(truncated.sum())
        self.terminated_episodes += int(terminated.sum())
        if bool(done.any()):
            obs = self.reset(done, rotate=True)
        else:
            self.observations = terminal_obs
            obs = terminal_obs
        return TaskStep(obs, reward.total, terminated, truncated, terminal_obs, metrics)

    @property
    def cfg(self):
        return asdict(self.config)

    def close(self):
        self.backend.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
